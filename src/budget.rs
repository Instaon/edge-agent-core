//! Request budget and pluggable context compaction.
//!
//! The bounded `Context` only caps *history*. What a model actually receives
//! is larger: system prompt + history + current input (strategy instructions,
//! tool results, images) and it must still leave room for the answer. The
//! kernel therefore checks the complete request before every inference call:
//!
//!   fits?            → send as is
//!   over budget      → run the compactor (default: summarize old history,
//!                      keep recent turns verbatim) and validate its result
//!   compaction fails → evict whole oldest messages (never cut one in half)
//!   still over       → fail loudly with `BudgetExceeded`
//!
//! All sizes are bytes, the same unit as `context_max_bytes`.

use crate::context::ContextEntry;
use crate::inference::UserInput;

/// Prefix of the system entry a summary replaces old history with. Shared
/// with idle-time compaction so both produce the same shape.
pub const SUMMARY_PREFIX: &str = "早期对话摘要：";
pub const SUMMARY_ROLE: &str = "system";

pub fn entry_cost(entry: &ContextEntry) -> usize {
    entry.role.len() + entry.content.len()
}

pub fn history_cost(entries: &[ContextEntry]) -> usize {
    entries.iter().map(entry_cost).sum()
}

/// The request could not be brought under budget even with no history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetExceeded {
    /// Bytes the request needs without any history (incl. output reserve).
    pub required: usize,
    pub limit: usize,
}

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "model request exceeds budget: {} bytes required without history (incl. output reserve), limit {}",
            self.required, self.limit
        )
    }
}

impl std::error::Error for BudgetExceeded {}

/// Budget of one complete model request.
#[derive(Debug, Clone, Copy)]
pub struct RequestBudget {
    pub max_bytes: usize,
    pub reserve_bytes: usize,
    pub image_cost_bytes: usize,
}

impl RequestBudget {
    /// Everything in the request except history, plus the output reserve.
    pub fn fixed_cost(&self, system: &str, input: &UserInput) -> usize {
        system.len()
            + input.text.len()
            + input.images.len() * self.image_cost_bytes
            + self.reserve_bytes
    }

    /// Bytes left for history, or the error when even an empty history
    /// cannot fit.
    pub fn history_budget(&self, system: &str, input: &UserInput) -> Result<usize, BudgetExceeded> {
        let required = self.fixed_cost(system, input);
        self.max_bytes.checked_sub(required).ok_or(BudgetExceeded {
            required,
            limit: self.max_bytes,
        })
    }
}

/// What a compactor is asked to do.
pub struct CompactionRequest<'a> {
    /// Current history, oldest first.
    pub entries: &'a [ContextEntry],
    /// The returned history must cost at most this many bytes.
    pub budget_bytes: usize,
    /// Hint: newest entries to keep verbatim (`compaction_keep_recent`).
    pub keep_recent: usize,
}

/// Turns old entries into a short text. The kernel provides one backed by
/// the configured model; tests and hosts can provide their own.
pub trait Summarizer {
    fn summarize(&mut self, entries: &[ContextEntry], max_bytes: usize) -> anyhow::Result<String>;
}

impl<F> Summarizer for F
where
    F: FnMut(&[ContextEntry], usize) -> anyhow::Result<String>,
{
    fn summarize(&mut self, entries: &[ContextEntry], max_bytes: usize) -> anyhow::Result<String> {
        self(entries, max_bytes)
    }
}

/// Replaceable compaction function. Its output is untrusted: the kernel
/// validates it and evicts whole messages instead when it is unusable.
/// Any `FnMut(&CompactionRequest, &mut dyn Summarizer) -> Result<Vec<ContextEntry>> + Send`
/// closure qualifies.
pub trait ContextCompactor: Send {
    fn compact(
        &mut self,
        req: &CompactionRequest,
        summarizer: &mut dyn Summarizer,
    ) -> anyhow::Result<Vec<ContextEntry>>;
}

impl<F> ContextCompactor for F
where
    F: FnMut(&CompactionRequest, &mut dyn Summarizer) -> anyhow::Result<Vec<ContextEntry>> + Send,
{
    fn compact(
        &mut self,
        req: &CompactionRequest,
        summarizer: &mut dyn Summarizer,
    ) -> anyhow::Result<Vec<ContextEntry>> {
        self(req, summarizer)
    }
}

/// Default compactor: one system summary of everything but the newest
/// `keep_recent` entries, followed by those entries verbatim.
pub struct SummarizeCompactor;

impl ContextCompactor for SummarizeCompactor {
    fn compact(
        &mut self,
        req: &CompactionRequest,
        summarizer: &mut dyn Summarizer,
    ) -> anyhow::Result<Vec<ContextEntry>> {
        let keep = req.keep_recent.min(req.entries.len());
        let (old, recent) = req.entries.split_at(req.entries.len() - keep);
        anyhow::ensure!(
            !old.is_empty(),
            "no history older than the kept recent turns"
        );
        let fixed = history_cost(recent) + SUMMARY_ROLE.len() + SUMMARY_PREFIX.len();
        let room = req
            .budget_bytes
            .checked_sub(fixed)
            .filter(|room| *room > 0)
            .ok_or_else(|| anyhow::anyhow!("recent turns alone exceed the history budget"))?;
        let summary = summarizer.summarize(old, room)?;
        let summary = summary.trim();
        anyhow::ensure!(!summary.is_empty(), "summarizer returned an empty summary");
        let mut out = Vec::with_capacity(recent.len() + 1);
        out.push(ContextEntry {
            role: SUMMARY_ROLE.into(),
            content: format!("{SUMMARY_PREFIX}{summary}"),
        });
        out.extend(recent.iter().cloned());
        Ok(out)
    }
}

/// Kernel-side check of a compactor's output.
pub fn validate_compaction(
    original: &[ContextEntry],
    result: &[ContextEntry],
    budget_bytes: usize,
) -> anyhow::Result<()> {
    let cost = history_cost(result);
    anyhow::ensure!(
        cost <= budget_bytes,
        "compacted history is {cost} bytes, budget {budget_bytes}"
    );
    anyhow::ensure!(
        cost < history_cost(original),
        "compaction did not shrink the history"
    );
    for (i, entry) in result.iter().enumerate() {
        anyhow::ensure!(
            matches!(entry.role.as_str(), "user" | "assistant" | "system"),
            "compacted entry {i} has invalid role '{}'",
            entry.role
        );
        anyhow::ensure!(
            !entry.content.trim().is_empty(),
            "compacted entry {i} is empty"
        );
    }
    Ok(())
}

/// Drop whole oldest messages until the history fits. Never truncates a
/// message: a half message is worse than a missing one.
pub fn evict_to_fit(entries: &[ContextEntry], budget_bytes: usize) -> Vec<ContextEntry> {
    let mut cost = history_cost(entries);
    let mut start = 0;
    while cost > budget_bytes && start < entries.len() {
        cost -= entry_cost(&entries[start]);
        start += 1;
    }
    entries[start..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(role: &str, content: &str) -> ContextEntry {
        ContextEntry {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn budget_counts_system_input_images_and_reserve() {
        let budget = RequestBudget {
            max_bytes: 100,
            reserve_bytes: 20,
            image_cost_bytes: 10,
        };
        let mut input = UserInput::text("hello"); // 5
        input.images.push(crate::inference::ImagePart {
            mime: "image/png".into(),
            data: vec![0; 4096],
        });
        assert_eq!(budget.fixed_cost("sys", &input), 3 + 5 + 10 + 20);
        assert_eq!(budget.history_budget("sys", &input), Ok(62));
        let huge = UserInput::text("x".repeat(100));
        assert_eq!(
            budget.history_budget("sys", &huge),
            Err(BudgetExceeded {
                required: 123,
                limit: 100
            })
        );
    }

    #[test]
    fn default_compactor_summarizes_old_and_keeps_recent() {
        let history = vec![
            e("user", "老问题一"),
            e("assistant", "老回答一"),
            e("user", "最近问题"),
            e("assistant", "最近回答"),
        ];
        let req = CompactionRequest {
            entries: &history,
            budget_bytes: 80,
            keep_recent: 2,
        };
        let mut summarizer = |old: &[ContextEntry], max: usize| -> anyhow::Result<String> {
            assert_eq!(old.len(), 2);
            assert!(max > 0);
            Ok("摘要".into())
        };
        let out = SummarizeCompactor.compact(&req, &mut summarizer).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].role, "system");
        assert_eq!(out[0].content, format!("{SUMMARY_PREFIX}摘要"));
        assert_eq!(out[1..], history[2..]);
        assert!(validate_compaction(&history, &out, 80).is_ok());
    }

    #[test]
    fn default_compactor_refuses_when_recent_turns_do_not_fit() {
        let history = vec![e("user", "old"), e("user", "a long recent message")];
        let req = CompactionRequest {
            entries: &history,
            budget_bytes: 10,
            keep_recent: 1,
        };
        let mut never = |_: &[ContextEntry], _: usize| -> anyhow::Result<String> {
            panic!("must not summarize when it cannot help")
        };
        assert!(SummarizeCompactor.compact(&req, &mut never).is_err());
    }

    #[test]
    fn validation_rejects_bad_compactions() {
        let history = vec![e("user", "aaaa"), e("assistant", "bbbb")];
        // Over budget.
        assert!(validate_compaction(&history, &[e("user", "aaaa")], 3).is_err());
        // Did not shrink.
        assert!(validate_compaction(&history, &history, 1000).is_err());
        // Invalid role / empty content.
        assert!(validate_compaction(&history, &[e("tool", "x")], 1000).is_err());
        assert!(validate_compaction(&history, &[e("user", "  ")], 1000).is_err());
        assert!(validate_compaction(&history, &[e("user", "ok")], 1000).is_ok());
    }

    #[test]
    fn eviction_drops_whole_oldest_messages() {
        let history = vec![e("user", "1111"), e("user", "2222"), e("user", "3333")];
        // 8 bytes per entry, 24 in total.
        assert_eq!(evict_to_fit(&history, 24), history);
        assert_eq!(evict_to_fit(&history, 16), history[1..]);
        assert_eq!(evict_to_fit(&history, 15), history[2..]);
        assert!(evict_to_fit(&history, 3).is_empty());
    }
}
