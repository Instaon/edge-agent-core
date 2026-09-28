//! Bounded conversation context with idle-time compaction and a hard byte cap.

use serde::Serialize;
use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextEntry {
    pub role: String, // "user" | "assistant" | "system"
    pub content: String,
}

pub struct ContextCompaction {
    entries: Vec<ContextEntry>,
    max_summary_bytes: usize,
    removed_bytes: usize,
}

impl ContextCompaction {
    pub fn entries(&self) -> &[ContextEntry] {
        &self.entries
    }

    pub fn max_summary_bytes(&self) -> usize {
        self.max_summary_bytes
    }
}

pub struct Context {
    entries: VecDeque<ContextEntry>,
    max_bytes: usize,
    compact_threshold_bytes: usize,
    cur_bytes: usize,
}

impl Context {
    pub fn new(max_bytes: usize, compact_threshold_percent: u8) -> Self {
        Self {
            entries: VecDeque::new(),
            max_bytes,
            compact_threshold_bytes: max_bytes.saturating_mul(compact_threshold_percent as usize)
                / 100,
            cur_bytes: 0,
        }
    }

    pub fn push(&mut self, role: &str, content: &str) {
        let cost = role.len() + content.len();
        // A single oversized entry is truncated rather than allowed to blow the budget.
        let content = if cost > self.max_bytes {
            let keep = self.max_bytes.saturating_sub(role.len());
            let mut end = keep.min(content.len());
            while end > 0 && !content.is_char_boundary(end) {
                end -= 1;
            }
            &content[..end]
        } else {
            content
        };
        self.cur_bytes += role.len() + content.len();
        self.entries.push_back(ContextEntry {
            role: role.into(),
            content: content.into(),
        });
        while self.cur_bytes > self.max_bytes {
            if let Some(old) = self.entries.pop_front() {
                self.cur_bytes -= old.role.len() + old.content.len();
            } else {
                break;
            }
        }
    }

    pub fn entries(&self) -> impl Iterator<Item = &ContextEntry> {
        self.entries.iter()
    }

    pub fn byte_len(&self) -> usize {
        self.cur_bytes
    }

    pub fn needs_compaction(&self) -> bool {
        self.cur_bytes >= self.compact_threshold_bytes && self.entries.len() > 2
    }

    pub fn compaction_candidate(&self) -> Option<ContextCompaction> {
        if !self.needs_compaction() {
            return None;
        }
        let summary_prefix = "早期对话摘要：";
        let summary_role = "system";
        let summary_overhead = summary_role.len() + summary_prefix.len();
        let mut removed_bytes = 0;
        let mut count = 0;
        let desired_summary_bytes = self.max_bytes / 4;
        while count + 2 < self.entries.len()
            && self.cur_bytes - removed_bytes + summary_overhead + desired_summary_bytes
                > self.compact_threshold_bytes
        {
            let entry = &self.entries[count];
            removed_bytes += entry.role.len() + entry.content.len();
            count += 1;
        }
        if count == 0
            || self.cur_bytes - removed_bytes + summary_overhead >= self.compact_threshold_bytes
        {
            return None;
        }
        let budget = desired_summary_bytes.min(
            self.compact_threshold_bytes - (self.cur_bytes - removed_bytes) - summary_overhead,
        );
        Some(ContextCompaction {
            entries: self.entries.iter().take(count).cloned().collect(),
            max_summary_bytes: budget,
            removed_bytes,
        })
    }

    pub fn apply_summary(&mut self, candidate: &ContextCompaction, summary: &str) -> bool {
        if !self
            .entries
            .iter()
            .take(candidate.entries.len())
            .eq(&candidate.entries)
        {
            return false;
        }
        let summary_prefix = "早期对话摘要：";
        let summary_role = "system";
        let summary_overhead = summary_role.len() + summary_prefix.len();
        let summary = summary.trim();
        if summary.is_empty()
            || summary.len() > candidate.max_summary_bytes
            || summary_overhead + summary.len() >= candidate.removed_bytes
            || self.cur_bytes - candidate.removed_bytes + summary_overhead + summary.len()
                > self.compact_threshold_bytes
        {
            return false;
        }
        for _ in 0..candidate.entries.len() {
            self.entries.pop_front();
        }
        self.entries.push_front(ContextEntry {
            role: summary_role.into(),
            content: format!("{summary_prefix}{summary}"),
        });
        self.cur_bytes =
            self.cur_bytes - candidate.removed_bytes + summary_overhead + summary.len();
        true
    }

    pub fn compact_with(
        &mut self,
        summarize: impl FnOnce(&[ContextEntry], usize) -> anyhow::Result<String>,
    ) -> anyhow::Result<bool> {
        let Some(candidate) = self.compaction_candidate() else {
            return Ok(false);
        };
        let summary = summarize(&candidate.entries, candidate.max_summary_bytes)?;
        Ok(self.apply_summary(&candidate, &summary))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_oldest_when_over_budget() {
        let mut ctx = Context::new(40, 75);
        ctx.push("user", "aaaaaaaaaa"); // 4 + 10
        ctx.push("user", "bbbbbbbbbb");
        ctx.push("user", "cccccccccc"); // 42 bytes total -> oldest dropped
        let all: Vec<_> = ctx.entries().map(|e| e.content.as_str()).collect();
        assert_eq!(all, vec!["bbbbbbbbbb", "cccccccccc"]);
        assert!(ctx.byte_len() <= 40);
    }

    #[test]
    fn single_entry_oversized_truncation() {
        let mut ctx = Context::new(20, 75);
        // "user" is 4 bytes. 20 - 4 = 16 bytes max for content.
        ctx.push("user", "0123456789abcdefghijklmn");
        assert_eq!(ctx.entries().count(), 1);
        let first = ctx.entries().next().unwrap();
        assert_eq!(first.role, "user");
        assert_eq!(first.content, "0123456789abcdef");
        assert_eq!(ctx.byte_len(), 20);
    }

    #[test]
    fn utf8_multibyte_boundary_truncation() {
        let mut ctx = Context::new(10, 75);
        // "user" is 4 bytes. budget for content is 6 bytes.
        // Each Chinese character "中" is 3 bytes (UTF-8).
        // "中文测试" = 12 bytes. 6 bytes = 2 characters "中文".
        ctx.push("user", "中文测试");
        let first = ctx.entries().next().unwrap();
        assert_eq!(first.content, "中文");
        assert_eq!(ctx.byte_len(), 10);

        // Test non-aligned budget: 11 bytes total -> 7 bytes for content -> 2 chars (6 bytes)
        let mut ctx2 = Context::new(11, 75);
        ctx2.push("user", "中文测试");
        let first2 = ctx2.entries().next().unwrap();
        assert_eq!(first2.content, "中文");
        assert_eq!(ctx2.byte_len(), 10);
    }

    #[test]
    fn empty_context() {
        let ctx = Context::new(100, 75);
        assert_eq!(ctx.byte_len(), 0);
        assert_eq!(ctx.entries().count(), 0);
    }

    #[test]
    fn compacts_oldest_and_preserves_recent_turns() {
        let mut ctx = Context::new(100, 75);
        for content in ["老对话甲甲甲甲", "老对话乙乙乙乙", "最近问题", "最近回答"]
        {
            ctx.push("user", content);
        }
        assert!(ctx.needs_compaction());
        assert!(
            ctx.compact_with(|old, budget| {
                assert_eq!(old.len(), 2);
                assert!(budget > 0);
                Ok("早期偏好".into())
            })
            .unwrap()
        );
        let entries: Vec<_> = ctx.entries().collect();
        assert_eq!(entries[0].role, "system");
        assert!(entries[0].content.contains("早期偏好"));
        assert_eq!(entries[1].content, "最近问题");
        assert_eq!(entries[2].content, "最近回答");
        assert!(ctx.byte_len() < 75);
    }

    #[test]
    fn failed_or_oversized_summary_keeps_context_unchanged() {
        let mut ctx = Context::new(100, 75);
        for content in ["老对话甲甲甲甲", "老对话乙乙乙乙", "最近问题", "最近回答"]
        {
            ctx.push("user", content);
        }
        let before: Vec<_> = ctx.entries().map(|entry| entry.content.clone()).collect();
        let bytes = ctx.byte_len();
        assert!(
            ctx.compact_with(|_, _| anyhow::bail!("model unavailable"))
                .is_err()
        );
        assert!(
            !ctx.compact_with(|_, budget| Ok("x".repeat(budget + 1)))
                .unwrap()
        );
        assert_eq!(
            ctx.entries()
                .map(|entry| entry.content.clone())
                .collect::<Vec<_>>(),
            before
        );
        assert_eq!(ctx.byte_len(), bytes);
    }

    #[test]
    fn stale_background_summary_cannot_replace_newer_context() {
        let mut ctx = Context::new(100, 75);
        for content in ["老对话甲甲甲甲", "老对话乙乙乙乙", "最近问题", "最近回答"]
        {
            ctx.push("user", content);
        }
        let candidate = ctx.compaction_candidate().unwrap();
        ctx.push("user", "一条更长的新对话让旧消息出队");
        let before: Vec<_> = ctx.entries().cloned().collect();
        assert!(!ctx.apply_summary(&candidate, "早期偏好"));
        assert_eq!(ctx.entries().cloned().collect::<Vec<_>>(), before);
    }
}
