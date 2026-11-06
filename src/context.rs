//! Bounded conversation context with idle-time compaction and a hard byte cap.

use serde::Serialize;
use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContextEntry {
    pub label: String,
    pub role: String, // "user" | "assistant" | "system"
    pub content: String,
}

pub struct ContextCompaction {
    revision: u64,
    label: String,
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
    allowed: Vec<String>,
    label: String,
    revision: u64,
    entries: VecDeque<ContextEntry>,
    max_bytes: usize,
    compact_threshold_bytes: usize,
    cur_bytes: usize,
}

impl Context {
    pub fn new(max_bytes: usize, compact_threshold_percent: u8) -> Self {
        Self {
            allowed: vec!["anonymous".into()],
            label: "anonymous".into(),
            revision: 0,
            entries: VecDeque::new(),
            max_bytes,
            compact_threshold_bytes: max_bytes.saturating_mul(compact_threshold_percent as usize)
                / 100,
            cur_bytes: 0,
        }
    }

    /// Host-selected labels are the isolation boundary. A switch invalidates in-flight summaries.
    pub fn current_label(&self)->&str {&self.label}

    pub fn set_scope(&mut self, label: &str, allowed: &[String]) {
        self.set_view(label, allowed, true);
    }

    /// Background views can temporarily hide a person's history without deleting it.
    pub fn set_view(&mut self, label: &str, allowed: &[String], discard_hidden: bool) {
        self.revision = self.revision.wrapping_add(1);
        self.label = label.to_string();
        self.allowed = allowed.to_vec();
        if discard_hidden {
            self.entries.retain(|e| allowed.contains(&e.label));
        }
        self.cur_bytes = self
            .entries
            .iter()
            .map(|e| e.role.len() + e.content.len())
            .sum();
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
            label: self.label.clone(),
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

    /// Replace the newest entry of `role`. Returns false when that entry is
    /// missing or is no longer the tail. Used so an operational rejection is
    /// not what the next model turn reads back.
    pub fn rewrite_last(&mut self, role: &str, content: &str) -> bool {
        let Some(index) = self.entries.iter().rposition(|entry| entry.role == role) else {
            return false;
        };
        if index + 1 != self.entries.len() || self.entries[index].label != self.label {
            return false;
        }
        let old_cost = self.entries[index].role.len() + self.entries[index].content.len();
        self.cur_bytes -= old_cost;
        let room = self.max_bytes.saturating_sub(role.len());
        let content = truncate_chars(content, room);
        self.entries[index].content = content;
        self.cur_bytes += role.len() + self.entries[index].content.len();
        while self.cur_bytes > self.max_bytes && self.entries.len() > 1 {
            if let Some(old) = self.entries.pop_front() {
                self.cur_bytes = self
                    .cur_bytes
                    .saturating_sub(old.role.len() + old.content.len());
            } else {
                break;
            }
        }
        true
    }

    pub fn entries(&self) -> impl Iterator<Item = &ContextEntry> {
        self.entries
            .iter()
            .filter(|e| self.allowed.contains(&e.label))
    }

    /// Owned copy of the history, oldest first.
    pub fn snapshot(&self) -> Vec<ContextEntry> {
        self.entries().cloned().collect()
    }

    /// Replace the whole history with a validated compaction result. The
    /// hard cap still applies: oldest entries go first if it is exceeded.
    pub fn replace_entries(&mut self, entries: Vec<ContextEntry>) {
        let mut hidden: VecDeque<_> = self
            .entries
            .iter()
            .filter(|e| !self.allowed.contains(&e.label))
            .cloned()
            .collect();
        hidden.extend(
            entries
                .into_iter()
                .filter(|e| self.allowed.contains(&e.label)),
        );
        self.entries = hidden;
        self.cur_bytes = self
            .entries
            .iter()
            .map(|e| e.role.len() + e.content.len())
            .sum();
        while self.cur_bytes > self.max_bytes {
            if let Some(old) = self.entries.pop_front() {
                self.cur_bytes -= old.role.len() + old.content.len();
            } else {
                break;
            }
        }
    }

    pub fn byte_len(&self) -> usize {
        self.cur_bytes
    }

    pub fn needs_compaction(&self) -> bool {
        self.cur_bytes >= self.compact_threshold_bytes && self.entries.len() > 2
    }

    pub fn compaction_candidate(&self) -> Option<ContextCompaction> {
        if !self.needs_compaction()
            || self
                .entries
                .iter()
                .any(|e| !self.allowed.contains(&e.label))
        {
            return None;
        }
        let summary_prefix = crate::budget::SUMMARY_PREFIX;
        let summary_role = crate::budget::SUMMARY_ROLE;
        let summary_overhead = summary_role.len() + summary_prefix.len();
        let mut removed_bytes = 0;
        let mut count = 0;
        let desired_summary_bytes = self.max_bytes / 4;
        while count + 2 < self.entries.len()
            && self.cur_bytes - removed_bytes + summary_overhead + desired_summary_bytes
                > self.compact_threshold_bytes
        {
            let entry = &self.entries[count];
            if entry.label != self.entries[0].label {
                break;
            }
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
            revision: self.revision,
            label: self.entries[0].label.clone(),
            entries: self.entries.iter().take(count).cloned().collect(),
            max_summary_bytes: budget,
            removed_bytes,
        })
    }

    pub fn apply_summary(&mut self, candidate: &ContextCompaction, summary: &str) -> bool {
        if candidate.revision != self.revision {
            return false;
        }
        if !self
            .entries
            .iter()
            .take(candidate.entries.len())
            .eq(&candidate.entries)
        {
            return false;
        }
        let summary_prefix = crate::budget::SUMMARY_PREFIX;
        let summary_role = crate::budget::SUMMARY_ROLE;
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
            label: candidate.label.clone(),
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

fn truncate_chars(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }
    let mut end = max_bytes.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    content[..end].to_string()
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
    fn rewrite_last_replaces_only_the_tail_and_keeps_the_budget() {
        let mut ctx = Context::new(80, 75);
        ctx.push("user", "开灯");
        ctx.push("assistant", "request safely rejected (inference error)");
        assert!(ctx.rewrite_last("assistant", "我这会儿没处理好，麻烦再跟我说一次。"));
        let all: Vec<_> = ctx.entries().map(|e| e.content.as_str()).collect();
        assert_eq!(all, vec!["开灯", "我这会儿没处理好，麻烦再跟我说一次。"]);
        let summed: usize = ctx.entries().map(|e| e.role.len() + e.content.len()).sum();
        assert_eq!(ctx.byte_len(), summed);
        assert!(ctx.byte_len() <= 80);
        ctx.push("user", "下一句");
        assert!(!ctx.rewrite_last("assistant", "太晚了"));
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

#[cfg(test)]
mod isolation_tests {
    use super::*;
    #[test]
    fn identity_switch_preserves_anonymous_and_removes_previous_person() {
        let mut c = Context::new(1000, 75);
        let anon = "anonymous:conversation".to_string();
        let a = "person:a".to_string();
        let b = "person:b".to_string();
        c.set_scope(&anon, std::slice::from_ref(&anon));
        c.push("user", "turn on lamp");
        c.set_scope(&a, &[anon.clone(), a.clone()]);
        c.push("user", "ALICE_PRIVATE");
        c.set_scope(&b, &[anon, b.clone()]);
        assert!(c.snapshot().iter().any(|x| x.content == "turn on lamp"));
        assert!(!c.snapshot().iter().any(|x| x.content == "ALICE_PRIVATE"));
    }
    #[test]
    fn scope_change_rejects_late_summary_even_if_prefix_unchanged() {
        let mut c = Context::new(100, 75);
        for text in ["老对话甲甲甲甲", "老对话乙乙乙乙", "最近问题", "最近回答"]
        {
            c.push("user", text);
        }
        let candidate = c.compaction_candidate().unwrap();
        c.set_scope("anonymous", &["anonymous".into()]);
        assert!(!c.apply_summary(&candidate, "迟到摘要"));
    }
}

#[cfg(test)]
mod background_isolation_tests {
    use super::*;
    #[test]
    fn background_view_cannot_read_or_compact_person_but_does_not_destroy_dialogue() {
        let mut c = Context::new(300, 75);
        c.set_scope("person:a", &["person:a".into()]);
        c.push("user", "A_PRIVATE");
        c.set_view("space:living", &["space:living".into()], false);
        assert!(c.snapshot().is_empty());
        assert!(c.compaction_candidate().is_none());
        c.push("system", "lamp on");
        c.replace_entries(vec![]);
        c.set_scope("person:a", &["person:a".into(), "space:living".into()]);
        assert_eq!(c.snapshot()[0].content, "A_PRIVATE");
        c.set_scope("person:b", &["person:b".into(), "space:living".into()]);
        assert!(!c.snapshot().iter().any(|e| e.content == "A_PRIVATE"));
    }
}
