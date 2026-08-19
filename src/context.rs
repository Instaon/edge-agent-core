//! Bounded conversation context. Hard byte budget: pushing past the cap
//! silently drops the oldest entries. The kernel never grows unbounded.

use serde::Serialize;
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize)]
pub struct ContextEntry {
    pub role: String, // "user" | "assistant" | "system"
    pub content: String,
}

pub struct Context {
    entries: VecDeque<ContextEntry>,
    max_bytes: usize,
    cur_bytes: usize,
}

impl Context {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_bytes,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_oldest_when_over_budget() {
        let mut ctx = Context::new(40);
        ctx.push("user", "aaaaaaaaaa"); // 4 + 10
        ctx.push("user", "bbbbbbbbbb");
        ctx.push("user", "cccccccccc"); // 42 bytes total -> oldest dropped
        let all: Vec<_> = ctx.entries().map(|e| e.content.as_str()).collect();
        assert_eq!(all, vec!["bbbbbbbbbb", "cccccccccc"]);
        assert!(ctx.byte_len() <= 40);
    }

    #[test]
    fn single_entry_oversized_truncation() {
        let mut ctx = Context::new(20);
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
        let mut ctx = Context::new(10);
        // "user" is 4 bytes. budget for content is 6 bytes.
        // Each Chinese character "中" is 3 bytes (UTF-8).
        // "中文测试" = 12 bytes. 6 bytes = 2 characters "中文".
        ctx.push("user", "中文测试");
        let first = ctx.entries().next().unwrap();
        assert_eq!(first.content, "中文");
        assert_eq!(ctx.byte_len(), 10);

        // Test non-aligned budget: 11 bytes total -> 7 bytes for content -> 2 chars (6 bytes)
        let mut ctx2 = Context::new(11);
        ctx2.push("user", "中文测试");
        let first2 = ctx2.entries().next().unwrap();
        assert_eq!(first2.content, "中文");
        assert_eq!(ctx2.byte_len(), 10);
    }

    #[test]
    fn empty_context() {
        let ctx = Context::new(100);
        assert_eq!(ctx.byte_len(), 0);
        assert_eq!(ctx.entries().count(), 0);
    }
}

