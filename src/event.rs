//! Unified event ingress. The kernel does not care how events are produced
//! (voice, sensor, timer, API) — business code enqueues them here. Priority
//! assignment is business policy; the queue merely honors the number.

use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    /// Free-form event kind, e.g. "command", "signal".
    pub kind: String,
    #[serde(default)]
    pub payload: serde_json::Value,
    /// Higher runs first. Equal priorities run FIFO.
    #[serde(default)]
    pub priority: u8,
    #[serde(default)]
    pub source: String,
}

struct Queued {
    event: Event,
    seq: u64,
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.event.priority == other.event.priority && self.seq == other.seq
    }
}
impl Eq for Queued {}
impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Queued {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher priority first, then lower seq (older) first.
        self.event
            .priority
            .cmp(&other.event.priority)
            .then(other.seq.cmp(&self.seq))
    }
}

#[derive(Default)]
pub struct EventQueue {
    heap: BinaryHeap<Queued>,
    next_seq: u64,
}

impl EventQueue {
    pub fn push(&mut self, event: Event) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(Queued { event, seq });
    }

    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop().map(|q| q.event)
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_ordering_higher_runs_first() {
        let mut q = EventQueue::default();
        q.push(Event {
            kind: "low".into(),
            payload: serde_json::json!("low_task"),
            priority: 1,
            source: "sensor".into(),
        });
        q.push(Event {
            kind: "high".into(),
            payload: serde_json::json!("high_task"),
            priority: 10,
            source: "voice".into(),
        });
        q.push(Event {
            kind: "mid".into(),
            payload: serde_json::json!("mid_task"),
            priority: 5,
            source: "timer".into(),
        });

        assert_eq!(q.len(), 3);
        assert!(!q.is_empty());

        assert_eq!(q.pop().unwrap().kind, "high");
        assert_eq!(q.pop().unwrap().kind, "mid");
        assert_eq!(q.pop().unwrap().kind, "low");
        assert!(q.pop().is_none());
        assert!(q.is_empty());
    }

    #[test]
    fn fifo_ordering_for_equal_priorities() {
        let mut q = EventQueue::default();
        q.push(Event {
            kind: "first".into(),
            payload: serde_json::json!(1),
            priority: 5,
            source: "app".into(),
        });
        q.push(Event {
            kind: "second".into(),
            payload: serde_json::json!(2),
            priority: 5,
            source: "app".into(),
        });
        q.push(Event {
            kind: "third".into(),
            payload: serde_json::json!(3),
            priority: 5,
            source: "app".into(),
        });

        assert_eq!(q.pop().unwrap().kind, "first");
        assert_eq!(q.pop().unwrap().kind, "second");
        assert_eq!(q.pop().unwrap().kind, "third");
        assert!(q.pop().is_none());
    }

    #[test]
    fn empty_queue_behavior() {
        let mut q = EventQueue::default();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        assert!(q.pop().is_none());
    }
}


