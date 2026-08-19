//! Execution circuit breaker: plain counters, fully deterministic.
//! Trips on consecutive failures or on a repeated identical action (dead loop).

pub struct Breaker {
    consecutive_failures: u32,
    max_failures: u32,
    last_action: Option<u64>,
    action_repeats: u32,
    max_repeats: u32,
    open: bool,
}

impl Breaker {
    pub fn new(max_failures: u32, max_repeats: u32) -> Self {
        Self {
            consecutive_failures: 0,
            max_failures,
            last_action: None,
            action_repeats: 0,
            max_repeats,
            open: false,
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.open = false;
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        if self.consecutive_failures >= self.max_failures {
            self.open = true;
        }
    }

    /// Record a fingerprint of the action the model just planned. Returns
    /// true if the breaker tripped because of a dead loop.
    pub fn record_action(&mut self, fingerprint: u64) -> bool {
        if self.last_action == Some(fingerprint) {
            self.action_repeats += 1;
        } else {
            self.last_action = Some(fingerprint);
            self.action_repeats = 1;
        }
        if self.action_repeats >= self.max_repeats {
            self.open = true;
            return true;
        }
        false
    }

    /// Manual reset, e.g. after a successful fallback answer calmed things down.
    pub fn reset(&mut self) {
        *self = Self::new(self.max_failures, self.max_repeats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trips_on_failures_and_repeats() {
        let mut b = Breaker::new(2, 3);
        b.record_failure();
        assert!(!b.is_open());
        b.record_failure();
        assert!(b.is_open());

        let mut b = Breaker::new(5, 3);
        assert!(!b.record_action(42));
        assert!(!b.record_action(42));
        assert!(b.record_action(42));
        assert!(b.is_open());
    }

    #[test]
    fn success_resets_failure_counter() {
        let mut b = Breaker::new(3, 3);
        b.record_failure();
        b.record_failure();
        assert!(!b.is_open());
        // Success clears the failure counter
        b.record_success();
        assert!(!b.is_open());
        // Now one more failure shouldn't trip it (needs 3 consecutive)
        b.record_failure();
        assert!(!b.is_open());
    }

    #[test]
    fn different_action_resets_repeat_counter() {
        let mut b = Breaker::new(3, 3);
        assert!(!b.record_action(100));
        assert!(!b.record_action(100));
        // Action changed
        assert!(!b.record_action(200));
        // Needs 3 consecutive 200s to trip
        assert!(!b.record_action(200));
        assert!(b.record_action(200));
        assert!(b.is_open());
    }

    #[test]
    fn manual_reset_clears_open_state() {
        let mut b = Breaker::new(2, 2);
        b.record_failure();
        b.record_failure();
        assert!(b.is_open());

        b.reset();
        assert!(!b.is_open());
    }
}

