//! Named resource locks. One physical device is controlled by at most one
//! task at any moment. No preemption: contention answers "busy" and the
//! caller decides (requeue / reject) — that policy lives outside the kernel.

use std::collections::HashMap;

#[derive(Default)]
pub struct ResourceLocks {
    // resource name -> holder task id
    held: HashMap<String, u64>,
}

impl ResourceLocks {
    /// Try to acquire every resource for `task`. All-or-nothing: on any
    /// conflict nothing is taken and the busy resource name is returned.
    pub fn acquire_all(&mut self, task: u64, resources: &[String]) -> Result<(), String> {
        for r in resources {
            if let Some(holder) = self.held.get(r) {
                if *holder != task {
                    return Err(r.clone());
                }
            }
        }
        for r in resources {
            self.held.insert(r.clone(), task);
        }
        Ok(())
    }

    pub fn release_task(&mut self, task: u64) {
        self.held.retain(|_, holder| *holder != task);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_resource_acquire_and_release() {
        let mut locks = ResourceLocks::default();
        let res = vec!["device:camera".to_string()];

        assert!(locks.acquire_all(1, &res).is_ok());
        // Task 2 fails while Task 1 holds it
        assert_eq!(locks.acquire_all(2, &res), Err("device:camera".to_string()));

        // Release Task 1
        locks.release_task(1);
        // Now Task 2 can acquire
        assert!(locks.acquire_all(2, &res).is_ok());
    }

    #[test]
    fn all_or_nothing_atomicity() {
        let mut locks = ResourceLocks::default();
        // Task 1 holds motor_a
        assert!(locks
            .acquire_all(1, &["device:motor_a".to_string()])
            .is_ok());

        // Task 2 tries to acquire motor_b AND motor_a
        let task2_req = vec!["device:motor_b".to_string(), "device:motor_a".to_string()];
        let err = locks.acquire_all(2, &task2_req);
        assert_eq!(err, Err("device:motor_a".to_string()));

        // Ensure motor_b was NOT acquired by Task 2
        let task3_req = vec!["device:motor_b".to_string()];
        assert!(locks.acquire_all(3, &task3_req).is_ok());
    }

    #[test]
    fn same_task_reacquire() {
        let mut locks = ResourceLocks::default();
        let res = vec!["device:relay".to_string()];
        assert!(locks.acquire_all(1, &res).is_ok());
        // Same task acquiring the same resource again should succeed
        assert!(locks.acquire_all(1, &res).is_ok());
    }

    #[test]
    fn release_only_affects_target_task() {
        let mut locks = ResourceLocks::default();
        assert!(locks.acquire_all(1, &["res1".to_string()]).is_ok());
        assert!(locks.acquire_all(2, &["res2".to_string()]).is_ok());

        locks.release_task(1);

        // res1 is free
        assert!(locks.acquire_all(3, &["res1".to_string()]).is_ok());
        // res2 is still held by task 2
        assert_eq!(locks.acquire_all(3, &["res2".to_string()]), Err("res2".to_string()));
    }
}

