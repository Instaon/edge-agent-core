//! End-to-end integration smoke tests for edge-agent-core.

use edge_agent_core::{
    config::{BackendConfig, Config},
    event::Event,
    kernel::Kernel,
    plugin::runtime::HostBridge,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Smoke Test 1: Basic end-to-end event execution with mock backend.
#[test]
fn smoke_basic_end_to_end_mock_flow() {
    let cfg = Config {
        plugins_dir: "target/empty_plugins".into(),
        dev_allow_unsigned: true,
        backend: BackendConfig::Mock,
        ..Default::default()
    };
    let mut kernel = Kernel::new(cfg, None).expect("failed to initialize kernel");

    // Push two command events
    kernel.queue.push(Event {
        kind: "command".into(),
        payload: serde_json::json!("turn on living room light"),
        priority: 1,
        source: "voice".into(),
    });
    kernel.queue.push(Event {
        kind: "command".into(),
        payload: serde_json::json!("set brightness to 50%"),
        priority: 1,
        source: "app".into(),
    });

    let outcomes = kernel.run_pending();
    assert_eq!(outcomes.len(), 2);

    assert_eq!(outcomes[0].status, "ok");
    assert_eq!(outcomes[0].via, "model");
    assert_eq!(outcomes[0].reply, "[mock] turn on living room light");

    assert_eq!(outcomes[1].status, "ok");
    assert_eq!(outcomes[1].via, "model");
    assert_eq!(outcomes[1].reply, "[mock] set brightness to 50%");

    kernel.shutdown();
}

/// Smoke Test 2: Multi-priority event queue scheduling and FIFO preservation.
#[test]
fn smoke_event_priority_scheduling() {
    let cfg = Config {
        plugins_dir: "target/empty_plugins".into(),
        dev_allow_unsigned: true,
        backend: BackendConfig::Mock,
        ..Default::default()
    };
    let mut kernel = Kernel::new(cfg, None).expect("failed to initialize kernel");

    // Push events with varying priorities and same priorities
    kernel.queue.push(Event {
        kind: "command".into(),
        payload: serde_json::json!("low_1"),
        priority: 1,
        source: "sensor".into(),
    });
    kernel.queue.push(Event {
        kind: "command".into(),
        payload: serde_json::json!("urgent"),
        priority: 100,
        source: "safety".into(),
    });
    kernel.queue.push(Event {
        kind: "command".into(),
        payload: serde_json::json!("mid_1"),
        priority: 10,
        source: "timer".into(),
    });
    kernel.queue.push(Event {
        kind: "command".into(),
        payload: serde_json::json!("mid_2"),
        priority: 10,
        source: "timer".into(),
    });
    kernel.queue.push(Event {
        kind: "command".into(),
        payload: serde_json::json!("low_2"),
        priority: 1,
        source: "sensor".into(),
    });

    let outcomes = kernel.run_pending();
    assert_eq!(outcomes.len(), 5);

    let replies: Vec<_> = outcomes.iter().map(|o| o.reply.as_str()).collect();
    assert_eq!(
        replies,
        vec![
            "[mock] urgent",
            "[mock] mid_1",
            "[mock] mid_2",
            "[mock] low_1",
            "[mock] low_2",
        ]
    );

    kernel.shutdown();
}

/// Smoke Test 3: Dead loop detection tripping breaker and fallback degradation.
#[test]
fn smoke_dead_loop_breaker_and_safe_reject() {
    // 1. Without strategy plugin: fallback safely rejects
    let cfg_no_strategy = Config {
        breaker_max_repeats: 3,
        plugins_dir: "target/empty_plugins".into(),
        dev_allow_unsigned: true,
        backend: BackendConfig::Mock,
        ..Default::default()
    };
    let mut kernel = Kernel::new(cfg_no_strategy, None).expect("failed to initialize kernel");

    // Repeating identical command payloads produces identical mock output.
    // 1st time: ok
    let o1 = kernel.handle_event(Event {
        kind: "command".into(),
        payload: serde_json::json!("repeat"),
        priority: 1,
        source: "test".into(),
    });
    assert_eq!(o1.status, "ok");

    // 2nd time: ok
    let o2 = kernel.handle_event(Event {
        kind: "command".into(),
        payload: serde_json::json!("repeat"),
        priority: 1,
        source: "test".into(),
    });
    assert_eq!(o2.status, "ok");

    // 3rd time: trips dead loop breaker -> enters fallback -> safe reject
    let o3 = kernel.handle_event(Event {
        kind: "command".into(),
        payload: serde_json::json!("repeat"),
        priority: 1,
        source: "test".into(),
    });
    assert_eq!(o3.status, "rejected");
    assert_eq!(o3.via, "fallback");
    assert!(o3.reply.contains("repeated action loop"));

    // After fallback, breaker is reset, so a fresh action succeeds
    let o4 = kernel.handle_event(Event {
        kind: "command".into(),
        payload: serde_json::json!("different_action"),
        priority: 1,
        source: "test".into(),
    });
    assert_eq!(o4.status, "ok");

    kernel.shutdown();
}

/// Smoke Test 4: Plugin hot update via kernel-reserved `plugin_reload` event.
#[test]
fn smoke_plugin_hot_reload() {
    let cfg = Config {
        dev_allow_unsigned: true,
        plugins_dir: "plugins".into(),
        backend: BackendConfig::Mock,
        ..Default::default()
    };
    let mut kernel = Kernel::new(cfg, None).expect("failed to initialize kernel");

    // Send plugin_reload event for strategy-demo
    let outcome = kernel.handle_event(Event {
        kind: "plugin_reload".into(),
        payload: serde_json::json!({ "name": "strategy-demo" }),
        priority: 255,
        source: "admin".into(),
    });

    assert_eq!(outcome.status, "ok");
    assert_eq!(outcome.via, "kernel");
    assert_eq!(outcome.reply, "plugin 'strategy-demo' reloaded");

    // Reloading nonexistent plugin should report error
    let bad_outcome = kernel.handle_event(Event {
        kind: "plugin_reload".into(),
        payload: serde_json::json!({ "name": "non_existent_plugin" }),
        priority: 255,
        source: "admin".into(),
    });
    assert_eq!(bad_outcome.status, "error");
    assert_eq!(bad_outcome.via, "kernel");
    assert!(bad_outcome.reply.contains("reload failed"));

    kernel.shutdown();
}

/// Smoke Test 5: Strategy plugin execution and deterministic rule handling.
#[test]
fn smoke_strategy_plugin_routing_and_fallback() {
    let cfg = Config {
        dev_allow_unsigned: true,
        plugins_dir: "plugins".into(),
        backend: BackendConfig::Mock,
        ..Default::default()
    };
    let mut kernel = Kernel::new(cfg, None).expect("failed to initialize kernel");

    let outcome = kernel.handle_event(Event {
        kind: "command".into(),
        payload: serde_json::json!("status check"),
        priority: 1,
        source: "app".into(),
    });

    // Valid outcome from strategy or model
    assert!(outcome.status == "ok" || outcome.status == "rejected");
    assert!(outcome.via == "rule" || outcome.via == "model" || outcome.via == "fallback");

    kernel.shutdown();
}

/// Custom test HostBridge for checking capability dispatch
struct TestBridge {
    call_count: AtomicUsize,
}

impl HostBridge for TestBridge {
    fn call(
        &self,
        plugin: &str,
        cap: &str,
        _op: Option<&str>,
        _args: &serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(serde_json::json!({
            "plugin": plugin,
            "cap": cap,
            "handled": true
        }))
    }
}

/// Smoke Test 6: HostBridge capability integration.
#[test]
fn smoke_host_bridge_integration() {
    let bridge = Arc::new(TestBridge {
        call_count: AtomicUsize::new(0),
    });

    let cfg = Config {
        dev_allow_unsigned: true,
        plugins_dir: "plugins".into(),
        backend: BackendConfig::Mock,
        ..Default::default()
    };

    let mut kernel = Kernel::new(cfg, Some(bridge.clone())).expect("failed to initialize kernel");
    let outcome = kernel.handle_event(Event {
        kind: "command".into(),
        payload: serde_json::json!("ping"),
        priority: 1,
        source: "test".into(),
    });

    assert_eq!(outcome.status, "ok");
    kernel.shutdown();
}
