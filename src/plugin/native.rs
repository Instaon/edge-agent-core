//! Native (in-process) plugins. Wasm is only one execution form — the one
//! that supports hot update from disk on a running device. Business code
//! built on this crate as a library registers plain Rust implementations
//! here: no sandbox, no signature, no serialization round-trip. Native code
//! is trusted by definition — it is compiled into the same binary as the
//! kernel — so it always sees the conversation context and needs no
//! capability adjudication (it can call hardware directly).
//!
//! Native and wasm plugins share the same `PluginInput`/`PluginOutput`
//! contract and the same kernel dispatch (tools, strategy, hooks). On a name
//! collision the native registration wins: in-tree code beats disk artifacts.

use super::abi::{PluginInput, PluginOutput};
use super::manifest::PluginKind;
use std::collections::HashMap;

/// Implement this (or just use a closure — any
/// `FnMut(&PluginInput) -> anyhow::Result<PluginOutput> + Send` qualifies)
/// and register it on `Kernel::builder`.
pub trait NativePlugin: Send {
    fn handle(&mut self, input: &PluginInput) -> anyhow::Result<PluginOutput>;
}

impl<F> NativePlugin for F
where
    F: FnMut(&PluginInput) -> anyhow::Result<PluginOutput> + Send,
{
    fn handle(&mut self, input: &PluginInput) -> anyhow::Result<PluginOutput> {
        self(input)
    }
}

struct Entry {
    kind: PluginKind,
    hooks: Vec<String>,
    /// `device:*` resources locked for the duration of a tool invocation —
    /// same arbitration as wasm tools declare via their manifest.
    devices: Vec<String>,
    plugin: Box<dyn NativePlugin>,
}

#[derive(Default)]
pub struct NativeRegistry {
    entries: HashMap<String, Entry>,
}

impl NativeRegistry {
    pub fn register_tool(
        &mut self,
        name: &str,
        devices: &[&str],
        plugin: impl NativePlugin + 'static,
    ) {
        self.entries.insert(
            name.to_string(),
            Entry {
                kind: PluginKind::Tool,
                hooks: vec![],
                devices: devices.iter().map(|d| d.to_string()).collect(),
                plugin: Box::new(plugin),
            },
        );
    }

    pub fn register_strategy(&mut self, name: &str, plugin: impl NativePlugin + 'static) {
        self.entries.insert(
            name.to_string(),
            Entry {
                kind: PluginKind::Strategy,
                hooks: vec![],
                devices: vec![],
                plugin: Box::new(plugin),
            },
        );
    }

    pub fn register_hook(
        &mut self,
        name: &str,
        points: &[&str],
        plugin: impl NativePlugin + 'static,
    ) {
        self.entries.insert(
            name.to_string(),
            Entry {
                kind: PluginKind::Hook,
                hooks: points.iter().map(|p| p.to_string()).collect(),
                devices: vec![],
                plugin: Box::new(plugin),
            },
        );
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Some(devices) iff `name` is a registered native tool.
    pub fn tool_devices(&self, name: &str) -> Option<Vec<String>> {
        self.entries
            .get(name)
            .filter(|e| e.kind == PluginKind::Tool)
            .map(|e| e.devices.clone())
    }

    pub fn tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.kind == PluginKind::Tool)
            .map(|(n, _)| n.clone())
            .collect();
        names.sort();
        names
    }

    /// The single active native strategy (deterministic pick if several exist).
    pub fn strategy_name(&self) -> Option<String> {
        let mut names: Vec<&String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.kind == PluginKind::Strategy)
            .map(|(n, _)| n)
            .collect();
        names.sort();
        names.first().map(|n| n.to_string())
    }

    pub fn hooks_for(&self, point: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.kind == PluginKind::Hook && e.hooks.iter().any(|h| h == point))
            .map(|(n, _)| n.clone())
            .collect();
        names.sort();
        names
    }

    pub fn invoke(
        &mut self,
        name: &str,
        input: &PluginInput,
    ) -> anyhow::Result<PluginOutput> {
        let entry = self
            .entries
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("native plugin '{name}' not registered"))?;
        entry.plugin.handle(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(kind: &str) -> PluginInput {
        PluginInput {
            kind: kind.into(),
            hook: None,
            event: None,
            context: None,
            args: serde_json::Value::Null,
        }
    }

    #[test]
    fn closure_plugin_registration_and_invoke() {
        let mut reg = NativeRegistry::default();
        reg.register_tool("relay", &["device:relay0"], |_in: &PluginInput| {
            Ok(PluginOutput::reply("clicked"))
        });
        assert!(reg.contains("relay"));
        assert_eq!(reg.tool_names(), vec!["relay"]);
        assert_eq!(
            reg.tool_devices("relay"),
            Some(vec!["device:relay0".to_string()])
        );
        let out = reg.invoke("relay", &input("tool")).unwrap();
        assert!(out.ok);
        assert_eq!(out.reply.as_deref(), Some("clicked"));
    }

    #[test]
    fn strategy_and_hook_queries() {
        let mut reg = NativeRegistry::default();
        reg.register_strategy("router-b", |_: &PluginInput| Ok(PluginOutput::model()));
        reg.register_strategy("router-a", |_: &PluginInput| Ok(PluginOutput::model()));
        reg.register_hook("audit", &["post_task", "on_degrade"], |_: &PluginInput| {
            Ok(PluginOutput::result(serde_json::Value::Null))
        });
        // Deterministic pick: lexicographically first.
        assert_eq!(reg.strategy_name().as_deref(), Some("router-a"));
        assert_eq!(reg.hooks_for("post_task"), vec!["audit"]);
        assert_eq!(reg.hooks_for("pre_task"), Vec::<String>::new());
        // Strategy is not a tool.
        assert_eq!(reg.tool_devices("router-a"), None);
    }

    #[test]
    fn invoke_unknown_plugin_errors() {
        let mut reg = NativeRegistry::default();
        assert!(reg.invoke("ghost", &input("tool")).is_err());
    }

    #[test]
    fn stateful_struct_plugin() {
        struct Counter {
            n: u32,
        }
        impl NativePlugin for Counter {
            fn handle(&mut self, _input: &PluginInput) -> anyhow::Result<PluginOutput> {
                self.n += 1;
                Ok(PluginOutput::result(serde_json::json!({ "count": self.n })))
            }
        }
        let mut reg = NativeRegistry::default();
        reg.register_tool("counter", &[], Counter { n: 0 });
        reg.invoke("counter", &input("tool")).unwrap();
        let out = reg.invoke("counter", &input("tool")).unwrap();
        assert_eq!(out.result["count"], 2);
    }
}
