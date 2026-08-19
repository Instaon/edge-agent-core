//! Kernel configuration. All limits are explicit budgets (设计原则: 资源即预算).

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Max bytes of conversation context kept in memory (hard cap, oldest dropped).
    pub context_max_bytes: usize,
    /// Max model-format retries before the circuit breaker trips for this task.
    pub max_format_retries: u32,
    /// Consecutive task failures before the kernel enters fallback-only mode.
    pub breaker_max_failures: u32,
    /// Identical consecutive actions treated as a dead loop.
    pub breaker_max_repeats: u32,
    /// Wasm fuel budget per plugin invocation (execution-time quota).
    pub plugin_fuel: u64,
    /// Wasm linear memory cap per plugin instance, in bytes.
    pub plugin_memory_limit: usize,
    /// Consecutive plugin failures before it is disabled and rolled back.
    pub plugin_max_failures: u32,
    /// Directory scanned for plugin packages: <dir>/<name>/<version>/{plugin.wasm,manifest.json}.
    pub plugins_dir: String,
    /// Hex-encoded ed25519 public key used to verify plugin signatures.
    pub trusted_pubkey: Option<String>,
    /// Optional wasm plugin that repairs malformed model output before the
    /// kernel's own strict validation (which always has the final say).
    pub repair_plugin: Option<String>,
    /// DEV ONLY: load unsigned plugins. Zero-trust default is off.
    pub dev_allow_unsigned: bool,
    pub backend: BackendConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BackendConfig {
    /// Deterministic echo backend, useful without any model installed.
    Mock,
    /// Any local/LAN service speaking the OpenAI chat-completions protocol.
    Openai {
        url: String,
        model: String,
        #[serde(default)]
        api_key: Option<String>,
    },
    /// Inference delegated to a wasm plugin: model access is just another
    /// replaceable runtime part. The plugin reaches its engine through
    /// `host_call` capabilities (net, vendor NPU bridge, ...).
    Plugin { name: String },
}

impl Default for Config {
    fn default() -> Self {
        Self {
            context_max_bytes: 64 * 1024,
            max_format_retries: 2,
            breaker_max_failures: 3,
            breaker_max_repeats: 3,
            plugin_fuel: 50_000_000,
            plugin_memory_limit: 16 * 1024 * 1024,
            plugin_max_failures: 3,
            plugins_dir: "plugins".into(),
            trusted_pubkey: None,
            repair_plugin: None,
            dev_allow_unsigned: false,
            backend: BackendConfig::Mock,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = Config::default();
        assert_eq!(cfg.context_max_bytes, 64 * 1024);
        assert_eq!(cfg.max_format_retries, 2);
        assert_eq!(cfg.breaker_max_failures, 3);
        assert_eq!(cfg.breaker_max_repeats, 3);
        assert_eq!(cfg.plugin_fuel, 50_000_000);
        assert_eq!(cfg.plugin_memory_limit, 16 * 1024 * 1024);
        assert_eq!(cfg.plugins_dir, "plugins");
        assert!(!cfg.dev_allow_unsigned);
        assert!(matches!(cfg.backend, BackendConfig::Mock));
    }

    #[test]
    fn parse_json_configs() {
        // 1. Mock Backend
        let json_mock = r#"{
            "backend": { "type": "mock" }
        }"#;
        let cfg: Config = serde_json::from_str(json_mock).unwrap();
        assert!(matches!(cfg.backend, BackendConfig::Mock));

        // 2. OpenAI Backend
        let json_openai = r#"{
            "context_max_bytes": 1024,
            "backend": {
                "type": "openai",
                "url": "http://127.0.0.1:8000/v1",
                "model": "qwen2.5-coder",
                "api_key": "sk-123"
            }
        }"#;
        let cfg: Config = serde_json::from_str(json_openai).unwrap();
        assert_eq!(cfg.context_max_bytes, 1024);
        match cfg.backend {
            BackendConfig::Openai { url, model, api_key } => {
                assert_eq!(url, "http://127.0.0.1:8000/v1");
                assert_eq!(model, "qwen2.5-coder");
                assert_eq!(api_key.as_deref(), Some("sk-123"));
            }
            _ => panic!("expected openai backend"),
        }

        // 3. Plugin Backend
        let json_plugin = r#"{
            "backend": {
                "type": "plugin",
                "name": "my-model-plugin"
            }
        }"#;
        let cfg: Config = serde_json::from_str(json_plugin).unwrap();
        match cfg.backend {
            BackendConfig::Plugin { name } => assert_eq!(name, "my-model-plugin"),
            _ => panic!("expected plugin backend"),
        }
    }

    #[test]
    fn reject_unknown_fields() {
        let json_invalid = r#"{
            "unknown_field": 123,
            "backend": { "type": "mock" }
        }"#;
        let res: Result<Config, _> = serde_json::from_str(json_invalid);
        assert!(res.is_err());
    }

    #[test]
    fn load_from_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_config.json");
        let content = r#"{
            "breaker_max_failures": 5,
            "dev_allow_unsigned": true,
            "backend": { "type": "mock" }
        }"#;
        std::fs::write(&path, content).unwrap();

        let loaded = Config::load(&path).unwrap();
        assert_eq!(loaded.breaker_max_failures, 5);
        assert!(loaded.dev_allow_unsigned);

        let _ = std::fs::remove_file(path);
    }
}

