//! Plugin registry: distribution-spec enforcement + health + rollback.
//!
//! On-disk layout (source-agnostic — cloud pull, LAN push or USB copy all
//! land the same way):
//!   <plugins_dir>/<name>/<semver>/plugin.wasm
//!   <plugins_dir>/<name>/<semver>/manifest.json
//!
//! Lifecycle: 获取 → 验签 → 加载 → 观察期 → 转正/回滚.
//! A `.disabled` marker persists a rollback decision across restarts.

use super::manifest::{Manifest, PluginKind};
use super::runtime::PluginRuntime;
use anyhow::Context;
use ed25519_dalek::VerifyingKey;
use semver::Version;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use wasmtime::Module;

pub struct LoadedPlugin {
    pub manifest: Manifest,
    pub module: Module,
    pub dir: PathBuf,
    pub consecutive_failures: u32,
}

#[derive(Default)]
pub struct Registry {
    plugins: HashMap<String, LoadedPlugin>,
}

pub struct ScanOptions<'a> {
    pub plugins_dir: &'a Path,
    pub pubkey: Option<&'a VerifyingKey>,
    pub allow_unsigned: bool,
}

impl Registry {
    /// Full scan. For each plugin name, versions are tried from highest semver
    /// down; the first one that verifies and compiles becomes active.
    pub fn scan(rt: &PluginRuntime, opts: &ScanOptions) -> Self {
        let mut reg = Registry::default();
        let Ok(entries) = std::fs::read_dir(opts.plugins_dir) else {
            return reg;
        };
        for entry in entries.flatten() {
            let name_dir = entry.path();
            if !name_dir.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            match load_best_version(rt, &name_dir, opts) {
                Ok(Some(p)) => {
                    eprintln!(
                        "[registry] loaded {} v{} ({:?})",
                        p.manifest.name, p.manifest.version, p.manifest.kind
                    );
                    reg.plugins.insert(name, p);
                }
                Ok(None) => {}
                Err(e) => eprintln!("[registry] plugin '{name}' rejected: {e:#}"),
            }
        }
        reg
    }

    /// Silent hot update: re-resolve one plugin from disk. Called between
    /// tasks, so in-flight work is never interrupted.
    pub fn reload(
        &mut self,
        rt: &PluginRuntime,
        opts: &ScanOptions,
        name: &str,
    ) -> anyhow::Result<()> {
        let name_dir = opts.plugins_dir.join(name);
        match load_best_version(rt, &name_dir, opts)? {
            Some(p) => {
                eprintln!("[registry] hot-updated {} -> v{}", name, p.manifest.version);
                self.plugins.insert(name.to_string(), p);
            }
            None => {
                self.plugins.remove(name);
                eprintln!("[registry] plugin '{name}' removed (no loadable version)");
            }
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&LoadedPlugin> {
        self.plugins.get(name)
    }

    /// The single active strategy plugin (deterministic pick if several exist).
    pub fn strategy(&self) -> Option<&LoadedPlugin> {
        let mut found: Vec<&LoadedPlugin> = self
            .plugins
            .values()
            .filter(|p| p.manifest.kind == PluginKind::Strategy)
            .collect();
        found.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
        found.first().copied()
    }

    pub fn hooks_for(&self, point: &str) -> Vec<&LoadedPlugin> {
        let mut found: Vec<&LoadedPlugin> = self
            .plugins
            .values()
            .filter(|p| {
                p.manifest.kind == PluginKind::Hook
                    && p.manifest.hooks.iter().any(|h| h == point)
            })
            .collect();
        found.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
        found
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.plugins.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .plugins
            .values()
            .filter(|p| p.manifest.kind == PluginKind::Tool)
            .map(|p| p.manifest.name.clone())
            .collect();
        names.sort();
        names
    }

    pub fn tool(&self, name: &str) -> Option<&LoadedPlugin> {
        self.plugins
            .get(name)
            .filter(|p| p.manifest.kind == PluginKind::Tool)
    }

    /// Health accounting. On reaching the failure threshold the active version
    /// is marked `.disabled` on disk and the previous stable version (if any)
    /// is activated. Returns true if a rollback/disable happened.
    pub fn report_result(
        &mut self,
        rt: &PluginRuntime,
        opts: &ScanOptions,
        name: &str,
        ok: bool,
        max_failures: u32,
    ) -> bool {
        let Some(p) = self.plugins.get_mut(name) else {
            return false;
        };
        if ok {
            p.consecutive_failures = 0;
            return false;
        }
        p.consecutive_failures += 1;
        if p.consecutive_failures < max_failures {
            return false;
        }
        let bad_version = p.manifest.version.clone();
        let marker = p.dir.join(".disabled");
        if let Err(e) = std::fs::write(&marker, b"auto-disabled: consecutive failures") {
            eprintln!("[registry] cannot write disable marker: {e}");
        }
        eprintln!("[registry] plugin '{name}' v{bad_version} disabled, rolling back");
        let _ = self.reload(rt, opts, name);
        true
    }
}

fn load_best_version(
    rt: &PluginRuntime,
    name_dir: &Path,
    opts: &ScanOptions,
) -> anyhow::Result<Option<LoadedPlugin>> {
    let mut versions: Vec<(Version, PathBuf)> = vec![];
    for entry in std::fs::read_dir(name_dir)
        .with_context(|| format!("cannot read {}", name_dir.display()))?
        .flatten()
    {
        let dir = entry.path();
        if !dir.is_dir() || dir.join(".disabled").exists() {
            continue;
        }
        if let Ok(v) = Version::parse(&entry.file_name().to_string_lossy()) {
            versions.push((v, dir));
        }
    }
    versions.sort_by(|a, b| b.0.cmp(&a.0)); // highest first

    let dir_name = name_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    for (version, dir) in versions {
        match load_version(rt, &dir, &dir_name, &version, opts) {
            Ok(p) => return Ok(Some(p)),
            Err(e) => eprintln!(
                "[registry] {}/{} skipped: {e:#}",
                dir_name, version
            ),
        }
    }
    Ok(None)
}

fn load_version(
    rt: &PluginRuntime,
    dir: &Path,
    expected_name: &str,
    expected_version: &Version,
    opts: &ScanOptions,
) -> anyhow::Result<LoadedPlugin> {
    let manifest_raw = std::fs::read_to_string(dir.join("manifest.json"))?;
    let manifest: Manifest = serde_json::from_str(&manifest_raw)?;
    let wasm = std::fs::read(dir.join("plugin.wasm"))?;

    // Directory layout must agree with the signed manifest.
    anyhow::ensure!(
        manifest.name == expected_name,
        "manifest name '{}' != directory '{}'",
        manifest.name,
        expected_name
    );
    anyhow::ensure!(
        Version::parse(&manifest.version)? == *expected_version,
        "manifest version '{}' != directory '{}'",
        manifest.version,
        expected_version
    );

    // 零信任验签: mandatory unless explicitly in dev mode.
    match (opts.pubkey, opts.allow_unsigned) {
        (Some(pk), _) => manifest.verify(&wasm, pk)?,
        (None, true) => eprintln!(
            "[registry] WARNING: loading UNSIGNED plugin '{}' (dev mode)",
            manifest.name
        ),
        (None, false) => anyhow::bail!("no trusted_pubkey configured and dev_allow_unsigned is off"),
    }

    let module = rt.compile(&wasm).context("wasm compilation failed")?;
    Ok(LoadedPlugin {
        manifest,
        module,
        dir: dir.to_path_buf(),
        consecutive_failures: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_WASM: &[u8] = &[0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

    #[test]
    fn scan_loads_highest_version_and_skips_disabled() {
        let rt = PluginRuntime::new().unwrap();
        let temp_dir = std::env::temp_dir().join(format!("test_reg_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let p_dir = temp_dir.join("my-tool");
        let v1_dir = p_dir.join("1.0.0");
        let v2_dir = p_dir.join("2.0.0");
        let v3_dir = p_dir.join("3.0.0");

        std::fs::create_dir_all(&v1_dir).unwrap();
        std::fs::create_dir_all(&v2_dir).unwrap();
        std::fs::create_dir_all(&v3_dir).unwrap();

        // Write v1
        let m1 = serde_json::json!({
            "name": "my-tool",
            "version": "1.0.0",
            "kind": "tool"
        });
        std::fs::write(v1_dir.join("manifest.json"), m1.to_string()).unwrap();
        std::fs::write(v1_dir.join("plugin.wasm"), MINIMAL_WASM).unwrap();

        // Write v2
        let m2 = serde_json::json!({
            "name": "my-tool",
            "version": "2.0.0",
            "kind": "tool"
        });
        std::fs::write(v2_dir.join("manifest.json"), m2.to_string()).unwrap();
        std::fs::write(v2_dir.join("plugin.wasm"), MINIMAL_WASM).unwrap();

        // Write v3 with .disabled marker
        let m3 = serde_json::json!({
            "name": "my-tool",
            "version": "3.0.0",
            "kind": "tool"
        });
        std::fs::write(v3_dir.join("manifest.json"), m3.to_string()).unwrap();
        std::fs::write(v3_dir.join("plugin.wasm"), MINIMAL_WASM).unwrap();
        std::fs::write(v3_dir.join(".disabled"), b"broken").unwrap();

        let opts = ScanOptions {
            plugins_dir: &temp_dir,
            pubkey: None,
            allow_unsigned: true,
        };

        let mut reg = Registry::scan(&rt, &opts);
        let loaded = reg.get("my-tool").expect("my-tool should be loaded");
        // v3 is disabled, so v2 must be selected
        assert_eq!(loaded.manifest.version, "2.0.0");
        assert_eq!(reg.tool_names(), vec!["my-tool"]);

        // Report failures on v2 to trigger rollback to v1
        assert!(!reg.report_result(&rt, &opts, "my-tool", false, 2));
        assert!(reg.report_result(&rt, &opts, "my-tool", false, 2)); // 2nd failure trips rollback

        // Now v2 is marked .disabled, active version should be 1.0.0
        let rolled = reg.get("my-tool").expect("my-tool should have rolled back to 1.0.0");
        assert_eq!(rolled.manifest.version, "1.0.0");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}

