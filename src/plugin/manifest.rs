//! Plugin package = plugin.wasm + manifest.json. The signature covers
//! sha256(wasm) plus name/version/kind and the canonicalized permission set,
//! so tampering with either code or declared permissions breaks verification.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub name: String,
    /// Semver, e.g. "1.2.0".
    pub version: String,
    /// "tool" | "strategy" | "hook"
    pub kind: PluginKind,
    /// Hook points this plugin attaches to (kind == "hook").
    #[serde(default)]
    pub hooks: Vec<String>,
    #[serde(default)]
    pub permissions: Permissions,
    /// Strategy concurrency declaration (kind == "strategy" only). Absent
    /// means "black box": the kernel runs the strategy strictly serially.
    #[serde(default, skip_serializing_if = "Concurrency::is_default")]
    pub concurrency: Concurrency,
    /// Hex ed25519 signature. Absent only in dev_allow_unsigned mode.
    #[serde(default)]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginKind {
    Tool,
    Strategy,
    Hook,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Permissions {
    /// May the plugin see conversation context in its input?
    pub context: bool,
    /// Capabilities reachable through `host_call`, e.g. "device:relay0",
    /// "net:192.168.1.10:80". Zero permissions by default.
    pub capabilities: Vec<String>,
}

/// What a strategy does to the outside world. Ordered from weakest to
/// strongest; the default is the conservative `Write`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideEffects {
    /// Pure function of its input: no `host_call` is reachable.
    None,
    /// Observes declared resources but changes nothing outside the kernel.
    Read,
    /// May change the world (actuate devices, send messages...).
    #[default]
    Write,
}

/// Concurrency contract a strategy declares. The kernel trusts nothing it
/// cannot enforce: a parallel strategy only reaches the resources it lists,
/// a side-effect-free one reaches none, and writers are never parallel.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Concurrency {
    /// The strategy may run concurrently with other parallel strategies
    /// (including speculatively, before earlier strategies have decided).
    pub parallel: bool,
    pub side_effects: SideEffects,
    /// Every resource the strategy touches (capabilities such as
    /// "device:camera" or logical names). Two strategies sharing a resource
    /// never run in the same parallel batch.
    pub resources: Vec<String>,
}

impl Concurrency {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// A parallel declaration is enforceable only when it is honest about
    /// effects and footprint. `caps` are the capabilities the plugin may
    /// otherwise reach through `host_call` (empty for native plugins).
    pub fn validate(&self, caps: &[String]) -> anyhow::Result<()> {
        if !self.parallel {
            return Ok(());
        }
        anyhow::ensure!(
            self.side_effects != SideEffects::Write,
            "a strategy with write side effects cannot be declared parallel"
        );
        if self.side_effects == SideEffects::Read {
            if let Some(cap) = caps.iter().find(|c| !self.resources.contains(c)) {
                anyhow::bail!(
                    "capability '{cap}' is not listed in concurrency.resources of a parallel strategy"
                );
            }
        }
        Ok(())
    }

    /// Capabilities a parallel strategy may actually reach at runtime.
    /// Non-parallel strategies keep their full manifest capability set.
    pub fn effective_caps(&self, caps: &[String]) -> Vec<String> {
        if !self.parallel {
            return caps.to_vec();
        }
        match self.side_effects {
            SideEffects::None => Vec::new(),
            _ => caps
                .iter()
                .filter(|c| self.resources.contains(c))
                .cloned()
                .collect(),
        }
    }

    /// Whether the scheduler may place this strategy in a parallel batch.
    pub fn parallel_eligible(&self) -> bool {
        self.parallel && self.side_effects != SideEffects::Write
    }
}

impl Manifest {
    /// The exact bytes the distribution signature covers.
    pub fn signing_payload(&self, wasm: &[u8]) -> Vec<u8> {
        let wasm_hash = Sha256::digest(wasm);
        let mut caps = self.permissions.capabilities.clone();
        caps.sort();
        let mut hooks = self.hooks.clone();
        hooks.sort();
        let mut canonical = serde_json::json!({
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "hooks": hooks,
            "context": self.permissions.context,
            "capabilities": caps,
            "wasm_sha256": hex::encode(wasm_hash),
        });
        // The concurrency contract widens what the scheduler will do, so it is
        // signed with the code. Omitted when absent to keep old signatures valid.
        if !self.concurrency.is_default() {
            let mut resources = self.concurrency.resources.clone();
            resources.sort();
            canonical["concurrency"] = serde_json::json!({
                "parallel": self.concurrency.parallel,
                "side_effects": self.concurrency.side_effects,
                "resources": resources,
            });
        }
        canonical.to_string().into_bytes()
    }

    /// Structural checks beyond the JSON schema, run at load time.
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.concurrency.is_default() {
            anyhow::ensure!(
                self.kind == PluginKind::Strategy,
                "concurrency is only meaningful for strategy plugins"
            );
            self.concurrency.validate(&self.permissions.capabilities)?;
        }
        Ok(())
    }

    pub fn verify(&self, wasm: &[u8], pubkey: &VerifyingKey) -> anyhow::Result<()> {
        let sig_hex = self
            .signature
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("manifest has no signature"))?;
        let sig_bytes: [u8; 64] = hex::decode(sig_hex)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("signature is not 64 bytes"))?;
        let sig = Signature::from_bytes(&sig_bytes);
        pubkey
            .verify(&self.signing_payload(wasm), &sig)
            .map_err(|e| anyhow::anyhow!("signature verification failed: {e}"))
    }
}

pub fn parse_pubkey(hex_key: &str) -> anyhow::Result<VerifyingKey> {
    let bytes: [u8; 32] = hex::decode(hex_key.trim())?
        .try_into()
        .map_err(|_| anyhow::anyhow!("public key is not 32 bytes"))?;
    Ok(VerifyingKey::from_bytes(&bytes)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn sign_verify_roundtrip_and_tamper_detection() {
        let key = SigningKey::generate(&mut rand_core::OsRng);
        let wasm = b"\0asm-fake-bytes";
        let mut m = Manifest {
            name: "demo".into(),
            version: "1.0.0".into(),
            kind: PluginKind::Tool,
            hooks: vec![],
            permissions: Permissions {
                context: false,
                capabilities: vec!["device:relay0".into()],
            },
            concurrency: Concurrency::default(),
            signature: None,
        };
        let sig = key.sign(&m.signing_payload(wasm));
        m.signature = Some(hex::encode(sig.to_bytes()));

        let pk = key.verifying_key();
        assert!(m.verify(wasm, &pk).is_ok());
        // Escalating permissions after signing must fail verification.
        m.permissions.capabilities.push("net:0.0.0.0".into());
        assert!(m.verify(wasm, &pk).is_err());
    }

    #[test]
    fn wasm_tamper_fails_verification() {
        let key = SigningKey::generate(&mut rand_core::OsRng);
        let wasm_original = b"\0asm-original";
        let wasm_tampered = b"\0asm-tampered";

        let mut m = Manifest {
            name: "tool".into(),
            version: "0.1.0".into(),
            kind: PluginKind::Tool,
            hooks: vec![],
            permissions: Permissions::default(),
            concurrency: Concurrency::default(),
            signature: None,
        };
        let sig = key.sign(&m.signing_payload(wasm_original));
        m.signature = Some(hex::encode(sig.to_bytes()));

        let pk = key.verifying_key();
        assert!(m.verify(wasm_original, &pk).is_ok());
        assert!(m.verify(wasm_tampered, &pk).is_err());
    }

    #[test]
    fn parse_pubkey_valid_and_invalid() {
        let key = SigningKey::generate(&mut rand_core::OsRng);
        let pk = key.verifying_key();
        let hex_pk = hex::encode(pk.to_bytes());

        let parsed = parse_pubkey(&hex_pk).unwrap();
        assert_eq!(parsed, pk);

        // Invalid hex
        assert!(parse_pubkey("invalid_hex").is_err());
        // Wrong length
        assert!(parse_pubkey("1234").is_err());
    }

    #[test]
    fn canonical_signing_payload_sorting() {
        let wasm = b"\0asm";
        let m1 = Manifest {
            name: "test".into(),
            version: "1.0.0".into(),
            kind: PluginKind::Hook,
            hooks: vec!["post_task".into(), "pre_task".into()],
            permissions: Permissions {
                context: true,
                capabilities: vec!["b_cap".into(), "a_cap".into()],
            },
            concurrency: Concurrency::default(),
            signature: None,
        };
        let m2 = Manifest {
            name: "test".into(),
            version: "1.0.0".into(),
            kind: PluginKind::Hook,
            hooks: vec!["pre_task".into(), "post_task".into()],
            permissions: Permissions {
                context: true,
                capabilities: vec!["a_cap".into(), "b_cap".into()],
            },
            concurrency: Concurrency::default(),
            signature: None,
        };
        // Payload must be identical regardless of original vec order
        assert_eq!(m1.signing_payload(wasm), m2.signing_payload(wasm));
    }

    fn strategy(concurrency: Concurrency, caps: &[&str]) -> Manifest {
        Manifest {
            name: "s".into(),
            version: "1.0.0".into(),
            kind: PluginKind::Strategy,
            hooks: vec![],
            permissions: Permissions {
                context: false,
                capabilities: caps.iter().map(|c| c.to_string()).collect(),
            },
            concurrency,
            signature: None,
        }
    }

    #[test]
    fn absent_concurrency_keeps_legacy_payload_and_parses() {
        let raw = r#"{"name":"s","version":"1.0.0","kind":"strategy"}"#;
        let m: Manifest = serde_json::from_str(raw).unwrap();
        assert!(m.concurrency.is_default());
        assert!(!m.concurrency.parallel_eligible());
        let payload = String::from_utf8(m.signing_payload(b"w")).unwrap();
        assert!(!payload.contains("concurrency"));
    }

    #[test]
    fn concurrency_declaration_is_signed() {
        let key = SigningKey::generate(&mut rand_core::OsRng);
        let mut m = strategy(
            Concurrency {
                parallel: true,
                side_effects: SideEffects::Read,
                resources: vec!["device:camera".into()],
            },
            &["device:camera"],
        );
        m.signature = Some(hex::encode(key.sign(&m.signing_payload(b"w")).to_bytes()));
        assert!(m.verify(b"w", &key.verifying_key()).is_ok());
        // Upgrading a serial strategy to parallel after signing must fail.
        m.concurrency.side_effects = SideEffects::None;
        assert!(m.verify(b"w", &key.verifying_key()).is_err());
    }

    #[test]
    fn concurrency_validation_rules() {
        let parallel = |side_effects, resources: &[&str]| Concurrency {
            parallel: true,
            side_effects,
            resources: resources.iter().map(|r| r.to_string()).collect(),
        };
        // Writers are never parallel.
        assert!(
            strategy(parallel(SideEffects::Write, &[]), &[])
                .validate()
                .is_err()
        );
        // Readers must list every capability they can reach.
        assert!(
            strategy(parallel(SideEffects::Read, &[]), &["device:camera"])
                .validate()
                .is_err()
        );
        assert!(
            strategy(
                parallel(SideEffects::Read, &["device:camera"]),
                &["device:camera"]
            )
            .validate()
            .is_ok()
        );
        // Pure strategies may keep caps; they simply lose them when parallel.
        let pure = parallel(SideEffects::None, &[]);
        assert!(
            strategy(pure.clone(), &["device:buzzer"])
                .validate()
                .is_ok()
        );
        assert!(pure.effective_caps(&["device:buzzer".into()]).is_empty());
        // Only strategies may declare concurrency.
        let mut tool = strategy(pure, &[]);
        tool.kind = PluginKind::Tool;
        assert!(tool.validate().is_err());
    }
}
