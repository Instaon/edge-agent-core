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

impl Manifest {
    /// The exact bytes the distribution signature covers.
    pub fn signing_payload(&self, wasm: &[u8]) -> Vec<u8> {
        let wasm_hash = Sha256::digest(wasm);
        let mut caps = self.permissions.capabilities.clone();
        caps.sort();
        let mut hooks = self.hooks.clone();
        hooks.sort();
        let canonical = serde_json::json!({
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "hooks": hooks,
            "context": self.permissions.context,
            "capabilities": caps,
            "wasm_sha256": hex::encode(wasm_hash),
        });
        canonical.to_string().into_bytes()
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
            signature: None,
        };
        // Payload must be identical regardless of original vec order
        assert_eq!(m1.signing_payload(wasm), m2.signing_payload(wasm));
    }
}

