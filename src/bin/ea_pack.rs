//! Distribution-spec tooling: key generation and package signing.
//!
//!   ea-pack keygen <out-prefix>          -> <prefix>.key / <prefix>.pub (hex)
//!   ea-pack sign <version-dir> <keyfile> -> signs manifest.json in place
//!
//! A "package" is a version directory holding plugin.wasm + manifest.json.

use ed25519_dalek::{Signer, SigningKey};
use edge_agent_core::plugin::manifest::Manifest;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, prefix] if cmd == "keygen" => keygen(prefix),
        [cmd, dir, key] if cmd == "sign" => sign(Path::new(dir), Path::new(key)),
        _ => {
            eprintln!("usage:\n  ea-pack keygen <out-prefix>\n  ea-pack sign <version-dir> <keyfile>");
            std::process::exit(2);
        }
    }
}

fn keygen(prefix: &str) -> anyhow::Result<()> {
    let key = SigningKey::generate(&mut rand_core::OsRng);
    std::fs::write(format!("{prefix}.key"), hex::encode(key.to_bytes()))?;
    std::fs::write(
        format!("{prefix}.pub"),
        hex::encode(key.verifying_key().to_bytes()),
    )?;
    println!("wrote {prefix}.key (secret, keep off the device) and {prefix}.pub");
    println!("put the .pub hex string into config as \"trusted_pubkey\"");
    Ok(())
}

fn sign(dir: &Path, keyfile: &Path) -> anyhow::Result<()> {
    let key_hex = std::fs::read_to_string(keyfile)?;
    let key_bytes: [u8; 32] = hex::decode(key_hex.trim())?
        .try_into()
        .map_err(|_| anyhow::anyhow!("secret key is not 32 bytes"))?;
    let key = SigningKey::from_bytes(&key_bytes);

    let manifest_path = dir.join("manifest.json");
    let mut manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
    let wasm = std::fs::read(dir.join("plugin.wasm"))?;

    let sig = key.sign(&manifest.signing_payload(&wasm));
    manifest.signature = Some(hex::encode(sig.to_bytes()));
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    println!(
        "signed {} v{} ({})",
        manifest.name,
        manifest.version,
        manifest_path.display()
    );
    Ok(())
}
