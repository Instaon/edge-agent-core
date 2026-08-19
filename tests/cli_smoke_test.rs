//! CLI binary smoke tests for edge-agent and ea-pack.

use edge_agent_core::plugin::manifest::{parse_pubkey, Manifest};
use std::io::Write;
use std::process::{Command, Stdio};

/// Helper to get target debug binary path
fn target_bin(name: &str) -> std::path::PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // drop test binary name
    if path.ends_with("deps") {
        path.pop(); // drop deps directory
    }
    path.push(name);
    path
}

#[test]
fn smoke_ea_pack_keygen_and_sign() {
    let temp_dir = std::env::temp_dir().join(format!(
        "test_cli_pack_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&temp_dir).unwrap();

    let ea_pack_bin = target_bin("ea-pack");

    // 1. Test keygen
    let key_prefix = temp_dir.join("test_key");
    let status = Command::new(&ea_pack_bin)
        .arg("keygen")
        .arg(&key_prefix)
        .status()
        .expect("failed to run ea-pack keygen");
    assert!(status.success());

    let key_file = temp_dir.join("test_key.key");
    let pub_file = temp_dir.join("test_key.pub");
    assert!(key_file.exists());
    assert!(pub_file.exists());

    let pub_hex = std::fs::read_to_string(&pub_file).unwrap();
    let pubkey = parse_pubkey(&pub_hex).expect("generated public key must be valid");

    // 2. Prepare mock plugin package for signing
    let pkg_dir = temp_dir.join("demo_plugin").join("1.0.0");
    std::fs::create_dir_all(&pkg_dir).unwrap();

    let wasm_bytes = b"\0asm\x01\0\0\0";
    std::fs::write(pkg_dir.join("plugin.wasm"), wasm_bytes).unwrap();

    let manifest_json = serde_json::json!({
        "name": "demo_plugin",
        "version": "1.0.0",
        "kind": "tool",
        "hooks": [],
        "permissions": {
            "context": false,
            "capabilities": ["device:sensor"]
        }
    });
    std::fs::write(
        pkg_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest_json).unwrap(),
    )
    .unwrap();

    // 3. Test sign
    let status = Command::new(&ea_pack_bin)
        .arg("sign")
        .arg(&pkg_dir)
        .arg(&key_file)
        .status()
        .expect("failed to run ea-pack sign");
    assert!(status.success());

    // 4. Verify the signed manifest
    let signed_raw = std::fs::read_to_string(pkg_dir.join("manifest.json")).unwrap();
    let signed_manifest: Manifest = serde_json::from_str(&signed_raw).unwrap();
    assert!(signed_manifest.signature.is_some());
    assert!(signed_manifest.verify(wasm_bytes, &pubkey).is_ok());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn smoke_edge_agent_stdin_stdout_pipeline() {
    let edge_agent_bin = target_bin("edge-agent");

    let mut child = Command::new(&edge_agent_bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn edge-agent");

    let event_json = r#"{"kind":"command","payload":"turn on the lamp","priority":1,"source":"cli"}"#;

    {
        let stdin = child.stdin.as_mut().expect("failed to open stdin");
        writeln!(stdin, "{}", event_json).expect("failed to write to stdin");
    } // stdin is closed here

    let output = child.wait_with_output().expect("failed to wait on edge-agent");
    assert!(output.status.success());

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout_str.trim();
    assert!(!trimmed.is_empty(), "expected stdout from edge-agent");

    let outcome: serde_json::Value = serde_json::from_str(trimmed)
        .expect("edge-agent stdout line should be valid JSON TaskOutcome");
    assert_eq!(outcome["status"], "ok");
    assert_eq!(outcome["via"], "model");
    assert_eq!(outcome["reply"], "[mock] turn on the lamp");
}
