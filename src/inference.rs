//! Inference backend abstraction. The model is a replaceable runtime part:
//! two real backends ship with the kernel — any OpenAI-compatible chat
//! service and Google AI Edge's LiteRT-LM runtime — and business code can
//! implement `InferenceBackend` itself (C-ABI wrappers over llama.cpp /
//! vendor NPU runtimes belong there) and inject it via `Kernel::builder`.
//!
//! The input is multimodal: text plus optional images. Backends that cannot
//! carry images must say so loudly instead of silently dropping them.

use crate::context::ContextEntry;
use anyhow::Context as _;
use base64::Engine as _;

/// One image attached to a user turn: raw bytes + mime type. How it reaches
/// the model is the backend's business (data URL, temp file, tensor...).
#[derive(Debug, Clone)]
pub struct ImagePart {
    /// e.g. "image/jpeg", "image/png".
    pub mime: String,
    pub data: Vec<u8>,
}

/// A user turn: text plus optional images.
#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub text: String,
    pub images: Vec<ImagePart>,
}

impl UserInput {
    pub fn text(t: impl Into<String>) -> Self {
        Self {
            text: t.into(),
            images: vec![],
        }
    }
}

pub trait InferenceBackend: Send {
    fn generate(
        &mut self,
        system: &str,
        context: &[ContextEntry],
        input: &UserInput,
    ) -> anyhow::Result<String>;
}

/// Deterministic backend for running the kernel with no model installed:
/// always answers with a plain reply command. Keeps the loop testable.
pub struct MockBackend;

impl InferenceBackend for MockBackend {
    fn generate(
        &mut self,
        _system: &str,
        _context: &[ContextEntry],
        input: &UserInput,
    ) -> anyhow::Result<String> {
        let mut text = format!("[mock] {}", input.text);
        if !input.images.is_empty() {
            text.push_str(&format!(" [+{} image(s)]", input.images.len()));
        }
        Ok(serde_json::json!({ "reply": text }).to_string())
    }
}

/// Any local/LAN service speaking the OpenAI chat-completions protocol
/// (llama.cpp server, ollama, vllm, vendor runtimes...). Images travel as
/// standard `image_url` content parts with base64 data URLs.
pub struct OpenAiBackend {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
}

/// Message building is separated from transport so the multimodal shape is
/// unit-testable without a live server.
pub fn build_openai_messages(
    system: &str,
    context: &[ContextEntry],
    input: &UserInput,
) -> Vec<serde_json::Value> {
    let mut messages = vec![serde_json::json!({"role": "system", "content": system})];
    for e in context {
        messages.push(serde_json::json!({"role": e.role, "content": e.content}));
    }
    let content = if input.images.is_empty() {
        // Plain string keeps compatibility with servers that reject arrays.
        serde_json::json!(input.text)
    } else {
        let mut parts = vec![serde_json::json!({"type": "text", "text": input.text})];
        for img in &input.images {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&img.data);
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": { "url": format!("data:{};base64,{b64}", img.mime) },
            }));
        }
        serde_json::json!(parts)
    };
    messages.push(serde_json::json!({"role": "user", "content": content}));
    messages
}

impl InferenceBackend for OpenAiBackend {
    fn generate(
        &mut self,
        system: &str,
        context: &[ContextEntry],
        input: &UserInput,
    ) -> anyhow::Result<String> {
        let messages = build_openai_messages(system, context, input);
        let mut req = ureq::post(&self.url).timeout(std::time::Duration::from_secs(120));
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let resp: serde_json::Value = req
            .send_json(serde_json::json!({
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
            }))
            .context("inference request failed")?
            .into_json()
            .context("inference response is not JSON")?;
        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .context("inference response missing choices[0].message.content")?;
        Ok(content.to_string())
    }
}

/// On-device inference through Google AI Edge's LiteRT-LM runtime, driven as
/// a subprocess of the `litert_lm_main` CLI (or any flag-compatible wrapper).
/// One-shot per request: system + bounded context + input are flattened into
/// a single prompt. A tighter in-process FFI binding can replace this behind
/// the same `InferenceBackend` trait without touching the kernel.
pub struct LitertLmBackend {
    /// Path to the litert_lm_main executable.
    pub binary: String,
    /// Path to the .litertlm / .task model bundle.
    pub model_path: String,
    /// Maps to litert_lm_main's `--backend` flag: "cpu" | "gpu" | "npu".
    pub accelerator: Option<String>,
    /// Flag used to pass image files (e.g. "--image_path"). Images are
    /// rejected with a clear error when this is unset, never dropped.
    pub image_arg: Option<String>,
    /// Extra flags appended verbatim.
    pub extra_args: Vec<String>,
}

fn mime_extension(mime: &str) -> &'static str {
    match mime {
        "image/jpeg" | "image/jpg" => "jpg",
        "image/png" => "png",
        "image/webp" => "webp",
        "image/bmp" => "bmp",
        _ => "bin",
    }
}

impl InferenceBackend for LitertLmBackend {
    fn generate(
        &mut self,
        system: &str,
        context: &[ContextEntry],
        input: &UserInput,
    ) -> anyhow::Result<String> {
        let mut prompt = String::new();
        if !system.is_empty() {
            prompt.push_str(system);
            prompt.push_str("\n\n");
        }
        for e in context {
            prompt.push_str(&format!("{}: {}\n", e.role, e.content));
        }
        prompt.push_str(&input.text);

        let mut cmd = std::process::Command::new(&self.binary);
        cmd.arg(format!("--model_path={}", self.model_path));
        if let Some(acc) = &self.accelerator {
            cmd.arg(format!("--backend={acc}"));
        }
        for a in &self.extra_args {
            cmd.arg(a);
        }

        // Images cross via temp files; the CLI only takes paths.
        let mut tmp_files: Vec<std::path::PathBuf> = vec![];
        if !input.images.is_empty() {
            let Some(flag) = &self.image_arg else {
                anyhow::bail!(
                    "litert_lm backend received {} image(s) but no image_arg is configured; \
                     set backend.image_arg (e.g. \"--image_path\") to a flag your \
                     litert_lm build supports",
                    input.images.len()
                );
            };
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            for (i, img) in input.images.iter().enumerate() {
                let path = std::env::temp_dir().join(format!(
                    "ea-litert-{}-{nonce}-{i}.{}",
                    std::process::id(),
                    mime_extension(&img.mime)
                ));
                std::fs::write(&path, &img.data)
                    .with_context(|| format!("cannot write temp image {}", path.display()))?;
                cmd.arg(format!("{flag}={}", path.display()));
                tmp_files.push(path);
            }
        }
        cmd.arg(format!("--input_prompt={prompt}"));

        let out = cmd
            .output()
            .with_context(|| format!("cannot run litert_lm binary '{}'", self.binary));
        for f in &tmp_files {
            let _ = std::fs::remove_file(f);
        }
        let out = out?;
        anyhow::ensure!(
            out.status.success(),
            "litert_lm exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        );
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_backend_generate() {
        let mut backend = MockBackend;
        let res = backend
            .generate("system prompt", &[], &UserInput::text("hello world"))
            .unwrap();
        let val: serde_json::Value = serde_json::from_str(&res).unwrap();
        assert_eq!(val["reply"], "[mock] hello world");
    }

    #[test]
    fn mock_backend_reports_images() {
        let mut backend = MockBackend;
        let input = UserInput {
            text: "what is this".into(),
            images: vec![ImagePart {
                mime: "image/png".into(),
                data: vec![1, 2, 3],
            }],
        };
        let res = backend.generate("s", &[], &input).unwrap();
        let val: serde_json::Value = serde_json::from_str(&res).unwrap();
        assert_eq!(val["reply"], "[mock] what is this [+1 image(s)]");
    }

    #[test]
    fn openai_messages_text_only_stays_string() {
        let ctx = vec![ContextEntry {
            role: "user".into(),
            content: "earlier".into(),
        }];
        let msgs = build_openai_messages("sys", &ctx, &UserInput::text("now"));
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["content"], "earlier");
        // No images => plain string content, not an array.
        assert!(msgs[2]["content"].is_string());
        assert_eq!(msgs[2]["content"], "now");
    }

    #[test]
    fn openai_messages_multimodal_content_parts() {
        let input = UserInput {
            text: "describe".into(),
            images: vec![ImagePart {
                mime: "image/jpeg".into(),
                data: b"fakejpg".to_vec(),
            }],
        };
        let msgs = build_openai_messages("sys", &[], &input);
        let content = &msgs[1]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "describe");
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,"));
        let b64 = url.trim_start_matches("data:image/jpeg;base64,");
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .unwrap(),
            b"fakejpg"
        );
    }

    #[cfg(unix)]
    fn fake_cli(dir: &std::path::Path, script_body: &str) -> String {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("fake_litert_lm.sh");
        std::fs::write(&path, format!("#!/bin/sh\n{script_body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().to_string()
    }

    #[cfg(unix)]
    #[test]
    fn litert_backend_invokes_cli_with_prompt() {
        let dir = std::env::temp_dir().join(format!("ea-litert-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Echo every argument on its own line so the test can inspect them.
        let bin = fake_cli(&dir, r#"for a in "$@"; do echo "$a"; done"#);
        let mut backend = LitertLmBackend {
            binary: bin,
            model_path: "/models/gemma.litertlm".into(),
            accelerator: Some("cpu".into()),
            image_arg: None,
            extra_args: vec!["--max_tokens=64".into()],
        };
        let out = backend
            .generate("be brief", &[], &UserInput::text("hello"))
            .unwrap();
        assert!(out.contains("--model_path=/models/gemma.litertlm"));
        assert!(out.contains("--backend=cpu"));
        assert!(out.contains("--max_tokens=64"));
        assert!(out.contains("be brief"));
        assert!(out.contains("hello"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn litert_backend_rejects_images_without_image_arg() {
        let mut backend = LitertLmBackend {
            binary: "/nonexistent".into(),
            model_path: "m".into(),
            accelerator: None,
            image_arg: None,
            extra_args: vec![],
        };
        let input = UserInput {
            text: "x".into(),
            images: vec![ImagePart {
                mime: "image/png".into(),
                data: vec![0],
            }],
        };
        let err = backend.generate("", &[], &input).unwrap_err();
        assert!(err.to_string().contains("image_arg"));
    }

    #[cfg(unix)]
    #[test]
    fn litert_backend_passes_images_as_temp_files() {
        let dir = std::env::temp_dir().join(format!("ea-litert-img-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let bin = fake_cli(&dir, r#"for a in "$@"; do echo "$a"; done"#);
        let mut backend = LitertLmBackend {
            binary: bin,
            model_path: "m".into(),
            accelerator: None,
            image_arg: Some("--image_path".into()),
            extra_args: vec![],
        };
        let input = UserInput {
            text: "look".into(),
            images: vec![ImagePart {
                mime: "image/jpeg".into(),
                data: vec![0xff, 0xd8],
            }],
        };
        let out = backend.generate("", &[], &input).unwrap();
        let img_line = out
            .lines()
            .find(|l| l.starts_with("--image_path="))
            .expect("image flag passed to CLI");
        assert!(img_line.ends_with(".jpg"));
        // Temp file is cleaned up after the call.
        let path = img_line.trim_start_matches("--image_path=");
        assert!(!std::path::Path::new(path).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
