//! Inference backend abstraction. The model is a replaceable runtime part:
//! implement `InferenceBackend` to plug in anything (C-ABI wrappers over
//! llama.cpp / vendor NPU runtimes belong here too — link them behind this
//! trait in business code).

use crate::context::ContextEntry;
use anyhow::Context as _;

pub trait InferenceBackend {
    fn generate(
        &mut self,
        system: &str,
        context: &[ContextEntry],
        input: &str,
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
        input: &str,
    ) -> anyhow::Result<String> {
        Ok(serde_json::json!({ "reply": format!("[mock] {input}") }).to_string())
    }
}

/// Any local/LAN service speaking the OpenAI chat-completions protocol
/// (llama.cpp server, ollama, vendor runtimes...).
pub struct OpenAiBackend {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
}

impl InferenceBackend for OpenAiBackend {
    fn generate(
        &mut self,
        system: &str,
        context: &[ContextEntry],
        input: &str,
    ) -> anyhow::Result<String> {
        let mut messages = vec![serde_json::json!({"role": "system", "content": system})];
        for e in context {
            messages.push(serde_json::json!({"role": e.role, "content": e.content}));
        }
        messages.push(serde_json::json!({"role": "user", "content": input}));

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_backend_generate() {
        let mut backend = MockBackend;
        let res = backend
            .generate("system prompt", &[], "hello world")
            .unwrap();
        let val: serde_json::Value = serde_json::from_str(&res).unwrap();
        assert_eq!(val["reply"], "[mock] hello world");
    }
}

