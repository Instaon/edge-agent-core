//! Wasm plugin ABI: the only contract between kernel and sandbox.
//!
//! Data flow — plugins never touch kernel memory, only serialized copies:
//!   1. kernel builds a `PluginInput` *view* filtered by the manifest's
//!      permissions (no `context` permission => no context bytes cross);
//!   2. input JSON is written into guest memory via the guest's `ea_alloc`;
//!   3. guest's `ea_handle(ptr, len) -> i64` returns packed (ptr << 32 | len)
//!      of its response buffer, 0 on internal error;
//!   4. kernel parses the response strictly as `PluginOutput`
//!      (`deny_unknown_fields`) — anything malformed is discarded and counted
//!      as a plugin failure.
//!
//! Capability calls happen *during* execution through the single host import
//! `edge.host_call(ptr, len) -> i64`, gated per-call against the manifest.

use serde::{Deserialize, Serialize};

/// What the kernel sends into the sandbox.
#[derive(Debug, Serialize)]
pub struct PluginInput {
    /// "tool" | "strategy" | "hook" | "infer" | "repair"
    /// (infer/repair are kernel-reserved invocations of tool-kind plugins
    /// designated in the config; the model can never call them directly.)
    pub kind: String,
    /// Hook point name when kind == "hook" (e.g. "wake", "pre_infer", "post_task").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hook: Option<String>,
    /// The triggering event, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<serde_json::Value>,
    /// Conversation context — present ONLY if the manifest declares
    /// `"context": true`. Untrusted plugins simply never see it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<ContextView>>,
    /// Tool arguments planned by the model / strategy.
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub args: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct ContextView {
    pub role: String,
    pub content: String,
}

/// The only shape a plugin may answer with. Strict on purpose: an edge
/// kernel would rather drop output than guess at it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginOutput {
    pub ok: bool,
    /// Free-form result payload (tool results, hook data).
    #[serde(default)]
    pub result: serde_json::Value,
    /// Strategy plugins only: "rule" (handled deterministically, use `reply`)
    /// or "model" (kernel should run the inference path).
    #[serde(default)]
    pub decision: Option<String>,
    /// Direct user-facing reply, when the plugin produced one.
    #[serde(default)]
    pub reply: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Constructors for native (in-process) plugins, which build the envelope
/// directly instead of serializing it across a sandbox boundary.
impl PluginOutput {
    pub fn reply(text: impl Into<String>) -> Self {
        Self {
            ok: true,
            result: serde_json::Value::Null,
            decision: None,
            reply: Some(text.into()),
            error: None,
        }
    }

    pub fn result(value: serde_json::Value) -> Self {
        Self {
            ok: true,
            result: value,
            decision: None,
            reply: None,
            error: None,
        }
    }

    pub fn fail(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            result: serde_json::Value::Null,
            decision: None,
            reply: None,
            error: Some(msg.into()),
        }
    }

    /// Strategy: handled deterministically, `reply` is the answer.
    pub fn rule(reply: impl Into<String>) -> Self {
        Self {
            ok: true,
            result: serde_json::Value::Null,
            decision: Some("rule".into()),
            reply: Some(reply.into()),
            error: None,
        }
    }

    /// Strategy: defer to the model path.
    pub fn model() -> Self {
        Self {
            ok: true,
            result: serde_json::Value::Null,
            decision: Some("model".into()),
            reply: None,
            error: None,
        }
    }
}

/// Payload of a `host_call` from the sandbox: `{"cap": "...", "op": ..., "args": ...}`.
/// `cap` must appear verbatim in the manifest's `permissions.capabilities`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostCallRequest {
    pub cap: String,
    #[serde(default)]
    pub op: Option<String>,
    #[serde(default)]
    pub args: serde_json::Value,
}

/// Guest-facing symbol names.
pub const GUEST_ALLOC: &str = "ea_alloc";
pub const GUEST_HANDLE: &str = "ea_handle";
pub const HOST_MODULE: &str = "edge";
pub const HOST_LOG: &str = "host_log";
pub const HOST_CALL: &str = "host_call";

pub fn pack_ptr_len(ptr: u32, len: u32) -> i64 {
    (((ptr as u64) << 32) | len as u64) as i64
}

pub fn unpack_ptr_len(v: i64) -> Option<(u32, u32)> {
    if v == 0 {
        return None;
    }
    let v = v as u64;
    Some(((v >> 32) as u32, (v & 0xffff_ffff) as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_and_unpack_roundtrip() {
        assert_eq!(unpack_ptr_len(0), None);

        let ptr = 0x1234_5678;
        let len = 0x0000_0042;
        let packed = pack_ptr_len(ptr, len);
        let unpacked = unpack_ptr_len(packed);
        assert_eq!(unpacked, Some((ptr, len)));

        // Max values for 32-bit fields
        let max_ptr = u32::MAX;
        let max_len = u32::MAX;
        let packed_max = pack_ptr_len(max_ptr, max_len);
        assert_eq!(unpack_ptr_len(packed_max), Some((max_ptr, max_len)));
    }

    #[test]
    fn plugin_input_serialization() {
        let input = PluginInput {
            kind: "tool".into(),
            hook: None,
            event: Some(serde_json::json!({ "kind": "command" })),
            context: Some(vec![ContextView {
                role: "user".into(),
                content: "ping".into(),
            }]),
            args: serde_json::json!({ "param": 1 }),
        };
        let s = serde_json::to_string(&input).unwrap();
        assert!(s.contains("\"kind\":\"tool\""));
        assert!(s.contains("\"context\":"));
        assert!(!s.contains("\"hook\"")); // None skipped
    }

    #[test]
    fn plugin_output_deserialization_and_strict_check() {
        // Valid
        let json_ok = r#"{
            "ok": true,
            "result": { "temperature": 25.5 },
            "reply": "Room is warm"
        }"#;
        let out: PluginOutput = serde_json::from_str(json_ok).unwrap();
        assert!(out.ok);
        assert_eq!(out.reply.as_deref(), Some("Room is warm"));
        assert_eq!(out.result["temperature"], 25.5);

        // Unknown field must be rejected
        let json_invalid = r#"{
            "ok": true,
            "extra_field": "disallowed"
        }"#;
        let res: Result<PluginOutput, _> = serde_json::from_str(json_invalid);
        assert!(res.is_err());
    }

    #[test]
    fn host_call_request_deserialization() {
        let json = r#"{
            "cap": "device:buzzer",
            "op": "beep",
            "args": { "frequency": 440 }
        }"#;
        let req: HostCallRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.cap, "device:buzzer");
        assert_eq!(req.op.as_deref(), Some("beep"));
        assert_eq!(req.args["frequency"], 440);
    }
}

