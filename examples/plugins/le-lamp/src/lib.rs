//! Strategy plugin: wake-word variants 「小灯 / 小登 / 小邓 / 小凳」 are forwarded
//! to the remote LeLamp agent instead of the on-device LLM.
//!
//! The lamp waits until its own Agent / tools / TTS finish, then returns.
//! `spoken: true` means the lamp already played the reply, so this plugin
//! returns an empty `reply` and the hub speaker stays silent.
//!
//! Match miss, intermediate ASR text, or a host-call/network failure all
//! return `decision: "model"` so the kernel continues the chain (native
//! rules → LLM). `ok` stays true on those paths so a flaky lamp cannot
//! auto-disable the plugin.
//!
//! Build: cargo build --release --target wasm32-unknown-unknown

use serde_json::{json, Value};

const KEYWORDS: &[&str] = &["小灯", "小登", "小邓", "小凳"];
const CAP: &str = "net:le-lamp";

#[link(wasm_import_module = "edge")]
extern "C" {
    fn host_log(ptr: *const u8, len: usize);
    fn host_call(ptr: *const u8, len: usize) -> i64;
}

#[no_mangle]
pub extern "C" fn ea_alloc(len: i32) -> i32 {
    let mut buf = Vec::<u8>::with_capacity(len as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr as i32
}

fn log(msg: &str) {
    unsafe { host_log(msg.as_ptr(), msg.len()) }
}

fn call_host(req: &Value) -> Value {
    let bytes = req.to_string().into_bytes();
    let packed = unsafe { host_call(bytes.as_ptr(), bytes.len()) };
    if packed == 0 {
        return json!({"ok": false, "error": "host_call failed"});
    }
    let ptr = (packed as u64 >> 32) as *const u8;
    let len = (packed as u64 & 0xffff_ffff) as usize;
    let raw = unsafe { std::slice::from_raw_parts(ptr, len) };
    serde_json::from_slice(raw).unwrap_or(json!({"ok": false, "error": "bad host reply"}))
}

#[no_mangle]
pub extern "C" fn ea_handle(ptr: i32, len: i32) -> i64 {
    let input = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    let output = match serde_json::from_slice::<Value>(input) {
        Ok(v) => handle(v),
        Err(_) => json!({"ok": false, "error": "bad input"}),
    };
    let bytes = output.to_string().into_bytes();
    let out_ptr = ea_alloc(bytes.len() as i32) as usize as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr, bytes.len()) };
    ((out_ptr as u64) << 32 | bytes.len() as u64) as i64
}

fn handle(input: Value) -> Value {
    let payload = input
        .get("event")
        .and_then(|e| e.get("payload"))
        .cloned()
        .unwrap_or(Value::Null);
    let text = payload
        .as_str()
        .or_else(|| payload.get("text").and_then(|t| t.as_str()))
        .unwrap_or("")
        .to_string();

    if !is_final(&payload) {
        log("le-lamp skip: intermediate asr text");
        return json!({"ok": true, "decision": "model"});
    }
    if !matches_keyword(&text) {
        return json!({"ok": true, "decision": "model"});
    }

    let session_id = payload
        .get("speaker")
        .and_then(|s| s.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("aios-edge-agent");
    let trace_id = payload
        .get("trace_id")
        .and_then(|s| s.as_str())
        .unwrap_or("");

    log(&format!("le-lamp match text={text} session={session_id}"));
    let reply = call_host(&json!({
        "cap": CAP,
        "op": "agent_text",
        "args": { "text": text, "session_id": session_id, "trace_id": trace_id }
    }));
    if !reply["ok"].as_bool().unwrap_or(false) {
        log(&format!(
            "le-lamp host failed, fall through to model: {}",
            reply["error"].as_str().unwrap_or("unknown")
        ));
        return json!({"ok": true, "decision": "model"});
    }

    let data = reply.get("data").cloned().unwrap_or(Value::Null);
    // Always return the lamp's answer so traces / chat have a sentence.
    // When spoken=true the host already published it with play=false
    // (gateway unblocks, hub speaker stays quiet).
    let spoken_text = extract_reply(&data).unwrap_or_else(|| "好的。".to_string());
    if data.get("spoken").and_then(|v| v.as_bool()) == Some(true) {
        log("le-lamp spoken by lamp, hub stays silent");
    }
    json!({"ok": true, "decision": "rule", "reply": spoken_text, "thought": "让台灯去办。"})
}

fn matches_keyword(text: &str) -> bool {
    KEYWORDS.iter().any(|k| text.contains(k))
}

/// Default true: the ASR service only publishes finalized segments. Explicit
/// `is_final=false` / `partial=true` still gates us if a producer sends
/// intermediate hypotheses.
fn is_final(payload: &Value) -> bool {
    if payload.get("partial").and_then(|v| v.as_bool()) == Some(true) {
        return false;
    }
    payload
        .get("is_final")
        .and_then(|v| v.as_bool())
        .or_else(|| {
            payload
                .get("asr")
                .and_then(|a| a.get("is_final"))
                .and_then(|v| v.as_bool())
        })
        .unwrap_or(true)
}

fn extract_reply(data: &Value) -> Option<String> {
    for key in ["answer", "reply", "text", "message", "tts", "speak"] {
        if let Some(s) = data.get(key).and_then(|t| t.as_str()) {
            let t = s.trim();
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    if let Some(obj) = data.get("data") {
        return extract_reply(obj);
    }
    if let Some(obj) = data.get("result") {
        return extract_reply(obj);
    }
    None
}
