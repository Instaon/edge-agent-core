//! Demo strategy plugin, showing the full guest side of the ABI:
//!   - `ea_alloc` / `ea_handle` exports for kernel <-> sandbox JSON exchange;
//!   - deterministic rule handling ("ping" answered without any model);
//!   - a capability call through `edge.host_call` (denied unless declared);
//!   - routing everything else to the model.
//!
//! Build: cargo build --release --target wasm32-unknown-unknown

use serde_json::{json, Value};

// ---- ABI plumbing -----------------------------------------------------------

#[link(wasm_import_module = "edge")]
extern "C" {
    fn host_log(ptr: *const u8, len: usize);
    fn host_call(ptr: *const u8, len: usize) -> i64;
}

#[no_mangle]
pub extern "C" fn ea_alloc(len: i32) -> i32 {
    let mut buf = Vec::<u8>::with_capacity(len as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf); // ownership crosses to the host; freed on instance drop
    ptr as i32
}

fn log(msg: &str) {
    unsafe { host_log(msg.as_ptr(), msg.len()) }
}

/// Call a capability on the host; returns the parsed `{ok, data|error}` reply.
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

// ---- Business logic ---------------------------------------------------------

fn handle(input: Value) -> Value {
    let phase = input["args"]["phase"].as_str().unwrap_or("route");
    let text = input["event"]["payload"].as_str().unwrap_or("");
    log(&format!("strategy phase={phase} text={text}"));

    // Deterministic rule: no model involved.
    if text.trim().eq_ignore_ascii_case("ping") {
        return json!({"ok": true, "decision": "rule", "reply": "pong"});
    }
    // Rule using a declared capability (works only if manifest grants it).
    if text.trim() == "beep" {
        let r = call_host(&json!({"cap": "device:buzzer", "op": "beep", "args": {}}));
        return if r["ok"].as_bool().unwrap_or(false) {
            json!({"ok": true, "decision": "rule", "reply": "beeped"})
        } else {
            json!({"ok": true, "decision": "rule",
                   "reply": format!("beep denied: {}", r["error"])})
        };
    }
    // Fallback phase: keep the device responsive with a canned answer.
    if phase == "fallback" {
        return json!({"ok": true, "decision": "rule",
                      "reply": "system degraded, request queued for later"});
    }
    // Everything else goes to the model.
    json!({"ok": true, "decision": "model"})
}
