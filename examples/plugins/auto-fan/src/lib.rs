//! Occupancy + temperature fan strategy.
//!
//! On a world-model change of occupancy (desk presence) / temperature:
//!   - person present AND temp > 28 °C AND fan off → turn the fan on
//!
//! Inverse (turn off) is not part of this rule. Misses — wrong event, missing
//! world data, host-call failure, conditions not met — return
//! `decision: "model"` so the kernel continues (other wasm → native rules →
//! LLM). `ok` stays true on those paths so a flaky world-model cannot
//! auto-disable the plugin.
//!
//! Occupancy is the desk presence sensor only — never the office composite
//! (`sensor.app_ban_gong_shi`), which is used for temperature.
//!
//! Build: cargo build --release --target wasm32-unknown-unknown

use serde_json::{json, Value};

const CAP: &str = "device:space-context";
/// "超过 28 摄氏度"：strictly greater than.
const TEMP_ON: f64 = 28.0;

/// Live hub ids.
const FAN_IDS: &[&str] = &["device.app_mtjk3r6x"];
const OCC_IDS: &[&str] =
    &["sensor.ha_binary_sensor_ren_ti_cun_zai_chuan_gan_qi_ban_gong_zhuo_qian_occupancy"];
const TEMP_IDS: &[&str] = &["sensor.app_ban_gong_shi"];

const TEMP_FIELDS: &[&str] = &["temperature"];
const OCC_FIELDS: &[&str] = &["occupancy", "presence", "motion"];

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

    if payload["self_originated"].as_bool().unwrap_or(false) {
        return fallthrough();
    }
    if !is_world_event(&payload) {
        return fallthrough();
    }
    if !is_auto_fan_event(&payload) {
        return fallthrough();
    }

    let world = call_host(&json!({
        "cap": CAP,
        "op": "get",
        "args": { "entities": focus_ids() }
    }));
    if !world["ok"].as_bool().unwrap_or(false) {
        log(&format!(
            "auto-fan skip trigger={}.{}={} reason=get world failed err={}",
            payload["entity"].as_str().unwrap_or("-"),
            payload["field"].as_str().unwrap_or("-"),
            payload["value"],
            world["error"].as_str().unwrap_or("unknown")
        ));
        return fallthrough();
    }

    let data = world.get("data").cloned().unwrap_or(Value::Null);
    let Some(plan) = plan_action(&data, &payload) else {
        log(&format!(
            "auto-fan skip trigger={}.{}={} reason=conditions not met or world incomplete",
            payload["entity"].as_str().unwrap_or("-"),
            payload["field"].as_str().unwrap_or("-"),
            payload["value"]
        ));
        return fallthrough();
    };

    if !plan.needs_write {
        log(&format!(
            "auto-fan eval trigger={}.{}={} want=on action=keep temp={:?} present={:?}",
            payload["entity"].as_str().unwrap_or("-"),
            payload["field"].as_str().unwrap_or("-"),
            payload["value"],
            plan.temp,
            plan.present
        ));
        // Fall through so a sibling wasm (auto-lamp) can still act on the
        // same occupancy event. First `decision: "rule"` wins the chain.
        return fallthrough();
    }

    log(&format!(
        "auto-fan eval trigger={}.{}={} want=on action=set_on fan={} temp={:?} present={:?}",
        payload["entity"].as_str().unwrap_or("-"),
        payload["field"].as_str().unwrap_or("-"),
        payload["value"],
        plan.fan,
        plan.temp,
        plan.present
    ));
    let written = call_host(&json!({
        "cap": CAP,
        "op": "set_desired",
        "args": {
            "entity": plan.fan,
            "field": "power",
            "value": true,
            "trace_id": payload["trace_id"].as_str().unwrap_or("")
        }
    }));
    if !written["ok"].as_bool().unwrap_or(false) {
        log(&format!(
            "auto-fan set_desired failed, fall through: {}",
            written["error"].as_str().unwrap_or("unknown")
        ));
        return fallthrough();
    }
    json!({"ok": true, "decision": "rule", "reply": "", "thought": "风扇就这样。"})
}

fn fallthrough() -> Value {
    json!({"ok": true, "decision": "model"})
}

fn is_world_event(payload: &Value) -> bool {
    matches!(
        payload["kind"].as_str().unwrap_or(""),
        "context_change" | "occurrence"
    )
}

fn is_auto_fan_event(payload: &Value) -> bool {
    let entity = payload["entity"].as_str().unwrap_or("");
    let field = payload["field"].as_str().unwrap_or("");
    if TEMP_IDS.contains(&entity) {
        return field.is_empty() || TEMP_FIELDS.contains(&field);
    }
    if OCC_IDS.contains(&entity) {
        return field.is_empty() || OCC_FIELDS.contains(&field);
    }
    TEMP_FIELDS.contains(&field) || (OCC_FIELDS.contains(&field) && !is_temp_source_id(entity))
}

fn is_temp_source_id(entity: &str) -> bool {
    TEMP_IDS.contains(&entity)
}

fn focus_ids() -> Vec<&'static str> {
    let mut ids = Vec::new();
    ids.extend_from_slice(FAN_IDS);
    ids.extend_from_slice(OCC_IDS);
    ids.extend_from_slice(TEMP_IDS);
    ids
}

struct Plan {
    fan: String,
    needs_write: bool,
    temp: Option<f64>,
    present: Option<bool>,
}

fn plan_action(world: &Value, event: &Value) -> Option<Plan> {
    let entities = world["entities"].as_array()?;
    let unrec = world["unreconciled"].as_array();

    let temp_ent = find_temp_entity(entities);
    let occ_ent = find_occ_entity(entities);
    let fan_ent = find_fan_entity(entities)?;

    let mut temp = temp_ent.and_then(|e| first_field(e, TEMP_FIELDS).and_then(as_f64));
    let mut present = occ_ent.and_then(|e| first_field(e, OCC_FIELDS).and_then(as_boolish));

    // The triggering event is newer than the 10s snapshot cache.
    let ev_entity = event["entity"].as_str().unwrap_or("");
    let ev_field = event["field"].as_str().unwrap_or("");
    let ev_value = &event["value"];
    let ev_is_temp_entity = temp_ent
        .map(|e| e["id"].as_str() == Some(ev_entity))
        .unwrap_or(false);
    let ev_is_occ_entity = occ_ent
        .map(|e| e["id"].as_str() == Some(ev_entity))
        .unwrap_or(false);
    if TEMP_FIELDS.contains(&ev_field) || (ev_is_temp_entity && !OCC_FIELDS.contains(&ev_field)) {
        if let Some(v) = as_f64(ev_value) {
            temp = Some(v);
        }
    }
    if ev_is_occ_entity && OCC_FIELDS.contains(&ev_field) {
        if let Some(v) = as_boolish(ev_value) {
            present = Some(v);
        }
    }

    let fan_id = fan_ent["id"].as_str()?.to_string();
    let current_on = fan_power(fan_ent, unrec, &fan_id)?;
    let temp = temp?;
    let present = present?;

    // Only the on-path. Person absent or temp ≤ 28 → None → fall through.
    if !(present && temp > TEMP_ON) {
        return None;
    }

    Some(Plan {
        fan: fan_id,
        needs_write: !current_on,
        temp: Some(temp),
        present: Some(present),
    })
}

fn find_temp_entity(entities: &[Value]) -> Option<&Value> {
    entities
        .iter()
        .find(|e| TEMP_IDS.contains(&e["id"].as_str().unwrap_or("")))
        .or_else(|| {
            entities.iter().find(|e| {
                first_field(e, TEMP_FIELDS).is_some()
                    || name_has(e, &["wen_du", "temperature", "温度", "ban_gong_shi"])
            })
        })
}

fn find_occ_entity(entities: &[Value]) -> Option<&Value> {
    entities
        .iter()
        .find(|e| OCC_IDS.contains(&e["id"].as_str().unwrap_or("")))
        .or_else(|| {
            entities.iter().find(|e| {
                !is_temp_source(e)
                    && (first_field(e, OCC_FIELDS).is_some()
                        || name_has(
                            e,
                            &["ren_ti", "人体存在", "办公桌", "occupancy", "presence"],
                        ))
            })
        })
}

fn is_temp_source(entity: &Value) -> bool {
    let id = entity["id"].as_str().unwrap_or("");
    TEMP_IDS.contains(&id) || first_field(entity, TEMP_FIELDS).is_some()
}

fn find_fan_entity(entities: &[Value]) -> Option<&Value> {
    entities
        .iter()
        .find(|e| FAN_IDS.contains(&e["id"].as_str().unwrap_or("")))
        .or_else(|| {
            entities.iter().find(|e| {
                e["kind"].as_str() == Some("device")
                    && writable_has(e, "power")
                    && name_has(e, &["fan", "feng_shan", "风扇"])
            })
        })
        .or_else(|| {
            let candidates: Vec<&Value> = entities
                .iter()
                .filter(|e| e["kind"].as_str() == Some("device") && writable_has(e, "power"))
                .collect();
            (candidates.len() == 1).then(|| candidates[0])
        })
}

fn fan_power(entity: &Value, unrec: Option<&Vec<Value>>, fan_id: &str) -> Option<bool> {
    if let Some(rows) = unrec {
        for row in rows {
            if row["entity"].as_str() == Some(fan_id) && row["field"].as_str() == Some("power") {
                let desired = row
                    .get("desired")
                    .and_then(|d| d.get("value"))
                    .unwrap_or(&row["desired"]);
                if let Some(b) = as_boolish(desired) {
                    return Some(b);
                }
            }
        }
    }
    as_boolish(entity["state"].get("power")?)
}

fn first_field<'a>(entity: &'a Value, fields: &[&str]) -> Option<&'a Value> {
    let state = entity.get("state")?;
    for f in fields {
        if let Some(v) = state.get(*f) {
            if !v.is_null() {
                return Some(v);
            }
        }
    }
    None
}

fn writable_has(entity: &Value, field: &str) -> bool {
    entity["writable"]
        .as_array()
        .map(|a| a.iter().any(|v| v.as_str() == Some(field)))
        .unwrap_or(false)
}

fn name_has(entity: &Value, needles: &[&str]) -> bool {
    let id = entity["id"].as_str().unwrap_or("").to_ascii_lowercase();
    let name = entity["name"].as_str().unwrap_or("").to_ascii_lowercase();
    needles.iter().any(|n| {
        let n = n.to_ascii_lowercase();
        id.contains(&n) || name.contains(&n)
    })
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|n| n as f64))
        .or_else(|| v.as_u64().map(|n| n as f64))
        .or_else(|| v.as_str()?.trim().parse().ok())
}

fn as_boolish(v: &Value) -> Option<bool> {
    if let Some(b) = v.as_bool() {
        return Some(b);
    }
    if let Some(n) = as_f64(v) {
        return Some(n > 0.0);
    }
    match v.as_str()?.trim().to_ascii_lowercase().as_str() {
        "on" | "home" | "true" | "occupied" | "present" | "yes" => Some(true),
        "off" | "away" | "false" | "empty" | "absent" | "unoccupied" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(id: &str, kind: &str, name: &str, state: Value, writable: &[&str]) -> Value {
        json!({
            "id": id,
            "kind": kind,
            "name": name,
            "space": "space.office",
            "online": true,
            "state": state,
            "writable": writable,
        })
    }

    fn world(temp: f64, present: bool, fan_on: bool) -> Value {
        json!({
            "entities": [
                entity(
                    TEMP_IDS[0],
                    "sensor",
                    "办公室传感器",
                    json!({"temperature": temp, "occupancy": !present, "illuminance": 200.0}),
                    &[],
                ),
                entity(
                    OCC_IDS[0],
                    "sensor",
                    "人体存在传感器办公桌前 此处有人",
                    json!({"occupancy": present}),
                    &[],
                ),
                entity(
                    FAN_IDS[0],
                    "device",
                    "电风扇",
                    json!({"power": fan_on}),
                    &["power", "fan_speed"],
                ),
            ],
            "unreconciled": [],
            "degraded": false,
        })
    }

    fn ev(field: &str, value: Value) -> Value {
        json!({
            "kind": "context_change",
            "entity": if field == "temperature" { TEMP_IDS[0] } else { OCC_IDS[0] },
            "field": field,
            "value": value,
            "self_originated": false,
        })
    }

    #[test]
    fn hot_and_present_turns_fan_on() {
        let p = plan_action(&world(29.0, true, false), &ev("occupancy", json!(true))).unwrap();
        assert_eq!(p.fan, FAN_IDS[0]);
        assert!(p.needs_write);
        assert_eq!(p.temp, Some(29.0));
        assert_eq!(p.present, Some(true));
    }

    #[test]
    fn hot_and_present_already_on_is_noop() {
        let p = plan_action(&world(29.0, true, true), &ev("temperature", json!(29.0))).unwrap();
        assert!(!p.needs_write);
    }

    #[test]
    fn exactly_28_does_not_act() {
        assert!(plan_action(&world(28.0, true, false), &ev("temperature", json!(28.0))).is_none());
    }

    #[test]
    fn cool_does_not_act() {
        assert!(plan_action(&world(23.6, true, false), &ev("occupancy", json!(true))).is_none());
    }

    #[test]
    fn absent_does_not_act_even_when_hot() {
        assert!(plan_action(&world(32.0, false, false), &ev("occupancy", json!(false))).is_none());
    }

    #[test]
    fn absent_does_not_turn_fan_off() {
        assert!(plan_action(&world(32.0, false, true), &ev("occupancy", json!(false))).is_none());
    }

    #[test]
    fn event_value_overrides_stale_snapshot() {
        let p = plan_action(&world(20.0, true, false), &ev("temperature", json!(29.5))).unwrap();
        assert!(p.needs_write);
        assert_eq!(p.temp, Some(29.5));
    }

    #[test]
    fn occupancy_event_overrides_stale_snapshot() {
        let p = plan_action(&world(29.0, false, false), &ev("occupancy", json!(true))).unwrap();
        assert!(p.needs_write);
        assert_eq!(p.present, Some(true));
    }

    #[test]
    fn desired_unreconciled_beats_debounced_current() {
        let mut w = world(29.0, true, false);
        w["unreconciled"] = json!([{
            "entity": FAN_IDS[0],
            "field": "power",
            "desired": { "value": true }
        }]);
        let p = plan_action(&w, &ev("occupancy", json!(true))).unwrap();
        assert!(!p.needs_write, "already asked the fan to turn on");
    }

    #[test]
    fn missing_occupancy_does_not_act() {
        let w = json!({
            "entities": [
                entity(TEMP_IDS[0], "sensor", "温度", json!({"temperature": 30.0}), &[]),
                entity(FAN_IDS[0], "device", "电风扇", json!({"power": false}), &["power"]),
            ],
            "unreconciled": [],
        });
        assert!(plan_action(&w, &ev("temperature", json!(30.0))).is_none());
    }

    #[test]
    fn unrelated_world_without_fan_does_not_act() {
        let w = json!({
            "entities": [
                entity(TEMP_IDS[0], "sensor", "温度", json!({"temperature": 30.0}), &[]),
                entity(OCC_IDS[0], "sensor", "此处有人", json!({"occupancy": true}), &[]),
            ],
            "unreconciled": [],
        });
        assert!(plan_action(&w, &ev("temperature", json!(30.0))).is_none());
    }

    #[test]
    fn composite_occupancy_is_ignored() {
        // 办公室综合传感器 occupancy=true，桌前人体=false：不应开风扇。
        assert!(plan_action(&world(30.0, false, false), &ev("occupancy", json!(false))).is_none());
    }

    #[test]
    fn trigger_field_gate() {
        let temp_ev = ev("temperature", json!(29.0));
        let occ_ev = ev("occupancy", json!(true));
        let composite_occ = json!({
            "kind": "context_change",
            "entity": TEMP_IDS[0],
            "field": "occupancy",
            "value": true,
        });
        let power = json!({
            "kind": "context_change",
            "entity": FAN_IDS[0],
            "field": "power",
            "value": true,
        });
        let illuminance = json!({
            "kind": "context_change",
            "entity": TEMP_IDS[0],
            "field": "illuminance",
            "value": 40.0,
        });
        assert!(is_auto_fan_event(&temp_ev));
        assert!(is_auto_fan_event(&occ_ev));
        assert!(
            !is_auto_fan_event(&composite_occ),
            "办公室综合传感器的 occupancy 不能当人在"
        );
        assert!(!is_auto_fan_event(&power));
        assert!(
            !is_auto_fan_event(&illuminance),
            "综合传感器的照度变化不是本策略的触发"
        );
    }
}
