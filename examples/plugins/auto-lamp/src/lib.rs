//! Occupancy + illuminance lighting strategy.
//!
//! On a world-model change of illuminance / occupancy (presence / motion):
//!   - lux < 70 AND person present AND lamp off  → turn the lamp on immediately
//!   - lux >= 70 AND lamp on                     → turn the lamp off
//!   - person absent AND lamp on                 → turn off only after occupancy
//!     has been *held* (world-model `lifecycle_*` event). Raw `device_change`
//!     ticks from a bouncing mmWave/PIR are treated as keep, not off.
//!
//! `set_desired` failure (device busy, etc.) stays `decision: "rule"` so the
//! model cannot toggle the lamp the other way. Incomplete world data still
//! returns `decision: "model"`. `ok` stays true on those paths so a flaky
//! world-model cannot auto-disable the plugin.
//!
//! Build: cargo build --release --target wasm32-unknown-unknown

use serde_json::{json, Value};

const CAP: &str = "device:space-context";
/// Turn on at or below this lux. Matches the on-site "光线低于 70" rule.
const LUX_ON: f64 = 70.0;
/// Turn off at or above this lux. Same threshold as on: gateway already
/// deadbands illuminance at 30 lux, so we will not flicker around 70.
const LUX_OFF: f64 = 70.0;

/// Live hub ids. Occupancy is the desk presence sensor only — never the
/// office composite (`sensor.app_ban_gong_shi`), which is used for lux.
const LAMP_IDS: &[&str] = &[
    "device.ha_light_xiaomi_cn_2052331724_lamp31_s_2_light",
    "device.welcome_lamp",
];
const OCC_IDS: &[&str] =
    &["sensor.ha_binary_sensor_ren_ti_cun_zai_chuan_gan_qi_ban_gong_zhuo_qian_occupancy"];
const LUX_IDS: &[&str] = &["sensor.app_ban_gong_shi"];

const LUX_FIELDS: &[&str] = &["illuminance"];
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

    let related = is_auto_lamp_event(&payload) || is_lamp_entity_event(&payload);

    if payload["self_originated"].as_bool().unwrap_or(false) {
        if related {
            log(&format!(
                "auto-lamp skip {} reason=self-originated",
                trigger_label(&payload)
            ));
        }
        return fallthrough();
    }
    if !is_world_event(&payload) {
        return fallthrough();
    }
    if !related {
        return fallthrough();
    }
    if !is_auto_lamp_event(&payload) {
        log(&format!(
            "auto-lamp skip {} reason=not a trigger field",
            trigger_label(&payload)
        ));
        return fallthrough();
    }

    let world = call_host(&json!({
        "cap": CAP,
        "op": "get",
        "args": { "entities": focus_ids() }
    }));
    if !world["ok"].as_bool().unwrap_or(false) {
        log(&format!(
            "auto-lamp skip {} reason=get world failed err={}",
            trigger_label(&payload),
            world["error"].as_str().unwrap_or("unknown")
        ));
        return fallthrough();
    }

    let data = world.get("data").cloned().unwrap_or(Value::Null);
    let Some(plan) = plan_action(&data, &payload) else {
        log(&format!(
            "auto-lamp skip {} reason=conditions not met or world incomplete",
            world_view(&data, &payload)
        ));
        return fallthrough();
    };

    let action = if !plan.needs_write {
        "keep"
    } else if plan.power {
        "set_on"
    } else {
        "set_off"
    };
    log(&format!(
        "auto-lamp eval {} want={} action={} decided_lux={:?} decided_present={:?} lux_on={} lux_off={}",
        world_view(&data, &payload),
        if plan.power { "on" } else { "off" },
        action,
        plan.lux,
        plan.present,
        LUX_ON,
        LUX_OFF
    ));

    if !plan.needs_write {
        return json!({"ok": true, "decision": "rule", "reply": "", "thought": "灯就这样。"});
    }

    let written = call_host(&json!({
        "cap": CAP,
        "op": "set_desired",
        "args": {
            "entity": plan.lamp,
            "field": "power",
            "value": plan.power,
            "trace_id": payload["trace_id"].as_str().unwrap_or("")
        }
    }));
    if !written["ok"].as_bool().unwrap_or(false) {
        log(&format!(
            "auto-lamp set_desired failed, keep rule (do not hand to model): {}",
            written["error"].as_str().unwrap_or("unknown")
        ));
        // Already decided. Falling through lets the LLM rewrite power.
        return json!({"ok": true, "decision": "rule", "reply": "", "thought": "灯就这样。"});
    }
    json!({"ok": true, "decision": "rule", "reply": "", "thought": "灯就这样。"})
}

fn fallthrough() -> Value {
    json!({"ok": true, "decision": "model"})
}

fn trigger_label(payload: &Value) -> String {
    format!(
        "trigger={}.{}={}",
        payload["entity"].as_str().unwrap_or("-"),
        payload["field"].as_str().unwrap_or("-"),
        payload["value"]
    )
}

fn is_lamp_entity_event(payload: &Value) -> bool {
    LAMP_IDS.contains(&payload["entity"].as_str().unwrap_or(""))
}

/// Snapshot + triggering event, for one-line eval logs.
fn world_view(world: &Value, event: &Value) -> String {
    let entities = match world["entities"].as_array() {
        Some(a) => a.as_slice(),
        None => return format!("{} entities=missing", trigger_label(event)),
    };
    let lux_ent = find_lux_entity(entities);
    let occ_ent = find_occ_entity(entities);
    let lamp_ent = find_lamp_entity(entities);

    let mut lux = lux_ent.and_then(|e| first_field(e, LUX_FIELDS).and_then(as_f64));
    let mut present = occ_ent.and_then(|e| first_field(e, OCC_FIELDS).and_then(as_boolish));
    let ev_entity = event["entity"].as_str().unwrap_or("");
    let ev_field = event["field"].as_str().unwrap_or("");
    let ev_value = &event["value"];
    let ev_is_lux_entity = lux_ent
        .map(|e| e["id"].as_str() == Some(ev_entity))
        .unwrap_or(false);
    let ev_is_occ_entity = occ_ent
        .map(|e| e["id"].as_str() == Some(ev_entity))
        .unwrap_or(false);
    if LUX_FIELDS.contains(&ev_field) || (ev_is_lux_entity && !OCC_FIELDS.contains(&ev_field)) {
        if let Some(v) = as_f64(ev_value) {
            lux = Some(v);
        }
    }
    if ev_is_occ_entity && OCC_FIELDS.contains(&ev_field) {
        if let Some(v) = as_boolish(ev_value) {
            present = Some(v);
        }
    }

    let lamp_id = lamp_ent.and_then(|e| e["id"].as_str()).unwrap_or("-");
    let unrec = world["unreconciled"].as_array();
    let (current, src) = match lamp_ent {
        Some(e) => lamp_power_src(e, unrec, lamp_id),
        None => (None, "missing"),
    };
    format!(
        "{} lux={:?} present={:?} lamp={} current={:?} current_src={}",
        trigger_label(event),
        lux,
        present,
        lamp_id,
        current,
        src
    )
}

fn lamp_power_src(
    entity: &Value,
    unrec: Option<&Vec<Value>>,
    lamp_id: &str,
) -> (Option<bool>, &'static str) {
    if let Some(rows) = unrec {
        for row in rows {
            if row["entity"].as_str() == Some(lamp_id) && row["field"].as_str() == Some("power") {
                let desired = row
                    .get("desired")
                    .and_then(|d| d.get("value"))
                    .unwrap_or(&row["desired"]);
                if let Some(b) = as_boolish(desired) {
                    return (Some(b), "unreconciled");
                }
            }
        }
    }
    match entity["state"].get("power").and_then(as_boolish) {
        Some(b) => (Some(b), "state"),
        None => (None, "missing"),
    }
}

fn is_world_event(payload: &Value) -> bool {
    matches!(
        payload["kind"].as_str().unwrap_or(""),
        "context_change" | "occurrence"
    )
}

fn is_auto_lamp_event(payload: &Value) -> bool {
    let entity = payload["entity"].as_str().unwrap_or("");
    let field = payload["field"].as_str().unwrap_or("");
    if LUX_IDS.contains(&entity) {
        return field.is_empty() || LUX_FIELDS.contains(&field);
    }
    if OCC_IDS.contains(&entity) {
        return field.is_empty() || OCC_FIELDS.contains(&field);
    }
    LUX_FIELDS.contains(&field) || (OCC_FIELDS.contains(&field) && !is_lux_source_id(entity))
}

fn is_lux_source_id(entity: &str) -> bool {
    LUX_IDS.contains(&entity)
}

fn focus_ids() -> Vec<&'static str> {
    let mut ids = Vec::new();
    ids.extend_from_slice(LAMP_IDS);
    ids.extend_from_slice(OCC_IDS);
    ids.extend_from_slice(LUX_IDS);
    ids
}

struct Plan {
    lamp: String,
    power: bool,
    needs_write: bool,
    lux: Option<f64>,
    present: Option<bool>,
}

fn plan_action(world: &Value, event: &Value) -> Option<Plan> {
    let entities = world["entities"].as_array()?;
    let unrec = world["unreconciled"].as_array();

    let lux_ent = find_lux_entity(entities);
    let occ_ent = find_occ_entity(entities);
    let lamp_ent = find_lamp_entity(entities)?;

    let mut lux = lux_ent.and_then(|e| first_field(e, LUX_FIELDS).and_then(as_f64));
    let mut present = occ_ent.and_then(|e| first_field(e, OCC_FIELDS).and_then(as_boolish));

    // The triggering event is newer than the 10s snapshot cache.
    let ev_entity = event["entity"].as_str().unwrap_or("");
    let ev_field = event["field"].as_str().unwrap_or("");
    let ev_value = &event["value"];
    let ev_is_lux_entity = lux_ent
        .map(|e| e["id"].as_str() == Some(ev_entity))
        .unwrap_or(false);
    let ev_is_occ_entity = occ_ent
        .map(|e| e["id"].as_str() == Some(ev_entity))
        .unwrap_or(false);
    if LUX_FIELDS.contains(&ev_field) || (ev_is_lux_entity && !OCC_FIELDS.contains(&ev_field)) {
        if let Some(v) = as_f64(ev_value) {
            lux = Some(v);
        }
    }
    if ev_is_occ_entity && OCC_FIELDS.contains(&ev_field) {
        if let Some(v) = as_boolish(ev_value) {
            present = Some(v);
        }
    }

    let lamp_id = lamp_ent["id"].as_str()?.to_string();
    let current_on = lamp_power(lamp_ent, unrec, &lamp_id)?;
    let lux = lux?;
    let present = present?;
    let occ_held = occupancy_is_held(event);

    let want_on = present && lux < LUX_ON;
    let want_off_bright = lux >= LUX_OFF;
    let want_off_absent = !present && occ_held;
    let target = if want_on {
        true
    } else if want_off_bright || want_off_absent {
        false
    } else {
        // Occupancy bounce while still dark: keep the current lamp state
        // and stay on the rule path so the LLM cannot toggle it.
        return Some(Plan {
            lamp: lamp_id,
            power: current_on,
            needs_write: false,
            lux: Some(lux),
            present: Some(present),
        });
    };

    Some(Plan {
        lamp: lamp_id,
        power: target,
        needs_write: current_on != target,
        lux: Some(lux),
        present: Some(present),
    })
}

/// World-model lifecycle events have already been held by origin-space-context
/// debounce. Gateway `device_change` ticks have not — those are the bounce.
fn occupancy_is_held(event: &Value) -> bool {
    event["event_type"]
        .as_str()
        .unwrap_or("")
        .starts_with("lifecycle")
}

fn find_lux_entity(entities: &[Value]) -> Option<&Value> {
    entities
        .iter()
        .find(|e| LUX_IDS.contains(&e["id"].as_str().unwrap_or("")))
        .or_else(|| {
            entities.iter().find(|e| {
                first_field(e, LUX_FIELDS).is_some()
                    || name_has(e, &["zhao_du", "illuminance", "lux", "照度"])
            })
        })
}

fn find_occ_entity(entities: &[Value]) -> Option<&Value> {
    entities
        .iter()
        .find(|e| OCC_IDS.contains(&e["id"].as_str().unwrap_or("")))
        .or_else(|| {
            entities.iter().find(|e| {
                !is_lux_source(e)
                    && (first_field(e, OCC_FIELDS).is_some()
                        || name_has(
                            e,
                            &["ren_ti", "人体存在", "办公桌", "occupancy", "presence"],
                        ))
            })
        })
}

fn is_lux_source(entity: &Value) -> bool {
    let id = entity["id"].as_str().unwrap_or("");
    LUX_IDS.contains(&id) || first_field(entity, LUX_FIELDS).is_some()
}

fn find_lamp_entity(entities: &[Value]) -> Option<&Value> {
    entities
        .iter()
        .find(|e| LAMP_IDS.contains(&e["id"].as_str().unwrap_or("")))
        .or_else(|| {
            entities.iter().find(|e| {
                e["kind"].as_str() == Some("device")
                    && writable_has(e, "power")
                    && name_has(e, &["lamp", "light", "灯", "welcome"])
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

fn lamp_power(entity: &Value, unrec: Option<&Vec<Value>>, lamp_id: &str) -> Option<bool> {
    if let Some(rows) = unrec {
        for row in rows {
            if row["entity"].as_str() == Some(lamp_id) && row["field"].as_str() == Some("power") {
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

    fn world(lux: f64, present: bool, lamp_on: bool) -> Value {
        json!({
            "entities": [
                entity(
                    LUX_IDS[0],
                    "sensor",
                    "办公室传感器",
                    json!({"illuminance": lux, "occupancy": !present}),
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
                    LAMP_IDS[0],
                    "device",
                    "迎宾灯",
                    json!({"power": lamp_on}),
                    &["power"],
                ),
            ],
            "unreconciled": [],
            "degraded": false,
        })
    }

    fn ev(field: &str, value: Value) -> Value {
        json!({
            "kind": "context_change",
            "entity": if field == "illuminance" { LUX_IDS[0] } else { OCC_IDS[0] },
            "field": field,
            "value": value,
            "self_originated": false,
        })
    }

    fn held_occ(value: Value) -> Value {
        let mut e = ev("occupancy", value);
        e["event_type"] = json!("lifecycle_began");
        e
    }

    fn bounce_occ(value: Value) -> Value {
        let mut e = ev("occupancy", value);
        e["event_type"] = json!("device_change");
        e
    }

    #[test]
    fn dark_and_present_turns_lamp_on() {
        let p = plan_action(&world(40.0, true, false), &ev("illuminance", json!(40.0))).unwrap();
        assert_eq!(p.lamp, LAMP_IDS[0]);
        assert!(p.power);
        assert!(p.needs_write);
    }

    #[test]
    fn dark_and_present_already_on_is_noop() {
        let p = plan_action(&world(40.0, true, true), &ev("occupancy", json!(true))).unwrap();
        assert!(p.power);
        assert!(!p.needs_write);
    }

    #[test]
    fn person_left_turns_lamp_off() {
        let p = plan_action(&world(40.0, false, true), &held_occ(json!(false))).unwrap();
        assert!(!p.power);
        assert!(p.needs_write);
    }

    #[test]
    fn bouncing_occupancy_false_keeps_lamp_on() {
        let p = plan_action(&world(40.0, false, true), &bounce_occ(json!(false))).unwrap();
        assert!(
            p.power && !p.needs_write,
            "raw device_change occupancy=false is sensor bounce, not leave"
        );
    }

    #[test]
    fn brighter_than_threshold_turns_lamp_off() {
        let p = plan_action(&world(90.0, true, true), &ev("illuminance", json!(90.0))).unwrap();
        assert!(!p.power);
        assert!(p.needs_write);
    }

    #[test]
    fn event_value_overrides_stale_snapshot() {
        // snapshot still says 100 lux; the event is the crossing below 70.
        let p = plan_action(&world(100.0, true, false), &ev("illuminance", json!(50.0))).unwrap();
        assert!(p.power);
        assert_eq!(p.lux, Some(50.0));
    }

    #[test]
    fn desired_unreconciled_beats_debounced_current() {
        let mut w = world(40.0, true, false);
        w["unreconciled"] = json!([{
            "entity": LAMP_IDS[0],
            "field": "power",
            "desired": { "value": true }
        }]);
        let p = plan_action(&w, &ev("occupancy", json!(true))).unwrap();
        assert!(p.power);
        assert!(!p.needs_write, "already asked the lamp to turn on");
    }

    #[test]
    fn missing_occupancy_does_not_act() {
        let w = json!({
            "entities": [
                entity(LUX_IDS[0], "sensor", "照度", json!({"illuminance": 40.0}), &[]),
                entity(LAMP_IDS[0], "device", "迎宾灯", json!({"power": false}), &["power"]),
            ],
            "unreconciled": [],
        });
        assert!(plan_action(&w, &ev("illuminance", json!(40.0))).is_none());
    }

    #[test]
    fn unrelated_world_without_lamp_does_not_act() {
        let w = json!({
            "entities": [
                entity(LUX_IDS[0], "sensor", "照度", json!({"illuminance": 40.0}), &[]),
                entity(OCC_IDS[0], "sensor", "此处有人", json!({"occupancy": true}), &[]),
            ],
            "unreconciled": [],
        });
        assert!(plan_action(&w, &ev("illuminance", json!(40.0))).is_none());
    }

    #[test]
    fn composite_occupancy_is_ignored() {
        // 办公室综合传感器 occupancy=true，桌前人体=false：应以桌前为准关灯。
        let p = plan_action(&world(40.0, false, true), &held_occ(json!(false))).unwrap();
        assert!(!p.present.unwrap());
        assert!(!p.power);
        assert!(p.needs_write);
    }

    #[test]
    fn trigger_field_gate() {
        let lux_ev = ev("illuminance", json!(1));
        let occ_ev = ev("occupancy", json!(true));
        let composite_occ = json!({
            "kind": "context_change",
            "entity": LUX_IDS[0],
            "field": "occupancy",
            "value": true,
        });
        let power = json!({
            "kind": "context_change",
            "entity": LAMP_IDS[0],
            "field": "power",
            "value": true,
        });
        assert!(is_auto_lamp_event(&lux_ev));
        assert!(is_auto_lamp_event(&occ_ev));
        assert!(
            !is_auto_lamp_event(&composite_occ),
            "办公室综合传感器的 occupancy 不能当人在"
        );
        assert!(!is_auto_lamp_event(&power));
        assert!(
            is_lamp_entity_event(&power),
            "灯自己的 power 变化要记一笔 skip，方便排查来回开关"
        );
    }
}
