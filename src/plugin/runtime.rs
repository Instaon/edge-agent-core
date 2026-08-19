//! Wasm sandbox execution. Fresh store per invocation (no state leaks across
//! calls), fuel budget as execution-time quota, hard linear-memory cap.
//! The sandbox has exactly two doors to the outside world: `host_log` and the
//! capability-gated `host_call`. Everything else is unreachable by construction.

use super::abi::{
    self, HostCallRequest, PluginInput, PluginOutput, GUEST_ALLOC, GUEST_HANDLE, HOST_CALL,
    HOST_LOG, HOST_MODULE,
};
use anyhow::{anyhow, bail, Context};
use std::collections::HashSet;
use std::sync::Arc;
use wasmtime::{Caller, Config as WtConfig, Engine, Linker, Module, Store, StoreLimits,
    StoreLimitsBuilder, TypedFunc};

/// Business-side capability provider. The kernel routes every approved
/// `host_call` here; registering providers is how tool plugins reach real
/// hardware / local network.
pub trait HostBridge: Send + Sync {
    fn call(
        &self,
        plugin: &str,
        cap: &str,
        op: Option<&str>,
        args: &serde_json::Value,
    ) -> anyhow::Result<serde_json::Value>;
}

/// Default bridge: nothing is reachable. Keeps the kernel runnable stand-alone.
pub struct NullBridge;
impl HostBridge for NullBridge {
    fn call(
        &self,
        _plugin: &str,
        cap: &str,
        _op: Option<&str>,
        _args: &serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        bail!("no provider registered for capability '{cap}'")
    }
}

struct StoreData {
    limits: StoreLimits,
    plugin_name: String,
    allowed_caps: HashSet<String>,
    bridge: Arc<dyn HostBridge>,
}

pub struct PluginRuntime {
    engine: Engine,
    linker: Linker<StoreData>,
}

impl PluginRuntime {
    pub fn new() -> anyhow::Result<Self> {
        let mut cfg = WtConfig::new();
        cfg.consume_fuel(true);
        let engine = Engine::new(&cfg)?;
        let mut linker: Linker<StoreData> = Linker::new(&engine);

        linker.func_wrap(
            HOST_MODULE,
            HOST_LOG,
            |mut caller: Caller<'_, StoreData>, ptr: i32, len: i32| {
                let msg = read_guest_bytes(&mut caller, ptr, len).unwrap_or_default();
                let name = caller.data().plugin_name.clone();
                eprintln!("[plugin:{name}] {}", String::from_utf8_lossy(&msg));
            },
        )?;

        linker.func_wrap(
            HOST_MODULE,
            HOST_CALL,
            |mut caller: Caller<'_, StoreData>, ptr: i32, len: i32| -> i64 {
                let resp = handle_host_call(&mut caller, ptr, len);
                let resp_json = match resp {
                    Ok(v) => serde_json::json!({ "ok": true, "data": v }),
                    Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
                };
                match write_guest_bytes(&mut caller, resp_json.to_string().as_bytes()) {
                    Ok((p, l)) => abi::pack_ptr_len(p, l),
                    Err(_) => 0,
                }
            },
        )?;

        Ok(Self { engine, linker })
    }

    pub fn compile(&self, wasm: &[u8]) -> anyhow::Result<Module> {
        Module::new(&self.engine, wasm)
    }

    pub fn invoke(
        &self,
        module: &Module,
        plugin_name: &str,
        allowed_caps: &[String],
        bridge: Arc<dyn HostBridge>,
        input: &PluginInput,
        fuel: u64,
        memory_limit: usize,
    ) -> anyhow::Result<PluginOutput> {
        let data = StoreData {
            limits: StoreLimitsBuilder::new()
                .memory_size(memory_limit)
                .memories(1)
                .build(),
            plugin_name: plugin_name.to_string(),
            allowed_caps: allowed_caps.iter().cloned().collect(),
            bridge,
        };
        let mut store = Store::new(&self.engine, data);
        store.limiter(|d| &mut d.limits);
        store.set_fuel(fuel)?;

        let instance = self.linker.instantiate(&mut store, module)?;
        let alloc: TypedFunc<i32, i32> =
            instance.get_typed_func(&mut store, GUEST_ALLOC)?;
        let handle: TypedFunc<(i32, i32), i64> =
            instance.get_typed_func(&mut store, GUEST_HANDLE)?;
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow!("plugin exports no linear memory"))?;

        let input_bytes = serde_json::to_vec(input)?;
        let in_ptr = alloc.call(&mut store, input_bytes.len() as i32)?;
        if in_ptr <= 0 {
            bail!("guest allocator returned null");
        }
        memory.write(&mut store, in_ptr as usize, &input_bytes)?;

        let packed = handle.call(&mut store, (in_ptr, input_bytes.len() as i32))?;
        let (out_ptr, out_len) =
            abi::unpack_ptr_len(packed).ok_or_else(|| anyhow!("plugin reported internal error"))?;
        if out_len as usize > 1024 * 1024 {
            bail!("plugin output exceeds 1MiB cap");
        }
        let mut out = vec![0u8; out_len as usize];
        memory.read(&store, out_ptr as usize, &mut out)?;

        // Strict envelope: unknown fields or malformed JSON => hard error.
        serde_json::from_slice::<PluginOutput>(&out)
            .context("plugin output violates the PluginOutput envelope")
    }
}

fn handle_host_call(
    caller: &mut Caller<'_, StoreData>,
    ptr: i32,
    len: i32,
) -> anyhow::Result<serde_json::Value> {
    let raw = read_guest_bytes(caller, ptr, len)?;
    let req: HostCallRequest =
        serde_json::from_slice(&raw).context("host_call payload violates HostCallRequest")?;
    // 权限收口：per-call adjudication against the signed manifest declaration.
    if !caller.data().allowed_caps.contains(&req.cap) {
        bail!("capability '{}' not declared in manifest — denied", req.cap);
    }
    let bridge = caller.data().bridge.clone();
    let plugin = caller.data().plugin_name.clone();
    bridge.call(&plugin, &req.cap, req.op.as_deref(), &req.args)
}

fn read_guest_bytes(
    caller: &mut Caller<'_, StoreData>,
    ptr: i32,
    len: i32,
) -> anyhow::Result<Vec<u8>> {
    let memory = caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .ok_or_else(|| anyhow!("plugin exports no linear memory"))?;
    if ptr < 0 || len < 0 {
        bail!("negative guest pointer/length");
    }
    let mut buf = vec![0u8; len as usize];
    memory.read(&*caller, ptr as usize, &mut buf)?;
    Ok(buf)
}

/// Write into guest memory via the guest's own allocator (re-entrant call),
/// so the host never picks addresses on the guest's behalf.
fn write_guest_bytes(
    caller: &mut Caller<'_, StoreData>,
    bytes: &[u8],
) -> anyhow::Result<(u32, u32)> {
    let alloc = caller
        .get_export(GUEST_ALLOC)
        .and_then(|e| e.into_func())
        .ok_or_else(|| anyhow!("plugin exports no {GUEST_ALLOC}"))?;
    let alloc: TypedFunc<i32, i32> = alloc.typed(&*caller)?;
    let ptr = alloc.call(&mut *caller, bytes.len() as i32)?;
    if ptr <= 0 {
        bail!("guest allocator returned null");
    }
    let memory = caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .ok_or_else(|| anyhow!("plugin exports no linear memory"))?;
    memory.write(&mut *caller, ptr as usize, bytes)?;
    Ok((ptr as u32, bytes.len() as u32))
}
