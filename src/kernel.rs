//! The kernel main loop. Single-threaded and synchronous on purpose: task
//! serialization is the first layer of concurrency arbitration, and every
//! path through the loop terminates behind a budget, a breaker or a lock.
//!
//! Degradation chain per task:
//!   strategy(rule) → model → format retries → breaker → strategy(fallback)
//!   → safe reject. Every stage is deterministic.

use crate::breaker::Breaker;
use crate::config::{BackendConfig, Config};
use crate::context::Context;
use crate::event::{Event, EventQueue};
use crate::inference::{
    ImagePart, InferenceBackend, LitertLmBackend, MockBackend, OpenAiBackend, UserInput,
};
use crate::lock::ResourceLocks;
use crate::plugin::abi::{ContextView, PluginInput, PluginOutput};
use crate::plugin::manifest::parse_pubkey;
use crate::plugin::native::{NativePlugin, NativeRegistry};
use crate::plugin::registry::{Registry, ScanOptions};
use crate::plugin::runtime::{HostBridge, NullBridge, PluginRuntime};
use base64::Engine as _;
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Serialize)]
pub struct TaskOutcome {
    /// "ok" | "rejected" | "busy" | "error"
    pub status: String,
    pub reply: String,
    /// Which path produced the answer: "rule" | "model" | "tool" | "fallback" | "kernel"
    pub via: String,
}

impl TaskOutcome {
    fn new(status: &str, via: &str, reply: impl Into<String>) -> Self {
        Self {
            status: status.into(),
            reply: reply.into(),
            via: via.into(),
        }
    }
}

/// The only two shapes the model may answer with. Anything else is rejected
/// and retried (输出格式强制纠偏 — 内核原生 JSON check).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelCommand {
    #[serde(default)]
    reply: Option<String>,
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    args: serde_json::Value,
}

/// How the kernel reaches a model. Native trait impls ship with the crate;
/// the Plugin path routes inference through the wasm runtime like everything
/// else — the model is a replaceable runtime part, not kernel substance.
enum InferencePath {
    Native(Box<dyn InferenceBackend>),
    Plugin(String),
}

pub struct Kernel {
    cfg: Config,
    pub queue: EventQueue,
    ctx: Context,
    locks: ResourceLocks,
    breaker: Breaker,
    infer: InferencePath,
    runtime: PluginRuntime,
    registry: Registry,
    /// In-process Rust plugins. Same dispatch as wasm; native wins on a
    /// name collision (in-tree code beats disk artifacts).
    native: NativeRegistry,
    bridge: Arc<dyn HostBridge>,
    plugins_dir: PathBuf,
    pubkey: Option<VerifyingKey>,
    task_seq: u64,
    /// Rollbacks detected mid-invocation; `on_rollback` hooks fire at the
    /// next safe boundary to avoid re-entering the plugin runtime.
    pending_rollbacks: Vec<String>,
}

/// Assembles a kernel: config, hardware bridge, an optional injected
/// inference backend (overrides `cfg.backend`), and native plugins — which
/// must be registered before construction so lifecycle hooks like
/// `kernel_start` see them.
pub struct KernelBuilder {
    cfg: Config,
    bridge: Option<Arc<dyn HostBridge>>,
    backend_override: Option<Box<dyn InferenceBackend>>,
    native: NativeRegistry,
}

impl KernelBuilder {
    pub fn bridge(mut self, bridge: Arc<dyn HostBridge>) -> Self {
        self.bridge = Some(bridge);
        self
    }

    /// Inject a custom `InferenceBackend` (own FFI binding, vendor NPU
    /// runtime...). Takes precedence over `cfg.backend`.
    pub fn backend(mut self, backend: Box<dyn InferenceBackend>) -> Self {
        self.backend_override = Some(backend);
        self
    }

    /// Register a native tool. `devices` lists the `device:*` resources
    /// locked for the duration of each invocation.
    pub fn register_tool(
        mut self,
        name: &str,
        devices: &[&str],
        plugin: impl NativePlugin + 'static,
    ) -> Self {
        self.native.register_tool(name, devices, plugin);
        self
    }

    pub fn register_strategy(mut self, name: &str, plugin: impl NativePlugin + 'static) -> Self {
        self.native.register_strategy(name, plugin);
        self
    }

    pub fn register_hook(
        mut self,
        name: &str,
        points: &[&str],
        plugin: impl NativePlugin + 'static,
    ) -> Self {
        self.native.register_hook(name, points, plugin);
        self
    }

    pub fn build(self) -> anyhow::Result<Kernel> {
        Kernel::assemble(self.cfg, self.bridge, self.backend_override, self.native)
    }
}

impl Kernel {
    pub fn builder(cfg: Config) -> KernelBuilder {
        KernelBuilder {
            cfg,
            bridge: None,
            backend_override: None,
            native: NativeRegistry::default(),
        }
    }

    pub fn new(cfg: Config, bridge: Option<Arc<dyn HostBridge>>) -> anyhow::Result<Self> {
        Self::assemble(cfg, bridge, None, NativeRegistry::default())
    }

    fn assemble(
        cfg: Config,
        bridge: Option<Arc<dyn HostBridge>>,
        backend_override: Option<Box<dyn InferenceBackend>>,
        native: NativeRegistry,
    ) -> anyhow::Result<Self> {
        let infer = match backend_override {
            Some(backend) => InferencePath::Native(backend),
            None => match &cfg.backend {
                BackendConfig::Mock => InferencePath::Native(Box::new(MockBackend)),
                BackendConfig::Openai {
                    url,
                    model,
                    api_key,
                } => InferencePath::Native(Box::new(OpenAiBackend {
                    url: url.clone(),
                    model: model.clone(),
                    api_key: api_key.clone(),
                })),
                BackendConfig::LitertLm {
                    binary,
                    model_path,
                    accelerator,
                    image_arg,
                    extra_args,
                } => InferencePath::Native(Box::new(LitertLmBackend {
                    binary: binary.clone(),
                    model_path: model_path.clone(),
                    accelerator: accelerator.clone(),
                    image_arg: image_arg.clone(),
                    extra_args: extra_args.clone(),
                })),
                BackendConfig::Plugin { name } => InferencePath::Plugin(name.clone()),
            },
        };
        let pubkey = match &cfg.trusted_pubkey {
            Some(hex_key) => Some(parse_pubkey(hex_key)?),
            None => None,
        };
        let runtime = PluginRuntime::new()?;
        let plugins_dir = PathBuf::from(&cfg.plugins_dir);
        let registry = Registry::scan(
            &runtime,
            &ScanOptions {
                plugins_dir: &plugins_dir,
                pubkey: pubkey.as_ref(),
                allow_unsigned: cfg.dev_allow_unsigned,
            },
        );
        let mut kernel = Self {
            ctx: Context::new(cfg.context_max_bytes),
            breaker: Breaker::new(cfg.breaker_max_failures, cfg.breaker_max_repeats),
            queue: EventQueue::default(),
            locks: ResourceLocks::default(),
            infer,
            runtime,
            registry,
            native,
            bridge: bridge.unwrap_or_else(|| Arc::new(NullBridge)),
            plugins_dir,
            pubkey,
            cfg,
            task_seq: 0,
            pending_rollbacks: Vec::new(),
        };
        let loaded = kernel.registry.names();
        kernel.run_hooks("kernel_start", None, serde_json::json!({ "plugins": loaded }));
        kernel.drain_rollback_hooks();
        Ok(kernel)
    }

    /// Graceful stop: fires `kernel_stop` hooks so plugins can flush state.
    pub fn shutdown(&mut self) {
        self.run_hooks("kernel_stop", None, serde_json::json!({}));
        self.drain_rollback_hooks();
    }

    /// Drain the queue. Business code pushes events and calls this; the
    /// stdin runner in main.rs does the same.
    pub fn run_pending(&mut self) -> Vec<TaskOutcome> {
        let mut outcomes = vec![];
        while let Some(ev) = self.queue.pop() {
            outcomes.push(self.handle_event(ev));
        }
        outcomes
    }

    pub fn handle_event(&mut self, event: Event) -> TaskOutcome {
        self.task_seq += 1;
        let task = self.task_seq;

        // Kernel-reserved event: silent hot update, applied between tasks.
        if event.kind == "plugin_reload" {
            let name = event.payload["name"].as_str().unwrap_or_default().to_string();
            let opts = ScanOptions {
                plugins_dir: &self.plugins_dir,
                pubkey: self.pubkey.as_ref(),
                allow_unsigned: self.cfg.dev_allow_unsigned,
            };
            let opts_result = self.registry.reload(&self.runtime, &opts, &name);
            return match opts_result {
                Ok(()) => TaskOutcome::new("ok", "kernel", format!("plugin '{name}' reloaded")),
                Err(e) => TaskOutcome::new("error", "kernel", format!("reload failed: {e:#}")),
            };
        }

        self.run_hooks("pre_task", Some(&event), serde_json::json!({}));
        let outcome = self.execute(task, &event);
        self.locks.release_task(task);
        self.run_hooks(
            "post_task",
            Some(&event),
            serde_json::json!({ "outcome": {
                "status": &outcome.status, "reply": &outcome.reply, "via": &outcome.via,
            }}),
        );
        self.drain_rollback_hooks();

        if event.kind == "command" {
            // Only text enters the bounded context; images are per-turn.
            let text = event
                .payload
                .as_str()
                .or_else(|| event.payload.get("text").and_then(|t| t.as_str()));
            if let Some(text) = text {
                self.ctx.push("user", text);
            }
            self.ctx.push("assistant", &outcome.reply);
        }
        outcome
    }

    fn execute(&mut self, task: u64, event: &Event) -> TaskOutcome {
        // 1) Strategy first: deterministic rules may finish the task with no model.
        let strategy_out = self.run_strategy(event, "route");
        let decision = match &strategy_out {
            Some(out) if out.ok => out.decision.clone().unwrap_or_else(|| "model".into()),
            Some(_) => "model".into(), // strategy failed: model path is the default
            None => "none".into(),     // no strategy plugin loaded
        };
        self.run_hooks(
            "post_route",
            Some(event),
            serde_json::json!({ "decision": &decision }),
        );
        if decision == "rule" {
            if let Some(out) = strategy_out {
                self.breaker.record_success();
                return TaskOutcome::new(
                    "ok",
                    "rule",
                    out.reply.unwrap_or_else(|| out.result.to_string()),
                );
            }
        }

        // 2) Breaker gate before spending any inference budget.
        if self.breaker.is_open() {
            let out = self.fallback(event, "breaker open");
            self.breaker.reset();
            return out;
        }

        // 3) Model path with format enforcement. The payload may be plain
        // text or a multimodal object ({"text": ..., "images": [...]}).
        let user_input = match user_input_from_event(event) {
            Ok(u) => u,
            Err(e) => return self.fallback(event, &format!("bad input payload: {e:#}")),
        };
        let system = self.system_prompt();
        let mut last_err = String::new();
        for attempt in 0..=self.cfg.max_format_retries {
            self.run_hooks(
                "pre_infer",
                Some(event),
                serde_json::json!({ "attempt": attempt }),
            );
            let raw = match self.infer_once(&system, &user_input) {
                Ok(r) => r,
                Err(e) => {
                    last_err = format!("inference error: {e:#}");
                    continue;
                }
            };
            // Raw output exposed before validation: the only place a monitor
            // can observe real model behavior on production devices.
            self.run_hooks(
                "post_infer",
                Some(event),
                serde_json::json!({ "attempt": attempt, "raw": &raw }),
            );
            let cmd = match parse_model_command(&raw) {
                Ok(c) => c,
                // Kernel's strict check failed; a configured repair plugin may
                // rewrite the output, but its result must pass the same strict
                // check — the kernel keeps the final say.
                Err(e) => match self.try_repair(&raw, &e) {
                    Some(c) => c,
                    None => {
                        last_err = format!("format violation: {e:#}");
                        continue;
                    }
                },
            };
            // A valid plan exists; audit point before anything executes.
            self.run_hooks(
                "on_plan",
                Some(event),
                serde_json::json!({ "plan": {
                    "reply": &cmd.reply, "tool": &cmd.tool, "args": &cmd.args,
                }}),
            );
            // Dead-loop guard on the planned action.
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (&cmd.reply, &cmd.tool, cmd.args.to_string()).hash(&mut h);
            if self.breaker.record_action(h.finish()) {
                let out = self.fallback(event, "repeated action loop");
                self.breaker.reset();
                return out;
            }
            match (&cmd.reply, &cmd.tool) {
                (Some(reply), None) => {
                    self.breaker.record_success();
                    return TaskOutcome::new("ok", "model", reply.clone());
                }
                (None, Some(tool)) => return self.run_tool(task, tool.clone(), cmd.args, event),
                _ => {
                    last_err = "command must set exactly one of reply/tool".into();
                    continue;
                }
            }
        }

        // 4) Retries exhausted → breaker → fallback.
        self.breaker.record_failure();
        self.fallback(event, &last_err)
    }

    fn run_tool(
        &mut self,
        task: u64,
        tool: String,
        args: serde_json::Value,
        event: &Event,
    ) -> TaskOutcome {
        // Native tools first, then wasm; both declare device resources.
        let devices: Vec<String> = if let Some(devices) = self.native.tool_devices(&tool) {
            devices
        } else if let Some(p) = self.registry.tool(&tool) {
            // 资源锁: every device:* capability the tool declares is locked
            // for the duration of the invocation. Busy => reject, no preemption.
            p.manifest
                .permissions
                .capabilities
                .iter()
                .filter(|c| c.starts_with("device:"))
                .cloned()
                .collect()
        } else {
            self.breaker.record_failure();
            return self.fallback(event, &format!("model asked for unknown tool '{tool}'"));
        };
        if let Err(busy) = self.locks.acquire_all(task, &devices) {
            return TaskOutcome::new("busy", "kernel", format!("resource '{busy}' is busy"));
        }

        // Locks held, about to touch a physical device: the audit boundary.
        self.run_hooks(
            "pre_tool",
            Some(event),
            serde_json::json!({ "tool": &tool, "args": &args }),
        );
        let input = PluginInput {
            kind: "tool".into(),
            hook: None,
            event: Some(serde_json::to_value(event).unwrap_or_default()),
            context: None, // filled by invoke_plugin if permitted
            args,
        };
        match self.invoke_plugin(&tool, input) {
            Ok(out) if out.ok => {
                self.run_hooks(
                    "post_tool",
                    Some(event),
                    serde_json::json!({ "tool": &tool, "ok": true }),
                );
                self.breaker.record_success();
                TaskOutcome::new(
                    "ok",
                    "tool",
                    out.reply.unwrap_or_else(|| out.result.to_string()),
                )
            }
            Ok(out) => {
                let why = out.error.unwrap_or_else(|| "tool reported failure".into());
                self.run_hooks(
                    "post_tool",
                    Some(event),
                    serde_json::json!({ "tool": &tool, "ok": false, "error": &why }),
                );
                self.breaker.record_failure();
                self.fallback(event, &why)
            }
            Err(e) => {
                let why = format!("tool '{tool}' crashed: {e:#}");
                self.run_hooks(
                    "post_tool",
                    Some(event),
                    serde_json::json!({ "tool": &tool, "ok": false, "error": &why }),
                );
                self.breaker.record_failure();
                self.fallback(event, &why)
            }
        }
    }

    /// 规则保底 → 安全拒绝. The strategy plugin gets a chance to answer
    /// deterministically; otherwise the kernel refuses safely and says why.
    fn fallback(&mut self, event: &Event, reason: &str) -> TaskOutcome {
        eprintln!("[kernel] degraded: {reason}");
        // The most important monitoring signal an edge device emits: the
        // deterministic chain had to leave the happy path, and why.
        self.run_hooks(
            "on_degrade",
            Some(event),
            serde_json::json!({ "reason": reason }),
        );
        if let Some(out) = self.run_strategy(event, "fallback") {
            if out.ok && out.decision.as_deref() == Some("rule") {
                return TaskOutcome::new(
                    "ok",
                    "fallback",
                    out.reply.unwrap_or_else(|| out.result.to_string()),
                );
            }
        }
        TaskOutcome::new(
            "rejected",
            "fallback",
            format!("request safely rejected ({reason})"),
        )
    }

    /// One inference attempt via whichever path is configured. The wasm path
    /// goes through the same `invoke_plugin` funnel as everything else:
    /// permission-filtered context, budgets, health accounting.
    fn infer_once(&mut self, system: &str, input: &UserInput) -> anyhow::Result<String> {
        let plugin = match &self.infer {
            InferencePath::Plugin(name) => Some(name.clone()),
            InferencePath::Native(_) => None,
        };
        match plugin {
            None => {
                let ctx: Vec<_> = self.ctx.entries().cloned().collect();
                let InferencePath::Native(backend) = &mut self.infer else {
                    unreachable!()
                };
                backend.generate(system, &ctx, input)
            }
            Some(name) => {
                // Images cross the sandbox boundary as base64 copies.
                let images: Vec<serde_json::Value> = input
                    .images
                    .iter()
                    .map(|img| {
                        serde_json::json!({
                            "mime": img.mime,
                            "b64": base64::engine::general_purpose::STANDARD.encode(&img.data),
                        })
                    })
                    .collect();
                let input_pl = PluginInput {
                    kind: "infer".into(),
                    hook: None,
                    event: None,
                    context: None, // filled by invoke_plugin if the manifest permits
                    args: serde_json::json!({
                        "system": system,
                        "input": input.text,
                        "images": images,
                    }),
                };
                let out = self.invoke_plugin(&name, input_pl)?;
                anyhow::ensure!(
                    out.ok,
                    "inference plugin failed: {}",
                    out.error.unwrap_or_default()
                );
                out.reply
                    .or_else(|| out.result.as_str().map(str::to_string))
                    .ok_or_else(|| anyhow::anyhow!("inference plugin returned no text"))
            }
        }
    }

    /// Optional wasm-based output repair. Whatever the plugin returns is
    /// re-validated by the kernel's own strict parser; repair can only help,
    /// never widen what the kernel accepts.
    fn try_repair(&mut self, raw: &str, err: &anyhow::Error) -> Option<ModelCommand> {
        let name = self.cfg.repair_plugin.clone()?;
        let input = PluginInput {
            kind: "repair".into(),
            hook: None,
            event: None,
            context: None,
            args: serde_json::json!({ "raw": raw, "error": err.to_string() }),
        };
        let out = self.invoke_plugin(&name, input).ok()?;
        if !out.ok {
            return None;
        }
        let fixed = out
            .reply
            .or_else(|| out.result.as_str().map(str::to_string))?;
        parse_model_command(&fixed).ok()
    }

    fn run_strategy(&mut self, event: &Event, phase: &str) -> Option<PluginOutput> {
        // Native strategy takes precedence over a wasm one.
        let name = self
            .native
            .strategy_name()
            .or_else(|| self.registry.strategy().map(|p| p.manifest.name.clone()))?;
        let input = PluginInput {
            kind: "strategy".into(),
            hook: None,
            event: Some(serde_json::to_value(event).unwrap_or_default()),
            context: None,
            args: serde_json::json!({ "phase": phase }),
        };
        self.invoke_plugin(&name, input).ok()
    }

    /// Lifecycle hooks are observers, not interceptors: their output never
    /// alters the deterministic chain, and a failing hook only logs. `data`
    /// carries the phase-specific payload documented in docs/03-usage.md.
    fn run_hooks(&mut self, point: &str, event: Option<&Event>, data: serde_json::Value) {
        let mut names: Vec<String> = self.native.hooks_for(point);
        for p in self.registry.hooks_for(point) {
            let n = p.manifest.name.clone();
            if !names.contains(&n) {
                names.push(n);
            }
        }
        for name in names {
            let input = PluginInput {
                kind: "hook".into(),
                hook: Some(point.into()),
                event: event.map(|e| serde_json::to_value(e).unwrap_or_default()),
                context: None,
                args: data.clone(),
            };
            if let Err(e) = self.invoke_plugin(&name, input) {
                eprintln!("[kernel] hook '{name}'@{point} failed: {e:#}");
            }
        }
    }

    /// Fire deferred `on_rollback` notifications at a safe boundary. Rollbacks
    /// discovered while these hooks run stay queued for the next boundary.
    fn drain_rollback_hooks(&mut self) {
        let pending = std::mem::take(&mut self.pending_rollbacks);
        for plugin in pending {
            self.run_hooks(
                "on_rollback",
                None,
                serde_json::json!({ "plugin": &plugin }),
            );
        }
    }

    /// Single funnel for plugin execution. Native plugins run in-process:
    /// trusted code, full context, no sandbox budget and no disk rollback —
    /// their failures surface to the caller and the task-level breaker.
    /// Wasm plugins get permission-filtered input, a budgeted sandbox run,
    /// strict output, and health accounting with auto-rollback.
    fn invoke_plugin(
        &mut self,
        name: &str,
        mut input: PluginInput,
    ) -> anyhow::Result<PluginOutput> {
        if self.native.contains(name) {
            input.context = Some(
                self.ctx
                    .entries()
                    .map(|e| ContextView {
                        role: e.role.clone(),
                        content: e.content.clone(),
                    })
                    .collect(),
            );
            return self.native.invoke(name, &input);
        }
        let (module, caps, may_see_context) = {
            let p = self
                .registry
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("plugin '{name}' not loaded"))?;
            (
                p.module.clone(),
                p.manifest.permissions.capabilities.clone(),
                p.manifest.permissions.context,
            )
        };
        // Context crosses the sandbox boundary only when the signed manifest
        // declared it, and always as a copy.
        if may_see_context {
            input.context = Some(
                self.ctx
                    .entries()
                    .map(|e| ContextView {
                        role: e.role.clone(),
                        content: e.content.clone(),
                    })
                    .collect(),
            );
        }
        let result = self.runtime.invoke(
            &module,
            name,
            &caps,
            self.bridge.clone(),
            &input,
            self.cfg.plugin_fuel,
            self.cfg.plugin_memory_limit,
        );
        let ok = matches!(&result, Ok(out) if out.ok || out.error.is_none());
        {
            let opts = ScanOptions {
                plugins_dir: &self.plugins_dir,
                pubkey: self.pubkey.as_ref(),
                allow_unsigned: self.cfg.dev_allow_unsigned,
            };
            let rolled_back = self.registry.report_result(
                &self.runtime,
                &opts,
                name,
                ok,
                self.cfg.plugin_max_failures,
            );
            if rolled_back {
                self.pending_rollbacks.push(name.to_string());
            }
        }
        result
    }

    fn system_prompt(&self) -> String {
        // Kernel-reserved plugins (inference / repair) are invoked by the
        // kernel itself and must never appear as model-callable tools.
        let mut tool_names = self.native.tool_names();
        tool_names.extend(self.registry.tool_names());
        tool_names.sort();
        tool_names.dedup();
        tool_names.retain(|n| {
            Some(n.as_str()) != self.cfg.repair_plugin.as_deref()
                && !matches!(&self.infer, InferencePath::Plugin(p) if p == n)
        });
        format!(
            "You are an edge device agent. Reply with ONE JSON object and nothing else.\n\
             Either {{\"reply\": \"<text>\"}} to answer the user,\n\
             or {{\"tool\": \"<name>\", \"args\": {{...}}}} to act.\n\
             Available tools: [{}]. If no tool fits, use reply.",
            tool_names.join(", ")
        )
    }
}

/// Build the multimodal user turn from an event payload. Accepted shapes:
///   "plain text"
///   { "text": "...", "images": [ {"mime": "image/png", "b64": "..."}
///                              | {"path": "/tmp/cam.jpg"} ] }
/// Image bytes are loaded eagerly so a bad path/encoding fails the task
/// up front instead of mid-inference.
fn user_input_from_event(event: &Event) -> anyhow::Result<UserInput> {
    if let Some(text) = event.payload.as_str() {
        return Ok(UserInput::text(text));
    }
    let Some(obj) = event.payload.as_object() else {
        return Ok(UserInput::text(event.payload.to_string()));
    };
    let text = obj
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .to_string();
    let mut images = vec![];
    if let Some(list) = obj.get("images").and_then(|i| i.as_array()) {
        for (i, img) in list.iter().enumerate() {
            if let Some(b64) = img.get("b64").and_then(|b| b.as_str()) {
                let data = base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| anyhow::anyhow!("images[{i}].b64 is not valid base64: {e}"))?;
                let mime = img
                    .get("mime")
                    .and_then(|m| m.as_str())
                    .unwrap_or("image/jpeg")
                    .to_string();
                images.push(ImagePart { mime, data });
            } else if let Some(path) = img.get("path").and_then(|p| p.as_str()) {
                let data = std::fs::read(path)
                    .map_err(|e| anyhow::anyhow!("cannot read images[{i}] '{path}': {e}"))?;
                let mime = img
                    .get("mime")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| mime_from_path(path));
                images.push(ImagePart { mime, data });
            } else {
                anyhow::bail!("images[{i}] needs either 'b64' or 'path'");
            }
        }
    }
    Ok(UserInput { text, images })
}

fn mime_from_path(path: &str) -> String {
    let ext = std::path::Path::new(path)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "png" => "image/png",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        _ => "image/jpeg",
    }
    .to_string()
}

/// Tolerant extraction (models love markdown fences), strict validation.
fn parse_model_command(raw: &str) -> anyhow::Result<ModelCommand> {
    let start = raw.find('{').ok_or_else(|| anyhow::anyhow!("no JSON object in output"))?;
    let end = raw
        .rfind('}')
        .ok_or_else(|| anyhow::anyhow!("no JSON object in output"))?;
    anyhow::ensure!(end > start, "malformed JSON bounds");
    let cmd: ModelCommand = serde_json::from_str(&raw[start..=end])?;
    anyhow::ensure!(
        cmd.reply.is_some() != cmd.tool.is_some(),
        "command must set exactly one of reply/tool"
    );
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_reply_command() {
        let raw = r#"{"reply": "Hello, how can I help you?"}"#;
        let cmd = parse_model_command(raw).unwrap();
        assert_eq!(cmd.reply.as_deref(), Some("Hello, how can I help you?"));
        assert_eq!(cmd.tool, None);
    }

    #[test]
    fn parse_valid_tool_command() {
        let raw = r#"{"tool": "turn_light_on", "args": {"brightness": 80}}"#;
        let cmd = parse_model_command(raw).unwrap();
        assert_eq!(cmd.reply, None);
        assert_eq!(cmd.tool.as_deref(), Some("turn_light_on"));
        assert_eq!(cmd.args["brightness"], 80);
    }

    #[test]
    fn parse_with_markdown_fences() {
        let raw = "Here is the response:\n```json\n{\"reply\": \"Done!\"}\n```\nHope that helps.";
        let cmd = parse_model_command(raw).unwrap();
        assert_eq!(cmd.reply.as_deref(), Some("Done!"));
    }

    #[test]
    fn parse_invalid_shapes() {
        // Both reply and tool
        let raw_both = r#"{"reply": "hi", "tool": "test", "args": {}}"#;
        assert!(parse_model_command(raw_both).is_err());

        // Neither reply nor tool
        let raw_neither = r#"{"args": {}}"#;
        assert!(parse_model_command(raw_neither).is_err());

        // Unknown field
        let raw_unknown = r#"{"reply": "hi", "extra": 123}"#;
        assert!(parse_model_command(raw_unknown).is_err());

        // Not a JSON
        let raw_not_json = "I cannot fulfill this request.";
        assert!(parse_model_command(raw_not_json).is_err());
    }

    #[test]
    fn user_input_multimodal_payload_parsing() {
        // Plain string
        let ev = Event {
            kind: "command".into(),
            payload: serde_json::json!("hi"),
            priority: 0,
            source: "t".into(),
        };
        let u = user_input_from_event(&ev).unwrap();
        assert_eq!(u.text, "hi");
        assert!(u.images.is_empty());

        // Object with base64 image
        let ev = Event {
            kind: "command".into(),
            payload: serde_json::json!({
                "text": "what is this",
                "images": [{ "mime": "image/png", "b64": base64::engine::general_purpose::STANDARD.encode(b"png-bytes") }],
            }),
            priority: 0,
            source: "t".into(),
        };
        let u = user_input_from_event(&ev).unwrap();
        assert_eq!(u.text, "what is this");
        assert_eq!(u.images.len(), 1);
        assert_eq!(u.images[0].mime, "image/png");
        assert_eq!(u.images[0].data, b"png-bytes");

        // Bad base64 fails up front
        let ev = Event {
            kind: "command".into(),
            payload: serde_json::json!({ "text": "x", "images": [{ "b64": "!!!" }] }),
            priority: 0,
            source: "t".into(),
        };
        assert!(user_input_from_event(&ev).is_err());

        // Path-based image with mime inferred from extension
        let img_path = std::env::temp_dir().join(format!("ea-kernel-test-{}.png", std::process::id()));
        std::fs::write(&img_path, b"disk-png").unwrap();
        let ev = Event {
            kind: "command".into(),
            payload: serde_json::json!({ "text": "cam", "images": [{ "path": img_path.to_str().unwrap() }] }),
            priority: 0,
            source: "t".into(),
        };
        let u = user_input_from_event(&ev).unwrap();
        assert_eq!(u.images[0].mime, "image/png");
        assert_eq!(u.images[0].data, b"disk-png");
        let _ = std::fs::remove_file(&img_path);
    }

    #[test]
    fn native_strategy_rule_short_circuits_model() {
        let cfg = Config {
            dev_allow_unsigned: true,
            backend: BackendConfig::Mock,
            ..Default::default()
        };
        let mut kernel = Kernel::builder(cfg)
            .register_strategy("router", |input: &PluginInput| {
                assert_eq!(input.kind, "strategy");
                Ok(PluginOutput::rule("handled by rule"))
            })
            .build()
            .unwrap();
        let outcome = kernel.handle_event(Event {
            kind: "command".into(),
            payload: serde_json::json!("anything"),
            priority: 1,
            source: "test".into(),
        });
        assert_eq!(outcome.status, "ok");
        assert_eq!(outcome.via, "rule");
        assert_eq!(outcome.reply, "handled by rule");
    }

    #[test]
    fn native_tool_dispatch_with_injected_backend() {
        struct ToolCaller;
        impl crate::inference::InferenceBackend for ToolCaller {
            fn generate(
                &mut self,
                system: &str,
                _context: &[crate::context::ContextEntry],
                _input: &UserInput,
            ) -> anyhow::Result<String> {
                // Native tools must be advertised to the model.
                assert!(system.contains("beeper"));
                Ok(r#"{"tool": "beeper", "args": {"hz": 440}}"#.into())
            }
        }
        let cfg = Config {
            dev_allow_unsigned: true,
            backend: BackendConfig::Mock, // overridden by injected backend
            ..Default::default()
        };
        let mut kernel = Kernel::builder(cfg)
            .backend(Box::new(ToolCaller))
            .register_tool("beeper", &["device:buzzer"], |input: &PluginInput| {
                assert_eq!(input.kind, "tool");
                assert_eq!(input.args["hz"], 440);
                Ok(PluginOutput::reply("beeped at 440Hz"))
            })
            .build()
            .unwrap();
        let outcome = kernel.handle_event(Event {
            kind: "command".into(),
            payload: serde_json::json!("beep please"),
            priority: 1,
            source: "test".into(),
        });
        assert_eq!(outcome.status, "ok");
        assert_eq!(outcome.via, "tool");
        assert_eq!(outcome.reply, "beeped at 440Hz");
    }

    #[test]
    fn native_hook_observes_lifecycle() {
        use std::sync::{Arc as StdArc, Mutex};
        let seen: StdArc<Mutex<Vec<String>>> = StdArc::new(Mutex::new(vec![]));
        let seen_in_hook = seen.clone();
        let cfg = Config {
            dev_allow_unsigned: true,
            backend: BackendConfig::Mock,
            ..Default::default()
        };
        let mut kernel = Kernel::builder(cfg)
            .register_hook(
                "audit",
                &["kernel_start", "pre_task", "post_task"],
                move |input: &PluginInput| {
                    seen_in_hook
                        .lock()
                        .unwrap()
                        .push(input.hook.clone().unwrap_or_default());
                    Ok(PluginOutput::result(serde_json::Value::Null))
                },
            )
            .build()
            .unwrap();
        kernel.handle_event(Event {
            kind: "command".into(),
            payload: serde_json::json!("hello"),
            priority: 1,
            source: "test".into(),
        });
        let seen = seen.lock().unwrap();
        assert_eq!(*seen, vec!["kernel_start", "pre_task", "post_task"]);
    }

    #[test]
    fn kernel_mock_event_handling() {
        let cfg = Config {
            dev_allow_unsigned: true,
            backend: BackendConfig::Mock,
            ..Default::default()
        };
        let mut kernel = Kernel::new(cfg, None).unwrap();
        let ev = Event {
            kind: "command".into(),
            payload: serde_json::json!("turn off the light"),
            priority: 1,
            source: "test".into(),
        };
        let outcome = kernel.handle_event(ev);
        assert_eq!(outcome.status, "ok");
        assert_eq!(outcome.via, "model");
        assert_eq!(outcome.reply, "[mock] turn off the light");
    }
}

