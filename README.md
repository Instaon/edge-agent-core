# Edge Agent Core

面向边缘设备的端侧 AI Agent 运行时内核：**最小可用、高确定性、可热更新**。

它让不可靠的端侧小模型（3B–8B）在 2GB–8GB 内存的设备上跑出可靠的设备控制。框架不含任何业务能力，只提供一个坚固的执行循环，和让能力安全插进来的机制。输入是多模态的（文字 + 图片），内置两种真实推理后端：OpenAI 兼容协议与 LiteRT-LM。

## 架构理念

**内核只提供机制，不提供策略；安全与 Loop 钉死在内核，其余皆为可替换插件。**

事件循环、资源锁、熔断、内存预算、验签与权限裁决留在内核——这些是"不死机、不失控"的底线，不能指望插件自律。路由规则、工具能力、生命周期观察、推理接入、格式修复都是可替换的插件，框架自带基础实现保证开箱即用。系统的能力上限由模型和插件决定，行为下限由内核保证。

插件有两种执行形态，共用同一套 `PluginInput`/`PluginOutput` 契约与内核调度（工具 / 策略 / 挂载点）：

- **原生 Rust 注册**（默认路径）：业务代码基于本框架开发时，直接实现 `NativePlugin` trait（或一个闭包）并在 `Kernel::builder` 上注册。进程内调用，无沙箱、无验签、无序列化开销——原生代码与内核编译在同一个二进制里，天然可信。
- **Wasm 插件**（热更形态）：Wasm 只是为了运行时动态加载与变更——设备不停机换能力、验签分发、失败自动回滚。不需要热更的能力没有理由付沙箱的开销。

同名冲突时原生注册优先（编译进来的代码优先于磁盘上的产物）。

## 项目结构

```
src/
├── kernel.rs            主循环与降级链（核心，所有路径在此汇合）
├── event.rs             统一事件入口：优先级队列
├── context.rs           对话上下文：硬字节预算，超限丢最旧
├── lock.rs              资源锁：同一设备任一时刻只被一个任务控制
├── breaker.rs           熔断器：连续失败 / 重复动作计数
├── inference.rs         多模态推理后端 trait + Mock / OpenAI 兼容 / LiteRT-LM 实现
├── plugin/
│   ├── abi.rs           插件契约（原生与 Wasm 共用）：输入输出信封、host_call 协议
│   ├── native.rs        原生插件注册表：业务 Rust 代码直接注册 tool/strategy/hook
│   ├── manifest.rs      插件清单 + ed25519 验签（签名覆盖代码与权限声明）
│   ├── registry.rs      Wasm 插件注册表：扫描、热更、健康统计、自动回滚
│   └── runtime.rs       wasmtime 沙箱：燃料配额、内存上限、权限逐次裁决
├── main.rs              最小 runner：stdin 事件进，stdout 结果出
└── bin/ea_pack.rs       分发工具：密钥生成、插件包签名
examples/plugins/        示例 Wasm 插件（策略插件，含完整 ABI 写法）
docs/                    详细文档
```

## 运行流程

内核是一个常驻的单线程主循环，所有事件走同一条**确定性降级链**：

```
事件（业务侧接入）→ 事件队列 → 策略路由
    │
    ├─ 规则命中 ──────────────→ 直接执行（不经过模型，确定性路径）
    │
    └─ 需要模型 → 本地推理 → 格式校验（可挂 Wasm 修复器，内核保留最终裁决）
          ├─ 合法 → 申请资源锁 → 执行工具调用 → 释放锁
          └─ 非法 → 有限次重试
                └─ 重试耗尽 → 熔断 → 规则保底 → 仍无法处理则安全拒绝
```

三条不变量贯穿全程：**内存有预算**（上下文硬上限 + 插件配额，永不 OOM）、**执行有熔断**（失控即中断降级）、**资源有锁**（设备控制无竞争）。主循环的每个阶段边界都暴露为生命周期挂载点（共 12 个），插件按 Manifest 声明的权限在沙箱内运行，越权即拒绝，连续失败自动回滚。

## 快速开始

```bash
cargo build
echo '{"kind":"command","payload":"hello"}' | cargo run --bin edge-agent
```

默认走 Mock 后端，无需模型即可跑通全流程。接入真实模型、编写/签名/分发插件，见使用手册。

## 作为依赖库引入

在你的业务 Rust 项目中，直接通过 `cargo add` 从 Git 仓库引入：

```bash
cargo add --git https://github.com/Instaon/edge-agent-core.git
```

或者手动在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
edge-agent-core = { git = "https://github.com/Instaon/edge-agent-core.git" }
```

### 业务接入示例

```rust
use edge_agent_core::{Config, Event, Kernel, PluginInput, PluginOutput};

fn main() -> anyhow::Result<()> {
    // 1. 组装内核：原生插件在构建期注册（这样 kernel_start 等挂载点能看到它们）
    let mut kernel = Kernel::builder(Config::default())
        // 原生工具：直接写 Rust，声明占用的设备资源锁
        .register_tool("living_room_light", &["device:light0"], |input: &PluginInput| {
            let on = input.args["on"].as_bool().unwrap_or(false);
            // ... 这里直接调 GPIO / 本地总线，无需经过沙箱 ...
            Ok(PluginOutput::reply(if on { "灯已打开" } else { "灯已关闭" }))
        })
        // 原生策略：规则命中就不进模型
        .register_strategy("router", |input: &PluginInput| {
            Ok(PluginOutput::model()) // 或 PluginOutput::rule("确定性回复")
        })
        // 原生挂载点：观察生命周期
        .register_hook("audit", &["post_task", "on_degrade"], |input: &PluginInput| {
            eprintln!("[audit] {:?}", input.hook);
            Ok(PluginOutput::result(serde_json::Value::Null))
        })
        .build()?;

    // 2. 生产事件并推入队列。payload 可以是纯文本，也可以带图片：
    //    {"text": "...", "images": [{"path": "/tmp/cam.jpg"} 或 {"mime": "image/png", "b64": "..."}]}
    kernel.queue.push(Event {
        kind: "command".into(),
        payload: serde_json::json!("打开客厅大灯"),
        priority: 10,
        source: "voice".into(),
    });

    // 3. 执行事件队列并获取结果
    for outcome in kernel.run_pending() {
        println!("status: {}, reply: {}", outcome.status, outcome.reply);
    }

    kernel.shutdown();
    Ok(())
}
```

### 推理后端配置

```jsonc
// OpenAI 兼容（llama.cpp server / ollama / vllm ...），图片走标准 image_url 内容分块
{ "backend": { "type": "openai", "url": "http://127.0.0.1:8000/v1/chat/completions", "model": "qwen2.5-vl" } }

// LiteRT-LM（Google AI Edge 端侧运行时，经 litert_lm_main CLI 驱动）
{ "backend": { "type": "litert_lm", "binary": "/usr/local/bin/litert_lm_main",
               "model_path": "/models/gemma3n.litertlm", "accelerator": "gpu",
               "image_arg": "--image_path" } }
```

也可以完全绕开配置，自己实现 `InferenceBackend` trait（自有 FFI 绑定、厂商 NPU 运行时）后通过 `Kernel::builder(cfg).backend(...)` 注入。

## 文档

- [架构与运行流程](docs/01-architecture.md) — 分层结构、主循环每一步的完整路径
- [核心模块设计与理念](docs/02-core-design.md) — 每个模块为什么这样设计、可替换边界、安全模型落地
- [使用手册](docs/03-usage.md) — 快速上手、多模态接入、原生插件注册、Wasm 插件写法、12 个生命周期挂载点、分发规范

