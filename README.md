# Edge Agent Core

面向边缘设备的端侧 AI Agent 运行时内核：**最小可用、高确定性、可热更新**。

它让不可靠的端侧小模型（3B–8B）在 2GB–8GB 内存的设备上跑出可靠的设备控制。框架不含任何业务能力，只提供一个坚固的执行循环，和让能力以 Wasm 插件形式安全插进来的机制。

## 架构理念

**内核只提供机制，不提供策略；安全与 Loop 钉死在内核，其余皆可 Wasm 化。**

事件循环、资源锁、熔断、内存预算、验签与权限裁决留在内核——这些是"不死机、不失控"的底线，不能指望插件自律。路由规则、工具能力、生命周期观察、推理接入、格式修复都是可替换的 Wasm 插件，框架自带基础实现保证开箱即用。系统的能力上限由模型和插件决定，行为下限由内核保证。

## 项目结构

```
src/
├── kernel.rs            主循环与降级链（核心，所有路径在此汇合）
├── event.rs             统一事件入口：优先级队列
├── context.rs           对话上下文：硬字节预算，超限丢最旧
├── lock.rs              资源锁：同一设备任一时刻只被一个任务控制
├── breaker.rs           熔断器：连续失败 / 重复动作计数
├── inference.rs         推理后端 trait + Mock / OpenAI 兼容实现
├── plugin/
│   ├── abi.rs           Wasm ABI 契约：输入输出信封、host_call 协议
│   ├── manifest.rs      插件清单 + ed25519 验签（签名覆盖代码与权限声明）
│   ├── registry.rs      插件注册表：扫描、热更、健康统计、自动回滚
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
use edge_agent_core::{Config, Event, Kernel};

fn main() -> anyhow::Result<()> {
    // 1. 初始化内核（使用默认配置或从 json 文件加载）
    let mut kernel = Kernel::new(Config::default(), None)?;

    // 2. 生产事件并推入队列
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

## 文档

- [架构与运行流程](docs/01-architecture.md) — 分层结构、主循环每一步的完整路径
- [核心模块设计与理念](docs/02-core-design.md) — 每个模块为什么这样设计、可替换边界、安全模型落地
- [使用手册](docs/03-usage.md) — 快速上手、Wasm 插件写法、12 个生命周期挂载点、分发规范

