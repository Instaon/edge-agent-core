# 更新日志（Changelog）

本文档记录 `edge-agent-core` 各版本的核心功能变更、配置新增与兼容性说明。

## [0.2.0] - 2026-11-05

### 新增（Added）

- **策略并发能力声明与并行批次调度**（`src/plugin/manifest.rs`、`src/plugin/native.rs`、`src/kernel.rs`）
  - 新增 `Concurrency` 与 `SideEffects`（`none` / `read` / `write`）结构，支持在 Wasm 插件 `manifest.json` 的 `concurrency` 字段或原生策略注册接口 `KernelBuilder::register_strategy_with` / `NativeRegistry::register_strategy_with` 中显式声明策略的并发能力、副作用等级与访问资源列表。
  - 新增 `Config::strategy_max_parallel` 配置项（默认 `4`，设为 `1` 则全局退回串行）。
  - 内核按策略链顺序将连续可并行（`parallel: true` 且 `side_effects != write`）且声明资源互不冲突的策略自动编入同一并行批次，使用 `std::thread::scope` 并发执行，并严格按原策略链顺序归并 `decision`、`observation`、`model_context` 与 Wasm 健康统计，保证并发调度的外部可观测行为与串行完全一致。
  - 副作用与资源访问强约束：
    - `side_effects: "write"` 的策略禁止声明 `parallel: true`（加载/注册期直接拒绝）；
    - `side_effects: "none"` 的并行策略在运行时剥夺全部 `host_call` 能力；
    - `side_effects: "read"` 的并行策略必须将其全部 capability 列入 `concurrency.resources`，且运行时仅放行声明在 `resources` 中的能力；
    - 未声明 `concurrency` 的策略默认视为黑盒（`parallel: false, side_effects: "write"`），独占单批次串行执行，且保持旧版插件签名完全兼容。

- **完整模型请求预算核算与可插拔上下文压缩**（`src/budget.rs`、`src/context.rs`、`src/config.rs`、`src/kernel.rs`）
  - 新增 `src/budget.rs` 模块，在每次模型推理（`infer_once`）前统一核算完整请求字节预算 `RequestBudget`，覆盖：`system prompt` + 对话历史 + 本轮输入文本 + 附带图片开销（`request_image_cost_bytes`）+ 模型输出预留（`response_reserve_bytes`）。
  - 新增配置项：
    - `request_max_bytes`（默认 `32768`，完整模型请求硬上限）
    - `response_reserve_bytes`（默认 `1024`，为模型输出预留的预算）
    - `request_image_cost_bytes`（默认 `1024`，每张输入图片计入请求预算的字节开销）
    - `context_compaction`（`"summarize"` | `"evict"`，默认 `"summarize"`）
    - `compaction_keep_recent`（默认 `4`，摘要压缩时保留最近原文的对话条数）
  - 支持通过 `KernelBuilder::compactor` 注入自定义压缩函数（实现 `ContextCompactor` trait 或传入闭包）。
  - 内核统一压缩与降级链路：
    1. 历史未超预算时原样发送；
    2. 超预算时调用压缩器（默认 `SummarizeCompactor` 调用当前模型生成早期历史摘要并保留最近对话），并对压缩结果执行 `validate_compaction` 严格校验（总字节不超预算、较原历史有效缩减、角色合法、内容非空）；
    3. 压缩失败或校验不通过时，自动退化为 `evict_to_fit` 按**完整最旧消息**逐条淘汰（绝不截断单条消息）；
    4. 若清空全部历史后单次请求固定开销仍超出 `request_max_bytes`，则以 `BudgetExceeded` 明确失败并立即进入降级链（`on_degrade` → fallback 策略 → safe reject），不再发起无意义的重试。
  - 新增 `on_compact` 生命周期挂载点，在请求级压缩或淘汰发生时携带 `{"method": "summary"|"evict", "before_bytes", "after_bytes", "budget_bytes", "error"}` 通知钩子插件。

### 兼容性说明（Compatibility）

- 旧版未包含 `concurrency` 字段的 Wasm 插件 `manifest.json` 及其 ed25519 签名保持 100% 兼容（`concurrency` 为默认值时不参与签名 canonical JSON 拼接）。
- 旧版配置文件无需修改即可直接升级，所有新增配置项均提供安全默认值。

---

## [0.1.0] - 2026-09-28

- 初始版本发布：单线程确定性主循环、优先级事件队列、资源锁、熔断器、有界上下文与闲时压缩、多模态推理后端（Mock / OpenAI / LiteRT-LM / Wasm Plugin）、原生与 Wasm 插件双形态统一调度、ed25519 签名校验与故障自动回滚。
