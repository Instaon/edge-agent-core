# 使用手册

面向两类读者：把内核跑起来的集成者，和给设备写能力包的插件开发者。

## 1. 快速上手

### 1.1 环境要求

- Rust stable（`edition = "2024"`，建议 rustc 1.85+）
- 编译插件需要 `wasm32-unknown-unknown` target：

```bash
rustup target add wasm32-unknown-unknown
```

### 1.2 跑起最小内核（无需任何模型）

```bash
cargo build
cp edge-agent.example.json edge-agent.json
echo '{"kind":"command","payload":"hello"}' | cargo run --bin edge-agent
```

默认配置用 `MockBackend`，会原样回显 `[mock] hello`。这一步只是确认内核本身能跑，不代表插件系统在工作——没有 `plugins/` 目录或目录为空时，内核照常运行，只是没有规则直达、没有工具可调。

### 1.3 接入真实模型

编辑 `edge-agent.json`，把 `backend` 换成任何讲 OpenAI chat-completions 协议的本地服务（llama.cpp server、ollama 等）：

```json
{
  "backend": {
    "type": "openai",
    "url": "http://127.0.0.1:8080/v1/chat/completions",
    "model": "qwen2.5-3b-instruct",
    "api_key": null
  }
}
```

如果你的推理库只有 C-ABI（比如直接链接 llama.cpp 的 `.so`），不要等框架内置支持——自己实现 `InferenceBackend` trait（见 [inference.rs](../src/inference.rs)），在 `generate()` 里做 FFI 调用，然后把 `edge-agent-core` 当库用，在 `Kernel::new` 时传入你的实现。`edge-agent` 这个二进制只是给内置后端用的最小 runner，接自定义 trait 实现需要写自己的 `main`。

第三种方式是把推理整个交给 wasm 插件（无需改 Rust 代码，插件可热更）：

```json
{ "backend": { "type": "plugin", "name": "my-infer" } }
```

内核会以 `kind: "infer"` 调用名为 `my-infer` 的插件，`args` 携带 `{"system": "...", "input": "..."}`；插件把模型的原始回答放进 `reply` 返回。推理插件通常需要在 manifest 声明 `context: true`（拿到对话历史）和网络类 capability（经 `host_call` 访问本地推理服务）。它和普通插件走同一套预算/健康统计——连续失败同样会被自动禁用回滚。

### 1.4 注意事项（容易踩的坑）

- **`trusted_pubkey` 为空且 `dev_allow_unsigned=false` 时，任何插件都加载不了。** 内核不会报错崩溃，只是插件列表为空，行为上看起来像"策略/工具都不生效，全部走模型直连"。排查时先看启动日志里的 `[registry]` 行。
- **`dev_allow_unsigned: true` 仅用于开发。** 生产设备上打开这个开关等于放弃零信任验签，framework 不会替你拦截，因为这是显式配置项。
- **`context_max_bytes` 设得太小会让策略/工具插件看到被截断的历史。** 只有声明了 `permissions.context: true` 的插件才受影响；没声明该权限的插件本来就看不到上下文，不受此项影响。
- **一个任务里工具调用失败不会自动重试**，会直接进入降级链（策略 fallback → 安全拒绝）。如果你的工具本身有瞬时性错误（比如串口偶发超时），重试逻辑要写在工具插件内部或者 `HostBridge` 实现里，内核不做这个决定。
- **`plugin_fuel` 太小会让复杂计算的插件被中途打断**，且不会有清晰的"超时"提示——燃料耗尽在 wasmtime 里表现为一次 trap，`invoke_plugin` 会把它当成插件失败计入健康统计。刚开始调试插件时可以把这个值调大（比如 5 亿），稳定后再收紧。

## 2. 编写一个 Wasm 插件

插件是一个普通的 Rust `cdylib`，编译到 `wasm32-unknown-unknown`。完整可运行的例子在 [examples/plugins/strategy-demo](../examples/plugins/strategy-demo)，本节讲清楚每一步在做什么、为什么必须这样写。

### 2.1 最小骨架

```toml
# Cargo.toml
[package]
name = "my-plugin"
version = "1.0.0"
edition = "2021"

[workspace]          # 独立于宿主工程的 workspace，避免被拉进主 crate 编译

[lib]
crate-type = ["cdylib"]

[dependencies]
serde_json = "1"

[profile.release]
opt-level = "s"       # 边缘设备优先体积
lto = true
strip = true
```

插件必须导出两个 `extern "C"` 函数，这是内核识别插件的唯一契约（见 [plugin/abi.rs](../src/plugin/abi.rs)）：

```rust
#[no_mangle]
pub extern "C" fn ea_alloc(len: i32) -> i32 {
    let mut buf = Vec::<u8>::with_capacity(len as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf); // 所有权交给宿主，实例销毁时随线性内存一起释放
    ptr as i32
}

#[no_mangle]
pub extern "C" fn ea_handle(ptr: i32, len: i32) -> i64 {
    let input = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    let output = serde_json::json!({ "ok": true, "reply": "hello from plugin" });
    let bytes = output.to_string().into_bytes();
    let out_ptr = ea_alloc(bytes.len() as i32) as usize as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr, bytes.len()) };
    ((out_ptr as u64) << 32 | bytes.len() as u64) as i64
}
```

`ea_handle` 的返回值是 `(ptr << 32 | len)` 打包成的 `i64`，`0` 表示内部错误。宿主用这个约定从 guest 内存里读回结果，不需要额外导出"取结果"的函数。

### 2.2 输入长什么样

内核传入的 JSON（对应 `PluginInput`）:

```json
{
  "kind": "tool",              // "tool" | "strategy" | "hook" | "infer" | "repair"（后两者是内核保留调用，见 2.4 末尾）
  "hook": null,                 // kind=hook 时是挂载点名，如 "pre_task"
  "event": { "kind": "command", "payload": "turn on light", "priority": 0, "source": "" },
  "context": null,              // 只有 manifest 声明 permissions.context=true 才非空
  "args": { "...": "..." }      // 工具参数 / 策略阶段信息
}
```

`context`（如果有）是一个数组：`[{"role": "user", "content": "..."}, ...]`，只读副本，改了也不会影响内核真实上下文。

### 2.3 输出必须是什么样

对应 `PluginOutput`，**严格信封，多一个字段都会被整体拒绝**：

```json
{
  "ok": true,
  "result": {},
  "decision": null,
  "reply": "已开灯",
  "error": null
}
```

字段含义：

| 字段 | 何时用 |
| --- | --- |
| `ok` | 必填。`false` 表示本次调用失败，会计入插件健康统计 |
| `result` | 工具/钩子的结构化返回值，被调用方（模型或业务）读取 |
| `decision` | 仅策略插件用："rule"（规则直达，用 `reply` 作为最终答案）或 "model"（交给模型处理） |
| `reply` | 直接面向用户的文本答案 |
| `error` | `ok=false` 时的原因，会出现在内核降级日志里 |

不要在 JSON 里加文档没提到的字段——`deny_unknown_fields` 会让整个输出被当作非法格式丢弃，即使 `ok: true` 也一样。

### 2.4 三类插件怎么写

**工具插件（tool）**：`ea_handle` 收到 `kind: "tool"`，`args` 是模型规划出的参数，执行具体动作后把结果放进 `result`/`reply`。

**策略插件（strategy）**：收到 `args.phase`，取值 `"route"`（每个任务开始时）或 `"fallback"`（降级发生时）。想做规则直达，返回 `{"ok": true, "decision": "rule", "reply": "..."}`；想交给模型，返回 `{"ok": true, "decision": "model"}`。见 [examples/plugins/strategy-demo/src/lib.rs](../examples/plugins/strategy-demo/src/lib.rs) 里 `ping` 走规则、其他文本走模型的完整示例。

**生命周期插件（hook）**：`manifest.json` 里 `hooks` 数组声明要挂载的点，`ea_handle` 收到 `hook` 字段告知当前是哪个点，`args` 字段携带该点的专属数据。钩子是**观察者而非拦截器**：它的 `result`/`reply` 不会影响主流程，失败也只是记日志——适合做状态上报、审计、指标采集这类旁路逻辑；想改变流程走向请写策略插件。

生命周期覆盖主循环的全部阶段边界和关键异常事件，共 12 个挂载点：

**内核级**（`event` 字段为 null）：

| 挂载点 | 触发时机 | `args` 携带 |
| --- | --- | --- |
| `kernel_start` | 内核启动、插件扫描完成后 | `{"plugins": [已加载插件名]}` |
| `kernel_stop` | 优雅停机（业务调用 `Kernel::shutdown`） | `{}` |

**任务级**（按一个任务内的触发顺序排列）：

| 挂载点 | 触发时机 | `args` 携带 |
| --- | --- | --- |
| `pre_task` | 事件刚出队、正式处理前 | `{}` |
| `post_route` | 策略路由决策产生后 | `{"decision": "rule"\|"model"\|"none"}` |
| `pre_infer` | 每次即将调用推理后端前（重试会多次触发） | `{"attempt": n}` |
| `post_infer` | 拿到模型原始输出、格式校验之前 | `{"attempt": n, "raw": "原始输出"}` |
| `on_plan` | 合法命令解析成功、执行之前（审计点） | `{"plan": {"reply", "tool", "args"}}` |
| `pre_tool` | 资源锁已获取、即将调用工具插件 | `{"tool": 名, "args": {...}}` |
| `post_tool` | 工具插件调用结束（无论成败） | `{"tool": 名, "ok": bool, "error"?}` |
| `post_task` | 任务产出结果、资源锁释放之后 | `{"outcome": {"status", "reply", "via"}}` |

**异常事件**（不一定每个任务都发生）：

| 挂载点 | 触发时机 | `args` 携带 |
| --- | --- | --- |
| `on_degrade` | 降级链被触发（熔断、格式重试耗尽、工具失败等） | `{"reason": "原因"}` |
| `on_rollback` | 某插件因连续失败被自动禁用回滚（延迟到任务边界触发） | `{"plugin": 名}` |

几个值得注意的点：

- `post_infer` 是量产设备上唯一能观察到**模型原始行为**的位置——格式校验之前的输出，包括乱码和非法 JSON，都会原样送到这里，适合做端侧模型质量统计。
- `on_plan` 与 `pre_tool` 的区别：`on_plan` 看到的是模型"想做什么"（包括纯 reply），`pre_tool` 看到的是"即将真的对设备做什么"（锁已到手）。设备操作审计挂 `pre_tool`，模型行为审计挂 `on_plan`。
- `on_degrade` 是边缘设备最重要的健康信号：它每触发一次，说明确定性链条离开了正常路径。生产环境建议必挂。
- 没有单独的"设备唤醒"挂载点——唤醒本身就是一次事件入队，`pre_task` 已覆盖该时刻。

**内核保留调用（infer / repair）**：这两种 `kind` 不由模型或事件触发，而是内核在配置指定后主动调用（manifest 里仍声明为 `kind: "tool"`）：

- **推理插件**（配置 `backend: {"type": "plugin", "name": ...}`）：收到 `kind: "infer"`，`args` 为 `{"system", "input"}`，manifest 声明了 `context: true` 则同时拿到对话历史；把模型原始回答放进 `reply` 返回。
- **修复插件**（配置 `repair_plugin: "名字"`）：模型输出未通过内核 JSON 校验时被调用，收到 `kind: "repair"`，`args` 为 `{"raw": 原始输出, "error": 校验错误}`；把修复后的文本放进 `reply`。注意：修复结果仍要通过内核同一套严格校验，修复器只能救回输出，不能放宽格式。

被配置为 infer/repair 的插件会自动从模型可调用的工具列表中剔除，模型永远无法直接调用它们。

### 2.5 调用外部能力（host_call）

插件不能直接访问文件、网络、硬件，唯一途径是：

```rust
#[link(wasm_import_module = "edge")]
extern "C" {
    fn host_call(ptr: *const u8, len: usize) -> i64;
}

fn call_host(req: &serde_json::Value) -> serde_json::Value {
    let bytes = req.to_string().into_bytes();
    let packed = unsafe { host_call(bytes.as_ptr(), bytes.len()) };
    // packed 解包方式与 ea_handle 返回值相同：ptr = packed>>32, len = packed & 0xffffffff
    // ...解析出 {"ok": bool, "data"|"error": ...}
}
```

请求体是 `{"cap": "device:relay0", "op": "on", "args": {}}`。`cap` 必须一字不差出现在 manifest 的 `permissions.capabilities` 里，否则内核直接拒绝，插件收到 `{"ok": false, "error": "capability '...' not declared ..."}`，不会 panic、不会崩溃沙箱。

`cap` 的具体行为（`op`/`args` 怎么解释）由业务实现的 `HostBridge` 决定，框架不规定命名规范，但建议约定俗成用 `device:<资源名>` / `net:<host>:<port>` 这类前缀，方便和资源锁的设备命名对齐（资源锁只识别 `device:` 前缀的能力项，见 [kernel.rs](../src/kernel.rs) 的 `run_tool`）。

### 2.6 日志

```rust
#[link(wasm_import_module = "edge")]
extern "C" { fn host_log(ptr: *const u8, len: usize); }
```

传入的字符串会打印到内核 stderr，格式 `[plugin:<name>] <msg>`。这是插件唯一的可观测手段，调试格式错误、权限拒绝时优先看这里。

### 2.7 编写清单（manifest.json）

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "kind": "tool",
  "hooks": [],
  "permissions": {
    "context": false,
    "capabilities": ["device:relay0"]
  },
  "signature": null
}
```

- `name`/`version` 必须和插件包所在目录名完全一致（见下节），否则加载时直接拒绝。
- `version` 必须是合法 semver（`x.y.z`）。
- `context: true` 会让插件在每次调用时都收到完整对话历史副本——不需要就别开，减少插件能看到的数据面。
- `capabilities` 只列这个插件真正要用的能力，声明越少，攻击面越小；分发时签名会覆盖这个列表，事后改不了。

## 3. 编译与本地测试

```bash
cd my-plugin
cargo build --release --target wasm32-unknown-unknown
ls target/wasm32-unknown-unknown/release/*.wasm   # 通常几十到一两百 KB
```

## 4. 分发规范：打包、签名、部署

插件包在磁盘上必须是这个结构，目录名即身份，加载时会核对目录名与 manifest 内容一致：

```
plugins/
└── <name>/
    └── <semver>/
        ├── plugin.wasm
        └── manifest.json
```

### 4.1 生成签名密钥（厂商/开发者只做一次）

```bash
cargo run --bin ea-pack -- keygen ./vendor-key
# 产出 vendor-key.key（私钥，绝不能进设备）和 vendor-key.pub（公钥）
```

把 `vendor-key.pub` 里的十六进制字符串填进设备的 `edge-agent.json`：

```json
{ "trusted_pubkey": "<vendor-key.pub 的内容>", "dev_allow_unsigned": false }
```

### 4.2 组装并签名一个包

```bash
mkdir -p plugins/my-plugin/1.0.0
cp target/wasm32-unknown-unknown/release/my_plugin.wasm plugins/my-plugin/1.0.0/plugin.wasm
cp manifest.json plugins/my-plugin/1.0.0/manifest.json
cargo run --bin ea-pack -- sign plugins/my-plugin/1.0.0 ./vendor-key.key
```

签名覆盖 `name + version + kind + hooks + context 权限 + capabilities 列表 + wasm 的 sha256`（详见 [02-core-design.md](02-core-design.md) 第 6.4 节）——签名之后再改任何一项，验签都会失败，不存在"改小权限不用重签"这种漏洞。

### 4.3 部署（三种渠道，验证逻辑完全一致）

- **随固件预置**：直接把 `plugins/` 目录打进镜像；
- **局域网/云端下发**：把整个版本目录（`plugin.wasm` + 签好名的 `manifest.json`）传到设备的 `plugins/<name>/<新版本号>/`，无需重启，见下节热更新；
- **U 盘拷贝**：同上，路径对了就行，加载逻辑不区分来源。

三种渠道走的是同一份验签代码（[plugin/registry.rs](../src/plugin/registry.rs) 的 `load_version`），不存在"内网渠道可以少验一步"的例外。

### 4.4 热更新

不需要特殊指令：把新版本目录放到位后，向内核发一个内核保留事件：

```bash
echo '{"kind":"plugin_reload","payload":{"name":"my-plugin"}}' | cargo run --bin edge-agent
```

内核会重新扫描该插件名下所有版本，选中**签名有效且未被 `.disabled` 标记的最高 semver**。正在执行的任务持有的是旧版本的模块句柄，会跑完；下一个任务开始时才用新版本。旧版本文件不需要手动删除，只有它是"未被选中"或"被标记 disabled"的状态。

### 4.5 回滚是自动的

新版本连续失败达到 `plugin_max_failures`（默认 3）次后，内核会：

1. 在该版本目录下写一个 `.disabled` 标记文件（持久化，重启也生效）；
2. 立即重新选版，自动回退到上一个未被标记的稳定版本。

不需要人工介入。如果想手动强制回滚，直接在对应版本目录下创建空文件 `.disabled` 即可，效果相同。想恢复某个被标记的版本，删掉 `.disabled` 文件并触发一次 `plugin_reload`。

## 5. 完整冒烟脚本

下面这段命令把"打包 → 签名 → 加载 → 篡改检测"跑一遍，可以直接复制运行来验证一套新环境是否配置正确：

```bash
# 1. 编译内核和示例插件
cargo build
(cd examples/plugins/strategy-demo && cargo build --release --target wasm32-unknown-unknown)

# 2. 生成密钥、部署、签名
cargo run --bin ea-pack -- keygen /tmp/ea-vendor
mkdir -p plugins/strategy-demo/1.0.0
cp examples/plugins/strategy-demo/target/wasm32-unknown-unknown/release/strategy_demo.wasm \
   plugins/strategy-demo/1.0.0/plugin.wasm
cp examples/plugins/strategy-demo/manifest.json plugins/strategy-demo/1.0.0/manifest.json
cargo run --bin ea-pack -- sign plugins/strategy-demo/1.0.0 /tmp/ea-vendor.key

# 3. 配置公钥并关闭 dev 模式
python3 -c "
import json
cfg = json.load(open('edge-agent.example.json'))
cfg['trusted_pubkey'] = open('/tmp/ea-vendor.pub').read().strip()
cfg['dev_allow_unsigned'] = False
json.dump(cfg, open('/tmp/ea-test.json','w'), indent=2)
"

# 4. 跑三条指令：规则直达 / 越权能力被拒 / 走模型
printf '%s\n' \
  '{"kind":"command","payload":"ping"}' \
  '{"kind":"command","payload":"beep"}' \
  '{"kind":"command","payload":"hello there"}' \
  | cargo run --bin edge-agent -- --config /tmp/ea-test.json
```

期望输出：`ping` → `{"status":"ok","reply":"pong","via":"rule"}`；`beep` → 因为没接 `HostBridge`，收到能力被拒的规则回复；`hello there` → `via":"model"`，走到 Mock 后端。
