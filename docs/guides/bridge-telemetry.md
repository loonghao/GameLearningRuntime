# Bridge 主动上报与统一观测

Bridge、学习器、采集器主动发布诊断数据，Agent 和 Dashboard 读取同一份持久化记录。
数据通道独立于游戏动作协议：上报不会执行动作、修改 run 状态、授予奖励或确认胜利。
运行时仍负责真实的目标绑定、动作执行和终态回执。

## 接入方式

| 方式 | 适用场景 | 写入入口 |
| --- | --- | --- |
| Python SDK | GLR 启动的 Bridge/sidecar | `BridgeTelemetry.from_env(source)` |
| HTTP JSON | Unity/C#、Unreal/C++、Godot、独立进程 | `POST /api/v1/telemetry` |
| CLI JSON / JSONL | Shell、批量采集、显式磁盘队列重放 | `glr telemetry ingest --file FILE [--jsonl]`；`--file -` 接受 stdin |

三者复用同一份 Rust 校验和事务写入实现。SDK 在 Dashboard 任务中优先使用 HTTP；
直接 `glr train` 启动时使用自动注入的 `GLR_CLI_PATH` 和项目根目录调用本机 CLI。
后者不需要 Web 写接口，适合低频或批量发送。高频生产者应在工作线程聚合发送，
避免在引擎主线程同步等待网络或子进程。

既有 Python `Telemetry` 仍可直接写学习器事件和指标；需要来源标识、重试去重和
Bridge 最新状态面板时使用 `BridgeTelemetry`。

## 数据契约

工作台支持 [Agent 优先与多类型数据视图](agent-workbench.md)：在 `bridge.state` 的
`payload.workbench` 中声明 `glr.workbench.v1`，即可展示自有数值、表格和文本面板。
无需在前端写死游戏类型；同一数据可由 CLI 和 HTTP 读取，并随历史归档持久化。

通过 `glr --json telemetry schema` 或 `GET /api/v1/telemetry/schema` 获取
[JSON Schema](../schemas/bridge-telemetry.schema.json)。CLI 读取 Schema 不需要训练工程。

```json
{
  "schema_version": "glr.bridge-telemetry.v1",
  "run_id": "run-existing-running-id",
  "source": "bridge.unity",
  "batch_id": "batch-0001",
  "events": [
    {"kind": "bridge.status", "step_id": 42,
     "payload": {"state": "ready", "message": "Synthetic navigation provider"}},
    {"kind": "navigation.route_sample", "episode_id": "episode-1", "step_id": 42,
     "payload": {"position": [10, 4, 2], "world_id": "map-1", "route_id": "main"}}
  ],
  "metrics": [{"name": "bridge.latency_ms", "value": 2.5, "step_id": 42}]
}
```

- `source` 是稳定的生产者名，如 `bridge.unity.navigation`，不使用 PID 或机器路径。
- `batch_id` 标识一次逻辑提交；重试保留相同 ID 和内容。
- `run_id` 必须是当前工程环境中已存在且运行中的 run，不会静默创建 run。
- `episode_id`、`step_id` 关联路线、决策和回执；没有可靠关联时省略。
- 事件可选 `observed_at_ns` 保存采样时间。服务分配 `timestamp_ns` 和 `sequence_id`，
  顺序以接收顺序为准，不依赖生产者时钟。
- `payload` 是 JSON 对象；顶层 `authority` 和 `_glr` 是保留字段，服务固定写入
  `authority=diagnostic`，并在 `_glr` 中记录 source、batch 和采样时间。

| 事件/记录 | 推荐内容 | 展示 |
| --- | --- | --- |
| `bridge.status` | `state`, `message` | 按来源保留最新状态卡片 |
| `bridge.state` | 场景、任务、实体、库存等有界 JSON | 最新状态卡片，点击检查完整结构 |
| `bridge.progress` | `fraction`：0..1，`label` / `message` | 最新进度条 |
| `bridge.log` | `level`, `message` 及结构化字段 | 可搜索的事件时间线 |
| `navigation.route_sample` | `position`：2 或 3 个有限坐标，route/world ID | 交互路线图 |
| `agent.decision`, `agent.execution` | 候选、选择和执行回执 | 时间线与 step 关联 |
| 自定义事件，如 `bridge.inventory_changed` | 项目自有语义字段 | 时间线与详情面板 |
| `metrics[]` | 有限数值、稳定名称、可选 step | 可选择的指标曲线 |

最新状态按 `run + source + kind` 保存，独立于浏览器 5,000 条事件窗口；最多投影
100 项并明确显示截断。每张卡片显示接收时间。旧的 `ready` 不证明当前在线，
消费者需按采样周期判断新鲜度。界面只渲染数据，不执行生产者提供的 HTML/JS。

## Python SDK

```python
from game_learning_runtime import BridgeTelemetry

publisher = BridgeTelemetry.from_env("bridge.example")
if publisher is not None:
    batch = publisher.prepare(
        events=[
            {
                "kind": "bridge.progress",
                "step_id": 42,
                "payload": {"fraction": 0.25, "label": "Exploring map"},
            },
            {
                "kind": "bridge.state",
                "step_id": 42,
                "payload": {"scene": "map-1", "inventory": {"potions": 3}},
            },
        ],
        metrics=[{"name": "bridge.fps", "value": 60, "step_id": 42}],
    )
    receipt = publisher.send(batch)
```

`prepare` 返回普通 JSON，可先追加到项目自有 JSONL 文件并 flush/fsync，再发送。
SDK 不隐式重试或建立无界队列；超时表示结果未知，保留原 batch 后重试。
新 ID 会被视为新记录。调用方捕获诊断失败，按预算缓冲，不得因此重放游戏动作。

## HTTP 和跨语言

Dashboard 生成随机令牌，给它启动的进程注入 `GLR_TELEMETRY_URL` 和
`GLR_TELEMETRY_TOKEN`；角色启动边界还会注入对应 `GLR_RUN_ID`。
独立 Bridge 通过自己的配置通道传递这些值。也可在启动 Dashboard 前指定
`GLR_TELEMETRY_TOKEN`：32..128 位 ASCII 字母、数字、`_`、`.` 或 `-`，保持高熵。
令牌仅进入环境，不写入预设、任务回执或查询结果。它是项目级诊断写凭证，
不是游戏动作授权；不要写入 URL、报告或版本库。

### 由 GLR 托管的自主循环

当策略循环是调用方自己的长生命周期进程（而不是 Dashboard 任务或工程角色）时，
用 `glr host` 把它放进一个 GLR 拥有的 run：

```powershell
glr --json host -- python my_loop.py --episodes 500
glr --json host --timeout-seconds 600 -- ./loop
glr --json host --no-telemetry -- ./loop   # 只要 run，不要写入口
```

`--` 之后的命令行按原样执行：GLR 不展开 `{...}` 占位符，也不要求配置 trainer 角色。
因此 `glr host -- jq '{a: 1}'` 这类含裸花括号的参数会原样传给子进程；同理
`{telemetry_token}` 不会被替换成令牌，凭据只经环境传递，不会出现在命令行上。
子进程会收到与角色一致的 `GLR_RUN_ID`、`GLR_RUN_DIR`、`GLR_STORE_PATH`、
`GLR_CLI_PATH` 和环境身份，并在启用写入时额外收到 `GLR_TELEMETRY_URL` 和
`GLR_TELEMETRY_TOKEN`，因此 `BridgeTelemetry.from_env("loop.example")` 可直接工作。
令牌由 CLI 生成并只写入这一个子进程的环境，不进入 run 记录、日志、报告或预设；
回执只报告写入通道是否可用，不回传令牌。run 结束时子进程被回收，终态 run 依旧
拒绝新批次：托管提供写入通道，不提供事后补录。

托管不等于授权：子进程以调用方权限运行，GLR 不沙箱化它，也不把它的退出码当作
学习效果的证据。需要 planner、evaluator、晋升或 checkpoint 语义时仍应使用 goal loop。


```powershell
$headers = @{ Authorization = "Bearer $env:GLR_TELEMETRY_TOKEN" }
Invoke-RestMethod -Method Post -Uri $env:GLR_TELEMETRY_URL `
  -Headers $headers -ContentType 'application/json; charset=utf-8' `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($batchJson))
```

C# sidecar 使用同一 JSON，不需要依赖 Python 或学习算法库：

```csharp
// System.Net.Http, System.Net.Http.Headers, System.Text, System.
// Reuse the client; run outside the engine main thread.
var handler = new HttpClientHandler { AllowAutoRedirect = false, UseProxy = false };
var client = new HttpClient(handler) { Timeout = TimeSpan.FromSeconds(5) };
using (var request = new HttpRequestMessage(HttpMethod.Post, telemetryUrl)) {
    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
    request.Content = new StringContent(batchJson, Encoding.UTF8, "application/json");
    using (var response = await client.SendAsync(request)) {
        response.EnsureSuccessStatusCode();
        string receiptJson = await response.Content.ReadAsStringAsync();
        // Check schema_version, run_id, source and batch_id before acknowledging.
    }
}
```

C++、Godot、Rust 使用各自已有 HTTP 实现发送同一请求即可，无需改变 Runtime Host
动作协议。此 C# 示例是接入说明，不是已经通过 Unity/引擎验收的插件。

## Agent 追溯和恢复

```powershell
glr --json telemetry ingest --file batch.json
glr --json telemetry ingest --file pending.jsonl --jsonl
glr --json telemetry state RUN_ID
glr --json runs trace RUN_ID --events-after -1 --metrics-after 0
glr --json backup create --output ../backups/bridge-history
```

Agent 使用 `GET /api/v1/telemetry/state?run=RUN_ID` 读取最新状态，通过
`GET /api/v1/snapshot?run=RUN_ID&events_after=-1&metrics_after=0` 分页追溯事件/指标。
Dashboard、报告和备份共享 run-store，重启保留记录和去重回执；
`glr observe --archive BACKUP` 也能展示归档中的 Bridge 状态。

每批最多 64 KiB、合计 1..64 条记录，每条事件 payload 最多 12 KiB。
类型错误、未知字段、非有限数值、越界进度和无效路线拒绝整个批次。
事务同时保存事件、指标、最新状态与回执。媒体使用现有受校验的产物契约，
不要内联截图、视频、账号、凭证或机器绑定信息。

JSONL 最多 1,000 批，每行单独提交。后续行失败，之前的提交仍保留，可用相同文件
再次导入去重；空行无效。终态 run 拒绝新批次，但仍确认已提交的重复批次。
离线队列须在 run 结束前发送，不通过改写历史 run 伪造补录。

HTTP 仅监听 loopback，写入要求令牌和 JSON Content-Type，拒绝外站 Origin。
`glr observe`（包括自动训练观测页面）不开启 HTTP 写入口；需要 HTTP 上报时启动
`glr dashboard`。401 表示令牌错误，400 表示契约拒绝，413 表示请求过大，429 表示
并发已满。诊断上报成功不证明动作成功、模型更新有效或游戏目标完成。
