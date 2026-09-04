# 使用 GLR Agent-first CLI

当一个已授权的游戏项目已经提供经过审查的 GLR 桥接器时，使用 `glr` CLI 让 Agent
启动项目角色、记录可查询证据、并发录制小窗视频、按预算追求目标，并在新实例中加载
空间知识或模型包。若仍需实现观察、动作、传输、生命周期、目标绑定或动作后验证，请使用
`glr-adapter-builder`，而不是把这些语义塞进 CLI。

## 安装主要入口

从同一个 GitHub Release 下载当前平台对应的 `glr-{version}-{rust-target}.zip` 与
`SHA256SUMS`。校验压缩包、解压，并把其中的 `glr` 和 `glr-hostd` 放入 `PATH`。
压缩包同时包含 `glr-cli` 与 `glr-adapter-builder` Skills。Rust CLI 可以独立运行；
只有项目角色需要 Python SDK 时才安装 Python 包。

操作项目之前先检查部署：

```powershell
glr --version
glr --project . --json doctor
glr --json update --check
```

`doctor` 校验项目契约、路径和已配置的可执行角色，但不证明实时桥接握手或实机验收。

## 更新 GLR 托管组件

`glr update --check` 是只读操作。用户明确要求更新后，执行 `glr update --yes`。Updater
只通过 HTTPS 下载准确 Rust target 的压缩包和 `SHA256SUMS`，校验 release manifest 与
摘要后，替换 CLI、同目录 Runtime Host 和项目 Skills。

```powershell
glr --json update --yes
glr --json update --yes --skills-dir .agents/skills
glr --json update --yes --no-skills
```

它不会运行安装脚本，也不会修改游戏代码、角色依赖、虚拟环境、模型、数据集或
`glr-project.json`。SHA-256 只验证同一 Release 的产物完整性，不等同于发布者签名。
更新后重新执行 `--version`、`doctor` 和 `update --check`。
检查默认匿名访问 GitHub。若遇到 API 限流，只通过当前进程的 `GLR_GITHUB_TOKEN`
传入已有 token；不要打印它，也不要把它写入项目配置。

## 配置项目

在项目根目录创建严格的 `glr-project.json`。路径必须相对项目，命令必须是固定 argv；
GLR 不调用 shell。

```json
{
  "schema_version": "glr.project.v1",
  "environment_id": "example.adventure-v1",
  "environment_family": "action-rpg",
  "protocol_version": "1.0",
  "data_dir": ".glr",
  "bridge_path": "bridge",
  "runtime": {"argv": ["python", "tools/runtime.py", "{bridge_path}"]},
  "trainer": {"argv": ["python", "tools/train.py"]},
  "player": {"argv": ["python", "tools/play.py", "{bundle}"]},
  "researcher": {"argv": ["python", "tools/research.py", "{research_path}"]},
  "planner": {"argv": ["python", "tools/plan.py", "{trial_path}"]},
  "evaluator": {"argv": ["python", "tools/evaluate.py", "{evaluation_path}"]},
  "progress": {
    "signal": "day_counter",
    "window_steps": 256,
    "max_stalled_rounds": 3
  },
  "capture": {
    "argv": ["python", "tools/record_window.py", "{capture_video}", "{capture_index}"],
    "required": true,
    "stop": "stdin-q",
    "video_file": "capture.mp4",
    "index_file": "capture-index.jsonl",
    "codec": "h264",
    "frame_rate": 12,
    "width": 640,
    "height": 360
  }
}
```

录制器属于项目，因为只有项目知道经过审查的游戏窗口和采集接口。对于训练时的小窗，
可以从 640×360、12 FPS 开始，以 UI 可辨认和训练可用为准，不追求展示级画质。

项目角色会收到 `GLR_PROJECT_ROOT`、`GLR_BRIDGE_PATH`、`GLR_RUN_ID`、`GLR_RUN_DIR`、
`GLR_STORE_PATH`、环境身份、录制输出路径和 goal loop 路径。每个角色仍需独立验证准确的
游戏目标，不能依赖 CLI 猜测进程或窗口。

进度检测默认关闭。声明 `progress` 后，完成的 trainer 必须在 `trainer.result.json` 中回传：

```json
{
  "schema_version": "glr.trainer-result.v1",
  "status": "completed",
  "metrics": {},
  "progress": {
    "signal": "day_counter",
    "first_value": 12,
    "last_value": 12,
    "steps_since_change": 256,
    "accepted_steps": 256
  }
}
```

声明的值在配置窗口内不变化时，CLI 将 trial 标为 `stalled`；达到连续轮次阈值后以退出码
`76` 中止 goal。结果包含 signal、首末值和未变化步数。没有 `progress` 时不会猜测信号，
也不会执行 stall 检测。

## 启动与训练

```powershell
glr --project . --json doctor
glr --project . --json runtime start
glr --project . --json train
```

`train` 会先启动录制 sidecar，再启动训练器，结束后停止录制器。完整录制包含：

- 供人 review 的小尺寸 H.264 MP4；
- 把 `(episode_id, step_id)` 对齐到视频 frame/PTS 的 NDJSON 索引；
- 记录 codec、FPS、分辨率、大小和 SHA-256 的 manifest。

这样后续可以从视频中筛选人认可的片段，再与动作和 transition 对齐。但普通 policy
rollout 不能自动标成专家示范；进入行为克隆或监督学习前，仍需通过 demonstration
provenance 门禁。只有明确不需要 review/监督数据时才使用 `--no-capture`。

录制器也可以选择加入 `glr.capture-session.v1` 生命周期握手：

```json
"session": {
  "status_file": "capture-status.jsonl",
  "startup_timeout_seconds": 5,
  "heartbeat_timeout_seconds": 5,
  "minimum_frames": 1,
  "minimum_steps": 1,
  "content_liveness": {
    "enabled": false,
    "required": false,
    "sample_every": 4,
    "max_bad_fraction": 0.5
  }
}
```

CLI 会写入带 `session_id` 的启动回执，并通过 `GLR_CAPTURE_STATUS` 告知录制器。录制器向
该路径追加严格的 NDJSON 状态，包含 `healthy`、`degraded`、`stopped`、`failed` 或
`completed` 状态、frame/step 计数、最近 frame 时间戳、丢帧数和可选原因。required capture
只有在握手健康、心跳未超时、出现 `completed` 终态、达到最小计数且
`glr.capture.v1` manifest 有效时才算成功。optional capture 不阻塞训练，但其生命周期会
以结构化的 `capture.lifecycle` 事件写入 run store，并出现在 `--json` 输出中。回执和状态
文件始终保留为 run artifact；视频、索引和 manifest 只有通过全部门禁后才登记。

启用后，content liveness 会稀疏采样连续帧对，并输出归一化的数值
`inter_frame_diff_mean`、`inter_frame_diff_max`、`luminance_mean` 和 `luminance_std`。
manifest 与 `--json` 会包含有界采样窗口、heartbeat、`content_static` /
`content_blank` 状态及原因。当坏内容比例超过配置阈值时，required capture 会在 artifact
登记前失败；optional capture 仍可完成，但会标记为 `degraded`。该功能默认关闭，且不会保留
帧内容。

## 用数值闭合视觉验证

`game_learning_runtime.visual_acceptance` 为无法直接查看像素的 Agent 提供与宿主无关的视觉
验收契约。`write_capture_atomically` 先写临时文件、flush/fsync 后再原子替换目标，并返回
回显的 `request_id`、尺寸、字节数和 SHA-256；`CaptureJobRegistry` 提供有界的
`pending`/`completed`/`failed` 轮询和严格请求关联。

`compute_visual_metrics` 与 `evaluate_visual_acceptance` 返回可直接写入 JSON 的 coverage、
包围盒/比例、distinct colours、chroma、luminance 以及可选 silhouette IoU。required 检查
使用 `require_visual_acceptance`；optional 检查可保留报告和失败原因供后续复核。该契约不
暗示任何宿主截图旁路，也不会保留图像内容。

## 让 Agent 按目标闭环

目标使用 `glr.agent-goal.v1`，必须包含稳定的目标 ID、环境品类、机器可判断的成功指标，
以及最大 trial、训练 step、总时长和资料源数量。例如：

```json
{
  "schema_version": "glr.agent-goal.v1",
  "goal_id": "goal.reach-destination",
  "objective": "到达指定地点，并由运行时确认已经抵达。",
  "environment_family": "action-rpg",
  "success_criteria": [
    {
      "metric": "objective.arrived",
      "operator": "gte",
      "target": 1,
      "source": "runtime.telemetry"
    }
  ],
  "budget": {
    "max_trials": 8,
    "max_training_steps": 50000,
    "max_wall_seconds": 14400,
    "max_research_sources": 64
  },
  "allowed_research_media": ["official-rules", "text-guide", "video-tutorial"]
}
```

执行：

```powershell
glr --project . --json goal run --goal goals/reach-destination.json
```

控制环顺序为：

```text
目标 + 硬预算
  -> researcher：带来源的攻略/规则/教程结论
  -> planner：本轮训练方案与声明式奖励项
  -> trainer：训练、指标、视频与产物
  -> evaluator：绑定已保存权威指标的证据
  -> 达标则停止；未达标则补充资料并调整下一轮
```

失败后的 research/planner 会得到上一轮资料和评估路径，因此可以在遇到难点时再看文字
攻略、修正视频教程假设或调整奖励。但所有轮次合计仍不能突破最初的资料源、trial、step
和时长预算。

官方规则、文字攻略和视频教程只能产生 advisory 结论。Evaluator 的 value、source、
authority 和 run ID 必须与当前 trial 新写入 SQLite 的 metric 完全匹配；只有
`authoritative` 运行时证据能满足成功条件。

## 查询训练历史与世界知识

```powershell
glr --project . --json runs list --limit 20
glr --project . --json runs show RUN_ID
glr --project . --json query entities --world forest --kind shrine --name 土地庙
glr --project . --json query routes --world forest --to-entity shrine.forest-1
glr --project . --json query edges --world forest --from-node node.spawn --at-ns 0
glr --project . --json query research --tag navigation --category strategy
glr --project . --json query research --verified-only
```

输出采用稳定的 `glr.cli-output.v1`。`runs show` 返回事件、指标、产物角色、哈希与元数据。
SQLite 只保存方便 Agent 查询的投影；transition、tensor、视频和模型仍是普通、可校验的
训练产物。

Research 查询会合并当前游戏、同品类和通用结论，并排除 rejected 结论。路线和攻略始终
是建议；Agent 在当前实例中仍需重新观察，并验证每个动作的后置状态。

## 在新实例迁移与复现

导出已经探索到的实体和路线：

```powershell
glr --project . --json knowledge export --output artifacts/spatial-knowledge.json
```

在同一环境/协议的新 checkout 或新游戏实例中导入：

```powershell
glr --project . --json knowledge import --input artifacts/spatial-knowledge.json
```

可选的 `glr.spatial-knowledge.v2` directed graph 也通过同一条导入命令处理。使用
`query edges` 查询有向边，并传入 `--status traversable` 只得到当前 frontier 候选；blocked 和
stale 边只有在显式查询时才会返回。负向遍历证据会保留其 advisory provenance，不会获得 action
authority。

环境或协议不一致时 GLR 会拒绝；导入后的坐标和路线会降级为 advisory，直到新实例再次
观察确认。对于同品类但不同游戏，只复用 family-scoped 攻略/策略结论，不能迁移世界坐标、
动作语义或默认认为模型兼容。

加载经过校验的模型包：

```powershell
glr --project . --json play --bundle artifacts/model-bundle
```

Checkpoint 合同预检与显式迁移：

```shell
glr --json checkpoint migrate \
  --manifest checkpoints/policy.manifest.json \
  --contract contracts/live-checkpoint-contract.json
glr --json checkpoint migrate \
  --manifest checkpoints/policy.manifest.json \
  --contract contracts/live-checkpoint-contract.json \
  --force
```

第一个命令对可迁移变化执行 dry-run（退出码 `3`）；action、observation、protocol
或 schema 变化会 fail-closed（退出码 `4`）。`--force` 会创建备份、保持 checkpoint
字节不变，并校验重写后的 manifest。

`play` 会先逐文件校验哈希，并要求 environment/protocol 完全一致，再启动项目 player。
“成功加载”不等于“成功复现”。必须在新实例再次运行权威 evaluator，并比较目标指标。

## 失败边界

以下情况 CLI 会停止：必需角色或录制失败、严格 JSON 无效、预算超限、资料类型未授权、
奖励引用未知结论、空间/模型身份不匹配，或 evaluator 证据找不到对应的已保存 metric。

使用返回的 run ID 执行 `runs show`。不要为了变绿而放松身份、权威性、来源或预算检查。
