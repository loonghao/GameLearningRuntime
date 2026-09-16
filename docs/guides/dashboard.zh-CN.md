# 训练 Dashboard、实时观测与持久化历史

## 视频、图片与文档

运行详情包含 MP4/WebM 回放、图片画廊、Markdown 笔记和文本日志。决策事件可打开
检查器；存在已登记且匹配的 capture manifest 时，可按 episode/step 跳转视频。
重复步骤无法唯一定位时不会跳转。播放依赖浏览器编解码支持，服务不转码。

媒体必须位于 `.glr/runs/RUN_ID` 并登记为 artifact。生产端先写入文件，再调用
`TrainingStore.register_artifact`，传入相对 `path`、实际 `source`、`role` 和
`media_type`；该接口记录元数据和摘要，不复制文件。现有历史归档会包含这些运行文件。

**Preview local media** 可临时预览本机视频或图片，不上传、不持久化，也不关联到当前
运行。Markdown 只解析指向已加载、已登记产物的相对链接；不加载外部图片和原始 HTML。
HTML/SVG 仅供下载。

媒体目录每页最多 100 项，文档预览最多 256 KiB，capture manifest 最多 8 MiB、
映射最多 25,000 帧。联动前验证 manifest 摘要、运行与环境、帧单调性和视频登记信息。
浏览时不重新计算整段视频的摘要，明确标记 `not_reverified`；完整校验仍由归档或报告
验证执行。原文件使用流式 Range 读取，可拖动播放大视频。

接口：`GET /api/v1/media?run=ID&after=PATH` 获取目录；
`GET/HEAD /api/v1/media/file?run=ID&path=PATH` 读取文件；
`GET /api/v1/media/document?run=ID&path=PATH` 预览文档；
`GET /api/v1/media/frames?run=ID&manifest=PATH&video=PATH` 获取帧映射。

```powershell
glr --project . dashboard
```

打开输出的 localhost 地址即可。后端 Axum 和页面资源都在 `glr` 二进制中，
不需要部署 Node/Python Web 服务。默认端口为 7432，`--port 0` 自动选空闲端口。

前端采用 React、TypeScript、shadcn/ui（Radix）与 Tailwind。Vite 构建的 HTML、
带内容哈希的 JS/CSS 在 CI 中生成，再编译进各平台 CLI。所有资源由同一个 localhost
服务提供，保持现有 CSP，安装后的 GLR 不需要 Node.js 或外部 CDN。

从源码开发时，先执行 `vx just dashboard-build dashboard-check`，再运行 Cargo 或
`just check/build`。修改前端后需要重新构建；Cargo 会校验源码指纹，拒绝缺失或过期
的资源。生成的 `dist/` 不入库。CI 将 `dashboard-ui` 产物传给 Rust 和打包任务，
发布时 Linux、Windows、macOS 共用不可变 release tag 对应的前端产物。
`/api/v1/health` 的 `dashboard_source_sha256` 可追溯内嵌源码版本。

- **Training presets**：点击预设开始训练；保存、导出、导入预设。默认预设使用工程已有
  trainer 和录制配置。不同算法、预算等通过工程的 `task run ... --set` 预设表达。
- **GLR operations**：表单直接来自 CLI 参数定义，支持训练、报告、模型播放、查询、
  工程包导入导出、备份恢复等操作。路径指运行 GLR 的本机路径。
- **Operation history**：查看参数、状态、退出码、stdout/stderr。重复请求 ID 不会重复启动。
- **Run history**：查看已持久化运行；按 step 过滤事件、检查决策候选和执行回执、
  选择学习指标、查看不同 episode/route 的 XY/XZ/YZ 路线。
- **Process / FFmpeg output**：选择 `capture.log` 查看 recorder 转发的 FFmpeg 输出。
  recorder 如果丢弃或私有缓存 stderr，GLR 无法自行恢复这些信息。

独立启动 Dashboard 后，CLI 启动的训练也能在其中看到。`train` 和 `goal run` 默认
额外输出只读观测地址，随命令退出而停止；`--no-observe` 可关闭这个自动服务。
`glr observe` 仅提供只读页面，不提供启动操作的接口。

## 接入学习器

Bridge 主动上报支持 Python SDK、带令牌的 localhost HTTP 和 CLI JSON/JSONL。
详见 [Bridge 接入指南](bridge-telemetry.md)，包含状态/进度、自定义事件、指标、
来源标识、批次去重、Agent 查询及 C# 示例。

```python
from game_learning_runtime import Telemetry

telemetry = Telemetry.from_env()
if telemetry is not None:
    telemetry.learning_update(step_id=40, metrics={"loss": 0.25, "entropy": 0.7})
    telemetry.route_sample([10, 4, 2], step_id=40, episode_id="episode-1")
```

通过 GLR 启动的 Python 学习器使用 `execute_decision(..., step_id=40)` 时，会自动记录
选择与执行结果。其他算法主动调用统一 Telemetry 接口即可；不会从日志猜测模型更新。
这些指标是诊断数据，不会被升级为奖励、胜利或学习质量的权威证据。

角色日志默认写磁盘，并镜像到终端 stderr；`--json` stdout 仍保持机器协议。
`GLR_LOG_STDERR=0` 关闭终端镜像，`GLR_TELEMETRY_STDERR=0` 关闭 SDK 的事件打印，
二者都不会删除 SQLite 中的数据。

## CLI 追溯与备份

```powershell
glr --json runs trace RUN_ID --events-after 1000 --metrics-after 500
glr --json runs log RUN_ID --path capture.log --offset 0
glr --json report build RUN_ID
glr --json dashboard presets
glr --json dashboard jobs
glr --json dashboard jobs --before JOB_ID
glr --json dashboard job-log JOB_ID --stream stderr
glr --json backup create --output ../backups/training-2026-09-16
glr --json backup verify ../backups/training-2026-09-16
glr --json backup restore ../backups/training-2026-09-16 --output ../restored-history
glr observe --archive ../restored-history
glr --json runs trace RUN_ID --archive ../restored-history
```

备份使用 SQLite 在线快照，可包含训练期间已提交的数据。完整归档已完成运行的目录、
日志、录制、注册产物，以及已完成 Dashboard 操作的日志；预设和操作回执随数据库保存。
训练中的 run 只备份数据库记录，清单用 `active_runs_database_only` 明示，
不把仍在写入的视频或模型伪装成一致快照。完整备份应在运行结束后再生成。

备份目录包含校验清单；恢复验证每个文件的大小、SHA-256 和 SQLite 完整性，
只写新目录，不覆盖正在使用的数据。恢复历史不等于恢复游戏现场或验证模型兼容性。
工程代码、配置和脚本通过现有 `package export/import` 迁移，源文件必须显式选择，
导入必须匹配环境和契约。预设 JSON 可加入工程包清单。

页面只保留最近 5,000 条事件/指标与有限日志窗口；导出按钮明确导出当前加载窗口。
数据库和原始文件不会自动清理。事件起始游标为 `-1`，可包含旧版序号为 `0` 的事件。
大于 16 KiB 的事件内容在实时接口中显示截断标记，
完整内容仍保存在 SQLite。游标查询可越过 1,000 条历史记录；离线报告超过 100,000 条
事件或指标会明确拒绝，避免静默遗漏。

Dashboard 一次执行一个任务，并检查项目是否有运行中的任务。服务崩溃后留下的
非终态回执需要核实，不会自动重跑。当前不自动计划备份、不自动删除历史、也不提供
通用“杀死所有进程”按钮。

详细 API、限制及示例见 [英文完整指南](dashboard.md)。所有写接口仅接受同源 localhost
JSON 请求；Agent 可继续直接使用 GLR CLI。
