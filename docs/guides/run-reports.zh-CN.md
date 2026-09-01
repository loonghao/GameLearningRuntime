# 生成可交互的训练回顾报告

GLR 可以把已经持久化的本地运行转换成自包含的 HTML 回顾物料：

```powershell
glr --project . --json report build run-0123456789abcdef
```

默认输出为 `.glr/runs/<run-id>/report/index.html`。`--output` 可以指定该
run 目录内的其他目录。命令会读取 SQLite 运行投影，按文件大小和 SHA-256
校验所有运行证据物料，并把生成的 HTML 登记为 `run-report` artifact。之前生成的
`run-report` 派生物料不会再次嵌入，避免重建时产生自引用摘要。它不会启动运行时、
发送动作或修改训练数据集。

报告完全离线、无需服务器，包含响应式总览卡片、指标条、可过滤事件时间线、
路线样本、进度/解锁事件、对战结果，以及带校验和的录像和战后截图链接。浏览器
端使用 `textContent` 展示运行时数据，不会把事件 payload 当成 HTML 或脚本执行。

## Adapter 事件约定

报告保持游戏中立。Adapter 可以向 run store 追加以下命名空间事件：

| 事件 | 约定 payload | 报告区域 |
| --- | --- | --- |
| `navigation.route_sample` | 有限数值 `position: [x, y, z]` | 2D 路线图 |
| `progression.item_unlocked` | `item_kind`、`item_id`，可选 `status` | 解锁表 |
| `progression.catalog_snapshot` | `catalog_kind` 和有界快照摘要 | 进度历史 |
| `match.result` | `match_kind`、`outcome`，可选 `turns`、`trophy_delta` | 对战表 |

这些事件是观察证据，不是动作。只有 Adapter 从获授权运行时状态读取坐标时，路线
才可以标记为 authoritative。只有渲染截图而没有语义状态时，只能把它作为媒体证据，
不能据此推断路线、解锁或胜负。

PvP 游戏必须显式设置 `match_kind=pvp`；普通战斗或打怪胜利不会被报告自动算成
PvP 胜场。战后卡牌截图应由项目自有的授权录制器采集，以项目相对路径和摘要登记，
再由对战事件引用。

## 证据与隐私

- 原始报告默认只在本地使用，发布前审查 metadata 和 payload。
- 不要在事件中写入账号、进程/窗口标识、主机名、绝对路径、凭据或私有状态。
- 媒体文件应放在 run 目录旁，并由 artifact 清单绑定 SHA-256；文件缺失或变更时，
  报告构建会失败关闭。
- 在事件中保留 `authority` 或 `status`，区分 authoritative、inferred、advisory 和
  需要人工确认的证据。
- 报告是回顾证据，不等同于实机验收；HTML 构建成功只证明已保存的运行数据和物料
  可读取且通过校验。

报告格式为 `glr.run-report.v1`，它以增量方式消费现有的 `glr.transition.v1`、run
store 和 capture 契约，不改变张量编码或学习器接口。
