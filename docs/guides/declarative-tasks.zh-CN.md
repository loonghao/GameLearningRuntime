# 使用声明式 VX 任务扩展 GLR

项目需要组合 Python、数据准备、训练赛季或打包流程，但这些流程尚未形成 GLR 核心契约
时，使用项目根目录的 `glr.toml`。运行时角色和环境身份继续放在
`glr-project.toml`（或兼容的 `glr-project.json`）中。

## 配置 VX 任务

先在 `vx.toml` 中锁定 Python、uv 等运行时版本，再声明任务：

```toml
schema_version = "glr.tasks.v1"

[tasks.prepare]
description = "准备训练输入"
runner = "vx"
argv = ["uv", "run", "--no-sync", "python", "tools/prepare.py"]
timeout_seconds = 600

[tasks.season]
description = "执行一个有界训练赛季"
runner = "vx"
argv = [
  "uv", "run", "--no-sync", "python", "tools/run_season.py",
  "--profile", "{profile}",
  "--max-matches", "{max_matches}",
  "--result", "{task_result}",
]
depends = ["prepare"]
timeout_seconds = 7200

[tasks.season.parameters.profile]
type = "string"
required = true

[tasks.season.parameters.max_matches]
type = "integer"
default = 20
minimum = 1
maximum = 100

[tasks.season.result]
schema = "glr.season-result.v1"
required = true
```

`runner = "vx"` 会在固定 argv 前添加 `vx`。VX 负责解析 uv、Python 与项目虚拟环境，
GLR 不重复实现 Python 安装和环境管理。环境已经由项目 setup 或 CI 锁定时可使用
`--no-sync`；任务明确允许 uv 同步锁定环境时可以省略它。

## 检查和执行

```powershell
glr --project . --json doctor
glr --project . --json task list
glr --project . --json task show season
glr --project . --json task run season `
  --set profile=league-legends/native-100024 `
  --set max_matches=20
```

参数使用可重复的 `--set NAME=VALUE`。支持 `string`、`integer`、`boolean` 和项目内
相对 `path`。依赖按照无环图顺序各执行一次；未知字段、循环依赖、重复或未知参数、
局部字符串占位符、越界 cwd、符号链接任务文件和非法超时都会在启动进程前被拒绝。

任务进程收到 `GLR_PROJECT_ROOT`、`GLR_TASK_NAME`、`GLR_TASK_DIR` 与
`GLR_TASK_RESULT`。也可以把 `{project_root}`、`{task_dir}`、`{task_result}` 作为完整
argv 参数使用。参数占位符同样必须独占一个 argv 项，禁止 `--profile={profile}` 这种
字符串插值。

每次执行会在 `.glr/tasks/<execution-id>/result.json` 写入
`glr.task-result.v1`，每个任务另有 `task.log`。超时返回 124；第一个失败的依赖会终止
后续任务。任务声明 `result` 时，GLR 还会要求 `{task_result}` 是普通 JSON 文件且
`schema_version` 与声明完全一致；无效结果返回 78。

任务成功只表示配置的进程正常退出，不证明桥接握手、权威比赛结果、赛季完成、策略提升
或模型验收。相应结论仍需 GLR run store 与 evaluator 的权威证据。

`glr.toml` 是受信任的本地项目配置。固定 argv 能消除 Shell 解析，但不会为所选进程提供
操作系统级沙箱。
