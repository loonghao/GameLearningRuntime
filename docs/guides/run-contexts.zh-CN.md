# 绑定单次调用上下文

当一次 GLR 调用必须使用一组确定的项目配置时，使用
`glr.run-context.v1`。准备、环境管理和流程编排仍放在 VX 驱动的
`glr.toml` task 中；context 只负责不可变身份与角色间传递。

```toml
schema_version = "glr.run-context.v1"
context_id = "ranked-2026"
environment_id = "example.context-v1"
protocol_version = "1.0"

[labels]
season = "ranked-2026"
ruleset = "standard"

[[inputs]]
owner = "training"
path = "config/training.json"
schema_version = "glr.training.v1"
```

```powershell
glr --project . --context config/contexts/ranked.toml --json doctor
glr --project . --context config/contexts/ranked.toml --json train
```

CLI 会校验环境与协议，冻结 context 源文件及所有声明的 JSON/TOML
输入，并在每个角色启动前重新校验字节。运行会保存
`run-context.json`、登记 `run-context` artifact，并写入
`context.selected` 事件。Python 角色通过
`load_inherited_run_context(project)` 校验同一份环境回执。

season、ruleset、experiment 等领域概念只是 labels 或项目 task 策略，
不会成为 GLR 核心子命令。哈希只证明输入身份，不证明运行时就绪、训练成功或
真实游戏验收。
