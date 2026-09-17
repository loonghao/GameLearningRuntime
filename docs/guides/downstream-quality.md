# 下游 GLR 训练项目质量基准

这套规范适用于使用 GLR 的项目，包括适配器、训练器、评估器、录制器和训练工具，
不仅适用于 GLR 框架本身。

完整规范位于 [Downstream Python quality baseline](../../.agents/skills/glr-adapter-builder/references/downstream-quality.md)。
它随 `glr-adapter-builder` Skill 分发，并由脚手架生成到下游项目的 `QUALITY.md`。

必须统一的边界：

- **包与导入**：`pyproject.toml`、`src/<namespace>`、可安装 wheel、显式依赖；
  禁止依赖 `sys.path` 修改、`PYTHONPATH` 或当前工作目录修补导入。
- **日志接口**：模块使用 `logging.getLogger(__name__)`，训练应用统一配置；
  保留异常堆栈、结构化事件和 run/episode/step 上下文。
- **并发与轮转**：多个生产线程通过有界队列交给单一监听器写入和轮转；
  多进程使用集中收集器或独立文件。必须定义拥塞、丢弃计数、磁盘故障和退出排空行为。
- **错误聚合**：Sentry 等 SDK 为可选依赖，由应用初始化；测试异常去重、脱敏、
  离线发送失败与限时退出。监控不可用不能改变训练结果。
- **测试与回归**：静态检查、单元测试、GLR 契约、安装后测试、确定性训练验证、
  checkpoint 恢复、日志故障测试分别验收；每个修复保留可离线运行的行为回归用例。

新脚手架提供 `vx run package-check`，在临时目录构建 wheel，安装到全新虚拟环境，
从源码目录外执行包导入与测试。`vx run check` 执行静态和源码项目测试。
`vx run ci` 将这两组检查组合为 CI 入口。
脚手架只提供初始基准，不代表已经实现生产日志系统、Sentry 接入或真实学习器。

已有项目按“记录当前行为 → 补离线回归 → 迁移一个完整包及调用者 → 统一日志 →
启用质量门禁”的顺序迁移。不要把测试文件数量、源码目录内通过测试，或能够启动训练，
当作整个项目已合规的证据。真实游戏验收单独记录。

## Agent 升级与代码、数据迁移

模块职责见 [Module boundaries](../../.agents/skills/glr-adapter-builder/references/module-boundaries.md)，
新项目会生成 `ARCHITECTURE.md`。规范涵盖契约、环境、奖励、策略、学习器、采集、
存储与应用组合；一个 wheel 可以包含多个模块，不要求每个概念单独拆包。
模块迁移必须检查旧 import、配置入口、序列化引用及观测/动作顺序，并验证离线行为一致。

升级必须遵循 [Framework migration contract](../../.agents/skills/glr-adapter-builder/references/framework-migration.md)。
脚手架同时生成 `FRAMEWORK_MIGRATION.md` 流程和 `MIGRATIONS.md` 升级记录模板。

Agent 必须确认实际安装版本与目标版本，核对 API、配置、数据库、数据集、模型及
优化器状态的兼容性；先做一致性备份并验证恢复，再向新目录迁移。验收涵盖记录数量、
引用关系、哈希、数据来源、离线回放及 checkpoint 恢复。不支持的迁移必须明确报告，
不能悄悄清空历史、丢弃权重或用更旧的本地仓库降级下游项目。

切换前保留旧包、锁文件、配置及数据快照，并验证回滚命令。回滚对切换后新增数据的
影响必须记录；只备份了文件不代表已经证明可以无损回滚。
