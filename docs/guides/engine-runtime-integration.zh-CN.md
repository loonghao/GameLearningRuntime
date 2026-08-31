# 接入 Unity 与 Unreal 游戏运行时

[English](engine-runtime-integration.md)

GLR 只维护一套环境契约，同时提供两种真实可信的部署 Profile。接入路径由你被授权
使用的运行时边界决定，而不是由 PPO、IMPALA 或 BC 等训练算法决定。

## 选择接入路径

| 边界 | 引擎插件 | 外部附着 |
| --- | --- | --- |
| 常见条件 | 拥有游戏源码，或引擎提供官方扩展 SDK | 只有二进制游戏，但存在获授权的外部接口 |
| 生命周期 | 物理 `reset` 或检查点恢复 | 默认只声明真实语义的 `attach` |
| 时钟 | 手动步进或受控时间缩放 | 实时运行 |
| 观察 | 引擎语义状态，可选渲染传感器 | 优先官方遥测/API，其次渲染输出 |
| 动作 | 原生游戏命令 | 官方动作 API 或受限输入词表 |
| 目标安全 | 绑定引擎实例和会话 | 每次修改前重新核对进程、窗口或会话 |
| 吞吐 | 无头构建、时间缩放、多个隔离实例 | 受实时执行和采集延迟限制 |

两条路径都返回相同的不可变 `EnvironmentSpec` 与 `TimeStep`，共用 episode/step
栅栏、Collector、数据集和训练器。

## 生成接入骨架

Unity 有源码环境：

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_unity `
  --package example_unity `
  --environment-id example.unity-v1 `
  --engine unity `
  --access source
```

已获授权的 Unreal 无源码环境：

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_unreal_external `
  --package example_unreal_external `
  --environment-id example.unreal.external-v1 `
  --engine unreal `
  --access external
```

脚手架会生成 `runtime-integration.json`、`training.json`、合成 conformance 环境、
研究清单、测试、`vx.toml` 和 `justfile`。首先只替换合成游戏语义，并始终保留契约
测试的 Red-Green-Refactor 闭环。

## 有源码：引擎内插件

引擎侧负责观察编码、动作执行、奖励、Mask、终止条件、物理 reset 和权威后状态：

1. 在引擎主线程或游戏线程采集语义状态和执行动作；
2. 暴露稳定的动作词表，不提供任意反射或脚本执行；
3. reset 或恢复检查点后才能返回 GLR step 0；
4. 每个被接受的动作只推进固定的模拟时间；
5. 返回动作后的状态，再确认动作成功；
6. 引擎允许时支持隔离的无头或打包实例并行训练。

Unity 可以把 `CollectObservations`、`ActionBuffers`、离散动作 Mask、
`OnActionReceived` 和 episode 回调映射到 GLR。ML-Agents 是可选兼容提供方，GLR
仍是学习器中立边界。决策频率可以跟随物理步，也可以由回合或游戏事件触发。参考
Unity 官方[环境设计文档](https://unity-technologies.github.io/ml-agents/Learning-Environment-Design/)。

Unreal 建议用插件组件或 Subsystem 实现 Adapter，把状态读取和修改切回 game thread。
需要时可把 Learning Agents 的 Interactor、Manager、Recorder、Training Environment
和 Communicator 映射到 GLR。Epic 当前提供共享内存和 Socket communicator，但 GLR
仍要求先通过配对基准再采用具体高速传输。参考官方
[Learning Agents API](https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/LearningAgents)
与[训练 API](https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/LearningAgentsTraining)。

## 无源码：外部附着

只能使用操作者明确获授权的运行时和交互接口。观察和动作优先级为：

1. 官方游戏、Mod、无障碍、遥测、Replay 或测试 API；
2. 有文档且新鲜度有界的日志或导出状态；
3. 渲染帧与明确受限的输入词表。

外部 Adapter 不得提供不可见隐藏状态、任意内存、反射、脚本、通用点击、无限制
进程发现或反作弊绕过。它通常只声明 `live-attach`；GLR 会拒绝虚假的 seeded reset
和手动时钟。每个修改动作都必须携带 episode 与预期 step，绑定到选定运行时，
必要时持有可释放的输入租约，并返回已经验证的后状态。

如果官方 API 同时提供语义观察和受限动作，可显式声明：

```python
from game_learning_runtime import (
    ActionMode,
    EngineFamily,
    ObservationMode,
    RuntimeIntegrationProfile,
    TransportMode,
)

profile = RuntimeIntegrationProfile.for_external(
    EngineFamily.UNITY,
    observation_mode=ObservationMode.OFFICIAL_API,
    action_mode=ActionMode.OFFICIAL_API,
    transport_mode=TransportMode.OFFICIAL_API,
)
environment = profile.connect(authorized_driver)
```

## 提效检查表

- 语义张量足够时，不要强制采集图像。
- 分离决策频率、渲染频率与物理频率。
- 通过批量 Agent 或多个隔离游戏实例扩展，而不是把学习器逻辑放进 Adapter。
- 为帧、队列、超时和主线程工作设置上限。
- 动作结果不确定时禁止自动重试，必须通过权威读回核对。
- 只有可复现基准达到 ADR-0004 门槛后，才把 framing、共享内存环或张量转换迁移到 Rust。

本地与 CI 使用同一个质量入口：

```powershell
vx setup
vx just check
```

合成 conformance 不能证明真实 Unity 或 Unreal 游戏已经接入。真实验收必须在小范围、
已授权运行时完成，并且公开材料只能包含聚合结果。
