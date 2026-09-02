# Runtime Host 与 Provider SDK

GLR Runtime Host 用来消除各游戏适配器重复的传输和生命周期代码，但不会假设一种框架能够启动所有引擎。

```text
Python learner / collector
  -> BridgeEnvironment
  -> HostBridgeDriver
  -> glr-hostd（Rust：帧、生命周期、fencing）
  -> engine provider（C# 或 C++：游戏语义、主线程、权威后状态）
  -> 官方插件 / BepInEx / UE4SS / 官方 Mod SDK
  -> 已授权运行时
```

## 运行一致性 Host

构建并执行真实的子进程边界：

```powershell
vx setup
vx just rust-check
vx just host-smoke
```

也可以由 Python 显式启动已下载或本地构建的二进制：

```python
from pathlib import Path

from game_learning_runtime import (
    BridgeEnvironment,
    ContractEnvironment,
    HostBridgeDriver,
    HostProcessConfig,
)

config = HostProcessConfig(executable=Path("C:/tools/glr-hostd.exe"))
driver = HostBridgeDriver.from_process(config)
environment = ContractEnvironment(
    BridgeEnvironment(
        driver,
        required_capabilities={"host-stdio", "reset", "step"},
    )
)
```

`HostProcessConfig` 会拒绝相对路径和不存在的程序，不通过 shell 启动，也不会搜索游戏。当前 Host 只提供 `synthetic-counter` 与串行 stdio；它是真实端到端契约测试，但不是 Unity/Unreal 实机验收。响应超时或帧损坏会让该子进程会话立即 fail-closed；调用方必须启动新 Host，不能让迟到响应与下一动作错配。

## 重连与未完成动作对账

能够证明 episode 状态已持久化的 Provider 可以声明
`reconnect-resume-v1`，并实现可选的 resumable Provider 契约。调用方发送
episode ID 和自己最后提交的 step；Provider 返回权威 `ProviderTimeStep`，并可为
传输中断时尚未确认的动作返回 `ActionReconciliation`。`applied`、`not_applied`、
`unknown` 是权威结果；只有 Provider 能证明重试安全时才能将 `retryable` 设为 true。
重连结果不能把游标推进到权威返回 step 之外，episode 或游标不匹配必须拒绝。

## 实现 Unity Provider

构建或下载 `GameLearningRuntime.Provider`，在已授权的 Unity 插件中引用这个 .NET Standard 2.0 程序集并实现 `IRuntimeProvider`。Unity 官方插件或 BepInEx 插件只保留薄启动职责：

1. 绑定经过审核的运行时实例；
2. 在 Unity 主线程执行 `Reset`、`Attach`、`Step`；
3. 把游戏语义状态与动作转换为复制后的 `TensorBuffer`；
4. 返回权威 `ProviderTimeStep` 后状态；
5. 在 `Dispose` 中释放 hook 与自有状态。

如果只能真实地 `Attach`，Provider 必须拒绝物理 `Reset`。不要暴露任意反射、通用方法调用或 C# 求值。

## 实现 Unreal Provider

在 Unreal Runtime Module 中包含 `sdk/cpp/include/glr/provider.hpp` 并实现 `glr::runtime_provider`。Unreal 层负责 Game Thread 调度与 UObject 生命周期。契约头文件不依赖 Unreal，因此普通 CI 可以先独立编译；带许可的引擎实机验收仍是单独门禁。

无源码且明确允许 Mod 时，可以继续用 UE4SS 做启动层，但仍需单独审核 Mod 政策、精确上游版本、游戏版本与 Game Thread 行为。GLR 不内置或静默安装 UE4SS。

## 训练与复现

未来接通实机 Provider 传输后，训练侧仍消费普通 `GameEnvironment`。已有 collector、奖励保护、BC 来源门禁、TorchRL 适配器和模型包流程均无需改变。模型包应记录 Runtime Host/Provider SDK 版本、runtime-integration 配置、奖励策略、随机种子、锁文件、模型与聚合指标；不得保存认证材料、本地程序路径或游戏安装路径。

## 当前能力边界

| 能力 | 当前状态 |
| --- | --- |
| Rust 生命周期与 fencing 核心 | 已实现并测试 |
| 有界 stdio client/host | 已实现；串行，硬上限 1 MiB |
| 合成子进程 smoke | 已实现；只输出聚合证据 |
| C# Unity/Provider 契约 | 已实现；.NET Standard 2.0 |
| C++ Unreal/Provider 契约 | 已实现；header-only C++20 |
| 重连/resume 动作对账 | 已实现；opt-in `reconnect-resume-v1` |
| 认证且 target-bound 的本地 IPC | 尚未实现 |
| 实机 C#/C++ Provider 连接 | 尚未实现 |
| 共享内存/异步 actor queue | 可选标准库 `BoundedActorQueue`；共享内存仍需基准达标 |
| 通用注入/启动器 | 明确不提供 |
