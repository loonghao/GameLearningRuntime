# Game Learning Runtime

[English](README.md) | 简体中文

[![PyPI](https://img.shields.io/pypi/v/game-learning-runtime.svg)](https://pypi.org/project/game-learning-runtime/)
[![CI](https://github.com/loonghao/GameLearningRuntime/actions/workflows/ci.yml/badge.svg)](https://github.com/loonghao/GameLearningRuntime/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB.svg)](pyproject.toml)
[![Rust](https://img.shields.io/badge/Rust-1.98.0-000000.svg)](rust-toolchain.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Game Learning Runtime（GLR）是连接游戏运行时与学习系统的学习器中立契约。
只需定义一次观察、动作、掩码、奖励、事件和 episode 边界，同一个适配器即可
复用于 TorchRL、自定义 PPO/IMPALA、行为克隆、离线数据集、评估与自动化 QA。

> 一个把游戏连接到学习系统和 AI Agent 的通用运行时。

GLR 只适用于你拥有或已获授权进行集成的游戏与测试环境。项目不包含反作弊绕过、
隐蔽注入或针对具体游戏的逆向代码。

## 查看 GLR 实际运行

![GLR collector 正在运行内置的合成计数器适配器](docs/assets/showcase/glr-counter-collector.gif)

该 GIF 来自真实的本地 `ContractEnvironment` + `SyncCollector` 运行，目标是 GLR
明确标记为合成环境的计数器适配器。它在不暴露游戏账号、本机路径、进程/窗口标识
或私有运行时数据的前提下展示公共契约。向仓库贡献真实适配器录像前，请先阅读
[展示素材来源与采集规则](docs/assets/showcase/README.md)。

## 一个边界，多种使用方

```text
游戏 / 模拟器
      │
      ▼
运行时适配器（C#、C++、Rust、Python、官方 API 等）
      │
      ▼
GLR 协议 + 环境契约
      │
      ├── TorchRL
      ├── 自定义 PPO / IMPALA
      ├── BC / DAgger / 离线学习
      ├── 录制 / 回放
      └── 评估 / 自动化 QA
```

游戏适配器不导入 PPO、IMPALA、BC 或 TorchRL。学习代码也无需知道运行时来自
Unity、Unreal、Source、原生程序还是测试模拟器。GLR 标准化的是数据与生命周期
边界，而不是强制某一种语言、引擎、传输或算法。

## GLR 当前标准化的内容

| 契约 | 当前能力 |
| --- | --- |
| 环境 | `reset`、真实语义的在线 `attach`、`step`、`close`、终止/截断、episode 与 step 标识 |
| 数据 | 递归张量规格、混合/参数化动作、掩码、事件、奖励和不可变 transition |
| 桥接 | 能力协商、reset/step 栅栏、默认拒绝元数据、传输中立的 driver 端口 |
| Runtime Host | Rust `glr-hostd`、有界 `glr.host.v1` stdio、Python `HostBridgeDriver` 与合成子进程 smoke |
| Provider SDK | 面向 Unity/BepInEx 的 .NET Standard 2.0 C# 契约，以及面向 Unreal/原生 Provider 的 header-only C++20 契约 |
| 运行时接入 | 向后兼容的 `glr.runtime-integration.v2`，区分有源码插件、获授权 Loader Plugin 与外部附着 |
| 训练配置 | 严格的 `glr.training.v1` 知识源、生命周期策略、桥接要求与可审计加权奖励 |
| 训练安全 | Episode shaping 预算、必需终局结果、失败回报上限与 BC 数据来源门禁 |
| 采集 | 面向 PPO/IMPALA 的定长或终止边界 unroll，以及面向 BC/离线训练的 `glr.transition.v1` JSONL |
| 集成 | 可选 Gymnasium、TorchRL 0.13 与模型中立的 PyTorch BC/PPO/GAE/V-trace objective |
| 验证 | 失败即关闭的契约包装器与隐私安全的合成 conformance profiles |
| Agent 工作流 | `glr-adapter-builder` Skill，用于带来源的玩法研究、有界宿主脚手架、部署暂存、训练和验证 |
| 模型复现 | `glr.model-bundle.v1` 保存配置、源码/锁文件、种子、版本、权重、指标与 SHA-256 溯源 |

具体游戏适配器、经过认证且 target-bound 的本地 Provider 传输、分布式 actor 传输、
生产训练器和参考策略仍属于[路线图](docs/planning/roadmap.md)工作。

## 快速开始

安装仅依赖 NumPy 的核心包：

```powershell
uv add game-learning-runtime
```

只在需要的位置添加可选集成：

```powershell
uv add "game-learning-runtime[torchrl]"
uv add "game-learning-runtime[torch]"
uv add "game-learning-runtime[gymnasium,torchrl]"
```

采集一个与学习器无关的 unroll：

```python
from game_learning_runtime import ContractEnvironment, SyncCollector
from game_learning_runtime.examples import CounterEnvironment, always_increment

environment = ContractEnvironment(CounterEnvironment(target=3))
collector = SyncCollector(environment, actor_id="local-actor")
unroll = collector.collect(always_increment, steps=16, policy_version=0)

print(len(unroll.transitions), unroll.total_reward)
```

对于已经运行、且明确获得授权的游戏，声明 `live-attach` 并显式选择该生命周期：

```python
environment = ContractEnvironment(authorized_live_adapter)
collector = SyncCollector(environment, start_mode="attach")
unroll = collector.collect(policy, steps=128, stop_on_done=True)
```

Attach 会从 step 0 开始一个新的 GLR 逻辑 episode，但绝不声称物理游戏世界已被重置
或设置了随机种子。

## Unity 与 Unreal 三条接入路径

GLR 只保留一套学习器侧契约，但会显式声明运行时边界：

- **有源码或官方扩展 SDK：** 使用引擎插件提供语义状态、原生动作、主线程调度、
  可控时钟和真实物理 reset。
- **已获授权的无源码运行时：** 通过官方 API、遥测、Replay，或受限的渲染观察与
  输入接口从外部 attach。默认要求实时运行、精确目标绑定、输入租约清理和后状态验证。
- **已获授权的 Mod Loader 运行时：** 通过 BepInEx 或 UE4SS 在进程内承载经过审查的
  有界命令 Adapter。它仍按实时 `attach` 工作，要求语义观察、游戏线程调度、准确的
  Loader/版本溯源；在游戏语义 handler 通过审查前，动作词表保持为空并默认拒绝。

```python
from game_learning_runtime import EngineFamily, RuntimeIntegrationProfile

profile = RuntimeIntegrationProfile.for_source(EngineFamily.UNITY)
environment = profile.connect(authorized_driver)
```

可以使用仓库 Skill 生成 Unity 或 Unreal Adapter 路径，再在保持契约测试通过的前提下
替换其中的合成语义。详见[引擎运行时接入指南](docs/guides/engine-runtime-integration.zh-CN.md)。

对于明确允许 Mod 的无源码游戏，可以直接生成 BepInEx 5 LTS Unity Mono 宿主或
UE4SS 3.x Lua 宿主：

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_loader `
  --package example_loader `
  --environment-id example.loader-v1 `
  --engine unity `
  --access loader `
  --loader bepinex `
  --loader-version v5.4.23.5
```

生成的部署命令只暂存带校验和的 payload，不会扫描或修改游戏安装目录。详见
[Loader Plugin 接入指南](docs/guides/loader-plugin-integration.zh-CN.md)。

## Runtime Host 与引擎 Provider

已实现的 Runtime Host 统一严格帧处理和生命周期 fencing，但不会试图替代引擎启动层：

```text
TorchRL / PPO / IMPALA / BC
  -> BridgeEnvironment -> HostBridgeDriver
  -> glr-hostd（Rust）
  -> C# Unity Provider / C++ Unreal Provider
  -> 官方插件、BepInEx、UE4SS 或官方 Mod SDK
```

执行真实跨进程一致性路径，并编译两套 Provider 契约：

```powershell
vx just host-smoke
vx just provider-sdk-check
```

当前 `glr-hostd` 只发布串行 stdio 的 `synthetic-counter`。它有 1 MiB 硬帧上限，
不会重试修改动作，但尚未声明认证或 target-bound IPC，也尚不能连接实机外部 C#/C++
Provider。准确能力边界和 Unity/Unreal 实现路径见
[Runtime Host 与 Provider SDK 指南](docs/guides/runtime-host-and-provider-sdks.zh-CN.md)。

## 可复现的本地开发环境

GLR 固定 Python、uv、just、rustup、Rust 与 .NET SDK 输入。本地与 GitHub Actions
执行相同 recipes：

```powershell
vx setup
vx just check
vx just ci
```

## 将知识库与奖励定义为数据

```python
from game_learning_runtime import (
    EpisodeRewardGuard,
    RewardSignal,
    load_reward_safety_config,
    load_training_config,
)

config = load_training_config("training.json")
guard = EpisodeRewardGuard(config, load_reward_safety_config("reward-safety.json"))
reward = guard.compose([RewardSignal(name="progress", source="runtime", value=0.25)])
print(reward.total, reward.contributions)
```

运行时遥测应标记为 `authoritative`；网页攻略和策略先验应标记为 `advisory`。
奖励项默认要求权威来源。配置只包含数据，GLR 不会把奖励表达式当作代码执行。
Episode guard 会限制每步和整局正向 shaping，要求权威的终局 outcome，并确保失败局
不会保留正回报。`DemonstrationGate` 则默认拒绝策略自模仿、失败局和未知来源样本进入
BC。详见[训练安全指南](docs/guides/training-safety.zh-CN.md)。

## 使用 Agent Skill 搭建适配器

仓库自带的 [glr-adapter-builder
Skill](.agents/skills/glr-adapter-builder/SKILL.md) 为新 Agent 提供边界明确的流程：

1. 研究当前玩法并保存来源；
2. 区分物理 `reset` 与真实语义的在线 `attach`；
3. 先搭建可训练的合成接缝；
4. 定义知识、奖励预算与 BC 数据来源策略；
5. 通过运行时桥接实现带栅栏的观察和动作；
6. 完成 conformance 验证后，再执行小范围、已授权的真实运行时 trace。

该 Skill 不会把网页策略升级为运行时权威，也不会把合成测试描述成真实游戏验收。

### 把 Skill 交给你的 Agent

最快的方式是克隆本仓库，并从仓库根目录启动 Codex。Codex 会自动发现
`.agents/skills` 下的仓库级 Skill。然后在提示词里显式调用：

```text
$glr-adapter-builder 为一个已获授权、拥有源码的 Unity 项目创建适配器。生成可训练环境、玩法研究清单、奖励配置和契约测试。
```

如果只有已获授权的二进制运行时，可以指定“无源码外部接入”。如果游戏明确允许
Mod，则指定 `BepInEx` 或 `UE4SS` 以及准确的兼容上游 tag。Skill 会选择真实语义的
`attach`、拒绝未知动作，并拒绝声明只有源码接入才能证明的能力。

如果希望在其他仓库使用，可以让 Codex 内置的安装器从 GitHub 安装：

```text
$skill-installer install https://github.com/loonghao/GameLearningRuntime/tree/main/.agents/skills/glr-adapter-builder
```

安装完成后，在新的 Agent turn 中调用 `$glr-adapter-builder`。团队需要可复现安装时，
请把 GitHub URL 中的 `main` 固定为 release tag 或 commit SHA。其他兼容开放 Agent
Skills 标准的 Agent，也可以把同一个 `glr-adapter-builder` 目录放进目标仓库的
`.agents/skills/`。参见 [Codex Skills 官方文档](https://developers.openai.com/codex/skills)。

生成结果包括环境骨架、`training.json`、`reward-safety.json`、
`demonstration-policy.json`、`runtime-integration.json`、带来源的研究清单、Agent 指令、
模型包冒烟训练器、测试、`vx.toml` 和 `justfile`。Loader 路径还会生成有界宿主源码
与部署清单。进入生成目录后运行：

```powershell
vx setup
vx run check
vx run train
vx run reproduce
```

`train` 会输出一个合成 BC 冒烟模型及自包含、带校验和的复现环境。替换真实 Learner
时仍应保留 `glr.model-bundle.v1` 校验门，详见
[可复现模型包](docs/guides/reproducible-model-bundles.zh-CN.md)。

## TorchRL 与自定义学习器

使用可选 TorchRL 适配器：

```python
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.integrations.torchrl import TorchRLEnvironment

env = TorchRLEnvironment(CounterEnvironment())
rollout = env.rollout(max_steps=32)
```

也可以在自定义 PyTorch 学习器中复用带掩码的 PPO objective：

```python
from game_learning_runtime.integrations.torch_objectives import ppo_loss

terms = ppo_loss(
    policy_logits=logits,
    actions=actions,
    old_log_prob=old_log_prob,
    advantages=advantages,
    values=values,
    value_targets=value_targets,
    action_mask=action_mask,
)
terms.loss.backward()
```

## 复用 CI 工作流

任何使用 uv 管理的 Python 仓库都可以调用 GLR 的公共可复用工作流：

```yaml
jobs:
  quality:
    uses: loonghao/GameLearningRuntime/.github/workflows/reusable-python-ci.yml@v0.3.0 # x-release-please-version
    with:
      python-versions: '["3.10", "3.12"]'
      sync-args: "--frozen --all-groups"
      lint-command: "uv run ruff check . && uv run mypy"
      test-command: "uv run pytest"
```

生产仓库应固定到 release tag 或 commit SHA。Release Please 会让示例 tag 与包版本同步。
可复用工作流不会接收部署密钥，只会检出并测试调用方仓库。

## 发布流程

`main` 上的 Conventional Commits 会创建或更新 Release Please PR。合并经过审阅的
Release PR 后，工作流会创建 tag 和 GitHub Release，校验并构建 tag 对应源码、附加
provenance，通过 Trusted Publishing 发布 Python 包，并附加 Linux、Windows、Intel
macOS、Apple Silicon 的 Runtime Host 校验和压缩包及 C# Provider 包。详见
[发布手册](docs/runbooks/release.md)。

## 文档

- [入门指南](docs/guides/getting-started.md)
- [构建可复用运行时桥接](docs/guides/runtime-bridges.md)
- [接入 Unity 与 Unreal 游戏运行时](docs/guides/engine-runtime-integration.zh-CN.md)
- [使用 Runtime Host 与 C#/C++ Provider SDK](docs/guides/runtime-host-and-provider-sdks.zh-CN.md)
- [接入获授权的 BepInEx 与 UE4SS Loader](docs/guides/loader-plugin-integration.zh-CN.md)
- [复现训练模型](docs/guides/reproducible-model-bundles.zh-CN.md)
- [配置知识源和奖励](docs/guides/knowledge-and-rewards.md)
- [限制奖励并验证 BC 数据来源](docs/guides/training-safety.zh-CN.md)
- [验证适配器](docs/guides/adapter-conformance.md)
- [适配现有 Gymnasium 环境](docs/guides/adapting-gymnasium.md)
- [组合自定义 Torch objectives](docs/guides/using-torch-objectives.md)
- [架构](docs/architecture/overview.md)与[数据流](docs/architecture/data-flow.md)
- [本地开发](docs/runbooks/local-development.md)
- [基准测试基线](docs/benchmarks/2026-08-31-data-plane-baseline.md)
- [路线图](docs/planning/roadmap.md)与[架构决策](docs/decisions/README.md)

开发契约见 [CONTRIBUTING.md](CONTRIBUTING.md)，私密漏洞报告流程见
[SECURITY.md](SECURITY.md)。GLR 使用 [MIT License](LICENSE)。
