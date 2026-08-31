# Game Learning Runtime

[English](README.md) | 简体中文

[![PyPI](https://img.shields.io/pypi/v/game-learning-runtime.svg)](https://pypi.org/project/game-learning-runtime/)
[![CI](https://github.com/loonghao/GameLearningRuntime/actions/workflows/ci.yml/badge.svg)](https://github.com/loonghao/GameLearningRuntime/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB.svg)](pyproject.toml)
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
| 训练配置 | 严格的 `glr.training.v1` 知识源、生命周期策略、桥接要求与可审计加权奖励 |
| 采集 | 面向 PPO/IMPALA 的定长或终止边界 unroll，以及面向 BC/离线训练的 `glr.transition.v1` JSONL |
| 集成 | 可选 Gymnasium、TorchRL 0.13 与模型中立的 PyTorch BC/PPO/GAE/V-trace objective |
| 验证 | 失败即关闭的契约包装器与隐私安全的合成 conformance profiles |
| Agent 工作流 | `glr-adapter-builder` Skill，用于带来源的玩法研究、桥接脚手架、奖励和验证 |

具体游戏适配器、具体传输实现、自动生成的 C#/C++/Rust SDK、分布式 actor 传输、
完整训练器和参考模型仍属于[路线图](docs/planning/roadmap.md)工作。

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

## 将知识库与奖励定义为数据

```python
from game_learning_runtime import RewardComposer, RewardSignal, load_training_config

config = load_training_config("training.json")
reward = RewardComposer(config).compose(
    [RewardSignal(name="progress", source="runtime", value=0.25)]
)
print(reward.total, reward.contributions)
```

运行时遥测应标记为 `authoritative`；网页攻略和策略先验应标记为 `advisory`。
奖励项默认要求权威来源。配置只包含数据，GLR 不会把奖励表达式当作代码执行。

## 使用 Agent Skill 搭建适配器

仓库自带的 [glr-adapter-builder
Skill](.agents/skills/glr-adapter-builder/SKILL.md) 为新 Agent 提供边界明确的流程：

1. 研究当前玩法并保存来源；
2. 区分物理 `reset` 与真实语义的在线 `attach`；
3. 先搭建可训练的合成接缝；
4. 定义知识与奖励配置；
5. 通过运行时桥接实现带栅栏的观察和动作；
6. 完成 conformance 验证后，再执行小范围、已授权的真实运行时 trace。

该 Skill 不会把网页策略升级为运行时权威，也不会把合成测试描述成真实游戏验收。

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
    uses: loonghao/GameLearningRuntime/.github/workflows/reusable-python-ci.yml@v0.2.0 # x-release-please-version
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
provenance，并通过 Trusted Publishing 把同一份分发包发布到 PyPI。详见
[发布手册](docs/runbooks/release.md)。

## 文档

- [入门指南](docs/guides/getting-started.md)
- [构建可复用运行时桥接](docs/guides/runtime-bridges.md)
- [配置知识源和奖励](docs/guides/knowledge-and-rewards.md)
- [验证适配器](docs/guides/adapter-conformance.md)
- [适配现有 Gymnasium 环境](docs/guides/adapting-gymnasium.md)
- [组合自定义 Torch objectives](docs/guides/using-torch-objectives.md)
- [架构](docs/architecture/overview.md)与[数据流](docs/architecture/data-flow.md)
- [本地开发](docs/runbooks/local-development.md)
- [基准测试基线](docs/benchmarks/2026-08-31-data-plane-baseline.md)
- [路线图](docs/planning/roadmap.md)与[架构决策](docs/decisions/README.md)

开发契约见 [CONTRIBUTING.md](CONTRIBUTING.md)，私密漏洞报告流程见
[SECURITY.md](SECURITY.md)。GLR 使用 [MIT License](LICENSE)。
