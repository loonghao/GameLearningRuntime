# Game Learning Runtime

[![CI](https://github.com/loonghao/GameLearningRuntime/actions/workflows/ci.yml/badge.svg)](https://github.com/loonghao/GameLearningRuntime/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB.svg)](pyproject.toml)

Game Learning Runtime (GLR) is a framework-neutral contract between game
runtimes and learning systems. A game adapter describes observations, actions,
action masks, rewards, events, and episode boundaries once; TorchRL, custom PPO
or IMPALA learners, behavior cloning, offline datasets, evaluators, and QA tools
can then consume the same interface.

> A universal runtime for connecting games to learning systems and AI agents.

GLR is intended for games and test environments you own or are authorized to
instrument. It does not include anti-cheat bypasses, stealth injection, or
game-specific reverse-engineering code.

## Why this boundary

```text
Game / simulator
      │
      ▼
Runtime adapter (C#, C++, Rust, Python, official API, ...)
      │
      ▼
GLR protocol + environment contract
      │
      ├── TorchRL
      ├── custom PPO / IMPALA
      ├── BC / DAgger / offline learning
      ├── recorder / replay
      └── evaluation / automated QA
```

Game adapters never import PPO, IMPALA, BC, or TorchRL. Learning code never
needs to know whether the game is Unity, Unreal, Source, native, or a test
simulator. The standardized boundary is the data and lifecycle contract, not a
single implementation language or transport.

## Current capabilities

- Recursive tensor-tree specs for continuous, discrete, multi-discrete, binary,
  hybrid, parameterized, and hierarchical data.
- A `GameEnvironment` port with reset, step, close, action masks, semantic
  events, terminated/truncated signals, episode IDs, and monotonic step IDs.
- A fail-closed `ContractEnvironment` wrapper that validates every boundary.
- Capability-gated live attachment for continuing games that cannot claim a
  physical or deterministic reset.
- Fixed-length actor `Unroll` collection suitable for custom PPO and IMPALA.
- Versioned `glr.transition.v1` JSONL records for BC, replay, and offline data.
- A packaged `glr.v1` Protobuf service with unary and bidirectional streaming
  interaction contracts.
- An optional TorchRL `EnvBase` adapter tested against TorchRL 0.13.
- Optional, model-neutral PyTorch objectives for masked BC, PPO/GAE, and
  IMPALA/V-trace custom learners.
- A privacy-safe adapter conformance runner plus synthetic turn-based,
  real-time combat, FPS, and ARPG contract profiles.

Game-specific runtime adapters, generated C#/C++/Rust protocol SDKs,
distributed actor transport, complete trainers, and reference model
architectures remain roadmap items.

## Install

Install the core package from PyPI:

```powershell
uv add game-learning-runtime
```

Add the TorchRL integration only where training requires it:

```powershell
uv add "game-learning-runtime[torchrl]"
```

Use the reusable objectives in a custom PyTorch learner without TorchRL:

```powershell
uv add "game-learning-runtime[torch]"
```

Reuse an existing Gymnasium environment without writing another TorchRL adapter:

```powershell
uv add "game-learning-runtime[gymnasium,torchrl]"
```

Pin a PyPI version or immutable GitHub release tag when reproducibility requires
an exact build.

## Minimal environment

```python
import numpy as np

from game_learning_runtime import ContractEnvironment, SyncCollector
from game_learning_runtime.examples import CounterEnvironment, always_increment

environment = ContractEnvironment(CounterEnvironment(target=3))
collector = SyncCollector(environment, actor_id="local-actor")
unroll = collector.collect(always_increment, steps=16, policy_version=0)

print(len(unroll.transitions), unroll.total_reward)
```

For an adapter bound to an already-running game, advertise `live-attach` and
select the lifecycle explicitly:

```python
environment = ContractEnvironment(authorized_live_adapter)
collector = SyncCollector(environment, start_mode="attach")
unroll = collector.collect(policy, steps=128)
```

Attach creates a fresh logical GLR episode at step zero. It never implies that
the game world was reset or seeded.

Run the complete example from a clone:

```powershell
uv sync --frozen
uv run python -c "from game_learning_runtime import *; from game_learning_runtime.examples import *; print(SyncCollector(ContractEnvironment(make_environment())).collect(always_increment, steps=4).total_reward)"
```

For TorchRL:

```python
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.integrations.torchrl import TorchRLEnvironment

env = TorchRLEnvironment(CounterEnvironment())
rollout = env.rollout(max_steps=32)
```

For a custom masked PPO update:

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

## Reuse the CI workflow

Any uv-managed Python repository can call the public reusable workflow:

```yaml
jobs:
  quality:
    uses: loonghao/GameLearningRuntime/.github/workflows/reusable-python-ci.yml@v0.1.0
    with:
      python-versions: '["3.10", "3.12"]'
      sync-args: "--frozen --all-groups"
      lint-command: "uv run ruff check . && uv run mypy"
      test-command: "uv run pytest"
```

Pin a release tag or commit SHA in production repositories. The workflow never
receives deployment secrets and only checks out/tests the calling repository.

## Documentation

- [Getting started](docs/guides/getting-started.md)
- [Adapt an existing Gymnasium environment](docs/guides/adapting-gymnasium.md)
- [Compose custom Torch objectives](docs/guides/using-torch-objectives.md)
- [Validate an adapter with the conformance runner](docs/guides/adapter-conformance.md)
- [Architecture](docs/architecture/overview.md)
- [Protocol and data flow](docs/architecture/data-flow.md)
- [Local development](docs/runbooks/local-development.md)
- [Data-plane benchmark baseline](docs/benchmarks/2026-08-31-data-plane-baseline.md)
- [Release runbook](docs/runbooks/release.md)
- [Roadmap](docs/planning/roadmap.md)
- [Architecture decisions](docs/decisions/README.md)

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development contract and
[SECURITY.md](SECURITY.md) for private vulnerability reporting. GLR is licensed
under the [MIT License](LICENSE).
