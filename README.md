# Game Learning Runtime

English | [简体中文](README.zh-CN.md)

[![PyPI](https://img.shields.io/pypi/v/game-learning-runtime.svg)](https://pypi.org/project/game-learning-runtime/)
[![CI](https://github.com/loonghao/GameLearningRuntime/actions/workflows/ci.yml/badge.svg)](https://github.com/loonghao/GameLearningRuntime/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB.svg)](pyproject.toml)
[![Rust](https://img.shields.io/badge/Rust-1.98.0-000000.svg)](rust-toolchain.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Game Learning Runtime (GLR) is an agent-first, learner-neutral control plane for
game learning. It gives an agent one stable JSON CLI and strict contracts to
start an authorized game bridge, train while recording review video, query prior
runs and world knowledge, research guides, revise bounded training and reward
plans, and reproduce a verified model bundle in a new game instance.

Define observations, actions, masks, rewards, events, and episode boundaries
once. The same adapter remains reusable with TorchRL, custom PPO or IMPALA,
behavior cloning, offline datasets, evaluation, and automated QA.

> A game-learning runtime designed to be operated by agents.

GLR is for games and test environments you own or are authorized to instrument.
It does not include anti-cheat bypasses, stealth injection, or game-specific
reverse-engineering code.

## See GLR running

![GLR collector running against the bundled synthetic counter adapter](docs/assets/showcase/glr-counter-collector.gif)

This GIF is generated from a real local `ContractEnvironment` + `SyncCollector`
run against GLR's explicitly synthetic counter adapter. It demonstrates the
public contract without exposing a game account, machine path, process/window
identity, or proprietary runtime data. See the [showcase provenance and capture
policy](docs/assets/showcase/README.md) before contributing footage from a live
adapter.

## One boundary, many consumers

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

Game adapters never import PPO, IMPALA, BC, or TorchRL. Learning code does not
need to know whether the runtime is Unity, Unreal, Source, native, or a test
simulator. GLR standardizes the data and lifecycle boundary, not one language,
engine, transport, or algorithm.

## What GLR standardizes

| Contract | Included today |
| --- | --- |
| Environment | `reset`, truthful live `attach`, `step`, `close`, termination, truncation, episode and step identity |
| Data | Recursive tensor specs, hybrid/parameterized actions, masks, events, rewards, immutable transitions |
| Bridge | Capability negotiation, reset/step fencing, metadata deny-by-default, transport-neutral driver ports |
| Runtime Host | Rust `glr-hostd`, bounded `glr.host.v1` stdio, Python `HostBridgeDriver`, synthetic process smoke |
| Provider SDKs | .NET Standard 2.0 C# contract for Unity/BepInEx and header-only C++20 contract for Unreal/native providers |
| Runtime integration | Backward-compatible `glr.runtime-integration.v2` profiles for source plugins, authorized loader plugins, and external attachments |
| Training config | Strict `glr.training.v1` knowledge sources, lifecycle policy, bridge requirements, auditable weighted rewards |
| Training safety | Episode shaping budgets, mandatory terminal outcomes, failed-return ceilings, BC provenance gates, and checksummed demonstration artifacts |
| Collection | Fixed-length or terminal-bounded unrolls for PPO/IMPALA plus `glr.transition.v1` JSONL for BC/offline use |
| Integrations | Optional Gymnasium, TorchRL 0.13, and model-neutral PyTorch BC/PPO/GAE/V-trace objectives |
| Validation | Fail-closed contract wrapper and privacy-safe synthetic conformance profiles |
| Agent control plane | Standalone Rust `glr` JSON CLI, strict project roles, bounded research/plan/train/evaluate goals, SQLite run queries, spatial knowledge transfer, managed binary/Skill updates |
| Review and supervised capture | Concurrent project-owned H.264 capture with checksummed episode/step-to-frame index |
| Agent workflow | Separate `glr-adapter-builder` and `glr-cli` Skills for adapter construction versus operation |
| Model reproduction | `glr.model-bundle.v1` copies config, source/lock inputs, seeds, versions, weights, metrics, and SHA-256 provenance |

Game-specific adapters, authenticated target-bound local provider transport,
distributed actor transport, production trainers, and reference policies remain
[roadmap](docs/planning/roadmap.md) work.

## Quick start

Install the matching standalone archive from the [latest GitHub
Release](https://github.com/loonghao/GameLearningRuntime/releases/latest). Verify
`glr-{version}-{rust-target}.zip` against `SHA256SUMS`, then put `glr` and
`glr-hostd` on `PATH`. The archive also carries the `glr-cli` and
`glr-adapter-builder` Skills.

The CLI is the primary deployment, integration, training, query, and playback
entrypoint. It does not require Python:

```powershell
glr --version
glr --project . --json doctor
glr --json update --check
```

Install the Python SDK only when a trainer, adapter, or learner imports it:

```powershell
uv add game-learning-runtime
```

Choose optional integrations only where they are needed:

```powershell
uv add "game-learning-runtime[torchrl]"
uv add "game-learning-runtime[torch]"
uv add "game-learning-runtime[gymnasium,torchrl]"
```

### Agent-operated project

Once an authorized project provides `glr-project.json` and its fixed bridge
roles, an agent can use the same machine-readable interface for the complete
workflow:

```powershell
glr --project . --json doctor
glr --project . --json runtime start
glr --project . --json train
glr --project . --json goal run --goal goals/reach-destination.json
glr --project . --json query entities --world forest --kind shrine
glr --project . --json query routes --world forest --to-entity shrine.forest-1
glr --project . --json query research --tag navigation
glr --project . --json play --bundle artifacts/model-bundle
```

The project manifest owns exact executable paths, environment identity, data
locations, and runtime/trainer/player/researcher/planner/evaluator/recorder
roles. GLR validates and orchestrates those roles; it does not embed a
game-specific launcher, scraper, or learning algorithm.

`glr update --check` only inspects the latest stable release. After an explicit
update request, `glr update --yes` verifies the exact target archive and
`SHA256SUMS`, then updates `glr`, its sibling `glr-hostd`, and the project-owned
Skills. It never changes game code, trainer dependencies, models, datasets, or
project configuration. Use `--no-skills` for binary-only maintenance or
`--skills-dir` for an explicitly selected Skills directory.

### Library integration

Collect a learner-neutral unroll:

```python
from game_learning_runtime import ContractEnvironment, SyncCollector
from game_learning_runtime.examples import CounterEnvironment, always_increment

environment = ContractEnvironment(CounterEnvironment(target=3))
collector = SyncCollector(environment, actor_id="local-actor")
unroll = collector.collect(always_increment, steps=16, policy_version=0)

print(len(unroll.transitions), unroll.total_reward)
```

For an authorized already-running game, advertise `live-attach` and select that
lifecycle explicitly:

```python
environment = ContractEnvironment(authorized_live_adapter)
collector = SyncCollector(environment, start_mode="attach")
unroll = collector.collect(policy, steps=128, stop_on_done=True)
```

Attach starts a fresh logical GLR episode at step zero. It never claims that the
physical game world was reset or seeded.

## From a user goal to verified replay

```text
user goal + hard budgets
          │
          ▼
allowed rules / text / video + prior runs / spatial knowledge
          │
          ▼
research ──► plan ──► train + indexed video ──► authoritative evaluation
                ▲                                      │
                └──────── bounded revision ────────────┘
                                                       │
                                                       ▼
                                   queryable evidence + model bundle
                                                       │
                                                       ▼
                                      verified replay in a new instance
```

Training can run a project-owned small-window recorder concurrently and bind its
H.264 output to episode/step IDs for human review and later supervised-data
selection. Goal runs gather allowed official rules, text guides, video tutorials,
and runtime traces through the configured researcher; adjust declarative reward
plans between bounded trials; and stop only on matching authoritative runtime
metrics. Guides and transferred knowledge remain advisory until fresh runtime
evidence verifies them. See [Operate GLR as an agent-first control
plane](docs/guides/agent-first-cli.md).

## Unity and Unreal integration lanes

GLR keeps one learner-facing contract while making the runtime boundary
explicit:

- **Source or official extension SDK:** run an engine plugin with semantic
  state, native actions, main-thread dispatch, controllable time, and a truthful
  physical reset.
- **Authorized binary-only runtime:** attach externally through an official API,
  telemetry, replay, or bounded rendered observation/input seam. It defaults to
  real-time `attach`, exact target binding, input lease cleanup, and verified
  post-state.
- **Authorized mod-loader runtime:** host a reviewed bounded-command adapter in
  BepInEx or UE4SS. It keeps truthful real-time `attach`, semantic observations,
  game-thread dispatch, exact loader/version provenance, and an empty-deny
  action vocabulary until game-specific handlers are reviewed.

```python
from game_learning_runtime import EngineFamily, RuntimeIntegrationProfile

profile = RuntimeIntegrationProfile.for_source(EngineFamily.UNITY)
environment = profile.connect(authorized_driver)
```

Generate a Unity or Unreal adapter lane with the repository-owned Skill, then
replace its synthetic semantics while keeping the contract tests green. See the
[engine runtime integration guide](docs/guides/engine-runtime-integration.md).

For no-source games that explicitly permit mods, GLR can generate a BepInEx 5
LTS Unity Mono host or a UE4SS 3.x Lua host:

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

The generated deployment command stages a checksummed payload; it never scans
for or modifies a game installation. See the [loader-plugin integration
guide](docs/guides/loader-plugin-integration.md).

## Runtime Host and engine providers

The implemented Runtime Host centralizes strict framing and lifecycle fencing
without trying to replace engine bootstraps:

```text
TorchRL / PPO / IMPALA / BC
  -> BridgeEnvironment -> HostBridgeDriver
  -> glr-hostd (Rust)
  -> C# Unity provider / C++ Unreal provider
  -> official plugin, BepInEx, UE4SS, or official mod SDK
```

Run the real cross-process conformance path and compile both provider contracts:

```powershell
vx just host-smoke
vx just provider-sdk-check
```

`glr-hostd` currently ships only `synthetic-counter` over serialized stdio. It
has a 1 MiB hard frame bound and never retries a mutating action, but it does
not yet claim authenticated or target-bound IPC and cannot yet connect a live
external C#/C++ provider. See the [Runtime Host and provider SDK guide](docs/guides/runtime-host-and-provider-sdks.md)
for the exact current boundary and Unity/Unreal implementation path.

## Reproducible local development

GLR pins Python, uv, just, rustup, Rust, and .NET SDK inputs. Local development
and GLR's GitHub Actions execute the same recipes:

```powershell
vx setup
vx just check
vx just ci
```

## Knowledge and rewards as data

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

Runtime telemetry should be `authoritative`; web guides and strategy priors
should be `advisory`. Reward terms require authoritative sources by default.
Configuration is data only: GLR does not evaluate reward expressions as code.
`KnowledgeInjector` additionally validates bounded `glr.knowledge-snapshot.v1`
payloads and selects stage/tag-relevant `acquire`, `engage`, `upgrade`, and
`avoid` advice into an immutable learner context. The learner owns encoding;
the context never gains action or reward authority.
The episode guard caps positive shaping per step and episode, requires an
authoritative terminal outcome, and prevents a failed episode from retaining a
positive return. `DemonstrationGate` separately rejects policy self-imitation,
failed episodes, and unknown provenance from BC by default. See [training
safety](docs/guides/training-safety.md).

## Build an adapter with the Agent Skill

The repository-owned [glr-adapter-builder
Skill](.agents/skills/glr-adapter-builder/SKILL.md) gives a new agent a bounded
workflow for:

1. researching current game mechanics with source provenance;
2. separating physical `reset` from truthful live `attach`;
3. scaffolding a synthetic trainable seam;
4. defining knowledge, reward budgets, and BC provenance policy;
5. implementing fenced observations/actions through a runtime bridge; and
6. validating conformance before a bounded authorized runtime trace.

The Skill never turns web strategy into runtime authority and never treats a
synthetic test as live-game acceptance.

### Give the Skill to your agent

The fastest path is to clone this repository and start Codex from its root.
Codex discovers repository skills under `.agents/skills` automatically. Invoke
the workflow explicitly in your prompt:

```text
$glr-adapter-builder Create an authorized Unity adapter with source access. Scaffold a trainable environment, research manifest, reward configuration, and contract tests.
```

For an authorized binary-only runtime, say `external access`. For an authorized
mod-enabled runtime, name `BepInEx` or `UE4SS` and the exact compatible upstream
tag. The Skill chooses truthful `attach`, denies unknown actions, and refuses
source-only capability claims.

To use the Skill from another repository, ask Codex's built-in installer to
install it from GitHub:

```text
$skill-installer install https://github.com/loonghao/GameLearningRuntime/tree/main/.agents/skills/glr-adapter-builder
```

Start a new agent turn after installation, then invoke
`$glr-adapter-builder`. Pin the GitHub URL to a release tag or commit SHA when
you need a reproducible team setup. Agents that implement the open Agent Skills
standard can instead place the same `glr-adapter-builder` directory under the
target repository's `.agents/skills/` directory. See the [official Codex Skills
documentation](https://developers.openai.com/codex/skills).

The generated lane includes the environment skeleton, `training.json`,
`reward-safety.json`, `demonstration-policy.json`, `runtime-integration.json`, a
provenance-aware research manifest, tests, Agent instructions, a model-bundle
smoke trainer, `vx.toml`, and a `justfile`.
Loader lanes also include bounded host source and a deployment manifest. From
that generated directory, run:

```powershell
vx setup
vx run check
vx run train
vx run reproduce
```

`train` emits a synthetic BC smoke model plus a self-contained checksummed
reproduction environment. Replace the learner while preserving the
`glr.model-bundle.v1` gate; see [reproducible model
bundles](docs/guides/reproducible-model-bundles.md).

For projects whose bridge already exists, use the separate
[`glr-cli` Skill](.agents/skills/glr-cli/SKILL.md). It teaches agents to configure
and operate runtime, training capture, goal loops, history queries, knowledge
transfer, verified playback, and explicitly authorized managed updates without
changing adapter internals. Both Skills ship in every standalone GLR archive.

## TorchRL and custom learners

Use the optional TorchRL adapter:

```python
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.integrations.torchrl import TorchRLEnvironment

env = TorchRLEnvironment(CounterEnvironment())
rollout = env.rollout(max_steps=32)
```

Or reuse the masked PPO objective in a custom PyTorch learner:

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

Any uv-managed Python repository can call GLR's public reusable workflow:

```yaml
jobs:
  quality:
    uses: loonghao/GameLearningRuntime/.github/workflows/reusable-python-ci.yml@v0.6.0 # x-release-please-version
    with:
      python-versions: '["3.10", "3.12"]'
      sync-args: "--frozen --all-groups"
      lint-command: "uv run ruff check . && uv run mypy"
      test-command: "uv run pytest"
```

Pin a release tag or commit SHA in production. Release Please keeps the example
tag synchronized with package releases. The reusable workflow receives no
deployment secrets and only checks out and tests the calling repository.

## Releases

Conventional Commits on `main` create or update a Release Please pull request.
Merging that reviewed PR creates the tag and GitHub Release, verifies and builds
the tagged source, attaches provenance, publishes the Python distributions to
PyPI through Trusted Publishing, and attaches checksummed unified GLR archives
for Linux, Windows, Intel macOS, and Apple Silicon. Each archive contains the
standalone Rust CLI, matching Runtime Host, install manifest, and both GLR
Skills; the release also includes the C# provider package.
See the [release
runbook](docs/runbooks/release.md).

## Documentation

- [Getting started](docs/guides/getting-started.md)
- [Operate GLR as an agent-first control plane](docs/guides/agent-first-cli.md)
- [Build a reusable runtime bridge](docs/guides/runtime-bridges.md)
- [Connect Unity and Unreal runtimes](docs/guides/engine-runtime-integration.md)
- [Use the Runtime Host and C#/C++ provider SDKs](docs/guides/runtime-host-and-provider-sdks.md)
- [Connect authorized BepInEx and UE4SS loaders](docs/guides/loader-plugin-integration.md)
- [Reproduce trained models](docs/guides/reproducible-model-bundles.md)
- [Configure knowledge sources and rewards](docs/guides/knowledge-and-rewards.md)
- [Enforce reward budgets and BC provenance](docs/guides/training-safety.md)
- [Validate an adapter](docs/guides/adapter-conformance.md)
- [Adapt an existing Gymnasium environment](docs/guides/adapting-gymnasium.md)
- [Compose custom Torch objectives](docs/guides/using-torch-objectives.md)
- [Architecture](docs/architecture/overview.md) and [data flow](docs/architecture/data-flow.md)
- [Local development](docs/runbooks/local-development.md)
- [Benchmark baseline](docs/benchmarks/2026-08-31-data-plane-baseline.md)
- [Roadmap](docs/planning/roadmap.md) and [architecture decisions](docs/decisions/README.md)

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development contract and
[SECURITY.md](SECURITY.md) for private vulnerability reporting. GLR is licensed
under the [MIT License](LICENSE).
