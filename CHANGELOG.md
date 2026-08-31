# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.4.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.3.0...v0.4.0) (2026-08-31)


### Features

* **gymnasium:** expose verified capabilities ([8fa57dd](https://github.com/loonghao/GameLearningRuntime/commit/8fa57dd115188836c4ba24f672a0fb32b15bac90))
* **torch:** support demonstration sample weights ([01c053e](https://github.com/loonghao/GameLearningRuntime/commit/01c053e9a5002fce935298e8f291c389cdd8be59))

## [0.3.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.2.0...v0.3.0) (2026-08-31)


### Features

* add engine runtime integration profiles ([955489b](https://github.com/loonghao/GameLearningRuntime/commit/955489b2f9dcdf0e8e72344db1bcabef1627f53e))
* add loader and training safety toolkit ([#18](https://github.com/loonghao/GameLearningRuntime/issues/18)) ([c22cbb2](https://github.com/loonghao/GameLearningRuntime/commit/c22cbb2a0fbb60c39ac5e81e07aa7ba228067908))
* add runtime host and provider SDKs ([e1bb13f](https://github.com/loonghao/GameLearningRuntime/commit/e1bb13f307e13674eed7f49a432fe60541528d16))


### Bug Fixes

* **ci:** harden release checksum glob ([9e01292](https://github.com/loonghao/GameLearningRuntime/commit/9e01292aaabb8d7f658070eeca5f87283801f3e0))

## [0.2.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.1.0...v0.2.0) (2026-08-31)

### Added

- Optional terminal-bounded synchronous collection with `stop_on_done=True`,
  preserving the default fixed-length cross-episode behavior.
- Capability-gated live attach across the core environment, synchronous
  collector, TorchRL adapter, and additive `glr.v1` protocol RPC, without
  aliasing attachment to deterministic reset.
- Explicit Gymnasium `attach_provider` forwarding for authorized running games,
  preserving the distinction between logical attachment and physical reset.
- Optional deny-by-default Gymnasium compatibility adapter for Box, Discrete,
  MultiDiscrete, MultiBinary, Dict, and Tuple spaces.
- Dedicated Gymnasium conformance CI and architecture decisions for adapter
  reuse, metadata privacy, and benchmark-gated Rust data-plane work.
- Optional model-neutral PyTorch objectives for masked behavior cloning,
  PPO/GAE, and IMPALA/V-trace, including explicit termination and truncation
  semantics.
- A privacy-safe synthetic data-plane benchmark for transition JSON round
  trips, trajectory construction, bounded actor-queue handoff, and mechanical
  application of the Rust adoption threshold.
- A reusable environment conformance runner and four synthetic contract
  profiles covering turn-based masks, real-time hybrid actions, FPS controls,
  and nested ARPG actions.
- Transport-neutral client and server bridge ports with protocol/capability
  negotiation, immutable reset/step requests, episode/step fencing, metadata
  deny-by-default, and no implicit action retry.
- Strict `glr.training.v1` knowledge-source, lifecycle, bridge-capability, and
  reward configuration with source authority checks and immutable reward
  contribution breakdowns.
- A project-owned adapter-builder Agent Skill with a synthetic reset/attach
  scaffold, provenance-first gameplay research workflow, and a privacy-safe
  research-manifest validator.

### Fixed

- Copy immutable NumPy bounds before creating TorchRL specs, avoiding
  non-writable tensor warnings and undefined write behavior.

## [0.1.0](https://github.com/loonghao/GameLearningRuntime/releases/tag/v0.1.0) (2026-08-31)

### Added

- Framework-neutral environment, tensor-tree, time-step, transition, and
  unroll contracts.
- Fail-closed runtime contract validation.
- Versioned Protobuf runtime service and JSONL transition records.
- Optional TorchRL environment adapter.
- Counter environment example, documentation, reusable CI, and release CD.
