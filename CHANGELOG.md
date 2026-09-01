# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.8.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.7.0...v0.8.0) (2026-09-01)


### Features

* trace knowledge injection queries ([e6c067d](https://github.com/loonghao/GameLearningRuntime/commit/e6c067d24c4ce89c49d2909875c4d21242a5646f))

## [0.7.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.6.1...v0.7.0) (2026-09-01)


### Features

* distribute skills as agent plugin ([af70610](https://github.com/loonghao/GameLearningRuntime/commit/af7061092994a4f4a35094b5a6d3157bd4081606))

## [0.6.1](https://github.com/loonghao/GameLearningRuntime/compare/v0.6.0...v0.6.1) (2026-09-01)


### Bug Fixes

* upload hidden release archives ([68a1d12](https://github.com/loonghao/GameLearningRuntime/commit/68a1d12dde856e3524d3938facc846b1ec25a67a))

## [0.6.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.5.0...v0.6.0) (2026-09-01)


### Features

* add agent-first training control plane ([240a02e](https://github.com/loonghao/GameLearningRuntime/commit/240a02e3e25f5575b710884891b28652cafa993d))
* make rust cli the primary entrypoint ([b3da49e](https://github.com/loonghao/GameLearningRuntime/commit/b3da49ea44054ac96da0ca8336c764fb1ef7c26c))


### Bug Fixes

* support virtual workspace releases ([9f87d76](https://github.com/loonghao/GameLearningRuntime/commit/9f87d7649a9acd7866c0ca17c1caa59c054035ea))

## [0.5.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.4.0...v0.5.0) (2026-08-31)


### Features

* add bounded knowledge injection ([08ca15f](https://github.com/loonghao/GameLearningRuntime/commit/08ca15fb2fd2e2dd74783300d16f69a633c3a4a4))
* bind demonstration provenance to trajectory bytes ([fcb3e6e](https://github.com/loonghao/GameLearningRuntime/commit/fcb3e6ef6dfbab4bd5416d41a57794a7ea26f695))

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
