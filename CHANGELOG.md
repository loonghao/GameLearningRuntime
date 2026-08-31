# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-08-31

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

## [0.1.0] - 2026-08-31

### Added

- Framework-neutral environment, tensor-tree, time-step, transition, and
  unroll contracts.
- Fail-closed runtime contract validation.
- Versioned Protobuf runtime service and JSONL transition records.
- Optional TorchRL environment adapter.
- Counter environment example, documentation, reusable CI, and release CD.

[Unreleased]: https://github.com/loonghao/GameLearningRuntime/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/loonghao/GameLearningRuntime/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/loonghao/GameLearningRuntime/releases/tag/v0.1.0
