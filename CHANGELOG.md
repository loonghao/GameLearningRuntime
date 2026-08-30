# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Optional deny-by-default Gymnasium compatibility adapter for Box, Discrete,
  MultiDiscrete, MultiBinary, Dict, and Tuple spaces.
- Dedicated Gymnasium conformance CI and architecture decisions for adapter
  reuse, metadata privacy, and benchmark-gated Rust data-plane work.
- Optional model-neutral PyTorch objectives for masked behavior cloning,
  PPO/GAE, and IMPALA/V-trace, including explicit termination and truncation
  semantics.

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

[Unreleased]: https://github.com/loonghao/GameLearningRuntime/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/loonghao/GameLearningRuntime/releases/tag/v0.1.0
