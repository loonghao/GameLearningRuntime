# Roadmap

This document describes future work, not current capability.

## Next

- Generate versioned Python, C#, C++, and Rust clients from `glr.v1`.
- Implement an authenticated local gRPC adapter with deadlines and health.
- Add bounded asynchronous actor queues and IMPALA backpressure metrics.
- Add a columnar/checksummed dataset container and deterministic replay.
- Define partial reset and multi-agent identity semantics.
- Publish the package to PyPI using trusted publishing.

## Later

- Unity, Unreal, and native adapter SDK templates.
- Gymnasium, JAX, and RLlib integrations.
- Reference BC, PPO, IMPALA/V-trace, and DAgger examples outside the core.
- Evaluation suites, curriculum contracts, snapshots, and benchmarks.

Game-specific adapters remain separate packages so their engine/runtime
dependencies never enter the core distribution.

