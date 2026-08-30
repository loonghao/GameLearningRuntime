# Roadmap

This document describes future work, not current capability.

## Next

- Generate versioned Python, C#, C++, and Rust clients from `glr.v1`.
- Prototype checksummed codec and multi-actor queue candidates against the
  recorded Python baseline; retain Rust only when ADR-0004's paired threshold
  is reproducible.
- Implement an authenticated local gRPC adapter with deadlines and health.
- Add bounded asynchronous actor queues and IMPALA backpressure metrics.
- Add a columnar/checksummed dataset container and deterministic replay.
- Define partial reset and multi-agent identity semantics.
- Add structured Gymnasium action-mask conformance fixtures.

## Later

- Unity, Unreal, and native adapter SDK templates.
- JAX and RLlib integrations.
- Reference BC, PPO, IMPALA/V-trace, and DAgger examples outside the core.
- Evaluation suites, curriculum contracts, snapshots, and benchmarks.

Game-specific adapters remain separate packages so their engine/runtime
dependencies never enter the core distribution.
