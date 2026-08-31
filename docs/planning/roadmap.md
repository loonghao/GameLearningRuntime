# Roadmap

This document describes future work, not current capability.

## Next

- Generate codecs from `glr.v1`/`glr.host.v1` after the implemented Python,
  Rust, C# provider, and C++ provider contracts stabilize.
- Prototype checksummed codec and multi-actor queue candidates against the
  recorded Python baseline; retain Rust only when ADR-0004's paired threshold
  is reproducible.
- Extend the implemented bounded stdio Runtime Host with authenticated,
  target-bound named-pipe/Unix-socket provider and learner connections,
  deadlines, health, and reconnect reconciliation.
- Add bounded asynchronous actor queues and IMPALA backpressure metrics.
- Add a columnar/checksummed dataset container and deterministic replay.
- Define partial reset and multi-agent identity semantics.
- Add structured Gymnasium action-mask conformance fixtures.
- Add freshness-aware knowledge snapshot validation and signed configuration
  bundles after real adapter requirements are measured.
- Add Unity and Unreal engine-plugin templates around the implemented
  engine-neutral C# and C++ provider contracts.
- Add an authorized external-attach reference driver with bounded rendered
  observations, input leases, exact target binding, and post-state receipts.
- Add separately tested BepInEx IL2CPP and UE4SS C++ loader templates after
  upstream compatibility matrices and live acceptance fixtures exist.
- Add signed model/dataset attestations and platform metadata without weakening
  the portable `glr.model-bundle.v1` core.

## Later

- Optional Unity ML-Agents and Unreal Learning Agents interoperability packages.
- JAX and RLlib integrations.
- Reference BC, PPO, IMPALA/V-trace, and DAgger examples outside the core.
- Evaluation suites, curriculum contracts, richer reward ablation reports, snapshots,
  and benchmarks.

Game-specific adapters remain separate packages so their engine/runtime
dependencies never enter the core distribution.
