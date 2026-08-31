# Roadmap

This document describes future work, not current capability.

## Next

- Generate versioned Python, C#, C++, and Rust clients from `glr.v1`.
- Prototype checksummed codec and multi-actor queue candidates against the
  recorded Python baseline; retain Rust only when ADR-0004's paired threshold
  is reproducible.
- Implement concrete authenticated local gRPC and framed-stream drivers with
  deadlines, health, exact runtime binding, and bounded payloads.
- Add bounded asynchronous actor queues and IMPALA backpressure metrics.
- Add a columnar/checksummed dataset container and deterministic replay.
- Define partial reset and multi-agent identity semantics.
- Add structured Gymnasium action-mask conformance fixtures.
- Add freshness-aware knowledge snapshot validation and signed configuration
  bundles after real adapter requirements are measured.
- Generate versioned Unity/C# and Unreal/C++ engine-plugin SDK templates from
  `glr.runtime-integration.v2` and `glr.v1`.
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
