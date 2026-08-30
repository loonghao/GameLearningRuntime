# Data-plane baseline — 2026-08-31

## Scope

This snapshot establishes a privacy-safe Python baseline and evaluates one
Rust-backed JSON candidate. It is evidence for ADR-0004, not a general hardware
or library ranking.

Runtime class: Windows AMD64, CPython 3.12.10. Workloads use only deterministic
synthetic arrays: 256 float32 observation elements, 32 transitions per
trajectory, and a bounded queue capacity of 256. No hostname, CPU model, path,
PID, game state, or private dataset was recorded.

## Python baseline

The longer baseline used 100 samples × 100 operations:

| Workload | Throughput | p50 | p95 |
|---|---:|---:|---:|
| Transition JSON round trip | 8,789 ops/s | 95.572 µs | 210.241 µs |
| Build a 32-transition trajectory | 1,465 ops/s | 687.041 µs | 799.902 µs |
| Bounded queue put/get round trip | 890,456 ops/s | 1.084 µs | 1.455 µs |

These values are a local comparison baseline, not release guarantees. The
queue case is single-threaded and cannot justify a production MPMC design by
itself.

## Rust-backed JSON candidate

`orjson` 3.12.0 was evaluated as an uncommitted benchmark-only candidate using
six alternating stdlib/candidate pairs. Each run used 40 samples × 20 complete
Transition→JSON→Transition operations with the same tensors.

| Decision metric | Paired median |
|---|---:|
| Throughput ratio | 1.069× |
| p95 reduction | -3.685% |
| Pairs meeting either ADR threshold | 0 / 6 |

One earlier unpaired run showed a 33.6% p95 reduction, but the alternating
pairs did not reproduce it. The candidate therefore does not satisfy the
benchmark gate and is not added as a runtime dependency.

## Decision

- Keep the current standard-library JSON path.
- Do not add a Rust JSON dependency based on this evidence.
- Retain Rust as a candidate for a future checksummed binary codec or bounded
  multi-actor queue only after those public semantics and representative
  workloads exist.
- Re-run the same schema and paired methodology when a candidate changes the
  actual end-to-end path rather than only one inner call.
