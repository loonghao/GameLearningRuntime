# Data-plane benchmarks

These benchmarks answer a narrow question: does a native candidate materially
improve a current GLR data-plane operation without using real game data or
publishing machine identity?

The suite uses deterministic synthetic tensors and reports only Python version,
platform class, architecture class, workload parameters, throughput, and
p50/p95 latency. It never reads a game adapter, dataset, hostname, user path,
process ID, or wall-clock timestamp.

Run the Python baseline:

```powershell
uv run python -m benchmarks.data_plane `
  --json-backend stdlib `
  --samples 100 `
  --operations-per-sample 100 `
  --observation-width 256 `
  --queue-capacity 256
```

Evaluate the Rust-backed JSON experiment without adding a project dependency:

```powershell
uv run --with orjson python -m benchmarks.data_plane `
  --json-backend orjson `
  --samples 100 `
  --operations-per-sample 100 `
  --observation-width 256 `
  --queue-capacity 256
```

Run candidates and baselines in alternating order multiple times. ADR-0004
permits adoption only when the representative paired result reaches at least
2× throughput or reduces p95 latency by at least 30%, without an unacceptable
memory, packaging, or portability regression. A single qualifying run is not
enough when repeated pairs do not reproduce it.

The queue workload is currently a single-thread, non-blocking baseline using
Python objects. It establishes a comparison point; it does not prove a future
MPMC queue's semantics or performance. Any Rust queue candidate must first
define bounded blocking, closure, ownership, backpressure, and cancellation
contracts and then pass a representative multi-actor benchmark.
