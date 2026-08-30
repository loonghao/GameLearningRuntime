# ADR-0004: Use Rust for benchmark-proven data-plane work

## Status

Accepted

## Context

High-throughput actors will eventually spend material time in protocol framing,
tensor validation, trajectory encoding, bounded queues, shared memory, and
dataset indexing. Rust can improve throughput, latency, memory predictability,
and native adapter reuse in those areas.

Moving every Python component to Rust would instead create duplicate domain
models, packaging overhead, and FFI copies. PyTorch and TorchRL tensor operations
already execute in native CPU or accelerator kernels, so rewriting their Python
orchestration does not by itself make training faster.

## Decision

Keep one semantic source of truth: the versioned GLR schemas and conformance
fixtures. Add Rust first for native SDKs and measured data-plane bottlenecks:

- Protobuf and local transport framing;
- validated tensor and trajectory codecs;
- checksummed replay/dataset containers;
- bounded actor queues, backpressure, and shared-memory ownership.

Do not reimplement learner algorithms or game semantics in Rust. A Rust
component must preserve byte/schema compatibility, fail closed on malformed
lengths and identities, and pass the same cross-language fixtures.

Adoption requires a representative benchmark. Prefer Rust when it provides at
least twice the throughput or reduces p95 latency by at least 30 percent without
an unacceptable memory, packaging, or portability regression. Record the
baseline, workload, hardware class, and result without publishing hostnames,
user paths, process IDs, or private dataset contents.

## Consequences

### Positive

- Rust work targets the places where language choice can materially help.
- Python, C#, C++, and Rust adapters share one wire and lifecycle contract.
- Cross-language fixtures prevent a second incompatible implementation.

### Negative

- Benchmark and FFI maintenance become release gates for native components.
- Platform wheels or standalone binaries add CI and signing cost.
- Some Python-only workloads will correctly remain Python-only.

### Neutral

- Early experiments may fail the adoption threshold and be removed.

## Alternatives considered

- Rewrite the full framework in Rust: rejected because learner ecosystems and
  existing game projects are Python/TorchRL-heavy and would require duplicate
  bindings.
- Never use Rust: rejected because native transport and storage paths need
  predictable throughput and memory behavior.
- Choose a Rust RL framework as the core: deferred; current Rust RL APIs remain
  younger than the Python ecosystem, and the runtime contract must stay learner
  neutral.
