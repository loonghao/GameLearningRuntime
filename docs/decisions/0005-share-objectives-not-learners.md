# ADR-0005: Share objective primitives, not learner implementations

## Status

Accepted

## Context

Several authorized game projects need the same masked behavior-cloning,
PPO/GAE, and IMPALA/V-trace mathematics. Copying those functions into every
project creates semantic drift, especially around invalid actions, termination,
truncation, and gradient detachment.

A shared trainer or model hierarchy would create a different problem. Network
architecture, observation encoding, reward shaping, optimization schedules,
collector topology, and operational policy are project decisions. Making them
part of the runtime would couple game adapters to a canonical learner and
violate the learner-neutral boundary in ADR-0001.

## Decision

Provide optional, pure PyTorch objective primitives at the outward integration
boundary. They accept explicit tensors, perform no collection or optimization,
and return typed loss components or detached targets.

The shared layer covers discrete masked logits, behavior cloning, GAE, clipped
PPO, V-trace, and the compositional IMPALA loss. It distinguishes `terminated`
from `truncated`: termination disables bootstrapping, while truncation retains
the next-state value but stops cross-episode recursion.

Keep models, optimizers, replay selection, batching, reward functions, encoder
semantics, and distributed actor orchestration in consuming projects. Keep
PyTorch optional through the `torch` and `torchrl` extras. Do not rewrite these
tensor objectives in Rust: PyTorch already dispatches their tensor operations
to native CPU or accelerator kernels, and ADR-0004 reserves Rust for measured
data-plane bottlenecks.

## Consequences

### Positive

- Projects share tested algorithm mathematics without sharing game semantics.
- Mask and episode-boundary behavior remains consistent across learners.
- Custom PyTorch users do not need to install TorchRL.
- Typed components remain easy to combine with project-specific metrics and
  optimization schedules.

### Negative

- The integration must track supported PyTorch behavior and packaging size.
- Callers remain responsible for batching, devices, mixed precision, and
  distributed execution.
- Only discrete categorical policies are covered initially.

### Neutral

- Full reference learners may be published later as examples outside the core.

## Alternatives considered

- Keep project-local copies: rejected because repeated bug fixes and subtle
  termination differences are already a maintenance cost.
- Add a canonical trainer and model base class: rejected because it would own
  project policy that is unrelated to the runtime contract.
- Use TorchRL loss modules exclusively: rejected because custom learners need a
  small stable surface and TorchRL must remain optional.
- Implement objectives in Rust: rejected unless future evidence identifies a
  non-kernel bottleneck that meets ADR-0004's benchmark gate.
