# ADR-0001: Keep the runtime contract learner-neutral

## Status

Accepted

## Context

The same game runtime data must support TorchRL, custom PPO and IMPALA,
behavior cloning, offline learning, evaluation, and QA. Coupling a game adapter
to one learner makes engines, datasets, and operational tooling hard to reuse.

## Decision

Define observations, structured actions, masks, rewards, events, reset, and
episode termination in a framework-neutral core. Game integrations implement
the `GameEnvironment` port. TorchRL and future learning frameworks live in
optional outward adapters.

## Consequences

### Positive

- Game adapters and datasets remain reusable across algorithms.
- The core installs without Torch or a game engine.
- Learner integrations can evolve independently.

### Negative

- Framework-specific conveniences require explicit adapters.
- Some zero-copy optimizations will need transport-specific extensions.

### Neutral

- Python/NumPy is the first reference implementation, not the wire-language
  requirement.

## Alternatives considered

- TorchRL as the core API: rejected because it makes Torch a runtime dependency.
- Gymnasium as the core API: rejected because nested masks, events, actor
  metadata, and streaming protocol semantics require additional contracts.
- One bridge per game and algorithm: rejected because reuse is the primary goal.

