# ADR-0006: Distinguish live attach from reset

## Status

Accepted

## Context

Some authorized real-time games expose a running world and safe structured
actions but cannot restore a deterministic checkpoint. Requiring those
adapters to implement `reset()` encourages them to label an observation-only
attachment, menu reload, or ordinary respawn as a physical environment reset.
That makes episode boundaries non-reproducible and corrupts training evidence.

The learning-side contract still needs a step-zero observation and a fresh
logical episode identity before it can validate actions and collect
transitions.

## Decision

Add an explicit, capability-gated `GameEnvironment.attach()` lifecycle.
Adapters must advertise `live-attach` before the contract wrapper will call it.
Attach starts a new logical GLR episode at step zero around the already-running
world; it does not claim that game state was restored, randomized, or reset.

`ContractEnvironment` applies the same tensor, terminal-state, fresh episode
identity, and step-zero checks to reset and attach. `SyncCollector` and the
TorchRL adapter accept an explicit `start_mode` whose default remains
`"reset"`. Attach mode rejects seeds because it cannot promise seeded world
initialization. The Gymnasium compatibility adapter advertises the capability
only when given an explicit `attach_provider`; it never infers attachment from
`reset()` or from an incidental method name. The v1 protocol exposes a distinct
additive `Attach` RPC.

## Consequences

### Positive

- Continuing games can use GLR without falsifying deterministic reset evidence.
- Learners and collectors select lifecycle semantics explicitly.
- Existing reset-based adapters and callers retain their behavior by default.
- Remote adapters can negotiate attach through the existing capability list.

### Negative

- Consumers that require reproducible episodes must reject attach mode.
- An attached logical episode may begin from an arbitrary world state.
- Game-specific adapters remain responsible for death, respawn, disconnect,
  and world-transition semantics.

### Neutral

- Attach is not partial reset and does not satisfy a deterministic-reset gate.

## Alternatives considered

- Treat attach as reset: rejected because it makes an unverifiable lifecycle
  claim.
- Make reset optional: rejected because existing collectors need a stable
  default and capability discovery would become ambiguous.
- Put attach only in game-specific adapters: rejected because collectors,
  TorchRL, and remote transports need one shared lifecycle meaning.
