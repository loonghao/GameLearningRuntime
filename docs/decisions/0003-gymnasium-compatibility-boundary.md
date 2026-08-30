# ADR-0003: Adapt Gymnasium at the outward boundary

## Status

Accepted

## Context

Several authorized game projects already expose the Gymnasium five-value
`reset`/`step` lifecycle, Box or Dict observations, Discrete actions, and an
optional action-mask method. Each project then repeats conversion code for
episode identity, tensor shapes, TorchRL, and replay metadata.

Gymnasium is useful as a compatibility surface, but it cannot replace the GLR
domain contract. GLR also requires monotonic step identity, structured masks,
semantic events, transport schemas, and explicit metadata policy.

## Decision

Provide Gymnasium as an optional outward integration. `GymnasiumEnvironment`
converts declared spaces and lifecycle results into `EnvironmentSpec` and
`TimeStep`, after which the existing `ContractEnvironment`, collectors,
recorders, and TorchRL adapter are reused.

The integration requires an explicit public `environment_id`. It never derives
metadata from module names, file locations, runtime processes, or the host.
Gymnasium `info` is discarded by default; a caller must provide an explicit
`info_transform` to export a stable allowlisted view.

Action-mask discovery is explicit rather than reflective. The first contract
supports a root Discrete action space because that is the common, unambiguous
mask representation in the current projects. Structured mask conventions will
be added only with conformance fixtures.

## Consequences

### Positive

- Existing Gymnasium environments gain GLR collectors and TorchRL without a
  project-specific TorchRL `EnvBase` implementation.
- Episode IDs, step IDs, shapes, dtypes, and masks are validated once.
- Incidental local metadata is denied by default.

### Negative

- Game-specific observation encoding and reward semantics remain in each game
  adapter.
- Structured and parameterized action-mask conventions need later schema work.
- Gymnasium remains an optional dependency and has a dedicated CI lane.

### Neutral

- Direct `GameEnvironment` implementations remain the preferred boundary for
  native transports and non-Gym runtimes.

## Alternatives considered

- Make Gymnasium the core contract: rejected because it does not express all GLR
  lifecycle, event, transport, and metadata requirements.
- Keep one TorchRL adapter per game: rejected because the lifecycle and tensor
  conversion code is duplicated.
- Reflectively export every Gymnasium `info` field: rejected because fields are
  not stable and may contain local or sensitive runtime data.
