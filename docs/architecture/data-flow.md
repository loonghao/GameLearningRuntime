# Protocol and data flow

## Environment handshake

1. A client calls `Describe` and obtains the environment ID, protocol version,
   tensor specs, masks, capabilities, and metadata.
2. The client decides whether it supports that exact contract.
3. `Reset` starts an episode and returns `step_id = 0`.
4. Every action carries the current `episode_id` and expected next step ID.
5. A time step returns observation, reward, terminated/truncated tensors,
   masks, events, and metadata.

```text
Client                         Runtime adapter
  │──── Describe ────────────────────▶│
  │◀─── EnvironmentDescriptor ────────│
  │──── Reset(seed/options) ─────────▶│
  │◀─── TimeStep(episode, step=0) ────│
  │──── Step(episode, expected=1) ───▶│
  │◀─── TimeStep(step=1) ─────────────│
```

`Interact` provides the same ordered step contract over a bidirectional stream
for future high-throughput actor adapters. It does not weaken identity or
ordering rules.

## Learning paths

- PPO consumes fixed-length `Unroll` values and computes returns/advantages.
- IMPALA actors attach `actor_id`, `sequence_id`, and `policy_version`; a future
  distributed queue transports those unrolls to a V-trace learner.
- BC writes expert actions as ordinary transitions, preserving masks and next
  observations for later DAgger or offline RL.
- TorchRL maps GLR trees to `TensorDict` only at the integration boundary.

## Compatibility

The Protobuf package is `glr.v1` and the JSONL record schema is
`glr.transition.v1`. Additive fields may remain within v1. Removing fields,
changing meaning, or changing tensor encoding requires a new major schema.

