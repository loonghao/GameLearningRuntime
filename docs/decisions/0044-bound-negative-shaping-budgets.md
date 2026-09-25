# ADR-0044: Bound negative shaping with optional budgets

## Status

Accepted

## Context

ADR-0011 caps positive shaping per step and per episode and deliberately
preserves negative shaping, so a terminal failure cannot be out-earned by dense
positive rewards. The symmetric hole is real: negative shaping was left
completely unbounded.

A positive-only budget caps the upside of shaping, not its downside. When a
negative shaping term keeps charging as the episode grows, the cumulative return
peaks *before* the episode ends. From that point on every further step destroys
value, so the optimal policy becomes dying early — the exact opposite of what a
terminal-dominance reward is for. Evidence from a live adapter (7 episodes,
mean regret about -9.7) shows the peak arriving before the end in every
episode, so the failure is reachable in practice and not theoretical.

The contract could not express a fix either. `RewardSafetyConfig.from_mapping`
rejects unknown fields, so `max_negative_shaping_per_episode` was a hard
`ValueError` and adapters had no way to protect themselves.

## Decision

Extend `glr.reward-safety.v1` in place with two optional fields:

- `max_negative_shaping_per_step`
- `max_negative_shaping_per_episode`

Both bound the **magnitude** of negative shaping and may be omitted. Omitting
them leaves negative shaping unbounded, which is exactly the behaviour of every
existing policy, so no configuration changes meaning and no schema version bump
is required: the fields are additive and every previously valid document stays
valid with identical semantics. `max_negative_shaping_per_step` cannot exceed
`max_negative_shaping_per_episode`, mirroring the positive pair.

`EpisodeRewardGuard` accumulates negative shaping on every step and applies the
same accept-and-scale rule used for positive shaping: the accepted magnitude is
the smaller of the observed magnitude, the per-step budget, and the remaining
per-episode budget, and every negative shaping contribution is scaled by the
accepted fraction. Results report `negative_shaping_total` and
`suppressed_negative_shaping` alongside the positive counters, so a budget that
fires constantly is visible before it silently reshapes the objective.

Because the default stays uncapped, the guard also names the gap. At construction
time, when no `max_negative_shaping_per_episode` is declared and a shaping term
can contribute a negative value, it logs one warning listing those terms. A term
"can contribute a negative value" when `clip(signal, minimum, maximum) * weight`
is negative for some finite signal: a zero weight never can, and the term bounds
decide the rest.

## Non-functional requirements

- **Correctness:** a bounded negative budget keeps the marginal value of
  surviving non-negative on the reproduced episode set; unbounded shaping does
  not.
- **Compatibility:** an existing `glr.reward-safety.v1` document keeps its exact
  behaviour, and `GuardedRewardResult` gains only optional counters.
- **Auditability:** accepted and suppressed negative shaping are reported per
  step, and a missing episode budget is named in a log record.
- **Learner neutrality:** the budget is declarative data applied after
  `RewardComposer`, with no learner-specific code.

## Failure modes and mitigation

- **Suicide-is-optimal shaping:** per-step and per-episode negative budgets stop
  unbounded accumulation in both directions.
- **Silent objective drift:** suppressed shaping is reported, and a missing
  episode budget is logged rather than assumed harmless.
- **Ambiguous configuration:** a step budget above the episode budget is
  rejected, and negative or non-numeric budgets are rejected.

## Consequences

### Positive

- Shaping can no longer make an early death the optimal policy.
- Adapters can bound the downside without waiting for a default change.
- The missing budget is observable before default values change.

### Negative

- A negative budget suppresses real evidence once exhausted, so a cap that is
  too tight hides genuine failure signal; measure before tightening.
- Two more counters and one log record per guard construction.

### Neutral

- Whether the default should become a real bound stays a product decision. The
  mechanism ships first so that decision has measured before/after data.

## Alternatives considered

**Per-term `episode_maximum`.** Most expressive, but it multiplies the schema
surface and does not bound the aggregate a policy actually feels.

**Versioned extensible schema plus a warning only.** Makes the hole visible
without closing it, so the demonstrated harm stays reachable.

**Tighten the default immediately.** Rescales every existing reward silently.
Rejected as a product call that this ADR deliberately defers.

## References

- ADR-0008, ADR-0011, and ADR-0043
- [Training safety guide](../guides/training-safety.md)
- Upstream issues `loonghao/GameLearningRuntime#160` and `#161`
