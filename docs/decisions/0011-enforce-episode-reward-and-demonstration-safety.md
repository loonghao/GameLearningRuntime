# ADR-0011: Enforce episode reward and demonstration safety

## Status

Accepted

## Context

Per-term and per-step reward clipping does not protect the actual learning
objective. A long episode can accumulate many small positive shaping rewards
and remain profitable after an authoritative terminal failure. Behavioral
cloning has a separate contamination risk: learner-selected actions can be
recorded and later relabelled as expert demonstrations, reinforcing the current
policy regardless of outcome.

Both failures produce plausible local behavior while weakening the terminal
objective. They must be rejected at the framework boundary, with reproducible
configuration and auditable evidence, rather than left to learner-specific
conventions.

## Decision

Add two strict, data-only contracts:

- `glr.reward-safety.v1` names one authoritative terminal outcome, identifies
  shaping terms, caps positive shaping per step and episode, requires terminal
  outcome evidence, and defines the maximum total return retained by a failed
  episode.
- `glr.demonstration-policy.v1` allowlists demonstration origins and episode
  outcomes, assigns deterministic sample weights, and fails closed on unknown
  provenance. The generated default rejects policy-originated and failed data.

`EpisodeRewardGuard` runs after `RewardComposer`. It proportionally suppresses
positive shaping when a budget is exhausted, preserves negative shaping, and
adds an auditable terminal correction when a failed episode would otherwise
exceed its configured ceiling. It closes at terminal and requires an explicit
reset before another episode.

`DemonstrationGate` runs before BC dataset ingestion. Every sample or trajectory
must carry immutable origin and authoritative episode outcome. A policy ID is
required for policy-originated data and forbidden for other origins, preventing
silent provenance relabelling. Policy data may enter only a separately reviewed
policy that explicitly allows it; it is never an implicit expert source.

The adapter scaffold emits both policies, validates them in tests and the smoke
trainer, and includes them in `glr.model-bundle.v1` reproduction inputs.

## Non-functional requirements

- **Correctness:** terminal failure dominates accumulated positive shaping;
  outcome signals are terminal-only and use a positive reward weight.
- **Security:** schemas reject executable or unknown fields and unknown
  provenance fails closed.
- **Auditability:** guarded results expose accepted shaping, suppressed shaping,
  contribution breakdown, and terminal correction.
- **Reproducibility:** reward budgets, provenance allowlists, and sample weights
  ship with the model bundle.
- **Learner neutrality:** reward and dataset gates do not import PPO, IMPALA,
  BC, TorchRL, or a model implementation.

## Failure modes and mitigation

- **Dense reward farming:** per-step and cumulative positive shaping budgets
  stop unbounded accumulation.
- **Positive failed return:** terminal correction caps the failed episode at the
  configured ceiling.
- **Missing result telemetry:** terminal composition fails before emitting a
  reward.
- **Policy self-imitation:** default demonstration policy rejects `policy` and
  `unknown` origins.
- **Failed demonstrations treated as expert:** default policy accepts only
  authoritative `success` outcomes.
- **Hidden weighting drift:** origin and outcome weights are versioned JSON and
  included in reproduction inputs.

## Consequences

### Positive

- Reward shaping cannot silently replace the terminal objective.
- BC datasets have an explicit, testable provenance boundary.
- Agents receive safe defaults and executable negative tests in every scaffold.

### Negative

- Adapters must provide authoritative terminal outcomes.
- Existing datasets need provenance and outcome migration before strict BC
  ingestion.
- A terminal correction indicates design pressure and must be monitored; it is
  not proof that the shaping terms are useful.

### Neutral

- Deliberate distillation from policy data remains possible through a separate
  explicit policy.
- The framework does not decide which game-specific signals constitute success.

## Alternatives considered

**Only clip each step.** Rejected because a long failed episode can still
accumulate a large positive return.

**Increase terminal reward magnitude.** Rejected because it is brittle across
episode lengths and hides the missing invariant.

**Filter demonstrations inside each trainer.** Rejected because different
learners would implement inconsistent or unverifiable provenance rules.

**Treat every completed episode as expert data.** Rejected because completion
does not imply success or expert authorship.

## References

- ADR-0005, ADR-0008, and ADR-0010
- [Configure knowledge sources and rewards](../guides/knowledge-and-rewards.md)
- [Training safety guide](../guides/training-safety.md)
