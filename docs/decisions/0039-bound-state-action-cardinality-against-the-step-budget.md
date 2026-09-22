# ADR-0039: Bound state-action cardinality against the step budget

## Status

Accepted

## Context

A run spends its whole budget on a configuration that cannot converge, and every
report still reads healthy: steps were taken, the process exited zero, and the
metrics moved. Nothing in the tree knows how big the space was, so nothing can
say the budget was never going to cover it.

The cost is not the failed run. It is that the failure is invisible, so an
operator cannot distinguish "this configuration is unlearnable" from "I have not
run enough steps yet" and keeps tuning the wrong variable — usually by buying
more steps for a space that needed fewer cells.

Both halves of the comparison already exist somewhere in the tree and are never
put next to each other:

- the adapter knows how it discretizes its state, and the action spec knows the
  action-set size, so `state_action_cells` is knowable at declaration time;
- the collector already counts steps and the run store already times them, so
  the observed step rate is knowable during the run.

What is missing is the division, and an authority to act on it.

## Decision

Add `game_learning_runtime.learnability` behind the optional
`learnability-budget-v1` capability, versioned by `glr.learnability-budget.v1`.

- **Declaration is a spec slot, not a new interface.** `EnvironmentSpec` gains
  an optional `learnability: LearnabilityDeclaration`. `state_action_cells` is
  `state.cells * action.cells` for a tabular configuration, or a declared
  `effective_capacity` used verbatim for a function approximator. Declaration
  alone is inert: no plan means no tracker, no metrics, no verdict.
- **`bins` is the discretization, and the runtime validates it.** A
  `SpaceCardinality` may carry a mapping of flattened observation leaf path to
  bin count, and `cells` must equal the product of those counts. When it does,
  a `StateCellResolver` derives the cell identity from the observation in mixed
  radix. When it does not, the adapter reports identity through
  `info["learnability_cell"]`, which may hold any integer — a `numpy` one
  included, because an observation is a `numpy` array and `observation[0]` is
  the natural thing to report — or a string.
- **Coverage is measured, and projected.** `coverage_ratio` is
  `distinct_cells_visited / state_action_cells`. Discovery efficiency is
  `distinct / min(steps, cells)`, and `projected_steps_to_k_visits` is
  `ceil(K * cells / efficiency)` with `K` defaulting to `4` — below roughly four
  visits a tabular value is a sample, not an estimate. The same efficiency
  yields `projected_coverage_at_budget`, which is what the gate reads.
- **The gate is coverage, not the projection.** Failing on
  `projected_steps_to_k_visits` would make the check a restatement of "the
  budget is too small", and every healthy tabular run would trip it. The gate is
  `projected_coverage_at_budget >= min_coverage`, with `min_coverage` defaulting
  to `0.5` when unconfigured: below half the declared space, the untouched half
  is initialization rather than learning.
- **Fail fast, with evidence.** `LearnabilityTracker.require()` raises as soon
  as the projection says the remaining budget cannot close the gap, but not
  before `MIN_EVIDENCE_STEPS` (8) steps, because a cold start cannot be told
  apart from an unlearnable configuration. In the 1,000-cell / 100-step
  fixture it raises at step 8, leaving 92% of the budget unspent.
- **A gate abort still settles the episode.** The gate raises mid-collection,
  so the episode it stopped would otherwise end without a terminal state and
  without a declared-metric audit. It is closed with
  `TerminationReason.FAILED` on the way out, like an environment error: no
  caller may observe an episode that neither recorded a reason nor raised a
  violation.
- **A distinct error type.** `LearnabilityBudgetError` subclasses `GLRError` and
  deliberately is *not* a `ContractViolation` and *not* a transport error:
  nothing was violated on the wire and nothing is retryable. Its message names
  all three remediation paths with the numbers that apply — shrink the space to
  at most N cells, switch to function approximation and declare an effective
  capacity, or raise throughput to M steps/s.
- **Both numbers are first-class.** The verdict is persisted as a
  `learnability.budget` event plus two metrics
  (`learnability.coverage_ratio`, `learnability.projected_steps_to_k_visits`)
  and projected into `glr.cli-output.v1` from `runs show` as `learnability` and
  `learnability_summary`. An absent verdict renders as `reported: false` with
  null fields, never as a passing one. The verdict window is bounded by
  verdicts, not by the events around them: a run with a long telemetry stream
  still yields its latest verdict, so a failed verdict cannot be crowded out
  of the window and read as a green run.
- **`glr train --min-coverage FRACTION`** carries the floor into the run. The
  trainer owns collection, so the floor travels as configuration (run metadata
  `learnability_min_coverage`, plus a `min_coverage` value for the trainer
  process) rather than as an assertion the host could not check on its own. A
  zero exit code plus a failed verdict is downgraded to a failed run.

## Consequences

- An unlearnable configuration is named before the budget is spent, and the
  message says which of the three variables to move and by how much.
- Both numbers are readable from the run store and the CLI JSON without parsing
  a log, so a scheduler can gate on them.
- An adapter that declares nothing behaves exactly as it did before this ADR.
- A caller can warn without stopping: `min_coverage=None` still produces the
  full verdict, the status, and the remedy list.
- `coverage_ratio` measures reach, not learning. A run can cover the space and
  still learn nothing; that question stays in `learning_status`.
- A declared `effective_capacity` is trusted verbatim, so the number is only as
  honest as the adapter declaring it. That is the deliberate trade for letting
  function approximation participate at all.

## Rejected alternatives

- **Compute the cardinality from the observation spec automatically:** rejected
  because most real adapters bin a continuous state in a way the contract does
  not express, and an encoding GLR invents is not the encoding the learner
  tabulates. `derive_state_cardinality` exists for the bounded discrete case and
  returns `None` when it cannot answer honestly.
- **Gate on `projected_steps_to_k_visits <= budget_steps`:** rejected because a
  tabular run that has converged still needs more steps than the budget to reach
  four visits everywhere, so the check would fail healthy runs. It stays as an
  informational projection.
- **Raise `ContractViolation`:** rejected because nothing was violated on the
  wire, and folding this into contract failures would make callers retry a
  configuration error.
- **Warn only, never fail:** rejected because an unattended run that cannot
  converge should stop spending its budget, not file a note for later.
- **Silently skip steps whose cell cannot be resolved:** rejected because an
  untracked step is not free; unresolved steps are charged against the budget
  and reported in `unresolved_steps`.
- **Default `min_coverage` to a hard failure:** rejected because it would break
  adapters that declare a loose upper bound. The default stays a warning;
  `--min-coverage` is what promotes it to a failure.

## Related

- ADR-0002 established the versioned, strict-schema contract this module
  follows (`from_mapping` / `to_mapping`, fail closed on unknown fields).
- ADR-0018 built on the run-store metric, namespaced-event, and metadata
  surfaces the verdict is persisted to.
- [Learnability budget guide](../guides/learnability-budget.md).
