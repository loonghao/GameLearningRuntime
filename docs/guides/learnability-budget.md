# Size the state-action space against the step budget

A run that cannot converge still reports healthy: steps were taken, the process
exited zero, and the report reads green. The expensive part is not the failure,
it is that the failure is **invisible** — an operator cannot tell "the
configuration is wrong" from "I have not run enough steps yet" and keeps tuning
the wrong variable.

The `learnability-budget-v1` capability puts the declared scale next to the
measured coverage so the question has an answer:

- **`state_action_cells`** — declared before the run. How big the space is that
  the learner would have to cover.
- **`coverage_ratio`** — measured during the run. How much of that space the run
  actually reached, projected forward against the observed step rate and the
  configured budget.

and one derived number that makes them comparable:

- **`projected_steps_to_k_visits`** — how many steps this configuration needs
  before every cell has been seen `K` times. `K` defaults to `4`, because below
  roughly four visits a tabular value is a sample, not an estimate.

Everything is opt-in and versioned by `glr.learnability-budget.v1`. An adapter
that declares nothing behaves exactly as it did before.

## Declare the cardinality

An adapter declares how it discretizes its state, or declares an upper bound,
and the runtime multiplies the state bound by the action-set size:

```python
from game_learning_runtime.learnability import (
    LEARNABILITY_BUDGET_SCHEMA_VERSION,
    LEARNABILITY_CAPABILITY,
    CardinalityKind,
    LearnabilityDeclaration,
    SpaceCardinality,
)
from game_learning_runtime.specs import EnvironmentSpec

spec = EnvironmentSpec(
    ...,
    capabilities=frozenset({LEARNABILITY_CAPABILITY}),
    learnability=LearnabilityDeclaration(
        schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
        kind=CardinalityKind.TABULAR,
        state=SpaceCardinality(cells=1_000, bins={"cell": 1_000}),
        action=SpaceCardinality(cells=4),
    ),
)
```

`state_action_cells` is then `1_000 * 4 == 4_000`.

`bins` is the discretization itself: a mapping of flattened observation leaf
path to the number of bins that leaf is encoded into, and `cells` must equal the
product of the bin counts. Supplying it lets the runtime resolve a cell identity
from an observation on its own. Omitting it declares a bound only, and cell
identity has to arrive through `TimeStep.info`:

```python
info = {"learnability_cell": current_cell}
```

The value may be any integer or a string. A `numpy` integer counts: an
observation is a `numpy` array, so `observation[0]` is the natural thing to
report, and `numpy` integers are not Python `int`. A value that is neither an
integer nor a string is not a cell identity, so the step is charged as
unresolved.

Either way a step that resolves to no cell is still charged against the budget:
an untracked step is not free, and the report says so in `notes`.

### Function approximation

A learner that does not tabulate the state has no cell product. It declares the
capacity that plays the same role and that number is used **verbatim** instead
of the product:

```python
LearnabilityDeclaration(
    schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
    kind=CardinalityKind.FUNCTION_APPROXIMATION,
    effective_capacity=512,
)
```

### Derive it from the contract you already declared

If the observation and action specs are already discrete and bounded, the bound
is implied and can be lifted instead of restated. A continuous or unbounded leaf
makes the product unknown, which is the honest answer rather than a guess:

```python
from game_learning_runtime.learnability import (
    derive_action_cardinality,
    derive_state_cardinality,
)

state = derive_state_cardinality(observation_spec)  # SpaceCardinality | None
action = derive_action_cardinality(action_spec)  # SpaceCardinality | None
```

## Opt in with a plan

Declaration alone is inert: nothing is measured and nothing is enforced until a
caller supplies a plan.

```python
from game_learning_runtime.collector import SyncCollector
from game_learning_runtime.learnability import LearnabilityBudget, LearnabilityPlan

plan = LearnabilityPlan(
    budget=LearnabilityBudget(
        budget_steps=100_000,
        budget_seconds=600.0,  # optional; turns remedy 3 into a steps/s floor
        min_coverage=None,  # warn only
        visit_target=4,  # K
    )
)

collector = SyncCollector(environment, learnability=plan)
collector.collect(policy, steps=100_000)
report = collector.learnability_report()
```

With `min_coverage=None` the run never stops early. It still reports a
`WARNING` status and a full remedy list, so an unattended run leaves the
evidence behind without being killed for it.

## Fail fast, before the budget is gone

Set `min_coverage` and the collector calls `require()` after every recorded
step. As soon as the projection says the remaining budget cannot reach the
floor, it raises:

```
learnability budget exhausted: visited 8 of 1000 state-action cells (coverage 0.008 < 0.500) after 8 of 100 budgeted steps
projected 4,000 steps to reach 4 visits per cell; observed throughput: unknown
remedies:
  1. shrink the declared state-action space from 1,000 to at most 200 cells (coarser state bins, fewer state features, or a shorter horizon); 1,000 cells needs about 4,000 steps for 4 visits per cell
  2. replace the tabular encoding with a function approximator and declare its effective capacity on the learnability-budget-v1 declaration instead of the 1,000-cell product; a table over 1,000 cells stays a sample collection below 4 visits per cell
  3. raise throughput from unknown to at least 50.0 steps/s (4,000 steps inside the 80s budget)
```

Three properties matter here:

- **It stops early.** The example raises at step 8 of a 100-step budget, so 92%
  of the budget is still unspent. Waiting until the budget is gone would make
  the check worthless.
- **It is a distinct error type.** `LearnabilityBudgetError` subclasses
  `GLRError` and is deliberately *not* a `ContractViolation` and *not* a
  transport error: nothing was violated on the wire and nothing is retryable.
  `error.report` carries the full verdict and `error.remedies` the three
  numbered paths.
- **The remedies carry numbers.** "Try a smaller state space" is not actionable.
  "Shrink to at most 200 cells", "at least 50.0 steps/s", and "4,000 steps"
  are.

The verdict needs evidence before it is trusted: nothing raises in the first
`MIN_EVIDENCE_STEPS` steps, because a few cold-start steps cannot distinguish an
unlearnable configuration from a slow start.

## Read all three numbers without parsing logs

Three surfaces carry the same `glr.learnability-budget.v1` payload:

- `SyncCollector.learnability_report()` during a run;
- the run store, as a `learnability.budget` event plus three first-class
  metrics — `learnability.coverage_ratio`,
  `learnability.projected_steps_to_k_visits` and
  `learnability.state_action_cells` (`TrainingStore.record_learnability` /
  `list_learnability`). The cell count is recorded as a metric because it is
  the scale the other two are fractions of: the same coverage ratio on 10
  cells and on 10,000 is not the same result, and a ratio alone cannot say
  which run this was;
- `glr.cli-output.v1` from `glr --project . --json runs show <run-id>`, which
  adds `learnability` and a `learnability_summary` next to the run record.

```json
"learnability_summary": {
  "schema_version": "glr.learnability-budget.v1",
  "reported": true,
  "status": "failed",
  "state_action_cells": 1000,
  "coverage_ratio": 0.095,
  "projected_steps_to_k_visits": 4211,
  "steps_per_second": 2.5,
  "budget_steps": 100,
  "min_coverage": 0.5
}
```

When no verdict was recorded, `reported` is `false` and every field is `null`.
An absent verdict is never silently rendered as a passing one.

## Enforce it from the CLI

`glr train --min-coverage FRACTION` carries the floor into the run. The trainer
owns collection, so the floor travels as configuration — the run metadata
records `learnability_min_coverage` and the trainer process receives a
`min_coverage` value — rather than as an assertion the host could not check on
its own. After the run, a zero exit code plus a failed verdict is downgraded to
a failed run: a trainer that measured coverage, found it short, and exited zero
must not read green.

`--min-coverage` must be a finite fraction in `(0, 1]`; anything else is a
`ContractViolation` at parse time.

## What is deliberately not checked

- **No automatic state encoding.** GLR never discretizes an adapter's state for
  it. The adapter declares the discretization, declares a bound, or reports cell
  identities in `info`.
- **No claim that coverage implies learning.** `coverage_ratio` says the run
  *reached* the space. Whether the learner used those visits well is a learning
  question, reported elsewhere as `learning_status`.
- **No enforcement without opt-in.** An undeclared adapter, or a caller that
  supplies no plan, gets no tracker, no metrics, and no verdict.
