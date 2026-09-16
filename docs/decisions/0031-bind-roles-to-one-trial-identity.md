# ADR-0031: Bind every training role to one trial identity

Status: Accepted with the trial-identity implementation

Related: issue #144, ADR-0017, ADR-0021, ADR-0024, the `glr.agent-goal.v1` and
`glr.trial-plan.v1` contracts.

## Decision

A role invocation always describes the trial it serves. `glr train` drives
exactly one implicit trial, so it publishes the same `trial_id` and `trial_path`
context keys the goal loop already publishes for its planner and trainer:
`GLR_TRIAL_ID` and `GLR_TRIAL_PATH`. The implicit trial uses the goal loop's
identifier and layout — `trial-1` under `trials/trial-1/`, with the plan at
`trials/trial-1/plan.json`.

`GLR_RUN_DIR` keeps its existing meaning: the run-scoped output root
`.glr/runs/<run-id>/`. A trial directory is a child of that root, never its
replacement, so a role that writes run evidence is unaffected by which command
started it.

Roles that own no trial receive no trial identity. `runtime start` and `play`
drive no trial, so they publish neither key.

## Why the role environment is now owned by the CLI

Publishing a trial identity is only meaningful if the CLI decides it. Before
this ADR, `Command` inherited the parent environment, and only
`GLR_RUN_CONTEXT`/`GLR_RUN_CONTEXT_SHA256` were explicitly removed. Any other
inherited `GLR_*` variable therefore outlived the value the CLI published, or
survived where the CLI published nothing. A trainer with no trial could observe a
forged `GLR_TRIAL_ID`, and a nested or resumed invocation could observe a stale
`GLR_RUN_ID` from an outer run.

The CLI now clears every inherited `GLR_*` variable before publishing the values
an invocation actually owns. The namespace is CLI-owned: a role's GLR
environment states what the CLI decided, not what the ambient environment
happened to contain. Parent-environment values that were never scrubbed are no
longer an input to role behavior.

## Consequences

A trainer written against the trial contract runs under `glr train` and under
`glr goal run` without a second convention. `{trial_id}` and `{trial_path}` are
expandable placeholders in `glr-project.json` for both entrypoints; previously
they were valid placeholders that `glr train` could not expand.

`glr train` does not plan. Writing a plan is the planner role's job in the goal
loop, and a standalone run has no planner. The CLI therefore creates the trial
directory and reserves the plan path, but never fabricates plan content: the
trainer receives a location, not an invented `glr.trial-plan.v1` document. A
trainer that requires a real plan must run under `glr goal run`.

Issuing a trial identity is not evidence that a trial succeeded, that a
checkpoint is promotable, or that a goal was met. Those remain separate
authoritative verdicts owned by the existing evaluation and promotion contracts.

Clearing the `GLR_*` namespace is a behavior change for any deployment that
relied on injecting its own `GLR_*` variables into a role. Such a variable must
now be declared through the project manifest context (explicit `extra` keys,
placeholders, or a run context) instead of the ambient environment. Variables
outside the `GLR_` prefix, including engine- and adapter-owned configuration, are
untouched.
