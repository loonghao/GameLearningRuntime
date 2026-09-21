# ADR-0034: Bind a default goal per project

## Status

Accepted

## Context

`glr goal run` requires `--goal` on every invocation, and `glr train` requires
`--context` whenever the frozen training and reward inputs matter. An agent
therefore has to repeat the same two paths in every script, and the pair that
makes a goal executable — the goal plus the run context that freezes its inputs
(ADR-0024) — is never recorded anywhere. Each invocation restates it, so a
drifted or deleted goal file is only discovered at the moment training starts.

Projects also asked for a goal template library and for goals to shape rewards
or judge completion. Neither is in scope here: a goal is structured metadata
plus an auditable receipt persisted with the run, and reward shaping and
completion judgement stay with the project planner and evaluator.

## Decision

Add a persisted per-project goal binding in `<data_dir>/goal-binding.json`
(`glr.goal-binding.v1`). Each entry records the goal ID, objective, environment
family, a project-relative path to the `glr.agent-goal.v1` file, its SHA-256,
and an optional project-relative path to a `glr.run-context.v1` file. One
`active_goal_id` pointer names the default; a project can save at most 64 goals.

New subcommands manage the store: `glr goal set --goal <path>` (with an optional
global `--context`) validates and binds a goal, `glr goal list` enumerates saved
goals, `glr goal show [goal-id]` inspects one, and `glr goal use <goal-id>` moves
the active pointer.

Resolution is explicit-first. `goal run` uses `--goal` when given and otherwise
falls back to the active binding; `train` uses `--context` when given and
otherwise inherits the context bound to the active goal. An explicit `--goal` or
`--context` always wins, so existing invocations keep their behaviour.

Binding is validated eagerly. `goal set` rejects a goal outside the project
root, an unusable goal, and a goal whose `environment_family` does not match the
project, so an unusable default is reported at bind time instead of at training
start. Stored paths are re-checked when the store is loaded and again when a
default goal is resolved: a store edited by hand into an absolute path, a parent
directory, or a link is refused instead of opened. `goal show` reports
`source_status` (`unchanged`, `changed`, `missing`) and `context_status`
(`unbound`, `bound`, `unresolved`), and a deleted default goal is reported as a
gap that asks for a rebind rather than as a silent failure.

`doctor` reports the active goal under `goal_binding`. Every `goal run` receipt
records `goal_binding` with `source` set to `default` or `explicit`, and the
goal file is copied into the run directory as an auditable receipt. The receipt
adds `context_source` (`explicit`, `default`, or `none`) next to `source`,
because an explicit `--goal` can still inherit the context bound to the active
goal, and adds the `source_status` of the bound goal, so a run records whether
its goal still matched the digest captured at bind time. Both fields are
additive: older receipts keep their `source`, `goal_id`, `goal_path`, and
`context_path` keys.

## Consequences

- One binding removes the repeated `--goal` and `--context` arguments from
  agent-written scripts.
- Stored paths are project-relative, so a portable project (ADR-0021) keeps its
  default goal when the project moves.
- A SHA-256 of the goal file makes a deleted default a hard stop at `goal run`
  and makes a drifted default visible: `goal show` reports it, and every
  `goal run` receipt records the status of the goal it actually ran. Drift is
  therefore visible when a run is inspected, not enforced as a gate before
  training starts.
- Revalidating stored paths on load turns a hand-edited store into an error
  every command reports, rather than a silent read outside the project root.
- A binding only selects a goal and a context. It never shapes rewards and never
  judges completion; those remain planner and evaluator responsibilities.

## Rejected alternatives

- Ship a goal template library in core: rejected as project and product policy,
  and orthogonal to default resolution.
- Let a goal contribute reward shaping or automatic achievement judgement:
  rejected because it would move reward policy out of the project planner and
  evaluator and into the control plane.
- Resolve the default from the most recently used goal path in the run store:
  rejected because run history is evidence, not configuration, and it cannot
  express an explicit choice.
- Store absolute paths: rejected because it breaks portable projects.
