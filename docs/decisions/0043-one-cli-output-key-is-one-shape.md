# ADR-0043: One `glr.cli-output.v1` key is one shape

## Status

Accepted

## Context

`glr.cli-output.v1` is read by agents and schedulers that do not know which verb
produced the payload they were handed. The contract already projects the run
store twice for exactly that reason: `declared_metrics` and `terminations` carry
the full history, and `declared_metrics_summary` and `termination_summary` carry
the newest entry so a caller can gate without walking the list.

The learnability verdict broke that symmetry. The key was named `learnability`
in two verbs and given two shapes:

- `glr --project . --json train` published the newest verdict there, as an
  object — the same projection the other verbs call a summary.
- `glr --project . --json runs show <run-id>` published the recorded history
  there, as an array.

Nothing crashed, and both shapes were pinned by tests. That is what made it
expensive: a caller that read the object as a history read `status` as a list
index, and a caller that read the history as an object got `0` where it expected
a fraction. The only way to read the key correctly was to first ask which verb
produced the payload, which is the question the projection exists to remove.

The drift was also structural, not accidental. Each verb added the key when it
needed it, and each test read the one verb it was written against, so no test
could see that the two disagreed.

## Decision

**One name in `glr.cli-output.v1` means one shape, in every verb that publishes
it.**

- The plural key is the history: an array of the recorded entries, oldest first,
  empty when nothing was recorded.
- The `<name>_summary` key is the newest entry, or an explicit `reported=false`
  with null fields when the history is empty. It is never omitted.
- `train` now publishes both `learnability` and `learnability_summary`, matching
  `runs show` key for key and shape for shape. The object it used to publish
  under `learnability` is still readable, under the name that says it is a
  summary.

This restates the convention `declared_metrics` / `declared_metrics_summary` and
`terminations` / `termination_summary` already follow. It does not invent a
third shape.

## Consequences

- A caller reads `learnability` as an array and `learnability_summary` as an
  object without knowing the verb. Shape parity between the two verbs is pinned
  by one test that runs both in a single run, because a test that reads only one
  verb is how the shapes drifted apart in the first place.
- A caller that read `train`'s `learnability.coverage_ratio` must read
  `learnability_summary.coverage_ratio` instead. The number is unchanged; only
  the name that carries it moved. Nothing in this repository consumed it — the
  dashboard, the `tools/` scripts, and the packaged skill reference read no
  learnability field — so the migration is a key rename with no in-tree caller.
- No verdict is silently rendered as a passing one: an empty history stays an
  empty array plus `reported=false`, as before.

## Rejected alternatives

- Keep both shapes and rename one key (`learnability_history`): rejected because
  it leaves two names for one concept. The contract already has a name for "the
  newest entry" — the `_summary` suffix — so a second vocabulary would make the
  next projection harder to name, not easier.
- Have `train` publish only the summary: rejected because a run can record more
  than one verdict, and the verb that created them should not be the one verb
  that cannot list them.
- Unify on the object and drop the list: rejected because `runs show` exists to
  return a history, and flattening it to the newest entry would throw away the
  earlier verdicts for every caller that reads the list today.
- Add a `shape` discriminator field: rejected because it moves the problem from
  the key to the value. A caller would still have to branch, and it would now
  have to branch on every read instead of once per upgrade.

## Related

- [ADR-0039: Bound state-action cardinality against the step budget](0039-bound-state-action-cardinality-against-the-step-budget.md)
- [ADR-0040: Fail closed on a declared metric that is never emitted](0040-fail-closed-on-a-declared-metric-that-is-never-emitted.md)
- [Learnability budget guide](../guides/learnability-budget.md)
