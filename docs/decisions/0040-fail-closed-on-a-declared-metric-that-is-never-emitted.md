# ADR-0040: Fail closed on a declared metric that is never emitted

## Status

Accepted

## Context

An adapter can declare the metrics it promises to emit, and a run can still
finish with one of them missing. Nothing in the tree noticed. The metric simply
was not in the run's metric table, and a missing row is indistinguishable from a
real zero: a dashboard renders `0`, a comparison ranks the run as if the metric
had been measured and came out empty, and the adapter keeps its promise on paper
while breaking it on every episode.

The gap is visible only to someone who remembers what was promised. Reading a
run back, you cannot tell "this adapter did not emit `steps_per_second`" from
"this run genuinely measured zero throughput", and an episode that crashes before
reaching its telemetry call looks exactly like an episode that emitted nothing
worth reporting.

The promise and the observation live in different places. The declaration is
part of the environment spec; the emissions arrive one at a time through
`Telemetry`, which is constructed independently of the collector that owns the
episode boundary. Reconciling them requires one place where both are in scope.

## Decision

Add `game_learning_runtime.declared_metrics` and reconcile the promise against
the observation at **episode close**, always recording the account and only
raising when strict mode is on.

- `EnvironmentSpec.metrics` carries a `MetricDeclaration`: `expected` (a promise
  — absent means gap), `optional` (an extra — counted, never missing), and
  `strict`. The field is `None` by default, and a spec that declares nothing has
  no ledger, no counters, no event, and no behaviour that differs from before the
  field existed.
- `DeclaredMetricLedger` counts the metrics it sees and snapshots an
  `DeclaredMetricAudit` per closed episode. The counters are properties of the
  name tuples, so a persisted audit can never claim a count its names disagree
  with.
- Counting happens in `TrainingStore.record_metric`, the single funnel every
  metric passes through. An adapter that routes metrics through `Telemetry` and
  one that writes straight to the store are counted the same way. A run-scoped
  registry keyed by `(absolute store path, run_id)` connects the store to the
  ledger without threading a new argument through every telemetry surface.
- `SyncCollector` closes the audit when `following.done`, before `stop_on_done`
  ends the run. A metric that never arrived is a property of the episode that
  just ended, so that is where it is checked — not when a report is finally
  rendered, by which time the run has already been reported complete.
- `close_episode` persists the audit **before** `require()` can raise. The
  account always lands, including on the run that fails.
- Three counters are recorded as first-class run metrics —
  `declared_metrics`, `emitted_metrics`, `missing_metrics` — and projected in
  `glr.cli-output.v1` through `runs show`. A scheduler reads all three without
  walking events or parsing a log.
- Strict mode is opt-in: the `strict-metrics-v1` capability (the `strict_metrics`
  spelling is accepted), `MetricDeclaration(strict=True)`, or an explicit
  `strict=True` to `build_declared_metrics`. Without it the run completes and the
  manifest still names the gap. Visibility is the half that fixes behaviour; the
  error is the half that prevents recurrence.
- `MissingDeclaredMetric` names **every** missing metric and carries the
  remediation in its message: emit it, drop it from the declaration, or mark it
  optional. It refuses to be constructed with an empty gap — an error that
  reports nothing missing would be noise.

## Consequences

- A declared-and-never-emitted metric is now reported by name instead of reading
  as a zero, whether or not strict mode is on.
- An adapter that declares nothing is measured exactly as before: no ledger is
  built, so there is no new event, no new counter, and no new failure mode.
- `summarize_declared_metrics` reports `reported=False` with null counters when
  no audit exists, rather than a passing zero. "Nothing was measured" and
  "nothing was missing" stay different facts.
- One audit per episode: a run of N episodes carries N audits, and the summary
  unions their gaps so a metric missing in any episode is named.
- Counting in the store couples declared-metric accounting to `record_metric`.
  A future metric path that bypasses the store would have to be counted too.
- Strict mode is a breaking change for an adapter that declares a metric it
  cannot emit. That is the point, and it is why the capability switch exists:
  the declaration can be adopted first and the gate turned on later.

## Rejected alternatives

- Infer the contract from the first episode's emissions: rejected because a
  first episode that crashes before its telemetry call would silently redefine
  the contract downward, which is the exact failure being fixed.
- Warn instead of raising: rejected because a warning is skippable, and a
  declared metric that is never emitted is a broken contract, not a style
  question. The warning half is kept as the non-strict default, which is why the
  counters are always recorded.
- Check at report render time: rejected because by then the run has been
  reported complete and the caller has already acted on the result.
- Count emissions in `Telemetry` only: rejected because the store is the one
  place every metric passes through, and counting in `Telemetry` would miss an
  adapter that writes to the store directly.
- Make strict mode the default: rejected because it would fail every existing
  adapter that declares a metric before it can emit one, and the visibility half
  is worth shipping on its own.

## Related

- ADR-0011 defines the reward and provenance safety envelope this extends toward
  metrics.
- [Declared metrics guide](../guides/declared-metrics.md).
