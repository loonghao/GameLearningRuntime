# Declared metrics

A declared metric that is never emitted does not show up as an error. It shows up
as a zero. The metric is simply absent from the run's metric table, a dashboard
renders `0`, a comparison ranks the run as if the metric had been measured and
came out empty, and an adapter keeps its promise on paper while breaking it on
every episode.

`game_learning_runtime.declared_metrics` closes that hole. An adapter declares
what it promises, the runtime counts what actually arrives, and every closed
episode carries the account — by name, whether or not strict mode is on.

## The three counters

| Counter | Meaning |
| --- | --- |
| `declared_metrics` | How many metrics the adapter promised as `expected`. |
| `emitted_metrics` | How many of those it actually emitted. |
| `missing_metrics_count` | How many were promised as `expected` and never emitted. |

They are recorded as first-class run metrics on every audited episode, so they
are readable from `glr runs show --json` and from the run's metric table without
parsing a log. The count is named `missing_metrics_count` because
`missing_metrics` in the same envelope is the list of names; one field cannot be
both a number and a list. `optional` metrics are counted in the audit payload
(`optional_metrics`, `emitted_optional_metrics`) and are never part of the
missing count.

## Declaring metrics

A Python adapter declares its promises on the spec:

```python
from game_learning_runtime.declared_metrics import MetricDeclaration
from game_learning_runtime.specs import EnvironmentSpec

spec = EnvironmentSpec(
    environment_id="example.counter-v1",
    observation=observation,
    action=action,
    capabilities={"strict-metrics-v1"},
    metrics=MetricDeclaration(
        expected=("episode_reward", "steps_per_second"),
        optional=("inherited_rows",),
    ),
)
```

An adapter that reaches GLR over the host wire — every C#, C++, Unreal, Godot
or Unity provider — puts the same declaration in its descriptor. The collector
is never in the loop for those adapters, so the wire is the only place they can
promise anything:

```json
{
  "schema_version": "glr.declared-metrics.v1",
  "expected": ["episode_reward", "steps_per_second"],
  "optional": ["inherited_rows"],
  "strict": false
}
```

The field is optional. A descriptor without `metrics` declares nothing, and an
unusable declaration is rejected at handshake time as a `HostProtocolError`
rather than silently ignored.

- `expected` is a promise. A name here that an episode never emits is a gap, and
  it is reported by name.
- `optional` names the extras. An optional metric is counted when it arrives and
  is never counted as missing, so a reader can tell "declared and absent" from
  "never part of the contract".
- A name cannot be both expected and optional; the declaration rejects the
  overlap rather than silently picking one.
- Names follow the run store's metric identifier rule (`[a-z][a-z0-9_.-]*`, at
  most 128 characters), and at most 64 metrics can be declared.

`spec.metrics` defaults to `None`. **A spec that declares nothing is measured
exactly as it was before the field existed**: no ledger, no counters, no event,
no new failure mode. Declaring metrics is opt-in per adapter.

## Auditing a run

The collector closes the audit at episode end, before the run can be reported
complete:

```python
from game_learning_runtime.collector import SyncCollector

collector = SyncCollector(
    environment,
    declared_metrics=environment.spec.metrics,
    store=store,
    run_id=run.run_id,
)
result = collector.collect(policy, steps=100, stop_on_done=True)
collector.release_declared_metrics()  # after the run ends

for audit in collector.declared_metric_audits():
    print(audit.episode_id, audit.declared_metrics, audit.emitted_metrics, audit.missing_metrics)
```

Pass a `MetricDeclaration` and the collector resolves strict mode from the
adapter's own capabilities. Pass a ready `DeclaredMetricLedger` when the caller
already bound one to a run — it keeps its store binding and its audits, so
`store` and `run_id` must be left out.

**`store` and `run_id` are what make the declaration count.** Emissions are
counted in `TrainingStore.record_metric`, the one funnel every metric passes
through, and the store finds the ledger through a run-scoped registry. A
collector built from a declaration alone keeps its audits in memory and writes
nothing: the run store counts nothing and `glr runs show` reports
`reported: false`, which is indistinguishable from an adapter that never
declared anything. Pass both, or bind the ledger yourself as shown below.

If you drive the loop yourself, `build_declared_metrics` and
`DeclaredMetricLedger.close_episode` give you the same two steps:

```python
from game_learning_runtime.declared_metrics import build_declared_metrics

ledger = build_declared_metrics(spec.metrics, capabilities=spec.capabilities)
...
audit = ledger.close_episode(str(episode_id))
audit.require()  # raises MissingDeclaredMetric in strict mode
```

`close_episode` returns the audit and records it; `require()` is what raises.
Keeping them separate means the account always lands, including on the run that
fails: the audit is persisted **before** `require()` can raise.

### How emissions are counted

Counting happens in `TrainingStore.record_metric` — the single funnel every
metric passes through. An adapter that emits through `Telemetry` and one that
writes straight to the store are counted the same way. A run-scoped registry
keyed by `(absolute store path, run_id)` connects the store to the ledger, so no
new argument is threaded through every telemetry surface:

```python
from game_learning_runtime.declared_metrics import bind_declared_metrics, build_declared_metrics

ledger = build_declared_metrics(spec.metrics, capabilities=spec.capabilities)
bind_declared_metrics(store, run.run_id, ledger)
# ... run ...
ledger.release()  # or release_declared_metrics(store, run.run_id)
```

## Strict mode

Strict mode is opt-in and turns a gap into a `MissingDeclaredMetric` failure. Any
one of these turns it on:

| Switch | Where |
| --- | --- |
| `strict-metrics-v1` capability | `EnvironmentSpec.capabilities` (the `strict_metrics` spelling is also accepted) |
| `MetricDeclaration(strict=True)` | the declaration itself |
| `build_declared_metrics(..., strict=True)` | an explicit override |

Without it the run completes and the manifest still names the gap. Visibility is
the half that fixes behaviour; the error is the half that prevents recurrence.
That split lets an adapter adopt the declaration first and turn the gate on once
its emissions are trustworthy.

```python
try:
    collector.collect(policy, steps=100, stop_on_done=True)
except MissingDeclaredMetric as error:
    print(error.missing_metrics, error.episode_id)
```

The error names **every** missing metric and carries the remediation in its
message: emit it, drop it from the declaration, or mark it optional. It refuses
to be constructed with an empty gap — an error that reports nothing missing would
be noise.

## Reading the result

`glr runs show --json <run-id>` carries both the per-episode audits and a
summary, inside the usual `glr.cli-output.v1` envelope:

```json
{
  "declared_metrics": [
    {
      "schema_version": "glr.declared-metrics.v1",
      "episode_id": "episode-1",
      "timestamp_ns": 1735689600000000000,
      "strict": true,
      "complete": false,
      "declared_metrics": 2,
      "emitted_metrics": 1,
      "missing_metrics": ["steps_per_second"],
      "missing_metrics_count": 1,
      "expected": ["episode_reward", "steps_per_second"],
      "emitted": ["episode_reward"],
      "optional": ["inherited_rows"],
      "emitted_optional": [],
      "optional_metrics": 1,
      "emitted_optional_metrics": 0
    }
  ],
  "declared_metrics_summary": {
    "schema_version": "glr.declared-metrics.v1",
    "reported": true,
    "audit_count": 1,
    "episode_id": "episode-1",
    "strict": true,
    "complete": false,
    "declared_metrics": 2,
    "emitted_metrics": 1,
    "missing_metrics": ["steps_per_second"],
    "missing_metrics_count": 1
  }
}
```

The three counters are also ordinary run metrics (`declared_metrics`,
`emitted_metrics`, `missing_metrics_count`), so a dashboard or a scheduler can
read them from the metric table alone.

`summary.missing_metrics` unions the gaps across every audited episode, so a
metric missing in any episode is named. When no audit exists the summary reports
`reported: false` with null counters, never a passing zero: "nothing was
measured" and "nothing was missing" are different facts and must not look the
same.

An episode ended by an environment error is audited too, and the audit is
written without raising: the environment failure is the error worth
propagating, and a truncated episode with no audit would look exactly like an
adapter that declared nothing.

## Design boundaries

- **Inert when absent.** No declaration means no ledger, and behavior identical
  to before the field existed.
- **Checked at episode close**, not at report render time — by the time a report
  renders, the run has already been reported complete.
- **Always recorded.** The audit is persisted before the error can raise, so a
  strict failure still leaves the counters on the record.
- **One audit per episode.** A run of N episodes carries N audits.
- **Counters are derived.** `DeclaredMetricAudit` computes its counts from its
  name tuples, so a persisted audit can never claim a count its names disagree
  with.

## Related

- [Training safety](training-safety.md) — reward budgets and BC provenance.
- [Runtime evidence](runtime-evidence.md) — the run-evidence contracts these
  counters sit alongside.
- ADR-0040 in [the decision index](../decisions/README.md).
