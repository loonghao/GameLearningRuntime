# Control local rollout attempts and policy updates

Use the Python SDK to retain collection attempt history and coordinate a policy
update with `BoundedActorQueue` learner leases. These APIs work with local
threads and SQLite. They do not schedule remote workers or retry game actions.
The architecture and source inspiration are recorded in
[ADR-0020](../decisions/0020-rollout-attempts-and-queue-barriers.md).

## Run the synthetic example

Set up the repository environment using the
[local development runbook](../runbooks/local-development.md), then run:

```powershell
.venv-glr/Scripts/python.exe scripts/rollout_control_demo.py --output-dir .glr/rollout-demo
```

The output directory must not already exist. Choose a new directory for each
run. The [complete example](../../scripts/rollout_control_demo.py) uses the
synthetic Counter environment, records one failed attempt and one successful
retry, and registers two attempt sidecars with SHA-256 artifact records.
Its JSON result contains these stable values, plus generated run and rollout
IDs:

```json
{
  "status": "succeeded",
  "attempts": 2,
  "committed_unrolls": 2,
  "stale_dropped_unrolls": 1,
  "artifact_count": 2
}
```

The example proves local orchestration behavior. It does not exercise a live
game or measure learning quality.

## Record an attempt and an explicit retry

Create the parent run with `TrainingStore.create_run()`, then call
`create_rollout(run_id, metadata=...)`. The returned `RolloutAttempt` starts in
`RolloutStatus.QUEUING` with `attempt_index=1`. Its UUID-based `rollout_id`
identifies the logical collection; `attempt_id` identifies this execution.
Metadata must be finite JSON with string keys, bounded to 64 KiB.

Call `update_rollout_attempt(attempt_id, status=..., expected_status=...)` to
advance the attempt. Supply `reason` when failing an attempt. Reasons must be
non-empty printable text of at most 256 characters.

| Current status | Allowed next status |
| --- | --- |
| `QUEUING` | `RUNNING`, `FAILED` |
| `RUNNING` | `SUCCEEDED`, `FAILED` |
| `SUCCEEDED` | None |
| `FAILED` | None; use `retry_rollout()` for a new attempt |

Enum members serialize as lowercase values. A stale `expected_status` raises
`ContractViolation` without changing the projection or appending an event.
Timestamps cannot precede the attempt's previous transition.
The first attempt cannot be queued before its parent run starts. Parent
completion cannot precede any child's latest transition, including terminal
children; rejected backdated writes leave the stored history unchanged.

After a failure, explicitly call `retry_rollout(attempt_id, reason=...)`.
Only the latest failed attempt can be retried, and it can have only one
successor. The successor keeps the rollout ID and metadata, has a fresh attempt
ID, increments the index, and records `parent_attempt_id` and `retry_reason`.
This operation only writes metadata: your application decides whether and how
to perform collection again.

Query `get_rollout_attempt(attempt_id)` or
`list_rollout_attempts(run_id=..., rollout_id=..., status=..., limit=...)`.
Filters are optional; the default limit is 100 and the maximum is 1,000.
Results are ordered by queued time, rollout ID, and attempt index. Queries
return attempt history rather than silently replacing older attempts.

Every state change appends a `rollout.<status>` run event in the same transaction.
`to_mapping()` returns the `glr.rollout.v1` payload, including the last run event
`sequence_id`; it does not write a file. The demo exports final projections and
uses `register_artifact()` to bind each sidecar to its bytes. Reopening the
Python store retains both the latest projections and earlier events.

`finish_run()` atomically marks outstanding attempts `FAILED` with a parent-run
reason. Existing terminal attempts remain unchanged. No attempt writes or
retries are accepted after the parent becomes terminal. An attempt's
`SUCCEEDED` status means collection completed, not that the game objective was
achieved or the samples satisfy a learner's acceptance rules.

## Establish a policy publication barrier

1. Call `queue.drain(timeout=...)`. It pauses new `get()` leases and waits for
   workers to commit or abort every existing lease. Calling `pause()` alone
   establishes the barrier without waiting.
2. After drain succeeds, publish the policy using your application's own actor
   synchronization, then call `set_learner_policy_version(version)` to update
   queue lag tracking and optional cutoffs.
3. Call `resume()` to permit new learner leases.

Queued unrolls stay in the queue. Producers may still enqueue while paused,
subject to capacity and overflow policy. Neither `pause()` nor `drain()` stops
game time, actor inference, or external model requests. A consumer blocked in
`get()` waits for resume and remains subject to its timeout or cancellation.

A drain timeout raises `ActorQueueFull`; cancellation raises
`ActorQueueCancelled`. Both leave the queue paused. Resolve outstanding leases
and retry the drain before treating the barrier as complete. `resume()` during
an active drain raises `ActorQueueCommitError`. Closing the queue wakes waiters
and releases the pause so remaining queued work can be consumed; it does not
constitute a successful policy publication barrier.

## Choose a stale-work cutoff

The constructor's `max_policy_version_lag=None` default disables stale-work
rejection. Set it to a non-negative integer to enforce
`max(0, learner_policy_version - unroll.policy_version) <= cutoff`.

The queue checks this condition on arrival, before returning a lease, and at
commit. A stale arrival raises `ActorQueueStaleUnroll`; already queued stale
work is dropped as `get()` searches for eligible work. A stale commit raises
`ActorQueueStaleUnroll` and keeps the lease in flight: explicitly `abort()` it
to release ownership. Commit cannot undo learner weights that were already
changed. Coordinate version publication with the lease barrier to avoid that
race. Queue cutoffs do not implement importance sampling or an RL algorithm.

## Read and persist queue metrics

`metrics().as_dict()` and `run_summary()` expose snapshots. Persist a snapshot
with `TrainingStore.append_event()` when your application reaches a useful
boundary, as the demo does; metrics are not stored automatically.

| Field | Meaning |
| --- | --- |
| `depth` | Currently queued unrolls, excluding active leases |
| `in_flight_unrolls` | Leases awaiting commit or abort |
| `uncommitted_unrolls` | Queued plus in-flight unrolls |
| `carry_over_unrolls` | Pending unrolls with a policy version older than the learner's current version |
| `oldest_pending_age_ns` | Maximum monotonic age since enqueue among pending unrolls; zero when empty |
| `max_policy_version_lag` | Highest observed lag; this metric is not the configured cutoff |
| `drain_count` | Successful drains only |
| `drain_latency_ns_total` | Accumulated duration of drain waits, including failed waits |
| `stale_dropped_unrolls` | Queued unrolls discarded during stale checks before leasing |
| `rejected_stale_unrolls` | Rejected stale arrivals; does not count rejected commits |

Carry-over is a pending-work snapshot, not a cumulative count of retries or
work inherited at the previous drain. Queue metrics contain no observation
tensors. They describe local execution and do not attest game completion.

## Storage and deployment scope

This first batch is a Python SDK feature. The additive rollout tables leave
the Python store schema at version 2. The Rust CLI store's pre-existing version
1 incompatibility is unchanged; do not assume it can query this database or
render rollout projections. Use the Python query methods above.

The queue remains in memory and is not restored from the attempt table after
a process exits. Cross-process worker ownership, remote scheduling, automatic
execution recovery, and actor inference barriers remain application concerns
or [future work](../planning/roadmap.md).
