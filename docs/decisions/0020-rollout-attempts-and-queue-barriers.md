# ADR-0020: Add rollout attempts and local queue barriers

- Status: Accepted
- Date: 2026-09-07

## Context

Long-running collection needs durable attempt history and a defined boundary
between unroll consumption and policy publication. Existing run events describe
execution, while actor queue leases already require an explicit commit or abort.
Neither previously represented retry lineage or paused new learner leases.

[Agent Lightning](https://github.com/microsoft/agent-lightning) provides the
inspiration: its [rollout and event model](https://microsoft.github.io/agent-lightning/stable/05-basics/)
separates execution status from training samples, and its
[asynchronous training design](https://microsoft.github.io/agent-lightning/stable/35-asynchronous-training/)
retains unfinished work across updates and drains active model requests before
publishing weights. GLR applies these ideas to its existing Python store and
thread queue; it does not add an Agent Lightning dependency.

## Decision

Add a local Python SDK projection, `RolloutAttempt`, to `TrainingStore`.
A logical rollout has a UUID-based `rollout_id`; every attempt has a new
UUID-based `attempt_id`. An explicit retry retains the rollout identity,
records the preceding failed attempt and reason, and increments the attempt
index. Only the latest failed attempt can acquire one successor.

Attempts move from `QUEUING` to `RUNNING` or `FAILED`, and from `RUNNING` to
`SUCCEEDED` or `FAILED`. Expected-status checks fence updates. Each projection
change and its append-only run event commit in one SQLite transaction. The
`glr.rollout.v1` event payload and `to_mapping()` sidecar include lineage,
timestamps, bounded metadata, and the corresponding run event sequence.
Finishing a parent run also fails its outstanding queued or running attempts.
Collection success does not establish game success or learner eligibility.

Extend `BoundedActorQueue` with a local learner-lease barrier. `pause()` stops
new leases; `drain()` pauses and waits for existing leases to commit or abort.
Queued work is retained, and the queue stays paused until `resume()`, including
after a failed drain wait. Producers can still enqueue under the existing
capacity and overflow rules. This boundary does not pause game time, actor
inference, or model requests. The application owns policy-object synchronization.

An optional policy-version lag cutoff rejects stale arrivals, drops stale
queued work before leasing, and rejects stale commits. It is disabled by
default. A rejected commit retains its lease for explicit abort and cannot
reverse optimizer changes. Applications must coordinate policy publication
with the barrier rather than treating commit as an optimizer transaction.

Queue summaries expose carry-over, in-flight leases, age, drain outcomes, and
stale rejection/drop counts. Applications choose when to persist snapshots as
run events. The [usage guide](../guides/rollout-control.md) defines the metric
semantics and the tested synthetic example.

## Consequences

Retry history survives Python store reopen, and concurrent state updates cannot
silently overwrite each other. The projection is additive and leaves the
Python store schema version at 2. Rust CLI store version 1 already differs from
Python version 2; this change does not resolve that pre-existing incompatibility
or promise that the CLI can read these records.

This batch adds no automatic game retries, remote scheduler, model gateway,
learner, or actor process supervisor. Durable attempt metadata does not restore
an in-memory queue after a process exits. Cross-process ownership, authenticated
multi-machine coordination, recovery, and actor policy publication protocols
remain future work in the [roadmap](../planning/roadmap.md).

## Alternatives considered

- Import Agent Lightning's trainer and gateway: unnecessary dependencies and
  model-specific behavior for a learner-neutral runtime.
- Retry failed runtime actions automatically: attempt metadata cannot establish
  whether a game action is safe to repeat.
- Discard all queued work at a policy update: makes carry-over impossible and
  forces a learner policy that should remain configurable.
- Treat queue pause as an inference pause: the queue does not own actor policy
  objects, game clocks, or external model endpoints.
