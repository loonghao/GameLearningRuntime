# ADR-0045: Admit remote roles with scoped capabilities, epoch-scoped fences, and coordinator-owned checkpoints

## Status

Accepted — **contract only, Phase 0, no code.** This record decides the contract
`glr.remote-admission.v1` and the invariants a conforming implementation must hold.
Nothing in it is implemented. Phase 1 (an in-process reference harness) is authorized to
begin; no distributed code may be written before the Phase 1 conformance suite is green
in CI.

This record supersedes [the original proposal record](../planning/remote-role-admission.md),
which is retained for its baseline survey and review history and is no longer maintained.
The design review that produced it closed on 2026-09-28 and answered all eight open
questions; the rulings are folded into the decisions below.

Related: issue #116, ADR-0001, ADR-0020, ADR-0027 (which names this contract its stage 4
and remote conformance its stage 5), ADR-0030, ADR-0032, ADR-0036, ADR-0041 (whose §D9
delegates cluster integration to a separate contract shaped exactly like this one), and
the roadmap entry "Add authenticated multi-machine coordination around the implemented
local agent control plane".

## Context

ADR-0027 already names this work as its stage 4 and defers it deliberately. Its first
three stages — the source-only envelope, its negative corpus, and locked offline
reproduction — are local, offline, and provable on one machine. This stage is not: it
introduces cross-machine authentication, leases, fencing, and reconciliation after a
disconnect, which is a different threat model from a wire-format review and cannot be
settled by folding it into one.

The current primitives cannot be stretched to cover it. Their recurring shape is that
**they are unique within one process and durable within one store**:

- `attempt_id` is `attempt-{uuid4().hex}` (`run_store.py` →
  `TrainingStore._insert_rollout_attempt`). Unique, but unordered, so it supports no
  monotonic check, no range dedupe, and no gap detection.
- `ExclusiveInstanceLease` (`supervision.py` → `ExclusiveInstanceLease`) is an in-process
  registry with no TTL; a crashed holder never releases.
- `checkpoint.py`'s no-replace plus `fsync` plus `os.replace` is a property of one local
  filesystem. Nothing stops two machines from each writing a manifest that would be valid
  on its own.
- `BoundedActorQueue` commit/abort fencing is enforced by an in-memory dict, and its lease
  token is an integer allocated from one queue object, reset on restart, meaningless in
  another process.

The nine-row survey behind those four, with the per-primitive reason each one cannot fence
a remote worker, is in Part 1 of the proposal record. What follows from it: remote
admission needs an identifier another machine can validate, a lease that expires without
the holder's cooperation, and a checkpoint path with one writer.

Three constraints shape every decision below. No scheduler, learner, or cloud vendor may
become a mandatory dependency (ADR-0005, ADR-0015, ADR-0032). GLR never interprets what a
policy means (ADR-0027); it only refuses to let a worker running different source or
different weights contribute to a run pinned to specific digests. And no concurrency or
recovery guarantee is claimed that the conformance checklist does not test.

## Decision

### D1. Contract identity and scope

The contract is `glr.remote-admission.v1`. It defines identifiers, state names, rejection
reasons, and invariants. It selects no transport, no authentication mechanism, no
scheduler, and no cloud provider, and names none.

Its scope is admitting a remote role into a training run and ingesting what it produces
back into the optimizer. Out of scope, explicitly: a remote scheduler or its selection; a
new mandatory dependency (any implementation is an optional, feature-gated adapter); any
change to the existing local queue, attempt, or store contracts, all of which stay
additive; and any claim that issue #116 stages 4–5 are satisfied by this document, which
is their design input rather than their evidence.

### D2. Roles, topology, and the v1 capability set

Two participants, defined by what they own rather than by what they are:

- **Coordinator** — the single writer for one run. It owns the optimizer, the policy
  version counter, the checkpoint generations, and the admission decisions.
- **Worker** — a process that admits itself to a run, receives one scoped capability,
  executes it, and returns results.

**One coordinator per run, not per trial.** The coordinator is the run's single writer; a
per-trial coordinator would need a leader-election story per trial and would leave
`coordinator_epoch` ambiguous for the ordered identifiers in D5. A "cluster" is one
coordinator plus zero or more workers; nothing in this contract requires a scheduler
between them, and the deployment chooses how a worker finds its coordinator.

A **capability** is an enumerated, deny-by-default grant, scoped by
`(project, environment_id, trial_id, package_digest, policy_digest, max_in_flight,
max_payload_bytes, max_duration)`. **Every v1 capability is trial-scoped.** The v1 set is
deliberately small and closed:

- `collect:unroll` — run one unroll and return its result.
- `evaluate:episode` — run one evaluation episode and return its result.
- `propose:checkpoint` — write checkpoint bytes to **per-lease staging only** and return
  their digest. It never writes to the run's checkpoint location; see D10.

**Why v1 has no `read:policy`.** The policy a worker executes against is already delivered
through the pinned package bound at admission and through `policy_digest` (ADR-0027's
offline source-package path). A separate read capability grants no capability admission
does not already convey, while adding a second authorization story and a
read-amplification surface with no caller that needs it. It is deferred, not forbidden: it
re-enters only with a concrete caller the pinned package cannot serve.

**Why there is no remote learner role.** ADR-0041 §D9 asks for "authenticated learner and
actor roles". This contract authenticates the **actor** side only and keeps the learner
behind the coordinator on one machine, because the coordinator is already the single
writer for the optimizer and the policy version counter. Admitting a remote learner would
turn policy publication into a distributed decision and would need a second single-writer
story that nothing here has earned. **v1 admits the actor role; a remote learner capability
is deferred, not designed away.** This is the explicit scope answer to ADR-0041 §D9.

### D3. Admission

A worker requests admission with:

| Field | Meaning |
| --- | --- |
| `worker_id` | Deployment-assigned identity. GLR does not mint it. |
| `package_digest` | The package content identity. **Version-independent** (ADR-0041 §D4): SHA-256 over the ordered selection and inventory entries only, with `tool_version` recorded in the manifest but excluded from the hash. See the dependency note below. |
| `policy_digest` | Digest of the policy artifact the worker will execute against (for a checkpoint, its manifest `checkpoint_sha256`). |
| `capabilities` | The capabilities the worker requests. |
| `lease_seconds` | Requested lease duration. |
| `nonce` | Freshness value, echoed in the grant. |

The coordinator grants, or refuses with a machine-readable reason:

| Field | Meaning |
| --- | --- |
| `lease_id` | Opaque, unique per grant. |
| `fence` | Strictly monotonic fencing token, per `(run_id, coordinator_epoch, resource)`. See D4. |
| `lease_seconds` | **A duration, not a deadline.** See D7. |
| `granted_capabilities` | The intersection of requested and permitted, never a superset. |
| `policy_version` | The version this worker will be measured against. |
| `max_in_flight`, `max_payload_bytes` | Backpressure terms. See D6. |

Refusal reasons: `ADMISSION_DIGEST_MISMATCH` (source or policy digest differs from what
the run pinned), `ADMISSION_CAPABILITY_DENIED`, `ADMISSION_CAPACITY_EXCEEDED`,
`ADMISSION_CLOSED` (the run is terminal).

**Binding the two digests is the learner-neutral core of this contract.** GLR never
interprets what a policy means; it only refuses to let a worker that is running different
source or different weights contribute to a run whose identity was pinned to specific
digests.

**Dependency: the ADR-0041 §D4 migration, and how implementations must read identity.**
ADR-0027 as shipped computes content identity over selection, `tool_version`, and
inventory. ADR-0041 §D4 changes that to selection and inventory only and leaves the
migration mechanism open (a new source-only schema revision, or a parallel identity field
with a documented transition). **This contract follows §D4 and inherits its migration
decision**, because admission is keyed on `package_digest` and refuses with
`ADMISSION_DIGEST_MISMATCH`: a digest that varies with the packaging CLI version would
reject two workers running byte-identical source, which is exactly the question §D4 wants
the identifier to answer.

Until that migration lands, `package_digest` is whatever `glr.source-package.v1` emits and
mismatch is expected to **over-reject across CLI versions**. That fails closed, so it is
accepted as the migration-period behaviour rather than worked around. What is not accepted
is pinning today's ADR-0027 computation into an implementation or a test: an
implementation must obtain `package_digest` through an **injected identity function** so
the migration changes one seam instead of rewriting the conformance suite. This is a Phase
1 gate, not a suggestion.

### D4. Epoch-scoped fencing tokens

`fence` is a strictly monotonic integer scoped by `(run_id, coordinator_epoch, resource)`,
issued by the coordinator and carried by every subsequent mutation. The coordinator keeps
`max_fence_seen` per scope **in memory** and rejects any mutation whose `fence` is lower
with `FENCE_STALE`, changing no state and incrementing a counter.

**`coordinator_epoch`** names one coordinator incarnation for a run. It increments on every
coordinator start or restart, and a new epoch allocates fences **from zero**.

Three rules keep the token usable:

- **Renewal preserves the fence.** A worker that renews before expiry keeps working with
  the same `fence`, so a renewal never invalidates its own in-flight attempts.
- **Re-admission issues a strictly greater fence.** Any grant after an expiry or a conflict
  is greater than every fence previously issued for that scope.
- **A fence from an earlier epoch is `EPOCH_STALE`.** Monotonicity across restarts comes
  from the epoch, not from a durable counter, so a restart needs no persisted
  `max_fence_seen` and this design keeps no persistence dependency it would otherwise have
  to justify. Rejecting an older epoch outright is also the safe answer: a restarted
  coordinator cannot know what its previous incarnation admitted.

What fencing buys, and what it does not: it protects **coordinator-side state** from a
worker that has lost its lease and does not know it. It does **not** protect **worker-side
effects** that already happened. A worker whose lease expired mid-action may already have
driven the environment. No token undoes that, which is why the state taxonomy below has a
terminal state for "we will never know".

### D5. Result ingestion: ordering, idempotency, lease-state recording

The local attempt identifier is a UUID and therefore unordered. Remote ingestion needs
order, so the coordinator allocates it:

- `attempt_seq` — monotonic per `(run_id, coordinator_epoch)`, issued by the coordinator.
- `attempt_id` — composite identity `(run_id, coordinator_epoch, attempt_seq, attempt_uuid)`,
  retaining a UUID for uniqueness across stores while ordering comes from `attempt_seq`.
- `ingest_id` — `sha256(attempt_id ‖ lease_id ‖ fence ‖ result_digest)`, the idempotency key.

Ingestion rules:

1. A result is **deduplicated on `ingest_id`**. A duplicate delivery returns the first
   recorded outcome and applies nothing. **A duplicate delivery never becomes a second
   optimizer update.**
2. The same `attempt_id` arriving with a *different* `result_digest` is `INGEST_CONFLICT`:
   recorded for inspection, and no optimizer update is applied either. Two different
   results for one attempt is a bug or an attack, never a merge decision.
3. Order of application is decided by the coordinator from `attempt_seq`, never by arrival
   order at the socket.
4. A result arriving with an expired lease, a stale fence, or a superseded epoch is
   `UNOWNED`: recorded as a metric, never applied.
5. Ingest records the **lease state it observed** — `live`, `expired`, or `superseded` —
   alongside the outcome and the digest. **An expiry never rolls back an update that was
   already ingested.** Ingestion is committed at ingest, so a partition wall appears as a
   metric rather than as a silent rewrite of optimizer history, and the recorded lease
   state is what makes that wall auditable after the fact.

### D6. Backpressure: reject, never silently drop

The local queue may `drop-oldest` because the coordinator of that queue is the same
process that produced the unroll and can account for the loss. A remote coordinator
cannot. A dropped remote result is indistinguishable from a result whose worker died, so
remote overflow is always **`reject`**: the worker is told to abort the attempt, and the
abort is recorded. Silent remote drops are not a supported policy.

Backpressure terms are `max_admitted_workers`, `max_in_flight` per worker,
`max_payload_bytes`, and `max_result_age`. The coordinator may tighten them at any time;
tightening never revokes a live lease retroactively.

### D7. Leases: duration, not deadline

A lease is granted as `lease_seconds` and measured by each side against **its own
monotonic clock from the moment of receipt**, not against an absolute expiry timestamp.
The design therefore does not assume synchronized clocks; wall-clock timestamps appear only
in diagnostics. Expiry is a local decision on each side, and the two sides may disagree for
the duration of one lease — which is exactly why every mutation carries a fence and why an
expired lease never invalidates a result already ingested (D5 rule 4 is about *subsequent*
accepts only).

### D8. Reconciliation after disconnect or reconnect

Reconnection is **re-admission, not resumption**:

1. The worker requests admission again and receives a strictly greater `fence`.
2. It reports the state of every attempt it owned:
   - completed, with digest → idempotent ingest (D5);
   - **in flight with an unknown outcome → `ORPHANED`**;
   - admitted but not started → returned to the queue.
3. An in-flight attempt is **never resumed in place** and **never replayed because a
   transport reconnected**. ADR-0020's reasoning applies with more force across a network:
   attempt metadata cannot establish whether a game action is safe to repeat. A retry is a
   *new* attempt with a new `attempt_seq`, and only the project or a human may decide to
   create one.

Cancellation follows the same shape: the coordinator requests it, the worker makes a
bounded best-effort attempt to stop and reports `CANCELLED`. If the outcome cannot be
established, the attempt is `ORPHANED`, not `CANCELLED`. An attempt is terminal exactly
once.

### D9. State taxonomy, and the two budgets

ADR-0020's local projection keeps its existing four states. The remote-only outcomes are
coordinator-side and are mirrored into the local store as `RolloutStatus.FAILED` with a
machine-readable `failure_reason`, so nothing that reads `RolloutAttempt` today has to
change. The mapping is one-to-one and lowercase: `ORPHANED` → `orphaned`, `STALE` →
`stale`, `UNOWNED` → `unowned`. `run_store.py` → `TrainingStore.update_rollout_attempt`
already rejects a `FAILED` attempt that carries no reason, so a remote outcome that loses
its reason fails closed instead of landing as an unexplained failure.

| State | Meaning | Applied to the optimizer |
| --- | --- | --- |
| `SUCCEEDED` | Result ingested and accepted. | Yes, exactly once. |
| `FAILED` | Worker reported a definite failure. | No. |
| `CANCELLED` | Worker confirmed it did not act. | No. |
| `ORPHANED` | Outcome is unknowable: expired lease, disconnect, or unconfirmed cancellation. | No, and **never retried automatically**. |
| `STALE` | Result accepted but `observed_policy_version` lags beyond the cutoff. | No, and counted. |
| `UNOWNED` | Arrived without a live lease or with a stale fence. | No, and counted. |

**Two budgets, because `ORPHANED` is not a training outcome.** Counting `ORPHANED` against
the run's training budget lets one network fault end the run; not counting it anywhere
leaves a silent worker leak with no bound. Both are unacceptable, so the two are separate:

- **`attempt_budget` (training).** Counts `SUCCEEDED`, `FAILED`, and `CANCELLED` only.
  `ORPHANED` never reached the optimizer, so it is not charged here. Exhausting this budget
  ends attempt allocation exactly as it does today.
- **`orphan_budget` (orchestration).** A separate bounded count and rate over `ORPHANED`
  outcomes. Exhausting it **stops admitting new workers and raises a run-level alarm**, and
  **does not terminate the run**. Ending a run is a project or human decision; a breached
  orchestration budget is evidence for that decision, not a substitute for it.

Both are counted per run and reported as metrics, so a run that is burning through its
orphan budget is visible before either limit is reached.

**These are attempt-level outcomes, not episode terminations.** `termination.py` →
`TerminationReason` is episode-level and is deliberately **not** reused here. In particular
`ENV_INDETERMINATE` ("the environment consequence of an action is unknown, so restart
rather than act again") answers a different question than attempt-level `ORPHANED` ("this
attempt's result never arrived and cannot be reconstructed"). An episode can end
`env_indeterminate` while its attempt ingests normally, and an attempt can be `ORPHANED`
with no episode termination recorded at all. The two must not be collapsed into a single
"we do not know" concept.

Policy lag mirrors the local cutoff with one remote difference: the worker is **told** to
stop producing, because continuing to spend its budget on results that will be rejected is
worse than pausing. A run may explicitly enable carry-over, in which case stale results are
recorded as `STALE` metrics rather than discarded — recorded either way.

### D10. Checkpoint ownership: staged on the worker, promoted by the coordinator

The run's checkpoint location has **exactly one writer on one machine: the coordinator**.
The no-replace plus `fsync` plus `os.replace` behaviour in `checkpoint.py` is a property of
one local filesystem; it does not survive being moved across machines, and pretending
otherwise turns "exactly one proposal wins" into a distributed negotiation. The write path
therefore splits in two:

- **Worker: staging only.** A worker holding `propose:checkpoint` writes its bytes to a
  **per-lease staging area** and returns `checkpoint_sha256`. Staging is addressed per
  lease, is not the run's checkpoint location, and is never resolved as one by any code
  path.
- **Coordinator: the only promote.** Promotion verifies the staged bytes against the
  returned digest and performs the single no-replace write into the run's checkpoint
  location. "Exactly one proposal wins" is then decided where no-replace actually holds —
  locally, on the coordinator.

- At most one live lease may hold a checkpoint-writing capability for a given
  `(trial_id, generation)`.
- Promotion requires a live lease, an in-epoch `fence >= max_fence_seen` for that
  generation, and the no-replace write semantics already used by `checkpoint.py` and
  ADR-0027. Exactly one proposal wins; the loser receives `PROMOTE_CONFLICT` and its
  staging bytes are retained for inspection.
- Staged bytes whose lease expires before promotion are labelled **unowned**, retained for
  inspection, and **never promoted and never referenced as the run's checkpoint**.

### D11. Trust boundary

- **The deployment owns authentication.** GLR defines the claim set and the refusal
  reasons; it does not choose mTLS, tokens, or workload identity, and it ships none.
- **Credentials and machine bindings stay deployment-local.** They are never part of a
  distribution package, and the package path stays offline and script-free (ADR-0027). A
  worker resolves its own credentials from its own deployment.
- **Ingest logs carry digests, counters, state transitions, and observed lease state —
  never observation data.** Following `ActorQueueMetrics`, the coordinator records digests,
  counters, and state transitions, plus the lease state observed at ingest (D5 rule 5). It
  never records observations or game content.
- **Ingest-log retention is bounded by the run.** Ingest records live and die with the run
  they belong to: there is no cross-run ingest log and no retention window that outlives
  the run. A deployment that needs a longer audit trail exports from these records on its
  own terms; GLR does not retain them on its behalf.
- **Blast radius, not immunity.** Scoped capabilities, expiring leases, epoch-scoped
  fences, deduplicated ingestion, and unique checkpoint ownership limit what a compromised
  or misbehaving worker can do. They do not make one safe.

## Conformance checklist

These are the tests an implementation must pass before it may claim anything about
authenticated multi-machine execution. ADR-0027 stage 5 names six families — duplicate
deliveries, stale ownership, dropped results, lease expiry, policy mismatch, and checkpoint
conflict. Tests 1–8 and 11 cover those families directly; the rest are this contract's
additions.

| # | Scenario | Invariant that must hold |
| --- | --- | --- |
| 1 | **Duplicate delivery** — one result, same `attempt_id` and `result_digest`, delivered twice. | The second ingest is a no-op returning the first outcome. Optimizer updates: exactly 1. |
| 2 | **Conflicting delivery** — same `attempt_id`, different `result_digest`. | `INGEST_CONFLICT`; no optimizer update; both payloads retained for inspection. |
| 3 | **Stale ownership** — mutation carrying `fence < max_fence_seen` inside the current epoch, and a mutation carrying a fence from an earlier `coordinator_epoch`. | `FENCE_STALE` in the in-epoch case and `EPOCH_STALE` in the cross-epoch case; state unchanged in both; the matching counter incremented. |
| 4 | **Lease expiry** — worker stops heartbeating with attempts in flight. | Lease expires on the coordinator; in-flight attempts become `ORPHANED`; no automatic retry; the next grant has a strictly greater `fence`. |
| 5 | **Late result after expiry** — result arrives carrying an expired lease's fence. | `UNOWNED`; recorded as a metric, including the lease state observed at ingest; never applied. |
| 6 | **Policy mismatch** — `observed_policy_version` lag exceeds the cutoff. | `STALE`; never applied; counted, and the worker is told to stop. |
| 7 | **Checkpoint conflict** — two workers propose promotion for one `(trial_id, generation)`. | Exactly one wins by no-replace on the coordinator; the loser gets `PROMOTE_CONFLICT`; only the winner's digest is referenced. |
| 8 | **Checkpoint from an expired lease** — staged bytes arrive after expiry. | Never promoted, whatever their digest. |
| 9 | **Reconnect storm** — one worker reconnects N times. | At most one live lease per `worker_id`; every superseded fence is rejected. |
| 10 | **Cancellation in flight** — cancel an admitted attempt. | Terminal exactly once: either `CANCELLED` or `ORPHANED`, never both, never neither. |
| 11 | **Backpressure overflow** — exceeds `max_in_flight` / `max_payload_bytes`. | `reject` plus a recorded abort. No silent drop. |
| 12 | **Clock skew** — worker monotonic-vs-wall-clock offset of ±1 hour. | Duration-based leases still behave; no premature expiry, no extended window. |
| 13 | **Package hygiene** — export a selection containing credential- or secret-shaped paths. | Rejected at export; no credential material in the archive. |
| 14 | **No replay of unknown outcomes** — an `ORPHANED` attempt exists. | The environment is not re-driven for it automatically; a retry is a new `attempt_seq` created by the project or a human. |
| 15 | **Determinism** — the whole suite runs in one process with an injected clock and fault injection. | Tests 1–12, 16, and 17 pass without a socket, so the concurrency invariants are provable in ordinary CI. |
| 16 | **Orphan budget breach** — `ORPHANED` outcomes exceed `orphan_budget`. | Admission of new workers stops and a run-level alarm is raised. The run is **not** terminated, and `attempt_budget` is untouched by the `ORPHANED` count. |
| 17 | **Staging is not a checkpoint** — a worker stages checkpoint bytes and its lease then expires. | The run's checkpoint location is unchanged, the staged bytes are labelled unowned and retained for inspection, and no code path resolves them as the run's checkpoint. |

Test 15 is the reason this contract is not simply "unprovable in CI". The *concurrency
contract* is designed to be provable in-process; only the *authentication and transport*
layers need real infrastructure, and only those layers' claims stay out of CI's scope.

## Non-functional requirements

- **Correctness:** a duplicate delivery never becomes a second optimizer update; a stale or
  foreign ownership claim never changes coordinator state; exactly one checkpoint proposal
  is promoted per `(trial_id, generation)`.
- **Compatibility:** everything is additive. ADR-0020's local queue, attempt, and store
  contracts are unchanged, and remote-only outcomes arrive as `FAILED` with a reason that
  today's readers already tolerate.
- **Auditability:** every ingest records digests, counters, state transitions, and the
  lease state observed; a partition wall is visible as a metric rather than as a silent
  rollback, and the orphan budget is observable before it is exhausted.
- **Learner neutrality:** the contract binds `package_digest` and `policy_digest` without
  interpreting either, and no scheduler, learner, or cloud vendor is named or required.
- **Security:** deny-by-default capabilities, expiring leases, epoch-scoped fences,
  deployment-owned credentials that never enter a package, and ingest records that never
  carry observation data.
- **Provability:** the concurrency invariants are testable in one process with an injected
  clock and fault injection, so they are enforceable in ordinary CI.

## Failure modes and mitigation

- **Duplicate delivery becomes a duplicate update.** `ingest_id` dedupe (D5 rule 1), and a
  conflicting digest for one attempt is `INGEST_CONFLICT`, never a merge.
- **Zombie worker keeps mutating state.** Epoch-scoped fences reject a stale fence with
  `FENCE_STALE` and a previous epoch with `EPOCH_STALE`, both without changing state.
- **Coordinator restart loses fencing history.** Fences are scoped by
  `coordinator_epoch`, so a new incarnation starts from zero with no persisted counter and
  rejects the old epoch outright instead of guessing.
- **Silent worker leak after repeated orphans.** A bounded `orphan_budget` stops admission
  and raises a run-level alarm while leaving the run alive for a human or project decision.
- **Network fault ends a training run.** `ORPHANED` is charged to `orphan_budget`, not to
  `attempt_budget`, so partition noise cannot terminate training.
- **Two machines write the same checkpoint.** Workers stage per lease; only the coordinator
  promotes, so no-replace stays a local filesystem property.
- **Silent remote result loss.** Overflow is always `reject` with a recorded abort, never
  `drop-oldest`.
- **Clock skew between machines.** Leases are durations measured on each side's own
  monotonic clock, and wall-clock timestamps are diagnostics only.
- **D4 migration churn.** Identity is read through an injected identity function, so the
  migration touches one seam rather than the conformance suite.
- **Compromised worker credential.** Bounded by scoped capabilities, expiring leases, and
  unique checkpoint ownership. Restricting blast radius is not immunity, and this contract
  does not claim otherwise.

## Consequences

### Positive

- Authenticated multi-machine execution has one reviewable contract with its own threat
  model, instead of being folded into a wire-format review.
- The concurrency invariants are provable in ordinary CI, so the parts that can be tested
  are tested before any distributed code exists.
- `ORPHANED` stops being either a run-killer or an unbounded leak: it has its own budget
  and its own alarm.
- Fencing needs no durable `max_fence_seen`, so a deliberately persistence-free design
  stays persistence-free across coordinator restarts.
- Checkpoint promotion stays a single-writer, single-filesystem operation instead of a
  distributed negotiation.

### Negative

- Two budgets instead of one, and one more alarm path to operate.
- Workers acquire a staging lifecycle: staged bytes must be labelled `unowned` on expiry,
  retained for inspection, and garbage-collected on deployment terms.
- `EPOCH_STALE` discards work from a previous coordinator incarnation rather than
  reconciling it; that is deliberate, but it is lost work after a restart.
- Until the ADR-0041 §D4 migration lands, admission over-rejects across CLI versions.
- The injected identity seam is a small amount of indirection carried for a migration that
  has not happened yet.

### Neutral

- Which transport and authentication mechanism a deployment uses stays a deployment
  decision, and this contract's claims stop at the concurrency and ownership layer.
- Whether the remote learner role is ever admitted stays open; v1 answers actor-only and
  names what a learner capability would have to solve first.

## Alternatives considered

**Keep fencing monotonic across restarts by persisting `max_fence_seen`.** It preserves
results from a previous incarnation, but it puts a persistence dependency into a design
that deliberately has none and still cannot tell a restarted coordinator what it admitted.
Rejected in favour of epoch scoping.

**Let workers write checkpoints and resolve conflicts by negotiation.** It avoids a
staging hop, but no-replace is a single-filesystem property; across machines "exactly one
proposal wins" would need a distributed agreement this contract has not earned. Rejected in
favour of worker staging plus coordinator-only promotion.

**Count `ORPHANED` in the training attempt budget.** Simpler, and it bounds the leak, but
it lets one network fault end a run. **Count it nowhere.** Also simpler, and it protects
training, but the leak is then unbounded and invisible. Both rejected in favour of a
separate bounded `orphan_budget` that alarms without terminating.

**Keep `read:policy` in v1.** Rejected: it grants nothing the pinned package and
`policy_digest` do not already convey, and it adds an authorization story plus a
read-amplification surface for no caller.

**Pin the ADR-0027 `package_digest` computation for v1.** Rejected: it contradicts
ADR-0041 §D4, rejects workers running byte-identical source, and would have to be undone
by the migration. Over-rejecting across CLI versions fails closed and is accepted instead.

**Promote the proposal to an ADR after Phase 1 instead of before.** Rejected: the previous
review already caught two documents contradicting each other; an unaccepted contract is
what invites implementation to outrun its own acceptance criteria.

## Phased delivery

| Phase | Content | Gate |
| --- | --- | --- |
| 0 | This contract. No code. | Human design review. **Closed 2026-09-28.** |
| 1 | In-process reference harness with an injected clock and fault injection, implementing the coordinator decision logic against the existing `BoundedActorQueue`. No transport, no authentication. `package_digest` is obtained through an **injected identity function**, never a copy of the ADR-0027 computation the harness happens to run against today. | Conformance tests 1–12, 16, and 17 green in CI. |
| 2 | Optional, feature-gated adapter crate (default off). Deployment-owned authentication. No change to the core queue, attempt, or store contracts. | The Phase 1 suite runs unchanged against the adapter. |
| 3 | Attended multi-machine conformance run over tests 1–8 and 13, with published results. | Results published **before** any claim of authenticated multi-machine execution. |

Phase 1 may begin now. No distributed code (Phase 2 or later) may be written before the
Phase 1 suite is green in CI.

## Explicitly not claimed

- Exactly-once execution. Exactly-once *application to the optimizer* is claimed, and only
  under the dedupe rules in D5.
- Linearizability, serializability, or any ordered guarantee beyond `attempt_seq` per
  coordinator epoch.
- Any bounded recovery time, or any guarantee that a run makes progress during a partition.
- That a lease holder has stopped working. A lease expiring is a coordinator-side decision;
  it is not knowledge about the worker.
- That any action is safe to replay. `ORPHANED` exists precisely because this cannot be
  established.
- Agreement between machine clocks. Leases are durations.
- Safety against a compromised worker credential, or against a malicious coordinator.

## References

- [Original proposal record](../planning/remote-role-admission.md) (accepted 2026-09-28;
  retained for its baseline survey and review history)
- [Training package phases](../planning/training-package-phases.md)
- ADR-0001, ADR-0005, ADR-0020, ADR-0027, ADR-0030, ADR-0032, ADR-0036
- ADR-0041 §D4 (content identity) and §D9 (cluster integration is a separate contract,
  outside the core)
- Upstream issue `loonghao/GameLearningRuntime#116`
