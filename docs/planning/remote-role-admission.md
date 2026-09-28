# Optional authenticated cluster distribution: remote role admission

> **Accepted proposal record — not normative.** The design review closed on 2026-09-28
> and this contract was promoted to
> [ADR-0045: Admit remote roles with scoped capabilities, epoch-scoped fences, and coordinator-owned checkpoints](../decisions/0045-remote-role-admission.md).
> **ADR-0045 is the normative record.** This document is kept as the original proposal
> and its review history and is no longer maintained; where the two differ, ADR-0045 wins.

- Status: **Accepted as a contract; not implemented.** The eight open questions were
  answered by the design review on 2026-09-28 and are recorded under "Design review
  rulings" below. Phase 1 may begin; no distributed code may be written before the
  Phase 1 conformance suite is green in CI.
- Contract name: `glr.remote-admission.v1` (assigned by ADR-0045)
- Date: 2026-09-19; accepted 2026-09-28
- Related: issue #116, ADR-0001, ADR-0020, ADR-0027 (which names this contract its stage
  4 and remote conformance its stage 5), ADR-0030, ADR-0032, ADR-0036, ADR-0041 (whose
  §D9 delegates cluster integration to a separate contract shaped exactly like this
  one), ADR-0045 (this contract, as accepted), and the roadmap entry "Add authenticated
  multi-machine coordination around the implemented local agent control plane".

## Why this is a separate proposal

ADR-0027 already names this work as its stage 4 and defers it deliberately. The first
three stages of issue #116 — the source-only envelope, its negative corpus, and locked
offline reproduction — are local, offline, and provable on one machine. This stage is
not: it introduces cross-machine authentication, leases, fencing, and reconciliation
after a disconnect, none of which can be proven by a single-machine test run.

Keeping the two in one change would bury a new threat model inside a wire-format
review. It would also violate the learner-neutral core contract (ADR-0001) the moment a
scheduler, a learner, or a cloud vendor became a mandatory dependency.

## Scope

This document defines the **contract** for admitting a remote role into a training run
and for ingesting what it produces back into the optimizer. It defines identifiers,
state names, rejection reasons, and the invariants a conforming implementation must
hold. It does not select a transport, an authentication mechanism, a scheduler, or a
cloud provider, and it names none.

## Non-goals

- No remote scheduler, and no scheduler or cloud-vendor selection.
- No new mandatory dependency. No scheduler, learner, or cloud SDK enters the core
  distribution; any implementation is an optional, feature-gated adapter.
- No change to the existing local queue, attempt, or store contracts. Everything
  proposed here is additive.
- No concurrency or recovery guarantee that the conformance suite in this document does
  not test. See "Explicitly not claimed".
- No claim that issue #116 stages 4–5 are satisfied by this document. It is the design
  input to them, not their evidence.

## Part 1 — Baseline: what exists today (implemented)

Every row below is shipped behavior. None of it is a remote-distribution primitive, and
the third column is why.

Anchors are **symbol references, not line numbers**, written `module.py` → `Symbol` with
`module.py` resolved under `src/game_learning_runtime/`. Line numbers shift on any rebase
and misdirect a reviewer with no signal; a symbol stays `grep`-stable for as long as the
behavior it names exists.

| Primitive | Implemented guarantee | Why it cannot fence a remote worker |
| --- | --- | --- |
| `BoundedActorQueue` (`collector.py` → `BoundedActorQueue`) | In-process, `threading.Condition`-based queue with capacity, `block` / `drop-oldest` / `fail` overflow policies, and `pause()` / `drain()` / `resume()` learner-lease barriers. | The lease is an integer counter (`QueuedUnroll.token`) allocated from one queue object. It is unique only inside that object, resets on process restart, and is meaningless in another process. There is no expiry. |
| `commit` / `abort` fencing (`collector.py` → `BoundedActorQueue.commit` / `BoundedActorQueue.abort`) | A leased unroll is finalized exactly once; a second or unknown finalization raises `ActorQueueCommitError` from `_validate_in_flight_locked`. | Exactly-once is enforced by an in-memory dict, not by a durable cross-machine token. A remote worker cannot prove it still holds anything. |
| Optional policy-lag cutoff (`collector.py` → `BoundedActorQueue._is_stale_locked`) | Configurable `max_policy_version_lag` drops stale queued unrolls and rejects stale commits; disabled by default. | The cutoff compares an in-process `learner_policy_version` against `Unroll.policy_version`. Both are process-local. |
| `RolloutAttempt` projection (`run_store.py` → `RolloutAttempt`, ADR-0020) | Retry lineage with `QUEUING` → `RUNNING` → `SUCCEEDED` / `FAILED`, expected-status fencing, and one SQLite transaction per projection change plus its append-only run event. | **`attempt_id` is `attempt-{uuid4().hex}`** (`run_store.py` → `TrainingStore._insert_rollout_attempt`). UUIDs are unique but carry no order, so they support no "monotonically increasing" check, no range dedupe, and no gap detection. Only `attempt_index` (per rollout lineage) and the append-only event `sequence_id` are ordered, and both are scoped to one store. |
| `ExclusiveInstanceLease` (`supervision.py` → `ExclusiveInstanceLease`) | One holder per in-process registry; `ProcessIdentity` is `(pid, start_time_ns)` to resist PID reuse; `ArtifactOwnershipError` gates artifact operations while the owner is alive. | Docstring is explicit: "Small in-process lease registry; adapters may replace it with a durable store." There is no TTL, no expiry, and a crashed holder never releases. |
| `ProcessSupervisor` (`supervision.py` → `ProcessSupervisor`, ADR-0032) | Explicit stop sequence, bounded waits, and exclusivity preserved across restarts. | Liveness is an adapter-supplied `ProcessProbe` for a PID on the local machine. |
| Checkpoint manifest (`checkpoint.py` → `CheckpointManifest`) | `checkpoint_sha256`, size, and a `CheckpointContract`; `write_checkpoint_manifest` refuses to overwrite an existing manifest; writes are temp-file + `fsync` + `os.replace`. | No-replace is a local filesystem property. Nothing prevents two machines from each writing a manifest that would be valid on its own. |
| Package import (ADR-0027) | Offline, script-free, atomic promote, and OS no-replace rename that prevents a racing destination from being overwritten. | Import is deliberately offline and non-executing. It has no notion of a worker identity. |
| Watchdog heartbeats (ADR-0036) | Bounded liveness decisions driven by one command and three exit codes. | A heartbeat proves a producer wrote a line recently. It proves neither that a lease is held nor that an action was *not* performed. |
| Workbench instance lease (ADR-0030) | Per-user lease registry; liveness by probing the port and comparing `instance_id`; stale and foreign leases are never claimed and never stopped. | The lease is explicitly "a hint, not the truth" — the right call for local discoverability, and explicitly not a fencing primitive. |

The recurring shape: **today's primitives are unique within one process and durable
within one store.** Remote admission needs an identifier that a different machine can
validate, and a lease that expires without the holder's cooperation.

## Part 2 — Proposed contract

### 2.1 Roles and topology

Two participants, defined by what they own rather than by what they are:

- **Coordinator** — the single writer for one run. It owns the optimizer, the policy
  version counter, the checkpoint generations, and the admission decisions.
- **Worker** — a process that admits itself to a run, receives one scoped capability,
  executes it, and returns results.

**One coordinator per run, not per trial** — confirmed by the design review. The
coordinator is the run's single writer; a per-trial coordinator would need a
leader-election story per trial and would leave `coordinator_epoch` ambiguous for the
ordered identifiers in 2.4. A "cluster" is one coordinator plus zero or more workers;
nothing in this contract requires a scheduler between them, and the deployment chooses how
a worker finds its coordinator.

A **capability** is an enumerated, deny-by-default grant, scoped by
`(project, environment_id, trial_id, package_digest, policy_digest, max_in_flight,
max_payload_bytes, max_duration)`. **Every v1 capability is trial-scoped.** The v1 set is
deliberately small and closed:

- `collect:unroll` — run one unroll and return its result.
- `evaluate:episode` — run one evaluation episode and return its result.
- `propose:checkpoint` — write checkpoint bytes to **per-lease staging only** and return
  their digest. It never writes to the run's checkpoint location; see 2.9.

**Why v1 has no `read:policy`.** The policy a worker executes against is already
delivered through the pinned package bound at admission and through `policy_digest`
(ADR-0027's offline source-package path). A separate read capability therefore grants no
capability that admission does not already convey, while adding a second authorization
story and a read-amplification surface with no caller that needs it. `read:policy` is
deferred, not forbidden: it re-enters only with a concrete caller that the pinned package
cannot serve.

**Why there is no remote learner role.** ADR-0041 §D9 asks for "authenticated learner
and actor roles". This contract authenticates the **actor** side only and keeps the
learner behind the coordinator on one machine, because the coordinator is already the
single writer for the optimizer and the policy version counter. Admitting a remote
learner would turn policy publication into a distributed decision and would need a
second single-writer story that nothing here has earned. The design review confirmed this
as v1's explicit answer to ADR-0041 §D9: **v1 admits the actor role; a remote learner
capability is deferred, not designed away.**

### 2.2 Admission

A worker requests admission with:

| Field | Meaning |
| --- | --- |
| `worker_id` | Deployment-assigned identity. GLR does not mint it. |
| `package_digest` | The package content identity. **Version-independent** (ADR-0041 §D4): SHA-256 over the ordered selection and inventory entries only, with `tool_version` recorded in the manifest but excluded from the hash. See the dependency note below for ADR-0027 as shipped. |
| `policy_digest` | Digest of the policy artifact the worker will execute against (for a checkpoint, its manifest `checkpoint_sha256`). |
| `capabilities` | The capabilities the worker requests. |
| `lease_seconds` | Requested lease duration. |
| `nonce` | Freshness value, echoed in the grant. |

The coordinator grants, or refuses with a machine-readable reason:

| Field | Meaning |
| --- | --- |
| `lease_id` | Opaque, unique per grant. |
| `fence` | Strictly monotonic fencing token, per `(run_id, resource)`. See 2.3. |
| `lease_seconds` | **A duration, not a deadline.** See 2.6. |
| `granted_capabilities` | The intersection of requested and permitted, never a superset. |
| `policy_version` | The version this worker will be measured against. |
| `max_in_flight`, `max_payload_bytes` | Backpressure terms. See 2.5. |

Refusal reasons: `ADMISSION_DIGEST_MISMATCH` (source or policy digest differs from what
the run pinned), `ADMISSION_CAPABILITY_DENIED`, `ADMISSION_CAPACITY_EXCEEDED`,
`ADMISSION_CLOSED` (the run is terminal).

**Binding the two digests is the learner-neutral core of this contract.** GLR never
interprets what a policy means (ADR-0027); it only refuses to let a worker that is
running different source or different weights contribute to a run whose identity was
pinned to specific digests.

**Dependency: the ADR-0041 §D4 migration.** ADR-0027 as shipped computes content
identity over selection, `tool_version`, and inventory. ADR-0041 §D4 changes that to
selection and inventory only, and leaves the migration mechanism open (a new source-only
schema revision, or a parallel identity field with a documented transition). This
contract requires the ADR-0041 §D4 form, because admission is keyed on `package_digest`
and refuses with `ADMISSION_DIGEST_MISMATCH`: a digest that varies with the packaging
CLI version would reject two workers running byte-identical source, which is exactly the
question ADR-0041 §D4 wants the identifier to answer. Until that migration lands,
`package_digest` is whatever `glr.source-package.v1` emits, and mismatch is expected to
over-reject across CLI versions. **Decided: this contract follows §D4 and inherits its
migration decision.** Over-rejecting across CLI versions fails closed, which is the safe
direction, so it is accepted as the migration-period behaviour rather than worked around.
What is not accepted is pinning today's ADR-0027 computation into the contract or into a
test: implementations must obtain `package_digest` through an **injected identity
function** so that the D4 migration changes one seam instead of rewriting the conformance
suite. See Part 4, Phase 1.

### 2.3 Fencing tokens

`fence` is a strictly monotonic integer scoped by `(run_id, coordinator_epoch, resource)`,
issued by the coordinator and carried by every subsequent mutation. The coordinator keeps
`max_fence_seen` per scope **in memory** and rejects any mutation whose `fence` is lower
with `FENCE_STALE`, changing no state and incrementing a counter.

**`coordinator_epoch`** names one coordinator incarnation for a run. It increments on every
coordinator start or restart, and a new epoch allocates fences **from zero**.

Three rules keep the token usable:

- **Renewal preserves the fence.** A worker that renews before expiry keeps working with
  the same `fence`, so a renewal never invalidates its own in-flight attempts.
- **Re-admission issues a strictly greater fence.** Any grant after an expiry or a
  conflict is greater than every fence previously issued for that scope.
- **A fence from an earlier epoch is `EPOCH_STALE`.** A restart therefore needs no
  persisted `max_fence_seen`: monotonicity across restarts comes from the epoch, not from
  a durable counter, so a design that is deliberately free of persistence dependencies
  stays that way. Rejecting an older epoch outright is also the safe answer, because a
  restarted coordinator cannot know what its previous incarnation admitted.

What fencing buys, and what it does not: it protects **coordinator-side state** from a
worker that has lost its lease and does not know it. It does **not** protect
**worker-side effects** that already happened. A worker whose lease expired mid-action
may already have driven the environment. No token undoes that, which is why the state
taxonomy below has a terminal state for "we will never know".

### 2.4 Result ingestion: ordering and idempotency

The local attempt identifier is a UUID and therefore unordered (Part 1). Remote
ingestion needs order, so the coordinator allocates it:

- `attempt_seq` — monotonic per `(run_id, coordinator_epoch)`, issued by the coordinator.
- `attempt_id` — composite identity `(run_id, coordinator_epoch, attempt_seq, attempt_uuid)`,
  retaining a UUID for uniqueness across stores while ordering comes from `attempt_seq`.
- `ingest_id` — `sha256(attempt_id ‖ lease_id ‖ fence ‖ result_digest)`, the idempotency key.

Ingestion rules:

1. A result is **deduplicated on `ingest_id`**. A duplicate delivery returns the first
   recorded outcome and applies nothing. **A duplicate delivery never becomes a second
   optimizer update.**
2. The same `attempt_id` arriving with a *different* `result_digest` is
   `INGEST_CONFLICT`: recorded for inspection, and no optimizer update is applied either.
   Two different results for one attempt is a bug or an attack, never a merge decision.
3. Order of application is decided by the coordinator from `attempt_seq`, never by
   arrival order at the socket.
4. A result arriving with an expired lease, a stale fence, or a superseded epoch is
   `UNOWNED`: recorded as a metric, never applied.
5. Ingest records the **lease state it observed** — `live`, `expired`, or `superseded` —
   alongside the outcome and the digest. **An expiry never rolls back an update that was
   already ingested.** Ingestion is committed at ingest, so a partition wall appears as a
   metric rather than as a silent rewrite of optimizer history, and the recorded lease
   state is what makes that wall auditable after the fact.

### 2.5 Backpressure: reject, never silently drop

The local queue may `drop-oldest` because the coordinator of that queue is the same
process that produced the unroll and can account for the loss. A remote coordinator
cannot. A dropped remote result is indistinguishable from a result whose worker died, so
remote overflow is always **`reject`**: the worker is told to abort the attempt, and the
abort is recorded. Silent remote drops are not a supported policy.

Backpressure terms are `max_admitted_workers`, `max_in_flight` per worker,
`max_payload_bytes`, and `max_result_age`. The coordinator may tighten them at any time;
tightening never revokes a live lease retroactively.

### 2.6 Leases: duration, not deadline

A lease is granted as `lease_seconds` and measured by each side against **its own
monotonic clock from the moment of receipt**, not against an absolute expiry timestamp.
The design therefore does not assume synchronized clocks; wall-clock timestamps appear
only in diagnostics. Expiry is a local decision on each side, and the two sides may
disagree for the duration of one lease — which is exactly why every mutation carries a
fence and why an expired lease never invalidates a result already ingested (rule 4 above
is about *subsequent* accepts only).

### 2.7 Reconciliation after disconnect or reconnect

Reconnection is **re-admission, not resumption**:

1. The worker requests admission again and receives a strictly greater `fence`.
2. It reports the state of every attempt it owned:
   - completed, with digest → idempotent ingest (2.4);
   - **in flight with an unknown outcome → `ORPHANED`**;
   - admitted but not started → returned to the queue.
3. An in-flight attempt is **never resumed in place** and **never replayed because a
   transport reconnected**. ADR-0020's reasoning applies with more force across a
   network: attempt metadata cannot establish whether a game action is safe to repeat.
   A retry is a *new* attempt with a new `attempt_seq`, and only the project or a human
   may decide to create one.

Cancellation follows the same shape: the coordinator requests it, the worker makes a
bounded best-effort attempt to stop and reports `CANCELLED`. If the outcome cannot be
established, the attempt is `ORPHANED`, not `CANCELLED`. An attempt is terminal exactly
once.

### 2.8 State taxonomy, and why it stays additive

ADR-0020's local projection keeps its existing four states. The remote-only outcomes are
coordinator-side and are mirrored into the local store as `RolloutStatus.FAILED` with a
machine-readable `failure_reason`, so nothing that reads `RolloutAttempt` today has to
change. The mapping is one-to-one and lowercase: `ORPHANED` → `orphaned`, `STALE` →
`stale`, `UNOWNED` → `unowned`. `run_store.py` → `TrainingStore.update_rollout_attempt`
already rejects a `FAILED` attempt that carries no reason, so a remote outcome that
loses its reason fails closed instead of landing as an unexplained failure.

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
leaves a silent worker leak with no bound. Both are unacceptable, so the two budgets are
separate:

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
`TerminationReason` is episode-level and is deliberately **not** reused here. In
particular `ENV_INDETERMINATE` ("the environment consequence of an action is unknown, so
restart rather than act again") answers a different question than attempt-level
`ORPHANED` ("this attempt's result never arrived and cannot be reconstructed"). An episode
can end `env_indeterminate` while its attempt ingests normally, and an attempt can be
`ORPHANED` with no episode termination recorded at all. The two must not be collapsed
into a single "we do not know" concept.

Policy lag mirrors the local cutoff with one remote difference: the worker is **told** to
stop producing, because continuing to spend its budget on results that will be rejected
is worse than pausing. A run may explicitly enable carry-over, in which case stale
results are recorded as `STALE` metrics rather than discarded — recorded either way.

### 2.9 Checkpoint ownership

The run's checkpoint location has **exactly one writer on one machine: the coordinator**.
The no-replace plus `fsync` plus `os.replace` behaviour in `checkpoint.py` is a property of
one local filesystem; it does not survive being moved across machines, and pretending
otherwise turns "exactly one proposal wins" into a distributed negotiation. The write path
therefore splits in two:

- **Worker: staging only.** A worker holding `propose:checkpoint` writes its bytes to a
  **per-lease staging area** and returns `checkpoint_sha256`. Staging is addressed per lease,
  is not the run's checkpoint location, and is never resolved as one by any code path.
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

### 2.10 Trust boundary

- **The deployment owns authentication.** GLR defines the claim set and the refusal
  reasons; it does not choose mTLS, tokens, or workload identity, and it ships none.
- **Credentials and machine bindings stay deployment-local.** They are never part of a
  distribution package, and the package path stays offline and script-free (ADR-0027).
  A worker resolves its own credentials from its own deployment.
- **Blast radius, not immunity.** Scoped capabilities, expiring leases, monotonic
  fences, deduplicated ingestion, and unique checkpoint ownership limit what a
  compromised or misbehaving worker can do. They do not make one safe.
- **Ingest logs carry digests, counters, state transitions, and observed lease state —
  never observation data.** Following `ActorQueueMetrics`, the coordinator records digests,
  counters, and state transitions, plus the lease state observed at ingest (2.4 rule 5). It
  never records observations or game content.
- **Ingest-log retention is bounded by the run.** Ingest records live and die with the run
  they belong to: there is no cross-run ingest log and no retention window that outlives the
  run. A deployment that needs a longer audit trail exports from these records on its own
  terms; GLR does not retain them on its behalf.

## Part 3 — Conformance checklist for a future implementation

These are the tests an implementation must pass before it may claim anything about
authenticated multi-machine execution. ADR-0027 stage 5 names six families — duplicate
deliveries, stale ownership, dropped results, lease expiry, policy mismatch, and
checkpoint conflict. Tests 1–8 and 11 below cover those families directly; the rest are
this proposal's additions.

| # | Scenario | Invariant that must hold |
| --- | --- | --- |
| 1 | **Duplicate delivery** — one result, same `attempt_id` and `result_digest`, delivered twice. | The second ingest is a no-op returning the first outcome. Optimizer updates: exactly 1. |
| 2 | **Conflicting delivery** — same `attempt_id`, different `result_digest`. | `INGEST_CONFLICT`; no optimizer update; both payloads retained for inspection. |
| 3 | **Stale ownership** — mutation carrying `fence < max_fence_seen` inside the current epoch, and a mutation carrying a fence from an earlier `coordinator_epoch`. | `FENCE_STALE` in the in-epoch case and `EPOCH_STALE` in the cross-epoch case; state unchanged in both; the matching counter incremented. |
| 4 | **Lease expiry** — worker stops heartbeating with attempts in flight. | Lease expires on the coordinator; in-flight attempts become `ORPHANED`; no automatic retry; the next grant has a strictly greater `fence`. |
| 5 | **Late result after expiry** — result arrives carrying an expired lease's fence. | `UNOWNED`; recorded as a metric; never applied. |
| 6 | **Policy mismatch** — `observed_policy_version` lag exceeds the cutoff. | `STALE`; never applied; counted, and the worker is told to stop. |
| 7 | **Checkpoint conflict** — two workers propose promotion for one `(trial_id, generation)`. | Exactly one wins by no-replace; the loser gets `PROMOTE_CONFLICT`; only the winner's digest is referenced. |
| 8 | **Checkpoint from an expired lease** — bytes arrive after expiry. | Never promoted, whatever their digest. |
| 9 | **Reconnect storm** — one worker reconnects N times. | At most one live lease per `worker_id`; every superseded fence is rejected. |
| 10 | **Cancellation in flight** — cancel an admitted attempt. | Terminal exactly once: either `CANCELLED` or `ORPHANED`, never both, never neither. |
| 11 | **Backpressure overflow** — exceeds `max_in_flight` / `max_payload_bytes`. | `reject` plus a recorded abort. No silent drop. |
| 12 | **Clock skew** — worker monotonic-vs-wall-clock offset of ±1 hour. | Duration-based leases still behave; no premature expiry, no extended window. |
| 13 | **Package hygiene** — export a selection containing credential- or secret-shaped paths. | Rejected at export; no credential material in the archive. |
| 14 | **No replay of unknown outcomes** — an `ORPHANED` attempt exists. | The environment is not re-driven for it automatically; a retry is a new `attempt_seq` created by the project or a human. |
| 15 | **Determinism** — the whole suite runs in one process with an injected clock and fault injection. | Tests 1–12, 16, and 17 pass without a socket, so the concurrency invariants are provable in ordinary CI. |
| 16 | **Orphan budget breach** — `ORPHANED` outcomes exceed `orphan_budget`. | Admission of new workers stops and a run-level alarm is raised. The run is **not** terminated, and `attempt_budget` is untouched by the `ORPHANED` count. |
| 17 | **Staging is not a checkpoint** — a worker stages checkpoint bytes and its lease then expires. | The run's checkpoint location is unchanged, the staged bytes are labelled unowned and retained for inspection, and no code path resolves them as the run's checkpoint. |

Test 15 is the reason this proposal is not simply "unprovable in CI". The *concurrency
contract* is designed to be provable in-process; only the *authentication and transport*
layers need real infrastructure, and only those layers' claims stay out of CI's scope.

## Part 4 — Phased delivery

| Phase | Content | Gate |
| --- | --- | --- |
| 0 | This document: contract, state taxonomy, conformance checklist. No code. | Human design review. **Closed 2026-09-28**; promoted to [ADR-0045](../decisions/0045-remote-role-admission.md). |
| 1 | In-process reference harness with an injected clock and fault injection, implementing the coordinator decision logic against the existing `BoundedActorQueue`. No transport, no authentication. `package_digest` is obtained through an **injected identity function**, never a copy of the ADR-0027 computation the harness happens to run against today. | Conformance tests 1–12, 16, and 17 green in CI. |
| 2 | Optional, feature-gated adapter crate (default off). Deployment-owned authentication. No change to the core queue, attempt, or store contracts. | The Phase 1 suite runs unchanged against the adapter. |
| 3 | Attended multi-machine conformance run over tests 1–8 and 13, with published results. | Results published **before** any claim of authenticated multi-machine execution. |

This document was accepted on 2026-09-28 and promoted to
[ADR-0045](../decisions/0045-remote-role-admission.md), which is the normative record.
Phase 1 may begin now; it was not permitted to begin before the promotion landed.

## Explicitly not claimed

- Exactly-once execution. Exactly-once *application to the optimizer* is claimed, and
  only under the dedupe rules in 2.4.
- Linearizability, serializability, or any ordered guarantee beyond `attempt_seq` per
  coordinator epoch.
- Any bounded recovery time, or any guarantee that a run makes progress during a
  partition.
- That a lease holder has stopped working. A lease expiring is a coordinator-side
  decision; it is not knowledge about the worker.
- That any action is safe to replay. `ORPHANED` exists precisely because this cannot be
  established.
- Agreement between machine clocks. Leases are durations.
- Safety against a compromised worker credential, or against a malicious coordinator.

## Design review rulings (2026-09-28)

All eight open questions were answered by the design review. The rulings are normative in
[ADR-0045](../decisions/0045-remote-role-admission.md); this table records the decision and
where it landed there.

| # | Question | Ruling | ADR-0045 |
| --- | --- | --- | --- |
| 1 | Checkpoint writing | **Coordinator-only persistence.** The run's checkpoint location has one writer on one machine; `propose:checkpoint` survives but is narrowed to "write per-lease staging, return the digest". | D10 |
| 2 | Does `ORPHANED` count against the budget? | **Two budgets.** Not in `attempt_budget` (training); counted in a separate bounded `orphan_budget` (orchestration) whose breach stops admission and alarms without terminating the run. | D9 |
| 3 | One coordinator per run or per trial? | **Per run** (confirmed), with fencing scoped to `(run_id, coordinator_epoch, resource)`; a fence from an earlier epoch is `EPOCH_STALE`. | D2, D4 |
| 4 | Does an expired lease roll back ingested updates? | **No rollback** (confirmed); ingest records the lease state it observed so the partition wall is auditable. | D5 |
| 5 | Minimum capability set | **v1 drops `read:policy`.** `collect:unroll` + `evaluate:episode` + `propose:checkpoint`, all trial-scoped. | D2 |
| 6 | Ingest-log content and retention | Digests, counters, state transitions, and observed lease state; no observation data. **Retention bounded by the run.** | D11 |
| 7 | Remote learner role | **v1 actor only** (confirmed) — the explicit scope answer to ADR-0041 §D9. | D2 |
| 8 | `package_digest` and the ADR-0041 §D4 migration | **Follow §D4** and inherit its migration decision; Phase 1 must read identity through an injected identity function. | D3, Phased delivery |

## Acceptance

- The document cleanly separates implemented behavior (ADR-0020 local leases and queue,
  Part 1 of this document) from proposed behavior (everything in Part 2).
- The conformance checklist in Part 3 covers duplicate delivery, stale ownership, lease
  expiry, policy mismatch, and checkpoint conflict, and each entry names the invariant
  rather than the mechanism.
- The limits of the local queue and attempt primitives are stated explicitly rather than
  left for an implementer to discover (Part 1, third column).
- Human design review is required. **Closed 2026-09-28** — see "Design review rulings";
  the accepted contract is [ADR-0045](../decisions/0045-remote-role-admission.md).
