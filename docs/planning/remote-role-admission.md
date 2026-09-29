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

**Superseded.** This proposal was accepted, with the eight open questions resolved, as
[ADR-0045: Admit remote roles with scoped capabilities, epoch-scoped fences, and
coordinator-owned checkpoints](../decisions/0045-remote-role-admission.md).

ADR-0045 is normative and carries the contract this part proposed: roles and topology
(D2), admission claims and refusal reasons (D3), epoch-scoped fencing tokens (D4),
result ingestion and idempotency (D5), backpressure (D6), leases as durations (D7),
reconciliation after disconnect (D8), the state taxonomy (D9), checkpoint ownership
(D10), and the trust boundary (D11).

The proposal text that used to live here is retained in this document's history for its
baseline survey. Do not implement from it: its §2.2 `fence` row and its Part 3 test 5
predate the design review and disagree with ADR-0045 on the fence scope and on what a
late result records. The design review rulings are tabulated at the end of this document,
and the conformance checklist in Part 3 has its normative counterpart in ADR-0045.

## Part 3 — Conformance checklist for a future implementation

These are the tests an implementation must pass before it may claim anything about
authenticated multi-machine execution. ADR-0027 stage 5 names six families — duplicate
deliveries, stale ownership, dropped results, lease expiry, policy mismatch, and
checkpoint conflict. Tests 1–8 and 11 below cover those families directly; the rest are
this proposal's additions.

**ADR-0045 carries the normative copy of this checklist**, with the design review
rulings applied — including test 5, which now records the lease state observed at
ingest. Where the two differ, ADR-0045 wins.

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
