# ADR-0049: Persist bounded learning campaigns and require host review

Status: Proposed implementation; maintainer acceptance pending.

## Context

A goal-driven agent needs to discover rules, propose interfaces, run bounded
experiments and retain evidence across restarts. A successful process or a
large historical score cannot establish that a new candidate is safe to use.
The existing goal-run checkpoint path could install from trainer metrics
before fixed evaluation and required capture checks. A metric value alone
does not bind candidate bytes to a supervisor and independent review.

## Decision

Keep shared admission, evidence, evaluation and promotion in GLR. Game
adapters own observation, legal actions, reset or attach, rule entry points
and authoritative success evidence. Learners and agents may propose passive
knowledge, policy and interface artifacts. Proposals do not execute code or
expand an adapter's action authority.

The Python SDK adds a persistent campaign kernel, revision-bound knowledge
and fixed offline replay evaluation. Admission requires a nonempty target,
exact environment/protocol/configuration, source revisions, artifact bytes,
declared actions, resource leases and finite goal budgets. Missing bindings
remain unknown. A resource lease coordinates participants using this store;
it does not replace the adapter's actual input mutex.

A protected host issues opaque capabilities for evaluation, supervision and
independent review. Capabilities bind the store instance, campaign, trial,
role and principal. Durable ownership is verified on each privileged call;
an existing unowned ledger cannot silently acquire a new authority. A typed
stop receipt must agree with persisted terminal worker evidence. Bare hashes
or worker-supplied stopped flags cannot release an unknown worker's lease.

Fixed evaluation binds the candidate, source, target, configuration, suite
and evaluator. Rust requires a separate `glr.checkpoint-evaluation.v1` report
with explicit measured coverage for all seven checks and the evidence bundle. All authoritative measurements of a fixed metric/source must
agree. Seven learning correctness counters must be measured and zero for a
policy evaluation. Explicit exclusions are permitted only in an inert
reference replay; they remain recorded exclusions, not measured successes.
Approval rechecks the terminal ledger, artifact and incumbent before moving
an artifact reference. Fixed external policy evaluation may update a logical
checkpoint digest and score. Python approval does not install a checkpoint
file, and inert replay cannot approve a policy reference.

The Rust control plane stages legacy goal candidates for review and closes
the old direct promotion method. Its positive checkpoint path is a protected
`PromotionHost`: an owner explicitly provisions an empty ledger; reopening
cannot bootstrap authority into an existing populated store. The host launches and retains opaque handles for the complete
declared direct workers, pins evaluator inputs, observes their terminal
states, records final evaluation, obtains independent review and consumes a
durable single-use authorization. Installation uses a recoverable journal
bound to the store epoch, authority, target, configuration and authorization.
Recovery refuses ambiguous, changed or incomplete evidence rather than
reconstructing approval from a high score or a new process.

The existing Python run-store schema remains version 2, including run state,
rollout attempts, termination evidence and configuration digests. The native
store retains its version-1 writer and version-2 readability without lowering
a database version. Old NULL records remain readable but cannot be used as
configuration-bound promotion evidence. CampaignStore uses its own version-1
database; its schema is not a replacement for TrainingStore. The Python artifact-reference flow
and Rust checkpoint flow have separate authorization contracts. Shared
storage readability does not imply a cross-language approval adapter.

## Consequences and limits

Old callers must obtain host review instead of directly installing a goal
candidate. Existing training and checkpoint files need no automatic rewrite.
Unknown stops retain their leases. Rejected candidates preserve the current
reference or checkpoint and keep an audit trail.

Host authority is an in-process trust boundary, not an OS sandbox, an account
authentication system or protection against another process that can write
the same files. An external supervisor must protect authority material and
observe actual process handles. Rust direct-child supervision does not prove
that every descendant or native game controller has stopped.

This implementation supplies bounded passes and restartable evidence. It
does not start a continuous service, authorize new game actions or establish
24-hour availability. Deterministic crash tests verify application recovery
boundaries; they do not establish power-loss durability on every filesystem.

See [the operator guide](../guides/continuous-learning.md).

Sudden host termination may leave child processes alive. Persisted unresolved
launches block new dispatch to the same target across run IDs and restarts.
The kernel cannot reconstruct an owned handle or infer stop from a PID; an
external protected supervisor must verify cleanup. Reattachment and automatic
release of unknown reservations are not implemented; coherent repair or
migration needs a separate owner review.

FixedReplaySuite freezes evaluator-owned reward training and safety contracts,
source run identity and strict adapter evidence independently of a learner
configuration. A captured epoch may appear only once within its source run;
a fresh logical replay UUID does not prove source reset freshness. Replay consumes the original captured action and contexts;
missing contracts, context or lifecycle measurements remain unknown. The
correlated reward boundary is defined in ADR-0050 and does not change the
provider wire format. Source and suite identities are checked before an
admitted callback; candidate artifacts remain inert. The evaluator fingerprints
its strict reward implementation dependencies, and verifies individual composed
reward terms instead of accepting an equal scalar sum.
