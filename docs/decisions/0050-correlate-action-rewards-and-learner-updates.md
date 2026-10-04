# ADR-0050: Correlate action rewards and learner updates

Status: Accepted

## Context

A reward source declaration proves which adapter may emit a signal. It does
not identify the action that caused the signal. Independently collected
timestamps, changed state, or process output cannot establish that link.
`ActionReceipt.step_id` names the authoritative post-state, while
`Transition.step_id` names the state before the action. Confusing these
identities rejects valid intervals or credits another action.

An accepted action can have no observed effect. Positive signal clipping and
negative weights can also create positive shaping even when the raw signal
is nonpositive. A budget limits the size of this error without correcting it.

## Decision

Add an opt-in strict collection path using `CorrelatedRewardGuard`. The owner
freezes run, environment, protocol, target and configuration identities in
`CorrelationPolicy`. The adapter emits `ObservationContext` for the exact
producer state and `RewardAttribution` for each positive reward contribution.
These are bounded, passive data; they grant no action or observation authority.

The runtime validates a live gameplay pre-state, a fresh post-state in the
same episode, post-step equal to pre-step plus one, producer sequences, target
and action timestamps. The existing action receipt must identify that exact
interval. Unknown and partial outcomes fail closed. A dead post-state can
close a terminal failure; it cannot claim positive outcome credit. Loading,
unknown lifecycle and reset boundaries cannot supply training transitions.

Positive contributions are checked after per-term clipping and weighting. Each
requires an accepted action, a live post-state, and an adapter-confirmed
effect for that signal and interval, including outcome and unbudgeted terms.
This conservatively rejects an unverified positive contribution even if a
later budget would suppress it. Terminal failure penalties are retained.
The existing `EpisodeRewardGuard` still enforces budgets and failure dominance.
Validation and a side-effect-free reward preview precede budget consumption.
Duplicate actions, skipped steps, reused episodes and exhausted collection
budgets are rejected. Consecutive intervals must reuse the exact previous
post-context and observation hash as their next pre-state. The strict path
verifies the observed scalar reward
against the composed reward; it does not silently replace a reward formula.

`SyncCollector` checks pre-action evidence before calling a policy or issuing
an action. It validates post-action evidence before recording a transition.
A failed interval closes the episode and requires a fresh start. Validated
receipts bind hashes of the actual observation, chosen action and next
observation. Hashing rejects object/structured arrays and has explicit size
and depth bounds. The transition and telemetry carry projected provenance,
without raw observations.

Strict collection requires `on_error="raise"`. An indeterminate outcome raises
even if earlier transitions were collected in the same call. It never returns
an artificial truncation whose reward receipt describes a different terminal
boundary. Default collection retains its existing partial behavior.

Observed rewards must be finite scalar float32 or float64 values equal to the
composition rounded to that declared dtype. Receipts retain both the composed
value and the actual observed value. TD checks use the actual learner reward,
so a float32 transport value is not silently replaced by a float64 formula.
Strict ports accept the base data contracts, preventing subclass overrides
from replacing identity validators or exporting extra fields.

Collector dispatch admission is separate from reward acceptance. Reset/attach
attempts are charged before adapter calls; action attempts are checked before
policy invocation and charged before dispatch. Failed attempts are not refunded.
Evidence-only direct composition does not launch or authorize an operation.

`ScalarLearningUpdate` reports scalar consumer operands. A frozen
`LearningConsumerPolicy` binds the learner, table and policy version. Runtime
telemetry checks operand and reward receipt hashes, TD arithmetic and terminal
bootstrap behavior. These events remain diagnostics: arithmetic validation
does not prove an external table was read or written. An uninstrumented
learner supplies no evidence of a learning update.

Fixed replay evaluation uses its independently frozen reward contract and
the captured source interval, through the same strict guard. Missing source
context or a missing frozen reward contract is unknown coverage and cannot
satisfy promotion checks. Candidate code cannot rewrite the evaluation suite
or derive evaluation authority from its own training metrics.

## Compatibility and consequences

Existing collectors retain their behavior unless the strict guard is supplied.
The protobuf transport, provider SDKs, transition wire schema and run store
version stay unchanged. Adapters explicitly opt in by supplying complete
data through `glr.observation-context` and `glr.reward-evidence`. A strict
collector requires its adapter's configuration snapshot to match the frozen
digest. It does not fabricate missing lifecycle fields from a phase label.

Game semantics remain in adapters: observation, legal actions, reset, rules,
effect measurement and success judgment. GLR owns identity validation,
composition, budget enforcement and evidence persistence. Adapter declarations
remain a trust boundary; a data receipt does not independently authenticate
a game. Fixed offline tests establish contract behavior, not gameplay quality.

Collection, evaluation, review and recoverable installation are seams for
a supervised continuous learning campaign. This decision installs no service,
scheduler, GPU workload, device permission or unattended game controller.
