# ADR-0051: Record safe decision evidence and read-only timelines

Status: Accepted

## Context

A launcher needs a shared way to show an observed decision, its execution,
reward evidence and reported learner update. Existing dynamic decision events
can include command parameters and adapter state. Forwarding those payloads
to a view can expose private data. Joining neighboring events by timestamp or
step can also attribute a reward or update to the wrong action.

The tensor policy contract returns an action. It does not expose candidate
scores, confidence, rule use or reasons. A receipt, changed policy digest or
successful process cannot fill those gaps. Observation, legal actions,
effect measurement and success judgment remain adapter responsibilities.

## Decision

Add passive `DecisionEvidence` using `glr.decision-evidence.v1` and a bounded
read-only projection using `glr.decision-timeline.v1`. Store decision evidence
as `agent.decision.evidence` in the existing schema-2 events table. This adds
no database migration, provider wire envelope or execution authority.

`capture_transition` hashes the captured observation and action, and projects
the actual action receipt. With a typed reward receipt it also verifies the
next-state hash. Its pre-step is the transition step; the receipt identifies
the next step. An optional typed reward receipt must
match the run, environment, protocol, configuration, target, episode, steps,
action interval, tensor hashes and scalar reward. Terminal consistency uses
the existing all-participant `Transition.done` contract.

Missing facts remain null or an explicit `unknown` state. Producers may
report bounded candidates, legality, finite scores and short basis codes at
selection time. The runtime does not infer them from a tensor action or
reconstruct reasons from later rewards. Training/evaluation mode and policy
or checkpoint identity are supplied explicitly. Raw frames, observations,
command arguments, incidental metadata and private reasoning are excluded.

Rule evidence reuses `RuleIndexBinding` and `DecisionConsumptionReceipt`.
`binding_checked` means that the supplied identities and revisions agree;
it does not mean that a rule was consumed. A retrieved receipt is not a used
receipt. A used receipt remains a consumer declaration linked to its evidence
hash, not runtime proof of an actual table read or useful effect. Missing
consumer instrumentation stays unknown, and an empty receipt list proves no
rule use.
The new safe projection restricts rule revisions to portable version labels
without paths, drive prefixes or free text. The existing knowledge-binding
contract is unchanged.

`Telemetry.decision_evidence` checks the durable run and configuration, then
persists locally. It stays silent even when the telemetry object's console
option is enabled. Existing raw decision and execution events are not
automatically converted. Producers choose when to emit this safe record.

`read_timeline` opens an existing regular database with SQLite `mode=ro` and
`query_only`; it does not construct a writable `TrainingStore`. Pages scan at
most 250 events and advance their cursor across skipped event kinds. Evidence
is bounded to 64 KiB per record. `project_events` links reported learner
updates only when run, episode, step, action, producer interval, reward receipt
hash and all three tensor hashes agree. Neighboring events remain separate.
Links outside the supplied bounded window remain unlinked.
Non-text, deeply nested or out-of-range payloads produce bounded invalid-data
warnings while subsequent valid events and the scanned cursor are retained.
Database/schema errors are not suppressed.

The timeline marks learner links as `reported_binding_checked`. It checks
record consistency; it does not read or mutate a learner table, rerun TD
arithmetic, authenticate game effects or establish policy improvement.
Protected fixed evaluation and review remain separate authority boundaries.
An optional comparison requires complete fixed evaluator/suite identities,
paired scores, direction, a positive step budget and a nonnegative improvement
threshold. A tie is not `candidate_better`. This diagnostic status neither
changes the incumbent nor authorizes installation.

## Integration boundary

The opt-in strict collector already emits `reward.correlated` when a store
and run are bound. It returns a projected reward mapping and hash in transition
provenance, not a typed reward receipt. Producers must not compose the same
interval again to obtain another receipt. A custom producer can retain its
first typed composition result and pass it to evidence capture.

Decision evidence, scalar learner update emission and timeline queries are
explicit APIs. This decision adds no collector auto-emission, HTTP endpoint,
dashboard or launcher UI. Adapter-owned launchers can consume the same safe
timeline shape while keeping game semantics and local data policy in their
adapters. Local diagnostic logs are not uploaded or published by this API.

## Consequences

- Adapters share one diagnostic format without copying runtime logic.
- Missing policy internals stay visible as unknown rather than plausible reasons.
- Legacy events retain their existing behavior and remain unverified projections.
- Sequence progress does not certify wall-clock freshness or OS authentication.
- A timeline record grants no action, device, evaluator or promotion capability.
- Integration still needs explicit producer instrumentation and adapter-owned UI.

See the [producer and consumer guide](../guides/decision-evidence-timeline.md),
[ADR-0025](0025-record-learner-owned-decisions.md) and
[ADR-0050](0050-correlate-action-rewards-and-learner-updates.md).
