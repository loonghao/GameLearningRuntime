# ADR-0053: Admit measured fleet data with local owner proof

Status: Proposed (bounded local implementation; draft review pending)

## Context

The first local fleet profile validates numeric shards and their declared
source identities. It deliberately carries no action receipt or incidental
metadata and quarantines non-simulated sources. That conservative profile
cannot represent a complete measured transition from the existing strict
collector. Removing its real-source gate would admit incomplete recordings;
keeping that gate as the final design would prevent legitimate measured data
from reaching a learner.

The SDK already defines `Transition`, `Unroll`, `ActionReceipt`,
`ObservationContext`, `CorrelatedRewardGuard` and reward-safety budgets. These
contracts establish data relationships when a producer supplies complete
facts. They do not authenticate a physical game effect. Generic transition
serialization also performs some type conversions, so it is not a substitute
for strict admission of untrusted records.

## Decision

Add an opt-in, detached, closed `glr.fleet.measured.v1` proof profile alongside
the existing `glr.fleet.shard.v1` numeric carrier. Preserve the old profile and
its defaults. A complete measured proof can make a non-simulated shard
structurally ready; actual learner selection additionally requires a fixed
holdout and an explicit, finite local owner enablement. An unconfigured owner,
unknown key, incomplete capture or missing enablement remains closed.

The profile belongs to the shared SDK's dataset boundary. It introduces no
provider action vocabulary, new game transport, device authority, external
HTTP ingress, key-provisioning service or continuous collector. Adapters keep
ownership of legal actions, original observations, resets, action encoding and
actual game-effect claims. Learners keep ownership of algorithms and parameter
writes; independent evaluation and checkpoint promotion remain separate.

### Preserve typed evidence beside the numeric carrier

A numeric carrier must not be described as the complete producer record. The
signed detached body retains the original typed evidence needed to reconstruct
the admitted `Transition`/`Unroll`: action receipt, two observation contexts,
raw reward inputs, attributions, observed scalar and full reward-result state.
The verifier reconstructs the complete typed data instead of treating a
metadata-stripped projection as evidence of causality. SDK serialization parity
tests must cover this complete round trip, including receipt fields, original
array dtypes, flags, timestamps and permitted metadata.

Reuse the strict collector's identity convention:

- `Transition.step_id` identifies the pre-state.
- `ActionReceipt.step_id` identifies the post-state, one greater than the
  pre-state step.
- `Transition.timestamp_ns` is the post-state timestamp.

Bind the actual observation, chosen action and next-observation tensor hashes
to that interval. Require complete original capture fields; do not generate
missing episode identifiers, infer timestamps from file order or upgrade a
sequence added by dispatch context into an original signed producer fact.
Validate native types, closed fields, shapes, dtypes, dimension products and
encoded byte limits before using the existing serializer. No object arrays,
raw frames, private logs, command arguments or hidden reasoning enter this
profile. Any allowed extra metadata is explicitly owner-scoped and bounded;
its privacy and meaning still require producer review.

An observed reward must be a finite scalar with its original float32/float64
dtype and must match the composition under that dtype exactly. Preserve an
explicit measured observed reward: equality of `reward` and `composed_reward`
in an older diagnostic mapping cannot establish that the scalar was checked.
Positive contributions require an accepted, consumed action, a live settled
post-state and a matching adapter-confirmed attribution. An effect claim is
still a producer claim, not independent physical authentication. Legitimate
terminal failure penalties remain valid; terminal transitions cannot bootstrap.

Preserve reward-safety configuration and explicit before/after budget state.
The older correlated receipt hash omits cumulative episode totals and shaping
suppression. Admission therefore validates a persistent episode tail across
shards, restarts and purges rather than resetting a reward guard per shard.
Verification does not compose the producer's reward again to create another
receipt or consume the producer's live budget twice.

### Local source grants and trust

A caller supplies an in-memory measured authority with explicitly registered
source grants and opaque key identifiers. Detached HMAC signs the canonical,
version-domain-separated body. Keys are caller-owned RAM values; there is no
automatic setup, persisted default production key or key material in public
representations, snapshots or dataset records.

Each grant binds the complete source declaration and the exporter revision,
target, clock domain, full training and reward-safety configuration, action
encoding and legal-mask mapping, quality claims, semantic evaluation domain,
evidence kind, age/skew limits, expiry and per-episode action limit. Source
identity includes the actual sampler/runtime, adapter, effective configuration,
behavior policy, run, source epoch, specifications, split and assignment.
The exporter revision must not replace the sampling runtime revision. Repeated
package version strings are not a source allowlist.

The trust scope is **authorized local source claims and byte integrity**.
An HMAC-valid record means the configured signer approved those bound claims.
A payload hash alone, a key proposed by the uploaded record, or a matching
package version establishes no source authority. This design does not prove a
physical game effect, resist a malicious local owner or grant a denied host
permission. It does not extend a Python approval into native or remote device
authority. Owner registry, grant, key, destination and expiry changes are
rechecked at admission and consumption boundaries.

Use separate evidence kinds for a `synthetic_contract_fixture` and an
`owner_authorized_local_measured` capture. A non-simulated fixture is useful for
testing the real-admission branch, but neither its flag nor its signature makes
it a real game observation.

### Fixed holdout, compatibility and consumption

Use an owner-registered semantic evaluation domain that survives source,
runtime, adapter and behavior-policy relabelling. Within that fixed domain,
reserve exact numeric input fingerprints for evaluation and reject their use
as training data, even under a new source label or episode identifier. Keep
unrelated game domains separate. This is exact-copy isolation; it does not
infer semantic equivalence of dtype transforms or near-duplicate samples.

A measured holdout must consist of complete admitted records and be frozen
against an explicit `MeasuredEvaluationSuite`. The suite retains bounded
evaluator artifact bytes, its source revision, exact case/shard/payload/proof
identities and fixed metric definitions with required sample counts. The hub
computes its content hash and keeps the immutable suite and artifact with the
snapshot; it does not execute the evaluator. Freeze this definition and data
independently from learner-configured training metrics. Binding a suite or
snapshot is not itself an external evaluation run or a measured game outcome.
Missing external results remain unknown.

An explicit `RealTrainingEnablement` binds the local destination, current
measured authority, permitted exact source specifications, frozen evaluation
identity/suite/snapshot, evidence kind, owner approval identity/hash, expiry,
maximum transitions and maximum callback attempts. Grant, suite and enablement
evidence kinds must agree. It is supplied
by the owner; constructing a matching configuration does not establish that a
production operator granted it. Existing on-policy behavior-artifact equality
and explicit off-policy algorithm/source/behavior allowlists remain required.
No automatic weighting or inference of missing algorithm statistics is added.

The consumer supplies the reconstructed typed `Unroll` through the existing
bounded queue and callback contract. Recheck authority, proof, destination,
source, enablement, expiry, holdout and persistent lifecycle before the claim,
before callback dispatch and after callback return. One plan still selects one
`Unroll`; before the claim, atomically reserve its actual transition count and
one callback attempt against the permit. An approval identity/hash binds one
immutable permit; reopening, changing learners or modifying its limits cannot
reset its spending. Failure, unknown effect and crashes do not refund these
reservations. Grant action/reward limits and finite hub quotas also apply.
These limits do not bound opaque optimizer internals. Revocation, mismatch or
unknown callback effect cannot silently make data available for a retry.

A consumption receipt records observed callback completion and its declared
update count. Queue commit does not prove parameter writes. A learner must
provide its own actual update/artifact evidence, and a fixed external evaluator
must independently establish benefit over the preserved baseline before any
separate promotion authority installs a candidate.

## Validation and scope

The first positive mechanism uses complete, explicitly synthetic non-simulated
measured fixtures, caller-supplied local grants and explicit enablement. It must
exercise proof verification, full typed serialization parity, frozen holdout,
compatible selection and bounded callbacks under a finite enablement.
Adversarial checks cover
signature/source substitution, incomplete capture, observed-scalar drift,
foreign identity, stale clocks, legal masks, episode budget reset, exact holdout
relabel, expiry/revocation and unknown callback effect. Installed-wheel
acceptance uses the exact source and artifact outside the source checkout.

True producer capture, actual loaded versions, legal host access, key setup,
remote authentication, central collection destinations and persistent services
remain integration work. Historical incomplete recordings can validate
rejection; they cannot supply a positive measured capture by reconstruction.
This decision authorizes neither live training nor game deployment.

## Consequences

Complete authorized measured data has a path to eligibility rather than a
permanent hardcoded real-source denial. The extra proof, grants, persistent
lifecycle and holdout state add bounded storage and validation cost. Strict
capture completeness and exact-copy isolation may conservatively reject useful
records; the runtime reports those limits instead of filling missing facts or
claiming approximate duplicate detection.

The existing numeric profile stays usable for its original purpose. Generic
serialization and diagnostics remain separate from source authority, physical
effect truth, model-update evidence, external evaluation and promotion.

See the [measured admission guide](../guides/measured-fleet-admission.md) and
[ADR-0052](0052-local-fleet-training-data-hub.md) for the underlying local hub.
