# Measured fleet admission with a local owner

Use this opt-in path when an authorized producer already has complete measured
`Transition`/`Unroll` records. It verifies a bounded proof and makes compatible
non-simulated data available to an explicitly enabled learner. It does not start
a collector, create keys, obtain host permissions, upgrade a running adapter or
approve a model for deployment.

The original numeric profile is unchanged. A plain non-simulated
`glr.fleet.shard.v1` shard remains quarantined. Complete
`glr.fleet.measured.v1` evidence adds an admission route; it does not turn an
incomplete log into a training record.

## Preserve identities and original capture

Record the exact sampling runtime, adapter, effective configuration, vector
specifications, reward contract, behavior policy, source epoch and run.
`SourceSpec.runtime_source_commit` is the sampling runtime revision, even when
an exporter or receiving SDK uses a newer build. Keep exporter and receiver
build identities separate. Package metadata alone is insufficient: two builds
can have the same version string and different interfaces.

An adapter must supply the original action receipt, both observation contexts,
legal masks and action encoding, reward inputs and attributions, observed scalar
and full reward-budget state. Preserve the first actual composition receipt;
do not compose again to manufacture another receipt. The shared verifier owns
relationships and bounds; the adapter still owns the truth of its observed
facts and confirmed effects.

The collector convention is precise: a transition uses the pre-state step,
its action receipt uses the post-state step (`pre + 1`), and the transition
stores the post-state timestamp. A trace line number, turn counter, later
screenshot or file modification time cannot fill these fields. Default episode
UUIDs and timestamps are not original producer evidence. Unknown lifecycle,
clock mapping, legality, behavior policy or reward attribution stays unknown.

`require_measured_capture(mapping)` checks capture completeness before admission.
It is not a source permit. The full verifier checks native types and bounds
before the general SDK record decoder can coerce values. This includes closed
fields, dtype/shape/size, finite scalar float32/float64 reward, masks, terminal
flags, timestamp and producer-sequence identity. The observed reward must match
the composition under its actual dtype exactly; an unmeasured receipt or a
similar numeric value is insufficient.

The numeric carrier and detached signed body serve different purposes. The
carrier preserves the existing shard transport and numeric layout. The body
retains the complete typed evidence so that verification reconstructs the
original admitted record, including permitted metadata. A stripped carrier is
not proof that an original receipt was absent or that causality was valid.
Do not upload raw frames, private logs, paths, command arguments, secrets or
hidden reasoning. Review the meaning of permitted numeric features and extras;
a vector encoding does not make private content safe.

## Run the installed synthetic example

In a normal installed SDK environment, replace the source-commit placeholder
with the exact 40-character lowercase build source revision. Choose a fresh
owned output directory; the example refuses to reuse one. It does not install
anything into a running adapter.

```shell
python -m game_learning_runtime.examples.measured_fleet_training --output-dir ./measured-fleet-example --runtime-source-commit "<exact-40-hex-build-source>"
```

The example constructs four complete numeric fixture sources with
`simulated=False` and `evidence_kind="synthetic_contract_fixture"`.
`actual_game_capture` remains false. The fixture caller supplies its test-only
in-memory authority; this command is not production key setup or a source grant
for a game. The default consumer performs zero callbacks. Explicit test-only
enablement permits two bounded callbacks and six actual NumPy reward-regression
parameter updates. Duplicate data causes no extra callback. Fixed holdout and
incompatible-game inputs are excluded from training.

The output contains:

- `demo-summary.json`, with fixture provenance, default/explicit callback
  counts, duplicate handling and actual before/after parameter hashes;
- `learner-model.npy`, `learner-recipe.json` and `learner-bundle/`, using
  `glr.model-bundle.v1` to bind the example code, selected manifests/chunks,
  measured proofs and test-only approval inputs to the saved artifact;
- `fixed-evaluation.json` and `fixed-evaluator-definition.json`, retaining the
  immutable suite, evaluator artifact and exact case/data/proof references;
- `test-only-owner-approval.json` and `test-only-enablement.json`, explicitly
  scoped to synthetic fixtures, plus `.glr/fleet/launcher-snapshot.json`.

The evaluator definition is retained, not executed; the external result is
`not_run`. These local parameter updates demonstrate the admission and
consumption path. They do not prove game learning benefit, a connected machine,
a production approval or a policy promotion. The example starts no game,
service or external collector.

## Owner configuration

The relevant APIs belong to these installed SDK modules:

| Module | API |
| --- | --- |
| `fleet_measured` | `MeasuredGrant`, `MeasuredAuthority`, `MeasuredActionBinding`, `MeasuredQuality`, `LegalActionEvidence`, `RewardBudgetState`, `MeasuredStep`; `encode_measured_shard`, `verify_measured_manifest`, `verify_measured`, `require_measured_capture` |
| `fleet_datahub` | `MeasuredDestination`, `MeasuredEvaluationCase`, `MeasuredEvaluationMetric`, `MeasuredEvaluationSuite`, `MeasuredEvaluationSnapshot`; `FleetHub.create/open`, `configure_measured`, `begin_measured_upload`, `ingest_measured`, `freeze_measured_evaluation` |
| `fleet_learner` | `RealTrainingEnablement`, `FleetConsumer`, `LearnerSelection`, `LearnerResult`; `configure_real`, `plan`, `consume_one` |

A `MeasuredGrant` fixes:

| Binding | Required owner-supplied information |
| --- | --- |
| Source | The complete `SourceSpec`, grant identifier and opaque key identifier; actual sampler/runtime, adapter, configuration, policy, run/epoch, schemas, split and assignment remain in that source. |
| Export and execution | Exporter source hash, target identity and explicit action-to-mask/command bindings. |
| Reward and quality | Full `TrainingConfig` and `RewardSafetyConfig`; original complete per-step quality/legal evidence and before/after budget state. |
| Evaluation domain | An owner-registered semantic domain that survives source/runtime/adapter/policy label changes; its fixed game/environment/layout semantics cannot be silently repinned. |
| Time and limits | `clock_domain="unix-utc-ns"`, maximum age/skew, expiry and maximum actions per episode. A monotonic clock requires an actual established mapping before producing this data. |
| Provenance kind | Exactly `synthetic_contract_fixture` or `owner_authorized_local_measured`. |
| Extra data | Explicit bounded info/provenance keys and event names; empty allowances are the default. |

`MeasuredAuthority` receives caller-owned in-memory grants and keys. Keys must
have at least 32 bytes and remain in RAM. An opaque key identifier is public
metadata, not the secret. There is no default production key, automatic key
setup or persistence of key material in the hub. A record cannot nominate its
own trusted key. Configure authority and destination explicitly on
`FleetHub.create/open`, or with `configure_measured`; the default is `None`.
After reopening, the caller supplies its authority again.

Detached HMAC verifies approved source claims and the complete signed body.
It does not authenticate a physical game effect, resist a malicious local owner
or supply a permission denied by the host. Typed contracts, matching source
hashes and a valid signature cannot establish which build is running without
an owner-provided loading receipt. They do not create remote authentication or
cross-language device authority.

## Encode and admit complete records

The following is the encoding sequence for **owner-prepared inputs**. It is not
a capture or key-bootstrap example. `source`, `unroll`, `grant`, `key` and
`steps` must already contain the complete approved data. The producer retains
its original records and first typed receipts locally. `MeasuredQuality`
explicitly supplies `before_fresh`, `after_fresh`, `mask_fresh`,
`observation_quality`, `reward_quality` and `legal_mask_quality`;
`LegalActionEvidence` binds the command identifiers, observation sequence and
actual action-mask hash. These are approved producer claims, not inferred
quality from a successful callback.

```python
from game_learning_runtime.fleet_measured import encode_measured_shard

encoded = encode_measured_shard(
    source,
    unroll,
    grant=grant,
    key=key,
    shard_seq=shard_seq,
    produced_at_utc_ms=produced_at_utc_ms,
    expires_at_utc_ms=expires_at_utc_ms,
    steps=steps,
)
```

`encoded.carrier` has the existing `manifest`, `payload` and `chunks`.
`encoded.envelope` retains the signed proof. Keep both; do not remove typed
receipt fields and mark a real sample simulated to pass the old profile.
Verification binds the three actual tensor hashes, action/context interval,
settled lifecycle, legal mask and declared target/configuration. It validates
reward-safety accumulation against a persistent episode tail. Reopening or
splitting the episode into shards does not grant a fresh reward budget.

For an explicitly configured hub, `ingest_measured` accepts the complete
manifest, payload and envelope:

```python
receipt = hub.ingest_measured(
    encoded.carrier.manifest,
    encoded.carrier.payload,
    encoded.envelope,
)
```

For bounded interrupted transfer, use `begin_measured_upload(manifest,
envelope)`, the existing `put_chunk` calls and `finish_upload`. A header-only
`verify_measured_manifest` check cannot certify numeric payload or full capture;
`verify_measured` requires the decoded carrier. Admission rechecks source grant,
key, proof, expiry and persistent episode identity. A structurally ready shard
still requires explicit consumer enablement and a frozen holdout.

## Freeze an immutable evaluation definition and holdout

Register and ingest complete measured `evaluation_holdout` sources before
training. Construct a `MeasuredEvaluationSuite` with:

- `suite_id`, fixed `evaluation_domain_id` and matching `evidence_kind`;
- `evaluator_source_commit` and actual bounded `evaluator_artifact` bytes;
- `MeasuredEvaluationCase` entries naming exact case/source/epoch/shard,
  payload hash and proof hash for all selected holdout shards;
- `MeasuredEvaluationMetric` entries fixing name, aggregation, direction and
  required sample count.

The suite content hash is computed by the SDK. The hub keeps its definition,
actual evaluator bytes and exact proof/data references immutable. It does not
execute that artifact. A bare chosen suite-hash string, learner metric names or
a successful dataset freeze cannot replace a fixed evaluation definition or an
external result.

With an owner-prepared suite and admitted holdout source:

```python
snapshot = hub.freeze_measured_evaluation(
    evaluation_id,
    ((holdout_source.source_id, holdout_source.source_epoch),),
    suite=fixed_suite,
)
```

The result includes `suite_sha256`, `snapshot_sha256`, exact source-spec hashes
and shard identities. The case set must cover the nonempty frozen holdout.
Upload after freeze, missing cases, mutable evaluator artifacts or kind/domain
mismatches cannot be silently accepted.

Exact numeric training copies of holdout inputs are excluded within the fixed
semantic domain across source/runtime/adapter/policy/episode relabelling.
Independent games may contain the same numbers. This gate does not detect
near-duplicates or infer semantic equivalence after dtype/value transforms.
Changed configuration/reward semantics require an audited new-hub migration;
they do not automatically alias an existing domain.

## Explicit finite consumption

After the immutable snapshot exists, the owner supplies a
`RealTrainingEnablement` with all of these fields:

| Fields | Meaning |
| --- | --- |
| `destination_id`, `destination_sha256` | Exact configured local destination; no network authorization. |
| `authority_sha256`, `allowed_source_spec_sha256s` | Current authority and allowlisted exact source declarations. |
| `evaluation_id`, `evaluation_suite_sha256`, `evaluation_snapshot_sha256` | Frozen evaluation definition and data snapshot. |
| `evidence_kind` | Must match grant and suite; a fixture cannot become a real capture by relabelling. |
| `approval_id`, `approval_sha256`, `expires_at_utc_ms` | Owner's finite approval binding and expiry; constructing these values is not proof of a production approval. |
| `max_transitions`, `max_callback_calls` | Maximum total transitions reserved for consumption and callback-attempt ceilings for this approval. |

The permit is supplied through `real_enablement`; the default is `None`.
`LearnerSelection` still requires exact compatibility, runtime/adapter and
behavior-policy binding. On-policy selection uses the exact behavior artifact.
Off-policy requires an explicit algorithm/hash and source/behavior allowlists;
algorithm-specific sampling statistics cannot be reconstructed from current
policy or replaced with invented weights.

The following is the consumer sequence, using an owner-supplied `enablement`,
compatible `selection`, an empty exclusive fail-policy `queue` and a bounded
`learner(Unroll, ConsumptionTicket) -> LearnerResult` callback:

```python
from game_learning_runtime.fleet_learner import FleetConsumer

consumer = FleetConsumer(
    hub,
    queue,
    learner_id="local-measured-learner",
    selection=selection,
    real_enablement=enablement,
)
plan = consumer.plan(max_shards=1)
if plan.shard_ids:
    consumed = consumer.consume_one(plan, learner)
```

One plan contains one `Unroll`. Empty plans do not claim data. Before the claim,
the hub atomically reserves the actual transition count and a callback attempt.
The approval identifier/hash binds one immutable permit. Reopening, switching
learner identities or altering limits cannot reset its budget. Failed or unknown
callbacks and crashes do not refund the reservation. These ceilings do not
limit an opaque optimizer's internal update steps.

Keys/grants, destination, proof, source, fixed holdout, permit and expiry are
rechecked before claim, before callback and after callback. Revocation or
unknown effects cannot silently requeue data for retry. A completed receipt
records callback completion and the callback's declared update count; queue
commit alone is not evidence of a parameter write or learning benefit. Save
actual model/parameter artifacts and checksums at the learner's update point.
Use the existing model-bundle contract for distributable reproduction inputs.
Keep fixed external evaluation and candidate-baseline promotion separate.

## Evidence limits and integration work

A complete non-simulated **synthetic_contract_fixture** can exercise verification,
holdout isolation, finite explicit enablement and a real bounded learner
callback. It remains synthetic. Acceptance needs adversarial lifecycle, parser,
signature, exact-reward, cumulative-budget and holdout tests, complete SDK
serialization parity, and normal installed-wheel checks outside the checkout.
This guide does not report that a game producer or production trainer ran.

True producer capture, actual loaded versions, legal host access, key setup,
central destinations, remote authentication and persistent services remain
owner integration work. A rejected connection remains rejected; the SDK does
not select a different transport or escalate access. Incomplete historical
records stay unknown and can test rejection. Their fields cannot be backfilled
into a positive capture from later outcomes.

Read [ADR-0053](../decisions/0053-admit-measured-fleet-data-with-local-owner-proof.md)
and the underlying [local fleet guide](local-fleet-training-data.md) for the
boundary between data admission, actual consumption, evaluation and promotion.
