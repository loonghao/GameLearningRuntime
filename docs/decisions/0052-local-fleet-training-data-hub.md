# ADR-0052: Add a local fleet training data hub

Status: Proposed (local prototype implemented; draft review pending)

## Context

Several producers can collect more training samples than one learner can use.
Those samples still need a common environment, observation and action contract,
known behavior policy, reward provenance and an explicit training assignment.
Combining files without these bindings can mix incompatible games, admit fixed
evaluation data into training or count the same transition twice.

The runtime already has `Transition`, `Unroll`, JSONL transition serialization,
a bounded actor queue and an embedded Launcher dashboard. Its bridge telemetry
and decision timeline are diagnostics. A recording, run status or successful
collection attempt does not establish training eligibility. The core provides
objective primitives; models, optimizers and algorithm semantics remain learner
responsibilities under [ADR-0005](0005-share-objectives-not-learners.md).

## Decision

Add an opt-in local data hub that validates bounded numeric shards, persists
transfer and selection identities, and supplies admitted `Unroll` objects to an
explicit learner consumer. Extend the existing Launcher with a read-only view
of that same hub. Keep this first implementation inside the local trusted-spool
boundary; it is not an authenticated distributed collector.

The first complete example uses two constructed **SIMULATED** vector producers
and a NumPy learner callback that performs real local parameter updates. It
exercises export, interrupted transfer, duplicate handling, compatible sample
selection and a durable consumer receipt. The example saves its two-parameter
reward-regression result with raw parameter hashes and a saved-file checksum.
Reuse `glr.model-bundle.v1` for copied reproduction inputs and the artifact,
including the exact example source, learner recipe and consumed training
shard manifests/chunks. Fixed evaluation and incompatible-game data remain
outside that bundle. This local data-only bundle is not an inference package
or checkpoint installation. These updates demonstrate the data path, not game performance or a successful
policy promotion. No producer is
presented as a connected real machine.

### Numeric payload and provenance

Reuse the existing `glr.transition.v1` record and `Transition`/`Unroll` classes.
Do not introduce a second batch hierarchy. The initial profile has explicit
numeric vector shapes, dtypes and byte limits. Before calling the existing
record decoder, validate the complete record, dimension products, encoded
sizes, finite values, terminal flags, action-vector representation and required
masks.
Object, string, structured and image/frame tensors are outside this profile.

The profile requires `action_receipt` to be null and admits no arbitrary
`info`, events or provenance payloads. A normal collector result is therefore
not automatically exportable: adapter metadata must never slip through the
existing general serializer. Producers construct the admitted projection from
known fields instead of uploading their logs or raw transition dictionaries.

Retain explicit machine pseudonym, source epoch and revision, game/run,
runtime and adapter revision, protocol, effective configuration, observation
and action specifications, reward contract, behavior policy and checkpoint
identity. Episode and step identities belong to the samples. Missing checkpoint
identity remains null. An artifact hash binds bytes; it does not authenticate a
machine, prove which build is running or establish a game effect.

Reward and execution provenance remain explicit. This profile cannot provide
real action-receipt or causal-effect evidence. Version 1 quarantines all
non-simulated sources. Only SIMULATED inputs explicitly allowed by the learner plan may
enter training. Constructed execution and reward values remain separate from
measurements of real game effects. A future profile for measured game data
needs its own reviewed contract; it cannot silently widen this one.

### Local transfer and durable index

Use an owner-selected hub directory with a separate `index.sqlite3` at schema
version 1. Do not migrate or lower the existing schema-2 run store. Keep staged
payloads, accepted shard identities and transfer offsets inside that directory.
The source epoch and immutable shard identity bind their content hash. Repeated
content is acknowledged without creating new samples; changed content under
an existing identity is rejected.

Producer publication uses atomic marker, catalog, manifest and chunk writes.
A retry may complete only the exact validated artifact inside the producer's
owned namespace. Existing conflicting or foreign fragments remain intact.
This recovery applies to owned partial publication as well as hub transfer;
it adds no filesystem permission or general repair operation.

Only completely received, hash-verified and profile-validated payloads become
ready. An interrupted sync resumes its recorded verified offset. A partial
file is not a successful shard. Bound transfer size, retained bytes and page
sizes. Use local configured spools and explicit sync-once operations; this adds
no external HTTP ingress, remote credentials, cloud storage or continuous
background service.

The index retains each episode's terminal state and last accepted step, time
and next-observation identity across shards. Continuation must retain its
exact run, source epoch and compatibility cohort, advance the step
consecutively, preserve timestamp ordering and match the prior observation
boundary. A terminated episode cannot continue. Earlier uploading fragments
must finalize before later shards; independent new episodes remain usable.
Purging payload bytes cannot erase these logical reservations.

### Training admission and consumer ownership

The hub owner freezes the train, evaluation-holdout or quarantine assignment.
Producers cannot choose a training label to override it. Content and source
identities retain their exclusion history through retries and renaming. Fixed
evaluation holdout data never enters a training plan. Before training its
cohort, the owner explicitly freezes the registered evaluation holdout with
`FleetHub.freeze_evaluation(source_id, source_epoch)`. A frozen holdout accepts
no further uploads. Exact numeric input fingerprints exclude incidental IDs
and timestamps so relabeling cannot erase exclusion. Episode/step and complete
content identities also remain indexed. An unfrozen registered holdout blocks
training in its cohort.

A late holdout that overlaps an already calling, consumed or uncertain sample
is rejected; it cannot retroactively turn a used training sample into a clean
baseline. Overlapping unconsumed training data is quarantined. These exact
fingerprints conservatively reject some legitimate equal numeric inputs. They
do not claim semantic near-duplicate detection.

Compatibility cohorts require exact game, environment, runtime, adapter,
protocol, configuration, observation/action specification and reward-contract
identities. Matching array shapes alone is insufficient. Each plan selects
one bounded shard as one SDK `Unroll`. `plan(max_shards=1)` rejects other
counts; it does not imply multi-shard concatenation. Empty plans are not
persisted, equivalent pending plans reuse their identity and the default
lifetime plan quota is 1,024. The data path does not merge model weights or
silently convert observations between adapters.

On-policy selection requires exact equality of the sampling behavior-policy
artifact hash. Policy epoch and version are frozen source provenance, not
independent target-epoch or target-version selection gates. The actor queue's
integer policy-lag check alone cannot establish artifact equality. Algorithms
that need behavior statistics must receive the statistics captured during
sampling. Missing log-probabilities cannot be reconstructed using the current
policy.
Off-policy selection requires a declared algorithm and implementation hash,
explicit source and behavior-policy allowlists, and all of the callback
algorithm's required inputs. Those declarations are owner-supplied; the hub
does not certify the algorithm's correctness. The hub invents no sample or
importance weights to make an incompatible dataset acceptable.

Materialize the verified samples into the existing `Unroll` contract, then
use the existing queue lease and commit/abort behavior. Persist a consumer
intent before invoking the configured callback. The consumer reports the
selected input identity, observed callback completion and its declared update
count. Models, optimizer operations and parameter evidence stay learner-owned.
A queue commit or telemetry event cannot substitute for callback invocation.

A completed callback receipt is not independent evidence of parameter
correctness, game improvement or evaluation success. An exception, process
interruption or receipt-persistence failure can leave an unknown effect.
Retain that outcome and prohibit automatic update retries. Evaluation, external
review and checkpoint installation retain their existing authority boundaries.

### Revocation and cleanup

Owner registration pins the source epoch and revision. Source revocation blocks
new transfers, plans and claims, with a final check before invoking a learner.
It cannot recall data already used or prove that an in-flight learner stopped.
An unknown consumer effect is not released or replayed after a timeout.

Purge only hub-owned payloads that are no longer pinned by active selection or
uncertain effects. Retain identity, assignment and receipt tombstones so deleted
bytes cannot erase holdout history or permit duplicate re-import. Cleanup does
not delete game files, existing run evidence, models or checkpoints.

### Existing Launcher projection

The Python hub explicitly writes a bounded atomic `glr.fleet.snapshot.v1` file
at `.glr/fleet/launcher-snapshot.json`. The existing Rust observation server
reads this closed projection through its existing local path and loopback
request guards. It does not ingest data, write a heartbeat or start another
server. The existing dashboard renders registered sources, compatible dataset
cohorts, accepted/duplicate/rejected counts, readiness and consumer receipts.
It remains a running-data view, not a daily report.

Source timestamps are producer declarations. Heartbeat and data receipt times
are measured by the hub when new forward evidence arrives. Duplicate transfers
and heartbeat retries cannot manufacture freshness. An unconnected source is
unknown; SIMULATED sources are never online. Snapshot and heartbeat freshness
have separate 120-second bounds. A stale snapshot vetoes a healthy presentation;
future timestamps indicate a clock mismatch instead of being clamped fresh.

Unknown timestamps, checkpoint identity, selection eligibility and declared
update count remain null. The snapshot contains no arrays, frames, logs, command
arguments, secrets, paths or arbitrary metadata. It has a 1 MiB byte limit,
at most 64 rows per array and only fixed reason codes. Local diagnostics stay
local by default.

## Consequences

- Compatible producers contribute to one larger, deduplicated training dataset
  without copying game logic or introducing another application.
- Transfer, quality, selection and callback receipts are visible in the same
  Launcher that already observes runs.
- A local index and explicit spool operation keep deployment and authorization
  costs small, but do not supply remote transport or source authentication.
- Exact compatibility and holdout isolation reject some otherwise parseable
  data. This is preferable to silently changing learner inputs or evaluation.
- The vector-only profile cannot admit ordinary unfiltered collector results
  or establish causal correctness for real games.
- Consumer failure can be indeterminate; larger datasets do not imply safe
  automatic update retries, improved policies or checkpoint promotion.

## Verification contract

The synthetic path must select data from both simulated producers, demonstrate
an interrupted transfer and duplicate without increasing sample counts, perform
an actual NumPy callback update and preserve its receipt. The existing Launcher
must show those same indexed rows and leave unconnected machines unknown.

Failure checks cover bounded payloads and forbidden metadata; source, episode
and schema drift; immutable deduplication and resume; holdout relabeling;
exact behavior-policy mismatch; undeclared off-policy use; consumer failure and
revocation; and truthful bounded Launcher freshness. Packaging acceptance uses
the normal installed wheel outside the checkout. Synthetic acceptance does not
claim real-game deployment or measured training benefit.

See [ADR-0028](0028-embedded-training-dashboard.md),
[ADR-0051](0051-record-safe-decision-evidence-and-readonly-timelines.md) and
[the local fleet guide](../guides/local-fleet-training-data.md).
