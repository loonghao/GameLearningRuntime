# Local fleet training data

The fleet SDK combines compatible samples from explicitly registered local
producers and supplies them to an owner-configured learner callback. The
existing Launcher shows the same indexed sources, datasets and receipts.
This is a finite local trusted-spool workflow. It adds no remote upload
endpoint, source authentication or continuous collection service.

Version 1 admits only a closed numeric-vector profile. All non-simulated
sources are quarantined. Training from SIMULATED sources requires an explicit
learner selection with `allow_simulated=True`. A digest identifies declared
content; it does not authenticate a machine or prove a real game effect.

## Run the finite example

Install the SDK normally, either as a wheel or as an editable development
install following [CONTRIBUTING](../../CONTRIBUTING.md). The example requires
only the base NumPy dependency. Its reusable implementation is packaged as
`game_learning_runtime.examples.fleet_training`.

Supply the lowercase 40-character source commit from your owned build receipt.
The example records this as **owner-declared** provenance; passing a commit
string is not a verification of the installed build. Keep the wheel and source
receipt separately. Choose a fresh output directory: an existing directory
is rejected instead of being cleared.

From the source checkout, the registered thin tool is:

```powershell
python tools/demo/fleet_training_demo.py --output-dir ../fleet-demo --runtime-source-commit $runtimeSourceCommit
```

Set `$runtimeSourceCommit` to the source commit described above before running
the command. To exercise a normal installation outside the checkout, save this
small caller as `fleet_example.py` in a temporary working directory:

```python
import argparse
import json
from pathlib import Path

from game_learning_runtime.examples.fleet_training import run_demo

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--runtime-source-commit", required=True)
args = parser.parse_args()
summary = run_demo(
    args.output_dir,
    runtime_source_commit=args.runtime_source_commit,
)
print(json.dumps(summary, indent=2, allow_nan=False))
```

Run that caller with the interpreter into which the SDK was installed:

```powershell
python fleet_example.py --output-dir ./fleet-demo --runtime-source-commit $runtimeSourceCommit
```

The example constructs two SIMULATED training producers with three transitions
each. It also constructs a three-transition evaluation holdout, a
three-transition incompatible-game source and an unattached fifth source.
It interrupts the first upload after a chunk, closes and reopens the hub, then
synchronizes the spools twice. The second sync acknowledges duplicate shards
without adding samples or refreshing their data receipt times. Before planning,
the owner explicitly freezes the holdout with
`hub.freeze_evaluation("holdout", "epoch-1")`.

Two actual callbacks consume the two compatible training sources through the
existing `Unroll` and `BoundedActorQueue` interfaces. The two-parameter NumPy
reward regressor performs six parameter updates and saves `learner-model.npy`
with pickling disabled. The summary records raw parameter hashes before and
after, the saved file's SHA-256, selected source IDs, input count and
training-fixture mean squared error. That error is measured on these constructed training samples. It is not
a fixed game evaluation, policy improvement or a checkpoint promotion;
`game_improvement` remains `"not_evaluated"`.

The output includes:

| Path within the output directory | Purpose |
| --- | --- |
| `glr-project.json` | Existing project contract for read-only observation. |
| `producers/` | Explicit local spool directories for the finite producers. |
| `.glr/fleet/index.sqlite3` | Independent schema-1 transfer and consumer ledger. |
| `.glr/fleet/launcher-snapshot.json` | Bounded safe Launcher projection. |
| `learner-model.npy` | Intermediate two-parameter reward-regression result. |
| `learner-recipe.json` | Zero initialization, float64 parameters, update rule, learning rate, actual source order and six inputs. |
| `learner-bundle/manifest.json` | Existing `glr.model-bundle.v1` inputs/artifact entries with relative paths, byte sizes and SHA-256. |
| `demo-summary.json` | Constructed inputs, actual callback receipts, model checksums and fixture measurements. |

The example reuses `build_model_bundle` and `verify_model_bundle` to save a
local `learner-bundle`. Its inputs are the exact example source bytes, learner
recipe and the actually consumed `producer-a`/`producer-b` manifests and
chunks. Holdout and incompatible-game samples are excluded. The recipe declares
seed 0 and zero random draws; it records the deterministic zero initialization
and update rule. The model array is the bundle artifact. `demo-summary.json`
records the bundle schema, relative path and manifest SHA-256 in addition to
the raw parameter and saved-model-file checksums.

This is a local reproduction bundle for an intermediate reward regressor.
It supplies no inference runtime and performs no model deployment, game
policy evaluation or checkpoint installation. See
[reproducible model bundles](reproducible-model-bundles.md) for the existing
verification API and integrity contract.

The fixture project has placeholder runtime, trainer and player commands.
Use it for observation; no role is executed by the example.

## Inspect it in the existing Launcher

Use a CLI build containing the fleet view, then point the existing read-only
observer at the example output:

```powershell
glr --project ./fleet-demo observe --port 0
```

Open the printed loopback URL and select **Fleet**. This command is a separate,
explicit observer; the Python example does not start a server. The Fleet panel
reads `GET /api/v1/fleet` and supports manual snapshot refresh. It does not
start a producer, import data or invoke a learner.

The endpoint returns a `glr.fleet.view.v1` envelope with status `available`,
`missing`, `invalid` or `unavailable`. Only an available, validated file has a
`glr.fleet.snapshot.v1` payload. Missing or rejected data leaves machine
activity unknown. GET preserves the recorded timestamps and never creates a
heartbeat or a new snapshot.

A snapshot and each heartbeat have separate 120-second freshness bounds. A
stale snapshot cannot show a healthy machine; future timestamps show a clock
mismatch. SIMULATED sources are never presented as online machines. The
unattached source has no heartbeat and remains unknown. Keep producer-declared
observation times separate from the hub's actual receipt times. Refreshing the
UI or resending identical content cannot manufacture freshness.

## Producer and admission contract

`fleet_payload` owns `VectorSpec`, `CompatibilitySpec`, `SourceSpec`,
`encode_shard` and `decode_shard`. `fleet_datahub` owns `FleetHub` and
`write_local_shard`. Register the exact source specification in the hub before
uploading. The owner freezes its source epoch, revision, assignment and split.
An upload cannot override that registration.

Construct an admitted projection from known fields. Reuse `Transition` and
`Unroll`; do not upload a normal collector dictionary without examining it.
The vector profile requires empty `info` and events, null arbitrary provenance
and null `action_receipt`. An ordinary adapter or `CounterEnvironment` may
supply metadata that makes its unfiltered transition ineligible. Raw frames,
commands, private paths and free-form logs are outside the profile.

The codec validates the complete closed record before using the existing
`transition_from_record` decoder. Vectors have fixed field names, dtypes and
lengths. Object, string, structured and image tensors are rejected. Reward
vectors are finite float32 or float64; terminal and mask vectors are boolean.
Episode, step, time and consecutive observation boundaries must be consistent.
The hub persists terminal state and episode tails across shards. Continuing an
episode must retain the exact run, source, epoch and compatibility cohort,
advance its step consecutively, preserve time ordering and match the previous
next-observation boundary. Terminal continuation is quarantined. Earlier
uploading fragments must finish before later shards can be admitted. A new
independent episode remains usable; purging files cannot erase these fences.
These structural checks do not prove that an action executed or caused a
reward. Real data needs a future reviewed evidence profile.

Source provenance retains a machine pseudonym, source revision and artifact
hash, runtime commit, adapter hash, run/game identity, environment/protocol,
configuration and vector specifications, reward contract, behavior-policy
hash, policy epoch/version and nullable checkpoint hash. UTC production and
receipt times are distinct. A compatibility group binds exact game, runtime,
adapter and compatibility fields; equal array shapes alone do not admit data.

`write_local_shard` writes a bounded immutable local spool. Marker, catalog,
manifest and chunks are published atomically. If publication stops partway,
retry the same validated artifact in that owned namespace. Existing matching
parts are verified and completed; conflicting or foreign fragments are not
overwritten or deleted. The hub supports
`begin_upload`, `put_chunk` and `finish_upload`, or an explicit
`sync_local_spools(directories, max_shards=32)` pass. Resume uses verified stored
chunk offsets. Changed bytes under the same identity are a conflict. Only a
complete, hash-verified, validated payload can become ready. The operation is
sync-once; scheduling and any separately authorized transport remain outside
this implementation.

Default `FleetLimits` cap a shard and a chunk at 1 MiB, a manifest at 16 KiB,
128 transitions per shard, 64 KiB per tensor, 8,192 values per vector,
64 sources, 1,024 shard identities, 1,024 lifetime plans and 64 MiB of retained
payload reservations.
The demo lowers chunk size to 512 bytes to exercise interruption. Configure
limits when creating a new owned hub; reopening uses its stored limits.

## Learner selection and receipts

`fleet_learner` owns `LearnerSelection`, `FleetConsumer`, `ConsumptionTicket`
and `LearnerResult`. A selection supplies the required compatibility contract,
game/runtime/adapter identity and behavior-policy hash. On-policy selection
requires exact equality of the behavior-policy artifact hash. Epoch and
version remain frozen source provenance; the selection has no independent
target-epoch or target-version field.

Off-policy selection additionally requires an explicit algorithm name and
implementation hash, allowed source IDs and allowed behavior-policy hashes.
The callback owns algorithm support and its required inputs. The hub does not
create log-probabilities, importance weights or model-weight averages.
Evaluation-holdout, quarantined, revoked and incompatible sources cannot enter
the training callback. A registered holdout blocks training of its exact
compatibility cohort until the owner calls
`FleetHub.freeze_evaluation(source_id, source_epoch)`. Freeze completes the
fixed-data assignment; uploading more data to it is rejected. Exact numeric
input fingerprints, excluding incidental IDs and timestamps, prevent a renamed
copy from becoming training data. Episode/step and full-content identities
also persist. Near-duplicate semantic content is not detected. Equal legitimate
numeric inputs can be conservatively excluded by these checks.

A late holdout overlapping calling, consumed or unknown-effect data is rejected.
Overlap with unconsumed training data quarantines those training samples.
SIMULATED samples need the explicit allowance even when all other identities
match.

`FleetConsumer.plan(max_shards=1)` selects one shard as one SDK `Unroll`.
Other `max_shards` values are rejected with `one_unroll_per_plan`; multiple
producers contribute through separate plans and callbacks. An empty plan is
not stored. The same selection and still-pending shard reuse the saved plan
identity. The lifetime plan quota defaults to 1,024; new plans beyond it are
rejected with `plan_quota`.

`consume_one(plan, callback)` materializes the admitted samples as an `Unroll`
and supplies it and a ticket to the configured callback. The example callback
returns `LearnerResult(declared_updates=len(unroll.transitions))` after its
actual NumPy updates. A completed callback with an unknown update count stays
`unknown_effect`; it cannot be automatically retried.

Persisted callback intent and queue lease handling fence consumption. The
receipt distinguishes observed callback completion from the learner-declared
update count. Queue commit alone is not evidence that parameters changed;
callback completion is not independent evidence of correct updates or
performance benefit. Preserve learner-owned model evidence and fixed external
evaluation separately. Dataset admission does not install a checkpoint.

If the callback raises, the process stops mid-call or completion cannot be
persisted, the effect can be unknown. Do not automatically retry an uncertain
update. Resume preserves an unknown effect instead of treating a timeout as a
successful stop or an unused sample.

## Revocation, cleanup and safe projection

`revoke_source` blocks further intake and selection for that source epoch,
including a final consumer check. It cannot undo a completed update or recall
an in-flight effect. `purge_shard` removes only owned payloads that are safe to
remove. It retains identity, split and receipt tombstones; it rejects active
or uncertain consumption. Removing bytes cannot erase holdout history or
make a consumed sample new again.

Publish a snapshot explicitly with `FleetHub.write_snapshot()`. Its arrays
contain source rows, dataset counts and actual durable consumer receipts.
Missing checkpoint, timestamps, update count or last selection eligibility
remain null. Last-plan reasons come from the actual selection; until then,
eligibility is unknown. The projection contains no raw vectors or arbitrary
metadata, has a 1 MiB bound and at most 64 rows per array, and uses fixed reason
codes. It stays local by default.

The fleet index remains separate from the schema-2 run store. Game adapters
own observation, action legality, reward/receipt evidence and their producer
projection. Learners own models and updates. The shared runtime owns bounded
admission, transfer/consumer records and the read-only view. Remote source
authentication, cross-machine transport and continuously running jobs require
separate design and authorization.

See [ADR-0052](../decisions/0052-local-fleet-training-data-hub.md),
[the dashboard guide](dashboard.md) and
[the decision evidence guide](decision-evidence-timeline.md).
