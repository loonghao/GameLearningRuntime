# Decision evidence and read-only timelines

Record what the learner selected and what the adapter observed in one bounded
diagnostic format. Use the same timeline projection in adapter-owned launcher
views. This SDK provides capture, local persistence and read-only queries;
launcher UI and automatic decision instrumentation require explicit integration.

Install the package using the [development baseline](../../CONTRIBUTING.md).
Keep action authorization, observation, legal-action masks, reset, rules and
success judgment in the adapter. A diagnostic record grants no capability.

## Identify the producer entry point

| Evidence | Producer behavior |
| --- | --- |
| `reward.correlated` | `SyncCollector` emits it automatically only when the strict reward guard and store/run binding are supplied |
| `agent.decision.evidence` | The producer explicitly calls `capture_transition` and `Telemetry.decision_evidence` |
| `learning.correlated-update` | The actual scalar learner explicitly calls `Telemetry.correlated_learning_update` at its update point |
| Timeline | A local consumer explicitly calls `read_timeline` or `project_events`; no new HTTP or UI endpoint is installed |
| Legacy decision/execution | Existing `execute_decision` hooks retain their behavior; raw payloads do not become verified evidence |

The normal `Policy` returns a tensor action. If it exposes no candidates,
scores, confidence or reason codes, record those fields as unknown. Do not
interpret a reward or changed checkpoint as the reason for an earlier action.

## Capture and query one synthetic transition

This complete example creates independent synthetic data. In production,
use the already captured `Transition`, its actual `ActionReceipt`, the durable
`RunRecord` and the real configuration digest. Do not dispatch a second action
to make a diagnostic record.

```python
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

import numpy as np

from game_learning_runtime.contracts import ActionOutcome, ActionReceipt, Transition
from game_learning_runtime.decision_evidence import capture_transition
from game_learning_runtime.decision_timeline import read_timeline
from game_learning_runtime.run_store import TrainingStore
from game_learning_runtime.telemetry import Telemetry

with TemporaryDirectory() as directory:
    store = TrainingStore(Path(directory) / "runs.sqlite3")
    run = store.create_run(
        environment_id="synthetic.timeline",
        protocol_version="1.0",
        kind="train",
        run_id="synthetic-run",
        environment_config_digest=hashlib.sha256(b"synthetic-config-v1").hexdigest(),
    )
    receipt = ActionReceipt(
        "synthetic-action", UUID(int=1), 1, ActionOutcome.ACCEPTED, 100, 110,
        authoritative_observation_sequence=11,
        issued_against_observation_sequence=10,
        target_id="synthetic-target",
    )
    transition = Transition(
        UUID(int=1), 0,
        {"state": np.array([0], np.float32)},
        {"choice": np.array([1], np.int64)},
        np.array([0.0], np.float32),
        {"state": np.array([1], np.float32)},
        np.array([False]), np.array([False]),
        action_receipt=receipt, timestamp_ns=110,
    )
    evidence = capture_transition(
        transition, run=run, decision_id="synthetic-decision",
        policy_version=1, mode="train",
    )
    Telemetry(store, run.run_id, console=False).decision_evidence(evidence)
    page = read_timeline(store.path, run.run_id, limit=100)
    record = page["entries"][0]["evidence"]
    assert record["selection"]["candidates"] is None
    assert record["rules"]["consumptions"] is None
    assert record["reward"] is None
    assert page["cursor"]["events_after"] == 1
    assert not page["more"]
    print("synthetic-decision candidates=unknown rule-use=unknown reward=unknown")
```

Expected output:

```text
synthetic-decision candidates=unknown rule-use=unknown reward=unknown
```

The action receipt and tensor hashes are captured facts. This example supplies
no reward composition or rule-consumer evidence, so those facts stay unknown.
The new telemetry method persists locally without printing the record, even
if its `console` option is enabled. Existing telemetry methods retain their
console behavior; use `console=False` for local-only producers.

## Supply evidence at the actual selection and update points

Provide `selection` only if the selecting policy reports it at decision time.
It contains exactly `candidates`, `chosen_candidate_id` and `basis`. Each
candidate contains `id`, `legal`, `rejection_reason`, `score` and `basis`.
Use nullable fields for unmeasured legality or scores. Basis codes are
`score_order`, `exploration`, `rule_requirement`, `legal_mask` and
`policy_reported`; rejection codes are `mask_rejected`, `capability_missing`,
`stale_observation`, `not_legal` and `unknown`. Null is permitted for an
unreported code. The action hash always comes from the captured tensor action.
Reported legality does not bypass an adapter's legal-action or dispatch gate.

Pass explicit `policy_version`, `policy_sha256`, `checkpoint_sha256` and
`mode` when those values are available. Modes are `train`, `evaluate` and
`unknown`. A run's label does not infer the mode of each decision.

For rule evidence, pass the exact `rule_source_id`, `RuleIndexBinding` and
`DecisionConsumptionReceipt` values tied to this `decision_id`. The binding
names the environment, protocol, rules revision/hash and index hash. Keep
the rules revision to 1-128 characters: a letter or digit first, then letters,
digits, `.`, `_`, `+` or `-`. This safe projection rejects paths, drive-relative
names and free text; the existing rule-binding API retains its own contract.
Keep
`RETRIEVED` separate from `USED`. A `USED` record declares a consumer artifact
hash; it is not SDK proof that a table was read or that the rule helped.
`binding_checked` only validates supplied records. Even an empty consumption
list can have a valid binding, so it cannot be displayed as actual rule use.
Missing binding or consumer evidence remains unknown.

For a custom strict reward producer, retain the base `CorrelatedRewardReceipt`
returned by its first composition and supply it as `correlated_reward` to
`capture_transition`. Use the same captured action, observations and scalar
reward. The capture API rejects another run/configuration/target, episode,
step, interval, action or tensor digest.
A receipt without an observed scalar remains `declared` after an exact
dtype-aware comparison; `binding_checked` additionally requires its observed
scalar to exactly match the captured float32 or float64 reward.
Read the
[strict reward guide](correlated-rewards.md) for lifecycle, effect and budget
validation. A consistent record does not independently authenticate a game.

When using `SyncCollector`, its transition provenance contains a projected
mapping and hash, not a typed `CorrelatedRewardReceipt`. Capture that returned
transition without the optional typed reward and retain unknown reward fields
in the decision record. The collector's existing `reward.correlated` event can
still appear separately in the timeline. Do not rerun composition for the same
interval or reconstruct a typed receipt by guessing missing fields. A complete
decision-to-reward binding requires producer instrumentation that retains the
first typed receipt; this API adds no such collector hook.

At an actual scalar learner update, report its operands using
`ScalarLearningUpdate` and the same reward receipt. Call
`telemetry.correlated_learning_update(receipt, update, consumer=consumer)`
with an owner-selected `LearningConsumerPolicy` binding the learner, table
and policy version. This existing method validates identity, tensor hashes,
reported TD arithmetic and terminal bootstrap behavior. It does not read or
write the learner table. An uninstrumented learner supplies no update evidence.

## Interpret the shared schema

`DecisionEvidence.to_mapping()` returns detached data using
`glr.decision-evidence.v1`. Unexpected fields and non-data objects are rejected.
Use null for absent quantities and identifiers, and the defined `unknown`
value for categorical states. Do not replace unknown values with zero or false.

For measured fields not exposed as capture arguments, construct
`DecisionEvidence(mapping)` from the exact safe schema before persistence.
Changing a detached mapping does not change the original evidence record.
Validation checks the declaration's format and consistency, not its external
truth; supply confidence, success or comparison only from the actual producer.

| Section | Evidence and limits |
| --- | --- |
| `identity` | Run, environment, protocol, configuration hash, target, episode, pre-step and optional decision ID |
| `observation` | Producer sequence, optional timestamp/confidence, state hash, lifecycle and sequence freshness |
| `selection` | At most 64 explicitly reported candidates, finite nullable scores, short codes, chosen ID and actual action hash |
| `rules` | Source, version-bound index and at most 64 declared consumption receipts; binding checks do not certify consumption |
| `execution` | Actual action receipt ID, outcome, target, interval and timestamps; arbitrary receipt details are excluded |
| `reward` | Optional receipt hash, interval, bounded term contributions, observed scalar, next-state hash and lifecycle |
| `outcome` | Captured terminal flags; known success additionally requires a source and evidence hash |
| `policy` | Explicit policy/checkpoint identity, version and training/evaluation mode |
| `comparison` | Paired baseline/candidate scores, fixed evaluator/suite hashes, positive budget, direction and minimum improvement, or unknown |

Each evidence record is limited to 64 KiB. Nesting and container sizes are
bounded; hashes reject unsuitable tensor data through the existing tensor
hash contract. Observation `freshness` describes producer sequence advance
over the recorded action interval. It is not a wall-clock age measurement,
OS authentication or independent live-game liveness check. Confidence remains
unknown unless a producer supplies an explicit measurement.

The optional comparison is diagnostic. Complete fixed inputs must agree with
its status. Direction is `min` or `max`; the minimum improvement is nonnegative
and `candidate_better` requires improvement strictly greater than that minimum.
Ties and missing inputs never claim a winning candidate. Keep the incumbent
and its fixed comparison budget until the separate
[evaluation and review process](continuous-learning.md) authorizes promotion.
This record does not create that authority or install a checkpoint.

## Read and render bounded pages

`read_timeline(database_path, run_id, after_sequence=-1, limit=250)` opens an
existing regular database with `mode=ro` and `query_only`. It never constructs
a writable store, initializes a database or migrates schema 2. A missing or
symlink path raises `FileNotFoundError`; an invalid cursor or page limit raises
`ValueError`. Database/schema errors remain query errors rather than an empty
success response.

The response uses `glr.decision-timeline.v1` with `entries`, `warnings`,
`unlinked_update_sequences`, `source_window`, `cursor.events_after` and `more`.
Use that cursor for the next page. It advances across all scanned events,
including ignored private logs; using the last displayed decision would lose
or repeatedly scan those rows. Recognized malformed evidence produces bounded
warning codes instead of exporting raw payloads or exception details.
Non-text payloads, excessive JSON nesting and out-of-range numeric values
also become invalid-evidence warnings; later valid rows and the scanned cursor
are retained. This does not hide database or schema errors.

`project_events(events, run_id=...)` accepts an explicitly bounded window of
existing `RunEvent` records. It links reported updates only on exact run,
episode, pre-step, action, producer interval, receipt hash and all three tensor
hashes. The resulting `reported_binding_checked` status means record
consistency, not verified table mutation, improvement or evaluator approval.
Do not join adjacent timestamps, equal steps or legacy event order. Links
outside a page remain unlinked; a consumer can retain at most 250 source events
and reproject that bounded window when needed.

Adapter-owned UI should render these projected sections and their status,
retain explicit unknown values, and show unlinked or invalid evidence as such.
No native endpoint or launcher UI is added here. Do not forward legacy state,
commands, parameters, raw observations, private reasoning, secrets or local
paths into this view. Keep local diagnostic storage local unless an owner
explicitly authorizes a separate safe export.

See [ADR-0051](../decisions/0051-record-safe-decision-evidence-and-readonly-timelines.md)
and the existing [learner decision guide](learner-owned-decisions.md).
