"""Synthetic checks for the restricted numeric-vector fleet boundary."""

import json
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

import game_learning_runtime.fleet_datahub as hub_module
import game_learning_runtime.fleet_payload as payload_module
from game_learning_runtime.collector import BoundedActorQueue
from game_learning_runtime.contracts import Transition, Unroll
from game_learning_runtime.fleet_datahub import FleetHub, write_local_shard
from game_learning_runtime.fleet_learner import FleetConsumer, LearnerResult, LearnerSelection
from game_learning_runtime.fleet_payload import (
    DEFAULT_LIMITS,
    CompatibilitySpec,
    FleetError,
    FleetLimits,
    SourceSpec,
    VectorSpec,
    decode_shard,
    encode_shard,
)


def source(source_id: str = "producer-a", **changes: object) -> SourceSpec:
    spec = CompatibilitySpec(
        environment_id="synthetic.environment",
        protocol_version="1.0",
        environment_config_sha256=sha256(b"config").hexdigest(),
        observation=(VectorSpec("state", "<f4", 2),),
        action=(VectorSpec("choice", "<i4", 1),),
        reward_contract_sha256=sha256(b"simulated-reward").hexdigest(),
    )
    value = SourceSpec(
        source_id=source_id,
        source_epoch="epoch-a",
        machine_id="simulated-a",
        source_revision="synthetic-v1",
        source_sha256=sha256(b"producer-source").hexdigest(),
        runtime_source_commit="a" * 40,
        adapter_source_sha256=sha256(b"adapter").hexdigest(),
        run_id="run-a",
        game_id="synthetic.game",
        compatibility=spec,
        policy_epoch="policy-a",
        policy_version=0,
        behavior_policy_sha256=sha256(b"policy").hexdigest(),
        assignment_id="assignment-a",
        simulated=True,
    )
    return replace(value, **changes)


def unroll(spec: SourceSpec, sequence: int = 0, count: int = 2) -> Unroll:
    transitions = tuple(
        Transition(
            episode_id=UUID(int=1),
            step_id=index + 1,
            timestamp_ns=100 + index,
            observation={"state": np.array([index, index + 1], dtype=np.float32)},
            action={"choice": np.array([0], dtype=np.int32)},
            reward=np.array([1], dtype=np.float32),
            next_observation={"state": np.array([index + 1, index + 2], dtype=np.float32)},
            terminated=np.array([index == count - 1], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
        )
        for index in range(count)
    )
    return Unroll(
        transitions,
        actor_id=spec.source_id,
        sequence_id=sequence,
        policy_version=spec.policy_version,
        environment_config_digest=spec.compatibility.environment_config_sha256,
    )


def test_roundtrip_uses_sdk_unroll_and_frozen_numeric_vectors() -> None:
    spec = source()
    encoded = encode_shard(spec, shard_seq=0, unroll=unroll(spec), produced_at_utc_ms=1000)
    decoded = decode_shard(encoded.manifest, encoded.payload)
    assert isinstance(decoded.unroll, Unroll)
    assert decoded.source == spec
    assert len(decoded.unroll.transitions) == 2
    assert not decoded.unroll.transitions[0].reward.flags.writeable


def episode_parts(spec: SourceSpec, *, terminal_first: bool = False) -> tuple[Unroll, Unroll]:
    whole = unroll(spec, count=4)
    first = replace(whole, transitions=whole.transitions[:2])
    if terminal_first:
        first = replace(
            first,
            transitions=(
                first.transitions[0],
                replace(first.transitions[1], terminated=np.array([True], dtype=np.bool_)),
            ),
        )
    return first, replace(whole, sequence_id=1, transitions=whole.transitions[2:])


def test_episode_terminal_fence_survives_new_shard_and_reopen(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: 2000)
    hub.register_source(spec)
    first, second = episode_parts(spec, terminal_first=True)
    encoded_first = encode_shard(spec, shard_seq=0, unroll=first, produced_at_utc_ms=1000)
    encoded_second = encode_shard(spec, shard_seq=1, unroll=second, produced_at_utc_ms=1000)
    assert hub.ingest(encoded_first.manifest, encoded_first.payload).status == "ready"
    hub.close()
    reopened = FleetHub.open(tmp_path / "hub", clock_ms=lambda: 2000)
    assert reopened.ingest(encoded_second.manifest, encoded_second.payload).status == "quarantine"
    assert reopened.snapshot()["datasets"][0]["ready_shard_count"] == 1


@pytest.mark.parametrize("change", ["gap", "time", "observation"])
def test_cross_shard_episode_continuity_is_checked(tmp_path: Path, change: str) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: 2000)
    hub.register_source(spec)
    first, second = episode_parts(spec)
    if change == "gap":
        second = replace(
            second, transitions=tuple(replace(t, step_id=t.step_id + 1) for t in second.transitions)
        )
    elif change == "time":
        second = replace(
            second,
            transitions=tuple(
                replace(t, timestamp_ns=99 + i) for i, t in enumerate(second.transitions)
            ),
        )
    else:
        second = replace(
            second,
            transitions=(
                replace(
                    second.transitions[0],
                    observation={"state": np.array([99, 100], dtype=np.float32)},
                ),
                second.transitions[1],
            ),
        )
    encoded_first = encode_shard(spec, shard_seq=0, unroll=first, produced_at_utc_ms=1000)
    encoded_second = encode_shard(spec, shard_seq=1, unroll=second, produced_at_utc_ms=1000)
    hub.ingest(encoded_first.manifest, encoded_first.payload)
    assert hub.ingest(encoded_second.manifest, encoded_second.payload).status == "quarantine"


def test_valid_episode_continuation_and_new_episode_are_ready(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: 2000)
    hub.register_source(spec)
    first, second = episode_parts(spec)
    for sequence, part in enumerate((first, second)):
        encoded = encode_shard(spec, shard_seq=sequence, unroll=part, produced_at_utc_ms=1000)
        assert hub.ingest(encoded.manifest, encoded.payload).status == "ready"
    third = replace(
        unroll(spec, sequence=2),
        transitions=tuple(replace(t, episode_id=UUID(int=2)) for t in unroll(spec).transitions),
    )
    encoded = encode_shard(spec, shard_seq=2, unroll=third, produced_at_utc_ms=1000)
    assert hub.ingest(encoded.manifest, encoded.payload).status == "ready"


def test_successor_cannot_finalize_ahead_of_uploading_predecessor(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: 2000)
    hub.register_source(spec)
    first, second = episode_parts(spec)
    encoded_first = encode_shard(spec, shard_seq=0, unroll=first, produced_at_utc_ms=1000)
    encoded_second = encode_shard(spec, shard_seq=1, unroll=second, produced_at_utc_ms=1000)
    hub.begin_upload(encoded_first.manifest)
    with pytest.raises(FleetError, match="upload_predecessor_incomplete"):
        hub.ingest(encoded_second.manifest, encoded_second.payload)
    assert hub.snapshot()["datasets"][0]["ready_shard_count"] == 0
    assert hub.ingest(encoded_first.manifest, encoded_first.payload).status == "ready"
    assert hub.ingest(encoded_second.manifest, encoded_second.payload).status == "ready"


@pytest.mark.parametrize("target", ["marker", "catalog", "manifest", "chunk"])
def test_producer_interrupted_publish_retries_only_owned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    spec = source()
    encoded = encode_shard(spec, shard_seq=0, unroll=unroll(spec), produced_at_utc_ms=1000)
    directory = tmp_path / "spool"
    original = hub_module._atomic
    failed = False

    def fail_once(path: Path, payload: bytes) -> None:
        nonlocal failed
        name = {
            "marker": "spool-owner.json",
            "catalog": "catalog.json",
            "manifest": "manifest.json",
            "chunk": "chunk-0.bin",
        }[target]
        if not failed and path.name == name:
            failed = True
            raise OSError("synthetic publication failure")
        original(path, payload)

    monkeypatch.setattr(hub_module, "_atomic", fail_once)
    with pytest.raises(OSError, match="synthetic publication failure"):
        write_local_shard(directory, encoded)
    assert failed
    shard_id = write_local_shard(directory, encoded)
    assert json.loads((directory / "catalog.json").read_text(encoding="utf-8")) == {
        "shards": [shard_id]
    }
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: 2000)
    hub.register_source(spec)
    assert hub.sync_local_spools([directory]).completed_shards == 1


def test_producer_retry_rejects_foreign_fragment_instead_of_claiming_it(tmp_path: Path) -> None:
    spec = source()
    encoded = encode_shard(spec, shard_seq=0, unroll=unroll(spec), produced_at_utc_ms=1000)
    directory = tmp_path / "spool"
    shard_id = write_local_shard(directory, encoded)
    output = directory / "shards" / shard_id
    (output / "manifest.json").unlink()
    foreign = output / "foreign.bin"
    foreign.write_bytes(b"must-preserve")
    with pytest.raises(FleetError, match="unowned_spool_fragment"):
        write_local_shard(directory, encoded)
    assert foreign.read_bytes() == b"must-preserve"


def test_producer_partial_retry_cannot_bypass_filled_catalog_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = source()
    limits = FleetLimits(max_shards=1)
    first = encode_shard(
        spec, shard_seq=0, unroll=unroll(spec), produced_at_utc_ms=1000, limits=limits
    )
    second = encode_shard(
        spec, shard_seq=1, unroll=unroll(spec, sequence=1), produced_at_utc_ms=1000, limits=limits
    )
    directory = tmp_path / "spool"
    original = hub_module._atomic
    failed = False

    def fail_manifest_once(path: Path, payload: bytes) -> None:
        nonlocal failed
        if not failed and path.name == "manifest.json":
            failed = True
            raise OSError("synthetic publication failure")
        original(path, payload)

    monkeypatch.setattr(hub_module, "_atomic", fail_manifest_once)
    with pytest.raises(OSError, match="synthetic publication failure"):
        write_local_shard(directory, first, limits=limits)
    second_id = write_local_shard(directory, second, limits=limits)
    with pytest.raises(FleetError, match="spool_shard_quota"):
        write_local_shard(directory, first, limits=limits)
    assert json.loads((directory / "catalog.json").read_text(encoding="utf-8")) == {
        "shards": [second_id]
    }
    assert write_local_shard(directory, second, limits=limits) == second_id


@pytest.mark.parametrize(
    "field,value",
    [("info", {"raw": "prohibited"}), ("events", (1,)), ("provenance", {"diagnostic": 1})],
)
def test_metadata_is_rejected_instead_of_silently_stripped(field: str, value: object) -> None:
    spec = source()
    original = unroll(spec)
    changed = replace(original.transitions[0], **{field: value})
    with pytest.raises(FleetError):
        encode_shard(
            spec,
            shard_seq=0,
            unroll=replace(original, transitions=(changed,)),
            produced_at_utc_ms=1000,
        )


@pytest.mark.parametrize(
    "value",
    [
        np.array([np.nan, 1], dtype=np.float32),
        np.array([[1, 2]], dtype=np.float32),
        np.array(["x", "y"]),
        np.array([object(), object()], dtype=object),
    ],
)
def test_unsafe_arrays_never_enter_the_portable_decoder(value: np.ndarray) -> None:
    spec = source()
    original = unroll(spec)
    changed = replace(original.transitions[0], observation={"state": value})
    with pytest.raises(FleetError):
        encode_shard(
            spec,
            shard_seq=0,
            unroll=replace(original, transitions=(changed,)),
            produced_at_utc_ms=1000,
        )


def test_content_digest_covers_reward_and_done() -> None:
    spec = source()
    original = unroll(spec)
    encoded = encode_shard(spec, shard_seq=0, unroll=original, produced_at_utc_ms=1000)
    changed = replace(original.transitions[0], reward=np.array([2], dtype=np.float32))
    tampered = encode_shard(
        spec,
        shard_seq=0,
        unroll=replace(original, transitions=(changed, original.transitions[1])),
        produced_at_utc_ms=1000,
    )
    with pytest.raises(FleetError):
        decode_shard(encoded.manifest, tampered.payload)


def test_terminal_mid_unroll_is_invalid() -> None:
    spec = source()
    original = unroll(spec)
    ended = replace(original.transitions[0], terminated=np.array([True], dtype=np.bool_))
    with pytest.raises(FleetError):
        encode_shard(
            spec,
            shard_seq=0,
            unroll=replace(original, transitions=(ended, original.transitions[1])),
            produced_at_utc_ms=1000,
        )


def test_simulation_cannot_mark_final_transition_both_terminated_and_truncated() -> None:
    spec = source()
    original = unroll(spec)
    ambiguous = replace(original.transitions[-1], truncated=np.array([True], dtype=np.bool_))
    with pytest.raises(FleetError, match="ambiguous_lifecycle"):
        encode_shard(
            spec,
            shard_seq=0,
            unroll=replace(original, transitions=(*original.transitions[:-1], ambiguous)),
            produced_at_utc_ms=1000,
        )


def test_byte_limit_and_boolean_limit_are_hard_bounds() -> None:
    with pytest.raises(FleetError):
        FleetLimits(max_chunk_bytes=True)
    spec = source()
    with pytest.raises(FleetError):
        encode_shard(
            spec,
            shard_seq=0,
            unroll=unroll(spec),
            produced_at_utc_ms=1000,
            limits=FleetLimits(max_shard_bytes=32),
        )


def test_small_budget_stops_before_encoding_all_records(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = source()
    calls = []
    original_codec = payload_module.transition_to_record

    def counted(transition: Transition) -> dict:
        calls.append(transition.step_id)
        return original_codec(transition)

    monkeypatch.setattr(payload_module, "transition_to_record", counted)
    with pytest.raises(FleetError):
        encode_shard(
            spec,
            shard_seq=0,
            unroll=unroll(spec, count=128),
            produced_at_utc_ms=1000,
            limits=FleetLimits(max_shard_bytes=1100),
        )
    assert len(calls) < 128


def test_closed_base_export_does_not_include_subclass_fields() -> None:
    @dataclass(frozen=True, slots=True)
    class ExtendedSource(SourceSpec):
        synthetic_extra: str = "synthetic-only"

    base = source()
    extended = ExtendedSource(
        **{name: getattr(base, name) for name in SourceSpec.__dataclass_fields__}
    )
    assert "synthetic_extra" not in SourceSpec.to_record(extended)


def test_codec_does_not_run_for_a_privileged_source_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True, slots=True)
    class ExtendedSource(SourceSpec):
        synthetic_extra: str = "synthetic-only"

    base = source()
    extended = ExtendedSource(
        **{name: getattr(base, name) for name in SourceSpec.__dataclass_fields__}
    )
    calls = []
    original_codec = payload_module.transition_to_record

    def counted(transition: Transition) -> dict:
        calls.append(transition.step_id)
        return original_codec(transition)

    monkeypatch.setattr(payload_module, "transition_to_record", counted)
    with pytest.raises(FleetError):
        encode_shard(extended, shard_seq=0, unroll=unroll(base), produced_at_utc_ms=1000)
    assert calls == []


def consumer(hub: FleetHub, spec: SourceSpec, *, learner_id: str = "learner-a") -> FleetConsumer:
    return FleetConsumer(
        hub,
        BoundedActorQueue(2, overflow_policy="fail"),
        learner_id=learner_id,
        selection=LearnerSelection(
            spec.compatibility,
            spec.behavior_policy_sha256,
            spec.game_id,
            spec.runtime_source_commit,
            spec.adapter_source_sha256,
            allow_simulated=True,
        ),
    )


def shard(
    spec: SourceSpec,
    seq: int = 0,
    *,
    episode: int = 1,
    offset: int = 0,
    limits: FleetLimits = DEFAULT_LIMITS,
):
    original = unroll(spec, sequence=seq)
    transitions = tuple(
        replace(
            item,
            episode_id=UUID(int=episode),
            observation={"state": item.observation["state"] + np.float32(offset)},
            next_observation={"state": item.next_observation["state"] + np.float32(offset)},
        )
        for item in original.transitions
    )
    return encode_shard(
        spec,
        shard_seq=seq,
        unroll=replace(original, transitions=transitions),
        produced_at_utc_ms=1000,
        limits=limits,
    )


def test_partial_upload_reopen_is_real_and_duplicate_does_not_refresh(tmp_path: Path) -> None:
    ticks = [1000]
    spec = source()
    limits = FleetLimits(max_chunk_bytes=512)
    encoded = shard(spec, limits=limits)
    assert len(encoded.chunks) > 1
    hub = FleetHub.create(tmp_path / "hub", limits=limits, clock_ms=lambda: ticks[0])
    hub.register_source(spec)
    receipt = hub.begin_upload(encoded.manifest)
    hub.put_chunk(receipt.shard_id, 0, encoded.chunks[0])
    hub.close()
    hub = FleetHub.open(tmp_path / "hub", clock_ms=lambda: ticks[0])
    restored = hub.begin_upload(encoded.manifest)
    assert restored.next_chunk_index == 1
    for index in range(1, len(encoded.chunks)):
        hub.put_chunk(restored.shard_id, index, encoded.chunks[index])
    assert hub.finish_upload(restored.shard_id).status == "ready"
    before = hub.snapshot()["machines"][0]
    ticks[0] = 9000
    assert hub.ingest(encoded.manifest, encoded.payload).duplicate
    after = hub.snapshot()["machines"][0]
    assert after["data_received_at_utc"] == before["data_received_at_utc"]
    assert after["data_observed_at_utc"] == "1970-01-01T00:00:00.000Z"
    assert before["data_received_at_utc"] != after["data_observed_at_utc"]


def test_missing_suffix_reports_fixed_error_and_never_ready(tmp_path: Path) -> None:
    spec = source()
    limits = FleetLimits(max_chunk_bytes=512)
    encoded = shard(spec, limits=limits)
    hub = FleetHub.create(tmp_path / "hub", limits=limits)
    hub.register_source(spec)
    receipt = hub.begin_upload(encoded.manifest)
    hub.put_chunk(receipt.shard_id, 0, encoded.chunks[0])
    with pytest.raises(FleetError, match="upload_incomplete"):
        hub.finish_upload(receipt.shard_id)
    assert consumer(hub, spec).plan().shard_ids == ()


def test_stored_chunk_drift_is_bounded_and_never_consumed(tmp_path: Path) -> None:
    spec = source()
    encoded = shard(spec)
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(spec)
    upload = hub.begin_upload(encoded.manifest)
    for index, part in enumerate(encoded.chunks):
        hub.put_chunk(upload.shard_id, index, part)
    (hub.root / "artifacts" / upload.shard_id / "chunk-0.bin").write_bytes(
        b"x" * (len(encoded.chunks[0]) + 1)
    )
    with pytest.raises(FleetError):
        hub.finish_upload(upload.shard_id)
    assert consumer(hub, spec).plan().shard_ids == ()


def test_step_identity_blocks_duplicate_across_shard_sequences(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(spec)
    original = shard(spec)
    assert hub.ingest(original.manifest, original.payload).status == "ready"
    duplicate = shard(spec, seq=1)
    assert hub.ingest(duplicate.manifest, duplicate.payload).status == "duplicate"
    learner = consumer(hub, spec)
    calls = []

    def learn(value, ticket):
        calls.append(value.actor_id)
        return LearnerResult(len(value.transitions))

    assert learner.consume_one(learner.plan(), learn).status == "consumed"
    assert learner.plan().shard_ids == ()
    assert len(calls) == 1
    dataset = hub.snapshot()["datasets"][0]
    assert dataset["accepted_transition_count"] == 2
    assert dataset["duplicate_transition_count"] == 2


def test_identity_content_conflict_quarantines_whole_new_shard(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(spec)
    initial = shard(spec)
    hub.ingest(initial.manifest, initial.payload)
    original = unroll(spec, sequence=1)
    changed = replace(original.transitions[0], reward=np.array([99], dtype=np.float32))
    conflict = encode_shard(
        spec,
        shard_seq=1,
        produced_at_utc_ms=1000,
        unroll=replace(original, transitions=(changed, original.transitions[1])),
    )
    assert hub.ingest(conflict.manifest, conflict.payload).status == "quarantine"
    assert hub.snapshot()["datasets"][0]["ready_shard_count"] == 1


def test_holdout_is_order_independent_and_copy_with_new_labels_is_quarantined(
    tmp_path: Path,
) -> None:
    training = source()
    evaluation = source(
        "evaluation", run_id="run-eval", assignment_id="assignment-eval", split="evaluation_holdout"
    )
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(training)
    hub.register_source(evaluation)
    train = shard(training)
    hub.ingest(train.manifest, train.payload)
    learner = consumer(hub, training)
    assert learner.plan().shard_ids == ()
    heldout = shard(evaluation, episode=2)
    assert hub.ingest(heldout.manifest, heldout.payload).status == "ready"
    hub.freeze_evaluation(evaluation.source_id, evaluation.source_epoch)
    assert learner.plan().shard_ids == ()
    copy = source("copy", run_id="renamed-run", assignment_id="copy-train")
    hub.register_source(copy)
    copied = shard(copy, episode=3)
    assert hub.ingest(copied.manifest, copied.payload).status == "quarantine"
    assert learner.plan().shard_ids == ()


def test_late_holdout_cannot_claim_already_consumed_inputs_unseen(tmp_path: Path) -> None:
    training = source()
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(training)
    train = shard(training)
    hub.ingest(train.manifest, train.payload)
    learner = consumer(hub, training)
    assert (
        learner.consume_one(learner.plan(), lambda value, ticket: LearnerResult(2)).status
        == "consumed"
    )
    evaluation = source(
        "evaluation", run_id="eval-run", assignment_id="eval-assignment", split="evaluation_holdout"
    )
    hub.register_source(evaluation)
    heldout = shard(evaluation, episode=2)
    assert hub.ingest(heldout.manifest, heldout.payload).status == "quarantine"
    with pytest.raises(FleetError, match="evaluation_incomplete_or_conflicted"):
        hub.freeze_evaluation(evaluation.source_id, evaluation.source_epoch)


def test_ambiguous_actor_pairs_have_distinct_real_queue_identities(tmp_path: Path) -> None:
    first = source("producer.a", source_epoch="epoch")
    second = source(
        "producer", source_epoch="a.epoch", run_id="run-second", assignment_id="assignment-second"
    )
    hub = FleetHub.create(tmp_path / "hub")
    for spec, episode in ((first, 1), (second, 2)):
        hub.register_source(spec)
        encoded = shard(spec, episode=episode, offset=episode)
        hub.ingest(encoded.manifest, encoded.payload)
    learner = consumer(hub, first)
    calls = []

    def learn(value, ticket):
        calls.append(value.actor_id)
        return LearnerResult(2)

    assert learner.consume_one(learner.plan(), learn).status == "consumed"
    assert learner.consume_one(learner.plan(), learn).status == "consumed"
    assert set(calls) == {"producer.a:epoch", "producer:a.epoch"}


@pytest.mark.parametrize("kind", ["exception", "unknown", "revocation", "binding"])
def test_callback_uncertainty_or_post_revocation_is_never_retried(
    tmp_path: Path, kind: str
) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(spec)
    encoded = shard(spec)
    hub.ingest(encoded.manifest, encoded.payload)
    learner = consumer(hub, spec)
    calls = []

    def learn(value, ticket):
        calls.append(ticket.ticket_id)
        if kind == "exception":
            raise RuntimeError("synthetic-only")
        if kind == "unknown":
            return LearnerResult(None)
        if kind == "binding":
            learner.learner_id = "different"
        else:
            hub.revoke_source(spec.source_id, spec.source_epoch)
        return LearnerResult(2)

    plan = learner.plan()
    receipt = learner.consume_one(plan, learn)
    assert receipt.status == "unknown_effect"
    assert receipt.transition_count == 0
    assert receipt.learner_id == "learner-a"
    hub.close()
    reopened = FleetHub.open(tmp_path / "hub")
    reopened.resume()
    assert consumer(reopened, spec).plan().shard_ids == ()
    assert len(calls) == 1
    assert reopened.snapshot()["consumer_receipts"][0]["status"] == "unknown_effect"


def test_revoked_between_plan_and_callback_runs_no_callback(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(spec)
    encoded = shard(spec)
    hub.ingest(encoded.manifest, encoded.payload)
    learner = consumer(hub, spec)
    plan = learner.plan()
    hub.revoke_source(spec.source_id, spec.source_epoch)
    calls = []
    with pytest.raises(FleetError, match="revoked"):
        learner.consume_one(plan, lambda value, ticket: calls.append(ticket))
    assert calls == []


def test_post_callback_journal_failure_and_revocation_cannot_falsely_consume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(spec)
    encoded = shard(spec)
    hub.ingest(encoded.manifest, encoded.payload)
    learner = consumer(hub, spec)
    plan = learner.plan()
    original = hub._connection
    called = []
    fault = [True]

    @contextmanager
    def failing(*, write=False):
        try:
            with original(write=write) as connection:
                yield connection
                if write and called and fault[0]:
                    fault[0] = False
                    raise sqlite3.OperationalError("synthetic commit failure")
        except sqlite3.OperationalError:
            FleetHub.open(hub.root).revoke_source(spec.source_id, spec.source_epoch)
            raise

    monkeypatch.setattr(hub, "_connection", failing)

    def learn(value, ticket):
        called.append(ticket.ticket_id)
        return LearnerResult(2)

    receipt = learner.consume_one(plan, learn)
    assert receipt.status == "unknown_effect"
    assert receipt.callback_completed
    assert len(called) == 1
    assert learner.plan().shard_ids == ()
    assert hub.snapshot()["consumer_receipts"][0]["status"] == "unknown_effect"


def test_real_owned_child_exit_in_callback_preserves_unknown_no_replay(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(spec)
    encoded = shard(spec)
    hub.ingest(encoded.manifest, encoded.payload)
    marker = tmp_path / "called.txt"
    code = """
import json, os, sys
from pathlib import Path
from game_learning_runtime.collector import BoundedActorQueue
from game_learning_runtime.fleet_datahub import FleetHub
from game_learning_runtime.fleet_payload import SourceSpec
from game_learning_runtime.fleet_learner import FleetConsumer, LearnerSelection
hub = FleetHub.open(Path(sys.argv[1]))
with hub._connection() as connection:
    record = connection.execute('SELECT spec_json FROM sources').fetchone()[0]
    source = SourceSpec.from_record(json.loads(record))
selection = LearnerSelection(source.compatibility, source.behavior_policy_sha256, source.game_id,
    source.runtime_source_commit, source.adapter_source_sha256, allow_simulated=True)
consumer = FleetConsumer(hub, BoundedActorQueue(1, overflow_policy='fail'),
    learner_id='learner-a', selection=selection)
def crash(unroll, ticket):
    Path(sys.argv[2]).write_text('called-once', encoding='utf-8')
    os._exit(86)
consumer.consume_one(consumer.plan(), crash)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(hub.root), str(marker)],
        cwd=tmp_path,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 86
    assert marker.read_text(encoding="utf-8") == "called-once"
    reopened = FleetHub.open(hub.root)
    assert reopened.resume()["unknown_effect_shards"] == 1
    assert consumer(reopened, spec).plan().shard_ids == ()
    assert reopened.snapshot()["consumer_receipts"][0]["status"] == "unknown_effect"


def test_purge_keeps_tombstone_and_never_deletes_external_weights(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub")
    hub.register_source(spec)
    encoded = shard(spec)
    receipt = hub.ingest(encoded.manifest, encoded.payload)
    weights = tmp_path / "owner-weights.bin"
    weights.write_bytes(b"synthetic-owned-weights")
    hub.purge_shard(receipt.shard_id)
    assert weights.read_bytes() == b"synthetic-owned-weights"
    assert hub.begin_upload(encoded.manifest).status == "purged"
    with pytest.raises(FleetError):
        hub.ingest(encoded.manifest, encoded.payload)
    assert consumer(hub, spec).plan().shard_ids == ()


def test_no_data_plans_are_not_persisted_and_valid_plan_quota_is_bounded(tmp_path: Path) -> None:
    spec = source()
    hub = FleetHub.create(tmp_path / "hub", limits=FleetLimits(max_plans=1))
    hub.register_source(spec)
    learner = consumer(hub, spec)
    for _ in range(20):
        assert learner.plan().shard_ids == ()
    with hub._connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0] == 0
    encoded = shard(spec)
    hub.ingest(encoded.manifest, encoded.payload)
    first = learner.plan()
    assert learner.plan().plan_id == first.plan_id
    learner.consume_one(first, lambda value, ticket: LearnerResult(2))
    second = shard(spec, seq=1, episode=2, offset=2)
    hub.ingest(second.manifest, second.payload)
    with pytest.raises(FleetError, match="plan_quota"):
        learner.plan()


def test_explicit_spool_sync_uses_real_files_and_snapshot_has_no_raw_paths(tmp_path: Path) -> None:
    spec = source()
    limits = FleetLimits(max_chunk_bytes=512)
    hub = FleetHub.create(tmp_path / "hub", limits=limits)
    hub.register_source(spec)
    write_local_shard(tmp_path / "producer", shard(spec, limits=limits), limits=limits)
    receipt = hub.sync_local_spools((tmp_path / "producer",))
    assert receipt.completed_shards == 1
    assert hub.sync_local_spools((tmp_path / "producer",)).duplicate_shards == 1
    file = hub.write_snapshot()
    data = json.loads(file.read_text(encoding="utf-8"))
    assert data["machines"][0]["simulated"]
    assert data["machines"][0]["heartbeat_received_at_utc"] is None
    assert str(tmp_path) not in json.dumps(data)
    assert "observation" not in json.dumps(data)
