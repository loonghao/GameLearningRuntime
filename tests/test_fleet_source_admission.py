"""Independent black-box admission checks using finite synthetic vector sources."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest

from game_learning_runtime.collector import BoundedActorQueue
from game_learning_runtime.contracts import Transition, Unroll
from game_learning_runtime.fleet_datahub import FleetHub
from game_learning_runtime.fleet_learner import FleetConsumer, LearnerResult, LearnerSelection
from game_learning_runtime.fleet_payload import (
    CompatibilitySpec,
    EncodedShard,
    FleetError,
    SourceSpec,
    VectorSpec,
    encode_shard,
)

NOW_MS = 1_700_000_000_000


def _source(**changes: Any) -> SourceSpec:
    compatibility = CompatibilitySpec(
        environment_id="synthetic.vector",
        protocol_version="1.0",
        environment_config_sha256="1" * 64,
        observation=(VectorSpec("state", "<f4", 1),),
        action=(VectorSpec("choice", "<i4", 1),),
        reward_contract_sha256="2" * 64,
    )
    return replace(
        SourceSpec(
            source_id="producer-a",
            source_epoch="epoch-1",
            machine_id="simulated-a",
            source_revision="fixture-v1",
            source_sha256="3" * 64,
            runtime_source_commit="4" * 40,
            adapter_source_sha256="5" * 64,
            run_id="run-a",
            game_id="synthetic-vector",
            compatibility=compatibility,
            policy_epoch="policy-epoch-1",
            policy_version=7,
            behavior_policy_sha256="6" * 64,
            assignment_id="assignment-a",
            simulated=True,
        ),
        **changes,
    )


def _shard(source: SourceSpec) -> EncodedShard:
    transition = Transition(
        observation={"state": np.array([2.0], dtype=np.float32)},
        action={"choice": np.array([1], dtype=np.int32)},
        reward=np.array([5.0], dtype=np.float32),
        next_observation={"state": np.array([3.0], dtype=np.float32)},
        terminated=np.array([False]),
        truncated=np.array([False]),
        episode_id=uuid5(NAMESPACE_URL, source.run_id),
        step_id=0,
        timestamp_ns=NOW_MS * 1_000_000,
    )
    unroll = Unroll(
        (transition,),
        actor_id=source.source_id,
        sequence_id=0,
        policy_version=source.policy_version,
        environment_config_digest=source.compatibility.environment_config_sha256,
    )
    return encode_shard(source, shard_seq=0, unroll=unroll, produced_at_utc_ms=NOW_MS)


def _selection(source: SourceSpec, **changes: Any) -> LearnerSelection:
    return replace(
        LearnerSelection(
            compatibility=source.compatibility,
            policy_sha256=source.behavior_policy_sha256,
            game_id=source.game_id,
            runtime_source_commit=source.runtime_source_commit,
            adapter_source_sha256=source.adapter_source_sha256,
            allow_simulated=True,
        ),
        **changes,
    )


def _ready(tmp_path: Path, source: SourceSpec) -> FleetHub:
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS)
    hub.register_source(source)
    shard = _shard(source)
    hub.ingest(shard.manifest, shard.payload)
    return hub


def _consumer(hub: FleetHub, selection: LearnerSelection) -> FleetConsumer:
    return FleetConsumer(
        hub,
        BoundedActorQueue(1, overflow_policy="fail"),
        learner_id="synthetic-learner",
        selection=selection,
    )


def _assert_not_consumed(hub: FleetHub, selection: LearnerSelection, reason: str) -> None:
    calls = []

    def learner(unroll: Unroll, ticket: Any) -> LearnerResult:
        calls.append(ticket.source_id)
        return LearnerResult(declared_updates=len(unroll.transitions))

    consumer = _consumer(hub, selection)
    plan = consumer.plan()
    assert plan.shard_ids == ()
    with pytest.raises(FleetError, match=r"^no_ready_data$"):
        consumer.consume_one(plan, learner)
    assert calls == []
    snapshot = hub.snapshot()
    assert snapshot["datasets"][0]["last_plan_reason_codes"] == [reason]
    assert snapshot["consumer_receipts"] == []


def test_exact_registration_survives_reopen_without_heartbeat(tmp_path: Path) -> None:
    source = _source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS)
    hub.register_source(source)
    hub.register_source(source)
    hub.close()
    reopened = FleetHub.open(tmp_path / "hub", clock_ms=lambda: NOW_MS)
    reopened.register_source(source)
    snapshot = reopened.snapshot()
    assert len(snapshot["machines"]) == 1
    machine = snapshot["machines"][0]
    assert machine["source_epoch"] == "epoch-1"
    assert machine["run_id"] == source.run_id
    assert machine["declared_status"] == "unknown"
    assert machine["heartbeat_received_at_utc"] is None
    assert machine["data_received_at_utc"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "run-b"},
        {"machine_id": "simulated-b"},
        {"source_revision": "fixture-v2"},
        {"source_sha256": "a" * 64},
        {"policy_epoch": "policy-epoch-2"},
        {"policy_version": 8},
        {"behavior_policy_sha256": "b" * 64},
        {"assignment_id": "assignment-b"},
        {"split": "evaluation_holdout"},
        {"simulated": False},
        {"checkpoint_sha256": "c" * 64},
    ],
)
def test_registered_source_fields_cannot_be_rewritten(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    source = _source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS)
    hub.register_source(source)
    with pytest.raises(FleetError, match=r"^source_conflict$"):
        hub.register_source(replace(source, **change))
    assert len(hub.snapshot()["machines"]) == 1


def test_new_epoch_needs_revocation_and_retains_old_identity(tmp_path: Path) -> None:
    source = _source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS)
    hub.register_source(source)
    successor = replace(source, source_epoch="epoch-2", run_id="run-b")
    with pytest.raises(FleetError, match=r"^source_epoch_busy$"):
        hub.register_source(successor)
    hub.revoke_source(source.source_id, source.source_epoch)
    hub.register_source(successor)
    rows = {row["source_epoch"]: row for row in hub.snapshot()["machines"]}
    assert rows["epoch-1"]["revoked"] is True
    assert rows["epoch-2"]["revoked"] is False
    assert rows["epoch-2"]["heartbeat_received_at_utc"] is None
    with pytest.raises(FleetError, match=r"^revoked$"):
        hub.receive_heartbeat(source.source_id, source.source_epoch, 0, declared_at_utc_ms=NOW_MS)


@pytest.mark.parametrize(
    "changes",
    [
        {"split": "evaluation_holdout"},
        {"assignment_id": "assignment-b"},
        {"game_id": "synthetic-other"},
    ],
)
def test_run_assignment_cannot_change_through_another_source(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    source = _source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS)
    hub.register_source(source)
    renamed = replace(source, source_id="producer-b", **changes)
    with pytest.raises(FleetError, match=r"^run_assignment_conflict$"):
        hub.register_source(renamed)
    assert [row["source_id"] for row in hub.snapshot()["machines"]] == [source.source_id]


@pytest.mark.parametrize(
    "registered_changes,spoof",
    [
        ({"simulated": False}, {"simulated": True}),
        ({"split": "evaluation_holdout"}, {"split": "train"}),
        ({}, {"source_id": "unregistered"}),
        ({}, {"source_epoch": "epoch-2"}),
        ({}, {"run_id": "run-b"}),
        ({}, {"machine_id": "simulated-b"}),
        ({}, {"source_sha256": "a" * 64}),
        ({}, {"behavior_policy_sha256": "b" * 64}),
    ],
)
def test_upload_manifest_cannot_spoof_registration(
    tmp_path: Path, registered_changes: dict[str, Any], spoof: dict[str, Any]
) -> None:
    source = _source(**registered_changes)
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS)
    hub.register_source(source)
    shard = _shard(source)
    record = json.loads(shard.manifest)
    record["source"].update(spoof)
    forged = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    with pytest.raises(FleetError, match=r"^source_not_admitted$"):
        hub.ingest(forged, shard.payload)
    assert all(row["accepted_transition_count"] == 0 for row in hub.snapshot()["datasets"])
    assert hub.snapshot()["consumer_receipts"] == []


def test_heartbeat_forward_receipt_is_distinct_from_declaration_and_replay(tmp_path: Path) -> None:
    clock = [NOW_MS]
    source = _source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: clock[0])
    hub.register_source(source)
    assert hub.receive_heartbeat(
        source.source_id,
        source.source_epoch,
        0,
        declared_at_utc_ms=NOW_MS - 1000,
        declared_status="running",
    )
    before = hub.snapshot()["machines"][0]
    assert before["heartbeat_declared_at_utc"] == "2023-11-14T22:13:19.000Z"
    assert before["heartbeat_received_at_utc"] == "2023-11-14T22:13:20.000Z"
    clock[0] += 5000
    assert (
        hub.receive_heartbeat(
            source.source_id,
            source.source_epoch,
            0,
            declared_at_utc_ms=NOW_MS - 1000,
            declared_status="running",
        )
        is False
    )
    replay = hub.snapshot()["machines"][0]
    assert replay == before
    assert hub.receive_heartbeat(
        source.source_id,
        source.source_epoch,
        1,
        declared_at_utc_ms=NOW_MS + 4000,
        declared_status="stopped",
    )
    forwarded = hub.snapshot()["machines"][0]
    assert forwarded["heartbeat_received_at_utc"] == "2023-11-14T22:13:25.000Z"
    assert forwarded["heartbeat_declared_at_utc"] == "2023-11-14T22:13:24.000Z"
    assert forwarded["declared_status"] == "stopped"
    assert forwarded["data_received_at_utc"] is None


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("first_gap", "heartbeat_sequence"),
        ("later_gap", "heartbeat_sequence"),
        ("old_replay", "heartbeat_sequence"),
        ("changed_replay", "heartbeat_conflict"),
        ("future", "future_timestamp"),
        ("rollback", "clock_mismatch"),
        ("invalid_utc", "invalid_utc"),
    ],
)
def test_invalid_heartbeat_does_not_change_received_evidence(
    tmp_path: Path, mode: str, expected: str
) -> None:
    clock = [NOW_MS]
    source = _source()
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: clock[0])
    hub.register_source(source)
    seq = 0
    declared = NOW_MS - 1000
    status = "running"
    if mode != "first_gap":
        hub.receive_heartbeat(
            source.source_id,
            source.source_epoch,
            0,
            declared_at_utc_ms=declared,
            declared_status=status,
        )
    if mode == "first_gap":
        seq = 1
    elif mode == "later_gap":
        seq = 2
    elif mode == "old_replay":
        hub.receive_heartbeat(
            source.source_id,
            source.source_epoch,
            1,
            declared_at_utc_ms=declared,
            declared_status=status,
        )
    elif mode == "changed_replay":
        status = "stopped"
    elif mode == "future":
        seq, declared = 1, NOW_MS + 1
    elif mode == "rollback":
        clock[0], seq = NOW_MS - 500, 1
    elif mode == "invalid_utc":
        seq, declared = 1, 253_402_300_800_000
    before = hub.snapshot()["machines"][0]
    with pytest.raises(FleetError, match=rf"^{expected}$"):
        hub.receive_heartbeat(
            source.source_id,
            source.source_epoch,
            seq,
            declared_at_utc_ms=declared,
            declared_status=status,
        )
    assert hub.snapshot()["machines"][0] == before


def test_exact_cohort_and_behavior_artifact_reach_actual_callback_once(tmp_path: Path) -> None:
    source = _source()
    hub = _ready(tmp_path, source)
    consumer = _consumer(hub, _selection(source))
    calls = []

    def learner(unroll: Unroll, ticket: Any) -> LearnerResult:
        calls.append(ticket.source_id)
        assert unroll.policy_version == 7
        assert unroll.transitions[0].observation["state"].tolist() == [2.0]
        assert unroll.transitions[0].reward.tolist() == [5.0]
        return LearnerResult(declared_updates=1)

    plan = consumer.plan()
    receipt = consumer.consume_one(plan, learner)
    assert calls == [source.source_id]
    assert receipt.status == "consumed"
    assert receipt.callback_completed is True
    assert receipt.transition_count == 1
    with pytest.raises(FleetError, match=r"^already_claimed$"):
        consumer.consume_one(plan, learner)
    assert calls == [source.source_id]


@pytest.mark.parametrize(
    "field",
    [
        "game",
        "runtime",
        "adapter",
        "environment",
        "protocol",
        "config",
        "observation",
        "action",
        "reward",
        "reward_dtype",
        "mask",
    ],
)
def test_incompatible_cohort_never_invokes_learner(tmp_path: Path, field: str) -> None:
    source = _source()
    hub = _ready(tmp_path, source)
    changes: dict[str, Any] = {}
    compatibility_changes: dict[str, Any] = {}
    if field == "game":
        changes["game_id"] = "synthetic-other"
    elif field == "runtime":
        changes["runtime_source_commit"] = "a" * 40
    elif field == "adapter":
        changes["adapter_source_sha256"] = "a" * 64
    elif field == "environment":
        compatibility_changes["environment_id"] = "synthetic.other"
    elif field == "protocol":
        compatibility_changes["protocol_version"] = "2.0"
    elif field == "config":
        compatibility_changes["environment_config_sha256"] = "a" * 64
    elif field == "observation":
        compatibility_changes["observation"] = (VectorSpec("state", "<f4", 2),)
    elif field == "action":
        compatibility_changes["action"] = (VectorSpec("choice", "<i8", 1),)
    elif field == "reward":
        compatibility_changes["reward_contract_sha256"] = "a" * 64
    elif field == "reward_dtype":
        compatibility_changes["reward_dtype"] = "<f8"
    elif field == "mask":
        compatibility_changes["masks"] = (VectorSpec("choice", "|b1", 1),)
    if compatibility_changes:
        changes["compatibility"] = replace(source.compatibility, **compatibility_changes)
    _assert_not_consumed(hub, _selection(source, **changes), "compatibility_mismatch")


def test_same_policy_version_with_different_artifact_is_not_on_policy(tmp_path: Path) -> None:
    source = _source(policy_version=7)
    hub = _ready(tmp_path, source)
    assert source.policy_version == 7
    _assert_not_consumed(hub, _selection(source, policy_sha256="a" * 64), "policy_mismatch")


@pytest.mark.parametrize(
    "missing",
    ["algorithm", "algorithm_sha256", "allowed_source_ids", "allowed_behavior_policy_sha256s"],
)
def test_off_policy_requires_declared_algorithm_and_both_allowlists(missing: str) -> None:
    source = _source()
    contract = {
        "mode": "off_policy",
        "algorithm": "synthetic-regression-v1",
        "algorithm_sha256": "a" * 64,
        "allowed_source_ids": (source.source_id,),
        "allowed_behavior_policy_sha256s": (source.behavior_policy_sha256,),
    }
    contract[missing] = () if missing.startswith("allowed_") else None
    with pytest.raises(FleetError, match=r"^off_policy_requires_explicit_contract$"):
        _selection(source, **contract)


@pytest.mark.parametrize("wrong", ["source", "behavior"])
def test_off_policy_contract_does_not_authorize_unlisted_data(tmp_path: Path, wrong: str) -> None:
    source = _source()
    hub = _ready(tmp_path, source)
    selection = _selection(
        source,
        mode="off_policy",
        policy_sha256="b" * 64,
        algorithm="synthetic-regression-v1",
        algorithm_sha256="a" * 64,
        allowed_source_ids=("other-source",) if wrong == "source" else (source.source_id,),
        allowed_behavior_policy_sha256s=("c" * 64,)
        if wrong == "behavior"
        else (source.behavior_policy_sha256,),
    )
    _assert_not_consumed(hub, selection, "off_policy_not_allowed")


def test_explicit_off_policy_contract_admits_named_behavior_artifact(tmp_path: Path) -> None:
    source = _source()
    hub = _ready(tmp_path, source)
    selection = _selection(
        source,
        mode="off_policy",
        policy_sha256="b" * 64,
        algorithm="synthetic-regression-v1",
        algorithm_sha256="a" * 64,
        allowed_source_ids=(source.source_id,),
        allowed_behavior_policy_sha256s=(source.behavior_policy_sha256,),
    )
    calls = []
    consumer = _consumer(hub, selection)

    def learner(unroll: Unroll, ticket: Any) -> LearnerResult:
        calls.append((ticket.source_id, len(unroll.transitions)))
        return LearnerResult(declared_updates=1)

    receipt = consumer.consume_one(consumer.plan(), learner)
    assert calls == [(source.source_id, 1)]
    assert receipt.status == "consumed"


@pytest.mark.parametrize(
    "source_changes,selection_changes,reason",
    [
        ({}, {"allow_simulated": False}, "simulated_not_allowed"),
        ({"simulated": False}, {}, "quarantine"),
        ({"split": "quarantine"}, {}, "quarantine"),
        ({"split": "evaluation_holdout"}, {}, "holdout"),
    ],
)
def test_simulated_allowance_and_nontraining_splits_cannot_be_bypassed(
    tmp_path: Path, source_changes: dict[str, Any], selection_changes: dict[str, Any], reason: str
) -> None:
    source = _source(**source_changes)
    hub = _ready(tmp_path, source)
    _assert_not_consumed(hub, _selection(source, **selection_changes), reason)
    if not source.simulated:
        assert hub.snapshot()["datasets"][0]["quality_status"] == "quarantine"
