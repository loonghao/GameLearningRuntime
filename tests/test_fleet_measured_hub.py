"""Measured storage/recovery checks using explicit public synthetic fixtures."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from game_learning_runtime.collector import BoundedActorQueue
from game_learning_runtime.contracts import Unroll
from game_learning_runtime.examples.measured_fleet_training import (
    SyntheticMeasuredFixture,
    build_synthetic_measured_fixture,
)
from game_learning_runtime.fleet_datahub import (
    FleetHub,
    MeasuredEvaluationCase,
    MeasuredEvaluationMetric,
    MeasuredEvaluationSuite,
)
from game_learning_runtime.fleet_learner import (
    FleetConsumer,
    LearnerSelection,
    RealTrainingEnablement,
)
from game_learning_runtime.fleet_measured import (
    MeasuredAuthority,
    encode_measured_shard,
    verify_measured,
)
from game_learning_runtime.fleet_payload import (
    FleetError,
    FleetLimits,
    decode_shard,
    encode_shard,
    parse_manifest,
    sha256,
)

NOW = 1_700_000_000_000


def _fixture(
    *,
    source_id: str = "producer-a",
    split: str = "train",
    start: int = 0,
    limits: FleetLimits | None = None,
) -> SyntheticMeasuredFixture:
    return build_synthetic_measured_fixture(
        runtime_source_commit="4" * 40,
        produced_at_utc_ms=NOW,
        source_id=source_id,
        split=split,
        start=start,
        limits=limits,
    )


def _hub(
    tmp_path: Path,
    fixture: SyntheticMeasuredFixture,
    *,
    limits: FleetLimits | None = None,
) -> FleetHub:
    hub = FleetHub.create(
        tmp_path / "hub",
        clock_ms=lambda: NOW,
        measured_authority=fixture.authority,
        measured_destination=fixture.destination,
        limits=limits or FleetLimits(),
    )
    hub.register_source(fixture.source)
    return hub


def _suite(
    fixture: SyntheticMeasuredFixture, *, artifact: bytes = b"raise RuntimeError('never execute')"
) -> MeasuredEvaluationSuite:
    manifest = parse_manifest(fixture.shard.manifest)
    return MeasuredEvaluationSuite(
        "synthetic-fixed-suite",
        fixture.grant.evaluation_domain_id,
        "synthetic_contract_fixture",
        "7" * 40,
        artifact,
        (
            MeasuredEvaluationCase(
                "case-a",
                fixture.source.source_id,
                fixture.source.source_epoch,
                manifest.shard_id,
                manifest.payload_sha256,
                sha256(fixture.envelope),
            ),
        ),
        (MeasuredEvaluationMetric("reward", "sum", "maximize", len(fixture.unroll.transitions)),),
    )


def test_partial_measured_upload_reopen_forgets_ram_trust_and_resumes_exact_proof(
    tmp_path: Path,
) -> None:
    limits = FleetLimits(max_chunk_bytes=512)
    fixture = _fixture(limits=limits)
    hub = _hub(tmp_path, fixture, limits=limits)
    upload = hub.begin_measured_upload(fixture.shard.manifest, fixture.envelope)
    assert len(fixture.shard.chunks) > 1
    hub.put_chunk(upload.shard_id, 0, fixture.shard.chunks[0])
    hub.close()
    closed = FleetHub.open(tmp_path / "hub", clock_ms=lambda: NOW)
    with pytest.raises(FleetError, match=r"^measured_authority_missing$"):
        closed.begin_measured_upload(fixture.shard.manifest, fixture.envelope)
    closed.close()
    resumed = FleetHub.open(
        tmp_path / "hub",
        clock_ms=lambda: NOW,
        measured_authority=fixture.authority,
        measured_destination=fixture.destination,
    )
    duplicate = resumed.begin_measured_upload(fixture.shard.manifest, fixture.envelope)
    assert duplicate.duplicate and duplicate.next_chunk_index == 1
    for index in range(1, len(fixture.shard.chunks)):
        resumed.put_chunk(upload.shard_id, index, fixture.shard.chunks[index])
    receipt = resumed.finish_upload(upload.shard_id)
    assert receipt.status == "ready" and receipt.trust_level == "measured_authenticated"
    assert not fixture.source.simulated
    selection = LearnerSelection(
        fixture.source.compatibility,
        fixture.source.behavior_policy_sha256,
        fixture.source.game_id,
        fixture.source.runtime_source_commit,
        fixture.source.adapter_source_sha256,
    )
    consumer = FleetConsumer(
        resumed,
        BoundedActorQueue(1, overflow_policy="fail"),
        learner_id="fixture-learner",
        selection=selection,
    )
    assert consumer.plan().shard_ids == ()
    assert resumed.snapshot()["consumer_receipts"] == []


def test_proof_bytes_are_reserved_before_any_upload_files_or_shard_row(tmp_path: Path) -> None:
    fixture = _fixture()
    requested = len(fixture.shard.manifest) + len(fixture.shard.payload) + len(fixture.envelope)
    limits = FleetLimits(max_retained_bytes=requested - 1)
    hub = _hub(tmp_path, fixture, limits=limits)
    with pytest.raises(FleetError, match=r"^retained_byte_quota$"):
        hub.begin_measured_upload(fixture.shard.manifest, fixture.envelope)
    with sqlite3.connect(hub.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM shards").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM measured_proofs").fetchone()[0] == 0
    assert list((hub.root / "artifacts").iterdir()) == []


def test_proof_over_limit_fails_before_json_parse_and_reservation(tmp_path: Path) -> None:
    fixture = _fixture()
    hub = _hub(tmp_path, fixture)
    with pytest.raises(FleetError, match=r"^measured_proof_byte_limit$"):
        hub.begin_measured_upload(fixture.shard.manifest, b"x" * (hub.limits.max_shard_bytes + 1))
    assert list((hub.root / "artifacts").iterdir()) == []


def test_duplicate_reception_preserves_original_data_receipt_time_and_proof(tmp_path: Path) -> None:
    fixture = _fixture()
    hub = _hub(tmp_path, fixture)
    first = hub.ingest_measured(fixture.shard.manifest, fixture.shard.payload, fixture.envelope)
    before = hub.snapshot()["machines"][0]["data_received_at_utc"]
    hub.close()
    later = FleetHub.open(
        tmp_path / "hub",
        clock_ms=lambda: NOW + 1000,
        measured_authority=fixture.authority,
        measured_destination=fixture.destination,
    )
    second = later.ingest_measured(fixture.shard.manifest, fixture.shard.payload, fixture.envelope)
    assert first.status == second.status == "ready" and second.duplicate
    assert later.snapshot()["machines"][0]["data_received_at_utc"] == before
    with sqlite3.connect(later.database) as connection:
        row = connection.execute("SELECT envelope FROM measured_proofs").fetchone()
        assert bytes(row[0]) == fixture.envelope


def test_proof_storage_tampering_is_rejected_on_durable_read(tmp_path: Path) -> None:
    fixture = _fixture()
    hub = _hub(tmp_path, fixture)
    receipt = hub.ingest_measured(fixture.shard.manifest, fixture.shard.payload, fixture.envelope)
    with sqlite3.connect(hub.database) as connection:
        connection.execute("UPDATE measured_proofs SET envelope=?", (fixture.envelope[:-1] + b"x",))
    with pytest.raises(FleetError, match=r"^stored_measured_proof_integrity$"):
        hub.finish_upload(receipt.shard_id)
    assert hub.snapshot()["consumer_receipts"] == []


def test_existing_v1_real_carrier_cannot_be_promoted_by_attaching_a_proof(tmp_path: Path) -> None:
    fixture = _fixture()
    hub = _hub(tmp_path, fixture)
    old = hub.ingest(fixture.shard.manifest, fixture.shard.payload)
    assert old.status == "quarantine"
    with pytest.raises(FleetError, match=r"^measured_proof_conflict$"):
        hub.ingest_measured(fixture.shard.manifest, fixture.shard.payload, fixture.envelope)
    assert hub.snapshot()["consumer_receipts"] == []


def test_actual_suite_artifact_and_metric_definition_are_frozen_without_execution(
    tmp_path: Path,
) -> None:
    fixture = _fixture(split="evaluation_holdout")
    hub = _hub(tmp_path, fixture)
    hub.ingest_measured(fixture.shard.manifest, fixture.shard.payload, fixture.envelope)
    suite = _suite(fixture)
    sources = ((fixture.source.source_id, fixture.source.source_epoch),)
    snapshot = hub.freeze_measured_evaluation("evaluation-a", sources, suite=suite)
    assert snapshot.suite_sha256 == suite.sha256
    assert hub.freeze_measured_evaluation("evaluation-a", sources, suite=suite) == snapshot
    modified = replace(suite, metrics=(replace(suite.metrics[0], direction="minimize"),))
    with pytest.raises(FleetError, match=r"^measured_evaluation_conflict$"):
        hub.freeze_measured_evaluation("evaluation-a", sources, suite=modified)
    with sqlite3.connect(hub.database) as connection:
        row = connection.execute(
            "SELECT evaluator_artifact,suite_sha256 FROM measured_evaluations"
        ).fetchone()
        assert bytes(row[0]) == suite.evaluator_artifact and row[1] == suite.sha256


def test_suite_false_case_hash_cannot_freeze_actual_heldout_data(tmp_path: Path) -> None:
    fixture = _fixture(split="evaluation_holdout")
    hub = _hub(tmp_path, fixture)
    hub.ingest_measured(fixture.shard.manifest, fixture.shard.payload, fixture.envelope)
    suite = _suite(fixture)
    wrong = replace(suite, cases=(replace(suite.cases[0], payload_sha256="f" * 64),))
    with pytest.raises(FleetError, match=r"^measured_evaluation_case_binding$"):
        hub.freeze_measured_evaluation(
            "evaluation-a", ((fixture.source.source_id, fixture.source.source_epoch),), suite=wrong
        )
    with sqlite3.connect(hub.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM measured_evaluations").fetchone()[0] == 0
        assert connection.execute("SELECT eval_frozen FROM sources").fetchone()[0] == 0


def test_evaluator_artifact_quota_rolls_back_freeze_without_hiding_ready_data(
    tmp_path: Path,
) -> None:
    fixture = _fixture(split="evaluation_holdout")
    requested = len(fixture.shard.manifest) + len(fixture.shard.payload) + len(fixture.envelope)
    hub = _hub(tmp_path, fixture, limits=FleetLimits(max_retained_bytes=requested + 4096))
    hub.ingest_measured(fixture.shard.manifest, fixture.shard.payload, fixture.envelope)
    with pytest.raises(FleetError, match=r"^retained_byte_quota$"):
        hub.freeze_measured_evaluation(
            "evaluation-a",
            ((fixture.source.source_id, fixture.source.source_epoch),),
            suite=_suite(fixture, artifact=b"fixture-evaluator" * 1024),
        )
    with sqlite3.connect(hub.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM measured_evaluations").fetchone()[0] == 0
        assert connection.execute("SELECT eval_frozen FROM sources").fetchone()[0] == 0
        assert connection.execute("SELECT status FROM shards").fetchone()[0] == "ready"


@pytest.mark.parametrize("same_game", [True, False])
def test_numeric_v1_sim_copy_cannot_relabel_a_measured_holdout_into_training(
    tmp_path: Path,
    same_game: bool,
) -> None:
    heldout = _fixture(split="evaluation_holdout")
    hub = _hub(tmp_path, heldout)
    hub.ingest_measured(heldout.shard.manifest, heldout.shard.payload, heldout.envelope)
    alias = replace(
        heldout.source,
        source_id="numeric-alias",
        source_epoch="new-epoch",
        run_id="alias-run",
        assignment_id="alias-assignment",
        simulated=True,
        split="train",
        runtime_source_commit="a" * 40,
        adapter_source_sha256="b" * 64,
        behavior_policy_sha256="c" * 64,
        game_id=heldout.source.game_id if same_game else "independent-game",
    )
    numeric = decode_shard(heldout.shard.manifest, heldout.shard.payload).unroll
    episode = uuid5(NAMESPACE_URL, "independent synthetic numeric alias")
    alias_unroll = Unroll(
        tuple(replace(item, episode_id=episode) for item in numeric.transitions),
        alias.source_id,
        0,
        alias.policy_version,
        environment_config_digest=alias.compatibility.environment_config_sha256,
    )
    shard = encode_shard(alias, shard_seq=0, unroll=alias_unroll, produced_at_utc_ms=NOW)
    hub.register_source(alias)
    receipt = hub.ingest(shard.manifest, shard.payload)
    assert receipt.status == ("quarantine" if same_game else "ready")


def test_measured_original_actor_metadata_and_queue_epoch_namespace_are_distinct(
    tmp_path: Path,
) -> None:
    original = _fixture(source_id="same-actor")
    successor_fixture = _fixture(source_id="next-actor", start=50)
    successor = replace(
        successor_fixture.source, source_id=original.source.source_id, source_epoch="second-epoch"
    )
    successor_grant = replace(successor_fixture.grant, source=successor)
    # This publicly known synthetic fixture key never grants production trust.
    key = b"GLR public SYNTHETIC contract fixture only; never production"
    authority = MeasuredAuthority(
        "synthetic-epoch-test", (original.grant, successor_grant), {original.grant.key_id: key}
    )
    unroll = replace(successor_fixture.unroll, actor_id=successor.source_id)
    packet = encode_measured_shard(
        successor,
        unroll,
        grant=successor_grant,
        key=key,
        shard_seq=0,
        produced_at_utc_ms=NOW,
        expires_at_utc_ms=NOW + 30000,
        steps=successor_fixture.steps,
    )
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW, measured_authority=authority)
    queue = BoundedActorQueue(1, overflow_policy="fail")
    for source, carrier, envelope in (
        (original.source, original.shard, original.envelope),
        (successor, packet.carrier, packet.envelope),
    ):
        if source == successor:
            hub.revoke_source(original.source.source_id, original.source.source_epoch)
        hub.register_source(source)
        receipt = hub.ingest_measured(carrier.manifest, carrier.payload, envelope)
        proof = verify_measured(
            envelope, decode_shard(carrier.manifest, carrier.payload), authority, NOW
        )
        assert proof.decoded.unroll.actor_id == source.source_id
        with hub._connection() as connection:
            row = connection.execute(
                "SELECT * FROM shards WHERE shard_id=?", (receipt.shard_id,)
            ).fetchone()
        projected = hub._learner_decoded(row).unroll
        assert projected.actor_id == f"{source.source_id}:{source.source_epoch}"
        assert (
            projected.transitions[0].action_receipt
            == proof.decoded.unroll.transitions[0].action_receipt
        )
        queue.put_nowait(projected)
        queue.commit(queue.get_nowait())
    assert queue.metrics().committed_unrolls == 2


def test_new_internal_real_refusal_projects_to_closed_snapshot_v1_reason(tmp_path: Path) -> None:
    fixture = _fixture()
    hub = _hub(tmp_path, fixture)
    hub.ingest_measured(fixture.shard.manifest, fixture.shard.payload, fixture.envelope)
    enablement = RealTrainingEnablement(
        fixture.destination.destination_id,
        fixture.destination.destination_sha256,
        fixture.authority.sha256,
        (fixture.grant.source_spec_sha256,),
        "absent-evaluation",
        "a" * 64,
        "b" * 64,
        "fixture-approval",
        "c" * 64,
        NOW + 30000,
        "synthetic_contract_fixture",
        3,
        1,
    )
    selection = LearnerSelection(
        fixture.source.compatibility,
        fixture.source.behavior_policy_sha256,
        fixture.source.game_id,
        fixture.source.runtime_source_commit,
        fixture.source.adapter_source_sha256,
    )
    consumer = FleetConsumer(
        hub,
        BoundedActorQueue(1, overflow_policy="fail"),
        learner_id="fixture-learner",
        selection=selection,
        real_enablement=enablement,
    )
    assert consumer.plan().shard_ids == ()
    assert hub.snapshot()["datasets"][0]["last_plan_reason_codes"] == ["quarantine"]
    with sqlite3.connect(hub.database) as connection:
        assert (
            connection.execute("SELECT last_plan_reasons FROM sources").fetchone()[0]
            == '["measured_evaluation_missing"]'
        )
        assert connection.execute("SELECT COUNT(*) FROM measured_attempts").fetchone()[0] == 0
