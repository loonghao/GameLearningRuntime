"""Host-ledger fault cases; synthetic receipts never claim OS process control."""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from test_continuous_learning import (
    _CANDIDATE,
    _HOST,
    _claimed,
    _correctness_evidence,
    _evaluate,
    _evaluation_run,
    _hash,
    _proposal,
    _review,
    _spec,
    _stopped,
    _submit_review,
)

from game_learning_runtime.continuous_learning import (
    EVALUATION_CHECK_SOURCE,
    EVALUATION_ZERO_METRICS,
    CampaignSpec,
    CampaignStore,
    EvidenceRef,
    HostAuthority,
    HostRoleCapability,
    Proposal,
    SupervisorStopReceipt,
    TrialTicket,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.run_store import RunStatus, TrainingStore


def _claimed_as(tmp_path: Path, authority: HostAuthority) -> tuple[CampaignStore, TrialTicket]:
    campaign = CampaignStore(tmp_path / "campaigns.sqlite3", host_authority=authority)
    spec = _spec(reviewers=tuple(set(authority.reviewer_ids)))
    campaign.create(spec, now_ns=0)
    proposal = _proposal(spec)
    campaign.propose(spec.campaign_id, proposal)
    ticket = campaign.claim(
        spec.campaign_id,
        proposal.proposal_id,
        worker_id="worker",
        training_steps=1,
        wall_seconds=2,
        now_ns=100,
    )
    root = campaign.candidate_directory(ticket)
    root.mkdir(parents=True)
    (root / "candidate.bin").write_bytes(_CANDIDATE)
    return campaign, ticket


def _worker_evidence(
    tmp_path: Path,
    ticket: TrialTicket,
    *,
    status: RunStatus | None = RunStatus.SUCCEEDED,
    exit_code: int | None = 0,
    persist_event: bool = True,
    supervisor_id: str = "supervisor",
    **receipt_changes: Any,
) -> tuple[TrainingStore, SupervisorStopReceipt]:
    spec = _spec()
    store = TrainingStore(tmp_path / "worker.sqlite3")
    identity = _hash("exact owned synthetic process identity")
    run = store.create_run(
        environment_id=spec.environment_id,
        protocol_version=spec.protocol_version,
        kind="campaign-worker",
        environment_config_digest=spec.environment_config_sha256,
        started_at_ns=ticket.started_at_ns,
        metadata={
            "campaign_trial_id": ticket.trial_id,
            "campaign_token": ticket.token,
            "target_id": spec.target_id,
            "worker_ids": [ticket.worker_id],
            "process_identity_sha256": identity,
        },
    )
    receipt = SupervisorStopReceipt(
        **{
            "receipt_id": "stop.synthetic",
            "supervisor_id": supervisor_id,
            "trial_id": ticket.trial_id,
            "trial_token": ticket.token,
            "worker_run_id": run.run_id,
            "worker_ids": (ticket.worker_id,),
            "environment_id": spec.environment_id,
            "protocol_version": spec.protocol_version,
            "target_id": spec.target_id,
            "environment_config_sha256": spec.environment_config_sha256,
            "process_identity_sha256": identity,
            "source_sha256": _hash("trusted observation"),
            "stopped_at_ns": 600,
            **receipt_changes,
        }
    )
    if persist_event:
        store.append_event(
            run.run_id,
            kind="supervisor.worker-stopped",
            payload=receipt.to_mapping(),
            timestamp_ns=600,
        )
    if status is not None:
        store.finish_run(run.run_id, status=status, exit_code=exit_code, finished_at_ns=600)
    return store, receipt


@pytest.mark.parametrize("operation", ["evaluate", "confirm_stopped", "review", "stop"])
def test_legacy_privileged_entry_points_cannot_use_worker_claim_as_authority(
    tmp_path: Path,
    operation: str,
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    unprivileged = CampaignStore(campaign.path)
    with pytest.raises(ContractViolation, match="host"):
        if operation == "evaluate":
            unprivileged.evaluate(
                ticket,
                store=training,
                run_id=run_id,
                evidence=[evidence, *_correctness_evidence(run_id)],
                candidate_path="candidate.bin",
                now_ns=500,
            )
        elif operation == "confirm_stopped":
            unprivileged.confirm_stopped(ticket, supervisor_receipt_sha256=_hash("caller claim"))
        elif operation == "review":
            unprivileged.review(ticket, _review(_hash("caller score")))
        else:
            unprivileged.stop(spec.campaign_id, reason="caller requested stop")
    assert campaign.snapshot(spec.campaign_id)["trials"][0]["status"] == "claimed"


def test_worker_cannot_provision_authority_over_an_existing_unowned_claim(tmp_path: Path) -> None:
    campaign = CampaignStore(tmp_path / "unowned.sqlite3")
    spec = _spec()
    campaign.create(spec, now_ns=0)
    proposal = _proposal(spec)
    campaign.propose(spec.campaign_id, proposal)
    campaign.claim(
        spec.campaign_id,
        proposal.proposal_id,
        worker_id="worker",
        training_steps=1,
        wall_seconds=1,
        now_ns=100,
    )
    with pytest.raises(ContractViolation, match="unknown"):
        CampaignStore(campaign.path, host_authority=_HOST)
    with sqlite3.connect(campaign.path) as connection:
        assert connection.execute("SELECT fingerprint FROM campaign_host").fetchone()[0] is None


def test_reopen_requires_the_original_host_secret_and_canonical_ledger(tmp_path: Path) -> None:
    campaign, _, _, ticket = _claimed(tmp_path)
    forged = replace(_HOST, secret=b"another-host-key-with-same-public-role-names")
    with pytest.raises(ContractViolation, match="owner"):
        CampaignStore(campaign.path, host_authority=forged)
    clone = tmp_path / "copy.sqlite3"
    shutil.copyfile(campaign.path, clone)
    with pytest.raises(ContractViolation, match="foreign"):
        CampaignStore(clone, host_authority=_HOST)
    reopened = CampaignStore(campaign.path, host_authority=_HOST)
    cap = campaign.role_capability(_HOST, ticket, role="evaluator", principal_id="evaluator")
    assert reopened._require_host(cap, ticket, "evaluator") == "evaluator"


@pytest.mark.parametrize(
    "changes",
    [
        {"store_epoch": "foreign"},
        {"store_binding_sha256": "0" * 64},
        {"campaign_id": "foreign"},
        {"trial_id": "foreign"},
        {"trial_token": "foreign"},
        {"role": "reviewer"},
        {"principal_id": "worker"},
        {"authority_fingerprint": "0" * 64},
        {"proof": b"0" * 32},
    ],
)
def test_role_capability_cannot_be_rebound_to_another_scope(
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    campaign, _, _, ticket = _claimed(tmp_path)
    cap = campaign.role_capability(_HOST, ticket, role="evaluator", principal_id="evaluator")
    with pytest.raises(ContractViolation, match="scope"):
        campaign._require_host(replace(cap, **changes), ticket, "evaluator")


@pytest.mark.parametrize("replacement", ["inode", "persisted-epoch", "persisted-owner"])
def test_existing_store_instance_rechecks_persisted_identity_before_cap_use(
    tmp_path: Path,
    replacement: str,
) -> None:
    campaign, _, _, ticket = _claimed(tmp_path)
    cap = campaign.role_capability(_HOST, ticket, role="evaluator", principal_id="evaluator")
    if replacement == "inode":
        old = campaign.path.with_suffix(".old")
        campaign.path.replace(old)
        shutil.copyfile(old, campaign.path)
    else:
        column = "epoch" if replacement == "persisted-epoch" else "fingerprint"
        with sqlite3.connect(campaign.path) as connection:
            connection.execute(f"UPDATE campaign_host SET {column}=?", ("changed",))
    with pytest.raises(ContractViolation, match=r"identity|authority"):
        campaign._require_host(cap, ticket, "evaluator")


@pytest.mark.parametrize("actor", ["planner", "worker"])
def test_candidate_actor_cannot_evaluate_itself_even_with_a_host_evaluator_role(
    tmp_path: Path,
    actor: str,
) -> None:
    authority = replace(_HOST, evaluator_ids=(actor,))
    campaign, ticket = _claimed_as(tmp_path, authority)
    spec = _spec()
    proposal = _proposal(spec)
    training, run_id, evidence = _evaluation_run(
        tmp_path, spec, proposal, ticket, metadata_changes={"evaluator_id": actor}
    )
    with pytest.raises(ContractViolation, match="independent"):
        campaign.evaluate(
            ticket,
            store=training,
            run_id=run_id,
            evidence=[evidence, *_correctness_evidence(run_id)],
            candidate_path="candidate.bin",
            now_ns=500,
            host=campaign.role_capability(authority, ticket, role="evaluator", principal_id=actor),
        )


@pytest.mark.parametrize("actor", ["evaluator", "supervisor"])
def test_evaluation_or_stop_actor_cannot_review_its_own_trial(tmp_path: Path, actor: str) -> None:
    authority = replace(_HOST, reviewer_ids=(actor,))
    campaign, ticket = _claimed_as(tmp_path, authority)
    spec = _spec()
    proposal = _proposal(spec)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = campaign.evaluate(
        ticket,
        store=training,
        run_id=run_id,
        evidence=[evidence, *_correctness_evidence(run_id)],
        candidate_path="candidate.bin",
        now_ns=500,
        host=campaign.role_capability(
            authority, ticket, role="evaluator", principal_id="evaluator"
        ),
    )
    worker_store, receipt = _worker_evidence(tmp_path, ticket)
    campaign.confirm_stopped(
        ticket,
        stop_receipt=receipt,
        store=worker_store,
        host=campaign.role_capability(
            authority, ticket, role="supervisor", principal_id="supervisor"
        ),
    )
    with pytest.raises(ContractViolation, match="independent"):
        campaign.review(
            ticket,
            _review(evaluation, reviewer_id=actor),
            store=training,
            now_ns=700,
            host=campaign.role_capability(authority, ticket, role="reviewer", principal_id=actor),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": None},
        {"persist_event": False},
        {"target_id": "foreign"},
        {"worker_ids": ("foreign",)},
        {"trial_token": "foreign"},
        {"environment_config_sha256": "0" * 64},
        {"process_identity_sha256": "0" * 64},
    ],
)
def test_unknown_or_foreign_worker_stop_preserves_quarantined_resource_lease(
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    campaign, spec, _, ticket = _claimed(tmp_path)
    campaign.quarantine(ticket, reason="unknown worker state")
    store, receipt = _worker_evidence(tmp_path, ticket, **changes)
    with pytest.raises(ContractViolation, match=r"terminal|receipt"):
        campaign.confirm_stopped(
            ticket,
            stop_receipt=receipt,
            store=store,
            host=campaign.role_capability(
                _HOST, ticket, role="supervisor", principal_id="supervisor"
            ),
        )
    snapshot = campaign.snapshot(spec.campaign_id)
    assert snapshot["trials"][0]["status"] == "quarantined"
    assert snapshot["trials"][0]["stopped_receipt"] is None
    with sqlite3.connect(campaign.path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM leases WHERE trial=?", (ticket.trial_id,)
        ).fetchone()[0] == len(spec.resources)


@pytest.mark.parametrize("status,exit_code", [(RunStatus.FAILED, 1), (RunStatus.INTERRUPTED, 130)])
def test_verified_failed_worker_is_terminal_and_can_release_resources(
    tmp_path: Path,
    status: RunStatus,
    exit_code: int,
) -> None:
    campaign, spec, _, ticket = _claimed(tmp_path)
    campaign.quarantine(ticket, reason="worker failed")
    store, receipt = _worker_evidence(tmp_path, ticket, status=status, exit_code=exit_code)
    campaign.confirm_stopped(
        ticket,
        stop_receipt=receipt,
        store=store,
        host=campaign.role_capability(_HOST, ticket, role="supervisor", principal_id="supervisor"),
    )
    snapshot = campaign.snapshot(spec.campaign_id)
    with sqlite3.connect(campaign.path) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM leases WHERE trial=?", (ticket.trial_id,)
            ).fetchone()[0]
            == 0
        )
    assert snapshot["trials"][0]["status"] == "rejected"
    assert snapshot["state"]["checkpoint_sha256"] == spec.baseline_checkpoint_sha256


@pytest.mark.parametrize("change", ["metric-value", "metric-source", "metric-config", "run-target"])
def test_changed_persisted_evaluation_cannot_be_approved(tmp_path: Path, change: str) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    with sqlite3.connect(training.path) as connection:
        if change == "metric-value":
            connection.execute("UPDATE metrics SET value=999 WHERE run_id=?", (run_id,))
        elif change == "metric-source":
            connection.execute(
                "UPDATE metrics SET metadata_json=json_set(metadata_json,'$.source','foreign') "
                "WHERE run_id=?",
                (run_id,),
            )
        elif change == "metric-config":
            connection.execute(
                "UPDATE metrics SET environment_config_digest=NULL WHERE run_id=?", (run_id,)
            )
        else:
            connection.execute(
                "UPDATE runs SET metadata_json=json_set(metadata_json,'$.target_id','foreign') "
                "WHERE run_id=?",
                (run_id,),
            )
    with pytest.raises(ContractViolation, match="unchanged"):
        _submit_review(campaign, ticket, _review(evaluation))
    assert (
        campaign.snapshot(spec.campaign_id)["state"]["checkpoint_sha256"]
        == spec.baseline_checkpoint_sha256
    )


@pytest.mark.parametrize("target", [None, "", " "])
def test_campaign_cannot_claim_an_unknown_or_empty_target(target: Any) -> None:
    with pytest.raises(ValueError):
        _spec(target_id=target)


def test_authorization_receipt_contains_scope_and_fixed_measurement_without_capability(
    tmp_path: Path,
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path, kind="interface")
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    _submit_review(campaign, ticket, _review(evaluation))
    authorization = campaign.events(spec.campaign_id)[-1]["body"]["authorization"]
    assert authorization["candidate_kind"] == "interface"
    assert authorization["evaluation_scope"] == "fixed-external"
    assert authorization["final_measurement"]["metric_id"] > 0
    assert authorization["target_id"] == spec.target_id
    public_json = json.dumps(campaign.events(spec.campaign_id))
    cap = campaign.role_capability(_HOST, ticket, role="reviewer", principal_id="reviewer")
    assert _HOST.secret.decode() not in repr(_HOST) + public_json
    assert cap.proof.hex() not in repr(cap) + public_json
    assert "proof" not in public_json and "secret" not in public_json


@pytest.mark.parametrize("actor", ["planner", "worker", "evaluator"])
def test_worker_or_evaluator_cannot_supply_its_own_stop_receipt(tmp_path: Path, actor: str) -> None:
    authority = replace(_HOST, supervisor_ids=(actor,))
    campaign, ticket = _claimed_as(tmp_path, authority)
    spec = _spec()
    proposal = _proposal(spec)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    campaign.evaluate(
        ticket,
        store=training,
        run_id=run_id,
        evidence=[evidence, *_correctness_evidence(run_id)],
        candidate_path="candidate.bin",
        now_ns=500,
        host=campaign.role_capability(
            authority, ticket, role="evaluator", principal_id="evaluator"
        ),
    )
    worker_store, receipt = _worker_evidence(tmp_path, ticket, supervisor_id=actor)
    with pytest.raises(ContractViolation, match="independent"):
        campaign.confirm_stopped(
            ticket,
            stop_receipt=receipt,
            store=worker_store,
            host=campaign.role_capability(authority, ticket, role="supervisor", principal_id=actor),
        )


def test_campaign_stop_capability_cannot_authorize_trial_stop_confirmation(tmp_path: Path) -> None:
    campaign, spec, _, ticket = _claimed(tmp_path)
    cap = campaign.role_capability(
        _HOST, spec.campaign_id, role="supervisor", principal_id="supervisor"
    )
    campaign.stop(spec.campaign_id, reason="host requested stop", host=cap)
    worker_store, receipt = _worker_evidence(tmp_path, ticket)
    with pytest.raises(ContractViolation, match="scope"):
        campaign.confirm_stopped(ticket, stop_receipt=receipt, store=worker_store, host=cap)
    assert campaign.snapshot(spec.campaign_id)["trials"][0]["stopped_receipt"] is None


def test_authority_replaced_between_cap_check_and_mutation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, spec, _, _ = _claimed(tmp_path)
    cap = campaign.role_capability(
        _HOST, spec.campaign_id, role="supervisor", principal_id="supervisor"
    )
    checked = campaign._require_host

    def replace_after_check(*args: Any, **kwargs: Any) -> str:
        principal = checked(*args, **kwargs)
        with sqlite3.connect(campaign.path) as connection:
            connection.execute("UPDATE campaign_host SET fingerprint='foreign'")
        return principal

    monkeypatch.setattr(campaign, "_require_host", replace_after_check)
    with pytest.raises(ContractViolation, match="authority"):
        campaign.stop(spec.campaign_id, reason="race must not authorize stop", host=cap)
    with sqlite3.connect(campaign.path) as connection:
        state = json.loads(connection.execute("SELECT state FROM campaigns").fetchone()[0])
    assert state["status"] == "running"


def test_unknown_config_is_never_inferred_from_campaign_or_run(tmp_path: Path) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(
        tmp_path, spec, proposal, ticket, metric_changes={"environment_config_digest": None}
    )
    # The current store inherits the run digest for ordinary metrics. Construct
    # an explicitly unknown persisted legacy row for this negative evidence case.
    with sqlite3.connect(training.path) as database:
        database.execute(
            "UPDATE metrics SET environment_config_digest=NULL WHERE run_id=?", (run_id,)
        )
    with pytest.raises(ContractViolation, match="persisted"):
        _evaluate(campaign, ticket, training, run_id, evidence)


@pytest.mark.parametrize("metric", EVALUATION_ZERO_METRICS)
@pytest.mark.parametrize("nonzero", [1e-13, -1e-13])
def test_every_persisted_correctness_counter_must_be_exact_zero_even_with_a_zero_row(
    tmp_path: Path,
    metric: str,
    nonzero: float,
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(
        tmp_path, spec, proposal, ticket, correctness_values={metric: nonzero}, status=None
    )
    training.record_metric(
        run_id,
        name=metric,
        value=0,
        environment_config_digest=spec.environment_config_sha256,
        timestamp_ns=300,
        metadata={
            "source": EVALUATION_CHECK_SOURCE,
            "authority": "authoritative",
            "coverage": "measured",
        },
    )
    training.finish_run(run_id, status=RunStatus.SUCCEEDED, exit_code=0, finished_at_ns=400)
    with pytest.raises(ContractViolation, match="persisted"):
        _evaluate(campaign, ticket, training, run_id, evidence)
    snapshot = campaign.snapshot(spec.campaign_id)
    assert snapshot["trials"][0]["status"] == "claimed"
    assert snapshot["state"]["checkpoint_sha256"] == spec.baseline_checkpoint_sha256


def test_objective_score_keeps_numeric_tolerance_without_relaxing_correctness(
    tmp_path: Path,
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(
        tmp_path, spec, proposal, ticket, metric_changes={"value": 0.5 + 1e-13}
    )
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    assert (
        evaluation
        and campaign.snapshot(spec.campaign_id)["trials"][0]["status"] == "awaiting-review"
    )


def test_worker_stop_evidence_changed_after_confirmation_cannot_authorize_approval(
    tmp_path: Path,
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    with sqlite3.connect(tmp_path / "worker-runs.sqlite3") as connection:
        connection.execute("UPDATE runs SET status='running',finished_at_ns=NULL")
    with pytest.raises(ContractViolation, match="terminal"):
        _submit_review(campaign, ticket, _review(evaluation))
    assert (
        campaign.snapshot(spec.campaign_id)["state"]["checkpoint_sha256"]
        == spec.baseline_checkpoint_sha256
    )


@dataclass(frozen=True, slots=True)
class _ExtraEvidence(EvidenceRef):
    host_material: HostAuthority | HostRoleCapability


@dataclass(frozen=True, slots=True)
class _ExtraProposal(Proposal):
    host_material: HostAuthority | HostRoleCapability = _HOST


@dataclass(frozen=True, slots=True)
class _ExtraSpec(CampaignSpec):
    host_material: HostAuthority | HostRoleCapability = _HOST


@dataclass(frozen=True, slots=True)
class _ExtraStop(SupervisorStopReceipt):
    host_material: HostAuthority | HostRoleCapability


@pytest.mark.parametrize("material_kind", ["authority", "capability"])
@pytest.mark.parametrize("contract_kind", ["evidence", "proposal", "spec", "stop"])
def test_public_campaign_mappings_export_only_declared_schema_fields(
    tmp_path: Path,
    material_kind: str,
    contract_kind: str,
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    material = (
        _HOST
        if material_kind == "authority"
        else campaign.role_capability(_HOST, ticket, role="reviewer", principal_id="reviewer")
    )
    if contract_kind == "evidence":
        source = proposal.sources[0]
        extended_source = _ExtraEvidence(
            source.source_id, source.revision_sha256, source.media_type, source.locator, material
        )
        extended = replace(proposal, sources=(extended_source,))
        actual = extended.to_mapping()
        expected = proposal.to_mapping()
        assert extended_source.to_mapping() == source.to_mapping()
        assert extended.sha256 == proposal.sha256
    elif contract_kind == "proposal":
        extended = _ExtraProposal(
            **{
                **{name: getattr(proposal, name) for name in proposal.__dataclass_fields__},
                "host_material": material,
            }
        )
        actual = extended.to_mapping()
        expected = proposal.to_mapping()
        assert extended.sha256 == proposal.sha256
    elif contract_kind == "spec":
        extended = _ExtraSpec(
            **{
                **{name: getattr(spec, name) for name in spec.__dataclass_fields__},
                "host_material": material,
            }
        )
        actual = extended.to_mapping()
        expected = spec.to_mapping()
    else:
        _, stop = _worker_evidence(tmp_path, ticket)
        extended = _ExtraStop(
            **{
                **{name: getattr(stop, name) for name in stop.__dataclass_fields__},
                "host_material": material,
            }
        )
        actual = extended.to_mapping()
        expected = stop.to_mapping()
    assert actual == expected
    public_json = json.dumps(actual)
    assert "host_material" not in public_json and _HOST.secret.decode() not in public_json
