"""Offline campaign tests: no game, controller, learner, or process is started."""

from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from game_learning_runtime.agent_goal import AgentGoal, GoalEvidence, ResearchMediaType
from game_learning_runtime.continuous_learning import (
    EVALUATION_CHECK_SOURCE,
    EVALUATION_ZERO_METRICS,
    CampaignSpec,
    CampaignStore,
    EvidenceRef,
    HostAuthority,
    Proposal,
    ReviewDecision,
    SupervisorStopReceipt,
    TrialTicket,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.run_store import RunStatus, TrainingStore
from game_learning_runtime.training import KnowledgeAuthority

_CANDIDATE = b"synthetic immutable policy candidate v1"
_CLAIM_TIME = 100
_EVALUATION_TIME = 500
_HOST = HostAuthority(
    "host.synthetic",
    ("evaluator",),
    ("supervisor",),
    ("reviewer", "planner", "worker"),
    b"synthetic-host-key-for-offline-tests-only",
)


def _campaign_store(path: Path) -> CampaignStore:
    return CampaignStore(path, host_authority=_HOST)


def _hash(value: str | bytes) -> str:
    encoded = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _goal(**budget: int) -> AgentGoal:
    return AgentGoal.from_mapping(
        {
            "schema_version": "glr.agent-goal.v1",
            "goal_id": "goal.synthetic",
            "objective": "Verify progress with a fixed external evaluation.",
            "environment_family": "synthetic",
            "success_criteria": [
                {
                    "metric": "objective.progress",
                    "operator": "gte",
                    "target": 1,
                    "source": "runtime.telemetry",
                }
            ],
            "promotion": {"metric": "objective.progress", "mode": "max"},
            "budget": {
                "max_trials": 2,
                "max_training_steps": 10,
                "max_wall_seconds": 10,
                "max_research_sources": 2,
                **budget,
            },
            "allowed_research_media": ["runtime-trace", "official-rules"],
        }
    )


def _spec(**changes: Any) -> CampaignSpec:
    fields = {
        "campaign_id": "campaign.synthetic",
        "goal": _goal(),
        "environment_id": "synthetic.game-v1",
        "protocol_version": "1.0",
        "target_id": "synthetic.target",
        "environment_config_sha256": _hash("fixed initial state"),
        "evaluator_sha256": _hash("independent evaluator v1"),
        "evaluation_suite_sha256": _hash("fixed evaluation scenarios v1"),
        "resources": ("synthetic.controller", "synthetic.device"),
        "admitted_actions": ("observe", "reset", "wait"),
        # Admitted identities still cannot review their own work.
        "reviewers": ("reviewer", "planner", "worker"),
        "baseline_checkpoint_sha256": _hash("baseline checkpoint"),
        "baseline_score": 0.0,
    }
    fields.update(changes)
    return CampaignSpec(**fields)


def _proposal(spec: CampaignSpec, **changes: Any) -> Proposal:
    fields = {
        "proposal_id": "proposal.synthetic",
        "proposer_id": "planner",
        "kind": "policy",
        "summary": "A synthetic hypothesis grounded in an immutable source revision.",
        "artifact_sha256": _hash(_CANDIDATE),
        "base_checkpoint_sha256": spec.baseline_checkpoint_sha256,
        "sources": (
            EvidenceRef(
                "trace.synthetic",
                _hash("synthetic observation revision v1"),
                ResearchMediaType.RUNTIME_TRACE,
                "episode.synthetic/step-3",
            ),
        ),
        "requested_actions": ("observe", "wait"),
    }
    fields.update(changes)
    return Proposal(**fields)


def _claimed(
    tmp_path: Path, *, spec: CampaignSpec | None = None, **proposal_changes: Any
) -> tuple[CampaignStore, CampaignSpec, Proposal, TrialTicket]:
    spec = _spec() if spec is None else spec
    campaign = _campaign_store(tmp_path / "campaigns.sqlite3")
    campaign.create(spec, now_ns=0)
    proposal = _proposal(spec, **proposal_changes)
    campaign.propose(spec.campaign_id, proposal)
    ticket = campaign.claim(
        spec.campaign_id,
        proposal.proposal_id,
        worker_id="worker",
        training_steps=5,
        wall_seconds=2,
        now_ns=_CLAIM_TIME,
    )
    root = campaign.candidate_directory(ticket)
    root.mkdir(parents=True)
    (root / "candidate.bin").write_bytes(_CANDIDATE)
    return campaign, spec, proposal, ticket


def _evaluation_run(
    tmp_path: Path,
    spec: CampaignSpec,
    proposal: Proposal,
    ticket: TrialTicket,
    *,
    value: float = 0.5,
    run_changes: dict[str, Any] | None = None,
    metadata_changes: dict[str, Any] | None = None,
    metric_changes: dict[str, Any] | None = None,
    persist_metric: bool = True,
    status: RunStatus | None = RunStatus.SUCCEEDED,
    exit_code: int = 0,
    finished_at_ns: int = 400,
    correctness_values: dict[str, float] | None = None,
    omitted_correctness_metric: str | None = None,
) -> tuple[TrainingStore, str, GoalEvidence]:
    training = TrainingStore(tmp_path / "runs.sqlite3")
    metadata = {
        "campaign_trial_id": ticket.trial_id,
        "campaign_token": ticket.token,
        "candidate_sha256": proposal.artifact_sha256,
        "evaluator_sha256": spec.evaluator_sha256,
        "evaluation_suite_sha256": spec.evaluation_suite_sha256,
        "target_id": spec.target_id,
        "evaluator_id": "evaluator",
        "proposal_sha256": proposal.sha256,
        "evaluation_scope": "fixed-external",
        **(metadata_changes or {}),
    }
    arguments = {
        "environment_id": spec.environment_id,
        "protocol_version": spec.protocol_version,
        "kind": "evaluation",
        "environment_config_digest": spec.environment_config_sha256,
        "metadata": metadata,
        "started_at_ns": 200,
        **(run_changes or {}),
    }
    run = training.create_run(**arguments)
    criterion = spec.goal.success_criteria[0]
    if persist_metric:
        training.record_metric(
            run.run_id,
            **{
                "name": criterion.metric,
                "value": value,
                "environment_config_digest": spec.environment_config_sha256,
                "metadata": {
                    "source": criterion.source,
                    "authority": "authoritative",
                    "coverage": "measured",
                },
                "timestamp_ns": 300,
                **(metric_changes or {}),
            },
        )
    for metric in EVALUATION_ZERO_METRICS:
        if metric != omitted_correctness_metric:
            training.record_metric(
                run.run_id,
                name=metric,
                value=(correctness_values or {}).get(metric, 0.0),
                metadata={
                    "source": EVALUATION_CHECK_SOURCE,
                    "authority": "authoritative",
                    "coverage": "measured",
                },
                timestamp_ns=300,
                environment_config_digest=spec.environment_config_sha256,
            )
    if status is not None:
        training.finish_run(
            run.run_id, status=status, exit_code=exit_code, finished_at_ns=finished_at_ns
        )
    evidence = GoalEvidence(
        criterion.metric, value, criterion.source, KnowledgeAuthority.AUTHORITATIVE, run.run_id
    )
    return training, run.run_id, evidence


def _evaluate(
    campaign: CampaignStore,
    ticket: TrialTicket,
    training: TrainingStore,
    run_id: str,
    objective_evidence: GoalEvidence,
    **changes: Any,
) -> str:
    return campaign.evaluate(
        ticket,
        **{
            "store": training,
            "run_id": run_id,
            "evidence": [objective_evidence, *_correctness_evidence(run_id)],
            "candidate_path": "candidate.bin",
            "now_ns": _EVALUATION_TIME,
            "host": campaign.role_capability(
                _HOST, ticket, role="evaluator", principal_id="evaluator"
            ),
            **changes,
        },
    )


def _correctness_evidence(
    run_id: str, values: dict[str, float] | None = None
) -> list[GoalEvidence]:
    return [
        GoalEvidence(
            metric,
            (values or {}).get(metric, 0.0),
            EVALUATION_CHECK_SOURCE,
            KnowledgeAuthority.AUTHORITATIVE,
            run_id,
        )
        for metric in EVALUATION_ZERO_METRICS
    ]


def _review(evaluation: str, **changes: Any) -> ReviewDecision:
    return ReviewDecision(
        **{
            "reviewer_id": "reviewer",
            "evaluation_sha256": evaluation,
            "approved": True,
            "receipt_sha256": _hash("external review receipt"),
            **changes,
        }
    )


def _stopped(campaign: CampaignStore, ticket: TrialTicket) -> None:
    fields = campaign.snapshot(ticket.campaign_id)["spec"]
    worker_store = TrainingStore(campaign.path.parent / "worker-runs.sqlite3")
    identity = _hash("synthetic process identity " + ticket.trial_id)
    stopped = max(600, ticket.started_at_ns + 1)
    run = worker_store.create_run(
        environment_id=fields["environment_id"],
        protocol_version=fields["protocol_version"],
        kind="campaign-worker",
        environment_config_digest=fields["environment_config_sha256"],
        started_at_ns=ticket.started_at_ns,
        metadata={
            "campaign_trial_id": ticket.trial_id,
            "campaign_token": ticket.token,
            "target_id": fields["target_id"],
            "worker_ids": [ticket.worker_id],
            "process_identity_sha256": identity,
        },
    )
    receipt = SupervisorStopReceipt(
        "stop-" + ticket.trial_id,
        "supervisor",
        ticket.trial_id,
        ticket.token,
        run.run_id,
        (ticket.worker_id,),
        fields["environment_id"],
        fields["protocol_version"],
        fields["target_id"],
        fields["environment_config_sha256"],
        identity,
        _hash("synthetic trusted supervisor observation"),
        stopped,
    )
    worker_store.append_event(
        run.run_id,
        kind="supervisor.worker-stopped",
        payload=receipt.to_mapping(),
        timestamp_ns=stopped,
    )
    worker_store.finish_run(
        run.run_id, status=RunStatus.SUCCEEDED, exit_code=0, finished_at_ns=stopped
    )
    campaign.confirm_stopped(
        ticket,
        stop_receipt=receipt,
        store=worker_store,
        host=campaign.role_capability(_HOST, ticket, role="supervisor", principal_id="supervisor"),
    )


def _submit_review(campaign: CampaignStore, ticket: TrialTicket, decision: ReviewDecision) -> None:
    principal = decision.reviewer_id if decision.reviewer_id in _HOST.reviewer_ids else "reviewer"
    campaign.review(
        ticket,
        decision,
        store=TrainingStore(campaign.path.parent / "runs.sqlite3"),
        host=campaign.role_capability(_HOST, ticket, role="reviewer", principal_id=principal),
        now_ns=700,
    )


def test_unprivileged_reopen_cannot_confirm_stop_with_only_a_digest(tmp_path: Path) -> None:
    campaign, _, _, ticket = _claimed(tmp_path)
    worker_store = CampaignStore(campaign.path)
    with pytest.raises(ContractViolation, match="host"):
        worker_store.confirm_stopped(ticket, supervisor_receipt_sha256=_hash("made-up stop"))
    assert worker_store.snapshot(ticket.campaign_id)["trials"][0]["stopped_receipt"] is None


def test_reviewed_candidate_survives_reopen_and_audit_retains_baseline(tmp_path: Path) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    assert campaign.snapshot(spec.campaign_id)["state"]["checkpoint_sha256"] == (
        spec.baseline_checkpoint_sha256
    )
    _stopped(campaign, ticket)
    _submit_review(campaign, ticket, _review(evaluation))

    reopened = _campaign_store(campaign.path)
    snapshot = reopened.snapshot(spec.campaign_id)
    assert snapshot["state"]["checkpoint_sha256"] == proposal.artifact_sha256
    assert snapshot["state"]["score"] == 0.5
    assert snapshot["state"]["status"] == "running"
    assert snapshot["state"]["reserved_steps"] == 5
    assert snapshot["trials"][0]["status"] == "promoted"
    assert snapshot["trials"][0]["evaluation_hash"] == evaluation
    events = reopened.events(spec.campaign_id)
    assert [event["kind"] for event in events] == [
        "campaign.created",
        "proposal.recorded",
        "trial.claimed",
        "trial.evaluated",
        "trial.stopped",
        "trial.reviewed",
    ]
    assert len({event["sequence"] for event in events}) == len(events)
    assert events[1]["body"]["base_checkpoint_sha256"] == spec.baseline_checkpoint_sha256
    assert events[1]["body"]["sources"][0]["revision_sha256"] == (
        proposal.sources[0].revision_sha256
    )
    with pytest.raises(ContractViolation):
        _submit_review(reopened, ticket, _review(evaluation))


def test_campaign_admission_is_frozen_and_exported_data_cannot_change_it(tmp_path: Path) -> None:
    spec = _spec()
    campaign = _campaign_store(tmp_path / "campaigns.sqlite3")
    campaign.create(spec, now_ns=0)
    with pytest.raises(FrozenInstanceError):
        spec.admitted_actions = ("execute-code",)
    exported = campaign.snapshot(spec.campaign_id)
    exported["spec"]["admitted_actions"] = ("execute-code",)
    exported["spec"]["goal"]["budget"]["max_trials"] = 1000
    with pytest.raises(ContractViolation, match="authority"):
        campaign.propose(spec.campaign_id, _proposal(spec, requested_actions=("execute-code",)))
    assert campaign.snapshot(spec.campaign_id)["spec"]["goal"]["budget"]["max_trials"] == 2
    with pytest.raises(sqlite3.IntegrityError):
        campaign.create(replace(spec, evaluator_sha256=_hash("changed evaluator")), now_ns=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact_sha256": "not-a-hash"},
        {"sources": ()},
        {"sources": [EvidenceRef("trace.one", _hash("one"), "runtime-trace", "step-1")]},
        {"requested_actions": ("wait", "wait")},
        {"summary": ""},
        {"kind": "execute"},
    ],
)
def test_proposals_require_content_addressed_bounded_provenance(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _proposal(_spec(), **changes)


def test_source_revisions_are_preserved_without_becoming_execution_authority(
    tmp_path: Path,
) -> None:
    campaign = _campaign_store(tmp_path / "campaigns.sqlite3")
    spec = _spec()
    campaign.create(spec, now_ns=0)
    first = _proposal(spec)
    second_source = replace(first.sources[0], revision_sha256=_hash("revised trace"))
    second = replace(first, proposal_id="proposal.revised", sources=(second_source,))
    campaign.propose(spec.campaign_id, first)
    campaign.propose(spec.campaign_id, second)
    revisions = [
        event["body"]["sources"][0]["revision_sha256"]
        for event in campaign.events(spec.campaign_id)
        if event["kind"] == "proposal.recorded"
    ]
    assert revisions == [first.sources[0].revision_sha256, second_source.revision_sha256]
    with pytest.raises(sqlite3.IntegrityError):
        campaign.propose(spec.campaign_id, replace(first, sources=(second_source,)))
    with pytest.raises(ContractViolation, match="source"):
        campaign.propose(
            spec.campaign_id,
            replace(
                first,
                proposal_id="proposal.disallowed",
                sources=(replace(first.sources[0], media_type=ResearchMediaType.TEXT_GUIDE),),
            ),
        )


def test_shared_resource_claim_is_atomic_between_independent_store_connections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaigns.sqlite3"
    stores = [_campaign_store(path), _campaign_store(path)]
    specs = [
        _spec(campaign_id="campaign.platformer", environment_id="synthetic.platformer-v1"),
        _spec(
            campaign_id="campaign.strategy",
            environment_id="synthetic.strategy-v1",
            admitted_actions=("choose_option", "refresh"),
        ),
    ]
    proposals = [_proposal(spec, requested_actions=spec.admitted_actions) for spec in specs]
    for campaign, spec, proposal in zip(stores, specs, proposals, strict=True):
        campaign.create(spec, now_ns=0)
        campaign.propose(spec.campaign_id, proposal)
    barrier = Barrier(2)

    def claim(index: int) -> TrialTicket | None:
        barrier.wait(timeout=5)
        try:
            return stores[index].claim(
                specs[index].campaign_id,
                proposals[index].proposal_id,
                worker_id=f"worker-{index}",
                training_steps=5,
                wall_seconds=2,
                now_ns=_CLAIM_TIME,
            )
        except ContractViolation:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, range(2)))
    assert sum(ticket is not None for ticket in results) == 1
    snapshots = [
        store.snapshot(spec.campaign_id) for store, spec in zip(stores, specs, strict=True)
    ]
    assert sorted(item["state"]["reserved_steps"] for item in snapshots) == [0, 5]
    assert sorted(item["state"]["trials"] for item in snapshots) == [0, 1]


def test_resource_conflict_rolls_back_every_lock_and_budget_reservation(tmp_path: Path) -> None:
    campaign, _, _, holder = _claimed(tmp_path)
    conflicting = _spec(
        campaign_id="campaign.conflict", resources=("synthetic.free", "synthetic.device")
    )
    independent = _spec(campaign_id="campaign.independent", resources=("synthetic.free",))
    for spec in (conflicting, independent):
        campaign.create(spec, now_ns=0)
        campaign.propose(spec.campaign_id, _proposal(spec))
    with pytest.raises(ContractViolation, match="resource"):
        campaign.claim(
            conflicting.campaign_id,
            "proposal.synthetic",
            worker_id="other-worker",
            training_steps=5,
            wall_seconds=2,
            now_ns=_CLAIM_TIME,
        )
    snapshot = campaign.snapshot(conflicting.campaign_id)
    assert snapshot["state"]["reserved_steps"] == 0
    assert snapshot["state"]["reserved_sources"] == 0
    assert snapshot["trials"] == []
    free = campaign.claim(
        independent.campaign_id,
        "proposal.synthetic",
        worker_id="independent-worker",
        training_steps=5,
        wall_seconds=2,
        now_ns=_CLAIM_TIME,
    )
    assert free.trial_id != holder.trial_id


def test_expired_claim_cannot_be_stolen_and_requires_confirmed_stop(tmp_path: Path) -> None:
    campaign, spec, _, ticket = _claimed(tmp_path)
    other = _spec(campaign_id="campaign.other")
    campaign.create(other, now_ns=0)
    campaign.propose(other.campaign_id, _proposal(other))
    reopened = _campaign_store(campaign.path)
    after_deadline = ticket.deadline_ns + 1
    with pytest.raises(ContractViolation, match="resource"):
        reopened.claim(
            other.campaign_id,
            "proposal.synthetic",
            worker_id="other-worker",
            training_steps=5,
            wall_seconds=2,
            now_ns=after_deadline,
        )
    reopened.quarantine(ticket, reason="worker deadline expired; exact stop is unconfirmed")
    with pytest.raises(ContractViolation, match="resource"):
        reopened.claim(
            other.campaign_id,
            "proposal.synthetic",
            worker_id="other-worker",
            training_steps=5,
            wall_seconds=2,
            now_ns=after_deadline,
        )
    with pytest.raises(ContractViolation, match="ticket"):
        _stopped(reopened, replace(ticket, token="stale-token"))
    _stopped(reopened, ticket)
    new = reopened.claim(
        other.campaign_id,
        "proposal.synthetic",
        worker_id="other-worker",
        training_steps=5,
        wall_seconds=2,
        now_ns=after_deadline,
    )
    assert new.token != ticket.token
    with pytest.raises(ContractViolation):
        _stopped(reopened, ticket)
    assert reopened.snapshot(spec.campaign_id)["state"]["reserved_steps"] == 5


@pytest.mark.parametrize(
    "budget",
    [{"max_trials": 1}, {"max_training_steps": 5}, {"max_research_sources": 1}],
)
def test_failure_and_reopen_do_not_refund_spent_campaign_budget(
    tmp_path: Path, budget: dict[str, int]
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path, spec=_spec(goal=_goal(**budget)))
    campaign.quarantine(ticket, reason="synthetic evaluation failure")
    _stopped(campaign, ticket)
    reopened = _campaign_store(campaign.path)
    second = replace(proposal, proposal_id="proposal.second")
    reopened.propose(spec.campaign_id, second)
    with pytest.raises(ContractViolation, match="budget"):
        reopened.claim(
            spec.campaign_id,
            second.proposal_id,
            worker_id="worker",
            training_steps=1,
            wall_seconds=1,
            now_ns=_CLAIM_TIME + 1,
        )
    state = reopened.snapshot(spec.campaign_id)["state"]
    assert (state["trials"], state["reserved_steps"], state["reserved_sources"]) == (1, 5, 1)
    assert state["checkpoint_sha256"] == spec.baseline_checkpoint_sha256


@pytest.mark.parametrize("claim_time", [-1, 10 * 10**9, 9 * 10**9])
def test_campaign_wall_budget_is_not_refreshed_on_reopen(tmp_path: Path, claim_time: int) -> None:
    spec = _spec()
    campaign = _campaign_store(tmp_path / "campaigns.sqlite3")
    campaign.create(spec, now_ns=0)
    campaign.propose(spec.campaign_id, _proposal(spec))
    reopened = _campaign_store(campaign.path)
    with pytest.raises((ContractViolation, ValueError)):
        reopened.claim(
            spec.campaign_id,
            "proposal.synthetic",
            worker_id="worker",
            training_steps=1,
            wall_seconds=2,
            now_ns=claim_time,
        )
    assert reopened.snapshot(spec.campaign_id)["state"]["trials"] == 0


def test_clock_rollback_before_campaign_creation_cannot_admit_work(tmp_path: Path) -> None:
    spec = _spec()
    campaign = _campaign_store(tmp_path / "campaigns.sqlite3")
    campaign.create(spec, now_ns=1000)
    campaign.propose(spec.campaign_id, _proposal(spec))
    with pytest.raises(ContractViolation):
        campaign.claim(
            spec.campaign_id,
            "proposal.synthetic",
            worker_id="worker",
            training_steps=1,
            wall_seconds=1,
            now_ns=999,
        )


@pytest.mark.parametrize(
    "field",
    [
        "campaign_trial_id",
        "campaign_token",
        "candidate_sha256",
        "evaluator_sha256",
        "evaluation_suite_sha256",
    ],
)
def test_evaluation_cannot_borrow_a_run_from_another_candidate_or_contract(
    tmp_path: Path, field: str
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(
        tmp_path, spec, proposal, ticket, metadata_changes={field: _hash("wrong binding")}
    )
    with pytest.raises(ContractViolation, match="bound"):
        _evaluate(campaign, ticket, training, run_id, evidence)
    snapshot = campaign.snapshot(spec.campaign_id)
    assert snapshot["trials"][0]["status"] == "claimed"
    assert snapshot["state"]["checkpoint_sha256"] == spec.baseline_checkpoint_sha256


@pytest.mark.parametrize(
    "changes",
    [
        {"run_changes": {"kind": "training"}},
        {"status": None},
        {"status": RunStatus.FAILED, "exit_code": 1},
        {"exit_code": 1},
        {"run_changes": {"environment_id": "other.game-v1"}},
        {"run_changes": {"protocol_version": "different"}},
        {"run_changes": {"environment_config_digest": _hash("other config")}},
        {"run_changes": {"started_at_ns": _CLAIM_TIME - 1}},
        {"finished_at_ns": _EVALUATION_TIME + 1},
    ],
)
def test_only_a_completed_evaluation_with_matching_environment_and_time_is_accepted(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket, **changes)
    with pytest.raises(ContractViolation, match="bound"):
        _evaluate(campaign, ticket, training, run_id, evidence)


@pytest.mark.parametrize(
    "changes",
    [
        {"persist_metric": False},
        {"metric_changes": {"value": 0.1}},
        {"metric_changes": {"metadata": {"source": "trainer", "authority": "authoritative"}}},
        {"metric_changes": {"metadata": {"source": "runtime.telemetry", "authority": "advisory"}}},
        {"metric_changes": {"environment_config_digest": _hash("other metric config")}},
        {"metric_changes": {"timestamp_ns": 199}},
        {"metric_changes": {"timestamp_ns": 401}},
    ],
)
def test_evaluation_requires_authoritative_persisted_metrics_from_this_run(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket, **changes)
    with pytest.raises(ContractViolation):
        _evaluate(campaign, ticket, training, run_id, evidence)
    assert campaign.snapshot(spec.campaign_id)["state"]["checkpoint_sha256"] == (
        spec.baseline_checkpoint_sha256
    )


@pytest.mark.parametrize(
    "changes",
    [{"run_id": "run-other"}, {"authority": KnowledgeAuthority.ADVISORY}],
)
def test_caller_cannot_relabel_evidence_to_replace_ledger_authority(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    with pytest.raises(ContractViolation):
        _evaluate(campaign, ticket, training, run_id, replace(evidence, **changes))


def test_conflicting_authoritative_scores_cannot_be_cherry_picked(tmp_path: Path) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket, status=None)
    training.record_metric(
        run_id,
        name=evidence.metric,
        value=-1,
        metadata={"source": evidence.source, "authority": "authoritative"},
        timestamp_ns=350,
    )
    training.finish_run(run_id, status=RunStatus.SUCCEEDED, exit_code=0, finished_at_ns=400)
    with pytest.raises(ContractViolation):
        _evaluate(campaign, ticket, training, run_id, evidence)


@pytest.mark.parametrize("metric", EVALUATION_ZERO_METRICS)
def test_high_score_cannot_override_a_failed_frozen_evaluation_gate(
    tmp_path: Path, metric: str
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    values = {metric: 1.0}
    training, run_id, evidence = _evaluation_run(
        tmp_path, spec, proposal, ticket, value=1.0, correctness_values=values
    )
    with pytest.raises(ContractViolation, match="correctness gate"):
        _evaluate(
            campaign,
            ticket,
            training,
            run_id,
            evidence,
            evidence=[evidence, *_correctness_evidence(run_id, values)],
        )
    assert campaign.snapshot(spec.campaign_id)["state"]["checkpoint_sha256"] == (
        spec.baseline_checkpoint_sha256
    )


@pytest.mark.parametrize("metric", EVALUATION_ZERO_METRICS)
def test_unpersisted_correctness_claim_cannot_complete_evaluation(
    tmp_path: Path, metric: str
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(
        tmp_path, spec, proposal, ticket, omitted_correctness_metric=metric
    )
    with pytest.raises(ContractViolation):
        _evaluate(campaign, ticket, training, run_id, evidence)


def test_evaluation_requires_all_fixed_evidence_and_rejects_duplicates(tmp_path: Path) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    with pytest.raises(ContractViolation, match="missing"):
        _evaluate(campaign, ticket, training, run_id, evidence, evidence=[evidence])
    with pytest.raises(ContractViolation):
        _evaluate(
            campaign,
            ticket,
            training,
            run_id,
            evidence,
            evidence=[evidence, evidence, *_correctness_evidence(run_id)],
        )


def test_missing_option_refresh_is_a_proposal_and_cannot_expand_adapter_authority(
    tmp_path: Path,
) -> None:
    # An independent synthetic capability claim exercises the admission boundary,
    # All records below are constructed data for this admission test.
    spec = _spec(
        environment_id="synthetic.strategy-v1", admitted_actions=("choose_option", "refresh")
    )
    campaign = _campaign_store(tmp_path / "campaigns.sqlite3")
    campaign.create(spec, now_ns=0)
    proposal = _proposal(
        spec,
        kind="interface",
        summary="A missing option observation requires adapter contract validation.",
        sources=(
            EvidenceRef(
                "synthetic.option-capability-gap",
                _hash("synthetic missing option observation"),
                ResearchMediaType.RUNTIME_TRACE,
                "synthetic-option-panel/step-1",
            ),
        ),
        requested_actions=("option_refresh",),
    )
    with pytest.raises(ContractViolation, match="authority"):
        campaign.propose(spec.campaign_id, proposal)
    campaign.propose(
        spec.campaign_id,
        replace(
            proposal, proposal_id="proposal.bounded-option", requested_actions=("choose_option",)
        ),
    )
    assert campaign.snapshot(spec.campaign_id)["spec"]["admitted_actions"] == (
        "choose_option",
        "refresh",
    )


@pytest.mark.parametrize(
    "source_id,metric,summary",
    [
        (
            "synthetic.reset-attribution",
            "evaluation.reward_attribution_errors",
            "A rejected state-restoration action cannot earn credit across an episode reset.",
        ),
        (
            "synthetic.effect-attribution",
            "evaluation.reward_attribution_errors",
            "An accepted environmental effect must reach its declared learner contribution.",
        ),
        (
            "synthetic.action-interval",
            "evaluation.action_interval_errors",
            "Multi-hop progress must use the whole action's start and end.",
        ),
        (
            "synthetic.stale-frame",
            "evaluation.stale_observation_updates",
            "Repeated or regressing observations cannot enter a learner update.",
        ),
        (
            "synthetic.inactive-frame",
            "evaluation.dead_or_loading_updates",
            "Dead and loading frames cannot start a new life or learner update.",
        ),
        (
            "synthetic.action-mask",
            "evaluation.illegal_action_bootstraps",
            "An unavailable high-value action cannot be used as a value bootstrap.",
        ),
    ],
)
def test_correctness_failure_cannot_be_hidden_by_a_high_goal_score(
    tmp_path: Path, source_id: str, metric: str, summary: str
) -> None:
    # Independent synthetic correctness fixtures exercise failures without running
    # target code or asserting that any live adapter emits these measurements.
    campaign, spec, proposal, ticket = _claimed(
        tmp_path,
        spec=_spec(environment_id="synthetic.platformer-v1"),
        summary=summary,
        sources=(
            EvidenceRef(source_id, _hash(summary), "runtime-trace", "synthetic-replay/step-1"),
        ),
    )
    values = {metric: 1.0}
    training, run_id, evidence = _evaluation_run(
        tmp_path, spec, proposal, ticket, value=1.0, correctness_values=values
    )
    with pytest.raises(ContractViolation, match="correctness gate"):
        _evaluate(
            campaign,
            ticket,
            training,
            run_id,
            evidence,
            evidence=[evidence, *_correctness_evidence(run_id, values)],
        )
    campaign.quarantine(ticket, reason="fixed evaluation reported a correctness regression")
    _stopped(campaign, ticket)
    snapshot = _campaign_store(campaign.path).snapshot(spec.campaign_id)
    assert snapshot["state"]["checkpoint_sha256"] == spec.baseline_checkpoint_sha256
    assert snapshot["trials"][0]["status"] == "rejected"


@pytest.mark.parametrize(
    "path",
    ["../candidate.bin", "/candidate.bin", "C:/candidate.bin", "a\\b", "a//b", "./candidate.bin"],
)
def test_candidate_paths_cannot_escape_the_isolated_trial(tmp_path: Path, path: str) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    with pytest.raises(ContractViolation):
        _evaluate(campaign, ticket, training, run_id, evidence, candidate_path=path)


def test_candidate_bytes_are_checked_before_evaluation_and_again_before_promotion(
    tmp_path: Path,
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    candidate = campaign.candidate_directory(ticket) / "candidate.bin"
    candidate.write_bytes(b"changed before evaluation")
    with pytest.raises(ContractViolation, match="bytes"):
        _evaluate(campaign, ticket, training, run_id, evidence)
    candidate.write_bytes(_CANDIDATE)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    candidate.write_bytes(b"changed after evaluation")
    with pytest.raises(ContractViolation, match="changed"):
        _submit_review(campaign, ticket, _review(evaluation))
    assert campaign.snapshot(spec.campaign_id)["state"]["checkpoint_sha256"] == (
        spec.baseline_checkpoint_sha256
    )


@pytest.mark.parametrize("reviewer", ["planner", "worker", "unadmitted-reviewer"])
def test_review_must_come_from_an_independent_admitted_identity(
    tmp_path: Path, reviewer: str
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    with pytest.raises(ContractViolation, match="independent"):
        _submit_review(campaign, ticket, _review(evaluation, reviewer_id=reviewer))
    assert campaign.snapshot(spec.campaign_id)["state"]["checkpoint_sha256"] == (
        spec.baseline_checkpoint_sha256
    )


def test_review_requires_bound_evaluation_and_supervisor_stop_receipt(tmp_path: Path) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    with pytest.raises(ContractViolation, match="stopped"):
        _submit_review(campaign, ticket, _review(evaluation))
    _stopped(campaign, ticket)
    with pytest.raises(ContractViolation, match="evaluation"):
        _submit_review(campaign, ticket, _review(_hash("another evaluation")))
    _submit_review(campaign, ticket, _review(evaluation, approved=False))
    snapshot = _campaign_store(campaign.path).snapshot(spec.campaign_id)
    assert snapshot["trials"][0]["status"] == "rejected"
    assert snapshot["state"]["checkpoint_sha256"] == spec.baseline_checkpoint_sha256
    assert snapshot["state"]["score"] == spec.baseline_score


def test_failed_promotion_commit_rolls_back_checkpoint_and_can_be_reviewed_after_reopen(
    tmp_path: Path,
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    before = campaign.snapshot(spec.campaign_id)
    # Inject a real database write failure after state/trial updates, proving the
    # public review transaction rolls back rather than mocking its domain logic.
    with sqlite3.connect(campaign.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_review_audit BEFORE INSERT ON audit "
            "WHEN NEW.kind = 'trial.reviewed' BEGIN "
            "SELECT RAISE(ABORT, 'synthetic storage failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic storage failure"):
        _submit_review(campaign, ticket, _review(evaluation))
    reopened = _campaign_store(campaign.path)
    assert reopened.snapshot(spec.campaign_id) == before
    assert all(event["kind"] != "trial.reviewed" for event in reopened.events(spec.campaign_id))
    with sqlite3.connect(campaign.path) as connection:
        connection.execute("DROP TRIGGER fail_review_audit")
    _submit_review(reopened, ticket, _review(evaluation))
    assert reopened.snapshot(spec.campaign_id)["state"]["checkpoint_sha256"] == (
        proposal.artifact_sha256
    )
    assert (
        sum(event["kind"] == "trial.reviewed" for event in reopened.events(spec.campaign_id)) == 1
    )


@pytest.mark.parametrize("value", [0.0, -1.0, 0.25])
def test_failed_or_insufficient_improvement_preserves_incumbent(
    tmp_path: Path, value: float
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path, spec=_spec(minimum_improvement=0.25))
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket, value=value)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    with pytest.raises(ContractViolation):
        _submit_review(campaign, ticket, _review(evaluation))
    snapshot = _campaign_store(campaign.path).snapshot(spec.campaign_id)
    assert snapshot["trials"][0]["status"] == "rejected"
    assert snapshot["state"]["checkpoint_sha256"] == spec.baseline_checkpoint_sha256
    assert snapshot["state"]["score"] == 0.0


@pytest.mark.parametrize("kind", ["knowledge", "interface"])
def test_approved_knowledge_or_interface_revision_does_not_replace_policy_checkpoint(
    tmp_path: Path, kind: str
) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path, kind=kind)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    _submit_review(campaign, ticket, _review(evaluation))
    state = campaign.snapshot(spec.campaign_id)["state"]
    assert state["accepted_proposals"] == {kind: proposal.proposal_id}
    assert state["checkpoint_sha256"] == spec.baseline_checkpoint_sha256
    assert state["score"] == spec.baseline_score


def test_promoted_checkpoint_fences_old_proposals(tmp_path: Path) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    old = replace(proposal, proposal_id="proposal.old-base")
    campaign.propose(spec.campaign_id, old)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    _submit_review(campaign, ticket, _review(evaluation))
    with pytest.raises(ContractViolation, match="stale"):
        campaign.claim(
            spec.campaign_id,
            old.proposal_id,
            worker_id="worker",
            training_steps=1,
            wall_seconds=1,
            now_ns=600,
        )


def test_successful_campaign_cannot_dispatch_again_after_reopen(tmp_path: Path) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket, value=1.0)
    evaluation = _evaluate(campaign, ticket, training, run_id, evidence)
    _stopped(campaign, ticket)
    _submit_review(campaign, ticket, _review(evaluation))
    reopened = _campaign_store(campaign.path)
    assert reopened.snapshot(spec.campaign_id)["state"]["status"] == "succeeded"
    with pytest.raises(ContractViolation):
        reopened.propose(spec.campaign_id, replace(proposal, proposal_id="proposal.after-success"))
    with pytest.raises(ContractViolation):
        reopened.claim(
            spec.campaign_id,
            proposal.proposal_id,
            worker_id="worker",
            training_steps=1,
            wall_seconds=1,
            now_ns=600,
        )


def test_stopping_campaign_preserves_uncertain_worker_lock_and_baseline(tmp_path: Path) -> None:
    campaign, spec, _, ticket = _claimed(tmp_path)
    campaign.stop(
        spec.campaign_id,
        reason="owner requested stop",
        host=campaign.role_capability(
            _HOST, spec.campaign_id, role="supervisor", principal_id="supervisor"
        ),
    )
    other = _spec(campaign_id="campaign.after-stop")
    campaign.create(other, now_ns=0)
    campaign.propose(other.campaign_id, _proposal(other))
    reopened = _campaign_store(campaign.path)
    with pytest.raises(ContractViolation, match="resource"):
        reopened.claim(
            other.campaign_id,
            "proposal.synthetic",
            worker_id="other-worker",
            training_steps=1,
            wall_seconds=1,
            now_ns=600,
        )
    snapshot = reopened.snapshot(spec.campaign_id)
    assert snapshot["state"]["status"] == "stopped"
    assert snapshot["state"]["checkpoint_sha256"] == spec.baseline_checkpoint_sha256
    _stopped(reopened, ticket)


def test_campaign_store_refuses_to_reuse_an_existing_training_database(tmp_path: Path) -> None:
    training = TrainingStore(tmp_path / "runs.sqlite3")
    with pytest.raises(ContractViolation, match="dedicated"):
        _campaign_store(training.path)


def test_completed_evaluation_releases_the_evidence_database_file(tmp_path: Path) -> None:
    campaign, spec, proposal, ticket = _claimed(tmp_path)
    training, run_id, evidence = _evaluation_run(tmp_path, spec, proposal, ticket)
    _evaluate(campaign, ticket, training, run_id, evidence)
    # A completed ledger may be archived immediately. On Windows a retained
    # SQLite read handle prevents this, even after its query has finished.
    destination = training.path.with_name("archived-evidence.sqlite3")
    training.path.rename(destination)
    assert destination.is_file()
