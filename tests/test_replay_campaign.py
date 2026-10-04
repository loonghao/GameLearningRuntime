"""Real replay-to-campaign integration using only inert, synthetic evidence."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from test_continuous_learning import _HOST, _stopped, _submit_review
from test_replay_evaluation import _changed, _decision, _frames, _spec, _suite

from game_learning_runtime.agent_goal import AgentGoal, ResearchMediaType
from game_learning_runtime.continuous_learning import (
    CampaignSpec,
    CampaignStore,
    EvidenceRef,
    Proposal,
    ReviewDecision,
    SupervisorStopReceipt,
    TrialTicket,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.replay_evaluation import (
    REPLAY_CHECK_METRICS,
    FixedReplaySuite,
    ReplayPolicy,
    evaluator_sha256,
)
from game_learning_runtime.run_store import RunStatus, TrainingStore

_INERT_REFERENCE = b'{"kind":"knowledge","revision":"offline-synthetic-v1"}'
_REFERENCE_SHA256 = hashlib.sha256(_INERT_REFERENCE).hexdigest()
_BASELINE_SHA256 = hashlib.sha256(b"untouched policy checkpoint").hexdigest()


def _callback(suite: FixedReplaySuite, fault: str | None = None) -> ReplayPolicy:
    def fixed_consumer(current, _legal, _binding):
        decision = _decision(suite, current.step_id)
        if current.step_id == 0 and fault == "unknown":
            return replace(decision, audit=replace(decision.audit, bootstrap=None))
        if current.step_id == 0 and fault == "nonzero":
            contribution = replace(decision.audit.reward_contributions[0], amount=2.0)
            return replace(
                decision, audit=replace(decision.audit, reward_contributions=(contribution,))
            )
        return decision

    return fixed_consumer


def _claim(
    tmp_path: Path,
    suite: FixedReplaySuite,
    policy: ReplayPolicy,
    *,
    kind: str = "knowledge",
    source_id: str | None = None,
    revision_sha256: str | None = None,
) -> tuple[CampaignStore, CampaignSpec, Proposal, TrialTicket, TrainingStore]:
    goal = AgentGoal.from_mapping(
        {
            "schema_version": "glr.agent-goal.v1",
            "goal_id": "goal.replay-reference",
            "objective": "Accept an inert reference after fixed offline contract checks.",
            "environment_family": "synthetic",
            "success_criteria": [
                {
                    "metric": "replay.contract_passed",
                    "operator": "eq",
                    "target": 1,
                    "source": "evaluation.replay",
                }
            ],
            "budget": {
                "max_trials": 2,
                "max_training_steps": 10,
                "max_wall_seconds": 10,
                "max_research_sources": 2,
            },
            "allowed_research_media": ["runtime-trace"],
        }
    )
    spec = CampaignSpec(
        campaign_id="campaign.replay-reference",
        goal=goal,
        environment_id=suite.spec.environment_id,
        protocol_version=suite.spec.protocol_version,
        target_id=suite.spec.metadata["target_id"],
        environment_config_sha256=suite.spec.metadata["environment_config_sha256"],
        evaluator_sha256=evaluator_sha256(policy),
        evaluation_suite_sha256=suite.sha256,
        resources=("synthetic.offline-fixture",),
        admitted_actions=("select-0", "select-1"),
        reviewers=_HOST.reviewer_ids,
        baseline_checkpoint_sha256=_BASELINE_SHA256,
    )
    episode = suite.episodes[0]
    proposal = Proposal(
        proposal_id="proposal.replay-reference",
        proposer_id="planner",
        kind=kind,
        summary="An inert knowledge revision bound to a captured synthetic source.",
        artifact_sha256=_REFERENCE_SHA256,
        base_checkpoint_sha256=_BASELINE_SHA256,
        sources=(
            EvidenceRef(
                source_id or episode.source_id,
                revision_sha256 or episode.source_sha256,
                ResearchMediaType.RUNTIME_TRACE,
                "source-step-1",
            ),
        ),
        requested_actions=("select-0",),
    )
    campaign = CampaignStore(tmp_path / "campaigns.sqlite3", host_authority=_HOST)
    campaign.create(spec, now_ns=0)
    campaign.propose(spec.campaign_id, proposal)
    ticket = campaign.claim(
        spec.campaign_id,
        proposal.proposal_id,
        worker_id="worker",
        training_steps=5,
        wall_seconds=2,
        now_ns=100,
    )
    candidate = campaign.candidate_directory(ticket)
    candidate.mkdir(parents=True)
    (candidate / "reference.json").write_bytes(_INERT_REFERENCE)
    return campaign, spec, proposal, ticket, TrainingStore(tmp_path / "runs.sqlite3")


def _run(
    campaign: CampaignStore,
    ticket: TrialTicket,
    training: TrainingStore,
    suite: FixedReplaySuite,
    policy: ReplayPolicy,
) -> str:
    # No caller evidence or zero-count mapping is supplied to this execution seam.
    return campaign.evaluate_replay(
        ticket,
        store=training,
        suite=suite,
        policy=policy,
        candidate_path="reference.json",
        now_ns=500,
        host=campaign.role_capability(_HOST, ticket, role="evaluator", principal_id="evaluator"),
    )


def _persisted_result(training: TrainingStore) -> tuple[str, dict[str, Any], dict[str, float]]:
    runs = training.list_runs()
    assert len(runs) == 1
    events = [
        event for event in training.list_events(runs[0].run_id) if event.kind == "evaluation.replay"
    ]
    assert len(events) == 1
    values = {metric.name: metric.value for metric in training.list_metrics(runs[0].run_id)}
    return runs[0].run_id, dict(events[0].payload), values


def test_real_replay_stopped_worker_and_independent_review_accept_only_inert_reference(
    tmp_path: Path,
) -> None:
    suite = _suite()
    policy = _callback(suite)
    campaign, spec, proposal, ticket, training = _claim(tmp_path, suite, policy)
    evaluation = _run(campaign, ticket, training, suite, policy)
    run_id, result, values = _persisted_result(training)
    assert result["passed"] is True
    assert result["completed_steps"] == result["expected_steps"] == 2
    assert result["evaluator_sha256"] == spec.evaluator_sha256
    assert result["suite_sha256"] == suite.sha256
    assert result["candidate_sha256"] == proposal.artifact_sha256
    assert result["source_provenance"] == [
        {"source_id": "frozen-export", "source_sha256": "3" * 64}
    ]
    assert values["replay.contract_passed"] == 1.0
    assert all(result["check_counts"][name] == values[name] == 0 for name in REPLAY_CHECK_METRICS)
    assert training.get_run(run_id).status is RunStatus.SUCCEEDED
    assert training.get_run(run_id).metadata["evaluation_scope"] == "offline-replay-inert-reference"
    decision = ReviewDecision("reviewer", evaluation, True, "7" * 64)
    with pytest.raises(ContractViolation, match="stopped worker"):
        _submit_review(campaign, ticket, decision)
    _stopped(campaign, ticket)
    worker_store = TrainingStore(tmp_path / "worker-runs.sqlite3")
    worker = worker_store.list_runs()[0]
    assert worker.status is RunStatus.SUCCEEDED
    observed = worker_store.list_events(worker.run_id)[0]
    receipt = SupervisorStopReceipt.from_mapping(observed.payload)
    assert receipt.worker_run_id == worker.run_id
    assert receipt.trial_id == ticket.trial_id
    assert receipt.trial_token == ticket.token
    assert receipt.worker_ids == (ticket.worker_id,)
    with pytest.raises(ContractViolation, match="independent"):
        _submit_review(campaign, ticket, replace(decision, reviewer_id="planner"))
    _submit_review(campaign, ticket, decision)
    state = campaign.snapshot(spec.campaign_id)["state"]
    assert state["accepted_proposals"] == {"knowledge": proposal.proposal_id}
    assert state["checkpoint_sha256"] == _BASELINE_SHA256
    assert state["score"] is None
    assert state["goal_satisfied"] is True
    reviewed = next(
        event for event in campaign.events(spec.campaign_id) if event["kind"] == "trial.reviewed"
    )["body"]
    assert reviewed["reviewer_id"] == "reviewer"
    authorization = reviewed["authorization"]
    assert authorization["candidate_kind"] == "knowledge"
    assert authorization["evaluation_scope"] == "offline-replay-inert-reference"
    assert authorization["evaluator_id"] != authorization["supervisor_id"] != "reviewer"


@pytest.mark.parametrize(
    "fault,metric",
    [
        ("unknown", REPLAY_CHECK_METRICS[4]),
        ("nonzero", REPLAY_CHECK_METRICS[5]),
        ("source_identity", REPLAY_CHECK_METRICS[1]),
    ],
)
def test_actual_replay_failures_persist_unknown_or_nonzero_counts_and_retain_claim(
    tmp_path: Path,
    fault: str,
    metric: str,
) -> None:
    frames = list(_frames())
    if fault == "source_identity":
        frames[1] = _changed(
            frames[1], timestep=replace(frames[1].snapshot(), episode_id=UUID(int=8))
        )
    suite = _suite(tuple(frames))
    policy = _callback(suite, fault)
    campaign, spec, _, ticket, training = _claim(tmp_path, suite, policy)
    with pytest.raises(ContractViolation, match="fixed replay failed or lacks evidence"):
        _run(campaign, ticket, training, suite, policy)
    run_id, result, values = _persisted_result(training)
    assert result["passed"] is False
    assert training.get_run(run_id).status is RunStatus.FAILED
    assert values["replay.contract_passed"] == 0.0
    if fault == "unknown":
        assert result["check_counts"][metric] is None
        assert result["coverage"][metric] == "unknown"
        assert metric not in values
    else:
        assert result["check_counts"][metric] == values[metric] == 1
    trial = campaign.snapshot(spec.campaign_id)["trials"][0]
    assert trial["status"] == "claimed"
    assert trial["evaluation_hash"] is None
    assert trial["stopped_receipt"] is None


@pytest.mark.parametrize("mismatch", ["source_id", "source_revision", "policy", "callback"])
def test_foreign_source_policy_and_callback_fail_before_an_evaluation_run(
    tmp_path: Path,
    mismatch: str,
) -> None:
    suite = _suite()
    policy = _callback(suite)
    campaign, spec, _, ticket, training = _claim(
        tmp_path,
        suite,
        policy,
        kind="policy" if mismatch == "policy" else "knowledge",
        source_id="absent-export" if mismatch == "source_id" else None,
        revision_sha256="4" * 64 if mismatch == "source_revision" else None,
    )
    supplied = _callback(suite, "unknown") if mismatch == "callback" else policy
    with pytest.raises(ContractViolation):
        _run(campaign, ticket, training, suite, supplied)
    assert training.list_runs() == ()
    trial = campaign.snapshot(spec.campaign_id)["trials"][0]
    assert trial["status"] == "claimed"
    assert trial["evaluation_hash"] is None


def test_targetless_replay_never_reaches_campaign_admission() -> None:
    spec = _spec()
    metadata = dict(spec.metadata, target_id=None)
    with pytest.raises(ValueError, match="identifier"):
        _suite(spec=replace(spec, metadata=metadata))
