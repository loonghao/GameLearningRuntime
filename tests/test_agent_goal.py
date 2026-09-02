from __future__ import annotations

import math
from copy import deepcopy

import pytest

from game_learning_runtime.agent_goal import (
    AgentGoal,
    GoalEvidence,
    GoalEvidenceBundle,
    ResearchBundle,
    ResearchFinding,
    ResearchSource,
    RewardTerm,
    TrialPlan,
)
from game_learning_runtime.training import KnowledgeAuthority


def test_agent_goal_combines_bounded_research_with_authoritative_success_evidence() -> None:
    goal = AgentGoal.from_mapping(
        {
            "schema_version": "glr.agent-goal.v1",
            "goal_id": "goal.reach-destination",
            "objective": "Reach the requested destination and verify arrival.",
            "environment_family": "action-rpg",
            "success_criteria": [
                {
                    "metric": "objective.arrived",
                    "operator": "gte",
                    "target": 1,
                    "source": "runtime.telemetry",
                }
            ],
            "budget": {
                "max_trials": 8,
                "max_training_steps": 50000,
                "max_wall_seconds": 14400,
                "max_research_sources": 64,
            },
            "allowed_research_media": ["official-rules", "text-guide", "video-tutorial"],
        }
    )
    video = ResearchSource.from_mapping(
        {
            "source_id": "guide.route-video",
            "media_type": "video-tutorial",
            "url": "https://example.com/tutorial",
            "publisher": "Example Publisher",
            "title": "Route tutorial",
            "accessed_at": "2026-09-01T10:00:00Z",
            "updated_at": None,
            "summary": "Shows a route and the hazards encountered along it.",
            "confidence": 0.8,
            "volatility": "medium",
        }
    )
    finding = ResearchFinding.from_mapping(
        {
            "finding_id": "strategy.route-hazards",
            "category": "strategy",
            "status": "unverified",
            "scope": "family",
            "scope_id": "action-rpg",
            "summary": "Use safe landmarks and re-observe after each hazard.",
            "source_ids": [video.source_id],
            "tags": ["navigate", "recover"],
            "confidence": 0.8,
            "locator": "00:01:10-00:02:05",
        }
    )

    evaluation = goal.evaluate(
        [
            GoalEvidence(
                metric="objective.arrived",
                value=1.0,
                source="runtime.telemetry",
                authority=KnowledgeAuthority.AUTHORITATIVE,
                run_id="run-evaluation",
            )
        ]
    )

    assert video.media_type.value == "video-tutorial"
    assert finding.scope.value == "family"
    assert evaluation.satisfied is True
    assert evaluation.criteria[0].passed is True

    bundle = ResearchBundle.from_mapping(
        {
            "schema_version": "glr.research-bundle.v1",
            "sources": [video.to_mapping()],
            "findings": [finding.to_mapping()],
        }
    )
    assert bundle.sources == (video,)
    assert bundle.findings == (finding,)


def _goal_value() -> dict[str, object]:
    return {
        "schema_version": "glr.agent-goal.v1",
        "goal_id": "goal.test",
        "objective": "Reach a bounded test objective.",
        "environment_family": "action-rpg",
        "success_criteria": [
            {"metric": "score", "operator": "gte", "target": 1, "source": "runtime"}
        ],
        "budget": {
            "max_trials": 2,
            "max_training_steps": 10,
            "max_wall_seconds": 10,
            "max_research_sources": 2,
        },
        "allowed_research_media": ["text-guide"],
    }


def _source_value(source_id: str = "guide.one") -> dict[str, object]:
    return {
        "source_id": source_id,
        "media_type": "text-guide",
        "url": "https://example.com/guide",
        "publisher": "Example",
        "title": "Guide",
        "accessed_at": "2026-09-01T00:00:00Z",
        "updated_at": "2026-08-01T00:00:00+00:00",
        "summary": "A compact guide summary.",
        "confidence": 0.5,
        "volatility": "low",
    }


def _finding_value(finding_id: str = "strategy.one") -> dict[str, object]:
    return {
        "finding_id": finding_id,
        "category": "strategy",
        "status": "unverified",
        "scope": "generic",
        "scope_id": None,
        "summary": "A bounded strategy hypothesis.",
        "source_ids": ["guide.one"],
        "tags": ["combat"],
        "confidence": 0.5,
        "locator": None,
    }


def test_goal_operators_and_advisory_evidence_fail_closed() -> None:
    value = _goal_value()
    value["success_criteria"] = [
        {"metric": "time", "operator": "lte", "target": 10, "source": "runtime"},
        {"metric": "phase", "operator": "eq", "target": 2, "source": "runtime"},
    ]
    goal = AgentGoal.from_mapping(value)
    evaluation = goal.evaluate(
        [
            GoalEvidence(
                metric="time",
                value=9,
                source="runtime",
                authority=KnowledgeAuthority.ADVISORY,
                run_id="run-one",
            ),
            GoalEvidence(
                metric="phase",
                value=2,
                source="runtime",
                authority=KnowledgeAuthority.AUTHORITATIVE,
                run_id="run-one",
            ),
        ]
    )
    assert evaluation.satisfied is False
    assert goal.to_mapping() == value
    with pytest.raises(TypeError, match="GoalEvidence"):
        goal.evaluate([object()])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="duplicate"):
        goal.evaluate(
            [
                GoalEvidence("time", 1, "runtime", KnowledgeAuthority.ADVISORY, "run-one"),
                GoalEvidence("time", 2, "runtime", KnowledgeAuthority.ADVISORY, "run-two"),
            ]
        )


def test_goal_checkpoint_promotion_is_optional_and_validated() -> None:
    value = _goal_value()
    value["promotion"] = {"metric": "victories", "mode": "max"}
    goal = AgentGoal.from_mapping(value)
    assert goal.promotion is not None
    assert goal.promotion.metric == "victories"
    assert goal.to_mapping()["promotion"] == {"metric": "victories", "mode": "max"}
    value["promotion"] = {"metric": "victories", "mode": "sideways"}
    with pytest.raises(ValueError, match=r"max.*min"):
        AgentGoal.from_mapping(value)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version="wrong"), "schema_version"),
        (lambda value: value.update(success_criteria=[]), "cannot be empty"),
        (
            lambda value: value.update(success_criteria=value["success_criteria"] * 2),
            "duplicate",
        ),
        (lambda value: value.update(allowed_research_media=["unknown"]), "unsupported"),
        (lambda value: value.update(allowed_research_media=[]), "non-empty"),
        (lambda value: value.update(goal_id="INVALID"), "must match"),
        (lambda value: value.update(objective=""), "bounded text"),
    ],
)
def test_agent_goal_rejects_invalid_contracts(mutate: object, message: str) -> None:
    value = _goal_value()
    mutate(value)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError), match=message):
        AgentGoal.from_mapping(value)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("url", "http://example.com", "HTTPS"),
        ("url", "https://user:secret@example.com", "credentials"),
        ("confidence", 2, "between 0 and 1"),
        ("accessed_at", "not-a-time", "RFC 3339"),
        ("accessed_at", "2026-09-01T00:00:00", "timezone"),
        ("source_id", "INVALID", "must match"),
    ],
)
def test_research_source_rejects_invalid_provenance(
    field: str, replacement: object, message: str
) -> None:
    value = _source_value()
    value[field] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        ResearchSource.from_mapping(value)


def test_research_bundle_rejects_scope_duplicates_and_false_verification() -> None:
    source = _source_value()
    finding = _finding_value()
    bad_scope = deepcopy(finding)
    bad_scope["scope_id"] = "unexpected"
    with pytest.raises(ValueError, match="generic"):
        ResearchFinding.from_mapping(bad_scope)
    duplicate_tags = deepcopy(finding)
    duplicate_tags["tags"] = ["combat", "combat"]
    with pytest.raises(ValueError, match="tags"):
        ResearchFinding.from_mapping(duplicate_tags)
    duplicate_sources = deepcopy(finding)
    duplicate_sources["source_ids"] = ["guide.one", "guide.one"]
    with pytest.raises(ValueError, match="source_ids"):
        ResearchFinding.from_mapping(duplicate_sources)
    bad_confidence = deepcopy(finding)
    bad_confidence["confidence"] = -1
    with pytest.raises(ValueError, match="between 0 and 1"):
        ResearchFinding.from_mapping(bad_confidence)

    bundle_value = {
        "schema_version": "glr.research-bundle.v1",
        "sources": [source],
        "findings": [finding],
    }
    unknown = deepcopy(bundle_value)
    unknown["findings"][0]["source_ids"] = ["guide.missing"]  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown sources"):
        ResearchBundle.from_mapping(unknown)
    duplicate = deepcopy(bundle_value)
    duplicate["sources"] = [source, source]
    with pytest.raises(ValueError, match="duplicate source_id"):
        ResearchBundle.from_mapping(duplicate)
    verified = deepcopy(bundle_value)
    verified["findings"][0]["status"] = "runtime-verified"  # type: ignore[index]
    with pytest.raises(ValueError, match="runtime trace"):
        ResearchBundle.from_mapping(verified)


def test_trial_and_evidence_contracts_round_trip_and_reject_unsafe_values() -> None:
    term_value = {
        "name": "arrival",
        "metric": "objective.arrived",
        "weight": 10,
        "rationale": "Reward verified arrival.",
        "source_finding_ids": ["strategy.one"],
    }
    term = RewardTerm.from_mapping(term_value)
    plan_value = {
        "schema_version": "glr.trial-plan.v1",
        "trial_id": "trial-one",
        "goal_id": "goal.test",
        "seed": 0,
        "max_steps": 10,
        "reward_terms": [term.to_mapping()],
        "notes": "A bounded trial.",
    }
    plan = TrialPlan.from_mapping(plan_value)
    assert plan.to_mapping() == plan_value
    with pytest.raises(ValueError, match="between -1000 and 1000"):
        RewardTerm.from_mapping({**term_value, "weight": 1001})
    with pytest.raises(ValueError, match="unique"):
        RewardTerm.from_mapping(
            {**term_value, "source_finding_ids": ["strategy.one", "strategy.one"]}
        )
    with pytest.raises(ValueError, match="seed"):
        TrialPlan.from_mapping({**plan_value, "seed": -1})
    with pytest.raises(ValueError, match="term names"):
        TrialPlan.from_mapping(
            {**plan_value, "reward_terms": [term.to_mapping(), term.to_mapping()]}
        )

    evidence_value = {
        "schema_version": "glr.goal-evidence.v1",
        "goal_id": "goal.test",
        "trial_id": "trial-one",
        "evidence": [
            {
                "metric": "score",
                "value": 1,
                "source": "runtime",
                "authority": "authoritative",
                "run_id": "run-one",
            }
        ],
    }
    evidence = GoalEvidenceBundle.from_mapping(evidence_value)
    assert evidence.to_mapping() == evidence_value
    with pytest.raises(ValueError, match="duplicate"):
        GoalEvidenceBundle.from_mapping(
            {**evidence_value, "evidence": evidence_value["evidence"] * 2}
        )


def test_agent_contracts_reject_wrong_shapes_versions_and_unbounded_counts() -> None:
    goal = _goal_value()
    with pytest.raises(TypeError, match="array"):
        AgentGoal.from_mapping({**goal, "success_criteria": "wrong"})
    with pytest.raises(TypeError, match="number"):
        criterion = deepcopy(goal["success_criteria"])
        criterion[0]["target"] = "one"  # type: ignore[index]
        AgentGoal.from_mapping({**goal, "success_criteria": criterion})
    with pytest.raises(ValueError, match="finite"):
        criterion = deepcopy(goal["success_criteria"])
        criterion[0]["target"] = math.inf  # type: ignore[index]
        AgentGoal.from_mapping({**goal, "success_criteria": criterion})
    with pytest.raises(ValueError, match="positive"):
        budget = dict(goal["budget"])  # type: ignore[arg-type]
        budget["max_trials"] = 0
        AgentGoal.from_mapping({**goal, "budget": budget})
    with pytest.raises(TypeError, match="timestamp"):
        ResearchSource.from_mapping({**_source_value(), "accessed_at": None})

    with pytest.raises(ValueError, match="schema_version"):
        ResearchBundle.from_mapping({"schema_version": "wrong", "sources": [], "findings": []})
    with pytest.raises(ValueError, match="bounded"):
        ResearchBundle.from_mapping(
            {
                "schema_version": "glr.research-bundle.v1",
                "sources": [{}] * 257,
                "findings": [],
            }
        )
    with pytest.raises(TypeError, match="object"):
        ResearchBundle.from_mapping(
            {
                "schema_version": "glr.research-bundle.v1",
                "sources": [1],
                "findings": [],
            }
        )
    source = _source_value()
    finding = _finding_value()
    with pytest.raises(ValueError, match="duplicate finding_id"):
        ResearchBundle.from_mapping(
            {
                "schema_version": "glr.research-bundle.v1",
                "sources": [source],
                "findings": [finding, finding],
            }
        )
    bundle = ResearchBundle.from_mapping(
        {
            "schema_version": "glr.research-bundle.v1",
            "sources": [source],
            "findings": [finding],
        }
    )
    assert bundle.to_mapping()["schema_version"] == "glr.research-bundle.v1"

    term = {
        "name": "reward",
        "metric": "score",
        "weight": 1,
        "rationale": "A bounded reward.",
        "source_finding_ids": [],
    }
    plan = {
        "schema_version": "glr.trial-plan.v1",
        "trial_id": "trial-one",
        "goal_id": "goal.test",
        "seed": 0,
        "max_steps": 1,
        "reward_terms": [],
        "notes": "A bounded trial.",
    }
    with pytest.raises(ValueError, match="schema_version"):
        TrialPlan.from_mapping({**plan, "schema_version": "wrong"})
    with pytest.raises(ValueError, match="128"):
        TrialPlan.from_mapping({**plan, "reward_terms": [term] * 129})
    with pytest.raises(ValueError, match="schema_version"):
        GoalEvidenceBundle.from_mapping(
            {
                "schema_version": "wrong",
                "goal_id": "goal.test",
                "trial_id": "trial-one",
                "evidence": [],
            }
        )
