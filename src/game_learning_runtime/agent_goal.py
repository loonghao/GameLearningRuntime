"""Agent-first goal, research, and authoritative evaluation contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from game_learning_runtime.training import KnowledgeAuthority

AGENT_GOAL_SCHEMA_VERSION = "glr.agent-goal.v1"
GOAL_EVIDENCE_SCHEMA_VERSION = "glr.goal-evidence.v1"
RESEARCH_BUNDLE_SCHEMA_VERSION = "glr.research-bundle.v1"
TRIAL_PLAN_SCHEMA_VERSION = "glr.trial-plan.v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")


class GoalOperator(str, Enum):
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"


class PromotionMode(str, Enum):
    MAX = "max"
    MIN = "min"


class ResearchMediaType(str, Enum):
    OFFICIAL_RULES = "official-rules"
    TEXT_GUIDE = "text-guide"
    VIDEO_TUTORIAL = "video-tutorial"
    RUNTIME_TRACE = "runtime-trace"


class ResearchCategory(str, Enum):
    MECHANIC = "mechanic"
    STRATEGY = "strategy"
    REWARD_HYPOTHESIS = "reward-hypothesis"
    SAFETY = "safety"
    NAVIGATION = "navigation"


class ResearchStatus(str, Enum):
    UNVERIFIED = "unverified"
    RUNTIME_VERIFIED = "runtime-verified"
    REJECTED = "rejected"


class ResearchScope(str, Enum):
    ENVIRONMENT = "environment"
    FAMILY = "family"
    GENERIC = "generic"


class Volatility(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class CheckpointPromotion:
    metric: str
    mode: PromotionMode

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CheckpointPromotion:
        _fields(value, expected=frozenset({"metric", "mode"}), path="goal.promotion")
        try:
            mode = PromotionMode(value["mode"])
        except ValueError as error:
            raise ValueError("goal.promotion.mode must be 'max' or 'min'") from error
        return cls(
            metric=_identifier(value["metric"], path="goal.promotion.metric"),
            mode=mode,
        )

    def to_mapping(self) -> dict[str, str]:
        return {"metric": self.metric, "mode": self.mode.value}


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    return value


def _sequence(value: object, *, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{path} must be an array")
    return value


def _fields(value: Mapping[str, Any], *, expected: frozenset[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(f"{path} has missing={missing} and unexpected={unexpected} fields")


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must match {_IDENTIFIER.pattern!r}")
    return value


def _text(value: object, *, path: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 and character not in "\n\t" for character in value)
    ):
        raise ValueError(f"{path} must be non-empty bounded text up to {maximum} characters")
    return value


def _number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _positive_integer(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _timestamp(value: object, *, path: str, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{path} must be an RFC 3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{path} must be an RFC 3339 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{path} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class GoalBudget:
    max_trials: int
    max_training_steps: int
    max_wall_seconds: int
    max_research_sources: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GoalBudget:
        expected = frozenset(
            {"max_trials", "max_training_steps", "max_wall_seconds", "max_research_sources"}
        )
        _fields(value, expected=expected, path="goal budget")
        return cls(
            max_trials=_positive_integer(value["max_trials"], path="budget.max_trials"),
            max_training_steps=_positive_integer(
                value["max_training_steps"], path="budget.max_training_steps"
            ),
            max_wall_seconds=_positive_integer(
                value["max_wall_seconds"], path="budget.max_wall_seconds"
            ),
            max_research_sources=_positive_integer(
                value["max_research_sources"], path="budget.max_research_sources"
            ),
        )


@dataclass(frozen=True, slots=True)
class SuccessCriterion:
    metric: str
    operator: GoalOperator
    target: float
    source: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SuccessCriterion:
        _fields(
            value,
            expected=frozenset({"metric", "operator", "target", "source"}),
            path="success criterion",
        )
        return cls(
            metric=_identifier(value["metric"], path="criterion.metric"),
            operator=GoalOperator(value["operator"]),
            target=_number(value["target"], path="criterion.target"),
            source=_identifier(value["source"], path="criterion.source"),
        )

    def accepts(self, value: float) -> bool:
        if self.operator is GoalOperator.GTE:
            return value >= self.target
        if self.operator is GoalOperator.LTE:
            return value <= self.target
        return math.isclose(value, self.target, rel_tol=1e-9, abs_tol=1e-12)


@dataclass(frozen=True, slots=True)
class GoalEvidence:
    metric: str
    value: float
    source: str
    authority: KnowledgeAuthority
    run_id: str

    def __post_init__(self) -> None:
        _identifier(self.metric, path="evidence.metric")
        _identifier(self.source, path="evidence.source")
        _identifier(self.run_id, path="evidence.run_id")
        object.__setattr__(self, "value", _number(self.value, path="evidence.value"))
        object.__setattr__(self, "authority", KnowledgeAuthority(self.authority))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GoalEvidence:
        _fields(
            value,
            expected=frozenset({"metric", "value", "source", "authority", "run_id"}),
            path="goal evidence",
        )
        return cls(
            metric=value["metric"],
            value=value["value"],
            source=value["source"],
            authority=KnowledgeAuthority(value["authority"]),
            run_id=value["run_id"],
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "value": self.value,
            "source": self.source,
            "authority": self.authority.value,
            "run_id": self.run_id,
        }


@dataclass(frozen=True, slots=True)
class CriterionEvaluation:
    criterion: SuccessCriterion
    observed: float | None
    evidence_run_id: str | None
    passed: bool


@dataclass(frozen=True, slots=True)
class GoalEvaluation:
    goal_id: str
    satisfied: bool
    criteria: tuple[CriterionEvaluation, ...]


@dataclass(frozen=True, slots=True)
class AgentGoal:
    goal_id: str
    objective: str
    environment_family: str
    success_criteria: tuple[SuccessCriterion, ...]
    budget: GoalBudget
    allowed_research_media: tuple[ResearchMediaType, ...]
    promotion: CheckpointPromotion | None = None
    schema_version: str = AGENT_GOAL_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AgentGoal:
        fields = dict(value)
        fields.pop("promotion", None)
        _fields(
            fields,
            expected=frozenset(
                {
                    "schema_version",
                    "goal_id",
                    "objective",
                    "environment_family",
                    "success_criteria",
                    "budget",
                    "allowed_research_media",
                }
            ),
            path="agent goal",
        )
        if value["schema_version"] != AGENT_GOAL_SCHEMA_VERSION:
            raise ValueError(f"goal.schema_version must be {AGENT_GOAL_SCHEMA_VERSION!r}")
        criteria = tuple(
            SuccessCriterion.from_mapping(_mapping(item, path="success_criteria[]"))
            for item in _sequence(value["success_criteria"], path="success_criteria")
        )
        if not criteria:
            raise ValueError("success_criteria cannot be empty")
        keys = [(criterion.metric, criterion.source) for criterion in criteria]
        if len(set(keys)) != len(keys):
            raise ValueError("success_criteria contains duplicate metric/source pairs")
        try:
            media = tuple(
                ResearchMediaType(item)
                for item in _sequence(
                    value["allowed_research_media"], path="allowed_research_media"
                )
            )
        except ValueError as error:
            raise ValueError("allowed_research_media contains an unsupported value") from error
        if not media or len(set(media)) != len(media):
            raise ValueError("allowed_research_media must be non-empty and unique")
        return cls(
            schema_version=value["schema_version"],
            goal_id=_identifier(value["goal_id"], path="goal.goal_id"),
            objective=_text(value["objective"], path="goal.objective", maximum=2048),
            environment_family=_identifier(
                value["environment_family"], path="goal.environment_family"
            ),
            success_criteria=criteria,
            budget=GoalBudget.from_mapping(_mapping(value["budget"], path="goal.budget")),
            allowed_research_media=media,
            promotion=(
                None
                if value.get("promotion") is None
                else CheckpointPromotion.from_mapping(
                    _mapping(value["promotion"], path="goal.promotion")
                )
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "goal_id": self.goal_id,
            "objective": self.objective,
            "environment_family": self.environment_family,
            "success_criteria": [
                {
                    "metric": criterion.metric,
                    "operator": criterion.operator.value,
                    "target": criterion.target,
                    "source": criterion.source,
                }
                for criterion in self.success_criteria
            ],
            "budget": {
                "max_trials": self.budget.max_trials,
                "max_training_steps": self.budget.max_training_steps,
                "max_wall_seconds": self.budget.max_wall_seconds,
                "max_research_sources": self.budget.max_research_sources,
            },
            "allowed_research_media": [media.value for media in self.allowed_research_media],
        }
        if self.promotion is not None:
            result["promotion"] = self.promotion.to_mapping()
        return result

    def evaluate(self, evidence: Iterable[GoalEvidence]) -> GoalEvaluation:
        by_key: dict[tuple[str, str], GoalEvidence] = {}
        for item in evidence:
            if not isinstance(item, GoalEvidence):
                raise TypeError("goal evidence must contain GoalEvidence values")
            key = (item.metric, item.source)
            if key in by_key:
                raise ValueError(
                    f"duplicate goal evidence for {item.metric!r} from {item.source!r}"
                )
            by_key[key] = item
        evaluations: list[CriterionEvaluation] = []
        for criterion in self.success_criteria:
            matched_evidence = by_key.get((criterion.metric, criterion.source))
            passed = (
                matched_evidence is not None
                and matched_evidence.authority is KnowledgeAuthority.AUTHORITATIVE
                and criterion.accepts(matched_evidence.value)
            )
            evaluations.append(
                CriterionEvaluation(
                    criterion=criterion,
                    observed=None if matched_evidence is None else matched_evidence.value,
                    evidence_run_id=(None if matched_evidence is None else matched_evidence.run_id),
                    passed=passed,
                )
            )
        return GoalEvaluation(
            goal_id=self.goal_id,
            satisfied=all(item.passed for item in evaluations),
            criteria=tuple(evaluations),
        )


@dataclass(frozen=True, slots=True)
class ResearchSource:
    source_id: str
    media_type: ResearchMediaType
    url: str
    publisher: str
    title: str
    accessed_at: datetime
    updated_at: datetime | None
    summary: str
    confidence: float
    volatility: Volatility

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchSource:
        _fields(
            value,
            expected=frozenset(
                {
                    "source_id",
                    "media_type",
                    "url",
                    "publisher",
                    "title",
                    "accessed_at",
                    "updated_at",
                    "summary",
                    "confidence",
                    "volatility",
                }
            ),
            path="research source",
        )
        url = _text(value["url"], path="research source.url", maximum=2048)
        parsed_url = urlparse(url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise ValueError("research source.url must be an HTTPS URL without credentials")
        confidence = _number(value["confidence"], path="research source.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("research source.confidence must be between 0 and 1")
        accessed = _timestamp(value["accessed_at"], path="research source.accessed_at")
        assert accessed is not None
        return cls(
            source_id=_identifier(value["source_id"], path="research source.source_id"),
            media_type=ResearchMediaType(value["media_type"]),
            url=url,
            publisher=_text(value["publisher"], path="research source.publisher", maximum=256),
            title=_text(value["title"], path="research source.title", maximum=512),
            accessed_at=accessed,
            updated_at=_timestamp(
                value["updated_at"], path="research source.updated_at", optional=True
            ),
            summary=_text(value["summary"], path="research source.summary", maximum=1024),
            confidence=confidence,
            volatility=Volatility(value["volatility"]),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "media_type": self.media_type.value,
            "url": self.url,
            "publisher": self.publisher,
            "title": self.title,
            "accessed_at": _format_timestamp(self.accessed_at),
            "updated_at": _format_timestamp(self.updated_at),
            "summary": self.summary,
            "confidence": self.confidence,
            "volatility": self.volatility.value,
        }


@dataclass(frozen=True, slots=True)
class ResearchFinding:
    finding_id: str
    category: ResearchCategory
    status: ResearchStatus
    scope: ResearchScope
    scope_id: str | None
    summary: str
    source_ids: tuple[str, ...]
    tags: tuple[str, ...]
    confidence: float
    locator: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchFinding:
        _fields(
            value,
            expected=frozenset(
                {
                    "finding_id",
                    "category",
                    "status",
                    "scope",
                    "scope_id",
                    "summary",
                    "source_ids",
                    "tags",
                    "confidence",
                    "locator",
                }
            ),
            path="research finding",
        )
        scope = ResearchScope(value["scope"])
        raw_scope_id = value["scope_id"]
        if scope is ResearchScope.GENERIC:
            if raw_scope_id is not None:
                raise ValueError("generic research findings cannot declare scope_id")
            scope_id = None
        else:
            scope_id = _identifier(raw_scope_id, path="research finding.scope_id")
        source_ids = tuple(
            _identifier(item, path="research finding.source_ids[]")
            for item in _sequence(value["source_ids"], path="research finding.source_ids")
        )
        if not source_ids or len(set(source_ids)) != len(source_ids):
            raise ValueError("research finding.source_ids must be non-empty and unique")
        tags = tuple(
            _identifier(item, path="research finding.tags[]")
            for item in _sequence(value["tags"], path="research finding.tags")
        )
        if len(set(tags)) != len(tags):
            raise ValueError("research finding.tags must be unique")
        confidence = _number(value["confidence"], path="research finding.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("research finding.confidence must be between 0 and 1")
        locator = value["locator"]
        if locator is not None:
            locator = _text(locator, path="research finding.locator", maximum=128)
        return cls(
            finding_id=_identifier(value["finding_id"], path="research finding.finding_id"),
            category=ResearchCategory(value["category"]),
            status=ResearchStatus(value["status"]),
            scope=scope,
            scope_id=scope_id,
            summary=_text(value["summary"], path="research finding.summary", maximum=1024),
            source_ids=source_ids,
            tags=tags,
            confidence=confidence,
            locator=locator,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            "category": self.category.value,
            "status": self.status.value,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "summary": self.summary,
            "source_ids": list(self.source_ids),
            "tags": list(self.tags),
            "confidence": self.confidence,
            "locator": self.locator,
        }


@dataclass(frozen=True, slots=True)
class ResearchBundle:
    """Provenance-bound, non-executable knowledge distilled from allowed sources."""

    sources: tuple[ResearchSource, ...]
    findings: tuple[ResearchFinding, ...]
    schema_version: str = RESEARCH_BUNDLE_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchBundle:
        _fields(
            value,
            expected=frozenset({"schema_version", "sources", "findings"}),
            path="research bundle",
        )
        if value["schema_version"] != RESEARCH_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"research bundle.schema_version must be {RESEARCH_BUNDLE_SCHEMA_VERSION!r}"
            )
        raw_sources = _sequence(value["sources"], path="research bundle.sources")
        raw_findings = _sequence(value["findings"], path="research bundle.findings")
        if len(raw_sources) > 256 or len(raw_findings) > 2048:
            raise ValueError("research bundle exceeds its bounded source or finding count")
        sources = tuple(
            ResearchSource.from_mapping(_mapping(item, path="research bundle.sources[]"))
            for item in raw_sources
        )
        findings = tuple(
            ResearchFinding.from_mapping(_mapping(item, path="research bundle.findings[]"))
            for item in raw_findings
        )
        source_ids = [source.source_id for source in sources]
        finding_ids = [finding.finding_id for finding in findings]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("research bundle contains duplicate source_id values")
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("research bundle contains duplicate finding_id values")
        known_sources = set(source_ids)
        for finding in findings:
            missing_sources = sorted(set(finding.source_ids) - known_sources)
            if missing_sources:
                raise ValueError(
                    f"research finding {finding.finding_id!r} references unknown sources "
                    f"{missing_sources}"
                )
            if finding.status is ResearchStatus.RUNTIME_VERIFIED and not any(
                source.source_id in finding.source_ids
                and source.media_type is ResearchMediaType.RUNTIME_TRACE
                for source in sources
            ):
                raise ValueError(
                    "runtime-verified finding "
                    f"{finding.finding_id!r} requires runtime trace provenance"
                )
        return cls(
            sources=sources,
            findings=findings,
            schema_version=RESEARCH_BUNDLE_SCHEMA_VERSION,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sources": [source.to_mapping() for source in self.sources],
            "findings": [finding.to_mapping() for finding in self.findings],
        }


@dataclass(frozen=True, slots=True)
class RewardTerm:
    """A bounded declarative reward adjustment, never executable code."""

    name: str
    metric: str
    weight: float
    rationale: str
    source_finding_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RewardTerm:
        _fields(
            value,
            expected=frozenset({"name", "metric", "weight", "rationale", "source_finding_ids"}),
            path="reward term",
        )
        weight = _number(value["weight"], path="reward term.weight")
        if abs(weight) > 1000:
            raise ValueError("reward term.weight must be between -1000 and 1000")
        source_ids = tuple(
            _identifier(item, path="reward term.source_finding_ids[]")
            for item in _sequence(
                value["source_finding_ids"], path="reward term.source_finding_ids"
            )
        )
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("reward term.source_finding_ids must be unique")
        return cls(
            name=_identifier(value["name"], path="reward term.name"),
            metric=_identifier(value["metric"], path="reward term.metric"),
            weight=weight,
            rationale=_text(value["rationale"], path="reward term.rationale", maximum=1024),
            source_finding_ids=source_ids,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "metric": self.metric,
            "weight": self.weight,
            "rationale": self.rationale,
            "source_finding_ids": list(self.source_finding_ids),
        }


@dataclass(frozen=True, slots=True)
class TrialPlan:
    """Learner-neutral plan exchanged between an agent planner and project trainer."""

    trial_id: str
    goal_id: str
    seed: int
    max_steps: int
    reward_terms: tuple[RewardTerm, ...]
    notes: str
    schema_version: str = TRIAL_PLAN_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrialPlan:
        _fields(
            value,
            expected=frozenset(
                {
                    "schema_version",
                    "trial_id",
                    "goal_id",
                    "seed",
                    "max_steps",
                    "reward_terms",
                    "notes",
                }
            ),
            path="trial plan",
        )
        if value["schema_version"] != TRIAL_PLAN_SCHEMA_VERSION:
            raise ValueError(f"trial plan.schema_version must be {TRIAL_PLAN_SCHEMA_VERSION!r}")
        seed = value["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("trial plan.seed must be a non-negative integer")
        terms = tuple(
            RewardTerm.from_mapping(_mapping(item, path="trial plan.reward_terms[]"))
            for item in _sequence(value["reward_terms"], path="trial plan.reward_terms")
        )
        if len(terms) > 128:
            raise ValueError("trial plan cannot contain more than 128 reward terms")
        names = [term.name for term in terms]
        if len(names) != len(set(names)):
            raise ValueError("trial plan reward term names must be unique")
        return cls(
            trial_id=_identifier(value["trial_id"], path="trial plan.trial_id"),
            goal_id=_identifier(value["goal_id"], path="trial plan.goal_id"),
            seed=seed,
            max_steps=_positive_integer(value["max_steps"], path="trial plan.max_steps"),
            reward_terms=terms,
            notes=_text(value["notes"], path="trial plan.notes", maximum=4096),
            schema_version=TRIAL_PLAN_SCHEMA_VERSION,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trial_id": self.trial_id,
            "goal_id": self.goal_id,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "reward_terms": [term.to_mapping() for term in self.reward_terms],
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class GoalEvidenceBundle:
    goal_id: str
    trial_id: str
    evidence: tuple[GoalEvidence, ...]
    schema_version: str = GOAL_EVIDENCE_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GoalEvidenceBundle:
        _fields(
            value,
            expected=frozenset({"schema_version", "goal_id", "trial_id", "evidence"}),
            path="goal evidence bundle",
        )
        if value["schema_version"] != GOAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"goal evidence.schema_version must be {GOAL_EVIDENCE_SCHEMA_VERSION!r}"
            )
        evidence = tuple(
            GoalEvidence.from_mapping(_mapping(item, path="goal evidence.evidence[]"))
            for item in _sequence(value["evidence"], path="goal evidence.evidence")
        )
        keys = [(item.metric, item.source) for item in evidence]
        if len(keys) != len(set(keys)):
            raise ValueError("goal evidence contains duplicate metric/source pairs")
        return cls(
            goal_id=_identifier(value["goal_id"], path="goal evidence.goal_id"),
            trial_id=_identifier(value["trial_id"], path="goal evidence.trial_id"),
            evidence=evidence,
            schema_version=GOAL_EVIDENCE_SCHEMA_VERSION,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "goal_id": self.goal_id,
            "trial_id": self.trial_id,
            "evidence": [item.to_mapping() for item in self.evidence],
        }


__all__ = [
    "AGENT_GOAL_SCHEMA_VERSION",
    "GOAL_EVIDENCE_SCHEMA_VERSION",
    "RESEARCH_BUNDLE_SCHEMA_VERSION",
    "TRIAL_PLAN_SCHEMA_VERSION",
    "AgentGoal",
    "CheckpointPromotion",
    "CriterionEvaluation",
    "GoalBudget",
    "GoalEvaluation",
    "GoalEvidence",
    "GoalEvidenceBundle",
    "GoalOperator",
    "PromotionMode",
    "ResearchBundle",
    "ResearchCategory",
    "ResearchFinding",
    "ResearchMediaType",
    "ResearchScope",
    "ResearchSource",
    "ResearchStatus",
    "RewardTerm",
    "SuccessCriterion",
    "TrialPlan",
    "Volatility",
]
