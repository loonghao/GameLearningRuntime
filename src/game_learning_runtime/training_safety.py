"""Fail-closed reward and demonstration-data safety contracts.

Policies are versioned JSON data without callbacks, expressions, import paths,
endpoints, or executable code. Adapters produce authoritative signals and
provenance; GLR validates them before reward or BC ingestion.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.training import RewardComposer, RewardSignal, TrainingConfig

REWARD_SAFETY_SCHEMA_VERSION = "glr.reward-safety.v1"
DEMONSTRATION_POLICY_SCHEMA_VERSION = "glr.demonstration-policy.v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_FAILURE_CORRECTION = "guardrail.failure-correction"
_EnumT = TypeVar("_EnumT", bound=Enum)


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} requires string keys")
    return value


def _sequence(value: object, *, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{path} must be an array")
    return value


def _reject_unknown(value: Mapping[str, Any], *, allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unexpected fields: {unknown}")


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must match {_IDENTIFIER.pattern!r}")
    return value


def _number(value: object, *, path: str, non_negative: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if non_negative and result < 0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _clip(value: float, minimum: float | None, maximum: float | None) -> float:
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


@dataclass(frozen=True, slots=True)
class RewardSafetyConfig:
    """Episode-level budget and terminal-outcome policy."""

    schema_version: str
    outcome_signal: str
    shaping_signals: tuple[str, ...]
    max_positive_shaping_per_step: float
    max_positive_shaping_per_episode: float
    failure_episode_maximum: float
    require_terminal_outcome: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RewardSafetyConfig:
        _reject_unknown(
            value,
            allowed=frozenset(
                {
                    "schema_version",
                    "outcome_signal",
                    "shaping_signals",
                    "max_positive_shaping_per_step",
                    "max_positive_shaping_per_episode",
                    "failure_episode_maximum",
                    "require_terminal_outcome",
                }
            ),
            path="reward safety",
        )
        schema_version = value.get("schema_version")
        if schema_version != REWARD_SAFETY_SCHEMA_VERSION:
            raise ValueError(
                "reward safety.schema_version must be "
                f"{REWARD_SAFETY_SCHEMA_VERSION!r}; received {schema_version!r}"
            )
        outcome = _identifier(value.get("outcome_signal"), path="outcome_signal")
        shaping = tuple(
            _identifier(item, path="shaping_signals[]")
            for item in _sequence(value.get("shaping_signals"), path="shaping_signals")
        )
        if not shaping:
            raise ValueError("shaping_signals requires at least one signal")
        if len(set(shaping)) != len(shaping):
            raise ValueError("shaping_signals contains duplicates")
        if outcome in shaping:
            raise ValueError("outcome_signal cannot also be a shaping signal")
        per_step = _number(
            value.get("max_positive_shaping_per_step"),
            path="max_positive_shaping_per_step",
            non_negative=True,
        )
        per_episode = _number(
            value.get("max_positive_shaping_per_episode"),
            path="max_positive_shaping_per_episode",
            non_negative=True,
        )
        if per_step > per_episode:
            raise ValueError(
                "max_positive_shaping_per_step cannot exceed max_positive_shaping_per_episode"
            )
        return cls(
            schema_version=schema_version,
            outcome_signal=outcome,
            shaping_signals=shaping,
            max_positive_shaping_per_step=per_step,
            max_positive_shaping_per_episode=per_episode,
            failure_episode_maximum=_number(
                value.get("failure_episode_maximum"), path="failure_episode_maximum"
            ),
            require_terminal_outcome=_boolean(
                value.get("require_terminal_outcome"), path="require_terminal_outcome"
            ),
        )


@dataclass(frozen=True, slots=True)
class GuardedRewardResult:
    """One guarded step plus episode-level audit counters."""

    total: float
    contributions: Mapping[str, float]
    episode_total: float
    positive_shaping_total: float
    suppressed_positive_shaping: float
    terminal: bool

    def __post_init__(self) -> None:
        for name in (
            "total",
            "episode_total",
            "positive_shaping_total",
            "suppressed_positive_shaping",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "contributions", MappingProxyType(dict(self.contributions)))


class EpisodeRewardGuard:
    """Apply shaping budgets and terminal failure dominance to rewards."""

    def __init__(self, training: TrainingConfig, safety: RewardSafetyConfig) -> None:
        terms = {term.name: term for term in training.reward.terms}
        outcome = terms.get(safety.outcome_signal)
        if outcome is None:
            raise ContractViolation(
                f"reward safety references unknown outcome signal {safety.outcome_signal!r}"
            )
        if outcome.weight <= 0:
            raise ContractViolation("the terminal outcome reward term must have a positive weight")
        missing_shaping = sorted(set(safety.shaping_signals) - terms.keys())
        if missing_shaping:
            raise ContractViolation(
                f"reward safety references unknown shaping signals: {missing_shaping}"
            )
        self._training = training
        self._safety = safety
        self._composer = RewardComposer(training)
        self.reset()

    def reset(self) -> None:
        """Open a fresh episode after a terminal transition."""

        self._episode_total = 0.0
        self._positive_shaping_total = 0.0
        self._closed = False

    def compose(
        self, signals: Iterable[RewardSignal], *, terminal: bool = False
    ) -> GuardedRewardResult:
        """Compose one step and enforce episode-level safety constraints."""

        if self._closed:
            raise ContractViolation("reward episode is terminal; call reset before composing")
        received = tuple(signals)
        outcome_signal = next(
            (signal for signal in received if signal.name == self._safety.outcome_signal), None
        )
        if outcome_signal is not None and not terminal:
            raise ContractViolation("the outcome signal may only be emitted on a terminal step")
        if terminal and self._safety.require_terminal_outcome and outcome_signal is None:
            raise ContractViolation("terminal outcome signal is required on every terminal step")

        composed = self._composer.compose(received)
        contributions = dict(composed.contributions)
        positive_shaping = math.fsum(
            value
            for name, value in contributions.items()
            if name in self._safety.shaping_signals and value > 0
        )
        remaining = max(
            0.0,
            self._safety.max_positive_shaping_per_episode - self._positive_shaping_total,
        )
        accepted = min(positive_shaping, self._safety.max_positive_shaping_per_step, remaining)
        if positive_shaping > 0:
            scale = accepted / positive_shaping
            for name in self._safety.shaping_signals:
                contribution = contributions.get(name)
                if contribution is not None and contribution > 0:
                    contributions[name] = contribution * scale

        step_total = _clip(
            math.fsum(contributions.values()),
            self._training.reward.minimum,
            self._training.reward.maximum,
        )
        self._positive_shaping_total += accepted
        self._episode_total += step_total
        failed = terminal and outcome_signal is not None and outcome_signal.value < 0
        if failed and self._episode_total > self._safety.failure_episode_maximum:
            correction = self._safety.failure_episode_maximum - self._episode_total
            contributions[_FAILURE_CORRECTION] = correction
            step_total += correction
            self._episode_total = self._safety.failure_episode_maximum

        if terminal:
            self._closed = True
        return GuardedRewardResult(
            total=step_total,
            contributions=contributions,
            episode_total=self._episode_total,
            positive_shaping_total=self._positive_shaping_total,
            suppressed_positive_shaping=positive_shaping - accepted,
            terminal=terminal,
        )


class DemonstrationOrigin(str, Enum):
    """Declared producer of a demonstration action."""

    HUMAN = "human"
    SCRIPTED_EXPERT = "scripted-expert"
    POLICY = "policy"
    UNKNOWN = "unknown"


class DemonstrationOutcome(str, Enum):
    """Authoritative episode result attached to a demonstration."""

    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


def _enum(value: object, enum_type: type[_EnumT], *, path: str) -> _EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{path} must be one of: {choices}") from error


def _weights(
    value: object,
    *,
    allowed: tuple[_EnumT, ...],
    enum_type: type[_EnumT],
    path: str,
) -> Mapping[_EnumT, float]:
    raw = _mapping(value, path=path)
    parsed = {
        _enum(key, enum_type, path=f"{path} key"): _number(weight, path=f"{path}.{key}")
        for key, weight in raw.items()
    }
    if set(parsed) != set(allowed):
        raise ValueError(f"{path} keys must exactly match the corresponding allowed values")
    if any(weight <= 0 for weight in parsed.values()):
        raise ValueError(f"{path} values must be positive")
    return MappingProxyType(parsed)


@dataclass(frozen=True, slots=True)
class DemonstrationPolicyConfig:
    """Allowlist and sample-weight policy for BC demonstrations."""

    schema_version: str
    allowed_origins: tuple[DemonstrationOrigin, ...]
    allowed_outcomes: tuple[DemonstrationOutcome, ...]
    origin_weights: Mapping[DemonstrationOrigin, float]
    outcome_weights: Mapping[DemonstrationOutcome, float]
    reject_unknown: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DemonstrationPolicyConfig:
        _reject_unknown(
            value,
            allowed=frozenset(
                {
                    "schema_version",
                    "allowed_origins",
                    "allowed_outcomes",
                    "origin_weights",
                    "outcome_weights",
                    "reject_unknown",
                }
            ),
            path="demonstration policy",
        )
        schema_version = value.get("schema_version")
        if schema_version != DEMONSTRATION_POLICY_SCHEMA_VERSION:
            raise ValueError(
                "demonstration policy.schema_version must be "
                f"{DEMONSTRATION_POLICY_SCHEMA_VERSION!r}; received {schema_version!r}"
            )
        origins = tuple(
            _enum(item, DemonstrationOrigin, path="allowed_origins[]")
            for item in _sequence(value.get("allowed_origins"), path="allowed_origins")
        )
        outcomes = tuple(
            _enum(item, DemonstrationOutcome, path="allowed_outcomes[]")
            for item in _sequence(value.get("allowed_outcomes"), path="allowed_outcomes")
        )
        if not origins or not outcomes:
            raise ValueError("allowed_origins and allowed_outcomes cannot be empty")
        if len(set(origins)) != len(origins) or len(set(outcomes)) != len(outcomes):
            raise ValueError("allowed demonstration values contain duplicates")
        reject_unknown = _boolean(value.get("reject_unknown"), path="reject_unknown")
        if reject_unknown and (
            DemonstrationOrigin.UNKNOWN in origins or DemonstrationOutcome.UNKNOWN in outcomes
        ):
            raise ValueError("reject_unknown cannot allow unknown origin or outcome")
        return cls(
            schema_version=schema_version,
            allowed_origins=origins,
            allowed_outcomes=outcomes,
            origin_weights=_weights(
                value.get("origin_weights"),
                allowed=origins,
                enum_type=DemonstrationOrigin,
                path="origin_weights",
            ),
            outcome_weights=_weights(
                value.get("outcome_weights"),
                allowed=outcomes,
                enum_type=DemonstrationOutcome,
                path="outcome_weights",
            ),
            reject_unknown=reject_unknown,
        )


@dataclass(frozen=True, slots=True)
class DemonstrationProvenance:
    """Immutable provenance carried by every BC sample or trajectory."""

    origin: DemonstrationOrigin
    outcome: DemonstrationOutcome
    policy_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "origin", _enum(self.origin, DemonstrationOrigin, path="demonstration origin")
        )
        object.__setattr__(
            self,
            "outcome",
            _enum(self.outcome, DemonstrationOutcome, path="demonstration outcome"),
        )
        if self.origin is DemonstrationOrigin.POLICY:
            if self.policy_id is None:
                raise ContractViolation("policy demonstrations require a policy_id")
            _identifier(self.policy_id, path="policy_id")
        elif self.policy_id is not None:
            raise ContractViolation("policy_id is forbidden unless demonstration origin is policy")


@dataclass(frozen=True, slots=True)
class DemonstrationDecision:
    """Accepted BC sample with a deterministic configured weight."""

    sample_weight: float


class DemonstrationGate:
    """Reject untrusted or self-generated examples before BC ingestion."""

    def __init__(self, config: DemonstrationPolicyConfig) -> None:
        self._config = config

    def validate(self, provenance: DemonstrationProvenance) -> DemonstrationDecision:
        if not isinstance(provenance, DemonstrationProvenance):
            raise TypeError("provenance must be a DemonstrationProvenance")
        if self._config.reject_unknown and (
            provenance.origin is DemonstrationOrigin.UNKNOWN
            or provenance.outcome is DemonstrationOutcome.UNKNOWN
        ):
            raise ContractViolation("unknown demonstration provenance is rejected")
        if provenance.origin not in self._config.allowed_origins:
            raise ContractViolation(
                f"demonstration origin {provenance.origin.value} is not allowed"
            )
        if provenance.outcome not in self._config.allowed_outcomes:
            raise ContractViolation(
                f"demonstration outcome {provenance.outcome.value} is not allowed"
            )
        return DemonstrationDecision(
            sample_weight=(
                self._config.origin_weights[provenance.origin]
                * self._config.outcome_weights[provenance.outcome]
            )
        )

    def validate_transition(self, transition: Any) -> DemonstrationDecision:
        """Validate provenance carried by a serialized/runtime transition."""
        provenance = getattr(transition, "provenance", None)
        if not isinstance(provenance, Mapping):
            raise ContractViolation("transition is missing demonstration provenance")
        try:
            parsed = DemonstrationProvenance(
                origin=DemonstrationOrigin(provenance["origin"]),
                outcome=DemonstrationOutcome(provenance["outcome"]),
                policy_id=provenance.get("policy_id"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractViolation("invalid transition demonstration provenance") from error
        return self.validate(parsed)


def load_reward_safety_config(path: str | Path) -> RewardSafetyConfig:
    """Load a strict reward-safety JSON document."""

    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    return RewardSafetyConfig.from_mapping(_mapping(value, path="reward safety"))


def load_demonstration_policy_config(path: str | Path) -> DemonstrationPolicyConfig:
    """Load a strict BC demonstration-policy JSON document."""

    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    return DemonstrationPolicyConfig.from_mapping(_mapping(value, path="demonstration policy"))


__all__ = [
    "DEMONSTRATION_POLICY_SCHEMA_VERSION",
    "REWARD_SAFETY_SCHEMA_VERSION",
    "DemonstrationDecision",
    "DemonstrationGate",
    "DemonstrationOrigin",
    "DemonstrationOutcome",
    "DemonstrationPolicyConfig",
    "DemonstrationProvenance",
    "EpisodeRewardGuard",
    "GuardedRewardResult",
    "RewardSafetyConfig",
    "load_demonstration_policy_config",
    "load_reward_safety_config",
]
