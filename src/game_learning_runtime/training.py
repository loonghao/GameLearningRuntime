"""Versioned, data-only configuration for knowledge sources and rewards.

The configuration deliberately contains no expression language, import path,
endpoint, credential, or executable callback.  Adapters own data acquisition
and emit named scalar signals; GLR validates and composes those signals.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from game_learning_runtime.errors import ContractViolation

TRAINING_SCHEMA_VERSION = "glr.training.v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")


class KnowledgeAuthority(str, Enum):
    """Trust level assigned to a configured knowledge source."""

    ADVISORY = "advisory"
    AUTHORITATIVE = "authoritative"


_AUTHORITY_RANK = {
    KnowledgeAuthority.ADVISORY: 0,
    KnowledgeAuthority.AUTHORITATIVE: 1,
}


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
        raise ValueError(
            f"{path} must match {_IDENTIFIER.pattern!r} and contain no local path or endpoint"
        )
    return value


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _optional_number(value: object, *, path: str) -> float | None:
    return None if value is None else _number(value, path=path)


def _authority(value: object, *, path: str) -> KnowledgeAuthority:
    try:
        return KnowledgeAuthority(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(authority.value for authority in KnowledgeAuthority)
        raise ValueError(f"{path} must be one of: {choices}") from error


def _validate_bounds(minimum: float | None, maximum: float | None, *, path: str) -> None:
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{path}.minimum cannot exceed {path}.maximum")


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    """How a collector starts and terminates interaction."""

    start_mode: Literal["reset", "attach"] = "reset"
    stop_on_done: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LifecycleConfig:
        _reject_unknown(
            value,
            allowed=frozenset({"start_mode", "stop_on_done"}),
            path="lifecycle",
        )
        start_mode = value.get("start_mode", "reset")
        if start_mode not in {"reset", "attach"}:
            raise ValueError("lifecycle.start_mode must be reset or attach")
        return cls(
            start_mode=start_mode,
            stop_on_done=_boolean(value.get("stop_on_done", True), path="lifecycle.stop_on_done"),
        )


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    """Capabilities a remote bridge must prove before collection starts."""

    required_capabilities: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BridgeConfig:
        _reject_unknown(
            value,
            allowed=frozenset({"required_capabilities"}),
            path="bridge",
        )
        capabilities = _sequence(
            value.get("required_capabilities", ()),
            path="bridge.required_capabilities",
        )
        normalized = frozenset(
            _identifier(item, path="bridge.required_capabilities[]") for item in capabilities
        )
        if len(normalized) != len(capabilities):
            raise ValueError("bridge.required_capabilities contains duplicates")
        return cls(required_capabilities=normalized)


@dataclass(frozen=True, slots=True)
class KnowledgeSourceSpec:
    """Bounded knowledge input declared without connection details."""

    source_id: str
    authority: KnowledgeAuthority
    required: bool = False
    max_age_seconds: float | None = None
    max_payload_bytes: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> KnowledgeSourceSpec:
        _reject_unknown(
            value,
            allowed=frozenset(
                {
                    "id",
                    "authority",
                    "required",
                    "max_age_seconds",
                    "max_payload_bytes",
                }
            ),
            path="knowledge_sources[]",
        )
        max_age = _optional_number(
            value.get("max_age_seconds"),
            path="knowledge_sources[].max_age_seconds",
        )
        if max_age is not None and max_age < 0:
            raise ValueError("knowledge_sources[].max_age_seconds cannot be negative")
        raw_payload = value.get("max_payload_bytes")
        if raw_payload is not None and (
            not isinstance(raw_payload, int) or isinstance(raw_payload, bool) or raw_payload <= 0
        ):
            raise ValueError("knowledge_sources[].max_payload_bytes must be a positive integer")
        return cls(
            source_id=_identifier(value.get("id"), path="knowledge_sources[].id"),
            authority=_authority(value.get("authority"), path="knowledge_sources[].authority"),
            required=_boolean(value.get("required", False), path="knowledge_sources[].required"),
            max_age_seconds=max_age,
            max_payload_bytes=raw_payload,
        )


@dataclass(frozen=True, slots=True)
class RewardTermSpec:
    """One named scalar reward contribution produced by adapter-owned code."""

    name: str
    source: str
    weight: float = 1.0
    minimum: float | None = None
    maximum: float | None = None
    required: bool = True
    minimum_authority: KnowledgeAuthority = KnowledgeAuthority.AUTHORITATIVE

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RewardTermSpec:
        _reject_unknown(
            value,
            allowed=frozenset(
                {
                    "name",
                    "source",
                    "weight",
                    "minimum",
                    "maximum",
                    "required",
                    "minimum_authority",
                }
            ),
            path="reward.terms[]",
        )
        minimum = _optional_number(value.get("minimum"), path="reward.terms[].minimum")
        maximum = _optional_number(value.get("maximum"), path="reward.terms[].maximum")
        _validate_bounds(minimum, maximum, path="reward.terms[]")
        return cls(
            name=_identifier(value.get("name"), path="reward.terms[].name"),
            source=_identifier(value.get("source"), path="reward.terms[].source"),
            weight=_number(value.get("weight", 1.0), path="reward.terms[].weight"),
            minimum=minimum,
            maximum=maximum,
            required=_boolean(value.get("required", True), path="reward.terms[].required"),
            minimum_authority=_authority(
                value.get("minimum_authority", "authoritative"),
                path="reward.terms[].minimum_authority",
            ),
        )


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Safe weighted composition policy for adapter-provided scalar signals."""

    terms: tuple[RewardTermSpec, ...]
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RewardConfig:
        _reject_unknown(
            value,
            allowed=frozenset({"terms", "minimum", "maximum"}),
            path="reward",
        )
        terms = tuple(
            RewardTermSpec.from_mapping(_mapping(item, path="reward.terms[]"))
            for item in _sequence(value.get("terms"), path="reward.terms")
        )
        if not terms:
            raise ValueError("reward.terms requires at least one term")
        names = [term.name for term in terms]
        if len(set(names)) != len(names):
            raise ValueError("reward.terms contains duplicate names")
        minimum = _optional_number(value.get("minimum"), path="reward.minimum")
        maximum = _optional_number(value.get("maximum"), path="reward.maximum")
        _validate_bounds(minimum, maximum, path="reward")
        return cls(terms=terms, minimum=minimum, maximum=maximum)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Top-level versioned configuration shared by adapters and collectors."""

    schema_version: str
    lifecycle: LifecycleConfig
    bridge: BridgeConfig
    knowledge_sources: tuple[KnowledgeSourceSpec, ...]
    reward: RewardConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingConfig:
        _reject_unknown(
            value,
            allowed=frozenset(
                {"schema_version", "lifecycle", "bridge", "knowledge_sources", "reward"}
            ),
            path="training",
        )
        schema_version = value.get("schema_version")
        if schema_version != TRAINING_SCHEMA_VERSION:
            raise ValueError(
                "training.schema_version must be "
                f"{TRAINING_SCHEMA_VERSION!r}; received {schema_version!r}"
            )
        knowledge_sources = tuple(
            KnowledgeSourceSpec.from_mapping(_mapping(item, path="knowledge_sources[]"))
            for item in _sequence(value.get("knowledge_sources"), path="knowledge_sources")
        )
        source_ids = [source.source_id for source in knowledge_sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("knowledge_sources contains duplicate ids")
        config = cls(
            schema_version=schema_version,
            lifecycle=LifecycleConfig.from_mapping(
                _mapping(value.get("lifecycle", {}), path="lifecycle")
            ),
            bridge=BridgeConfig.from_mapping(_mapping(value.get("bridge", {}), path="bridge")),
            knowledge_sources=knowledge_sources,
            reward=RewardConfig.from_mapping(_mapping(value.get("reward"), path="reward")),
        )
        config._validate_reward_authority()
        return config

    @property
    def knowledge_by_id(self) -> Mapping[str, KnowledgeSourceSpec]:
        """Return an immutable lookup without exposing mutable input data."""

        return MappingProxyType({source.source_id: source for source in self.knowledge_sources})

    def _validate_reward_authority(self) -> None:
        sources = self.knowledge_by_id
        for term in self.reward.terms:
            source = sources.get(term.source)
            if source is None:
                raise ContractViolation(
                    f"reward term {term.name!r} references unknown source {term.source!r}"
                )
            if _AUTHORITY_RANK[source.authority] < _AUTHORITY_RANK[term.minimum_authority]:
                raise ContractViolation(
                    f"reward term {term.name!r} requires {term.minimum_authority.value} "
                    f"source, but {term.source!r} is {source.authority.value}"
                )


@dataclass(frozen=True, slots=True)
class RewardSignal:
    """One scalar value emitted by adapter-owned, reviewed code."""

    name: str
    source: str
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, path="reward signal name"))
        object.__setattr__(self, "source", _identifier(self.source, path="reward signal source"))
        object.__setattr__(self, "value", _number(self.value, path="reward signal value"))


@dataclass(frozen=True, slots=True)
class RewardResult:
    """Immutable composed reward and its auditable contributions."""

    total: float
    contributions: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "total", _number(self.total, path="reward total"))
        object.__setattr__(self, "contributions", MappingProxyType(dict(self.contributions)))


class RewardComposer:
    """Validate and compose signals using a declarative ``TrainingConfig``."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._terms = {term.name: term for term in config.reward.terms}

    def compose(self, signals: Iterable[RewardSignal]) -> RewardResult:
        received: dict[str, RewardSignal] = {}
        for signal in signals:
            if not isinstance(signal, RewardSignal):
                raise TypeError("reward signals must be RewardSignal instances")
            if signal.name in received:
                raise ContractViolation(f"duplicate reward signal {signal.name!r}")
            term = self._terms.get(signal.name)
            if term is None:
                raise ContractViolation(f"unknown reward signal {signal.name!r}")
            if signal.source != term.source:
                raise ContractViolation(
                    f"reward signal {signal.name!r} expected source {term.source}, "
                    f"received {signal.source}"
                )
            received[signal.name] = signal

        missing = sorted(
            term.name
            for term in self._config.reward.terms
            if term.required and term.name not in received
        )
        if missing:
            raise ContractViolation(f"missing required reward signals: {missing}")

        contributions: dict[str, float] = {}
        for term in self._config.reward.terms:
            received_signal = received.get(term.name)
            if received_signal is None:
                continue
            raw = _clip(received_signal.value, term.minimum, term.maximum)
            contributions[term.name] = raw * term.weight

        total = _clip(
            math.fsum(contributions.values()),
            self._config.reward.minimum,
            self._config.reward.maximum,
        )
        return RewardResult(total=total, contributions=contributions)


def _clip(value: float, minimum: float | None, maximum: float | None) -> float:
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a UTF-8 JSON configuration using only the Python standard library."""

    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    return TrainingConfig.from_mapping(_mapping(value, path="training"))


__all__ = [
    "TRAINING_SCHEMA_VERSION",
    "BridgeConfig",
    "KnowledgeAuthority",
    "KnowledgeSourceSpec",
    "LifecycleConfig",
    "RewardComposer",
    "RewardConfig",
    "RewardResult",
    "RewardSignal",
    "RewardTermSpec",
    "TrainingConfig",
    "load_training_config",
]
