"""Passive, version-bound knowledge and missing-capability evidence.

These records do not authorize actions, expand an adapter's capabilities, or
prove learning, reward, success or improvement. Producers and decision consumers
remain responsible for the artifacts identified by the supplied SHA-256 hashes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from game_learning_runtime.errors import ContractViolation

RULE_INDEX_BINDING_SCHEMA_VERSION = "glr.rule-index-binding.v1"
DECISION_CONSUMPTION_SCHEMA_VERSION = "glr.decision-consumption.v1"
CAPABILITY_GAP_SCHEMA_VERSION = "glr.capability-gap.v1"
_ID = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_SEMANTICS = 256


def _identifier(value: object, *, path: str) -> None:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{path} must be a bounded portable identifier")


def _digest(value: object, *, path: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")


def _version(value: object, *, path: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{path} must be bounded nonempty version text")


def _fields(value: Mapping[str, Any], *, expected: set[str], schema: str) -> None:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("evidence must be a mapping with string keys")
    if set(value) != expected | {"schema_version"}:
        raise ValueError("evidence has missing or unexpected fields")
    if value["schema_version"] != schema:
        raise ValueError("unsupported evidence schema_version")


def _semantics(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or len(value) > _MAX_SEMANTICS:
        raise ValueError("available_semantics must be a bounded tuple")
    for semantic in value:
        _identifier(semantic, path="available_semantics[]")
    if len(set(value)) != len(value):
        raise ValueError("available_semantics must be unique")
    return tuple(sorted(value))


@dataclass(frozen=True, slots=True)
class RuleIndexBinding:
    """One immutable index revision bound to its exact rules and environment.

    ``rules_sha256`` covers the adapter's complete rules scope, including any
    build, mode or rule modifiers relevant to that adapter. The runtime does
    not infer those game-specific fields from a version label.
    """

    environment_id: str
    protocol_version: str
    rules_version: str
    rules_sha256: str
    index_sha256: str
    index_rules_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.environment_id, path="environment_id")
        _version(self.protocol_version, path="protocol_version")
        _version(self.rules_version, path="rules_version")
        for name in ("rules_sha256", "index_sha256", "index_rules_sha256"):
            _digest(getattr(self, name), path=name)
        if self.index_rules_sha256 != self.rules_sha256:
            raise ContractViolation("index was built for different rules")

    def assert_current(
        self,
        *,
        environment_id: str,
        protocol_version: str,
        rules_version: str,
        rules_sha256: str,
    ) -> None:
        """Refuse this binding before retrieval or direct-table decision use."""
        _identifier(environment_id, path="environment_id")
        _version(protocol_version, path="protocol_version")
        _version(rules_version, path="rules_version")
        _digest(rules_sha256, path="rules_sha256")
        for name, current in (
            ("environment_id", environment_id),
            ("protocol_version", protocol_version),
            ("rules_version", rules_version),
            ("rules_sha256", rules_sha256),
        ):
            if getattr(self, name) != current:
                raise ContractViolation(f"rule index binding has stale or foreign {name}")

    def to_mapping(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible record."""
        return {
            "schema_version": RULE_INDEX_BINDING_SCHEMA_VERSION,
            "environment_id": self.environment_id,
            "protocol_version": self.protocol_version,
            "rules_version": self.rules_version,
            "rules_sha256": self.rules_sha256,
            "index_sha256": self.index_sha256,
            "index_rules_sha256": self.index_rules_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RuleIndexBinding:
        """Parse the exact v1 schema; unknown authority fields are rejected."""
        _fields(
            value,
            expected={
                "environment_id",
                "protocol_version",
                "rules_version",
                "rules_sha256",
                "index_sha256",
                "index_rules_sha256",
            },
            schema=RULE_INDEX_BINDING_SCHEMA_VERSION,
        )
        return cls(**{key: item for key, item in value.items() if key != "schema_version"})


class ConsumptionState(str, Enum):
    """The consumer's evidence level, independent of measured task outcome."""

    RETRIEVED = "retrieved"
    USED = "used"


@dataclass(frozen=True, slots=True)
class DecisionConsumptionReceipt:
    """A finding's provenance and actual decision consumer, not a success claim.

    ``RETRIEVED`` means a finding was supplied to the named consumer; it does
    not prove use. ``USED`` requires an artifact from that consumer linking the
    finding to this decision. Direct-table consumption uses ``USED`` as well;
    it does not imply that a separate retrieval occurred.
    """

    decision_id: str
    finding_id: str
    finding_sha256: str
    consumer_id: str
    binding: RuleIndexBinding
    state: ConsumptionState
    use_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("decision_id", "finding_id", "consumer_id"):
            _identifier(getattr(self, name), path=name)
        _digest(self.finding_sha256, path="finding_sha256")
        if not isinstance(self.binding, RuleIndexBinding):
            raise TypeError("binding must be a RuleIndexBinding")
        object.__setattr__(self, "state", ConsumptionState(self.state))
        if self.state is ConsumptionState.USED:
            _digest(self.use_evidence_sha256, path="use_evidence_sha256")
        elif self.use_evidence_sha256 is not None:
            raise ValueError("retrieved-only receipt cannot claim use evidence")

    @property
    def is_used(self) -> bool:
        """Whether consumer-linked use evidence is declared, not whether it helped."""
        return self.state is ConsumptionState.USED

    def assert_for_decision(
        self,
        *,
        decision_id: str,
        finding_id: str,
        finding_sha256: str,
        consumer_id: str,
        binding: RuleIndexBinding,
    ) -> None:
        """Reject receipts copied from another decision, finding, consumer or index."""
        _identifier(decision_id, path="decision_id")
        _identifier(finding_id, path="finding_id")
        _digest(finding_sha256, path="finding_sha256")
        _identifier(consumer_id, path="consumer_id")
        if not isinstance(binding, RuleIndexBinding):
            raise TypeError("binding must be a RuleIndexBinding")
        if self.decision_id != decision_id or self.consumer_id != consumer_id:
            raise ContractViolation("consumption receipt belongs to another decision or consumer")
        if self.finding_id != finding_id or self.finding_sha256 != finding_sha256:
            raise ContractViolation("consumption receipt belongs to another finding revision")
        if self.binding != binding:
            raise ContractViolation("consumption receipt belongs to another rule index binding")

    def to_mapping(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible receipt."""
        return {
            "schema_version": DECISION_CONSUMPTION_SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "finding_id": self.finding_id,
            "finding_sha256": self.finding_sha256,
            "consumer_id": self.consumer_id,
            "binding": RuleIndexBinding.to_mapping(self.binding),
            "state": self.state.value,
            "use_evidence_sha256": self.use_evidence_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DecisionConsumptionReceipt:
        """Parse the exact v1 schema and revalidate its nested rules binding."""
        _fields(
            value,
            expected={
                "decision_id",
                "finding_id",
                "finding_sha256",
                "consumer_id",
                "binding",
                "state",
                "use_evidence_sha256",
            },
            schema=DECISION_CONSUMPTION_SCHEMA_VERSION,
        )
        fields = {key: item for key, item in value.items() if key != "schema_version"}
        fields["binding"] = RuleIndexBinding.from_mapping(fields["binding"])
        return cls(**fields)


class CapabilityGapKind(str, Enum):
    """Which adapter or consumer surface needs owner review."""

    MISSING_ACTION = "missing_action"
    MISSING_OBSERVATION = "missing_observation"
    MISSING_CONSUMER = "missing_consumer"


@dataclass(frozen=True, slots=True)
class CapabilityGap:
    """Passive feedback about a missing semantic in a frozen evidence surface.

    No method grants permission, constructs an action or executes a producer.
    An owner can use this evidence to propose an isolated interface change.
    """

    decision_id: str
    kind: CapabilityGapKind
    required_semantic: str
    available_semantics: tuple[str, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.decision_id, path="decision_id")
        _identifier(self.required_semantic, path="required_semantic")
        _digest(self.evidence_sha256, path="evidence_sha256")
        object.__setattr__(self, "kind", CapabilityGapKind(self.kind))
        object.__setattr__(self, "available_semantics", _semantics(self.available_semantics))
        self.verify_missing()

    def verify_missing(self, available_semantics: tuple[str, ...] | None = None) -> None:
        """Reject a stale gap if its required semantic exists in a supplied surface."""
        surface = (
            self.available_semantics
            if available_semantics is None
            else _semantics(available_semantics)
        )
        if self.required_semantic in surface:
            raise ContractViolation("required semantic is present; capability gap is invalid")

    def to_mapping(self) -> dict[str, Any]:
        """Return versioned passive data, with no action or authorization fields."""
        return {
            "schema_version": CAPABILITY_GAP_SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "kind": self.kind.value,
            "required_semantic": self.required_semantic,
            "available_semantics": list(self.available_semantics),
            "evidence_sha256": self.evidence_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CapabilityGap:
        """Parse the exact v1 schema, refusing extra executable fields."""
        _fields(
            value,
            expected={
                "decision_id",
                "kind",
                "required_semantic",
                "available_semantics",
                "evidence_sha256",
            },
            schema=CAPABILITY_GAP_SCHEMA_VERSION,
        )
        if not isinstance(value["available_semantics"], list):
            raise TypeError("available_semantics must be an array")
        fields = {key: item for key, item in value.items() if key != "schema_version"}
        fields["available_semantics"] = tuple(fields["available_semantics"])
        return cls(**fields)


__all__ = [
    "CAPABILITY_GAP_SCHEMA_VERSION",
    "DECISION_CONSUMPTION_SCHEMA_VERSION",
    "RULE_INDEX_BINDING_SCHEMA_VERSION",
    "CapabilityGap",
    "CapabilityGapKind",
    "ConsumptionState",
    "DecisionConsumptionReceipt",
    "RuleIndexBinding",
]
