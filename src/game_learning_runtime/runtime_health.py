"""Stable runtime identity and read-only health contracts.

These values let an owning launcher verify that a session is still connected
to the executable it selected before starting or draining work.  They contain
no paths, process identifiers, hostnames, or credentials; executable
ownership and replacement remain launcher responsibilities.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

RUNTIME_HEALTH_SCHEMA_VERSION = "glr.runtime-health.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a bounded runtime identifier")
    return value


def _non_negative(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _positive(value: object, *, path: str) -> int:
    result = _non_negative(value, path=path)
    if result == 0:
        raise ValueError(f"{path} must be positive")
    return result


def _fields(value: Mapping[str, Any], expected: frozenset[str], *, path: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(f"{path} has missing={missing} and unexpected={unexpected} fields")


class RuntimeHealthStatus(str, Enum):
    """Read-only lifecycle state reported by a runtime provider."""

    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Stable public identity for one runtime family and executable version."""

    runtime_id: str
    runtime_version: str

    def __post_init__(self) -> None:
        _identifier(self.runtime_id, path="runtime identity.runtime_id")
        _identifier(self.runtime_version, path="runtime identity.runtime_version")

    def to_mapping(self) -> dict[str, str]:
        return {"runtime_id": self.runtime_id, "runtime_version": self.runtime_version}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RuntimeIdentity:
        _fields(value, frozenset({"runtime_id", "runtime_version"}), path="runtime identity")
        return cls(
            runtime_id=_identifier(value["runtime_id"], path="runtime identity.runtime_id"),
            runtime_version=_identifier(
                value["runtime_version"], path="runtime identity.runtime_version"
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeLease:
    """Optional launcher/provider lease metadata observed with a health read."""

    lease_id: str
    owner_id: str
    expires_at_ns: int

    def __post_init__(self) -> None:
        _identifier(self.lease_id, path="runtime lease.lease_id")
        _identifier(self.owner_id, path="runtime lease.owner_id")
        _positive(self.expires_at_ns, path="runtime lease.expires_at_ns")

    def to_mapping(self) -> dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "owner_id": self.owner_id,
            "expires_at_ns": self.expires_at_ns,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RuntimeLease:
        _fields(
            value,
            frozenset({"lease_id", "owner_id", "expires_at_ns"}),
            path="runtime lease",
        )
        return cls(
            lease_id=_identifier(value["lease_id"], path="runtime lease.lease_id"),
            owner_id=_identifier(value["owner_id"], path="runtime lease.owner_id"),
            expires_at_ns=_positive(value["expires_at_ns"], path="runtime lease.expires_at_ns"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeHealth:
    """Bounded, read-only health snapshot for launcher coordination."""

    identity: RuntimeIdentity
    status: RuntimeHealthStatus
    observed_at_ns: int
    accepting_new_sessions: bool
    active_sessions: int
    lease: RuntimeLease | None = None
    schema_version: str = RUNTIME_HEALTH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_HEALTH_SCHEMA_VERSION:
            raise ValueError(f"unsupported runtime health schema: {self.schema_version!r}")
        if not isinstance(self.identity, RuntimeIdentity):
            raise TypeError("runtime health identity must be a RuntimeIdentity")
        object.__setattr__(self, "status", RuntimeHealthStatus(self.status))
        _non_negative(self.observed_at_ns, path="runtime health.observed_at_ns")
        if not isinstance(self.accepting_new_sessions, bool):
            raise TypeError("runtime health.accepting_new_sessions must be a bool")
        _non_negative(self.active_sessions, path="runtime health.active_sessions")
        if self.lease is not None:
            if not isinstance(self.lease, RuntimeLease):
                raise TypeError("runtime health.lease must be a RuntimeLease or None")
            if self.lease.expires_at_ns <= self.observed_at_ns:
                raise ValueError("runtime health lease must expire after observed_at_ns")

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "identity": self.identity.to_mapping(),
            "status": self.status.value,
            "observed_at_ns": self.observed_at_ns,
            "accepting_new_sessions": self.accepting_new_sessions,
            "active_sessions": self.active_sessions,
        }
        if self.lease is not None:
            value["lease"] = self.lease.to_mapping()
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RuntimeHealth:
        required = frozenset(
            {
                "schema_version",
                "identity",
                "status",
                "observed_at_ns",
                "accepting_new_sessions",
                "active_sessions",
            }
        )
        missing = sorted(required - set(value))
        unexpected = sorted(set(value) - (required | {"lease"}))
        if missing or unexpected:
            raise ValueError(
                f"runtime health has missing={missing} and unexpected={unexpected} fields"
            )
        identity = value["identity"]
        lease = value.get("lease")
        if not isinstance(identity, Mapping):
            raise TypeError("runtime health.identity must be an object")
        if lease is not None and not isinstance(lease, Mapping):
            raise TypeError("runtime health.lease must be an object or null")
        return cls(
            schema_version=value["schema_version"],
            identity=RuntimeIdentity.from_mapping(identity),
            status=RuntimeHealthStatus(value["status"]),
            observed_at_ns=_non_negative(
                value["observed_at_ns"], path="runtime health.observed_at_ns"
            ),
            accepting_new_sessions=value["accepting_new_sessions"],
            active_sessions=_non_negative(
                value["active_sessions"], path="runtime health.active_sessions"
            ),
            lease=None if lease is None else RuntimeLease.from_mapping(lease),
        )


__all__ = [
    "RUNTIME_HEALTH_SCHEMA_VERSION",
    "RuntimeHealth",
    "RuntimeHealthStatus",
    "RuntimeIdentity",
    "RuntimeLease",
]
