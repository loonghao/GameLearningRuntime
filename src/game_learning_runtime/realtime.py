"""Learner-neutral realtime timing and input-lease contracts.

The objects in this module describe bounded provider behavior.  They do not
provide a generic automation API and cannot grant action authority by
themselves; a provider remains responsible for target identity and
authoritative post-action observation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from time import time_ns
from typing import Any

REALTIME_CONTROL_SCHEMA_VERSION = "glr.realtime-control.v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]*$")


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase bounded identifier")
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


class RealtimeActionStatus(str, Enum):
    """Typed result of a realtime action dispatch attempt."""

    CONSUMED = "consumed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class InputLeaseOperation(str, Enum):
    """Lifecycle operation for one target-bound input lease."""

    ACQUIRE = "acquire"
    RENEW = "renew"
    RELEASE = "release"
    PREEMPT = "preempt"


class InputLeaseStatus(str, Enum):
    """Typed result of a lease operation."""

    ACQUIRED = "acquired"
    RENEWED = "renewed"
    RELEASED = "released"
    PREEMPTED = "preempted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RealtimeTimingContract:
    """Descriptor-level timing bounds, expressed in nanoseconds."""

    minimum_hold_ns: int
    maximum_hold_ns: int
    settle_deadline_ns: int
    simulation_quantum_ns: int
    clock_source: str = "monotonic"
    schema_version: str = REALTIME_CONTROL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REALTIME_CONTROL_SCHEMA_VERSION:
            raise ValueError(f"unsupported realtime control schema: {self.schema_version!r}")
        for name in (
            "minimum_hold_ns",
            "maximum_hold_ns",
            "settle_deadline_ns",
            "simulation_quantum_ns",
        ):
            _positive(getattr(self, name), path=f"realtime timing.{name}")
        if self.minimum_hold_ns > self.maximum_hold_ns:
            raise ValueError("realtime timing minimum_hold_ns cannot exceed maximum_hold_ns")
        if self.maximum_hold_ns > self.settle_deadline_ns:
            raise ValueError("realtime timing maximum_hold_ns cannot exceed settle_deadline_ns")
        if not isinstance(self.clock_source, str) or not self.clock_source:
            raise ValueError("realtime timing clock_source cannot be empty")

    def validate_step(
        self, *, deadline_ns: int, quantum_ns: int, hold_ns: int | None = None
    ) -> None:
        """Validate a bounded step request against this descriptor."""

        _positive(deadline_ns, path="step.deadline_ns")
        _positive(quantum_ns, path="step.quantum_ns")
        if deadline_ns > self.settle_deadline_ns:
            raise ValueError("step.deadline_ns exceeds realtime timing settle_deadline_ns")
        if quantum_ns > self.simulation_quantum_ns:
            raise ValueError("step.quantum_ns exceeds realtime timing simulation_quantum_ns")
        if hold_ns is not None:
            _positive(hold_ns, path="step.hold_ns")
            if hold_ns < self.minimum_hold_ns or hold_ns > self.maximum_hold_ns:
                raise ValueError("step.hold_ns is outside realtime timing hold bounds")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "minimum_hold_ns": self.minimum_hold_ns,
            "maximum_hold_ns": self.maximum_hold_ns,
            "settle_deadline_ns": self.settle_deadline_ns,
            "simulation_quantum_ns": self.simulation_quantum_ns,
            "clock_source": self.clock_source,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RealtimeTimingContract:
        _fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "minimum_hold_ns",
                    "maximum_hold_ns",
                    "settle_deadline_ns",
                    "simulation_quantum_ns",
                    "clock_source",
                }
            ),
            path="realtime timing",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class RealtimeStepTiming:
    """Per-action timing request carried by a fenced step."""

    deadline_ns: int
    quantum_ns: int
    hold_ns: int | None = None

    def __post_init__(self) -> None:
        _positive(self.deadline_ns, path="step.deadline_ns")
        _positive(self.quantum_ns, path="step.quantum_ns")
        if self.hold_ns is not None:
            _positive(self.hold_ns, path="step.hold_ns")

    def validate_against(self, contract: RealtimeTimingContract) -> None:
        contract.validate_step(
            deadline_ns=self.deadline_ns,
            quantum_ns=self.quantum_ns,
            hold_ns=self.hold_ns,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "deadline_ns": self.deadline_ns,
            "quantum_ns": self.quantum_ns,
            "hold_ns": self.hold_ns,
        }


@dataclass(frozen=True, slots=True)
class InputLeaseToken:
    """Opaque target/session binding required for a mutating input step."""

    lease_id: str
    session_id: str
    target_id: str

    def __post_init__(self) -> None:
        _identifier(self.lease_id, path="lease.lease_id")
        _identifier(self.session_id, path="lease.session_id")
        _identifier(self.target_id, path="lease.target_id")

    def to_mapping(self) -> dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "session_id": self.session_id,
            "target_id": self.target_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> InputLeaseToken:
        _fields(value, frozenset({"lease_id", "session_id", "target_id"}), path="lease token")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class InputLeaseRequest:
    """Acquire, renew, release, or preempt one target-bound lease."""

    operation: InputLeaseOperation
    session_id: str
    target_id: str
    lease_id: str | None = None
    expires_at_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", InputLeaseOperation(self.operation))
        _identifier(self.session_id, path="lease request.session_id")
        _identifier(self.target_id, path="lease request.target_id")
        if self.lease_id is not None:
            _identifier(self.lease_id, path="lease request.lease_id")
        if self.expires_at_ns is not None:
            _positive(self.expires_at_ns, path="lease request.expires_at_ns")
        if self.operation is InputLeaseOperation.ACQUIRE and self.lease_id is not None:
            raise ValueError("acquire must not provide an existing lease_id")
        if self.operation is not InputLeaseOperation.ACQUIRE and self.lease_id is None:
            raise ValueError("lease_id is required for an existing lease operation")


@dataclass(frozen=True, slots=True)
class InputLeaseReceipt:
    """Result of a lease operation, including the exact target/session binding."""

    status: InputLeaseStatus
    token: InputLeaseToken | None
    observed_at_ns: int
    expires_at_ns: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", InputLeaseStatus(self.status))
        if self.token is not None and not isinstance(self.token, InputLeaseToken):
            raise TypeError("lease receipt token must be an InputLeaseToken or None")
        _non_negative(self.observed_at_ns, path="lease receipt.observed_at_ns")
        if self.expires_at_ns is not None:
            _positive(self.expires_at_ns, path="lease receipt.expires_at_ns")
            if self.expires_at_ns <= self.observed_at_ns:
                raise ValueError("lease receipt expires_at_ns must be after observed_at_ns")
        if self.reason is not None and (not self.reason or len(self.reason) > 256):
            raise ValueError("lease receipt reason must contain 1-256 characters")


@dataclass(frozen=True, slots=True)
class RealtimeActionReceipt:
    """Typed timing result linked to one action and its deadline."""

    action_id: str
    status: RealtimeActionStatus
    deadline_ns: int
    quantum_ns: int
    issued_at_ns: int
    consumed_at_ns: int | None = None
    settled_at_ns: int | None = None
    cancellation_token: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.action_id, path="realtime receipt.action_id")
        object.__setattr__(self, "status", RealtimeActionStatus(self.status))
        _positive(self.deadline_ns, path="realtime receipt.deadline_ns")
        _positive(self.quantum_ns, path="realtime receipt.quantum_ns")
        _non_negative(self.issued_at_ns, path="realtime receipt.issued_at_ns")
        for name in ("consumed_at_ns", "settled_at_ns"):
            value = getattr(self, name)
            if value is not None:
                _non_negative(value, path=f"realtime receipt.{name}")
                if value < self.issued_at_ns:
                    raise ValueError(f"realtime receipt.{name} cannot precede issued_at_ns")
        if (
            self.consumed_at_ns is not None
            and self.consumed_at_ns > self.issued_at_ns + self.deadline_ns
        ):
            raise ValueError("realtime receipt.consumed_at_ns exceeds deadline")
        if (
            self.settled_at_ns is not None
            and self.consumed_at_ns is not None
            and self.settled_at_ns < self.consumed_at_ns
        ):
            raise ValueError("realtime receipt.settled_at_ns cannot precede consumed_at_ns")
        if self.cancellation_token is not None:
            _identifier(self.cancellation_token, path="realtime receipt.cancellation_token")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "status": self.status.value,
            "deadline_ns": self.deadline_ns,
            "quantum_ns": self.quantum_ns,
            "issued_at_ns": self.issued_at_ns,
            "consumed_at_ns": self.consumed_at_ns,
            "settled_at_ns": self.settled_at_ns,
            "cancellation_token": self.cancellation_token,
        }


class InputLeaseBook:
    """Deterministic lease fencing primitive for synthetic providers and hosts."""

    def __init__(self, *, clock: Callable[[], int] = time_ns) -> None:
        self._clock = clock
        self._active: InputLeaseToken | None = None
        self._expires_at_ns: int | None = None

    @property
    def active(self) -> InputLeaseToken | None:
        return self._active

    def apply(self, request: InputLeaseRequest, *, now_ns: int | None = None) -> InputLeaseReceipt:
        now = self._clock() if now_ns is None else _non_negative(now_ns, path="now_ns")
        active = self._active
        if request.operation is InputLeaseOperation.ACQUIRE:
            if active is not None and self._expires_at_ns is not None and now < self._expires_at_ns:
                return InputLeaseReceipt(
                    InputLeaseStatus.REJECTED,
                    active,
                    now,
                    self._expires_at_ns,
                    "lease already held",
                )
            lease_id = request.session_id + ".lease"
            token = InputLeaseToken(lease_id, request.session_id, request.target_id)
            expires = request.expires_at_ns or now + 1
            self._active, self._expires_at_ns = token, expires
            return InputLeaseReceipt(InputLeaseStatus.ACQUIRED, token, now, expires)
        if active is None or self._expires_at_ns is None or now >= self._expires_at_ns:
            return InputLeaseReceipt(
                InputLeaseStatus.REJECTED,
                None,
                now,
                reason="lease is absent or expired",
            )
        if (
            request.lease_id != active.lease_id
            or request.session_id != active.session_id
            or request.target_id != active.target_id
        ):
            return InputLeaseReceipt(
                InputLeaseStatus.REJECTED,
                active,
                now,
                self._expires_at_ns,
                "lease identity mismatch",
            )
        if request.operation is InputLeaseOperation.RENEW:
            expires = request.expires_at_ns or self._expires_at_ns
            if expires <= now:
                return InputLeaseReceipt(
                    InputLeaseStatus.REJECTED,
                    active,
                    now,
                    self._expires_at_ns,
                    "renewal is expired",
                )
            self._expires_at_ns = expires
            return InputLeaseReceipt(InputLeaseStatus.RENEWED, active, now, expires)
        status = (
            InputLeaseStatus.PREEMPTED
            if request.operation is InputLeaseOperation.PREEMPT
            else InputLeaseStatus.RELEASED
        )
        self._active, self._expires_at_ns = None, None
        return InputLeaseReceipt(status, active, now)

    def authorize(self, token: InputLeaseToken, *, now_ns: int | None = None) -> bool:
        now = self._clock() if now_ns is None else _non_negative(now_ns, path="now_ns")
        return (
            self._active == token and self._expires_at_ns is not None and now < self._expires_at_ns
        )


__all__ = [
    "REALTIME_CONTROL_SCHEMA_VERSION",
    "InputLeaseBook",
    "InputLeaseOperation",
    "InputLeaseReceipt",
    "InputLeaseRequest",
    "InputLeaseStatus",
    "InputLeaseToken",
    "RealtimeActionReceipt",
    "RealtimeActionStatus",
    "RealtimeStepTiming",
    "RealtimeTimingContract",
]
