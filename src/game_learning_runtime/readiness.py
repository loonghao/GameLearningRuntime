"""Side-effect-free host readiness probes.

Readiness is deliberately separate from transport health and episode
progress. A probe may be run before attaching to an external target and,
optionally, while an episode is running. The probe never starts, stops, or
mutates the target; callers decide how to park and retry a not-ready host.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from time import monotonic_ns, sleep
from typing import Protocol

from game_learning_runtime.errors import GLRError

READINESS_SCHEMA_VERSION = "glr.environment-readiness.v1"


class ReadinessState(str, Enum):
    """The bounded result of a readiness probe."""

    READY = "ready"
    NOT_READY = "not_ready"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """One immutable probe result suitable for manifests and CLI JSON."""

    state: ReadinessState
    reason: str = ""
    checked_at_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.state, ReadinessState):
            try:
                object.__setattr__(self, "state", ReadinessState(self.state))
            except ValueError as error:
                raise ValueError(f"unsupported readiness state: {self.state!r}") from error
        if len(self.reason) > 256:
            raise ValueError("readiness reason cannot exceed 256 characters")
        if any(ord(character) < 0x20 for character in self.reason):
            raise ValueError("readiness reason cannot contain control characters")
        if self.checked_at_ns < 0:
            raise ValueError("checked_at_ns cannot be negative")
        if not self.checked_at_ns:
            object.__setattr__(self, "checked_at_ns", monotonic_ns())

    @property
    def ready(self) -> bool:
        return self.state is ReadinessState.READY

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": READINESS_SCHEMA_VERSION,
            "state": self.state.value,
            "reason": self.reason,
            "checked_at_ns": self.checked_at_ns,
        }


class ReadinessProbe(Protocol):
    """A side-effect-free probe implemented by an adapter or host."""

    def probe(self) -> ReadinessResult:
        """Return the current host/target readiness without mutation."""
        ...


class EnvironmentReadinessError(GLRError):
    """Raised when an attach/reset gate observes a non-ready environment."""

    def __init__(self, result: ReadinessResult) -> None:
        if result.ready:
            raise ValueError("a readiness error requires a non-ready result")
        self.result = result
        super().__init__(f"environment {result.state.value}: {result.reason or 'no reason given'}")


class ReadinessMonitor:
    """Cache and gate probe results without consuming an episode budget."""

    def __init__(self, probe: ReadinessProbe | Callable[[], ReadinessResult]) -> None:
        self._probe = probe
        self._last: ReadinessResult | None = None

    @property
    def last_result(self) -> ReadinessResult | None:
        return self._last

    def check(self) -> ReadinessResult:
        result = self._probe() if callable(self._probe) else self._probe.probe()
        if not isinstance(result, ReadinessResult):
            raise TypeError("readiness probe must return ReadinessResult")
        self._last = result
        return result

    def require_ready(self) -> ReadinessResult:
        result = self.check()
        if not result.ready:
            raise EnvironmentReadinessError(result)
        return result

    def wait_until_ready(
        self, *, timeout_seconds: float, poll_interval_seconds: float
    ) -> ReadinessResult:
        """Park and re-probe until ready, with a bounded timeout."""

        if timeout_seconds < 0 or poll_interval_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be non-negative and poll_interval_seconds positive"
            )
        deadline = monotonic_ns() + int(timeout_seconds * 1_000_000_000)
        while True:
            result = self.check()
            if result.ready:
                return result
            if monotonic_ns() >= deadline:
                raise EnvironmentReadinessError(result)
            remaining = deadline - monotonic_ns()
            sleep(min(poll_interval_seconds, remaining / 1_000_000_000))


__all__ = [
    "READINESS_SCHEMA_VERSION",
    "EnvironmentReadinessError",
    "ReadinessMonitor",
    "ReadinessProbe",
    "ReadinessResult",
    "ReadinessState",
]
