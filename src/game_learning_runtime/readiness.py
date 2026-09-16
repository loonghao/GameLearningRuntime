"""Side-effect-free host readiness probes.

Readiness is deliberately separate from transport health and episode
progress. A probe may be run before attaching to an external target and,
optionally, while an episode is running. The probe never starts, stops, or
mutates the target; callers decide how to park and retry a not-ready host.

:func:`run_readiness_window` is that parking policy for a startup verb. A role
that cannot serve yet publishes the very same ``ReadinessResult`` mapping this
module already defines, and the caller re-invokes the role until the host
reports ready, reports a terminal state, or the declared window expires. The
window therefore belongs to the project contract instead of to a settle
constant re-implemented inside every adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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


def readiness_from_mapping(value: object, *, path: str = "readiness receipt") -> ReadinessResult:
    """Read back the published ``glr.environment-readiness.v1`` mapping.

    A role reports readiness to its caller through this mapping, so the shared
    reader is strict about the declared schema and named state, and tolerant of
    extra keys a producer may add. It never guesses a state.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    if value.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ValueError(f"{path} must declare schema_version {READINESS_SCHEMA_VERSION!r}")
    state = value.get("state")
    if not isinstance(state, str):
        raise TypeError(f"{path}.state must be a readiness state string")
    try:
        resolved_state = ReadinessState(state)
    except ValueError as error:
        states = ", ".join(item.value for item in ReadinessState)
        raise ValueError(f"{path}.state must be one of {states}") from error
    reason = value.get("reason", "")
    if not isinstance(reason, str):
        raise TypeError(f"{path}.reason must be a string")
    checked_at_ns = value.get("checked_at_ns", 0)
    if not isinstance(checked_at_ns, int) or isinstance(checked_at_ns, bool) or checked_at_ns < 0:
        raise ValueError(f"{path}.checked_at_ns must be a non-negative integer")
    return ReadinessResult(resolved_state, reason, checked_at_ns)


class ReadinessWindowVerdict(str, Enum):
    """The bounded verdict of one declared startup window."""

    SUCCEEDED = "succeeded"
    NOT_READY = "not_ready"
    UNAVAILABLE = "unavailable"
    INCONSISTENT = "inconsistent"
    UNREPORTED = "unreported"


@dataclass(frozen=True, slots=True)
class ReadinessAttempt:
    """One role invocation and the readiness receipt it published, if any."""

    index: int
    exit_code: int
    result: ReadinessResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 1:
            raise ValueError("readiness attempt index must be a positive integer")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise ValueError("readiness attempt exit_code must be an integer")
        if self.result is not None and not isinstance(self.result, ReadinessResult):
            raise TypeError("readiness attempt result must be a ReadinessResult or None")

    def to_mapping(self) -> dict[str, object]:
        return {
            "index": self.index,
            "exit_code": self.exit_code,
            "readiness": None if self.result is None else self.result.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ReadinessWindowOutcome:
    """The reusable result of parking a startup verb inside one window."""

    verdict: ReadinessWindowVerdict
    attempts: tuple[ReadinessAttempt, ...]
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, ReadinessWindowVerdict):
            raise TypeError("readiness window verdict must be a ReadinessWindowVerdict")
        if not self.attempts:
            raise ValueError("a readiness window must contain at least one attempt")
        if tuple(attempt.index for attempt in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise ValueError("readiness window attempts must be numbered from one")

    @property
    def succeeded(self) -> bool:
        return self.verdict is ReadinessWindowVerdict.SUCCEEDED

    @property
    def exhausted(self) -> bool:
        """True when the host was still parking on a retryable state."""

        return self.verdict is ReadinessWindowVerdict.NOT_READY

    @property
    def last_attempt(self) -> ReadinessAttempt:
        return self.attempts[-1]

    @property
    def last_result(self) -> ReadinessResult | None:
        return self.last_attempt.result

    def to_mapping(self) -> dict[str, object]:
        """Describe the window and every receipt without restating a schema.

        The embedded ``readiness`` mappings keep ``to_mapping()`` of
        :class:`ReadinessResult` verbatim; the window itself is not a probe
        result, so it deliberately declares no ``schema_version``.
        """

        return {
            "verdict": self.verdict.value,
            "exhausted": self.exhausted,
            "timeout_seconds": self.timeout_seconds,
            "attempts": [attempt.to_mapping() for attempt in self.attempts],
        }


def run_readiness_window(
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    attempt: Callable[[int], ReadinessAttempt],
    sleep_seconds: Callable[[float], None] = sleep,
) -> ReadinessWindowOutcome:
    """Re-invoke a role until it is judged ready, terminal, or out of window.

    ``attempt`` performs one bounded role invocation and returns its exit code
    with the readiness receipt the role published. Only an explicit
    ``not_ready`` receipt is retryable: a missing, unreadable, ``unavailable``,
    or contradictory receipt is terminal on the first observation, so a crash
    is never retried and never mistaken for a host that is still starting.
    """

    if not (timeout_seconds > 0 and poll_interval_seconds > 0):
        raise ValueError("timeout_seconds and poll_interval_seconds must be positive")
    deadline = monotonic_ns() + int(timeout_seconds * 1_000_000_000)
    attempts: list[ReadinessAttempt] = []
    while True:
        record = attempt(len(attempts) + 1)
        if not isinstance(record, ReadinessAttempt):
            raise TypeError("a readiness attempt must return ReadinessAttempt")
        attempts.append(record)
        verdict = _attempt_verdict(record)
        if verdict is not None:
            return ReadinessWindowOutcome(
                verdict=verdict,
                attempts=tuple(attempts),
                timeout_seconds=timeout_seconds,
            )
        now = monotonic_ns()
        if now >= deadline:
            return ReadinessWindowOutcome(
                verdict=ReadinessWindowVerdict.NOT_READY,
                attempts=tuple(attempts),
                timeout_seconds=timeout_seconds,
            )
        sleep_seconds(min(poll_interval_seconds, (deadline - now) / 1_000_000_000))


def _attempt_verdict(record: ReadinessAttempt) -> ReadinessWindowVerdict | None:
    """Return a terminal verdict for one attempt, or None to keep parking."""

    if record.exit_code == 0:
        return ReadinessWindowVerdict.SUCCEEDED
    result = record.result
    if result is None:
        return ReadinessWindowVerdict.UNREPORTED
    if result.ready:
        return ReadinessWindowVerdict.INCONSISTENT
    if result.state is ReadinessState.UNAVAILABLE:
        return ReadinessWindowVerdict.UNAVAILABLE
    return None


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
    "ReadinessAttempt",
    "ReadinessMonitor",
    "ReadinessProbe",
    "ReadinessResult",
    "ReadinessState",
    "ReadinessWindowOutcome",
    "ReadinessWindowVerdict",
    "readiness_from_mapping",
    "run_readiness_window",
]
