"""Bounded watchdog policy for scheduler-driven supervision and recovery.

A watchdog answers one question per supervised source: is the source still
proving that it is alive, and if not, what bounded recovery is allowed?  The
policy is deliberately finite.  A source that exhausts its recovery attempt
budget is escalated instead of being restarted forever, because an unbounded
restart loop hides a real defect and burns scheduler capacity.

The module is policy-only.  Process liveness and restart mechanics stay behind
:class:`ProcessSupervisor` or an injected recovery runner, so the watchdog can
be driven from a cron job, a task scheduler, or a long-lived daemon without
changing its decisions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from subprocess import CompletedProcess, run
from time import monotonic_ns, sleep
from typing import Protocol

from game_learning_runtime.errors import GLRError
from game_learning_runtime.supervision import ProcessSupervisor

WATCHDOG_SCHEMA_VERSION = "glr.watchdog-report.v1"
HEARTBEAT_SCHEMA_VERSION = "glr.heartbeat.v1"

#: Exit codes are part of the scheduler contract. A scheduler can branch on
#: them without parsing output: 0 = healthy, 3 = a recovery attempt succeeded,
#: 4 = escalation required because an intervention failed, is unavailable, or
#: its attempt budget is exhausted.
WATCHDOG_EXIT_HEALTHY = 0
WATCHDOG_EXIT_RECOVERED = 3
WATCHDOG_EXIT_ESCALATED = 4

_NS_PER_SECOND = 1_000_000_000


class WatchdogStatus(str, Enum):
    """Observed lifecycle state of one supervised source."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STARVED = "starved"
    RECOVERING = "recovering"
    FAILED = "failed"


class WatchdogAction(str, Enum):
    """Bounded action the watchdog takes for one source in one pass."""

    NONE = "none"
    NOTIFY = "notify"
    RESTART = "restart"
    ESCALATE = "escalate"


class RecoveryRunner(Protocol):
    """Runs one bounded recovery attempt for a supervised source."""

    def __call__(self, argv: Sequence[str], *, timeout_seconds: float) -> bool: ...


def _positive_number(value: float, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{path} must be finite")
    if result <= 0:
        raise ValueError(f"{path} must be positive")
    return result


def _non_negative_number(value: float, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be a number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{path} must be finite")
    if result < 0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _non_negative_int(value: int, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _identifier(value: str, *, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{path} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{path} must not contain control characters")
    return value


@dataclass(frozen=True, slots=True)
class WatchdogPolicy:
    """Finite recovery budget applied to one supervised source.

    ``heartbeat_timeout_seconds`` is the longest gap tolerated before a
    heartbeat is considered late. ``max_missed_heartbeats`` converts repeated
    lateness into starvation. ``restart_attempt_limit`` is the total number of
    automatic recovery *attempts* allowed for one source, counted whether each
    attempt succeeds or fails; once exhausted the watchdog escalates and stops
    touching the source.
    """

    heartbeat_timeout_seconds: float = 30.0
    max_missed_heartbeats: int = 3
    restart_attempt_limit: int = 3
    restart_backoff_seconds: float = 5.0
    restart_cooldown_seconds: float = 30.0
    recovery_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        _positive_number(self.heartbeat_timeout_seconds, path="watchdog heartbeat_timeout_seconds")
        _positive_number(self.recovery_timeout_seconds, path="watchdog recovery_timeout_seconds")
        _non_negative_number(self.restart_backoff_seconds, path="watchdog restart_backoff_seconds")
        _non_negative_number(
            self.restart_cooldown_seconds, path="watchdog restart_cooldown_seconds"
        )
        if not isinstance(self.max_missed_heartbeats, int) or isinstance(
            self.max_missed_heartbeats, bool
        ):
            raise ValueError("watchdog max_missed_heartbeats must be an integer")
        if self.max_missed_heartbeats < 1:
            raise ValueError("watchdog max_missed_heartbeats must be at least 1")
        _non_negative_int(self.restart_attempt_limit, path="watchdog restart_attempt_limit")

    @property
    def starvation_seconds(self) -> float:
        """Age in seconds after which a source is considered starved."""

        return self.heartbeat_timeout_seconds * self.max_missed_heartbeats

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "max_missed_heartbeats": self.max_missed_heartbeats,
            "restart_attempt_limit": self.restart_attempt_limit,
            "restart_backoff_seconds": self.restart_backoff_seconds,
            "restart_cooldown_seconds": self.restart_cooldown_seconds,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
            "starvation_seconds": self.starvation_seconds,
        }


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """One bounded liveness proof emitted by a supervised source."""

    source: str
    sequence: int
    observed_at_ns: int
    state: str = "running"
    detail: str = ""

    def __post_init__(self) -> None:
        _identifier(self.source, path="heartbeat source")
        _non_negative_int(self.sequence, path="heartbeat sequence")
        _non_negative_int(self.observed_at_ns, path="heartbeat observed_at_ns")
        _identifier(self.state, path="heartbeat state")
        if not isinstance(self.detail, str):
            raise ValueError("heartbeat detail must be a string")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": HEARTBEAT_SCHEMA_VERSION,
            "source": self.source,
            "sequence": self.sequence,
            "observed_at_ns": self.observed_at_ns,
            "state": self.state,
            "detail": self.detail,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> Heartbeat:
        """Rebuild a heartbeat from a stored mapping, rejecting unknown fields.

        ``schema_version`` is the envelope written by :meth:`to_mapping` and is
        accepted but ignored, so a stored line round-trips unchanged.
        """

        expected = frozenset(
            {"schema_version", "source", "sequence", "observed_at_ns", "state", "detail"}
        )
        unexpected = sorted(set(mapping) - expected)
        if unexpected:
            raise ValueError(f"heartbeat mapping has unexpected fields {unexpected}")
        missing = sorted({"source", "sequence", "observed_at_ns"} - set(mapping))
        if missing:
            raise ValueError(f"heartbeat mapping is missing fields {missing}")
        return cls(
            source=mapping["source"],  # type: ignore[arg-type]
            sequence=mapping["sequence"],  # type: ignore[arg-type]
            observed_at_ns=mapping["observed_at_ns"],  # type: ignore[arg-type]
            state=mapping.get("state", "running"),  # type: ignore[arg-type]
            detail=mapping.get("detail", ""),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class WatchdogDecision:
    """One source's status and the bounded action taken for it."""

    source: str
    status: WatchdogStatus
    action: WatchdogAction
    reason: str
    age_seconds: float | None
    missed_heartbeats: int
    restart_attempts: int
    restart_attempt_limit: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "source": self.source,
            "status": self.status.value,
            "action": self.action.value,
            "reason": self.reason,
            "age_seconds": self.age_seconds,
            "missed_heartbeats": self.missed_heartbeats,
            "restart_attempts": self.restart_attempts,
            "restart_attempt_limit": self.restart_attempt_limit,
        }


@dataclass(frozen=True, slots=True)
class WatchdogReport:
    """Result of one watchdog pass across every registered source."""

    decisions: tuple[WatchdogDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.decisions, tuple):
            raise ValueError("watchdog decisions must be a tuple")
        for decision in self.decisions:
            if not isinstance(decision, WatchdogDecision):
                raise ValueError("watchdog decisions must contain WatchdogDecision values")

    @property
    def escalated(self) -> tuple[WatchdogDecision, ...]:
        return tuple(d for d in self.decisions if d.action is WatchdogAction.ESCALATE)

    @property
    def recovered(self) -> tuple[WatchdogDecision, ...]:
        """Sources whose recovery attempt actually succeeded this pass."""

        return tuple(d for d in self.decisions if d.action is WatchdogAction.RESTART)

    @property
    def recovery_failures(self) -> tuple[WatchdogDecision, ...]:
        """Sources whose recovery attempt failed this pass.

        Reported separately from :attr:`escalated` so an operator can tell a
        failed intervention from one that was never attempted. A failed attempt
        is still escalated, so it never reports exit code ``3``.
        """

        return tuple(d for d in self.decisions if d.reason == "restart-failed")

    @property
    def exit_code(self) -> int:
        """Scheduler-facing exit code for this pass.

        ``3`` means a recovery *succeeded*. A recovery attempt that failed is
        escalated instead, so a scheduler can never read a failed intervention
        as "recovered".
        """

        if self.escalated:
            return WATCHDOG_EXIT_ESCALATED
        if self.recovered:
            return WATCHDOG_EXIT_RECOVERED
        return WATCHDOG_EXIT_HEALTHY

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "exit_code": self.exit_code,
            "escalated": [d.source for d in self.escalated],
            "recovered": [d.source for d in self.recovered],
            "recovery_failures": [d.source for d in self.recovery_failures],
            "decisions": [d.to_mapping() for d in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class WatchdogTarget:
    """One registered source with its recovery wiring and optional policy."""

    name: str
    policy: WatchdogPolicy
    supervisor: ProcessSupervisor | None = None
    recovery_command: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, path="watchdog target name")
        if not isinstance(self.policy, WatchdogPolicy):
            raise ValueError("watchdog target policy must be a WatchdogPolicy")
        if self.recovery_command is not None:
            if not isinstance(self.recovery_command, tuple) or not self.recovery_command:
                raise ValueError("watchdog recovery_command must be a non-empty tuple")
            for part in self.recovery_command:
                if not isinstance(part, str) or not part:
                    raise ValueError("watchdog recovery_command parts must be non-empty strings")

    @property
    def restartable(self) -> bool:
        return self.supervisor is not None or self.recovery_command is not None


class WatchdogStateError(GLRError):
    """Raised when a watchdog operation targets an unknown source."""


def _command_recovery_runner(argv: Sequence[str], *, timeout_seconds: float) -> bool:
    """Run one recovery command to completion and report whether it succeeded.

    A scheduler-driven recovery should not flash a console window, so on Windows
    the child is started without one. The argv list is still passed with
    ``shell=False``, which keeps the no-injection, no-quoting property.
    """

    command = list(argv)
    if sys.platform == "win32":
        # `CREATE_NO_WINDOW` keeps a scheduled pass from flashing a console.
        completed: CompletedProcess[str] = run(
            command,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        completed = run(
            command,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
    return completed.returncode == 0


class HeartbeatLog:
    """Append-only JSON Lines heartbeat log shared between a runner and cron.

    A scheduled trainer appends one line per heartbeat; a scheduled
    ``glr watchdog tick`` reads the newest heartbeat per source.  The log is
    written by an external process, so reads tolerate truncated trailing lines
    and never mutate the file.
    """

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise ValueError("heartbeat log path must be a Path")
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def append(self, heartbeat: Heartbeat) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(heartbeat.to_mapping(), ensure_ascii=False, separators=(",", ":"))
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"{payload}\n")

    def read(self) -> tuple[Heartbeat, ...]:
        """Return every parsable heartbeat, newest last, skipping partial lines."""

        if not self._path.is_file():
            return ()
        heartbeats: list[Heartbeat] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    mapping = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(mapping, dict):
                    continue
                try:
                    heartbeats.append(Heartbeat.from_mapping(mapping))
                except (ValueError, TypeError):
                    continue
        return tuple(heartbeats)

    def latest_by_source(self) -> dict[str, Heartbeat]:
        """Return the newest heartbeat per source, compared by sequence."""

        latest: dict[str, Heartbeat] = {}
        for heartbeat in self.read():
            current = latest.get(heartbeat.source)
            if current is None or heartbeat.sequence >= current.sequence:
                latest[heartbeat.source] = heartbeat
        return latest


@dataclass(slots=True)
class _SourceState:
    registered_at_ns: int
    last_heartbeat: Heartbeat | None = None
    restart_attempts: int = 0
    last_restart_at_ns: int | None = None


class SupervisionWatchdog:
    """Evaluate heartbeats against a finite policy and act once per pass.

    The watchdog never blocks indefinitely and never restarts without a budget.
    A single pass is safe to run from cron: it reads heartbeats, decides,
    performs at most one bounded recovery per source, and returns a report whose
    :attr:`WatchdogReport.exit_code` is the scheduler contract.
    """

    def __init__(
        self,
        *,
        policy: WatchdogPolicy | None = None,
        clock: Callable[[], int] = monotonic_ns,
        sleep_fn: Callable[[float], None] = sleep,
        recovery_runner: RecoveryRunner | None = None,
    ) -> None:
        self._default_policy = policy or WatchdogPolicy()
        if not isinstance(self._default_policy, WatchdogPolicy):
            raise ValueError("watchdog policy must be a WatchdogPolicy")
        self._clock = clock
        self._sleep = sleep_fn
        self._recovery_runner: RecoveryRunner = recovery_runner or _command_recovery_runner
        self._targets: dict[str, WatchdogTarget] = {}
        self._states: dict[str, _SourceState] = {}

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self._targets)

    def register(self, target: WatchdogTarget) -> None:
        if target.name in self._targets:
            raise WatchdogStateError(f"watchdog source {target.name!r} is already registered")
        self._targets[target.name] = target
        self._states[target.name] = _SourceState(registered_at_ns=self._clock())

    def observe(self, heartbeat: Heartbeat) -> None:
        """Record one heartbeat, ignoring stale or unknown sources."""

        state = self._states.get(heartbeat.source)
        if state is None:
            return
        if state.last_heartbeat is not None and heartbeat.sequence < state.last_heartbeat.sequence:
            return
        state.last_heartbeat = heartbeat

    def observe_all(self, heartbeats: Sequence[Heartbeat]) -> None:
        for heartbeat in heartbeats:
            self.observe(heartbeat)

    def restart_attempts(self, source: str) -> int:
        state = self._states.get(source)
        if state is None:
            raise WatchdogStateError(f"watchdog source {source!r} is not registered")
        return state.restart_attempts

    def evaluate(self, source: str, *, now_ns: int | None = None) -> WatchdogDecision:
        """Decide status and action for one source without acting."""

        target = self._targets.get(source)
        state = self._states.get(source)
        if target is None or state is None:
            raise WatchdogStateError(f"watchdog source {source!r} is not registered")
        now = self._clock() if now_ns is None else now_ns
        policy = target.policy

        if state.last_heartbeat is None:
            age_seconds = (now - state.registered_at_ns) / _NS_PER_SECOND
            missed = policy.max_missed_heartbeats
            if age_seconds <= policy.starvation_seconds:
                return WatchdogDecision(
                    source=source,
                    status=WatchdogStatus.DEGRADED,
                    action=WatchdogAction.NOTIFY,
                    reason="no-heartbeat-yet",
                    age_seconds=age_seconds,
                    missed_heartbeats=missed,
                    restart_attempts=state.restart_attempts,
                    restart_attempt_limit=policy.restart_attempt_limit,
                )
            return self._starved_decision(target, state, age_seconds, missed, now)

        age_ns = now - state.last_heartbeat.observed_at_ns
        age_seconds = max(0.0, age_ns / _NS_PER_SECOND)
        missed = int(age_seconds // policy.heartbeat_timeout_seconds)

        if state.last_restart_at_ns is not None:
            since_restart = (now - state.last_restart_at_ns) / _NS_PER_SECOND
            if since_restart < policy.restart_cooldown_seconds:
                return WatchdogDecision(
                    source=source,
                    status=WatchdogStatus.RECOVERING,
                    action=WatchdogAction.NONE,
                    reason="awaiting-heartbeat-after-restart",
                    age_seconds=age_seconds,
                    missed_heartbeats=missed,
                    restart_attempts=state.restart_attempts,
                    restart_attempt_limit=policy.restart_attempt_limit,
                )

        if age_seconds <= policy.heartbeat_timeout_seconds:
            return WatchdogDecision(
                source=source,
                status=WatchdogStatus.HEALTHY,
                action=WatchdogAction.NONE,
                reason="heartbeat-current",
                age_seconds=age_seconds,
                missed_heartbeats=0,
                restart_attempts=state.restart_attempts,
                restart_attempt_limit=policy.restart_attempt_limit,
            )
        if age_seconds <= policy.starvation_seconds:
            return WatchdogDecision(
                source=source,
                status=WatchdogStatus.DEGRADED,
                action=WatchdogAction.NOTIFY,
                reason="heartbeat-late",
                age_seconds=age_seconds,
                missed_heartbeats=missed,
                restart_attempts=state.restart_attempts,
                restart_attempt_limit=policy.restart_attempt_limit,
            )
        return self._starved_decision(target, state, age_seconds, missed, now)

    def tick(self, *, now_ns: int | None = None) -> WatchdogReport:
        """Run one full pass: evaluate every source and act once per source."""

        now = self._clock() if now_ns is None else now_ns
        decisions: list[WatchdogDecision] = []
        for source in self._targets:
            decision = self.evaluate(source, now_ns=now)
            if decision.action is WatchdogAction.RESTART:
                decision = self._recover(source, decision, now)
            decisions.append(decision)
        return WatchdogReport(tuple(decisions))

    def run_once(self, *, now_ns: int | None = None) -> int:
        """Run one pass and return the scheduler-facing exit code."""

        return self.tick(now_ns=now_ns).exit_code

    def run(
        self,
        *,
        interval_seconds: float,
        max_ticks: int | None = None,
    ) -> WatchdogReport:
        """Run repeated passes for long-lived supervisor processes.

        The loop is bounded by ``max_ticks``. Escalation ends the loop and
        returns the escalating report, so a supervisor host can alert instead
        of continuing to poke a broken source.
        """

        _positive_number(interval_seconds, path="watchdog interval_seconds")
        if max_ticks is not None and (
            not isinstance(max_ticks, int) or isinstance(max_ticks, bool) or max_ticks < 1
        ):
            raise ValueError("watchdog max_ticks must be a positive integer or None")
        completed = 0
        while True:
            report = self.tick()
            completed += 1
            if report.exit_code == WATCHDOG_EXIT_ESCALATED:
                return report
            if max_ticks is not None and completed >= max_ticks:
                return report
            self._sleep(interval_seconds)

    def _starved_decision(
        self,
        target: WatchdogTarget,
        state: _SourceState,
        age_seconds: float,
        missed: int,
        now: int,
    ) -> WatchdogDecision:
        policy = target.policy
        if state.restart_attempts >= policy.restart_attempt_limit:
            return WatchdogDecision(
                source=target.name,
                status=WatchdogStatus.FAILED,
                action=WatchdogAction.ESCALATE,
                reason="restart-attempts-exhausted",
                age_seconds=age_seconds,
                missed_heartbeats=missed,
                restart_attempts=state.restart_attempts,
                restart_attempt_limit=policy.restart_attempt_limit,
            )
        if state.last_restart_at_ns is not None:
            since_restart = (now - state.last_restart_at_ns) / _NS_PER_SECOND
            if since_restart < policy.restart_backoff_seconds:
                return WatchdogDecision(
                    source=target.name,
                    status=WatchdogStatus.RECOVERING,
                    action=WatchdogAction.NONE,
                    reason="restart-backoff-active",
                    age_seconds=age_seconds,
                    missed_heartbeats=missed,
                    restart_attempts=state.restart_attempts,
                    restart_attempt_limit=policy.restart_attempt_limit,
                )
        if not target.restartable:
            # Nothing can restart this source, so the only honest signal is to
            # escalate: a scheduled watchdog that reports "healthy" here would
            # hide a silent trainer from the very job that is watching it.
            return WatchdogDecision(
                source=target.name,
                status=WatchdogStatus.STARVED,
                action=WatchdogAction.ESCALATE,
                reason="detect-only-no-recovery-wiring",
                age_seconds=age_seconds,
                missed_heartbeats=missed,
                restart_attempts=state.restart_attempts,
                restart_attempt_limit=policy.restart_attempt_limit,
            )
        return WatchdogDecision(
            source=target.name,
            status=WatchdogStatus.STARVED,
            action=WatchdogAction.RESTART,
            reason="heartbeat-starved",
            age_seconds=age_seconds,
            missed_heartbeats=missed,
            restart_attempts=state.restart_attempts,
            restart_attempt_limit=policy.restart_attempt_limit,
        )

    def _recover(self, source: str, decision: WatchdogDecision, now: int) -> WatchdogDecision:
        target = self._targets[source]
        state = self._states[source]
        policy = target.policy
        recovered = False
        if target.supervisor is not None:
            try:
                target.supervisor.restart()
                recovered = True
            except GLRError:
                recovered = False
        elif target.recovery_command is not None:
            try:
                recovered = self._recovery_runner(
                    target.recovery_command, timeout_seconds=policy.recovery_timeout_seconds
                )
            except (OSError, subprocess.TimeoutExpired):
                # `subprocess.run(timeout=...)` raises `TimeoutExpired`, which is
                # neither `OSError` nor `TimeoutError`. Swallowing it here keeps
                # the scheduler contract intact: a timed-out recovery is a failed
                # intervention, reported as exit code 4 rather than a traceback.
                recovered = False
        state.restart_attempts += 1
        state.last_restart_at_ns = now
        if recovered:
            return WatchdogDecision(
                source=source,
                status=WatchdogStatus.RECOVERING,
                action=WatchdogAction.RESTART,
                reason="restart-issued",
                age_seconds=decision.age_seconds,
                missed_heartbeats=decision.missed_heartbeats,
                restart_attempts=state.restart_attempts,
                restart_attempt_limit=policy.restart_attempt_limit,
            )
        # A failed intervention never reports "recovered". It escalates on the
        # first failure so a scheduler cannot be told the run is fine; further
        # attempts still happen on later passes until the budget is exhausted.
        return WatchdogDecision(
            source=source,
            status=WatchdogStatus.FAILED,
            action=WatchdogAction.ESCALATE,
            reason="restart-failed",
            age_seconds=decision.age_seconds,
            missed_heartbeats=decision.missed_heartbeats,
            restart_attempts=state.restart_attempts,
            restart_attempt_limit=policy.restart_attempt_limit,
        )


def watchdog_policy_from_mapping(mapping: Mapping[str, object]) -> WatchdogPolicy:
    """Build a policy from a configuration mapping, rejecting unknown keys."""

    expected = frozenset(
        {
            "heartbeat_timeout_seconds",
            "max_missed_heartbeats",
            "restart_attempt_limit",
            "restart_backoff_seconds",
            "restart_cooldown_seconds",
            "recovery_timeout_seconds",
        }
    )
    unexpected = sorted(set(mapping) - expected)
    if unexpected:
        raise ValueError(f"watchdog policy has unexpected fields {unexpected}")
    return WatchdogPolicy(**mapping)  # type: ignore[arg-type]


__all__ = [
    "HEARTBEAT_SCHEMA_VERSION",
    "WATCHDOG_EXIT_ESCALATED",
    "WATCHDOG_EXIT_HEALTHY",
    "WATCHDOG_EXIT_RECOVERED",
    "WATCHDOG_SCHEMA_VERSION",
    "Heartbeat",
    "HeartbeatLog",
    "RecoveryRunner",
    "SupervisionWatchdog",
    "WatchdogAction",
    "WatchdogDecision",
    "WatchdogPolicy",
    "WatchdogReport",
    "WatchdogStateError",
    "WatchdogStatus",
    "WatchdogTarget",
    "watchdog_policy_from_mapping",
]
