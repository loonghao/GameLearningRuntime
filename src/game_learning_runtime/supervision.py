"""Optional lifecycle supervision for externally owned target processes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic_ns, sleep
from typing import Protocol

from game_learning_runtime.errors import GLRError

SUPERVISION_SCHEMA_VERSION = "glr.process-supervision.v1"


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """PID plus a start timestamp, preventing PID-reuse confusion."""

    pid: int
    start_time_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.pid, int) or isinstance(self.pid, bool) or self.pid <= 0:
            raise ValueError("process pid must be a positive integer")
        if (
            not isinstance(self.start_time_ns, int)
            or isinstance(self.start_time_ns, bool)
            or self.start_time_ns < 0
        ):
            raise ValueError("process start_time_ns must be a non-negative integer")

    def to_mapping(self) -> dict[str, int]:
        return {"pid": self.pid, "start_time_ns": self.start_time_ns}


@dataclass(frozen=True, slots=True)
class StopAction:
    name: str
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.name or any(ord(character) < 32 for character in self.name):
            raise ValueError("stop action name must be printable and non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("stop action timeout_seconds must be positive")


class ProcessProbe(Protocol):
    """Adapter-owned, bounded process operations used by the supervisor."""

    def is_alive(self, identity: ProcessIdentity) -> bool: ...

    def invoke(self, action: str, identity: ProcessIdentity) -> None: ...

    def launch(self) -> ProcessIdentity: ...


class LeaseConflictError(GLRError):
    """Raised when a second environment attaches to a live target lease."""

    def __init__(self, identity: ProcessIdentity) -> None:
        self.identity = identity
        super().__init__(f"process lease is already held for pid {identity.pid}")


class SupervisionError(GLRError):
    """Raised when a declared lifecycle step cannot be verified."""


class ArtifactOwnershipError(GLRError):
    """Raised when an artifact operation is attempted while its owner lives."""


class ExclusiveInstanceLease:
    """Small in-process lease registry; adapters may replace it with a durable store."""

    def __init__(self) -> None:
        self._identity: ProcessIdentity | None = None

    @property
    def identity(self) -> ProcessIdentity | None:
        return self._identity

    def acquire(self, identity: ProcessIdentity) -> None:
        if self._identity is not None:
            raise LeaseConflictError(self._identity)
        self._identity = identity

    def release(self, identity: ProcessIdentity) -> None:
        if self._identity != identity:
            raise SupervisionError("process lease identity does not match the held instance")
        self._identity = None


@dataclass(frozen=True, slots=True)
class StopResult:
    identity: ProcessIdentity
    stopped: bool
    ended_by: str | None
    restart_count: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": SUPERVISION_SCHEMA_VERSION,
            "identity": self.identity.to_mapping(),
            "stopped": self.stopped,
            "ended_by": self.ended_by,
            "restart_count": self.restart_count,
        }


class ProcessSupervisor:
    """Run an explicit stop sequence and preserve exclusivity across restarts."""

    def __init__(
        self,
        probe: ProcessProbe,
        *,
        stop_sequence: tuple[StopAction, ...] = (
            StopAction("request-close", 5.0),
            StopAction("terminate", 5.0),
            StopAction("kill", 2.0),
        ),
        lease: ExclusiveInstanceLease | None = None,
        sleep_fn: Callable[[float], None] = sleep,
    ) -> None:
        if not stop_sequence:
            raise ValueError("stop_sequence cannot be empty")
        self._probe = probe
        self._sequence = tuple(stop_sequence)
        self._lease = lease or ExclusiveInstanceLease()
        self._sleep = sleep_fn
        self._identity: ProcessIdentity | None = None
        self._restart_count = 0

    @property
    def identity(self) -> ProcessIdentity | None:
        return self._identity

    @property
    def restart_count(self) -> int:
        return self._restart_count

    def attach(self, identity: ProcessIdentity) -> None:
        if self._identity is not None:
            raise SupervisionError("supervisor is already attached")
        if not self._probe.is_alive(identity):
            raise SupervisionError("cannot acquire a lease for a process that is not alive")
        self._lease.acquire(identity)
        self._identity = identity

    def require_artifact_stopped(self, operation: str) -> None:
        if not operation or any(ord(character) < 32 for character in operation):
            raise ValueError("artifact operation must be printable and non-empty")
        if self._identity is not None and self._probe.is_alive(self._identity):
            raise ArtifactOwnershipError(
                f"artifact operation {operation!r} requires the target process to be stopped"
            )

    def stop(self) -> StopResult:
        identity = self._identity
        if identity is None:
            raise SupervisionError("stop requires an attached process")
        ended_by: str | None = None
        for action in self._sequence:
            if not self._probe.is_alive(identity):
                ended_by = ended_by or "already-stopped"
                break
            self._probe.invoke(action.name, identity)
            if self._wait_gone(identity, action.timeout_seconds):
                ended_by = action.name
                break
        stopped = not self._probe.is_alive(identity)
        if stopped:
            self._lease.release(identity)
            self._identity = None
        return StopResult(identity, stopped, ended_by, self._restart_count)

    def restart(self) -> ProcessIdentity:
        result = self.stop()
        if not result.stopped:
            raise SupervisionError("restart refused because the target did not stop")
        identity = self._probe.launch()
        self.attach(identity)
        self._restart_count += 1
        return identity

    def _wait_gone(self, identity: ProcessIdentity, timeout_seconds: float) -> bool:
        deadline = monotonic_ns() + int(timeout_seconds * 1_000_000_000)
        while self._probe.is_alive(identity):
            if monotonic_ns() >= deadline:
                return False
            self._sleep(min(0.01, (deadline - monotonic_ns()) / 1_000_000_000))
        return True


__all__ = [
    "SUPERVISION_SCHEMA_VERSION",
    "ArtifactOwnershipError",
    "ExclusiveInstanceLease",
    "LeaseConflictError",
    "ProcessIdentity",
    "ProcessProbe",
    "ProcessSupervisor",
    "StopAction",
    "StopResult",
    "SupervisionError",
]
