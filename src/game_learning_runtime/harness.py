"""Optional task harness contracts for external agent providers.

The harness boundary is deliberately separate from GLR's environment and
learner contracts.  A provider can propose or transform structured tasks, but
it never receives authority to mutate a game runtime through this module.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from time import monotonic, time_ns
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable
from uuid import UUID, uuid4

from game_learning_runtime.errors import GLRError

HARNESS_SCHEMA_VERSION = "glr.harness.v1"


class HarnessError(GLRError):
    """Base error for the optional harness boundary."""


class HarnessDisabledError(HarnessError):
    """Raised when a provider was not explicitly enabled."""


class HarnessPermissionError(HarnessError):
    """Raised when a task requests a capability not granted to the provider."""


class HarnessUnavailableError(HarnessError):
    """Raised when an enabled provider has no configured transport."""


class HarnessRecoveryError(HarnessError):
    """Raised when a persisted state belongs to another provider or schema."""


class HarnessPermission(str, Enum):
    """Capabilities that a harness may request; runtime mutation is absent."""

    CONTEXT_READ = "context.read"
    TASK_SUBMIT = "task.submit"
    EVENT_EMIT = "event.emit"
    RUNTIME_OBSERVE = "runtime.observe"
    RUNTIME_ACT = "runtime.act"


class HarnessResultStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate JSON-shaped data and copy it so callers cannot mutate records."""

    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        copied = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError("harness payload must be JSON-serializable") from error
    if not isinstance(copied, dict):
        raise TypeError("harness payload must be a JSON object")
    return cast(Mapping[str, Any], _freeze_json(copied))


@dataclass(frozen=True, slots=True)
class HarnessCapabilities:
    """Provider capability declaration negotiated before task submission."""

    provider: str
    schema_version: str = HARNESS_SCHEMA_VERSION
    task_kinds: frozenset[str] = frozenset()
    permissions: frozenset[HarnessPermission] = frozenset()
    events: bool = True
    state_recovery: bool = True

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider cannot be empty")
        if self.schema_version != HARNESS_SCHEMA_VERSION:
            raise ValueError(f"unsupported harness schema: {self.schema_version!r}")
        if not self.task_kinds or any(not item.strip() for item in self.task_kinds):
            raise ValueError("task_kinds must contain at least one non-empty kind")
        if not isinstance(self.events, bool) or not isinstance(self.state_recovery, bool):
            raise TypeError("events and state_recovery must be bools")

    def to_mapping(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "schema_version": self.schema_version,
            "task_kinds": sorted(self.task_kinds),
            "permissions": sorted(item.value for item in self.permissions),
            "events": self.events,
            "state_recovery": self.state_recovery,
        }


@dataclass(frozen=True, slots=True)
class HarnessTask:
    """A bounded, structured unit of work submitted to a provider."""

    kind: str
    payload: Mapping[str, Any]
    idempotency_key: str
    task_id: UUID = field(default_factory=uuid4)
    permissions: frozenset[HarnessPermission] = frozenset({HarnessPermission.CONTEXT_READ})
    deadline_ms: int = 30_000
    schema_version: str = HARNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("task kind cannot be empty")
        if not self.idempotency_key.strip() or len(self.idempotency_key) > 256:
            raise ValueError("idempotency_key must be 1..256 characters")
        if self.deadline_ms <= 0 or self.deadline_ms > 300_000:
            raise ValueError("deadline_ms must be between 1 and 300000")
        if self.schema_version != HARNESS_SCHEMA_VERSION:
            raise ValueError(f"unsupported harness schema: {self.schema_version!r}")
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))
        object.__setattr__(self, "permissions", frozenset(self.permissions))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_id": str(self.task_id),
            "kind": self.kind,
            "payload": dict(self.payload),
            "idempotency_key": self.idempotency_key,
            "permissions": sorted(item.value for item in self.permissions),
            "deadline_ms": self.deadline_ms,
        }


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """Exactly-once result associated with one idempotency key."""

    task_id: UUID
    idempotency_key: str
    status: HarnessResultStatus
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", _freeze_mapping(self.output))
        if self.elapsed_ms < 0:
            raise ValueError("elapsed_ms cannot be negative")
        if self.status is HarnessResultStatus.COMPLETED and self.error is not None:
            raise ValueError("completed result cannot contain an error")
        if self.status is not HarnessResultStatus.COMPLETED and not self.error:
            raise ValueError("failed results require an error")

    def to_mapping(self) -> dict[str, object]:
        return {
            "task_id": str(self.task_id),
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
            "output": dict(self.output),
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True, slots=True)
class HarnessEvent:
    """Ordered event emitted by an orchestrator for review and recovery."""

    event_type: str
    task_id: UUID
    sequence: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=time_ns)

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type cannot be empty")
        if self.sequence < 0:
            raise ValueError("event sequence cannot be negative")
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))

    def to_mapping(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "task_id": str(self.task_id),
            "sequence": self.sequence,
            "payload": dict(self.payload),
            "timestamp_ns": self.timestamp_ns,
        }


@dataclass(frozen=True, slots=True)
class HarnessSnapshot:
    """Portable provider state; active work is never serialized."""

    provider: str
    schema_version: str
    completed: tuple[HarnessResult, ...] = ()
    events: tuple[HarnessEvent, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "schema_version": self.schema_version,
            "completed": [item.to_mapping() for item in self.completed],
            "events": [item.to_mapping() for item in self.events],
        }


@runtime_checkable
class HarnessProvider(Protocol):
    """Minimal provider port implemented by DeepSeek or another backend."""

    @property
    def capabilities(self) -> HarnessCapabilities: ...

    def submit(self, task: HarnessTask) -> HarnessResult: ...

    def snapshot(self) -> HarnessSnapshot: ...

    def restore(self, snapshot: HarnessSnapshot) -> None: ...


@runtime_checkable
class HarnessOrchestrator(Protocol):
    """Optional orchestration port; GLR does not require an orchestrator."""

    def submit(self, task: HarnessTask) -> HarnessResult: ...

    def events(self) -> tuple[HarnessEvent, ...]: ...

    def snapshot(self) -> HarnessSnapshot: ...

    def restore(self, snapshot: HarnessSnapshot) -> None: ...


Handler = Callable[[HarnessTask], Mapping[str, Any]]


class DeepSeekHarnessProvider:
    """A disabled-by-default DeepSeek adapter with no network dependency.

    Integrators provide a reviewed handler/transport explicitly.  The provider
    only handles bounded task validation, permissions, idempotency and state;
    it never discovers credentials or grants ``runtime.act`` authority.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        handler: Handler | None = None,
        allowed_permissions: Sequence[HarnessPermission] = (
            HarnessPermission.CONTEXT_READ,
            HarnessPermission.TASK_SUBMIT,
            HarnessPermission.EVENT_EMIT,
        ),
        task_kinds: Sequence[str] = ("completion", "analysis", "plan"),
    ) -> None:
        requested_permissions = frozenset(allowed_permissions)
        if HarnessPermission.RUNTIME_ACT in requested_permissions:
            raise HarnessPermissionError("deepseek-harness cannot be granted runtime.act authority")
        self._enabled = enabled
        self._handler = handler
        self._capabilities = HarnessCapabilities(
            provider="deepseek-harness",
            task_kinds=frozenset(task_kinds),
            permissions=requested_permissions,
        )
        self._completed: dict[str, HarnessResult] = {}
        self._events: list[HarnessEvent] = []
        self._lock = RLock()

    @property
    def capabilities(self) -> HarnessCapabilities:
        return self._capabilities

    @property
    def enabled(self) -> bool:
        """Whether this provider may submit work (off by default)."""

        return self._enabled

    def submit(self, task: HarnessTask) -> HarnessResult:
        if not self._enabled:
            raise HarnessDisabledError("deepseek-harness is disabled; enable it explicitly")
        if task.kind not in self._capabilities.task_kinds:
            raise HarnessPermissionError(f"unsupported task kind: {task.kind}")
        denied = task.permissions - self._capabilities.permissions
        if denied:
            names = ", ".join(sorted(item.value for item in denied))
            raise HarnessPermissionError(f"task requests denied permissions: {names}")
        with self._lock:
            cached = self._completed.get(task.idempotency_key)
            if cached is not None:
                return cached
        if self._handler is None:
            raise HarnessUnavailableError("deepseek-harness has no explicit handler")

        started = monotonic()
        try:
            output = self._handler(task)
            elapsed_ms = int((monotonic() - started) * 1000)
            if elapsed_ms > task.deadline_ms:
                result = HarnessResult(
                    task_id=task.task_id,
                    idempotency_key=task.idempotency_key,
                    status=HarnessResultStatus.TIMED_OUT,
                    error="provider deadline exceeded",
                    elapsed_ms=elapsed_ms,
                )
            else:
                result = HarnessResult(
                    task_id=task.task_id,
                    idempotency_key=task.idempotency_key,
                    status=HarnessResultStatus.COMPLETED,
                    output=output,
                    elapsed_ms=elapsed_ms,
                )
        except TimeoutError:
            result = HarnessResult(
                task_id=task.task_id,
                idempotency_key=task.idempotency_key,
                status=HarnessResultStatus.TIMED_OUT,
                error="provider timed out",
                elapsed_ms=int((monotonic() - started) * 1000),
            )
        except Exception as error:  # provider failures are data, not retries
            result = HarnessResult(
                task_id=task.task_id,
                idempotency_key=task.idempotency_key,
                status=HarnessResultStatus.FAILED,
                error=f"{type(error).__name__}: {error}",
                elapsed_ms=int((monotonic() - started) * 1000),
            )
        with self._lock:
            self._completed.setdefault(task.idempotency_key, result)
        return result

    def snapshot(self) -> HarnessSnapshot:
        with self._lock:
            return HarnessSnapshot(
                provider=self.capabilities.provider,
                schema_version=HARNESS_SCHEMA_VERSION,
                completed=tuple(self._completed.values()),
                events=tuple(self._events),
            )

    def restore(self, snapshot: HarnessSnapshot) -> None:
        if snapshot.provider != self.capabilities.provider:
            raise HarnessRecoveryError("snapshot provider does not match deepseek-harness")
        if snapshot.schema_version != HARNESS_SCHEMA_VERSION:
            raise HarnessRecoveryError("snapshot schema is not supported")
        with self._lock:
            self._completed = {item.idempotency_key: item for item in snapshot.completed}
            self._events = list(snapshot.events)


class LocalHarnessOrchestrator:
    """Small optional eventing wrapper around a single provider."""

    def __init__(self, provider: HarnessProvider) -> None:
        self._provider = provider
        self._events: list[HarnessEvent] = []

    def submit(self, task: HarnessTask) -> HarnessResult:
        self._events.append(HarnessEvent("task.submitted", task.task_id, len(self._events)))
        result = self._provider.submit(task)
        self._events.append(
            HarnessEvent(
                f"task.{result.status.value}",
                task.task_id,
                len(self._events),
                {"idempotency_key": result.idempotency_key, "error": result.error},
            )
        )
        return result

    def events(self) -> tuple[HarnessEvent, ...]:
        return tuple(self._events)

    def snapshot(self) -> HarnessSnapshot:
        state = self._provider.snapshot()
        return HarnessSnapshot(state.provider, state.schema_version, state.completed, self.events())

    def restore(self, snapshot: HarnessSnapshot) -> None:
        self._provider.restore(snapshot)
        self._events = list(snapshot.events)


__all__ = [
    "HARNESS_SCHEMA_VERSION",
    "DeepSeekHarnessProvider",
    "HarnessCapabilities",
    "HarnessDisabledError",
    "HarnessError",
    "HarnessEvent",
    "HarnessOrchestrator",
    "HarnessPermission",
    "HarnessPermissionError",
    "HarnessProvider",
    "HarnessRecoveryError",
    "HarnessResult",
    "HarnessResultStatus",
    "HarnessSnapshot",
    "HarnessTask",
    "HarnessUnavailableError",
    "LocalHarnessOrchestrator",
]
