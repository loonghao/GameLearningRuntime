"""Transport-neutral client boundary for authorized game runtime bridges.

Transport implementations own framing, authentication, deadlines, and
main-thread dispatch.  This module owns the learner-facing environment
lifecycle and request fencing shared by all transports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from time import time_ns
from types import MappingProxyType
from typing import Any, Protocol
from uuid import UUID

from game_learning_runtime.contracts import (
    ActionReconciliation,
    TensorTree,
    TimeStep,
    freeze_tree,
)
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.realtime import (
    InputLeaseBook,
    InputLeaseReceipt,
    InputLeaseRequest,
    InputLeaseToken,
    RealtimeStepTiming,
)
from game_learning_runtime.specs import EnvironmentSpec

_MAX_UINT64 = (1 << 64) - 1


def _copy_string_options(options: Mapping[Any, Any]) -> Mapping[str, str]:
    copied = dict(options)
    if not all(isinstance(key, str) for key in copied):
        raise TypeError("bridge options require string keys")
    if not all(isinstance(value, str) for value in copied.values()):
        raise TypeError("bridge options require string values")
    return MappingProxyType({str(key): str(value) for key, value in copied.items()})


@dataclass(frozen=True, slots=True)
class BridgeAttachRequest:
    """One live-attach request that makes no physical reset claim."""

    options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", _copy_string_options(self.options))


@dataclass(frozen=True, slots=True)
class BridgeResetRequest:
    """One deterministic reset request matching the versioned wire contract."""

    seed: int | None = None
    options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed is not None and (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed <= _MAX_UINT64
        ):
            raise ValueError("seed must be an unsigned 64-bit integer or None")
        object.__setattr__(self, "options", _copy_string_options(self.options))


@dataclass(frozen=True, slots=True)
class BridgeStepRequest:
    """One fenced action request sent to a bridge driver."""

    episode_id: UUID
    expected_step_id: int
    action: TensorTree
    action_id: str | None = None
    issued_at_ns: int | None = None
    deadline_ns: int | None = None
    quantum_ns: int | None = None
    hold_ns: int | None = None
    lease: InputLeaseToken | None = None
    cancellation_token: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        if (
            not isinstance(self.expected_step_id, int)
            or isinstance(self.expected_step_id, bool)
            or not 1 <= self.expected_step_id <= _MAX_UINT64
        ):
            raise ValueError("expected_step_id must be an unsigned non-zero 64-bit integer")
        object.__setattr__(self, "action", freeze_tree(self.action))
        if self.action_id is not None and (
            not isinstance(self.action_id, str) or not self.action_id or len(self.action_id) > 128
        ):
            raise ValueError("action_id must contain 1-128 characters or be None")
        if self.issued_at_ns is not None and (
            not isinstance(self.issued_at_ns, int)
            or isinstance(self.issued_at_ns, bool)
            or self.issued_at_ns < 0
        ):
            raise ValueError("issued_at_ns must be a non-negative integer or None")
        timing_values = (self.deadline_ns, self.quantum_ns, self.hold_ns)
        if any(value is not None for value in timing_values):
            if self.deadline_ns is None or self.quantum_ns is None:
                raise ValueError("deadline_ns and quantum_ns must be provided together")
            RealtimeStepTiming(
                deadline_ns=self.deadline_ns,
                quantum_ns=self.quantum_ns,
                hold_ns=self.hold_ns,
            )
        if self.lease is not None and not isinstance(self.lease, InputLeaseToken):
            raise TypeError("lease must be an InputLeaseToken or None")
        if self.cancellation_token is not None and not self.cancellation_token:
            raise ValueError("cancellation_token cannot be empty")


@dataclass(frozen=True, slots=True)
class BridgeResumeRequest:
    """Reconnect request carrying the last client-committed cursor."""

    episode_id: UUID
    last_committed_step_id: int
    target_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        if (
            not isinstance(self.last_committed_step_id, int)
            or isinstance(self.last_committed_step_id, bool)
            or self.last_committed_step_id < 0
        ):
            raise ValueError("last_committed_step_id must be a non-negative integer")
        if self.target_id is not None:
            if not isinstance(self.target_id, str):
                raise TypeError("target_id must be a string or None")
            if not self.target_id or len(self.target_id) > 128:
                raise ValueError("target_id must contain 1-128 characters or None")


@dataclass(frozen=True, slots=True)
class BridgeResumeResult:
    """Authoritative cursor plus an optional in-flight action verdict."""

    timestep: TimeStep
    committed_step_id: int
    reconciliation: ActionReconciliation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestep, TimeStep):
            raise TypeError("timestep must be a TimeStep")
        if (
            not isinstance(self.committed_step_id, int)
            or isinstance(self.committed_step_id, bool)
            or self.committed_step_id < 0
        ):
            raise ValueError("committed_step_id must be a non-negative integer")
        if self.timestep.step_id != self.committed_step_id:
            raise ValueError("committed_step_id must match timestep.step_id")
        if (
            self.reconciliation is not None
            and self.reconciliation.episode_id != self.timestep.episode_id
        ):
            raise ValueError("reconciliation episode_id must match timestep.episode_id")


class BridgeDriver(Protocol):
    """Transport-side port implemented by HTTP, JSONL, pipe, or native drivers."""

    def describe(self) -> EnvironmentSpec:
        """Return the remote runtime contract after transport authentication."""
        ...

    def reset(self, request: BridgeResetRequest) -> TimeStep:
        """Reset the runtime and return step zero."""
        ...

    def attach(self, request: BridgeAttachRequest) -> TimeStep:
        """Attach to an existing world without claiming a physical reset."""
        ...

    def step(self, request: BridgeStepRequest) -> TimeStep:
        """Apply one fenced action and return its authoritative post-state."""
        ...

    def resume(self, request: BridgeResumeRequest) -> BridgeResumeResult:
        """Reconcile an in-flight action and return the authoritative cursor."""
        ...

    def lease(self, request: InputLeaseRequest) -> InputLeaseReceipt:
        """Apply one explicit target-bound input lease operation."""
        ...

    def cancel(self, action_id: str) -> None:
        """Fence one obsolete action before provider dispatch."""
        ...

    def close(self) -> None:
        """Release transport resources and any owned input lease."""
        ...


class EnvironmentBridgeDriver:
    """Server-side bridge kernel for one local ``GameEnvironment``.

    A transport server can decode a request, call this driver, and encode the
    returned contract object.  Requests are serialized so adapters never need
    to reimplement episode and step fencing for each transport.
    """

    def __init__(self, environment: GameEnvironment) -> None:
        self._environment = (
            environment
            if isinstance(environment, ContractEnvironment)
            else ContractEnvironment(environment)
        )
        self._current: TimeStep | None = None
        self._closed = False
        self._lease_book = InputLeaseBook()
        self._cancelled_actions: set[str] = set()
        self._lock = RLock()

    def describe(self) -> EnvironmentSpec:
        with self._lock:
            self._ensure_open()
            return self._environment.spec

    def reset(self, request: BridgeResetRequest) -> TimeStep:
        with self._lock:
            self._ensure_open()
            result = self._environment.reset(seed=request.seed, options=request.options)
            if result.step_id != 0:
                raise ContractViolation(
                    f"bridge reset returned step {result.step_id}; expected step 0"
                )
            self._current = result
            self._lease_book = InputLeaseBook()
            self._cancelled_actions.clear()
            return result

    def attach(self, request: BridgeAttachRequest) -> TimeStep:
        with self._lock:
            self._ensure_open()
            result = self._environment.attach(options=request.options)
            if result.step_id != 0:
                raise ContractViolation(
                    f"bridge attach returned step {result.step_id}; expected step 0"
                )
            self._current = result
            self._lease_book = InputLeaseBook()
            self._cancelled_actions.clear()
            return result

    def step(self, request: BridgeStepRequest) -> TimeStep:
        with self._lock:
            self._ensure_open()
            current = self._current
            if current is None:
                raise ContractViolation("bridge driver step requires reset first")
            if request.episode_id != current.episode_id:
                raise ContractViolation("bridge request episode_id does not match current episode")
            expected_step_id = current.step_id + 1
            if request.expected_step_id != expected_step_id:
                raise ContractViolation(
                    "bridge request expected_step_id does not match current step; "
                    f"expected {expected_step_id}, received {request.expected_step_id}"
                )
            if request.action_id is not None and request.action_id in self._cancelled_actions:
                raise ContractViolation("realtime action was cancelled before provider dispatch")
            if request.deadline_ns is not None:
                if self._environment.spec.realtime_timing is None:
                    raise ContractViolation("bridge does not advertise realtime timing")
                if request.issued_at_ns is not None and (
                    time_ns() >= request.issued_at_ns + request.deadline_ns
                ):
                    raise ContractViolation("realtime action deadline expired before dispatch")
                try:
                    RealtimeStepTiming(
                        request.deadline_ns, request.quantum_ns or 0, request.hold_ns
                    ).validate_against(self._environment.spec.realtime_timing)
                except ValueError as error:
                    raise ContractViolation(str(error)) from error
            if request.lease is not None and not self._lease_book.authorize(request.lease):
                raise ContractViolation("realtime step lease is absent, stale, or mismatched")
            result = self._environment.step(request.action)
            if result.episode_id != current.episode_id:
                raise ContractViolation("bridge environment changed episode_id during step")
            if result.step_id != expected_step_id:
                raise ContractViolation(
                    f"bridge environment returned step {result.step_id}; "
                    f"expected step {expected_step_id}"
                )
            self._current = result
            return result

    def resume(self, request: BridgeResumeRequest) -> BridgeResumeResult:
        with self._lock:
            self._ensure_open()
            current = self._current
            if current is None:
                raise ContractViolation("bridge resume requires reset or attach first")
            if request.episode_id != current.episode_id:
                raise ContractViolation("bridge resume episode_id does not match current episode")
            if request.last_committed_step_id > current.step_id:
                raise ContractViolation("bridge resume cursor is ahead of the authoritative step")
            return BridgeResumeResult(timestep=current, committed_step_id=current.step_id)

    def lease(self, request: InputLeaseRequest) -> InputLeaseReceipt:
        with self._lock:
            self._ensure_open()
            return self._lease_book.apply(request)

    def cancel(self, action_id: str) -> None:
        with self._lock:
            self._ensure_open()
            if not action_id or len(action_id) > 128:
                raise ValueError("action_id must contain 1-128 characters")
            self._cancelled_actions.add(action_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._current = None
            self._lease_book = InputLeaseBook()
            self._cancelled_actions.clear()
            self._environment.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractViolation("bridge driver is closed")


class BridgeEnvironment(GameEnvironment):
    """Expose a transport driver through the standard ``GameEnvironment`` port.

    The environment never retries ``step`` because a transport failure can
    leave the action outcome unknown.  Drivers may reconnect for later
    observations, but repeating an action requires adapter-specific evidence.
    """

    def __init__(
        self,
        driver: BridgeDriver,
        *,
        protocol_version: str = "1.0",
        metadata_allowlist: Iterable[str] = (),
        required_capabilities: Iterable[str] = (),
    ) -> None:
        if not protocol_version:
            raise ValueError("protocol_version cannot be empty")
        self._driver = driver
        self._closed = False
        self._current: TimeStep | None = None
        try:
            described = driver.describe()
            if not isinstance(described, EnvironmentSpec):
                raise TypeError("bridge driver describe() must return EnvironmentSpec")
            if described.protocol_version != protocol_version:
                raise ContractViolation(
                    "bridge protocol version mismatch; "
                    f"expected {protocol_version}, received {described.protocol_version}"
                )
            required = frozenset(required_capabilities)
            if any(not capability for capability in required):
                raise ValueError("required capabilities cannot contain empty values")
            missing = sorted(required - described.capabilities)
            if missing:
                raise ContractViolation(f"bridge is missing required capabilities: {missing}")
            allowed = frozenset(metadata_allowlist)
            self._spec = EnvironmentSpec(
                environment_id=described.environment_id,
                observation=described.observation,
                action=described.action,
                reward=described.reward,
                done=described.done,
                action_mask=described.action_mask,
                protocol_version=described.protocol_version,
                capabilities=described.capabilities,
                realtime_timing=described.realtime_timing,
                metadata={
                    key: value for key, value in described.metadata.items() if key in allowed
                },
            )
        except Exception:
            self._closed = True
            driver.close()
            raise

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, object] | None = None
    ) -> TimeStep:
        self._ensure_open()
        request = BridgeResetRequest(
            seed=seed,
            options={} if options is None else _copy_string_options(options),
        )
        result = self._driver.reset(request)
        return self._accept_start(result, operation="reset")

    def attach(self, *, options: Mapping[str, object] | None = None) -> TimeStep:
        self._ensure_open()
        if "live-attach" not in self._spec.capabilities:
            raise ContractViolation("bridge does not declare live-attach capability")
        request = BridgeAttachRequest(
            options={} if options is None else _copy_string_options(options)
        )
        result = self._driver.attach(request)
        return self._accept_start(result, operation="attach")

    def step(self, action: TensorTree) -> TimeStep:
        self._ensure_open()
        current = self._current
        if current is None:
            raise ContractViolation("bridge step requires reset first")
        if current.done:
            raise ContractViolation("bridge episode is terminal; reset before stepping")
        self._spec.action.validate(action, path="action")
        expected_step_id = current.step_id + 1
        request = BridgeStepRequest(
            episode_id=current.episode_id,
            expected_step_id=expected_step_id,
            action=action,
        )
        result = self._driver.step(request)
        if not isinstance(result, TimeStep):
            raise TypeError("bridge driver step() must return TimeStep")
        if result.episode_id != current.episode_id:
            raise ContractViolation("bridge step returned a different episode_id")
        if result.step_id != expected_step_id:
            raise ContractViolation(
                f"bridge step returned step {result.step_id}; expected step {expected_step_id}"
            )
        self._current = result
        return result

    def resume(
        self,
        *,
        episode_id: UUID,
        last_committed_step_id: int,
        target_id: str | None = None,
    ) -> BridgeResumeResult:
        """Reconnect without replaying a mutating action."""

        self._ensure_open()
        if "reconnect-resume-v1" not in self._spec.capabilities:
            raise ContractViolation("bridge does not declare reconnect-resume-v1 capability")
        request = BridgeResumeRequest(
            episode_id=episode_id,
            last_committed_step_id=last_committed_step_id,
            target_id=target_id,
        )
        resume = getattr(self._driver, "resume", None)
        if resume is None:
            raise ContractViolation("bridge declares reconnect-resume-v1 but has no resume method")
        result = resume(request)
        if not isinstance(result, BridgeResumeResult):
            raise TypeError("bridge driver resume() must return BridgeResumeResult")
        if result.timestep.episode_id != request.episode_id:
            raise ContractViolation("bridge resume returned a different episode_id")
        if result.committed_step_id < request.last_committed_step_id:
            raise ContractViolation("bridge resume returned a cursor older than the client cursor")
        if self._current is not None and result.committed_step_id < self._current.step_id:
            raise ContractViolation("bridge resume would rewind the authoritative cursor")
        self._current = result.timestep
        return result

    def step_realtime(
        self,
        action: TensorTree,
        *,
        deadline_ns: int,
        quantum_ns: int,
        hold_ns: int | None = None,
        lease: InputLeaseToken | None = None,
        cancellation_token: str | None = None,
        action_id: str | None = None,
        issued_at_ns: int | None = None,
    ) -> TimeStep:
        """Apply one bounded realtime step without implicit retries."""

        self._ensure_open()
        current = self._current
        if current is None:
            raise ContractViolation("bridge step requires reset first")
        if current.done:
            raise ContractViolation("bridge episode is terminal; reset before stepping")
        self._spec.action.validate(action, path="action")
        if action_id is None:
            action_id = f"step-{current.step_id + 1}"
        issued_at_ns = time_ns() if issued_at_ns is None else issued_at_ns
        timing = RealtimeStepTiming(deadline_ns, quantum_ns, hold_ns)
        if self._spec.realtime_timing is None:
            raise ContractViolation("bridge does not advertise realtime timing")
        try:
            timing.validate_against(self._spec.realtime_timing)
        except ValueError as error:
            raise ContractViolation(str(error)) from error
        result = self._driver.step(
            BridgeStepRequest(
                episode_id=current.episode_id,
                expected_step_id=current.step_id + 1,
                action=action,
                action_id=action_id,
                issued_at_ns=issued_at_ns,
                deadline_ns=timing.deadline_ns,
                quantum_ns=timing.quantum_ns,
                hold_ns=timing.hold_ns,
                lease=lease,
                cancellation_token=cancellation_token,
            )
        )
        if not isinstance(result, TimeStep):
            raise TypeError("bridge driver step() must return TimeStep")
        if result.episode_id != current.episode_id or result.step_id != current.step_id + 1:
            raise ContractViolation("bridge realtime step returned a stale timestep")
        self._current = result
        return result

    def lease(self, request: InputLeaseRequest) -> InputLeaseReceipt:
        """Apply one lease operation through a driver that supports leases."""

        self._ensure_open()
        lease = getattr(self._driver, "lease", None)
        if lease is None:
            raise ContractViolation("bridge driver does not support input leases")
        result = lease(request)
        if not isinstance(result, InputLeaseReceipt):
            raise TypeError("bridge driver lease() must return InputLeaseReceipt")
        return result

    def cancel(self, action_id: str) -> None:
        """Fence an obsolete action through a driver that supports cancellation."""

        self._ensure_open()
        cancel = getattr(self._driver, "cancel", None)
        if cancel is None:
            raise ContractViolation("bridge driver does not support realtime cancellation")
        cancel(action_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._current = None
        self._driver.close()

    def _accept_start(self, result: TimeStep, *, operation: str) -> TimeStep:
        if not isinstance(result, TimeStep):
            raise TypeError(f"bridge driver {operation}() must return TimeStep")
        if result.step_id != 0:
            raise ContractViolation(
                f"bridge {operation} returned step {result.step_id}; expected step 0"
            )
        self._current = result
        return result

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractViolation("bridge environment is closed")


__all__ = [
    "BridgeAttachRequest",
    "BridgeDriver",
    "BridgeEnvironment",
    "BridgeResetRequest",
    "BridgeResumeRequest",
    "BridgeResumeResult",
    "BridgeStepRequest",
    "EnvironmentBridgeDriver",
]
