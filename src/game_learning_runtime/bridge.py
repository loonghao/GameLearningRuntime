"""Transport-neutral client boundary for authorized game runtime bridges.

Transport implementations own framing, authentication, deadlines, and
main-thread dispatch.  This module owns the learner-facing environment
lifecycle and request fencing shared by all transports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol
from uuid import UUID

from game_learning_runtime.contracts import TensorTree, TimeStep, freeze_tree
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.errors import ContractViolation
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

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._current = None
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
    "BridgeStepRequest",
    "EnvironmentBridgeDriver",
]
