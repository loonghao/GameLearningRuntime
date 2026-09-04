"""Optional non-gameplay phase contracts for realtime environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic_ns
from types import MappingProxyType

from game_learning_runtime.errors import ContractViolation, GLRError

PHASE_SCHEMA_VERSION = "glr.environment-phase.v1"


class EnvironmentPhase(str, Enum):
    GAMEPLAY = "gameplay"
    MENU = "menu"
    LOADING = "loading"
    CUTSCENE = "cutscene"
    MODAL = "modal"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PhasePolicy:
    """Action, observation, training, and wall-clock policy for one phase."""

    training: bool = False
    actions_allowed: bool = False
    budget_ms: float | None = None
    observations_expected: bool = True

    def __post_init__(self) -> None:
        for name in ("training", "actions_allowed", "observations_expected"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.budget_ms is not None and self.budget_ms <= 0:
            raise ValueError("budget_ms must be positive or None")


def _default_policies() -> dict[EnvironmentPhase, PhasePolicy]:
    return {
        EnvironmentPhase.GAMEPLAY: PhasePolicy(training=True, actions_allowed=True),
        EnvironmentPhase.MENU: PhasePolicy(actions_allowed=True),
        EnvironmentPhase.LOADING: PhasePolicy(),
        EnvironmentPhase.CUTSCENE: PhasePolicy(observations_expected=False),
        EnvironmentPhase.MODAL: PhasePolicy(),
        EnvironmentPhase.UNAVAILABLE: PhasePolicy(),
        EnvironmentPhase.UNKNOWN: PhasePolicy(),
    }


class PhaseTimeoutError(GLRError):
    """Raised when a declared phase exceeds its wall-clock budget."""

    def __init__(self, phase: EnvironmentPhase, elapsed_ms: float, budget_ms: float) -> None:
        self.phase = phase
        self.elapsed_ms = elapsed_ms
        self.budget_ms = budget_ms
        super().__init__(
            f"phase_timeout:{phase.value} after {elapsed_ms:.1f} ms (budget {budget_ms:.1f} ms)"
        )


class PhaseActionError(GLRError):
    """Raised when an action is attempted in a phase that disallows actions."""


class PhaseObservationError(GLRError):
    """Raised when an expected observation is absent for the current phase."""


@dataclass(frozen=True, slots=True)
class PhaseStep:
    phase: EnvironmentPhase
    elapsed_ms: float
    training_eligible: bool
    observation_expected: bool
    phase_timeout: str | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "elapsed_ms": self.elapsed_ms,
            "training_eligible": self.training_eligible,
            "observation_expected": self.observation_expected,
            "phase_timeout": self.phase_timeout,
        }


@dataclass(frozen=True, slots=True)
class PhaseMetrics:
    """Per-phase time and step totals for a run manifest."""

    time_ms: Mapping[str, float] = field(default_factory=dict)
    steps: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.time_ms.values()):
            raise ValueError("phase time totals cannot be negative")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.steps.values()
        ):
            raise ValueError("phase step totals must be non-negative integers")
        object.__setattr__(self, "time_ms", MappingProxyType(dict(self.time_ms)))
        object.__setattr__(self, "steps", MappingProxyType(dict(self.steps)))

    def to_mapping(self) -> dict[str, object]:
        return {"time_ms": dict(self.time_ms), "steps": dict(self.steps)}


class PhaseMonitor:
    """Enforce declared phase policy and aggregate run metrics."""

    def __init__(
        self, policies: Mapping[EnvironmentPhase | str, PhasePolicy] | None = None
    ) -> None:
        resolved = _default_policies()
        if policies is not None:
            for phase, policy in policies.items():
                resolved[EnvironmentPhase(phase)] = policy
        self._policies = MappingProxyType(resolved)
        self._phase: EnvironmentPhase | None = None
        self._entered_ns: int | None = None
        self._last_ns: int | None = None
        self._time_ms: dict[str, float] = {}
        self._steps: dict[str, int] = {}

    @property
    def metrics(self) -> PhaseMetrics:
        return PhaseMetrics(self._time_ms, self._steps)

    def policy(self, phase: EnvironmentPhase | str) -> PhasePolicy:
        return self._policies[EnvironmentPhase(phase)]

    def allow_action(self, phase: EnvironmentPhase | str) -> None:
        resolved = EnvironmentPhase(phase)
        if not self.policy(resolved).actions_allowed:
            raise PhaseActionError(f"actions are not allowed in phase {resolved.value}")

    def record_step(
        self,
        phase: EnvironmentPhase | str,
        *,
        observation_present: bool = True,
        now_ns: int | None = None,
    ) -> PhaseStep:
        resolved = EnvironmentPhase(phase)
        if not isinstance(observation_present, bool):
            raise TypeError("observation_present must be bool")
        now = monotonic_ns() if now_ns is None else now_ns
        if not isinstance(now, int) or isinstance(now, bool) or now < 0:
            raise ValueError("now_ns must be a non-negative integer")
        if self._phase is None:
            self._phase = resolved
            self._entered_ns = now
            self._last_ns = now
        elif resolved != self._phase:
            self._accumulate(now)
            self._phase = resolved
            self._entered_ns = now
            self._last_ns = now
        assert self._entered_ns is not None
        elapsed = max(0.0, (now - self._entered_ns) / 1_000_000)
        policy = self.policy(resolved)
        if not observation_present and policy.observations_expected:
            raise PhaseObservationError(f"observation is absent in phase {resolved.value}")
        self._steps[resolved.value] = self._steps.get(resolved.value, 0) + 1
        if policy.budget_ms is not None and elapsed >= policy.budget_ms:
            raise PhaseTimeoutError(resolved, elapsed, policy.budget_ms)
        return PhaseStep(
            resolved,
            elapsed,
            policy.training,
            policy.observations_expected,
        )

    def _accumulate(self, now_ns: int) -> None:
        assert self._phase is not None and self._last_ns is not None
        elapsed = max(0.0, (now_ns - self._last_ns) / 1_000_000)
        self._time_ms[self._phase.value] = self._time_ms.get(self._phase.value, 0.0) + elapsed
        self._last_ns = now_ns


def validate_phase(value: EnvironmentPhase | str) -> EnvironmentPhase:
    try:
        return EnvironmentPhase(value)
    except ValueError as error:
        raise ContractViolation(f"unsupported environment phase: {value!r}") from error


__all__ = [
    "PHASE_SCHEMA_VERSION",
    "EnvironmentPhase",
    "PhaseActionError",
    "PhaseMetrics",
    "PhaseMonitor",
    "PhaseObservationError",
    "PhasePolicy",
    "PhaseStep",
    "PhaseTimeoutError",
    "validate_phase",
]
