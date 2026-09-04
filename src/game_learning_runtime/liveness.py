"""Runtime-owned observation liveness and progress declaration guards."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns

from game_learning_runtime.errors import ContractViolation, GLRError

LIVENESS_SCHEMA_VERSION = "glr.environment-liveness.v1"


@dataclass(frozen=True, slots=True)
class LivenessSnapshot:
    """Freshness telemetry derived by the runtime, never by an adapter."""

    observation_sequence: int
    observation_age_ms: float
    last_sequence_change_ms: float
    env_frozen: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.observation_sequence, int)
            or isinstance(self.observation_sequence, bool)
            or self.observation_sequence < 0
        ):
            raise ValueError("observation_sequence must be a non-negative integer")
        for name in ("observation_age_ms", "last_sequence_change_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
        if not isinstance(self.env_frozen, bool):
            raise TypeError("env_frozen must be bool")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": LIVENESS_SCHEMA_VERSION,
            "observation_sequence": self.observation_sequence,
            "observation_age_ms": self.observation_age_ms,
            "last_sequence_change_ms": self.last_sequence_change_ms,
            "env_frozen": self.env_frozen,
        }


class EnvironmentFrozenError(GLRError):
    """Raised when observations stop changing for the configured window."""

    def __init__(self, snapshot: LivenessSnapshot) -> None:
        if not snapshot.env_frozen:
            raise ValueError("an environment-frozen error requires a frozen snapshot")
        self.snapshot = snapshot
        super().__init__(
            "environment observation is frozen at sequence "
            f"{snapshot.observation_sequence} for {snapshot.last_sequence_change_ms:.1f} ms"
        )


class LivenessMonitor:
    """Track sequence freshness with a bounded, optional freeze threshold."""

    def __init__(self, *, freeze_after_ms: float | None = None) -> None:
        if freeze_after_ms is not None and freeze_after_ms <= 0:
            raise ValueError("freeze_after_ms must be positive or None")
        self.freeze_after_ms = freeze_after_ms
        self._sequence: int | None = None
        self._last_change_ns: int | None = None
        self._last_produced_ns: int | None = None

    def observe(
        self,
        observation_sequence: int,
        *,
        produced_at_ns: int | None = None,
        now_ns: int | None = None,
    ) -> LivenessSnapshot:
        if (
            not isinstance(observation_sequence, int)
            or isinstance(observation_sequence, bool)
            or observation_sequence < 0
        ):
            raise ValueError("observation_sequence must be a non-negative integer")
        now = monotonic_ns() if now_ns is None else now_ns
        if not isinstance(now, int) or isinstance(now, bool) or now < 0:
            raise ValueError("now_ns must be non-negative")
        if produced_at_ns is not None and (
            not isinstance(produced_at_ns, int) or produced_at_ns < 0
        ):
            raise ValueError("produced_at_ns must be a non-negative integer or None")
        if self._sequence is None or observation_sequence != self._sequence:
            self._sequence = observation_sequence
            self._last_change_ns = now
            self._last_produced_ns = now if produced_at_ns is None else produced_at_ns
        elif produced_at_ns is not None:
            self._last_produced_ns = produced_at_ns
        assert self._last_change_ns is not None
        assert self._last_produced_ns is not None
        age_ms = max(0.0, (now - self._last_produced_ns) / 1_000_000)
        change_ms = max(0.0, (now - self._last_change_ns) / 1_000_000)
        frozen = self.freeze_after_ms is not None and change_ms >= self.freeze_after_ms
        return LivenessSnapshot(observation_sequence, age_ms, change_ms, frozen)

    def require_live(self, snapshot: LivenessSnapshot) -> LivenessSnapshot:
        if snapshot.env_frozen:
            raise EnvironmentFrozenError(snapshot)
        return snapshot


@dataclass(frozen=True, slots=True)
class ProgressSignalDeclaration:
    """Adapter-declared progress field, kept distinct from liveness."""

    field: str
    liveness_field: str = "observation_sequence"

    def __post_init__(self) -> None:
        if not self.field or not self.liveness_field:
            raise ValueError("progress and liveness fields cannot be empty")
        if self.field == self.liveness_field:
            raise ContractViolation(
                f"progress signal cannot use the runtime liveness counter {self.liveness_field!r}"
            )


def validate_progress_field(field: str, *, liveness_field: str = "observation_sequence") -> str:
    """Validate a declaration at adapter startup and return its field."""

    return ProgressSignalDeclaration(field, liveness_field).field


__all__ = [
    "LIVENESS_SCHEMA_VERSION",
    "EnvironmentFrozenError",
    "LivenessMonitor",
    "LivenessSnapshot",
    "ProgressSignalDeclaration",
    "validate_progress_field",
]
