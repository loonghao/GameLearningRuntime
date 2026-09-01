"""Learner-neutral runtime evidence and artifact lineage contracts.

Adapters own game semantics and authoritative observation. These immutable
records only make settled route transitions, motion health, modal boundaries,
and cross-artifact identity explicit and reproducible.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

from game_learning_runtime.errors import ContractViolation

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _id(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a public identifier")
    return value


def _sha(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _seq(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _position(value: object) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("position must contain exactly three coordinates")
    return tuple(_finite(item, path="position[]") for item in value)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RouteTransitionEvidence:
    """Authoritative adapter evidence for one route edge observation."""

    episode_id: UUID
    route_id: str
    edge_id: str
    map_sha256: str
    game_image_sha256: str
    producer_state_seq: int
    settled: bool
    reached: bool
    position: tuple[float, float, float]
    yaw_deg: float
    heading_error_deg: float
    encounter_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        object.__setattr__(self, "route_id", _id(self.route_id, path="route_id"))
        object.__setattr__(self, "edge_id", _id(self.edge_id, path="edge_id"))
        object.__setattr__(self, "map_sha256", _sha(self.map_sha256, path="map_sha256"))
        object.__setattr__(
            self,
            "game_image_sha256",
            _sha(self.game_image_sha256, path="game_image_sha256"),
        )
        object.__setattr__(
            self,
            "producer_state_seq",
            _seq(self.producer_state_seq, path="producer_state_seq"),
        )
        if not isinstance(self.settled, bool) or not isinstance(self.reached, bool):
            raise TypeError("settled and reached must be booleans")
        object.__setattr__(self, "position", _position(self.position))
        yaw = _finite(self.yaw_deg, path="yaw_deg")
        if not -180.0 <= yaw < 180.0:
            raise ValueError("yaw_deg must be in [-180, 180)")
        object.__setattr__(self, "yaw_deg", yaw)
        heading = _finite(self.heading_error_deg, path="heading_error_deg")
        if not 0.0 <= heading <= 180.0:
            raise ValueError("heading_error_deg must be in [0, 180]")
        object.__setattr__(self, "heading_error_deg", heading)
        if self.encounter_id is not None:
            object.__setattr__(self, "encounter_id", _id(self.encounter_id, path="encounter_id"))

    @property
    def edge_succeeded(self) -> bool:
        """Whether the adapter has enough settled evidence to confirm the edge."""

        return self.settled and self.reached


@dataclass(frozen=True, slots=True)
class RouteHealthTelemetry:
    """Read-only motion health counters; no recovery action is implied."""

    episode_id: UUID
    route_id: str
    producer_state_seq: int
    displacement_m: float
    heading_delta_deg: float
    stall_ticks: int = 0
    oscillation_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        object.__setattr__(self, "route_id", _id(self.route_id, path="route_id"))
        object.__setattr__(
            self,
            "producer_state_seq",
            _seq(self.producer_state_seq, path="producer_state_seq"),
        )
        displacement = _finite(self.displacement_m, path="displacement_m")
        if displacement < 0:
            raise ValueError("displacement_m cannot be negative")
        object.__setattr__(self, "displacement_m", displacement)
        heading = _finite(self.heading_delta_deg, path="heading_delta_deg")
        if not 0.0 <= heading <= 180.0:
            raise ValueError("heading_delta_deg must be in [0, 180]")
        object.__setattr__(self, "heading_delta_deg", heading)
        object.__setattr__(self, "stall_ticks", _seq(self.stall_ticks, path="stall_ticks"))
        object.__setattr__(
            self, "oscillation_count", _seq(self.oscillation_count, path="oscillation_count")
        )

    @property
    def stalled(self) -> bool:
        return self.stall_ticks > 0

    @property
    def oscillating(self) -> bool:
        return self.oscillation_count > 0

    @property
    def requires_observation(self) -> bool:
        return self.stalled or self.oscillating


class ModalState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ModalNavigationBoundary:
    """Authoritative modal state that gates navigation pause/resume."""

    episode_id: UUID
    modal_id: str
    state: ModalState
    producer_state_seq: int
    authoritative: bool
    previous: ModalNavigationBoundary | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        object.__setattr__(self, "modal_id", _id(self.modal_id, path="modal_id"))
        if not isinstance(self.state, ModalState):
            object.__setattr__(self, "state", ModalState(self.state))
        object.__setattr__(
            self,
            "producer_state_seq",
            _seq(self.producer_state_seq, path="producer_state_seq"),
        )
        if not isinstance(self.authoritative, bool):
            raise TypeError("authoritative must be a boolean")
        if self.previous is not None:
            if (
                self.previous.episode_id != self.episode_id
                or self.previous.modal_id != self.modal_id
            ):
                raise ContractViolation("modal boundary previous identity does not match")
            if self.producer_state_seq <= self.previous.producer_state_seq:
                raise ContractViolation("modal boundary producer_state_seq must increase")

    @property
    def navigation_paused(self) -> bool:
        return self.state is ModalState.OPEN or not self.authoritative


@dataclass(frozen=True, slots=True)
class ArtifactLineage:
    """Stable association across trajectory, recording, route, and build identity."""

    episode_id: UUID
    trajectory_id: str
    recording_id: str
    route_id: str
    map_sha256: str
    game_image_sha256: str
    encounter_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        for field_name in ("trajectory_id", "recording_id", "route_id"):
            object.__setattr__(self, field_name, _id(getattr(self, field_name), path=field_name))
        object.__setattr__(self, "map_sha256", _sha(self.map_sha256, path="map_sha256"))
        object.__setattr__(
            self,
            "game_image_sha256",
            _sha(self.game_image_sha256, path="game_image_sha256"),
        )
        if self.encounter_id is not None:
            object.__setattr__(self, "encounter_id", _id(self.encounter_id, path="encounter_id"))

    def to_mapping(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "episode_id": str(self.episode_id),
            "trajectory_id": self.trajectory_id,
            "recording_id": self.recording_id,
            "route_id": self.route_id,
            "map_sha256": self.map_sha256,
            "game_image_sha256": self.game_image_sha256,
        }
        if self.encounter_id is not None:
            value["encounter_id"] = self.encounter_id
        return value


__all__ = [
    "ArtifactLineage",
    "ModalNavigationBoundary",
    "ModalState",
    "RouteHealthTelemetry",
    "RouteTransitionEvidence",
]
