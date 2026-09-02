"""Portable, exact-environment snapshots of observed entities and advisory routes."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from time import time_ns
from types import MappingProxyType
from typing import Any

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.run_store import (
    RouteWaypoint,
    SpatialEntity,
    SpatialRoute,
    TrainingStore,
)
from game_learning_runtime.training import KnowledgeAuthority

SPATIAL_KNOWLEDGE_SCHEMA_VERSION = "glr.spatial-knowledge.v1"
SPATIAL_KNOWLEDGE_V2_SCHEMA_VERSION = "glr.spatial-knowledge.v2"
_GRAPH_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_MAX_GRAPH_OBJECTS = 100_000


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    return value


def _sequence(value: object, *, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{path} must be an array")
    return value


def _fields(value: Mapping[str, Any], expected: frozenset[str], *, path: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(f"{path} has missing={missing} and unexpected={unexpected} fields")


def _entity_mapping(entity: SpatialEntity) -> dict[str, object]:
    return {
        "environment_id": entity.environment_id,
        "world_id": entity.world_id,
        "entity_id": entity.entity_id,
        "kind": entity.kind,
        "label": entity.label,
        "position": list(entity.position),
        "coordinate_frame": entity.coordinate_frame,
        "authority": entity.authority.value,
        "confidence": entity.confidence,
        "observed_at_ns": entity.observed_at_ns,
        "source_run_id": entity.source_run_id,
        "metadata": dict(entity.metadata),
    }


def _entity_from_mapping(value: Mapping[str, Any]) -> SpatialEntity:
    _fields(
        value,
        frozenset(
            {
                "environment_id",
                "world_id",
                "entity_id",
                "kind",
                "label",
                "position",
                "coordinate_frame",
                "authority",
                "confidence",
                "observed_at_ns",
                "source_run_id",
                "metadata",
            }
        ),
        path="spatial entity",
    )
    return SpatialEntity(
        environment_id=value["environment_id"],
        world_id=value["world_id"],
        entity_id=value["entity_id"],
        kind=value["kind"],
        label=value["label"],
        position=tuple(_sequence(value["position"], path="spatial entity.position")),
        coordinate_frame=value["coordinate_frame"],
        authority=KnowledgeAuthority(value["authority"]),
        confidence=value["confidence"],
        observed_at_ns=value["observed_at_ns"],
        source_run_id=value["source_run_id"],
        metadata=_mapping(value["metadata"], path="spatial entity.metadata"),
    )


def _route_mapping(route: SpatialRoute) -> dict[str, object]:
    return {
        "environment_id": route.environment_id,
        "world_id": route.world_id,
        "route_id": route.route_id,
        "name": route.name,
        "from_entity_id": route.from_entity_id,
        "to_entity_id": route.to_entity_id,
        "coordinate_frame": route.coordinate_frame,
        "confidence": route.confidence,
        "verified_at_ns": route.verified_at_ns,
        "source_run_id": route.source_run_id,
        "waypoints": [
            {
                "index": waypoint.index,
                "position": list(waypoint.position),
                "tolerance": waypoint.tolerance,
                "label": waypoint.label,
            }
            for waypoint in route.waypoints
        ],
        "metadata": dict(route.metadata),
    }


def _route_from_mapping(value: Mapping[str, Any]) -> SpatialRoute:
    _fields(
        value,
        frozenset(
            {
                "environment_id",
                "world_id",
                "route_id",
                "name",
                "from_entity_id",
                "to_entity_id",
                "coordinate_frame",
                "confidence",
                "verified_at_ns",
                "source_run_id",
                "waypoints",
                "metadata",
            }
        ),
        path="spatial route",
    )
    waypoints: list[RouteWaypoint] = []
    for item in _sequence(value["waypoints"], path="spatial route.waypoints"):
        waypoint = _mapping(item, path="spatial route.waypoints[]")
        _fields(
            waypoint,
            frozenset({"index", "position", "tolerance", "label"}),
            path="spatial waypoint",
        )
        waypoints.append(
            RouteWaypoint(
                index=waypoint["index"],
                position=tuple(_sequence(waypoint["position"], path="spatial waypoint.position")),
                tolerance=waypoint["tolerance"],
                label=waypoint["label"],
            )
        )
    return SpatialRoute(
        environment_id=value["environment_id"],
        world_id=value["world_id"],
        route_id=value["route_id"],
        name=value["name"],
        from_entity_id=value["from_entity_id"],
        to_entity_id=value["to_entity_id"],
        coordinate_frame=value["coordinate_frame"],
        confidence=value["confidence"],
        verified_at_ns=value["verified_at_ns"],
        source_run_id=value["source_run_id"],
        waypoints=tuple(waypoints),
        metadata=_mapping(value["metadata"], path="spatial route.metadata"),
    )


@dataclass(frozen=True, slots=True)
class SpatialKnowledgeBundle:
    environment_id: str
    protocol_version: str
    exported_at_ns: int
    entities: tuple[SpatialEntity, ...]
    routes: tuple[SpatialRoute, ...]
    schema_version: str = SPATIAL_KNOWLEDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPATIAL_KNOWLEDGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported spatial knowledge schema: {self.schema_version!r}")
        if not isinstance(self.protocol_version, str) or not self.protocol_version:
            raise ValueError("spatial knowledge protocol_version cannot be empty")
        if (
            not isinstance(self.exported_at_ns, int)
            or isinstance(self.exported_at_ns, bool)
            or self.exported_at_ns < 0
        ):
            raise ValueError("spatial knowledge exported_at_ns must be non-negative")
        if len(self.entities) > 100_000 or len(self.routes) > 100_000:
            raise ValueError("spatial knowledge exceeds the bounded object count")
        if any(entity.environment_id != self.environment_id for entity in self.entities):
            raise ValueError("spatial entity environment_id differs from its bundle")
        if any(route.environment_id != self.environment_id for route in self.routes):
            raise ValueError("spatial route environment_id differs from its bundle")

    @classmethod
    def from_store(
        cls,
        store: TrainingStore,
        *,
        environment_id: str,
        protocol_version: str,
    ) -> SpatialKnowledgeBundle:
        return cls(
            environment_id=environment_id,
            protocol_version=protocol_version,
            exported_at_ns=time_ns(),
            entities=store.list_entities(environment_id=environment_id),
            routes=store.list_routes(environment_id=environment_id),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpatialKnowledgeBundle:
        _fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "environment_id",
                    "protocol_version",
                    "exported_at_ns",
                    "entities",
                    "routes",
                }
            ),
            path="spatial knowledge",
        )
        return cls(
            schema_version=value["schema_version"],
            environment_id=value["environment_id"],
            protocol_version=value["protocol_version"],
            exported_at_ns=value["exported_at_ns"],
            entities=tuple(
                _entity_from_mapping(_mapping(item, path="spatial knowledge.entities[]"))
                for item in _sequence(value["entities"], path="spatial knowledge.entities")
            ),
            routes=tuple(
                _route_from_mapping(_mapping(item, path="spatial knowledge.routes[]"))
                for item in _sequence(value["routes"], path="spatial knowledge.routes")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "protocol_version": self.protocol_version,
            "exported_at_ns": self.exported_at_ns,
            "entities": [_entity_mapping(entity) for entity in self.entities],
            "routes": [_route_mapping(route) for route in self.routes],
        }

    def import_into(
        self,
        store: TrainingStore,
        *,
        environment_id: str,
        protocol_version: str,
        source_run_id: str,
    ) -> tuple[int, int]:
        if self.environment_id != environment_id:
            raise ContractViolation("spatial knowledge environment_id does not match")
        if self.protocol_version != protocol_version:
            raise ContractViolation("spatial knowledge protocol_version does not match")
        for entity in self.entities:
            store.upsert_entity(
                SpatialEntity(
                    environment_id=entity.environment_id,
                    world_id=entity.world_id,
                    entity_id=entity.entity_id,
                    kind=entity.kind,
                    label=entity.label,
                    position=entity.position,
                    coordinate_frame=entity.coordinate_frame,
                    authority=KnowledgeAuthority.ADVISORY,
                    confidence=entity.confidence,
                    observed_at_ns=entity.observed_at_ns,
                    source_run_id=source_run_id,
                    metadata={
                        **dict(entity.metadata),
                        "imported_authority": entity.authority.value,
                        "imported_source_run_id": entity.source_run_id,
                    },
                )
            )
        for route in self.routes:
            store.record_route(
                SpatialRoute(
                    environment_id=route.environment_id,
                    world_id=route.world_id,
                    route_id=route.route_id,
                    name=route.name,
                    from_entity_id=route.from_entity_id,
                    to_entity_id=route.to_entity_id,
                    coordinate_frame=route.coordinate_frame,
                    confidence=route.confidence,
                    verified_at_ns=route.verified_at_ns,
                    source_run_id=source_run_id,
                    waypoints=route.waypoints,
                    metadata={
                        **dict(route.metadata),
                        "imported_source_run_id": route.source_run_id,
                        "advisory": True,
                    },
                )
            )
        return len(self.entities), len(self.routes)


class TraversabilityStatus(str, Enum):
    """Advisory status of a directed edge at a point in time."""

    UNKNOWN = "unknown"
    TRAVERSABLE = "traversable"
    BLOCKED = "blocked"
    STALE = "stale"


class SpatialHazard(str, Enum):
    """Portable blocker reasons; none of these grants action authority."""

    DYNAMIC_HAZARD = "dynamic-hazard"
    INSUFFICIENT_CLEARANCE = "insufficient-clearance"
    NO_NAV_PROJECTION = "no-nav-projection"
    STEEP_SLOPE = "steep-slope"
    GEOMETRY_BLOCKED = "geometry-blocked"
    TRANSIENT_FAILURE = "transient-failure"
    UNKNOWN_BLOCKER = "unknown-blocker"


def _graph_identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _GRAPH_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must match {_GRAPH_IDENTIFIER.pattern!r}")
    return value


def _graph_position(value: object, *, path: str) -> tuple[float, float, float]:
    values = _sequence(value, path=path)
    if len(values) != 3:
        raise ValueError(f"{path} must contain exactly three coordinates")
    result = tuple(float(item) for item in values)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{path} coordinates must be finite")
    return result  # type: ignore[return-value]


def _graph_optional_position(value: object, *, path: str) -> tuple[float, float, float] | None:
    return None if value is None else _graph_position(value, path=path)


def _graph_non_negative(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _graph_float(value: object, *, path: str, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = f" greater than or equal to {minimum}" if minimum is not None else " finite"
        raise ValueError(f"{path} must be{suffix}")
    return result


def _graph_metadata(value: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    _mapping(value, path=path)
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain finite JSON data") from error
    if len(encoded.encode("utf-8")) > 1024 * 1024:
        raise ValueError(f"{path} exceeds the 1 MiB limit")
    return MappingProxyType(json.loads(encoded))


@dataclass(frozen=True, slots=True)
class SpatialFrameTransform:
    """Bounded transform between the frames used by a graph edge."""

    from_frame: str
    to_frame: str
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        _graph_identifier(self.from_frame, path="transform.from_frame")
        _graph_identifier(self.to_frame, path="transform.to_frame")
        object.__setattr__(
            self, "translation", _graph_position(self.translation, path="transform.translation")
        )
        rotation = tuple(float(item) for item in self.rotation_quaternion)
        if len(rotation) != 4 or any(not math.isfinite(item) for item in rotation):
            raise ValueError("transform.rotation_quaternion must contain four finite values")
        if math.isclose(sum(item * item for item in rotation), 0.0):
            raise ValueError("transform.rotation_quaternion cannot be zero")
        object.__setattr__(self, "rotation_quaternion", rotation)
        scale = tuple(
            _graph_float(item, path="transform.scale[]", minimum=0.0) for item in self.scale
        )
        if len(scale) != 3 or any(item == 0.0 for item in scale):
            raise ValueError("transform.scale must contain three positive values")
        object.__setattr__(self, "scale", scale)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpatialFrameTransform:
        _fields(
            value,
            frozenset({"from_frame", "to_frame", "translation", "rotation_quaternion", "scale"}),
            path="spatial transform",
        )
        return cls(
            from_frame=value["from_frame"],
            to_frame=value["to_frame"],
            translation=_graph_position(value["translation"], path="transform.translation"),
            rotation_quaternion=tuple(value["rotation_quaternion"]),
            scale=tuple(value["scale"]),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "translation": list(self.translation),
            "rotation_quaternion": list(self.rotation_quaternion),
            "scale": list(self.scale),
        }


@dataclass(frozen=True, slots=True)
class SpatialGraphNode:
    """A stable, non-executable 3D graph node."""

    node_id: str
    world_id: str
    position: tuple[float, float, float]
    coordinate_frame: str
    source_run_id: str
    ground_z: float | None = None
    nav_z: float | None = None
    observed_at_ns: int = 0
    authority: KnowledgeAuthority = KnowledgeAuthority.ADVISORY
    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for path, value in (
            ("node_id", self.node_id),
            ("world_id", self.world_id),
            ("coordinate_frame", self.coordinate_frame),
            ("source_run_id", self.source_run_id),
        ):
            _graph_identifier(value, path=f"spatial node.{path}")
        object.__setattr__(self, "position", _graph_position(self.position, path="node.position"))
        for path in ("ground_z", "nav_z"):
            value = getattr(self, path)
            if value is not None:
                object.__setattr__(self, path, _graph_float(value, path=f"node.{path}"))
        _graph_non_negative(self.observed_at_ns, path="node.observed_at_ns")
        object.__setattr__(self, "authority", KnowledgeAuthority(self.authority))
        confidence = _graph_float(self.confidence, path="node.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("node.confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        frozen = _graph_metadata(self.metadata, path="node.metadata")
        object.__setattr__(self, "metadata", frozen)

    @property
    def vertical_delta(self) -> float | None:
        if self.ground_z is None or self.nav_z is None:
            return None
        return self.nav_z - self.ground_z

    def to_mapping(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "world_id": self.world_id,
            "position": list(self.position),
            "coordinate_frame": self.coordinate_frame,
            "ground_z": self.ground_z,
            "nav_z": self.nav_z,
            "observed_at_ns": self.observed_at_ns,
            "source_run_id": self.source_run_id,
            "authority": self.authority.value,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpatialGraphNode:
        _fields(
            value,
            frozenset(
                {
                    "node_id",
                    "world_id",
                    "position",
                    "coordinate_frame",
                    "ground_z",
                    "nav_z",
                    "observed_at_ns",
                    "source_run_id",
                    "authority",
                    "confidence",
                    "metadata",
                }
            ),
            path="spatial node",
        )
        return cls(
            node_id=value["node_id"],
            world_id=value["world_id"],
            position=_graph_position(value["position"], path="node.position"),
            coordinate_frame=value["coordinate_frame"],
            source_run_id=value["source_run_id"],
            ground_z=value["ground_z"],
            nav_z=value["nav_z"],
            observed_at_ns=value["observed_at_ns"],
            authority=KnowledgeAuthority(value["authority"]),
            confidence=value["confidence"],
            metadata=_mapping(value["metadata"], path="node.metadata"),
        )


@dataclass(frozen=True, slots=True)
class NegativeTraversalEvidence:
    """A provenance-bound failed traversal observation."""

    reason: SpatialHazard
    observed_at_ns: int
    source_run_id: str
    expires_at_ns: int | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", SpatialHazard(self.reason))
        _graph_non_negative(self.observed_at_ns, path="negative evidence.observed_at_ns")
        _graph_identifier(self.source_run_id, path="negative evidence.source_run_id")
        if self.expires_at_ns is not None:
            _graph_non_negative(self.expires_at_ns, path="negative evidence.expires_at_ns")
            if self.expires_at_ns < self.observed_at_ns:
                raise ValueError("negative evidence.expires_at_ns cannot precede observed_at_ns")
        if self.detail is not None and (not self.detail.strip() or len(self.detail) > 512):
            raise ValueError("negative evidence.detail must be bounded text")

    def to_mapping(self) -> dict[str, object]:
        return {
            "reason": self.reason.value,
            "observed_at_ns": self.observed_at_ns,
            "source_run_id": self.source_run_id,
            "expires_at_ns": self.expires_at_ns,
            "detail": self.detail,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NegativeTraversalEvidence:
        _fields(
            value,
            frozenset({"reason", "observed_at_ns", "source_run_id", "expires_at_ns", "detail"}),
            path="negative evidence",
        )
        return cls(
            reason=SpatialHazard(value["reason"]),
            observed_at_ns=value["observed_at_ns"],
            source_run_id=value["source_run_id"],
            expires_at_ns=value["expires_at_ns"],
            detail=value["detail"],
        )


@dataclass(frozen=True, slots=True)
class SpatialGraphEdge:
    """A directed, advisory traversal observation between two graph nodes."""

    edge_id: str
    world_id: str
    from_node_id: str
    to_node_id: str
    coordinate_frame: str
    source_run_id: str
    passability: TraversabilityStatus = TraversabilityStatus.UNKNOWN
    cost: float | None = None
    success_count: int = 0
    failure_count: int = 0
    last_verified_at_ns: int = 0
    expires_at_ns: int | None = None
    ground_projection: tuple[float, float, float] | None = None
    nav_projection: tuple[float, float, float] | None = None
    vertical_delta: float | None = None
    slope: float | None = None
    clearance: float | None = None
    hazard_reasons: tuple[SpatialHazard, ...] = ()
    negative_evidence: tuple[NegativeTraversalEvidence, ...] = ()
    transform: SpatialFrameTransform | None = None
    authority: KnowledgeAuthority = KnowledgeAuthority.ADVISORY
    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for path, value in (
            ("edge_id", self.edge_id),
            ("world_id", self.world_id),
            ("from_node_id", self.from_node_id),
            ("to_node_id", self.to_node_id),
            ("coordinate_frame", self.coordinate_frame),
            ("source_run_id", self.source_run_id),
        ):
            _graph_identifier(value, path=f"spatial edge.{path}")
        if self.from_node_id == self.to_node_id:
            raise ValueError("spatial edge cannot connect a node to itself")
        object.__setattr__(self, "passability", TraversabilityStatus(self.passability))
        if self.cost is not None:
            object.__setattr__(self, "cost", _graph_float(self.cost, path="edge.cost", minimum=0.0))
        for path in ("success_count", "failure_count", "last_verified_at_ns"):
            _graph_non_negative(getattr(self, path), path=f"edge.{path}")
        if self.expires_at_ns is not None:
            _graph_non_negative(self.expires_at_ns, path="edge.expires_at_ns")
            if self.expires_at_ns < self.last_verified_at_ns:
                raise ValueError("edge.expires_at_ns cannot precede last_verified_at_ns")
        for path in ("ground_projection", "nav_projection"):
            value = getattr(self, path)
            if value is not None:
                object.__setattr__(self, path, _graph_position(value, path=f"edge.{path}"))
        for path in ("vertical_delta", "slope", "clearance"):
            value = getattr(self, path)
            if value is not None:
                object.__setattr__(
                    self,
                    path,
                    _graph_float(
                        value,
                        path=f"edge.{path}",
                        minimum=0.0 if path != "vertical_delta" else None,
                    ),
                )
        if self.slope is not None and self.slope > 90.0:
            raise ValueError("edge.slope must be at most 90 degrees")
        object.__setattr__(
            self,
            "hazard_reasons",
            tuple(SpatialHazard(item) for item in self.hazard_reasons),
        )
        evidence = tuple(self.negative_evidence)
        if any(not isinstance(item, NegativeTraversalEvidence) for item in evidence):
            raise TypeError("edge.negative_evidence must contain NegativeTraversalEvidence values")
        object.__setattr__(self, "negative_evidence", evidence)
        if self.transform is not None and not isinstance(self.transform, SpatialFrameTransform):
            raise TypeError("edge.transform must be a SpatialFrameTransform")
        object.__setattr__(self, "authority", KnowledgeAuthority(self.authority))
        confidence = _graph_float(self.confidence, path="edge.confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("edge.confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        frozen = _graph_metadata(self.metadata, path="edge.metadata")
        object.__setattr__(self, "metadata", frozen)

    def status_at(self, observed_at_ns: int) -> TraversabilityStatus:
        """Resolve expiry and active negative evidence without granting action authority."""

        _graph_non_negative(observed_at_ns, path="observed_at_ns")
        if self.expires_at_ns is not None and observed_at_ns >= self.expires_at_ns:
            return TraversabilityStatus.STALE
        if any(
            evidence.expires_at_ns is None or observed_at_ns < evidence.expires_at_ns
            for evidence in self.negative_evidence
        ):
            return TraversabilityStatus.BLOCKED
        if self.passability is TraversabilityStatus.STALE:
            return TraversabilityStatus.STALE
        return self.passability

    def to_mapping(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "world_id": self.world_id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "coordinate_frame": self.coordinate_frame,
            "source_run_id": self.source_run_id,
            "passability": self.passability.value,
            "cost": self.cost,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_verified_at_ns": self.last_verified_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "ground_projection": (
                None if self.ground_projection is None else list(self.ground_projection)
            ),
            "nav_projection": None if self.nav_projection is None else list(self.nav_projection),
            "vertical_delta": self.vertical_delta,
            "slope": self.slope,
            "clearance": self.clearance,
            "hazard_reasons": [item.value for item in self.hazard_reasons],
            "negative_evidence": [item.to_mapping() for item in self.negative_evidence],
            "transform": None if self.transform is None else self.transform.to_mapping(),
            "authority": self.authority.value,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpatialGraphEdge:
        _fields(
            value,
            frozenset(
                {
                    "edge_id",
                    "world_id",
                    "from_node_id",
                    "to_node_id",
                    "coordinate_frame",
                    "source_run_id",
                    "passability",
                    "cost",
                    "success_count",
                    "failure_count",
                    "last_verified_at_ns",
                    "expires_at_ns",
                    "ground_projection",
                    "nav_projection",
                    "vertical_delta",
                    "slope",
                    "clearance",
                    "hazard_reasons",
                    "negative_evidence",
                    "transform",
                    "authority",
                    "confidence",
                    "metadata",
                }
            ),
            path="spatial edge",
        )
        transform = value["transform"]
        return cls(
            edge_id=value["edge_id"],
            world_id=value["world_id"],
            from_node_id=value["from_node_id"],
            to_node_id=value["to_node_id"],
            coordinate_frame=value["coordinate_frame"],
            source_run_id=value["source_run_id"],
            passability=TraversabilityStatus(value["passability"]),
            cost=value["cost"],
            success_count=value["success_count"],
            failure_count=value["failure_count"],
            last_verified_at_ns=value["last_verified_at_ns"],
            expires_at_ns=value["expires_at_ns"],
            ground_projection=_graph_optional_position(
                value["ground_projection"], path="edge.ground_projection"
            ),
            nav_projection=_graph_optional_position(
                value["nav_projection"], path="edge.nav_projection"
            ),
            vertical_delta=value["vertical_delta"],
            slope=value["slope"],
            clearance=value["clearance"],
            hazard_reasons=tuple(
                SpatialHazard(item)
                for item in _sequence(value["hazard_reasons"], path="edge.hazard_reasons")
            ),
            negative_evidence=tuple(
                NegativeTraversalEvidence.from_mapping(
                    _mapping(item, path="edge.negative_evidence[]")
                )
                for item in _sequence(value["negative_evidence"], path="edge.negative_evidence")
            ),
            transform=None
            if transform is None
            else SpatialFrameTransform.from_mapping(_mapping(transform, path="edge.transform")),
            authority=KnowledgeAuthority(value["authority"]),
            confidence=value["confidence"],
            metadata=_mapping(value["metadata"], path="edge.metadata"),
        )


@dataclass(frozen=True, slots=True)
class SpatialKnowledgeGraph:
    """Versioned directed traversability graph with advisory query projections."""

    environment_id: str
    protocol_version: str
    exported_at_ns: int
    nodes: tuple[SpatialGraphNode, ...]
    edges: tuple[SpatialGraphEdge, ...]
    transforms: tuple[SpatialFrameTransform, ...] = ()
    schema_version: str = SPATIAL_KNOWLEDGE_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPATIAL_KNOWLEDGE_V2_SCHEMA_VERSION:
            raise ValueError(f"unsupported spatial knowledge schema: {self.schema_version!r}")
        _graph_identifier(self.environment_id, path="spatial graph.environment_id")
        if not isinstance(self.protocol_version, str) or not self.protocol_version:
            raise ValueError("spatial graph protocol_version cannot be empty")
        _graph_non_negative(self.exported_at_ns, path="spatial graph.exported_at_ns")
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        transforms = tuple(self.transforms)
        if len(nodes) > _MAX_GRAPH_OBJECTS or len(edges) > _MAX_GRAPH_OBJECTS:
            raise ValueError("spatial graph exceeds the bounded object count")
        if any(not isinstance(node, SpatialGraphNode) for node in nodes):
            raise TypeError("spatial graph.nodes must contain SpatialGraphNode values")
        if any(not isinstance(edge, SpatialGraphEdge) for edge in edges):
            raise TypeError("spatial graph.edges must contain SpatialGraphEdge values")
        if any(not isinstance(transform, SpatialFrameTransform) for transform in transforms):
            raise TypeError("spatial graph.transforms must contain SpatialFrameTransform values")
        node_ids = {node.node_id for node in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError("spatial graph.nodes contains duplicate node_id values")
        edge_ids = {edge.edge_id for edge in edges}
        if len(edge_ids) != len(edges):
            raise ValueError("spatial graph.edges contains duplicate edge_id values")
        if any(node.world_id == "" or node.source_run_id == "" for node in nodes):
            raise ValueError("spatial graph nodes require world and source run identifiers")
        if any(
            edge.from_node_id not in node_ids or edge.to_node_id not in node_ids for edge in edges
        ):
            raise ValueError("spatial graph edge references an unknown node")
        node_by_id = {node.node_id: node for node in nodes}
        if any(
            edge.world_id != node_by_id[edge.from_node_id].world_id
            or edge.world_id != node_by_id[edge.to_node_id].world_id
            for edge in edges
        ):
            raise ValueError("spatial graph edge crosses world boundaries")
        transform_pairs = {(item.from_frame, item.to_frame) for item in transforms}
        if len(transform_pairs) != len(transforms):
            raise ValueError("spatial graph.transforms contains duplicate frame pairs")
        if any(
            edge.transform is not None and edge.transform.from_frame != edge.coordinate_frame
            for edge in edges
        ):
            raise ValueError("spatial graph edge transform does not match coordinate_frame")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "transforms", transforms)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpatialKnowledgeGraph:
        _fields(
            value,
            frozenset(
                {
                    "schema_version",
                    "environment_id",
                    "protocol_version",
                    "exported_at_ns",
                    "nodes",
                    "edges",
                    "transforms",
                }
            ),
            path="spatial graph",
        )
        return cls(
            schema_version=value["schema_version"],
            environment_id=value["environment_id"],
            protocol_version=value["protocol_version"],
            exported_at_ns=value["exported_at_ns"],
            nodes=tuple(
                SpatialGraphNode.from_mapping(_mapping(item, path="spatial graph.nodes[]"))
                for item in _sequence(value["nodes"], path="spatial graph.nodes")
            ),
            edges=tuple(
                SpatialGraphEdge.from_mapping(_mapping(item, path="spatial graph.edges[]"))
                for item in _sequence(value["edges"], path="spatial graph.edges")
            ),
            transforms=tuple(
                SpatialFrameTransform.from_mapping(
                    _mapping(item, path="spatial graph.transforms[]")
                )
                for item in _sequence(value["transforms"], path="spatial graph.transforms")
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "protocol_version": self.protocol_version,
            "exported_at_ns": self.exported_at_ns,
            "nodes": [node.to_mapping() for node in self.nodes],
            "edges": [edge.to_mapping() for edge in self.edges],
            "transforms": [transform.to_mapping() for transform in self.transforms],
        }

    def edge_status(self, edge_id: str, *, observed_at_ns: int) -> TraversabilityStatus:
        for edge in self.edges:
            if edge.edge_id == edge_id:
                return edge.status_at(observed_at_ns)
        raise KeyError(f"unknown spatial edge: {edge_id}")

    def frontier_candidates(
        self,
        *,
        world_id: str,
        from_node_id: str,
        observed_at_ns: int,
        to_node_id: str | None = None,
        limit: int = 100,
    ) -> tuple[SpatialGraphEdge, ...]:
        """Return advisory outgoing edges, excluding blocked or stale evidence."""

        _graph_identifier(world_id, path="world_id")
        _graph_identifier(from_node_id, path="from_node_id")
        if to_node_id is not None:
            _graph_identifier(to_node_id, path="to_node_id")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        candidates = [
            edge
            for edge in self.edges
            if edge.world_id == world_id
            and edge.from_node_id == from_node_id
            and (to_node_id is None or edge.to_node_id == to_node_id)
            and edge.status_at(observed_at_ns)
            in (TraversabilityStatus.UNKNOWN, TraversabilityStatus.TRAVERSABLE)
        ]
        candidates.sort(
            key=lambda edge: (
                0 if edge.status_at(observed_at_ns) is TraversabilityStatus.TRAVERSABLE else 1,
                edge.cost is None,
                edge.cost if edge.cost is not None else math.inf,
                -edge.confidence,
                edge.edge_id,
            )
        )
        return tuple(candidates[:limit])

    def imported_advisory(self, *, source_run_id: str) -> SpatialKnowledgeGraph:
        """Bind imported nodes/edges to a new run while retaining negative evidence provenance."""

        _graph_identifier(source_run_id, path="source_run_id")
        nodes = tuple(
            replace(
                node,
                source_run_id=source_run_id,
                authority=KnowledgeAuthority.ADVISORY,
                metadata={
                    **dict(node.metadata),
                    "imported_source_run_id": node.source_run_id,
                    "advisory": True,
                },
            )
            for node in self.nodes
        )
        edges = tuple(
            replace(
                edge,
                source_run_id=source_run_id,
                authority=KnowledgeAuthority.ADVISORY,
                metadata={
                    **dict(edge.metadata),
                    "imported_source_run_id": edge.source_run_id,
                    "advisory": True,
                },
            )
            for edge in self.edges
        )
        return replace(self, nodes=nodes, edges=edges)


__all__ = [
    "SPATIAL_KNOWLEDGE_SCHEMA_VERSION",
    "SPATIAL_KNOWLEDGE_V2_SCHEMA_VERSION",
    "NegativeTraversalEvidence",
    "SpatialFrameTransform",
    "SpatialGraphEdge",
    "SpatialGraphNode",
    "SpatialHazard",
    "SpatialKnowledgeBundle",
    "SpatialKnowledgeGraph",
    "TraversabilityStatus",
]
