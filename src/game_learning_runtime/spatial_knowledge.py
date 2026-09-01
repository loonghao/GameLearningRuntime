"""Portable, exact-environment snapshots of observed entities and advisory routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import time_ns
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


__all__ = ["SPATIAL_KNOWLEDGE_SCHEMA_VERSION", "SpatialKnowledgeBundle"]
