from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.run_store import (
    RouteWaypoint,
    SpatialEntity,
    SpatialRoute,
    TrainingStore,
)
from game_learning_runtime.spatial_knowledge import SpatialKnowledgeBundle
from game_learning_runtime.training import KnowledgeAuthority


def test_spatial_knowledge_moves_only_between_exact_environment_contracts(
    tmp_path: Path,
) -> None:
    source_store = TrainingStore(tmp_path / "source.sqlite3")
    source_run = source_store.create_run(
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        kind="training",
    )
    entity = SpatialEntity(
        environment_id=source_run.environment_id,
        world_id="forest",
        entity_id="shrine.forest-1",
        kind="shrine",
        label="林外土地庙",
        position=(100.0, 25.0, 4.0),
        coordinate_frame="world",
        authority=KnowledgeAuthority.AUTHORITATIVE,
        confidence=1.0,
        observed_at_ns=1000,
        source_run_id=source_run.run_id,
    )
    source_store.upsert_entity(entity)
    source_store.record_route(
        SpatialRoute(
            environment_id=source_run.environment_id,
            world_id="forest",
            route_id="route.spawn-to-shrine",
            name="出生点到林外土地庙",
            from_entity_id="spawn.main",
            to_entity_id=entity.entity_id,
            coordinate_frame="world",
            confidence=0.9,
            verified_at_ns=1100,
            source_run_id=source_run.run_id,
            waypoints=(
                RouteWaypoint(index=0, position=(0.0, 0.0, 0.0), tolerance=2.0),
                RouteWaypoint(index=1, position=entity.position, tolerance=3.0),
            ),
        )
    )
    bundle = SpatialKnowledgeBundle.from_store(
        source_store,
        environment_id=source_run.environment_id,
        protocol_version=source_run.protocol_version,
    )
    restored = SpatialKnowledgeBundle.from_mapping(bundle.to_mapping())

    target_store = TrainingStore(tmp_path / "target.sqlite3")
    target_run = target_store.create_run(
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        kind="knowledge-import",
    )
    restored.import_into(
        target_store,
        environment_id=target_run.environment_id,
        protocol_version=target_run.protocol_version,
        source_run_id=target_run.run_id,
    )

    imported = target_store.query_entities(
        environment_id=target_run.environment_id, world_id="forest"
    )[0]
    assert imported.source_run_id == target_run.run_id
    assert imported.metadata["imported_source_run_id"] == source_run.run_id
    assert (
        target_store.query_routes(environment_id=target_run.environment_id, world_id="forest")[
            0
        ].route_id
        == "route.spawn-to-shrine"
    )
    with pytest.raises(ContractViolation, match="environment_id"):
        restored.import_into(
            target_store,
            environment_id="example.other-v1",
            protocol_version="1.0",
            source_run_id=target_run.run_id,
        )
    with pytest.raises(ContractViolation, match="protocol_version"):
        restored.import_into(
            target_store,
            environment_id=target_run.environment_id,
            protocol_version="2.0",
            source_run_id=target_run.run_id,
        )


def test_spatial_knowledge_rejects_invalid_snapshots(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1", protocol_version="1.0", kind="training"
    )
    bundle = SpatialKnowledgeBundle.from_store(
        store, environment_id=run.environment_id, protocol_version=run.protocol_version
    )
    with pytest.raises(ValueError, match="unsupported"):
        replace(bundle, schema_version="wrong")
    with pytest.raises(ValueError, match="protocol_version"):
        replace(bundle, protocol_version="")
    with pytest.raises(ValueError, match="exported_at_ns"):
        replace(bundle, exported_at_ns=-1)
    with pytest.raises(ValueError, match="missing"):
        SpatialKnowledgeBundle.from_mapping({"schema_version": "glr.spatial-knowledge.v1"})
    with pytest.raises(TypeError, match="array"):
        SpatialKnowledgeBundle.from_mapping(
            {
                **bundle.to_mapping(),
                "entities": "not-an-array",
            }
        )
