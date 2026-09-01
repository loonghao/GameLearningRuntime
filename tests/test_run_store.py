from __future__ import annotations

import math
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from game_learning_runtime.agent_goal import ResearchBundle, ResearchCategory
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.run_store import (
    RouteWaypoint,
    RunStatus,
    SpatialEntity,
    SpatialRoute,
    TrainingStore,
)
from game_learning_runtime.training import KnowledgeAuthority


def test_training_store_records_queryable_run_lifecycle_and_events(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        kind="training",
        metadata={"algorithm": "bc"},
    )

    event = store.append_event(
        run.run_id,
        kind="episode.started",
        payload={"seed": 7},
        episode_id="episode-1",
        step_id=0,
    )
    store.record_metric(run.run_id, name="reward.total", value=3.5, step_id=12)
    finished = store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)

    assert event.sequence_id == 1
    assert finished.status is RunStatus.SUCCEEDED
    assert store.get_run(run.run_id) == finished
    assert store.list_runs(environment_id="example.adventure-v1") == (finished,)
    assert store.list_events(run.run_id)[0].payload == {"seed": 7}
    assert store.list_metrics(run.run_id)[0].value == 3.5


def test_training_store_queries_observed_entities_and_advisory_routes(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        kind="training",
    )
    shrine = SpatialEntity(
        environment_id=run.environment_id,
        world_id="forest",
        entity_id="shrine.forest-1",
        kind="shrine",
        label="林外土地庙",
        position=(100.0, 25.0, 4.0),
        coordinate_frame="world",
        authority=KnowledgeAuthority.AUTHORITATIVE,
        confidence=1.0,
        observed_at_ns=1000,
        source_run_id=run.run_id,
    )
    store.upsert_entity(shrine)
    route = SpatialRoute(
        environment_id=run.environment_id,
        world_id="forest",
        route_id="route.spawn-to-shrine",
        name="出生点到林外土地庙",
        from_entity_id="spawn.main",
        to_entity_id=shrine.entity_id,
        coordinate_frame="world",
        confidence=0.9,
        verified_at_ns=1100,
        source_run_id=run.run_id,
        waypoints=(
            RouteWaypoint(index=0, position=(0.0, 0.0, 0.0), tolerance=2.0),
            RouteWaypoint(index=1, position=shrine.position, tolerance=3.0),
        ),
    )
    store.record_route(route)

    nearby = store.query_entities(
        environment_id=run.environment_id,
        world_id="forest",
        kind="shrine",
        near=(98.0, 25.0, 4.0),
        radius=5.0,
    )
    routes = store.query_routes(
        environment_id=run.environment_id,
        world_id="forest",
        to_entity_id=shrine.entity_id,
    )

    assert nearby == (shrine,)
    assert routes == (route,)


def test_training_store_binds_queryable_artifacts_to_exact_bytes(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        kind="training",
    )
    source = tmp_path / "capture.mp4"
    source.write_bytes(b"video")

    artifact = store.register_artifact(
        run.run_id,
        path="capture.mp4",
        source=source,
        role="review-video",
        media_type="video/mp4",
    )

    assert artifact.size_bytes == 5
    assert len(artifact.sha256) == 64
    assert store.list_artifacts(run.run_id) == (artifact,)


def test_training_store_queries_provenance_bound_research_by_reuse_scope(
    tmp_path: Path,
) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    bundle = ResearchBundle.from_mapping(
        {
            "schema_version": "glr.research-bundle.v1",
            "sources": [
                {
                    "source_id": "guide.dodge-video",
                    "media_type": "video-tutorial",
                    "url": "https://example.com/dodge",
                    "publisher": "Example Publisher",
                    "title": "Dodge timing tutorial",
                    "accessed_at": "2026-09-01T10:00:00Z",
                    "updated_at": None,
                    "summary": "Demonstrates a timing rule for a similar game family.",
                    "confidence": 0.8,
                    "volatility": "medium",
                }
            ],
            "findings": [
                {
                    "finding_id": "strategy.dodge-window",
                    "category": "strategy",
                    "status": "unverified",
                    "scope": "family",
                    "scope_id": "action-rpg",
                    "summary": "Observe the attack cue before committing to the dodge.",
                    "source_ids": ["guide.dodge-video"],
                    "tags": ["combat", "dodge"],
                    "confidence": 0.8,
                    "locator": "00:00:30-00:01:15",
                },
                {
                    "finding_id": "strategy.rejected",
                    "category": "strategy",
                    "status": "rejected",
                    "scope": "generic",
                    "scope_id": None,
                    "summary": "A disproven timing rule.",
                    "source_ids": ["guide.dodge-video"],
                    "tags": ["combat"],
                    "confidence": 0.2,
                    "locator": None,
                },
            ],
        }
    )

    store.upsert_research_bundle(bundle)
    findings = store.query_research(
        environment_id="example.adventure-v1",
        environment_family="action-rpg",
        tags=("dodge",),
    )

    assert findings == (bundle.findings[0],)
    assert store.get_research_sources(findings[0]) == bundle.sources


def test_training_store_fails_closed_on_schema_lifecycle_and_payload_errors(
    tmp_path: Path,
) -> None:
    incompatible = tmp_path / "incompatible.sqlite3"
    connection = sqlite3.connect(incompatible)
    connection.execute("PRAGMA user_version = 99")
    connection.close()
    with pytest.raises(ContractViolation, match="schema version"):
        TrainingStore(incompatible)

    store = TrainingStore(tmp_path / "runs.sqlite3")
    with pytest.raises(ValueError, match="run_id"):
        store.create_run(
            environment_id="example.adventure-v1",
            protocol_version="1.0",
            kind="training",
            run_id="INVALID",
        )
    with pytest.raises(ValueError, match="protocol_version"):
        store.create_run(
            environment_id="example.adventure-v1", protocol_version="", kind="training"
        )
    with pytest.raises(ValueError, match="finite JSON"):
        store.create_run(
            environment_id="example.adventure-v1",
            protocol_version="1.0",
            kind="training",
            metadata={"bad": math.nan},
        )
    with pytest.raises(ValueError, match="started_at_ns"):
        store.create_run(
            environment_id="example.adventure-v1",
            protocol_version="1.0",
            kind="training",
            started_at_ns=-1,
        )

    run = store.create_run(
        environment_id="example.adventure-v1", protocol_version="1.0", kind="training"
    )
    with pytest.raises(ValueError, match="step_id"):
        store.append_event(run.run_id, kind="event.test", payload={}, step_id=-1)
    with pytest.raises(ValueError, match="timestamp_ns"):
        store.append_event(run.run_id, kind="event.test", payload={}, timestamp_ns=-1)
    with pytest.raises(ValueError, match="finite"):
        store.record_metric(run.run_id, name="reward", value=math.inf)
    with pytest.raises(ValueError, match="step_id"):
        store.record_metric(run.run_id, name="reward", value=1, step_id=-1)
    with pytest.raises(ValueError, match="limit"):
        store.list_events(run.run_id, limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.list_metrics(run.run_id, limit=0)
    with pytest.raises(TypeError, match="exit_code"):
        store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=True)
    with pytest.raises(ValueError, match="finished_at_ns"):
        store.finish_run(
            run.run_id,
            status=RunStatus.SUCCEEDED,
            exit_code=0,
            finished_at_ns=-1,
        )
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
    with pytest.raises(ContractViolation, match="terminal"):
        store.append_event(run.run_id, kind="event.test", payload={})
    with pytest.raises(ContractViolation, match="terminal"):
        store.record_metric(run.run_id, name="reward", value=1)
    with pytest.raises(ContractViolation, match="already terminal"):
        store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
    with pytest.raises(KeyError, match="unknown"):
        store.get_run("run-missing")
    with pytest.raises(ValueError, match="limit"):
        store.list_runs(limit=0)


def test_spatial_contracts_and_queries_reject_ambiguous_coordinates(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1", protocol_version="1.0", kind="training"
    )
    entity = SpatialEntity(
        environment_id=run.environment_id,
        world_id="forest",
        entity_id="shrine.one",
        kind="shrine",
        label="Shrine",
        position=(1.0, 2.0, 3.0),
        coordinate_frame="world",
        authority=KnowledgeAuthority.AUTHORITATIVE,
        confidence=1.0,
        observed_at_ns=1,
        source_run_id=run.run_id,
    )
    for changes, message in (
        ({"label": ""}, "entity label"),
        ({"position": (1.0, 2.0)}, "three coordinates"),
        ({"confidence": 2}, "between 0 and 1"),
        ({"observed_at_ns": -1}, "non-negative"),
        ({"metadata": {"bad": math.nan}}, "finite JSON"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(entity, **changes)
    store.upsert_entity(entity)

    with pytest.raises(ValueError, match="radius is required"):
        store.query_entities(environment_id=run.environment_id, world_id="forest", near=(1, 2, 3))
    with pytest.raises(ValueError, match="near is required"):
        store.query_entities(environment_id=run.environment_id, world_id="forest", radius=1)
    with pytest.raises(ValueError, match="positive"):
        store.query_entities(
            environment_id=run.environment_id,
            world_id="forest",
            near=(1, 2, 3),
            radius=0,
        )
    with pytest.raises(ValueError, match="limit"):
        store.query_routes(environment_id=run.environment_id, world_id="forest", limit=0)

    waypoint = RouteWaypoint(index=0, position=(0, 0, 0), tolerance=1)
    with pytest.raises(ValueError, match="index"):
        replace(waypoint, index=-1)
    with pytest.raises(ValueError, match="positive"):
        replace(waypoint, tolerance=0)
    route = SpatialRoute(
        environment_id=run.environment_id,
        world_id="forest",
        route_id="route.one",
        name="Route",
        from_entity_id=None,
        to_entity_id=entity.entity_id,
        coordinate_frame="world",
        confidence=0.8,
        verified_at_ns=2,
        source_run_id=run.run_id,
        waypoints=(waypoint, RouteWaypoint(index=1, position=(1, 2, 3), tolerance=1)),
    )
    for changes, message in (
        ({"confidence": 2}, "between 0 and 1"),
        ({"verified_at_ns": -1}, "non-negative"),
        ({"waypoints": (waypoint,)}, "at least two"),
        (
            {
                "waypoints": (
                    waypoint,
                    RouteWaypoint(index=2, position=(1, 2, 3), tolerance=1),
                )
            },
            "contiguous",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            replace(route, **changes)


def test_artifact_and_research_queries_validate_inputs(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1", protocol_version="1.0", kind="training"
    )
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"data")
    with pytest.raises(ValueError, match="portable relative"):
        store.register_artifact(
            run.run_id,
            path="../artifact.bin",
            source=source,
            role="model",
            media_type="application/octet-stream",
        )
    with pytest.raises(ValueError, match="media_type"):
        store.register_artifact(
            run.run_id, path="artifact.bin", source=source, role="model", media_type=""
        )
    with pytest.raises(FileNotFoundError, match="regular"):
        store.register_artifact(
            run.run_id,
            path="missing.bin",
            source=tmp_path / "missing.bin",
            role="model",
            media_type="application/octet-stream",
        )

    bundle = ResearchBundle.from_mapping(
        {
            "schema_version": "glr.research-bundle.v1",
            "sources": [
                {
                    "source_id": "trace.one",
                    "media_type": "runtime-trace",
                    "url": "https://example.com/trace",
                    "publisher": "Runtime",
                    "title": "Trace",
                    "accessed_at": "2026-09-01T00:00:00Z",
                    "updated_at": None,
                    "summary": "An authoritative trace.",
                    "confidence": 1,
                    "volatility": "low",
                }
            ],
            "findings": [
                {
                    "finding_id": "mechanic.verified",
                    "category": "mechanic",
                    "status": "runtime-verified",
                    "scope": "environment",
                    "scope_id": run.environment_id,
                    "summary": "A verified mechanic.",
                    "source_ids": ["trace.one"],
                    "tags": ["mechanic"],
                    "confidence": 1,
                    "locator": "step-1",
                }
            ],
        }
    )
    store.upsert_research_bundle(bundle)
    assert (
        store.query_research(
            environment_id=run.environment_id,
            environment_family="action-rpg",
            category=ResearchCategory.MECHANIC,
            include_unverified=False,
        )
        == bundle.findings
    )
    with pytest.raises(ValueError, match="unique"):
        store.query_research(
            environment_id=run.environment_id,
            environment_family="action-rpg",
            tags=("mechanic", "mechanic"),
        )
    with pytest.raises(TypeError, match="ResearchFinding"):
        store.get_research_sources(object())  # type: ignore[arg-type]
