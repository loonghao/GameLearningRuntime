from __future__ import annotations

from dataclasses import replace

import pytest

from game_learning_runtime import (
    KnowledgeAuthority,
    NegativeTraversalEvidence,
    SpatialFrameTransform,
    SpatialGraphEdge,
    SpatialGraphNode,
    SpatialHazard,
    SpatialKnowledgeGraph,
    TraversabilityStatus,
)


def _graph() -> SpatialKnowledgeGraph:
    nodes = (
        SpatialGraphNode(
            node_id="node.spawn",
            world_id="forest",
            position=(0.0, 0.0, 1.0),
            coordinate_frame="world",
            source_run_id="run.observed",
            ground_z=0.0,
            nav_z=1.0,
            observed_at_ns=100,
            authority=KnowledgeAuthority.AUTHORITATIVE,
            confidence=0.9,
        ),
        SpatialGraphNode(
            node_id="node.shrine",
            world_id="forest",
            position=(10.0, 0.0, 2.0),
            coordinate_frame="world",
            source_run_id="run.observed",
            ground_z=1.0,
            nav_z=2.0,
            observed_at_ns=100,
            authority=KnowledgeAuthority.AUTHORITATIVE,
            confidence=0.8,
        ),
        SpatialGraphNode(
            node_id="node.cliff",
            world_id="forest",
            position=(0.0, 10.0, 8.0),
            coordinate_frame="world",
            source_run_id="run.observed",
            ground_z=0.0,
            nav_z=8.0,
            observed_at_ns=100,
        ),
    )
    return SpatialKnowledgeGraph(
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        exported_at_ns=100,
        nodes=nodes,
        edges=(
            SpatialGraphEdge(
                edge_id="edge.spawn-shrine",
                world_id="forest",
                from_node_id="node.spawn",
                to_node_id="node.shrine",
                coordinate_frame="world",
                source_run_id="run.observed",
                passability=TraversabilityStatus.TRAVERSABLE,
                cost=10.0,
                success_count=2,
                last_verified_at_ns=100,
                expires_at_ns=500,
                ground_projection=(5.0, 0.0, 1.0),
                nav_projection=(5.0, 0.0, 1.5),
                vertical_delta=0.5,
                slope=5.0,
                clearance=2.0,
                transform=SpatialFrameTransform(
                    from_frame="world",
                    to_frame="nav",
                    translation=(0.0, 0.0, 1.0),
                ),
                authority=KnowledgeAuthority.AUTHORITATIVE,
                confidence=0.9,
            ),
            SpatialGraphEdge(
                edge_id="edge.spawn-cliff",
                world_id="forest",
                from_node_id="node.spawn",
                to_node_id="node.cliff",
                coordinate_frame="world",
                source_run_id="run.observed",
                passability=TraversabilityStatus.BLOCKED,
                failure_count=1,
                last_verified_at_ns=100,
                hazard_reasons=(SpatialHazard.STEEP_SLOPE,),
                negative_evidence=(
                    NegativeTraversalEvidence(
                        reason=SpatialHazard.STEEP_SLOPE,
                        observed_at_ns=120,
                        source_run_id="run.observed",
                        detail="nav projection exceeded the bounded slope",
                    ),
                ),
                slope=75.0,
                clearance=0.5,
                authority=KnowledgeAuthority.AUTHORITATIVE,
                confidence=0.95,
            ),
            SpatialGraphEdge(
                edge_id="edge.shrine-cliff",
                world_id="forest",
                from_node_id="node.shrine",
                to_node_id="node.cliff",
                coordinate_frame="world",
                source_run_id="run.observed",
                passability=TraversabilityStatus.UNKNOWN,
                last_verified_at_ns=100,
            ),
            SpatialGraphEdge(
                edge_id="edge.cliff-spawn",
                world_id="forest",
                from_node_id="node.cliff",
                to_node_id="node.spawn",
                coordinate_frame="world",
                source_run_id="run.observed",
                passability=TraversabilityStatus.TRAVERSABLE,
                last_verified_at_ns=100,
                expires_at_ns=150,
            ),
        ),
        transforms=(
            SpatialFrameTransform(
                from_frame="world",
                to_frame="nav",
                translation=(0.0, 0.0, 1.0),
            ),
        ),
    )


def test_spatial_graph_round_trips_3d_edges_and_expiry() -> None:
    graph = _graph()
    restored = SpatialKnowledgeGraph.from_mapping(graph.to_mapping())

    assert restored.to_mapping() == graph.to_mapping()
    assert (
        restored.edge_status("edge.spawn-shrine", observed_at_ns=200)
        is TraversabilityStatus.TRAVERSABLE
    )
    assert (
        restored.edge_status("edge.spawn-cliff", observed_at_ns=200) is TraversabilityStatus.BLOCKED
    )
    assert (
        restored.edge_status("edge.shrine-cliff", observed_at_ns=200)
        is TraversabilityStatus.UNKNOWN
    )
    assert (
        restored.edge_status("edge.cliff-spawn", observed_at_ns=200) is TraversabilityStatus.STALE
    )
    assert restored.edges[1].negative_evidence[0].source_run_id == "run.observed"


def test_negative_evidence_is_advisory_and_excludes_frontier_candidates() -> None:
    graph = _graph()
    candidates = graph.frontier_candidates(
        world_id="forest",
        from_node_id="node.spawn",
        observed_at_ns=200,
    )
    assert [edge.edge_id for edge in candidates] == ["edge.spawn-shrine"]

    imported = graph.imported_advisory(source_run_id="run.imported")
    assert imported.edges[1].authority is KnowledgeAuthority.ADVISORY
    assert imported.edges[1].metadata["advisory"] is True
    assert imported.edges[1].metadata["imported_source_run_id"] == "run.observed"
    assert imported.edges[1].negative_evidence[0].source_run_id == "run.observed"
    assert (
        imported.edge_status("edge.spawn-cliff", observed_at_ns=200) is TraversabilityStatus.BLOCKED
    )


def test_spatial_graph_rejects_ambiguous_or_unbounded_data() -> None:
    graph = _graph()
    with pytest.raises(ValueError, match="unsupported"):
        replace(graph, schema_version="glr.spatial-knowledge.v1")
    with pytest.raises(ValueError, match="unknown node"):
        SpatialKnowledgeGraph(
            environment_id=graph.environment_id,
            protocol_version=graph.protocol_version,
            exported_at_ns=graph.exported_at_ns,
            nodes=graph.nodes,
            edges=(replace(graph.edges[0], to_node_id="node.missing"),),
        )
    with pytest.raises(ValueError, match="expires_at_ns"):
        replace(graph.edges[0], expires_at_ns=99)
    with pytest.raises(ValueError, match="slope"):
        replace(graph.edges[0], slope=91.0)
