from __future__ import annotations

import math
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


def test_spatial_graph_validation_rejects_malformed_scalars_and_metadata() -> None:
    graph = _graph()
    node = graph.nodes[0]
    edge = graph.edges[0]

    with pytest.raises(TypeError, match="must be an array"):
        replace(node, position="0,0,0")
    with pytest.raises(ValueError, match="exactly three"):
        replace(node, position=(0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        replace(node, position=(0.0, 0.0, math.nan))
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(node, observed_at_ns=-1)
    with pytest.raises(TypeError, match="must be a number"):
        replace(node, ground_z="unknown")
    with pytest.raises(ValueError, match="finite"):
        replace(node, nav_z=math.inf)
    with pytest.raises(ValueError, match="between 0 and 1"):
        replace(node, confidence=1.1)
    with pytest.raises(ValueError, match="finite JSON"):
        replace(node, metadata={"score": math.nan})
    with pytest.raises(ValueError, match="1 MiB"):
        replace(node, metadata={"payload": "x" * (1024 * 1024)})
    with pytest.raises(ValueError, match="must match"):
        replace(node, node_id="Node/invalid")

    with pytest.raises(ValueError, match="non-negative integer"):
        replace(edge, success_count=-1)
    with pytest.raises(TypeError, match="must be a number"):
        replace(edge, cost="cheap")
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        replace(edge, cost=-1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        replace(edge, confidence=-0.1)
    with pytest.raises(TypeError, match="NegativeTraversalEvidence"):
        replace(edge, negative_evidence=("failed",))
    with pytest.raises(TypeError, match="SpatialFrameTransform"):
        replace(edge, transform="world->nav")
    with pytest.raises(ValueError, match="cannot connect"):
        replace(edge, to_node_id=edge.from_node_id)


def test_spatial_graph_validation_covers_transforms_evidence_and_status() -> None:
    graph = _graph()
    node = graph.nodes[0]

    assert replace(node, ground_z=1.0, nav_z=None).vertical_delta is None
    assert replace(node, ground_z=1.0, nav_z=3.5).vertical_delta == 2.5

    with pytest.raises(ValueError, match="four finite"):
        SpatialFrameTransform("world", "nav", rotation_quaternion=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="four finite"):
        SpatialFrameTransform("world", "nav", rotation_quaternion=(0.0, 0.0, 0.0, math.nan))
    with pytest.raises(ValueError, match="cannot be zero"):
        SpatialFrameTransform("world", "nav", rotation_quaternion=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="positive values"):
        SpatialFrameTransform("world", "nav", scale=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="positive values"):
        SpatialFrameTransform("world", "nav", scale=(1.0, 1.0))

    with pytest.raises(ValueError, match="non-negative integer"):
        NegativeTraversalEvidence(
            SpatialHazard.STEEP_SLOPE, observed_at_ns=1, source_run_id="run", expires_at_ns=-1
        )
    with pytest.raises(ValueError, match="cannot precede"):
        NegativeTraversalEvidence(
            SpatialHazard.STEEP_SLOPE, observed_at_ns=10, source_run_id="run", expires_at_ns=9
        )
    with pytest.raises(ValueError, match="bounded text"):
        NegativeTraversalEvidence(
            SpatialHazard.STEEP_SLOPE, observed_at_ns=1, source_run_id="run", detail=" "
        )

    stale = SpatialGraphEdge(
        edge_id="edge.stale",
        world_id="forest",
        from_node_id="node.spawn",
        to_node_id="node.shrine",
        coordinate_frame="world",
        source_run_id="run",
        passability=TraversabilityStatus.STALE,
    )
    assert stale.status_at(0) is TraversabilityStatus.STALE
    with pytest.raises(KeyError, match="unknown spatial edge"):
        graph.edge_status("edge.missing", observed_at_ns=0)
    with pytest.raises(ValueError, match="must match"):
        graph.frontier_candidates(
            world_id="forest",
            from_node_id="node.spawn",
            to_node_id="Node/invalid",
            observed_at_ns=0,
        )
    with pytest.raises(ValueError, match="between 1 and 1000"):
        graph.frontier_candidates(
            world_id="forest", from_node_id="node.spawn", observed_at_ns=0, limit=0
        )


def test_spatial_graph_validation_rejects_structural_inconsistencies() -> None:
    graph = _graph()
    node = graph.nodes[0]
    edge = graph.edges[0]
    transform = graph.transforms[0]

    with pytest.raises(ValueError, match="protocol_version"):
        replace(graph, protocol_version="")
    with pytest.raises(TypeError, match="nodes must contain"):
        replace(graph, nodes=(object(),))
    with pytest.raises(TypeError, match="edges must contain"):
        replace(graph, edges=(object(),))
    with pytest.raises(TypeError, match="transforms must contain"):
        replace(graph, transforms=(object(),))
    with pytest.raises(ValueError, match="duplicate node"):
        replace(graph, nodes=(node, node))
    with pytest.raises(ValueError, match="duplicate edge"):
        replace(graph, edges=(edge, edge))
    with pytest.raises(ValueError, match="crosses world"):
        replace(graph, edges=(replace(edge, world_id="desert"),))
    with pytest.raises(ValueError, match="duplicate frame pairs"):
        replace(graph, transforms=(transform, transform))
    mismatched = replace(edge, transform=SpatialFrameTransform("nav", "world"))
    with pytest.raises(ValueError, match="does not match"):
        replace(graph, edges=(mismatched,))
