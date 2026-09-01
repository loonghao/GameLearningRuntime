from __future__ import annotations

from uuid import UUID

import pytest

from game_learning_runtime import (
    ArtifactLineage,
    ModalNavigationBoundary,
    ModalState,
    RouteHealthTelemetry,
    RouteTransitionEvidence,
)
from game_learning_runtime.errors import ContractViolation

EPISODE = UUID("12345678-1234-5678-8123-456789abcdef")
MAP_SHA = "a" * 64
IMAGE_SHA = "b" * 64


def test_route_transition_evidence_requires_settled_authoritative_edge_proof() -> None:
    evidence = RouteTransitionEvidence(
        episode_id=EPISODE,
        route_id="route.alpha",
        edge_id="edge.03",
        map_sha256=MAP_SHA,
        game_image_sha256=IMAGE_SHA,
        producer_state_seq=41,
        settled=True,
        reached=True,
        position=(1.0, 0.0, 2.5),
        yaw_deg=-179.0,
        heading_error_deg=2.0,
    )

    assert evidence.edge_succeeded
    assert evidence.producer_state_seq == 41
    assert evidence.yaw_deg == -179.0

    unsettled = RouteTransitionEvidence(
        episode_id=EPISODE,
        route_id="route.alpha",
        edge_id="edge.03",
        map_sha256=MAP_SHA,
        game_image_sha256=IMAGE_SHA,
        producer_state_seq=42,
        settled=False,
        reached=True,
        position=(1.0, 0.0, 2.5),
        yaw_deg=0.0,
        heading_error_deg=2.0,
    )
    assert not unsettled.edge_succeeded


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"route_id": "local/path"}, "route_id"),
        ({"map_sha256": "x"}, "SHA-256"),
        ({"producer_state_seq": -1}, "non-negative"),
        ({"yaw_deg": 180.0}, "yaw_deg"),
        ({"heading_error_deg": 181.0}, "heading_error_deg"),
        ({"position": (1.0, 2.0)}, "position"),
    ],
)
def test_route_transition_evidence_rejects_ambiguous_or_unbounded_values(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "episode_id": EPISODE,
        "route_id": "route.alpha",
        "edge_id": "edge.03",
        "map_sha256": MAP_SHA,
        "game_image_sha256": IMAGE_SHA,
        "producer_state_seq": 41,
        "settled": True,
        "reached": True,
        "position": (1.0, 0.0, 2.5),
        "yaw_deg": 0.0,
        "heading_error_deg": 2.0,
    }
    values.update(changes)

    with pytest.raises((ContractViolation, TypeError, ValueError), match=message):
        RouteTransitionEvidence(**values)  # type: ignore[arg-type]


def test_route_health_telemetry_captures_stall_and_oscillation_without_recovery_authority() -> None:
    telemetry = RouteHealthTelemetry(
        episode_id=EPISODE,
        route_id="route.alpha",
        producer_state_seq=42,
        displacement_m=0.0,
        heading_delta_deg=179.5,
        stall_ticks=12,
        oscillation_count=3,
    )

    assert telemetry.stalled
    assert telemetry.oscillating
    assert telemetry.requires_observation


def test_modal_boundary_only_resumes_after_authoritative_close() -> None:
    opened = ModalNavigationBoundary(
        episode_id=EPISODE,
        modal_id="vendor",
        state=ModalState.OPEN,
        producer_state_seq=8,
        authoritative=True,
    )
    closed_unverified = ModalNavigationBoundary(
        episode_id=EPISODE,
        modal_id="vendor",
        state=ModalState.CLOSED,
        producer_state_seq=9,
        authoritative=False,
    )
    closed = ModalNavigationBoundary(
        episode_id=EPISODE,
        modal_id="vendor",
        state=ModalState.CLOSED,
        producer_state_seq=10,
        authoritative=True,
    )

    assert opened.navigation_paused
    assert closed_unverified.navigation_paused
    assert not closed.navigation_paused
    with pytest.raises(ContractViolation, match="producer_state_seq"):
        ModalNavigationBoundary(
            episode_id=EPISODE,
            modal_id="vendor",
            state=ModalState.CLOSED,
            producer_state_seq=7,
            authoritative=True,
            previous=opened,
        )


def test_artifact_lineage_binds_episode_route_build_and_encounter_identity() -> None:
    lineage = ArtifactLineage(
        episode_id=EPISODE,
        trajectory_id="trajectory.01",
        recording_id="recording.01",
        route_id="route.alpha",
        map_sha256=MAP_SHA,
        game_image_sha256=IMAGE_SHA,
        encounter_id="encounter.elite.01",
    )

    assert lineage.encounter_id == "encounter.elite.01"
    assert lineage.to_mapping()["episode_id"] == str(EPISODE)
    assert "display_name" not in lineage.to_mapping()


def test_artifact_lineage_rejects_display_names_and_invalid_hashes() -> None:
    with pytest.raises(ValueError, match="public identifier"):
        ArtifactLineage(
            episode_id=EPISODE,
            trajectory_id="trajectory.01",
            recording_id="recording.01",
            route_id="route.alpha",
            map_sha256=MAP_SHA,
            game_image_sha256=IMAGE_SHA,
            encounter_id="Elite Display Name",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        ArtifactLineage(
            episode_id=EPISODE,
            trajectory_id="trajectory.01",
            recording_id="recording.01",
            route_id="route.alpha",
            map_sha256="z" * 64,
            game_image_sha256=IMAGE_SHA,
        )
