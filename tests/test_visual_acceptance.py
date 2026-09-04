from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.visual_acceptance import (
    VISUAL_ACCEPTANCE_SCHEMA_VERSION,
    CaptureArtifact,
    CaptureCorrelationError,
    CaptureJobRegistry,
    CaptureJobStatus,
    CaptureRequest,
    CaptureResponse,
    VisualAcceptanceConfig,
    VisualAcceptanceError,
    VisualAcceptanceReport,
    VisualMetrics,
    compute_visual_metrics,
    correlate_capture_response,
    evaluate_visual_acceptance,
    require_visual_acceptance,
    silhouette_iou,
    write_capture_atomically,
)


def _image() -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[2:6, 2:6] = np.array([255, 32, 16], dtype=np.uint8)
    return image


def test_atomic_capture_is_durable_and_correlated(tmp_path: Path) -> None:
    request = CaptureRequest(width=8, height=8)
    destination = tmp_path / "nested" / "capture.png"
    artifact = write_capture_atomically(destination, b"png-bytes", request=request)
    assert destination.read_bytes() == b"png-bytes"
    assert artifact.path == destination.as_posix()
    assert artifact.size_bytes == 9
    assert artifact.request_id == request.request_id
    replacement = write_capture_atomically(destination, memoryview(b"new"), request=request)
    assert replacement.size_bytes == 3
    with pytest.raises(ValueError, match="empty"):
        write_capture_atomically(tmp_path / "empty", b"", request=request)
    with pytest.raises(TypeError, match="bytes-like"):
        write_capture_atomically(tmp_path / "bad", "text", request=request)  # type: ignore[arg-type]
    symlink = tmp_path / "link"
    symlink.symlink_to(destination)
    with pytest.raises(FileExistsError, match="symlink"):
        write_capture_atomically(symlink, b"x", request=request)


def test_capture_response_correlation_and_job_lifecycle(tmp_path: Path) -> None:
    request = CaptureRequest(request_id=uuid4(), width=8, height=8)
    other = CaptureRequest(width=8, height=8)
    registry = CaptureJobRegistry(max_jobs=2)
    pending = registry.submit(request)
    assert pending.status is CaptureJobStatus.PENDING
    assert registry.poll(request, pending.job_id).to_mapping()["status"] == "pending"
    artifact = write_capture_atomically(tmp_path / "capture.bin", b"data", request=request)
    completed = registry.complete(request, pending.job_id, artifact)
    assert completed.status is CaptureJobStatus.COMPLETED
    assert registry.poll(request, pending.job_id).artifact == artifact
    with pytest.raises(ContractViolation, match="terminal"):
        registry.complete(request, pending.job_id, artifact)
    second = registry.submit(other)
    failed = registry.fail(other, second.job_id, "device unavailable")
    assert failed.status is CaptureJobStatus.FAILED
    with pytest.raises(ContractViolation, match="terminal"):
        registry.fail(other, second.job_id, "again")
    with pytest.raises(KeyError, match="unknown"):
        registry.poll(request, uuid4())
    with pytest.raises(CaptureCorrelationError, match="request_id"):
        registry.poll(other, pending.job_id)
    with pytest.raises(ContractViolation, match="full"):
        registry.submit(CaptureRequest(width=1, height=1))


def test_capture_contract_rejects_invalid_states_and_dimensions() -> None:
    with pytest.raises(ValueError, match="width"):
        CaptureRequest(width=0)
    with pytest.raises(ValueError, match="UUID"):
        CaptureRequest(request_id="not-a-uuid")
    request = CaptureRequest(width=2, height=3)
    with pytest.raises(ValueError, match="completed"):
        CaptureResponse(request.request_id, CaptureJobStatus.COMPLETED)
    with pytest.raises(ValueError, match="pending"):
        CaptureResponse(request.request_id, CaptureJobStatus.PENDING)
    with pytest.raises(ValueError, match="error"):
        CaptureResponse(request.request_id, CaptureJobStatus.FAILED)
    with pytest.raises(ValueError, match="positive"):
        CaptureArtifact(request.request_id, "x", 0, 2, 3, "0" * 64)
    with pytest.raises(ValueError, match="SHA"):
        CaptureArtifact(request.request_id, "x", 1, 2, 3, "bad")
    artifact = CaptureArtifact(request.request_id, "x", 1, 2, 3, "0" * 64)
    response = CaptureResponse(request.request_id, CaptureJobStatus.COMPLETED, artifact=artifact)
    assert correlate_capture_response(request, response) is response
    wrong_dimensions = CaptureRequest(request_id=request.request_id, width=4, height=3)
    with pytest.raises(CaptureCorrelationError, match="artifact"):
        correlate_capture_response(wrong_dimensions, response)


def test_visual_metrics_cover_subject_mask_and_texture_statistics() -> None:
    image = _image()
    metrics = compute_visual_metrics(image)
    assert metrics.coverage_pct == pytest.approx(25.0)
    assert metrics.bbox == (2, 2, 6, 6)
    assert metrics.aspect == pytest.approx(1.0)
    assert metrics.distinct_colours == 1
    assert metrics.chroma > 0
    assert metrics.luminance > 0
    assert metrics.to_mapping()["bbox"] == [2, 2, 6, 6]
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 2:6] = True
    masked = compute_visual_metrics(image, subject_mask=mask)
    assert masked.coverage_pct == pytest.approx(12.5)
    blank = compute_visual_metrics(np.zeros((4, 4), dtype=np.uint8))
    assert blank.bbox is None
    assert blank.aspect is None
    assert blank.distinct_colours == 0
    assert blank.chroma == 0
    assert blank.luminance == 0


def test_visual_metrics_support_downsampling_and_silhouette_iou() -> None:
    first = np.zeros((32, 32), dtype=np.uint8)
    second = np.zeros((16, 16), dtype=np.uint8)
    first[8:24, 8:24] = 255
    second[4:12, 4:12] = 255
    metrics = compute_visual_metrics(first, downsample_size=8)
    assert metrics.width == 8 and metrics.height == 8
    assert silhouette_iou(first, first) == 1.0
    assert silhouette_iou(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0
    assert 0 < silhouette_iou(first, second) < 1
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="boolean"):
        compute_visual_metrics(first, subject_mask=np.ones((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match="matching"):
        compute_visual_metrics(first, subject_mask=mask)
    with pytest.raises(ValueError, match="downsample_size"):
        compute_visual_metrics(first, downsample_size=0)


def test_acceptance_pass_fail_and_reference_assertions() -> None:
    image = _image()
    passing = evaluate_visual_acceptance(
        image,
        VisualAcceptanceConfig(
            min_coverage_pct=20,
            min_distinct_colours=1,
            min_chroma=0.1,
            expected_aspect=1,
            min_silhouette_iou=0.9,
        ),
        reference=image,
    )
    assert passing.passed
    assert passing.failures == ()
    assert passing.silhouette_iou == 1.0
    assert passing.to_mapping()["schema_version"] == VISUAL_ACCEPTANCE_SCHEMA_VERSION
    require_visual_acceptance(passing)
    failing = evaluate_visual_acceptance(
        image,
        VisualAcceptanceConfig(
            min_coverage_pct=50,
            max_coverage_pct=60,
            min_distinct_colours=99,
            min_chroma=0.9,
            expected_aspect=2,
            aspect_tolerance=0.01,
            min_luminance=0.9,
            min_silhouette_iou=0.9,
        ),
        reference=np.zeros_like(image),
    )
    assert not failing.passed
    assert len(failing.failures) == 5
    with pytest.raises(VisualAcceptanceError, match="coverage"):
        require_visual_acceptance(failing)
    require_visual_acceptance(failing, required=False)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: VisualAcceptanceConfig(min_coverage_pct=101),
        lambda: VisualAcceptanceConfig(min_coverage_pct=80, max_coverage_pct=20),
        lambda: VisualAcceptanceConfig(min_distinct_colours=-1),
        lambda: VisualAcceptanceConfig(min_chroma=2),
        lambda: VisualAcceptanceConfig(expected_aspect=0),
        lambda: VisualAcceptanceConfig(aspect_tolerance=2),
        lambda: VisualAcceptanceConfig(min_luminance=0.8, max_luminance=0.2),
        lambda: VisualAcceptanceConfig(min_silhouette_iou=2),
    ],
)
def test_acceptance_config_rejects_invalid_bounds(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_visual_contract_rejects_invalid_metrics_and_reports() -> None:
    with pytest.raises(ValueError, match="coverage_pct"):
        VisualMetrics(101, None, None, 0, 0, 0, 1, 1)
    with pytest.raises(ValueError, match="bbox"):
        VisualMetrics(0, (1, 2), None, 0, 0, 0, 1, 1)  # type: ignore[arg-type]
    metrics = VisualMetrics(0, None, None, 0, 0, 0, 1, 1)
    with pytest.raises(ValueError, match="passed"):
        VisualAcceptanceReport(metrics, True, failures=("failure",))
    with pytest.raises(ValueError, match="schema"):
        VisualAcceptanceReport(metrics, True, schema_version="wrong")
