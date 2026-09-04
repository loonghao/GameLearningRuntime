from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from game_learning_runtime.capture import (
    CaptureFrame,
    CaptureIndexWriter,
    build_capture_manifest,
    verify_capture_manifest,
)
from game_learning_runtime.capture_liveness import (
    CAPTURE_LIVENESS_SCHEMA_VERSION,
    ContentLivenessConfig,
    ContentLivenessGateError,
    ContentLivenessMonitor,
    ContentLivenessReport,
    ContentLivenessSample,
    ContentLivenessState,
    ContentStatistics,
    evaluate_content_liveness,
    measure_content,
    record_content_liveness,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.run_store import TrainingStore


def _moving_frames() -> list[np.ndarray]:
    first = np.full((8, 8, 3), 32, dtype=np.uint8)
    second = first.copy()
    second[:, 2:4, 0] = 255
    third = second.copy()
    third[:, 4:6, 1] = 220
    return [first, second, third]


def test_measure_content_is_normalized_and_uses_luminance() -> None:
    stats = measure_content(np.zeros((4, 4), dtype=np.uint8), np.full((4, 4), 255, dtype=np.uint8))
    assert stats.to_mapping() == {
        "inter_frame_diff_mean": 1.0,
        "inter_frame_diff_max": 1.0,
        "luminance_mean": 1.0,
        "luminance_std": 0.0,
    }
    rgb = measure_content(
        np.zeros((4, 4, 4), dtype=np.float32),
        np.dstack([np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4))]),
    )
    assert rgb.luminance_mean == pytest.approx(0.2126, abs=1e-4)
    config = ContentLivenessConfig(enabled=True, max_samples=8)
    assert ContentLivenessConfig.from_mapping(config.to_mapping()) == config
    with pytest.raises(ValueError, match="unexpected"):
        ContentLivenessConfig.from_mapping({"unknown": True})


@pytest.mark.parametrize(
    ("previous", "current", "state"),
    [
        (
            np.zeros((8, 8), dtype=np.uint8) + 32,
            np.ones((8, 8), dtype=np.uint8) * 255,
            ContentLivenessState.LIVE,
        ),
        (
            np.zeros((8, 8), dtype=np.uint8) + 32,
            np.zeros((8, 8), dtype=np.uint8) + 32,
            ContentLivenessState.CONTENT_STATIC,
        ),
        (
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((8, 8), dtype=np.uint8),
            ContentLivenessState.CONTENT_BLANK,
        ),
    ],
)
def test_monitor_classifies_moving_static_and_blank_content(
    previous: np.ndarray, current: np.ndarray, state: ContentLivenessState
) -> None:
    monitor = ContentLivenessMonitor(ContentLivenessConfig(enabled=True))
    sample = monitor.observe(previous, current, observed_at_ns=12)
    assert sample is not None
    assert sample.state is state
    report = monitor.report()
    assert report.schema_version == CAPTURE_LIVENESS_SCHEMA_VERSION
    assert report.sample_count == 1
    assert report.to_mapping()["samples"][0]["inter_frame_diff_mean"] >= 0
    assert ContentLivenessReport.from_mapping(report.to_mapping()) == report


def test_short_frozen_stretch_is_degraded_and_named() -> None:
    moving = _moving_frames()
    frozen = np.full((8, 8, 3), 64, dtype=np.uint8)
    frames = [moving[0], moving[1], frozen, frozen, moving[2]]
    report = evaluate_content_liveness(
        frames,
        ContentLivenessConfig(enabled=True, max_bad_fraction=0.75),
    )
    assert report.state is ContentLivenessState.DEGRADED
    assert "content_static" in report.reason or "content_blank" in report.reason
    assert report.bad_fraction > 0
    assert all(
        isinstance(value, (int, float, str, bool, list, dict, type(None)))
        for value in report.to_mapping().values()
    )


def test_monitor_is_disabled_without_frame_work_and_samples_sparsely() -> None:
    disabled = ContentLivenessMonitor()
    assert disabled.observe(np.zeros((4, 4)), np.ones((4, 4))) is None
    assert disabled.report().state is ContentLivenessState.DISABLED
    monitor = ContentLivenessMonitor(ContentLivenessConfig(enabled=True, sample_every=2))
    frame = np.full((4, 4), 32, dtype=np.uint8)
    assert monitor.observe(frame, frame, frame_pair_index=0) is not None
    assert monitor.observe(frame, frame, frame_pair_index=1) is None
    assert monitor.sample_count == 1


def test_monitor_keeps_recent_samples_but_aggregates_the_full_duration() -> None:
    frame = np.full((4, 4), 64, dtype=np.uint8)
    monitor = ContentLivenessMonitor(
        ContentLivenessConfig(enabled=True, max_samples=2, max_bad_fraction=0.5)
    )
    for index in range(5):
        monitor.observe(frame, frame, frame_pair_index=index, observed_at_ns=index)
    report = monitor.report()
    assert report.sample_count == 5
    assert report.content_static_count == 5
    assert len(report.samples) == 2
    assert report.bad_fraction == 1.0
    with pytest.raises(ContentLivenessGateError):
        monitor.gate(required=True)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ContentLivenessConfig(static_mean_threshold=-1),
        lambda: ContentLivenessConfig(sample_every=0),
        lambda: ContentLivenessConfig(downsample_size=3),
        lambda: ContentLivenessConfig(max_samples=0),
        lambda: ContentStatistics(0.2, 0.1, 0.0, 0.0),
        lambda: ContentLivenessSample(
            -1, ContentStatistics(0, 0, 0, 0), ContentLivenessState.LIVE, 0
        ),
    ],
)
def test_content_contract_rejects_invalid_values(factory: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        factory()  # type: ignore[operator]


def test_measure_content_rejects_invalid_frames_and_pair_index() -> None:
    with pytest.raises(ValueError, match="same shape"):
        measure_content(np.zeros((4, 4)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="channel count"):
        measure_content(np.zeros((4, 4, 2)), np.zeros((4, 4, 2)))
    with pytest.raises(ValueError, match="finite"):
        measure_content(np.full((4, 4), np.nan), np.zeros((4, 4)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        measure_content(np.full((4, 4), 300.0), np.zeros((4, 4)))
    monitor = ContentLivenessMonitor(ContentLivenessConfig(enabled=True))
    with pytest.raises(ValueError, match="frame_pair_index"):
        monitor.observe(np.zeros((4, 4)), np.zeros((4, 4)), frame_pair_index=-1)


def test_required_gate_fails_before_manifest_and_optional_gate_returns_report(
    tmp_path: Path,
) -> None:
    blank = np.zeros((8, 8), dtype=np.uint8)
    monitor = ContentLivenessMonitor(ContentLivenessConfig(enabled=True, required=True))
    monitor.observe(blank, blank, observed_at_ns=1)
    with pytest.raises(ContentLivenessGateError, match="gate failed"):
        monitor.gate()
    optional = ContentLivenessMonitor(ContentLivenessConfig(enabled=True, required=False))
    optional.observe(blank, blank, observed_at_ns=2)
    report = optional.gate()
    assert report.state is ContentLivenessState.CONTENT_BLANK

    allowed = ContentLivenessMonitor(
        ContentLivenessConfig(enabled=True, max_bad_fraction=1.0, required=True)
    )
    moving = _moving_frames()
    allowed.observe(moving[0], moving[1], observed_at_ns=3)
    allowed.observe(moving[1], moving[1], observed_at_ns=4)
    assert allowed.gate().state is ContentLivenessState.DEGRADED
    with pytest.raises(ContentLivenessGateError, match="disabled"):
        ContentLivenessMonitor(ContentLivenessConfig(required=True)).gate()

    video = tmp_path / "capture.mp4"
    index = tmp_path / "capture-index.jsonl"
    manifest_path = tmp_path / "capture.manifest.json"
    video.write_bytes(b"video")
    frame = CaptureFrame("run", uuid4(), 0, 0, 0, 1)
    with CaptureIndexWriter(index) as writer:
        writer.write(frame)
    with pytest.raises(ContentLivenessGateError):
        build_capture_manifest(
            manifest_path,
            environment_id="example.v1",
            run_id="run",
            video_path=video,
            index_path=index,
            codec="h264",
            frame_rate=30,
            width=8,
            height=8,
            content_liveness=monitor.report(),
            content_liveness_required=True,
        )


def test_manifest_embeds_metrics_and_old_manifest_remains_readable(tmp_path: Path) -> None:
    moving = _moving_frames()
    report = evaluate_content_liveness(moving, ContentLivenessConfig(enabled=True))
    video = tmp_path / "capture.mp4"
    index = tmp_path / "capture-index.jsonl"
    manifest_path = tmp_path / "manifest.json"
    video.write_bytes(b"video")
    with CaptureIndexWriter(index) as writer:
        writer.write(CaptureFrame("run", uuid4(), 0, 0, 0, 1))
    manifest = build_capture_manifest(
        manifest_path,
        environment_id="example.v1",
        run_id="run",
        video_path=video,
        index_path=index,
        codec="h264",
        frame_rate=30,
        width=8,
        height=8,
        content_liveness=report,
    )
    loaded = verify_capture_manifest(manifest_path, expected_environment_id="example.v1")
    assert loaded.content_liveness == report.to_mapping()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(payload["content_liveness"]["bad_fraction"], float)
    payload.pop("content_liveness")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    old = verify_capture_manifest(manifest_path, expected_environment_id="example.v1")
    assert old.content_liveness is None
    assert manifest.content_liveness is not None


def test_run_store_records_reason_and_measured_values(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite")
    store.create_run(
        run_id="run", environment_id="example.v1", protocol_version="v1", kind="capture"
    )
    report = evaluate_content_liveness(
        [np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)],
        ContentLivenessConfig(enabled=True),
    )
    metrics, event = record_content_liveness(store, "run", report, timestamp_ns=10)
    assert len(metrics) == 6
    assert event is not None
    assert event.payload["reason"]
    assert {metric.name for metric in store.list_metrics("run")} >= {
        "capture.content_liveness.bad_fraction",
        "capture.content_liveness.luminance_std",
    }


def test_empty_and_malformed_reports_are_bounded() -> None:
    empty = ContentLivenessReport(
        state=ContentLivenessState.DEGRADED,
        enabled=True,
        reason="no samples",
    )
    assert empty.bad_fraction == 0.0
    assert empty.last_sample_index is None
    assert not empty.usable
    with pytest.raises(ContentLivenessGateError, match="no samples"):
        empty.require_usable()
    with pytest.raises(ValueError, match="schema"):
        ContentLivenessReport(state=ContentLivenessState.LIVE, schema_version="wrong")
    with pytest.raises((ValueError, TypeError), match=r"samples|monotonic"):
        ContentLivenessReport(
            state=ContentLivenessState.LIVE,
            samples=(
                ContentLivenessSample(
                    1, ContentStatistics(0, 0, 1, 0), ContentLivenessState.LIVE, 0
                ),
                ContentLivenessSample(
                    0, ContentStatistics(0, 0, 1, 0), ContentLivenessState.LIVE, 0
                ),
            ),
            enabled=True,
        )
    with pytest.raises(TypeError, match="samples"):
        ContentLivenessReport(state=ContentLivenessState.LIVE, samples=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="enabled"):
        ContentLivenessReport(state=ContentLivenessState.LIVE, enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative integer"):
        ContentLivenessReport(state=ContentLivenessState.LIVE, total_sample_count=-1)
    malformed = empty.to_mapping()
    malformed["bad_fraction"] = 0.5
    with pytest.raises(ContractViolation, match="bad_fraction"):
        ContentLivenessReport.from_mapping(malformed)


def test_validation_edges_cover_bounds_and_json_contract() -> None:
    with pytest.raises(ValueError, match="text"):
        ContentLivenessReport(state=ContentLivenessState.LIVE, reason="x" * 257)
    with pytest.raises(ValueError, match="control"):
        ContentLivenessReport(state=ContentLivenessState.LIVE, reason="bad\nreason")
    with pytest.raises(ValueError, match="enabled"):
        ContentLivenessConfig(enabled=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="required"):
        ContentLivenessConfig(required=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        ContentLivenessConfig(max_bad_fraction=float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        ContentLivenessConfig(sample_every=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        ContentLivenessConfig(max_samples=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        ContentStatistics(float("inf"), 1, 0, 0)
    with pytest.raises(ValueError, match="between"):
        ContentStatistics(-0.1, 0, 0, 0)
    with pytest.raises(TypeError, match="statistics"):
        ContentLivenessSample(0, object(), ContentLivenessState.LIVE, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="observed_at_ns"):
        ContentLivenessSample(0, ContentStatistics(0, 0, 0, 0), ContentLivenessState.LIVE, -1)
    with pytest.raises(ValueError, match="aggregate"):
        ContentLivenessReport(
            state=ContentLivenessState.LIVE,
            samples=(),
            enabled=True,
            total_sample_count=1,
            total_content_static_count=1,
            total_content_blank_count=1,
        )


def test_mapping_rejects_shape_and_heartbeat_tampering() -> None:
    report = evaluate_content_liveness(
        [np.full((4, 4), 32, dtype=np.uint8), np.full((4, 4), 32, dtype=np.uint8)],
        ContentLivenessConfig(enabled=True),
    )
    mapping = report.to_mapping()
    for key, value in (("samples", {}), ("heartbeat", [])):
        invalid = dict(mapping)
        invalid[key] = value
        with pytest.raises((TypeError, ValueError)):
            ContentLivenessReport.from_mapping(invalid)
    invalid = dict(mapping)
    invalid["recent_sample_count"] = 0
    with pytest.raises(ContractViolation, match="recent_sample_count"):
        ContentLivenessReport.from_mapping(invalid)
    invalid = dict(mapping)
    invalid["heartbeat"] = dict(mapping["heartbeat"])
    invalid["heartbeat"]["sample_count"] = 2  # type: ignore[index]
    with pytest.raises(ContractViolation, match="heartbeat"):
        ContentLivenessReport.from_mapping(invalid)
    invalid = dict(mapping)
    invalid["heartbeat"] = dict(mapping["heartbeat"])
    invalid["heartbeat"]["last_observed_at_ns"] = 99  # type: ignore[index]
    with pytest.raises(ContractViolation, match="timestamp"):
        ContentLivenessReport.from_mapping(invalid)
    invalid = dict(mapping)
    invalid["samples"] = [object()]
    invalid["recent_sample_count"] = 1
    with pytest.raises(TypeError, match="sample"):
        ContentLivenessReport.from_mapping(invalid)
    invalid = dict(mapping)
    invalid_sample = dict(mapping["samples"][0])  # type: ignore[index]
    invalid_sample.pop("state")
    invalid["samples"] = [invalid_sample]
    invalid["recent_sample_count"] = 1
    with pytest.raises(ValueError, match="fields"):
        ContentLivenessReport.from_mapping(invalid)


def test_frame_conversion_downsampling_and_empty_inputs() -> None:
    large = np.zeros((80, 90), dtype=np.uint8)
    changed = large.copy()
    changed[::10, ::10] = 255
    assert measure_content(large, changed, downsample_size=8).inter_frame_diff_max > 0
    with pytest.raises(ValueError, match="non-empty"):
        measure_content(np.zeros((0, 4)), np.zeros((0, 4)))
    with pytest.raises(ValueError, match="HxW"):
        measure_content(np.zeros((4,)), np.zeros((4,)))
    with pytest.raises(TypeError, match="numeric"):
        measure_content(np.full((4, 4), "x"), np.full((4, 4), "x"))
    with pytest.raises(ValueError, match="normalized"):
        measure_content(np.full((4, 4), -1.0), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="downsample_size"):
        measure_content(large, large, downsample_size=3)
    assert evaluate_content_liveness([], ContentLivenessConfig(enabled=True)).sample_count == 0


def test_long_frozen_capture_reason_is_bounded() -> None:
    frame = np.full((4, 4), 64, dtype=np.uint8)
    report = evaluate_content_liveness(
        [frame for _ in range(20)], ContentLivenessConfig(enabled=True, max_samples=32)
    )
    assert report.state is ContentLivenessState.CONTENT_STATIC
    assert report.reason.endswith(",...")


def test_record_rejects_invalid_timestamp(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite")
    store.create_run(
        run_id="run", environment_id="example.v1", protocol_version="v1", kind="capture"
    )
    report = ContentLivenessReport(state=ContentLivenessState.DISABLED)
    with pytest.raises(ValueError, match="timestamp_ns"):
        record_content_liveness(store, "run", report, timestamp_ns=-1)
