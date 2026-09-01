from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from game_learning_runtime.capture import (
    CaptureFile,
    CaptureFrame,
    CaptureIndexWriter,
    CaptureManifest,
    build_capture_manifest,
    read_capture_index,
    verify_capture_manifest,
)
from game_learning_runtime.errors import ContractViolation


def test_capture_manifest_binds_review_video_to_step_aligned_training_index(
    tmp_path: Path,
) -> None:
    run_id = "run-capture"
    episode_id = uuid4()
    video = tmp_path / "capture.mp4"
    index = tmp_path / "capture-index.jsonl"
    manifest_path = tmp_path / "capture.manifest.json"
    video.write_bytes(b"synthetic-mp4-bytes")
    with CaptureIndexWriter(index) as writer:
        writer.write(
            CaptureFrame(
                run_id=run_id,
                episode_id=episode_id,
                step_id=0,
                frame_index=0,
                pts_ns=0,
                observation_timestamp_ns=100,
            )
        )
        writer.write(
            CaptureFrame(
                run_id=run_id,
                episode_id=episode_id,
                step_id=1,
                frame_index=3,
                pts_ns=100_000_000,
                observation_timestamp_ns=100_000_100,
            )
        )

    manifest = build_capture_manifest(
        manifest_path,
        environment_id="example.adventure-v1",
        run_id=run_id,
        video_path=video,
        index_path=index,
        codec="h264",
        frame_rate=30.0,
        width=640,
        height=360,
    )
    verified = verify_capture_manifest(
        manifest_path, expected_environment_id=manifest.environment_id
    )

    assert verified == manifest
    assert verified.video.path == "capture.mp4"
    assert verified.index.path == "capture-index.jsonl"
    assert len(verified.frames) == 2


def _frame(**overrides: object) -> CaptureFrame:
    values: dict[str, object] = {
        "run_id": "run-capture",
        "episode_id": uuid4(),
        "step_id": 0,
        "frame_index": 0,
        "pts_ns": 0,
        "observation_timestamp_ns": 1,
    }
    values.update(overrides)
    return CaptureFrame(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "wrong", "unsupported"),
        ("run_id", "", "cannot be empty"),
        ("step_id", -1, "non-negative"),
        ("frame_index", True, "non-negative"),
        ("pts_ns", -1, "non-negative"),
    ],
)
def test_capture_frame_rejects_invalid_identity_and_time(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _frame(**{field: value})


def test_capture_index_rejects_duplicates_non_monotonic_and_existing_output(
    tmp_path: Path,
) -> None:
    episode = uuid4()
    index = tmp_path / "index.jsonl"
    with (
        pytest.raises(ContractViolation, match="duplicate"),
        CaptureIndexWriter(index) as writer,
    ):
        writer.write(_frame(episode_id=episode))
        writer.write(_frame(episode_id=episode, frame_index=1, pts_ns=1))
    with pytest.raises(FileExistsError, match="absent or empty"):
        CaptureIndexWriter(index).__enter__()

    second = tmp_path / "second.jsonl"
    with (
        pytest.raises(ContractViolation, match="monotonic"),
        CaptureIndexWriter(second) as writer,
    ):
        writer.write(_frame(episode_id=episode, frame_index=1, pts_ns=2))
        writer.write(_frame(episode_id=episode, step_id=1, frame_index=0, pts_ns=1))


def test_capture_index_reader_rejects_invalid_json_and_multiple_runs(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        tuple(read_capture_index(invalid))

    multiple = tmp_path / "multiple.jsonl"
    first = _frame().to_mapping()
    second = _frame(run_id="run-other", step_id=1, frame_index=1, pts_ns=1).to_mapping()
    multiple.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(ContractViolation, match="multiple run"):
        tuple(read_capture_index(multiple))


def test_capture_manifest_rejects_invalid_files_dimensions_and_tampering(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="portable relative"):
        CaptureFile(path="../video.mp4", sha256="0" * 64, size_bytes=1)
    with pytest.raises(ValueError, match="SHA-256"):
        CaptureFile(path="video.mp4", sha256="bad", size_bytes=1)
    with pytest.raises(ValueError, match="non-negative"):
        CaptureFile(path="video.mp4", sha256="0" * 64, size_bytes=-1)
    entry = CaptureFile(path="video.mp4", sha256="0" * 64, size_bytes=1)
    with pytest.raises(ValueError, match="at least one"):
        CaptureManifest(
            environment_id="example.adventure-v1",
            run_id="run-one",
            video=entry,
            index=CaptureFile(path="index.jsonl", sha256="0" * 64, size_bytes=1),
            codec="h264",
            frame_rate=12,
            width=640,
            height=360,
            frames=(),
        )
    with pytest.raises(ValueError, match="positive"):
        CaptureManifest(
            environment_id="example.adventure-v1",
            run_id="run-one",
            video=entry,
            index=CaptureFile(path="index.jsonl", sha256="0" * 64, size_bytes=1),
            codec="h264",
            frame_rate=0,
            width=640,
            height=360,
            frames=(_frame(run_id="run-one"),),
        )

    video = tmp_path / "capture.mp4"
    index = tmp_path / "index.jsonl"
    manifest = tmp_path / "manifest.json"
    video.write_bytes(b"video")
    with CaptureIndexWriter(index) as writer:
        writer.write(_frame(run_id="run-one"))
    build_capture_manifest(
        manifest,
        environment_id="example.adventure-v1",
        run_id="run-one",
        video_path=video,
        index_path=index,
        codec="h264",
        frame_rate=12,
        width=640,
        height=360,
    )
    with pytest.raises(ContractViolation, match="environment_id"):
        verify_capture_manifest(manifest, expected_environment_id="example.other-v1")
    video.write_bytes(b"tampered")
    with pytest.raises(ContractViolation, match="sha256"):
        verify_capture_manifest(manifest, expected_environment_id="example.adventure-v1")


def test_capture_contract_rejects_wrong_shapes_and_manifest_locations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        CaptureFrame.from_mapping({"schema_version": "glr.capture-frame.v1"})
    with pytest.raises(ValueError, match="fields"):
        CaptureFile.from_mapping({"path": "video.mp4"})
    writer = CaptureIndexWriter(tmp_path / "not-open.jsonl")
    with pytest.raises(RuntimeError, match="context manager"):
        writer.write(_frame())
    with (
        pytest.raises(TypeError, match="CaptureFrame"),
        CaptureIndexWriter(tmp_path / "wrong-type.jsonl") as opened,
    ):
        opened.write(object())  # type: ignore[arg-type]

    entry = CaptureFile(path="video.mp4", sha256="0" * 64, size_bytes=1)
    index_entry = CaptureFile(path="index.jsonl", sha256="0" * 64, size_bytes=1)
    for changes, message in (
        ({"environment_id": ""}, "cannot be empty"),
        ({"width": 0}, "positive integer"),
        ({"frames": (_frame(run_id="other"),)}, "run_id"),
    ):
        values = {
            "environment_id": "example.adventure-v1",
            "run_id": "run-one",
            "video": entry,
            "index": index_entry,
            "codec": "h264",
            "frame_rate": 12,
            "width": 640,
            "height": 360,
            "frames": (_frame(run_id="run-one"),),
            **changes,
        }
        with pytest.raises((ContractViolation, ValueError), match=message):
            CaptureManifest(**values)  # type: ignore[arg-type]

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()
    outside_video = tmp_path / "outside.mp4"
    outside_video.write_bytes(b"video")
    index = capture_dir / "index.jsonl"
    with CaptureIndexWriter(index) as opened:
        opened.write(_frame(run_id="run-one"))
    with pytest.raises(ValueError, match="inside"):
        build_capture_manifest(
            capture_dir / "manifest.json",
            environment_id="example.adventure-v1",
            run_id="run-one",
            video_path=outside_video,
            index_path=index,
            codec="h264",
            frame_rate=12,
            width=640,
            height=360,
        )

    invalid_manifest = tmp_path / "invalid-manifest.json"
    invalid_manifest.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="UTF-8 JSON"):
        verify_capture_manifest(invalid_manifest, expected_environment_id="example.adventure-v1")
    invalid_manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="object"):
        verify_capture_manifest(invalid_manifest, expected_environment_id="example.adventure-v1")
