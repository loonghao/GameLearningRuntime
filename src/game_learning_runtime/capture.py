"""Checksummed review video and step-aligned capture index contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import IO, Any
from uuid import UUID

from game_learning_runtime.capture_liveness import (
    ContentLivenessReport,
)
from game_learning_runtime.errors import ContractViolation

CAPTURE_MANIFEST_SCHEMA_VERSION = "glr.capture.v1"
CAPTURE_FRAME_SCHEMA_VERSION = "glr.capture-frame.v1"
_MAX_INDEX_LINE_BYTES = 1024 * 1024


def _relative_path(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"{path} must be a portable relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{path} must be a portable relative path")
    return candidate.as_posix()


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be an existing regular non-symlink file")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_fields(value: Mapping[str, Any], *, expected: frozenset[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(f"{path} has missing={missing} and unexpected={unexpected} fields")


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    """Mapping from one environment step to one decoded video frame."""

    run_id: str
    episode_id: UUID
    step_id: int
    frame_index: int
    pts_ns: int
    observation_timestamp_ns: int
    schema_version: str = CAPTURE_FRAME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_FRAME_SCHEMA_VERSION:
            raise ValueError(f"unsupported capture frame schema: {self.schema_version!r}")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("capture frame run_id cannot be empty")
        if not isinstance(self.episode_id, UUID):
            object.__setattr__(self, "episode_id", UUID(str(self.episode_id)))
        for name, value in (
            ("step_id", self.step_id),
            ("frame_index", self.frame_index),
            ("pts_ns", self.pts_ns),
            ("observation_timestamp_ns", self.observation_timestamp_ns),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"capture frame {name} must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "episode_id": str(self.episode_id),
            "step_id": self.step_id,
            "frame_index": self.frame_index,
            "pts_ns": self.pts_ns,
            "observation_timestamp_ns": self.observation_timestamp_ns,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CaptureFrame:
        _strict_fields(
            value,
            expected=frozenset(
                {
                    "schema_version",
                    "run_id",
                    "episode_id",
                    "step_id",
                    "frame_index",
                    "pts_ns",
                    "observation_timestamp_ns",
                }
            ),
            path="capture frame",
        )
        return cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            episode_id=UUID(str(value["episode_id"])),
            step_id=value["step_id"],
            frame_index=value["frame_index"],
            pts_ns=value["pts_ns"],
            observation_timestamp_ns=value["observation_timestamp_ns"],
        )


class CaptureIndexWriter:
    """Append one unique step-to-frame mapping at a time and flush immediately."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._stream: IO[str] | None = None
        self._last_frame_index = -1
        self._last_pts_ns = -1
        self._steps: set[tuple[UUID, int]] = set()

    def __enter__(self) -> CaptureIndexWriter:
        if self._path.exists() and (self._path.is_symlink() or self._path.stat().st_size > 0):
            raise FileExistsError("capture index output must be absent or empty")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self._path.open("x" if not self._path.exists() else "w", encoding="utf-8")
        return self

    def write(self, frame: CaptureFrame) -> None:
        if self._stream is None:
            raise RuntimeError("capture index writer must be used as a context manager")
        if not isinstance(frame, CaptureFrame):
            raise TypeError("frame must be a CaptureFrame")
        key = (frame.episode_id, frame.step_id)
        if key in self._steps:
            raise ContractViolation("capture index contains a duplicate episode/step mapping")
        if frame.frame_index <= self._last_frame_index or frame.pts_ns < self._last_pts_ns:
            raise ContractViolation("capture frame indexes and PTS must be monotonic")
        encoded = json.dumps(frame.to_mapping(), separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > _MAX_INDEX_LINE_BYTES:
            raise ValueError("capture index line exceeds the 1 MiB limit")
        self._stream.write(encoded + "\n")
        self._stream.flush()
        self._steps.add(key)
        self._last_frame_index = frame.frame_index
        self._last_pts_ns = frame.pts_ns

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def read_capture_index(path: str | Path) -> Iterator[CaptureFrame]:
    source = _regular_file(Path(path), label="capture index")
    previous_frame = -1
    previous_pts = -1
    steps: set[tuple[UUID, int]] = set()
    run_id: str | None = None
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > _MAX_INDEX_LINE_BYTES:
                raise ValueError(f"capture index line {line_number} exceeds the 1 MiB limit")
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError("capture index record must be an object")
                frame = CaptureFrame.from_mapping(value)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid capture index record at line {line_number}") from error
            if run_id is None:
                run_id = frame.run_id
            elif frame.run_id != run_id:
                raise ContractViolation("capture index contains multiple run IDs")
            key = (frame.episode_id, frame.step_id)
            if key in steps:
                raise ContractViolation("capture index contains a duplicate episode/step mapping")
            if frame.frame_index <= previous_frame or frame.pts_ns < previous_pts:
                raise ContractViolation("capture frame indexes and PTS must be monotonic")
            steps.add(key)
            previous_frame = frame.frame_index
            previous_pts = frame.pts_ns
            yield frame


@dataclass(frozen=True, slots=True)
class CaptureFile:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, path="capture file path"))
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("capture file sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("capture file size_bytes must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CaptureFile:
        _strict_fields(
            value,
            expected=frozenset({"path", "sha256", "size_bytes"}),
            path="capture file",
        )
        return cls(path=value["path"], sha256=value["sha256"], size_bytes=value["size_bytes"])


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    environment_id: str
    run_id: str
    video: CaptureFile
    index: CaptureFile
    codec: str
    frame_rate: float
    width: int
    height: int
    frames: tuple[CaptureFrame, ...] = field(repr=False)
    schema_version: str = CAPTURE_MANIFEST_SCHEMA_VERSION
    content_liveness: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported capture manifest schema: {self.schema_version!r}")
        for name, value in (
            ("environment_id", self.environment_id),
            ("run_id", self.run_id),
            ("codec", self.codec),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"capture manifest {name} cannot be empty")
        rate = float(self.frame_rate)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("capture frame_rate must be positive and finite")
        object.__setattr__(self, "frame_rate", rate)
        for name, dimension in (("width", self.width), ("height", self.height)):
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise ValueError(f"capture {name} must be a positive integer")
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("capture manifest requires at least one indexed frame")
        if any(frame.run_id != self.run_id for frame in frames):
            raise ContractViolation("capture frame run_id does not match the manifest")
        object.__setattr__(self, "frames", frames)
        if self.content_liveness is not None:
            if not isinstance(self.content_liveness, Mapping):
                raise TypeError("capture content_liveness must be a mapping or None")
            try:
                liveness = ContentLivenessReport.from_mapping(self.content_liveness)
                encoded = json.dumps(liveness.to_mapping(), allow_nan=False)
            except (TypeError, ValueError, ContractViolation) as error:
                raise ValueError("capture content_liveness must be JSON-compatible") from error
            if len(encoded.encode("utf-8")) > _MAX_INDEX_LINE_BYTES:
                raise ValueError("capture content_liveness exceeds the 1 MiB limit")
            object.__setattr__(self, "content_liveness", liveness.to_mapping())

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "run_id": self.run_id,
            "video": self.video.to_mapping(),
            "index": self.index.to_mapping(),
            "codec": self.codec,
            "frame_rate": self.frame_rate,
            "width": self.width,
            "height": self.height,
        }
        if self.content_liveness is not None:
            mapping["content_liveness"] = dict(self.content_liveness)
        return mapping


def _file_entry(path: Path, *, relative_to: Path, label: str) -> CaptureFile:
    source = _regular_file(path, label=label).resolve()
    try:
        relative = source.relative_to(relative_to).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must be inside the capture manifest directory") from error
    return CaptureFile(path=relative, sha256=_sha256(source), size_bytes=source.stat().st_size)


def build_capture_manifest(
    manifest_path: str | Path,
    *,
    environment_id: str,
    run_id: str,
    video_path: str | Path,
    index_path: str | Path,
    codec: str,
    frame_rate: float,
    width: int,
    height: int,
    content_liveness: ContentLivenessReport | Mapping[str, object] | None = None,
    content_liveness_required: bool = False,
) -> CaptureManifest:
    requested_target = Path(manifest_path)
    if requested_target.is_symlink():
        raise FileExistsError("capture manifest output cannot be a symlink")
    target = requested_target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"capture manifest already exists: {target}")
    frames = tuple(read_capture_index(index_path))
    if isinstance(content_liveness, ContentLivenessReport):
        content_liveness.require_usable(
            required=content_liveness_required,
            max_bad_fraction=content_liveness.max_bad_fraction,
        )
        liveness_mapping: Mapping[str, object] | None = content_liveness.to_mapping()
    elif content_liveness is None:
        liveness_mapping = None
    elif isinstance(content_liveness, Mapping):
        report = ContentLivenessReport.from_mapping(content_liveness)
        report.require_usable(required=content_liveness_required)
        liveness_mapping = report.to_mapping()
    else:
        raise TypeError("content_liveness must be a ContentLivenessReport, mapping, or None")
    manifest = CaptureManifest(
        environment_id=environment_id,
        run_id=run_id,
        video=_file_entry(Path(video_path), relative_to=target.parent, label="capture video"),
        index=_file_entry(Path(index_path), relative_to=target.parent, label="capture index"),
        codec=codec,
        frame_rate=frame_rate,
        width=width,
        height=height,
        frames=frames,
        content_liveness=liveness_mapping,
    )
    payload = json.dumps(manifest.to_mapping(), indent=2, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(target)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return manifest


def verify_capture_manifest(
    manifest_path: str | Path, *, expected_environment_id: str
) -> CaptureManifest:
    source = _regular_file(Path(manifest_path), label="capture manifest").resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("capture manifest must be UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError("capture manifest must contain an object")
    expected_fields = frozenset(
        {
            "schema_version",
            "environment_id",
            "run_id",
            "video",
            "index",
            "codec",
            "frame_rate",
            "width",
            "height",
        }
    )
    if "content_liveness" in value:
        _strict_fields(
            value,
            expected=expected_fields | {"content_liveness"},
            path="capture manifest",
        )
    else:
        _strict_fields(value, expected=expected_fields, path="capture manifest")
    if value["environment_id"] != expected_environment_id:
        raise ContractViolation("capture environment_id does not match the expected project")
    video = CaptureFile.from_mapping(value["video"])
    index = CaptureFile.from_mapping(value["index"])
    for entry, label in ((video, "capture video"), (index, "capture index")):
        candidate = _regular_file(source.parent / PurePosixPath(entry.path), label=label)
        if candidate.stat().st_size != entry.size_bytes or _sha256(candidate) != entry.sha256:
            raise ContractViolation(f"sha256 mismatch for {entry.path}")
    frames = tuple(read_capture_index(source.parent / PurePosixPath(index.path)))
    return CaptureManifest(
        schema_version=value["schema_version"],
        environment_id=value["environment_id"],
        run_id=value["run_id"],
        video=video,
        index=index,
        codec=value["codec"],
        frame_rate=value["frame_rate"],
        width=value["width"],
        height=value["height"],
        frames=frames,
        content_liveness=value.get("content_liveness"),
    )


__all__ = [
    "CAPTURE_FRAME_SCHEMA_VERSION",
    "CAPTURE_MANIFEST_SCHEMA_VERSION",
    "CaptureFile",
    "CaptureFrame",
    "CaptureIndexWriter",
    "CaptureManifest",
    "build_capture_manifest",
    "read_capture_index",
    "verify_capture_manifest",
]
