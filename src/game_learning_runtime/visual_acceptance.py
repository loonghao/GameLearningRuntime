"""Deterministic capture and numeric visual-acceptance contracts.

This module intentionally never decides that an image "looks right".  It
returns bounded scalar measurements which can be checked by an agent without a
vision model, and a small request/job protocol that makes capture results
durable and correlatable.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import time_ns
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np

from game_learning_runtime.errors import ContractViolation

VISUAL_ACCEPTANCE_SCHEMA_VERSION = "glr.visual-acceptance.v1"
_MAX_FAILURES = 32


def _text(value: object, *, path: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{path} must be non-empty text up to {maximum} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


def _finite(
    value: object, *, path: str, minimum: float = 0.0, maximum: float | None = None
) -> float:
    numeric = float(cast(Any, value))
    if (
        not math.isfinite(numeric)
        or numeric < minimum
        or (maximum is not None and numeric > maximum)
    ):
        bound = f"between {minimum} and {maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{path} must be finite and {bound}")
    return numeric


def _positive_int(value: object, *, path: str, maximum: int = 4096) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"{path} must be an integer between 1 and {maximum}")
    return value


class CaptureJobStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    """Correlation token and expected dimensions for one capture request."""

    request_id: UUID | str = field(default_factory=uuid4)
    width: int = 1
    height: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", UUID(str(self.request_id)))
        object.__setattr__(self, "width", _positive_int(self.width, path="capture width"))
        object.__setattr__(self, "height", _positive_int(self.height, path="capture height"))

    def to_mapping(self) -> dict[str, object]:
        return {"request_id": str(self.request_id), "width": self.width, "height": self.height}


@dataclass(frozen=True, slots=True)
class CaptureArtifact:
    """Durably written capture metadata; only path and numeric facts are retained."""

    request_id: UUID | str
    path: str
    size_bytes: int
    width: int
    height: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", UUID(str(self.request_id)))
        object.__setattr__(self, "path", _text(self.path, path="capture artifact path"))
        if any(ord(character) < 32 for character in self.path):
            raise ValueError("capture artifact path cannot contain control characters")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes <= 0
        ):
            raise ValueError("capture artifact size_bytes must be a positive integer")
        object.__setattr__(self, "width", _positive_int(self.width, path="capture width"))
        object.__setattr__(self, "height", _positive_int(self.height, path="capture height"))
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("capture artifact sha256 must be a lowercase SHA-256 digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "request_id": str(self.request_id),
            "path": self.path,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CaptureResponse:
    """Correlated response, including pending state for long-running jobs."""

    request_id: UUID | str
    status: CaptureJobStatus
    job_id: UUID | str | None = None
    artifact: CaptureArtifact | None = None
    error: str = ""
    observed_at_ns: int = field(default_factory=time_ns)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", UUID(str(self.request_id)))
        object.__setattr__(self, "status", CaptureJobStatus(self.status))
        if self.job_id is not None:
            object.__setattr__(self, "job_id", UUID(str(self.job_id)))
        object.__setattr__(
            self,
            "error",
            _text(self.error, path="capture response error", maximum=512) if self.error else "",
        )
        if (
            not isinstance(self.observed_at_ns, int)
            or isinstance(self.observed_at_ns, bool)
            or self.observed_at_ns < 0
        ):
            raise ValueError("capture response observed_at_ns must be a non-negative integer")
        if self.status is CaptureJobStatus.PENDING and self.job_id is None:
            raise ValueError("pending capture response requires a job_id")
        if self.status is CaptureJobStatus.COMPLETED and self.artifact is None:
            raise ValueError("completed capture response requires an artifact")
        if self.status is CaptureJobStatus.FAILED and not self.error:
            raise ValueError("failed capture response requires an error")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": VISUAL_ACCEPTANCE_SCHEMA_VERSION,
            "request_id": str(self.request_id),
            "status": self.status.value,
            "job_id": str(self.job_id) if self.job_id is not None else None,
            "artifact": self.artifact.to_mapping() if self.artifact is not None else None,
            "error": self.error,
            "observed_at_ns": self.observed_at_ns,
        }


class CaptureCorrelationError(ContractViolation):
    """Raised when a response cannot be proven to belong to the request."""


def correlate_capture_response(
    request: CaptureRequest, response: CaptureResponse
) -> CaptureResponse:
    if response.request_id != request.request_id:
        raise CaptureCorrelationError("capture response request_id does not match request")
    if response.artifact is not None and (
        response.artifact.request_id != request.request_id
        or response.artifact.width != request.width
        or response.artifact.height != request.height
    ):
        raise CaptureCorrelationError("capture artifact does not match request correlation")
    return response


def write_capture_atomically(
    destination: str | Path,
    payload: bytes | bytearray | memoryview,
    *,
    request: CaptureRequest,
) -> CaptureArtifact:
    """Write bytes via temp-file/fsync/replace and report durable facts."""

    target = Path(destination)
    if target.is_symlink():
        raise FileExistsError("capture destination cannot be a symlink")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("capture payload must be bytes-like")
    data = bytes(payload)
    if not data:
        raise ValueError("capture payload cannot be empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(target)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return CaptureArtifact(
        request_id=request.request_id,
        path=target.as_posix(),
        size_bytes=len(data),
        width=request.width,
        height=request.height,
        sha256=hashlib.sha256(data).hexdigest(),
    )


class CaptureJobRegistry:
    """Small explicit async state registry with strict request correlation."""

    def __init__(self, *, max_jobs: int = 64) -> None:
        self._max_jobs = _positive_int(max_jobs, path="max_jobs", maximum=1024)
        self._jobs: dict[UUID, CaptureResponse] = {}

    def submit(self, request: CaptureRequest) -> CaptureResponse:
        if len(self._jobs) >= self._max_jobs:
            raise ContractViolation("capture job registry is full")
        job_id = uuid4()
        response = CaptureResponse(request.request_id, CaptureJobStatus.PENDING, job_id=job_id)
        self._jobs[job_id] = response
        return response

    def poll(self, request: CaptureRequest, job_id: UUID | str) -> CaptureResponse:
        resolved_job = UUID(str(job_id))
        response = self._jobs.get(resolved_job)
        if response is None:
            raise KeyError(f"unknown capture job: {resolved_job}")
        return correlate_capture_response(request, response)

    def complete(
        self, request: CaptureRequest, job_id: UUID | str, artifact: CaptureArtifact
    ) -> CaptureResponse:
        current = self.poll(request, job_id)
        if current.status is not CaptureJobStatus.PENDING:
            raise ContractViolation("capture job is already terminal")
        response = CaptureResponse(
            request.request_id,
            CaptureJobStatus.COMPLETED,
            job_id=current.job_id,
            artifact=artifact,
        )
        self._jobs[UUID(str(job_id))] = correlate_capture_response(request, response)
        return response

    def fail(self, request: CaptureRequest, job_id: UUID | str, error: str) -> CaptureResponse:
        current = self.poll(request, job_id)
        if current.status is not CaptureJobStatus.PENDING:
            raise ContractViolation("capture job is already terminal")
        response = CaptureResponse(
            request.request_id,
            CaptureJobStatus.FAILED,
            job_id=current.job_id,
            error=error,
        )
        self._jobs[UUID(str(job_id))] = response
        return response


def _image_array(image: Any, *, path: str) -> np.ndarray[Any, Any]:
    array = np.asarray(image)
    if array.ndim not in {2, 3} or array.size == 0:
        raise ValueError(f"{path} must be a non-empty HxW or HxWxC image")
    if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"{path} channel count must be 1, 3, or 4")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{path} must contain numeric pixels")
    values = array.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} pixels must be finite")
    if np.issubdtype(array.dtype, np.integer):
        values = values / float(max(np.iinfo(array.dtype).max, 1))
    elif float(values.max(initial=0.0)) > 1.0:
        if float(values.max(initial=0.0)) > 255.0:
            raise ValueError(f"{path} float pixels must be in [0, 1] or [0, 255]")
        values = values / 255.0
    if float(values.min(initial=0.0)) < 0.0 or float(values.max(initial=0.0)) > 1.0:
        raise ValueError(f"{path} normalized pixels must be in [0, 1]")
    return values


def _downsample(image: np.ndarray[Any, Any], size: int) -> np.ndarray[Any, Any]:
    height, width = image.shape[:2]
    if height <= size and width <= size:
        return image
    rows = np.linspace(0, height - 1, num=min(size, height), dtype=np.intp)
    columns = np.linspace(0, width - 1, num=min(size, width), dtype=np.intp)
    return image[rows][:, columns]


def _luminance(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if image.ndim == 2 or image.shape[2] == 1:
        return image if image.ndim == 2 else image[..., 0]
    return np.tensordot(
        image[..., :3], np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=1
    )


def _mask(image: np.ndarray[Any, Any], subject_mask: Any) -> np.ndarray[Any, Any]:
    if subject_mask is None:
        return _luminance(image) > 0.0
    mask = np.asarray(subject_mask)
    if mask.shape != image.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("subject_mask must be a boolean HxW mask matching image")
    return mask


def _sample_mask(mask: Any, *, source_shape: tuple[int, int], size: int) -> Any:
    array = np.asarray(mask)
    if array.shape != source_shape or array.dtype != np.bool_:
        raise ValueError("subject_mask must be a boolean HxW mask matching image")
    return _downsample(array, size)


def _bbox(mask: np.ndarray[Any, Any]) -> tuple[int, int, int, int] | None:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    return int(x0), int(y0), int(x1 + 1), int(y1 + 1)


@dataclass(frozen=True, slots=True)
class VisualMetrics:
    coverage_pct: float
    bbox: tuple[int, int, int, int] | None
    aspect: float | None
    distinct_colours: int
    chroma: float
    luminance: float
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coverage_pct", _finite(self.coverage_pct, path="coverage_pct", maximum=100)
        )
        if self.bbox is not None and (
            len(self.bbox) != 4 or any(not isinstance(value, int) for value in self.bbox)
        ):
            raise ValueError("bbox must contain four integer coordinates or None")
        if self.aspect is not None:
            object.__setattr__(self, "aspect", _finite(self.aspect, path="aspect", minimum=1e-12))
        if (
            not isinstance(self.distinct_colours, int)
            or isinstance(self.distinct_colours, bool)
            or self.distinct_colours < 0
        ):
            raise ValueError("distinct_colours must be a non-negative integer")
        object.__setattr__(self, "chroma", _finite(self.chroma, path="chroma", maximum=1))
        object.__setattr__(self, "luminance", _finite(self.luminance, path="luminance", maximum=1))
        object.__setattr__(self, "width", _positive_int(self.width, path="width"))
        object.__setattr__(self, "height", _positive_int(self.height, path="height"))

    def to_mapping(self) -> dict[str, object]:
        return {
            "coverage_pct": self.coverage_pct,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "aspect": self.aspect,
            "distinct_colours": self.distinct_colours,
            "chroma": self.chroma,
            "luminance": self.luminance,
            "width": self.width,
            "height": self.height,
        }


def compute_visual_metrics(
    image: Any, *, subject_mask: Any = None, downsample_size: int = 256
) -> VisualMetrics:
    """Compute bounded metrics for a render without retaining the image."""

    size = _positive_int(downsample_size, path="downsample_size", maximum=1024)
    source = _image_array(image, path="image")
    array = _downsample(source, size)
    sampled_mask = (
        None
        if subject_mask is None
        else _sample_mask(subject_mask, source_shape=source.shape[:2], size=size)
    )
    mask = _mask(array, sampled_mask)
    luminance = _luminance(array)
    count = int(mask.sum())
    total = int(mask.size)
    box = _bbox(mask)
    aspect = None if box is None else (box[2] - box[0]) / (box[3] - box[1])
    selected = array[mask]
    if selected.size == 0:
        distinct = 0
        chroma = 0.0
        mean_luminance = 0.0
    else:
        channels = selected[:, None] if array.ndim == 2 else selected[:, :3]
        quantized = np.rint(channels * 255).astype(np.uint8)
        distinct = int(np.unique(quantized, axis=0).shape[0])
        chroma = (
            0.0 if array.ndim == 2 else float((channels.max(axis=1) - channels.min(axis=1)).mean())
        )
        mean_luminance = float(luminance[mask].mean())
    return VisualMetrics(
        coverage_pct=100.0 * count / total,
        bbox=box,
        aspect=aspect,
        distinct_colours=distinct,
        chroma=chroma,
        luminance=mean_luminance,
        width=int(array.shape[1]),
        height=int(array.shape[0]),
    )


def silhouette_iou(
    image: Any,
    reference: Any,
    *,
    subject_mask: Any = None,
    reference_mask: Any = None,
    downsample_size: int = 128,
) -> float:
    """Return intersection-over-union of two fixed-grid subject silhouettes."""

    size = _positive_int(downsample_size, path="downsample_size", maximum=1024)
    first_source = _image_array(image, path="image")
    second_source = _image_array(reference, path="reference")
    first = _downsample(first_source, size)
    second = _downsample(second_source, size)
    first_sampled_mask = (
        None
        if subject_mask is None
        else _sample_mask(subject_mask, source_shape=first_source.shape[:2], size=size)
    )
    second_sampled_mask = (
        None
        if reference_mask is None
        else _sample_mask(reference_mask, source_shape=second_source.shape[:2], size=size)
    )
    first_mask = _mask(first, first_sampled_mask)
    second_mask = _mask(second, second_sampled_mask)
    if first_mask.shape != second_mask.shape:
        rows = min(first_mask.shape[0], second_mask.shape[0])
        columns = min(first_mask.shape[1], second_mask.shape[1])
        first_mask = first_mask[:rows, :columns]
        second_mask = second_mask[:rows, :columns]
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_mask, second_mask).sum() / union)


@dataclass(frozen=True, slots=True)
class VisualAcceptanceConfig:
    min_coverage_pct: float = 0.0
    max_coverage_pct: float = 100.0
    min_distinct_colours: int = 0
    min_chroma: float = 0.0
    expected_aspect: float | None = None
    aspect_tolerance: float = 0.05
    min_luminance: float = 0.0
    max_luminance: float = 1.0
    min_silhouette_iou: float | None = None

    def __post_init__(self) -> None:
        minimum = _finite(self.min_coverage_pct, path="min_coverage_pct", maximum=100)
        maximum = _finite(self.max_coverage_pct, path="max_coverage_pct", maximum=100)
        if minimum > maximum:
            raise ValueError("min_coverage_pct cannot exceed max_coverage_pct")
        object.__setattr__(self, "min_coverage_pct", minimum)
        object.__setattr__(self, "max_coverage_pct", maximum)
        if (
            not isinstance(self.min_distinct_colours, int)
            or isinstance(self.min_distinct_colours, bool)
            or self.min_distinct_colours < 0
        ):
            raise ValueError("min_distinct_colours must be a non-negative integer")
        object.__setattr__(
            self, "min_chroma", _finite(self.min_chroma, path="min_chroma", maximum=1)
        )
        if self.expected_aspect is not None:
            object.__setattr__(
                self,
                "expected_aspect",
                _finite(self.expected_aspect, path="expected_aspect", minimum=1e-12),
            )
        object.__setattr__(
            self,
            "aspect_tolerance",
            _finite(self.aspect_tolerance, path="aspect_tolerance", maximum=1),
        )
        object.__setattr__(
            self, "min_luminance", _finite(self.min_luminance, path="min_luminance", maximum=1)
        )
        object.__setattr__(
            self, "max_luminance", _finite(self.max_luminance, path="max_luminance", maximum=1)
        )
        if self.min_luminance > self.max_luminance:
            raise ValueError("min_luminance cannot exceed max_luminance")
        if self.min_silhouette_iou is not None:
            object.__setattr__(
                self,
                "min_silhouette_iou",
                _finite(self.min_silhouette_iou, path="min_silhouette_iou", maximum=1),
            )


@dataclass(frozen=True, slots=True)
class VisualAcceptanceReport:
    metrics: VisualMetrics
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)
    silhouette_iou: float | None = None
    schema_version: str = VISUAL_ACCEPTANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VISUAL_ACCEPTANCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported visual acceptance schema: {self.schema_version!r}")
        if not isinstance(self.metrics, VisualMetrics):
            raise TypeError("metrics must be VisualMetrics")
        if not isinstance(self.passed, bool):
            raise ValueError("visual acceptance passed must be a boolean")
        failures = tuple(_text(item, path="visual failure", maximum=256) for item in self.failures)
        if len(failures) > _MAX_FAILURES:
            raise ValueError("visual acceptance failures are bounded")
        object.__setattr__(self, "failures", failures)
        if self.silhouette_iou is not None:
            object.__setattr__(
                self,
                "silhouette_iou",
                _finite(self.silhouette_iou, path="silhouette_iou", maximum=1),
            )
        if self.passed != (not failures):
            raise ValueError("visual acceptance passed must match failures")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": self.metrics.to_mapping(),
            "silhouette_iou": self.silhouette_iou,
        }


class VisualAcceptanceError(ContractViolation):
    """Raised when a required numeric visual assertion fails."""

    def __init__(self, report: VisualAcceptanceReport) -> None:
        self.report = report
        super().__init__("visual acceptance failed: " + "; ".join(report.failures))


def evaluate_visual_acceptance(
    image: Any,
    config: VisualAcceptanceConfig | None = None,
    *,
    subject_mask: Any = None,
    reference: Any = None,
    reference_mask: Any = None,
    downsample_size: int = 256,
) -> VisualAcceptanceReport:
    resolved = config or VisualAcceptanceConfig()
    metrics = compute_visual_metrics(
        image, subject_mask=subject_mask, downsample_size=downsample_size
    )
    failures: list[str] = []
    if not resolved.min_coverage_pct <= metrics.coverage_pct <= resolved.max_coverage_pct:
        failures.append("coverage_pct_out_of_bounds")
    if metrics.distinct_colours < resolved.min_distinct_colours:
        failures.append("distinct_colours_below_threshold")
    if metrics.chroma < resolved.min_chroma:
        failures.append("chroma_below_threshold")
    if not resolved.min_luminance <= metrics.luminance <= resolved.max_luminance:
        failures.append("luminance_out_of_bounds")
    if resolved.expected_aspect is not None and (
        metrics.aspect is None
        or abs(metrics.aspect - resolved.expected_aspect) > resolved.aspect_tolerance
    ):
        failures.append("aspect_out_of_bounds")
    iou = None
    if reference is not None:
        iou = silhouette_iou(
            image,
            reference,
            subject_mask=subject_mask,
            reference_mask=reference_mask,
            downsample_size=min(downsample_size, 1024),
        )
        if resolved.min_silhouette_iou is not None and iou < resolved.min_silhouette_iou:
            failures.append("silhouette_iou_below_threshold")
    return VisualAcceptanceReport(
        metrics=metrics,
        passed=not failures,
        failures=tuple(failures),
        silhouette_iou=iou,
    )


def require_visual_acceptance(report: VisualAcceptanceReport, *, required: bool = True) -> None:
    if required and not report.passed:
        raise VisualAcceptanceError(report)


capture_atomically = write_capture_atomically


__all__ = [
    "VISUAL_ACCEPTANCE_SCHEMA_VERSION",
    "CaptureArtifact",
    "CaptureCorrelationError",
    "CaptureJobRegistry",
    "CaptureJobStatus",
    "CaptureRequest",
    "CaptureResponse",
    "VisualAcceptanceConfig",
    "VisualAcceptanceError",
    "VisualAcceptanceReport",
    "VisualMetrics",
    "capture_atomically",
    "compute_visual_metrics",
    "correlate_capture_response",
    "evaluate_visual_acceptance",
    "require_visual_acceptance",
    "silhouette_iou",
    "write_capture_atomically",
]
