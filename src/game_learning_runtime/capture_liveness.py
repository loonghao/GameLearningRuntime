"""Cheap, game-neutral telemetry for detecting frozen or blank captures.

The monitor deliberately accepts two frames for each sample instead of keeping a
previous frame internally.  Callers therefore control sampling and the monitor
retains only bounded numeric statistics; captured pixels never become part of a
run record or a report.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from time import time_ns
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from game_learning_runtime.errors import ContractViolation

if TYPE_CHECKING:
    from game_learning_runtime.run_store import MetricRecord, RunEvent


CAPTURE_LIVENESS_SCHEMA_VERSION = "glr.capture-content-liveness.v1"
_MAX_REASON_LENGTH = 256


class ContentLivenessState(str, Enum):
    """Aggregated state of sampled capture content."""

    DISABLED = "disabled"
    LIVE = "live"
    CONTENT_STATIC = "content_static"
    CONTENT_BLANK = "content_blank"
    DEGRADED = "degraded"


def _bounded_reason(value: str, *, path: str = "reason") -> str:
    if not isinstance(value, str) or len(value) > _MAX_REASON_LENGTH:
        raise ValueError(f"{path} must be text up to {_MAX_REASON_LENGTH} characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


def _threshold(value: float, *, path: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{path} must be finite and between 0 and 1")
    return numeric


@dataclass(frozen=True, slots=True)
class ContentLivenessConfig:
    """Bounded thresholds and sampling controls for content liveness.

    The default is disabled, preserving the cost and behaviour of existing
    capture sessions.  Thresholds operate on normalized [0, 1] pixel/luminance
    values, independent of the source's integer bit depth.
    """

    enabled: bool = False
    static_mean_threshold: float = 0.002
    static_max_threshold: float = 0.02
    blank_luminance_threshold: float = 0.01
    blank_luminance_std_threshold: float = 0.01
    max_bad_fraction: float = 0.5
    required: bool = False
    sample_every: int = 1
    downsample_size: int = 64
    max_samples: int = 32

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, path: str = "content_liveness"
    ) -> ContentLivenessConfig:
        allowed = {
            "enabled",
            "static_mean_threshold",
            "static_max_threshold",
            "blank_luminance_threshold",
            "blank_luminance_std_threshold",
            "max_bad_fraction",
            "required",
            "sample_every",
            "downsample_size",
            "max_samples",
        }
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be an object")
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            raise ValueError(f"{path} has unexpected fields: {unexpected}")
        return cls(
            enabled=value.get("enabled", False),
            static_mean_threshold=value.get("static_mean_threshold", 0.002),
            static_max_threshold=value.get("static_max_threshold", 0.02),
            blank_luminance_threshold=value.get("blank_luminance_threshold", 0.01),
            blank_luminance_std_threshold=value.get("blank_luminance_std_threshold", 0.01),
            max_bad_fraction=value.get("max_bad_fraction", 0.5),
            required=value.get("required", False),
            sample_every=value.get("sample_every", 1),
            downsample_size=value.get("downsample_size", 64),
            max_samples=value.get("max_samples", 32),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("content liveness enabled must be a boolean")
        if not isinstance(self.required, bool):
            raise ValueError("content liveness required must be a boolean")
        for name in (
            "static_mean_threshold",
            "static_max_threshold",
            "blank_luminance_threshold",
            "blank_luminance_std_threshold",
            "max_bad_fraction",
        ):
            object.__setattr__(self, name, _threshold(getattr(self, name), path=name))
        if (
            not isinstance(self.sample_every, int)
            or isinstance(self.sample_every, bool)
            or self.sample_every < 1
        ):
            raise ValueError("sample_every must be a positive integer")
        if (
            not isinstance(self.downsample_size, int)
            or isinstance(self.downsample_size, bool)
            or not 4 <= self.downsample_size <= 512
        ):
            raise ValueError("downsample_size must be an integer between 4 and 512")
        if (
            not isinstance(self.max_samples, int)
            or isinstance(self.max_samples, bool)
            or not 1 <= self.max_samples <= 256
        ):
            raise ValueError("max_samples must be an integer between 1 and 256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "static_mean_threshold": self.static_mean_threshold,
            "static_max_threshold": self.static_max_threshold,
            "blank_luminance_threshold": self.blank_luminance_threshold,
            "blank_luminance_std_threshold": self.blank_luminance_std_threshold,
            "max_bad_fraction": self.max_bad_fraction,
            "required": self.required,
            "sample_every": self.sample_every,
            "downsample_size": self.downsample_size,
            "max_samples": self.max_samples,
        }


@dataclass(frozen=True, slots=True)
class ContentStatistics:
    """Numeric measurements for one sampled consecutive frame pair."""

    inter_frame_diff_mean: float
    inter_frame_diff_max: float
    luminance_mean: float
    luminance_std: float

    def __post_init__(self) -> None:
        for name in (
            "inter_frame_diff_mean",
            "inter_frame_diff_max",
            "luminance_mean",
            "luminance_std",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
            object.__setattr__(self, name, value)
        if self.inter_frame_diff_mean > self.inter_frame_diff_max:
            raise ValueError("inter-frame difference mean cannot exceed max")

    def to_mapping(self) -> dict[str, float]:
        return {
            "inter_frame_diff_mean": self.inter_frame_diff_mean,
            "inter_frame_diff_max": self.inter_frame_diff_max,
            "luminance_mean": self.luminance_mean,
            "luminance_std": self.luminance_std,
        }


@dataclass(frozen=True, slots=True)
class ContentLivenessSample:
    """One bounded telemetry sample; no pixel data is retained."""

    sample_index: int
    statistics: ContentStatistics
    state: ContentLivenessState
    observed_at_ns: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_index, int)
            or isinstance(self.sample_index, bool)
            or self.sample_index < 0
        ):
            raise ValueError("sample_index must be a non-negative integer")
        if not isinstance(self.statistics, ContentStatistics):
            raise TypeError("statistics must be ContentStatistics")
        object.__setattr__(self, "state", ContentLivenessState(self.state))
        if (
            not isinstance(self.observed_at_ns, int)
            or isinstance(self.observed_at_ns, bool)
            or self.observed_at_ns < 0
        ):
            raise ValueError("observed_at_ns must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "state": self.state.value,
            "observed_at_ns": self.observed_at_ns,
            **self.statistics.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ContentLivenessReport:
    """Bounded aggregate suitable for manifests, CLI JSON, and run stores."""

    state: ContentLivenessState
    samples: tuple[ContentLivenessSample, ...] = field(default_factory=tuple)
    reason: str = ""
    enabled: bool = False
    schema_version: str = CAPTURE_LIVENESS_SCHEMA_VERSION
    total_sample_count: int | None = None
    total_content_static_count: int | None = None
    total_content_blank_count: int | None = None
    max_bad_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_LIVENESS_SCHEMA_VERSION:
            raise ValueError(f"unsupported content liveness schema: {self.schema_version!r}")
        object.__setattr__(self, "state", ContentLivenessState(self.state))
        samples = tuple(self.samples)
        if any(not isinstance(sample, ContentLivenessSample) for sample in samples):
            raise TypeError("samples must contain ContentLivenessSample values")
        if tuple(sample.sample_index for sample in samples) != tuple(
            sorted(sample.sample_index for sample in samples)
        ):
            raise ContractViolation("content liveness sample indexes must be monotonic")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "reason", _bounded_reason(self.reason))
        if not isinstance(self.enabled, bool):
            raise ValueError("content liveness enabled must be a boolean")
        object.__setattr__(
            self, "max_bad_fraction", _threshold(self.max_bad_fraction, path="max_bad_fraction")
        )
        for name in (
            "total_sample_count",
            "total_content_static_count",
            "total_content_blank_count",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        total = len(samples) if self.total_sample_count is None else self.total_sample_count
        static = (
            sum(sample.state is ContentLivenessState.CONTENT_STATIC for sample in samples)
            if self.total_content_static_count is None
            else self.total_content_static_count
        )
        blank = (
            sum(sample.state is ContentLivenessState.CONTENT_BLANK for sample in samples)
            if self.total_content_blank_count is None
            else self.total_content_blank_count
        )
        if total < len(samples) or static + blank > total:
            raise ValueError("content liveness aggregate counts are inconsistent")
        object.__setattr__(self, "total_sample_count", total)
        object.__setattr__(self, "total_content_static_count", static)
        object.__setattr__(self, "total_content_blank_count", blank)

    @property
    def sample_count(self) -> int:
        return self.total_sample_count or 0

    @property
    def content_static_count(self) -> int:
        return self.total_content_static_count or 0

    @property
    def content_blank_count(self) -> int:
        return self.total_content_blank_count or 0

    @property
    def bad_fraction(self) -> float:
        if not self.samples:
            return 0.0
        return (self.content_static_count + self.content_blank_count) / self.sample_count

    @property
    def last_sample_index(self) -> int | None:
        return self.samples[-1].sample_index if self.samples else None

    @property
    def usable(self) -> bool:
        return self.enabled and bool(self.samples) and self.state is ContentLivenessState.LIVE

    def require_usable(
        self, *, required: bool = True, max_bad_fraction: float | None = None
    ) -> None:
        """Raise before registration when a required capture is not usable."""

        limit = (
            self.max_bad_fraction
            if max_bad_fraction is None
            else _threshold(max_bad_fraction, path="max_bad_fraction")
        )
        if not required:
            return
        if not self.enabled:
            raise ContentLivenessGateError(self, "content liveness is required but disabled")
        if not self.samples:
            raise ContentLivenessGateError(self, "content liveness has no samples")
        if self.bad_fraction > limit:
            detail = self.reason or self.state.value
            raise ContentLivenessGateError(self, f"content liveness gate failed: {detail}")

    def to_mapping(self) -> dict[str, object]:
        # Keep all values JSON scalar/array values so CLI consumers cannot
        # accidentally serialize numpy scalars or enum instances.
        return {
            "schema_version": self.schema_version,
            "enabled": self.enabled,
            "state": self.state.value,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "recent_sample_count": len(self.samples),
            "content_static_count": self.content_static_count,
            "content_blank_count": self.content_blank_count,
            "bad_fraction": self.bad_fraction,
            "max_bad_fraction": self.max_bad_fraction,
            "heartbeat": {
                "sample_count": self.sample_count,
                "last_sample_index": self.last_sample_index,
                "last_observed_at_ns": self.samples[-1].observed_at_ns if self.samples else None,
            },
            "samples": [sample.to_mapping() for sample in self.samples],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContentLivenessReport:
        expected = {
            "schema_version",
            "enabled",
            "state",
            "reason",
            "sample_count",
            "recent_sample_count",
            "content_static_count",
            "content_blank_count",
            "bad_fraction",
            "max_bad_fraction",
            "heartbeat",
            "samples",
        }
        unexpected = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if missing or unexpected:
            raise ValueError(
                f"content liveness report has missing={missing} and unexpected={unexpected} fields"
            )
        samples_value = value["samples"]
        if not isinstance(samples_value, list):
            raise TypeError("content liveness samples must be a list")
        if value["recent_sample_count"] != len(samples_value):
            raise ContractViolation("content liveness recent_sample_count does not match samples")
        heartbeat = value["heartbeat"]
        if not isinstance(heartbeat, Mapping) or set(heartbeat) != {
            "sample_count",
            "last_sample_index",
            "last_observed_at_ns",
        }:
            raise ValueError("content liveness heartbeat has invalid fields")
        samples: list[ContentLivenessSample] = []
        for item in samples_value:
            if not isinstance(item, Mapping):
                raise TypeError("content liveness sample must be an object")
            sample_fields = {
                "sample_index",
                "state",
                "observed_at_ns",
                "inter_frame_diff_mean",
                "inter_frame_diff_max",
                "luminance_mean",
                "luminance_std",
            }
            if set(item) != sample_fields:
                raise ValueError("content liveness sample has invalid fields")
            samples.append(
                ContentLivenessSample(
                    sample_index=item["sample_index"],
                    statistics=ContentStatistics(
                        inter_frame_diff_mean=item["inter_frame_diff_mean"],
                        inter_frame_diff_max=item["inter_frame_diff_max"],
                        luminance_mean=item["luminance_mean"],
                        luminance_std=item["luminance_std"],
                    ),
                    state=item["state"],
                    observed_at_ns=item["observed_at_ns"],
                )
            )
        report = cls(
            schema_version=value["schema_version"],
            enabled=value["enabled"],
            state=value["state"],
            reason=value["reason"],
            samples=tuple(samples),
            total_sample_count=value["sample_count"],
            total_content_static_count=value["content_static_count"],
            total_content_blank_count=value["content_blank_count"],
            max_bad_fraction=value["max_bad_fraction"],
        )
        expected_fraction = report.bad_fraction
        actual_fraction = float(value["bad_fraction"])
        if not math.isfinite(actual_fraction) or not math.isclose(
            expected_fraction, actual_fraction, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ContractViolation("content liveness bad_fraction does not match counts")
        if heartbeat["sample_count"] != report.sample_count:
            raise ContractViolation("content liveness heartbeat sample_count does not match report")
        if heartbeat["last_sample_index"] != report.last_sample_index:
            raise ContractViolation(
                "content liveness heartbeat last_sample_index does not match report"
            )
        expected_observed = report.samples[-1].observed_at_ns if report.samples else None
        if heartbeat["last_observed_at_ns"] != expected_observed:
            raise ContractViolation("content liveness heartbeat timestamp does not match report")
        return report


class ContentLivenessGateError(ContractViolation):
    """Raised when required capture content cannot be registered as usable."""

    def __init__(self, report: ContentLivenessReport, message: str | None = None) -> None:
        self.report = report
        super().__init__(message or "content liveness gate failed")


class ContentLivenessSink(Protocol):
    """Minimal sink protocol for run-store telemetry."""

    def record_metric(
        self, run_id: str, *, name: str, value: float, metadata: Mapping[str, Any]
    ) -> Any: ...

    def append_event(
        self, run_id: str, *, kind: str, payload: Mapping[str, Any], timestamp_ns: int
    ) -> Any: ...


def _frame_array(frame: Any, *, path: str) -> np.ndarray[Any, Any]:
    array = np.asarray(frame)
    if array.ndim not in {2, 3} or array.size == 0:
        raise ValueError(f"{path} must be a non-empty HxW or HxWxC frame")
    if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"{path} channel count must be 1, 3, or 4")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{path} must contain numeric pixels")
    values = array.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} pixels must be finite")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(info.max, 1))
        values = values / scale
    elif float(values.max(initial=0.0)) > 1.0:
        if float(values.max(initial=0.0)) > 255.0:
            raise ValueError(f"{path} float pixels must be in [0, 1] or [0, 255]")
        values = values / 255.0
    if float(values.min(initial=0.0)) < 0.0 or float(values.max(initial=0.0)) > 1.0:
        raise ValueError(f"{path} normalized pixels must be in [0, 1]")
    return values


def _downsample(array: np.ndarray[Any, Any], size: int) -> np.ndarray[Any, Any]:
    height, width = array.shape[:2]
    if height <= size and width <= size:
        return array
    row_indexes = np.linspace(0, height - 1, num=min(height, size), dtype=np.intp)
    column_indexes = np.linspace(0, width - 1, num=min(width, size), dtype=np.intp)
    return array[row_indexes][:, column_indexes]


def _luminance(frame: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if frame.ndim == 2 or frame.shape[2] == 1:
        return frame if frame.ndim == 2 else frame[..., 0]
    # ITU-R BT.709 weights; alpha, when present, is intentionally ignored.
    return np.tensordot(
        frame[..., :3], np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=1
    )


def measure_content(
    previous_frame: Any, current_frame: Any, *, downsample_size: int = 64
) -> ContentStatistics:
    """Compute normalized differences and luminance stats for one frame pair."""

    if (
        not isinstance(downsample_size, int)
        or isinstance(downsample_size, bool)
        or not 4 <= downsample_size <= 512
    ):
        raise ValueError("downsample_size must be an integer between 4 and 512")
    previous = _downsample(_frame_array(previous_frame, path="previous_frame"), downsample_size)
    current = _downsample(_frame_array(current_frame, path="current_frame"), downsample_size)
    if previous.shape != current.shape:
        raise ValueError("previous_frame and current_frame must have the same shape")
    difference = np.abs(current - previous)
    luminance = _luminance(current)
    return ContentStatistics(
        inter_frame_diff_mean=float(difference.mean()),
        inter_frame_diff_max=float(difference.max(initial=0.0)),
        luminance_mean=float(luminance.mean()),
        luminance_std=float(luminance.std()),
    )


class ContentLivenessMonitor:
    """Sparse monitor retaining only fixed-size scalar samples."""

    def __init__(self, config: ContentLivenessConfig | None = None) -> None:
        self.config = config or ContentLivenessConfig()
        self._pair_count = 0
        self._samples: deque[ContentLivenessSample] = deque(maxlen=self.config.max_samples)
        self._sampled_count = 0
        self._static_count = 0
        self._blank_count = 0

    @property
    def sample_count(self) -> int:
        return self._sampled_count

    def observe(
        self,
        previous_frame: Any,
        current_frame: Any,
        *,
        frame_pair_index: int | None = None,
        observed_at_ns: int | None = None,
    ) -> ContentLivenessSample | None:
        """Sample a pair when enabled and due; pixels are released on return."""

        if not self.config.enabled:
            return None
        pair_index = self._pair_count if frame_pair_index is None else frame_pair_index
        if not isinstance(pair_index, int) or isinstance(pair_index, bool) or pair_index < 0:
            raise ValueError("frame_pair_index must be a non-negative integer or None")
        self._pair_count = max(self._pair_count, pair_index + 1)
        if pair_index % self.config.sample_every:
            return None
        statistics = measure_content(
            previous_frame, current_frame, downsample_size=self.config.downsample_size
        )
        if (
            statistics.luminance_mean <= self.config.blank_luminance_threshold
            and statistics.luminance_std <= self.config.blank_luminance_std_threshold
        ):
            state = ContentLivenessState.CONTENT_BLANK
        elif (
            statistics.inter_frame_diff_mean <= self.config.static_mean_threshold
            and statistics.inter_frame_diff_max <= self.config.static_max_threshold
        ):
            state = ContentLivenessState.CONTENT_STATIC
        else:
            state = ContentLivenessState.LIVE
        sample = ContentLivenessSample(
            sample_index=pair_index,
            statistics=statistics,
            state=state,
            observed_at_ns=time_ns() if observed_at_ns is None else observed_at_ns,
        )
        self._sampled_count += 1
        if state is ContentLivenessState.CONTENT_STATIC:
            self._static_count += 1
        elif state is ContentLivenessState.CONTENT_BLANK:
            self._blank_count += 1
        self._samples.append(sample)
        return sample

    def report(self) -> ContentLivenessReport:
        if not self.config.enabled:
            return ContentLivenessReport(
                state=ContentLivenessState.DISABLED,
                enabled=False,
                reason="content liveness disabled",
                max_bad_fraction=self.config.max_bad_fraction,
            )
        samples = tuple(self._samples)
        if self._sampled_count == 0:
            return ContentLivenessReport(
                state=ContentLivenessState.DEGRADED,
                samples=samples,
                enabled=True,
                reason="content liveness has no samples",
                max_bad_fraction=self.config.max_bad_fraction,
            )
        static_count = self._static_count
        blank_count = self._blank_count
        bad_count = static_count + blank_count
        static_indexes = tuple(
            sample.sample_index
            for sample in samples
            if sample.state is ContentLivenessState.CONTENT_STATIC
        )
        blank_indexes = tuple(
            sample.sample_index
            for sample in samples
            if sample.state is ContentLivenessState.CONTENT_BLANK
        )

        def describe(kind: str, indexes: tuple[int, ...]) -> str:
            rendered = ",".join(str(index) for index in indexes[:12])
            if len(indexes) > 12:
                rendered += ",..."
            return f"{kind}:samples={rendered}"

        if bad_count == 0:
            state = ContentLivenessState.LIVE
            reason = ""
        elif blank_count == self._sampled_count:
            state = ContentLivenessState.CONTENT_BLANK
            reason = describe("content_blank", blank_indexes)
        elif static_count == self._sampled_count:
            state = ContentLivenessState.CONTENT_STATIC
            reason = describe("content_static", static_indexes)
        else:
            state = ContentLivenessState.DEGRADED
            reason = "degraded:" + ";".join(
                part
                for part in (
                    describe("content_static", static_indexes) if static_indexes else "",
                    describe("content_blank", blank_indexes) if blank_indexes else "",
                )
                if part
            )
        return ContentLivenessReport(
            state=state,
            samples=samples,
            enabled=True,
            reason=reason,
            total_sample_count=self._sampled_count,
            total_content_static_count=static_count,
            total_content_blank_count=blank_count,
            max_bad_fraction=self.config.max_bad_fraction,
        )

    def gate(self, *, required: bool | None = None) -> ContentLivenessReport:
        report = self.report()
        report.require_usable(
            required=self.config.required if required is None else required,
            max_bad_fraction=self.config.max_bad_fraction,
        )
        return report


def evaluate_content_liveness(
    frames: Iterable[Any], config: ContentLivenessConfig | None = None
) -> ContentLivenessReport:
    """Evaluate consecutive frames without retaining the input sequence."""

    monitor = ContentLivenessMonitor(config)
    iterator = iter(frames)
    try:
        previous = next(iterator)
    except StopIteration:
        return monitor.report()
    for current in iterator:
        monitor.observe(previous, current)
        previous = current
    return monitor.report()


def record_content_liveness(
    store: ContentLivenessSink,
    run_id: str,
    report: ContentLivenessReport,
    *,
    timestamp_ns: int | None = None,
) -> tuple[tuple[MetricRecord, ...], RunEvent | None]:
    """Persist report reason and measured values as run-store evidence."""

    timestamp = time_ns() if timestamp_ns is None else timestamp_ns
    if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
        raise ValueError("timestamp_ns must be a non-negative integer")
    metadata = {
        "schema_version": report.schema_version,
        "state": report.state.value,
        "reason": report.reason,
        "sample_count": report.sample_count,
        "content_static_count": report.content_static_count,
        "content_blank_count": report.content_blank_count,
    }
    metrics: list[MetricRecord] = []
    metrics.extend(
        (
            store.record_metric(
                run_id,
                name="capture.content_liveness.bad_fraction",
                value=report.bad_fraction,
                metadata=metadata,
            ),
            store.record_metric(
                run_id,
                name="capture.content_liveness.sample_count",
                value=float(report.sample_count),
                metadata=metadata,
            ),
        )
    )
    for name, values in (
        (
            "capture.content_liveness.inter_frame_diff_mean",
            [s.statistics.inter_frame_diff_mean for s in report.samples],
        ),
        (
            "capture.content_liveness.inter_frame_diff_max",
            [s.statistics.inter_frame_diff_max for s in report.samples],
        ),
        (
            "capture.content_liveness.luminance_mean",
            [s.statistics.luminance_mean for s in report.samples],
        ),
        (
            "capture.content_liveness.luminance_std",
            [s.statistics.luminance_std for s in report.samples],
        ),
    ):
        if values:
            metrics.append(
                store.record_metric(run_id, name=name, value=float(values[-1]), metadata=metadata)
            )
    event = store.append_event(
        run_id,
        kind="capture.content_liveness",
        payload=report.to_mapping(),
        timestamp_ns=timestamp,
    )
    return tuple(metrics), event


__all__ = [
    "CAPTURE_LIVENESS_SCHEMA_VERSION",
    "ContentLivenessConfig",
    "ContentLivenessGateError",
    "ContentLivenessMonitor",
    "ContentLivenessReport",
    "ContentLivenessSample",
    "ContentLivenessState",
    "ContentStatistics",
    "evaluate_content_liveness",
    "measure_content",
    "record_content_liveness",
]
