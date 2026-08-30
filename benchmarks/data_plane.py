"""Reproducible synthetic benchmarks for candidate GLR data-plane hot paths."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from queue import Queue
from time import perf_counter_ns
from typing import Any, Protocol, cast
from uuid import UUID

import numpy as np

from game_learning_runtime.contracts import Transition, Unroll
from game_learning_runtime.serialization import transition_from_record, transition_to_record

REPORT_SCHEMA = "glr.benchmark.data-plane.v1"
TRAJECTORY_LENGTH = 32


class _JsonBackend(Protocol):
    def dumps(self, value: Mapping[str, Any]) -> bytes: ...

    def loads(self, value: bytes) -> Mapping[str, Any]: ...


class _OrjsonModule(Protocol):
    def dumps(self, value: Any) -> bytes: ...

    def loads(self, value: bytes) -> Any: ...


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    """Latency and throughput summary for one named synthetic workload."""

    name: str
    samples: int
    operations: int
    elapsed_seconds: float
    throughput_ops_s: float
    p50_latency_us: float
    p95_latency_us: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RustAdoptionAssessment:
    """Mechanical application of ADR-0004's Rust adoption threshold."""

    qualifies: bool
    reason: str | None
    throughput_ratio: float
    p95_reduction_percent: float


class _StdlibJson:
    def dumps(self, value: Mapping[str, Any]) -> bytes:
        return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def loads(self, value: bytes) -> Mapping[str, Any]:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("JSON backend returned a non-object transition record")
        return decoded


class _Orjson:
    def __init__(self) -> None:
        try:
            module = import_module("orjson")
        except ModuleNotFoundError as error:  # pragma: no cover - depends on benchmark environment
            raise RuntimeError(
                "orjson benchmark requires `uv run --with orjson python -m benchmarks.data_plane`"
            ) from error
        self._orjson = cast(_OrjsonModule, module)

    def dumps(self, value: Mapping[str, Any]) -> bytes:
        return self._orjson.dumps(value)

    def loads(self, value: bytes) -> Mapping[str, Any]:
        decoded = self._orjson.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("JSON backend returned a non-object transition record")
        return decoded


def _json_backend(name: str) -> _JsonBackend:
    if name == "stdlib":
        return _StdlibJson()
    if name == "orjson":
        return _Orjson()
    raise ValueError("json_backend must be 'stdlib' or 'orjson'")


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def measure(
    name: str,
    operation: Callable[[], None],
    *,
    samples: int,
    operations_per_sample: int,
    warmup_operations: int = 10,
) -> BenchmarkMeasurement:
    """Measure one operation without collecting machine identity or wall time."""

    if not name:
        raise ValueError("benchmark name cannot be empty")
    if samples <= 0 or operations_per_sample <= 0:
        raise ValueError("samples and operations_per_sample must be positive")
    if warmup_operations < 0:
        raise ValueError("warmup_operations cannot be negative")
    for _ in range(warmup_operations):
        operation()

    per_operation_us: list[float] = []
    elapsed_ns = 0
    for _ in range(samples):
        started = perf_counter_ns()
        for _ in range(operations_per_sample):
            operation()
        sample_elapsed_ns = max(1, perf_counter_ns() - started)
        elapsed_ns += sample_elapsed_ns
        per_operation_us.append(sample_elapsed_ns / operations_per_sample / 1_000.0)

    operations = samples * operations_per_sample
    elapsed_seconds = elapsed_ns / 1_000_000_000.0
    return BenchmarkMeasurement(
        name=name,
        samples=samples,
        operations=operations,
        elapsed_seconds=elapsed_seconds,
        throughput_ops_s=operations / elapsed_seconds,
        p50_latency_us=_percentile(per_operation_us, 0.50),
        p95_latency_us=_percentile(per_operation_us, 0.95),
    )


def assess_rust_candidate(
    *, baseline: BenchmarkMeasurement, candidate: BenchmarkMeasurement
) -> RustAdoptionAssessment:
    """Apply ADR-0004: 2x throughput or at least 30 percent lower p95."""

    if baseline.name != candidate.name:
        raise ValueError("baseline and candidate workload names must match")
    if baseline.throughput_ops_s <= 0.0 or baseline.p95_latency_us <= 0.0:
        raise ValueError("baseline throughput and p95 latency must be positive")
    throughput_ratio = candidate.throughput_ops_s / baseline.throughput_ops_s
    p95_reduction_percent = (
        (baseline.p95_latency_us - candidate.p95_latency_us) / baseline.p95_latency_us * 100.0
    )
    if throughput_ratio >= 2.0:
        reason = "throughput"
    elif p95_reduction_percent >= 30.0:
        reason = "p95_latency"
    else:
        reason = None
    return RustAdoptionAssessment(
        qualifies=reason is not None,
        reason=reason,
        throughput_ratio=throughput_ratio,
        p95_reduction_percent=p95_reduction_percent,
    )


def safe_runtime_metadata() -> dict[str, str]:
    """Return only coarse runtime class data; never machine or user identity."""

    return {
        "architecture": platform.machine() or "unknown",
        "platform": sys.platform,
        "python": platform.python_version(),
    }


def _synthetic_transition(observation_width: int) -> Transition:
    if observation_width <= 0:
        raise ValueError("observation_width must be positive")
    observation = np.linspace(-1.0, 1.0, observation_width, dtype=np.float32)
    action_mask = np.ones(16, dtype=np.bool_)
    return Transition(
        episode_id=UUID(int=1),
        step_id=7,
        timestamp_ns=1,
        observation={"vector": observation},
        action={"choice": np.array([3], dtype=np.int64)},
        action_mask={"choice": action_mask},
        reward=np.array([0.25], dtype=np.float32),
        next_observation={"vector": observation + np.float32(0.01)},
        next_action_mask={"choice": action_mask},
        terminated=np.array([False], dtype=np.bool_),
        truncated=np.array([False], dtype=np.bool_),
        info={},
    )


def run_suite(
    *,
    json_backend: str,
    samples: int,
    operations_per_sample: int,
    observation_width: int,
    queue_capacity: int,
) -> dict[str, Any]:
    """Run privacy-safe codec, trajectory, and bounded-queue workloads."""

    if queue_capacity <= 0:
        raise ValueError("queue_capacity must be positive")
    backend = _json_backend(json_backend)
    transition = _synthetic_transition(observation_width)
    trajectory_seed = tuple(transition for _ in range(TRAJECTORY_LENGTH))
    queue: Queue[Unroll] = Queue(maxsize=queue_capacity)
    queued_unroll = Unroll(
        trajectory_seed,
        actor_id="synthetic-actor",
        sequence_id=0,
        policy_version=0,
    )

    def transition_json_round_trip() -> None:
        record = transition_to_record(transition)
        payload = backend.dumps(record)
        restored = transition_from_record(backend.loads(payload))
        if restored.step_id != transition.step_id:
            raise AssertionError("transition JSON round trip changed the step")

    def trajectory_build() -> None:
        transitions = tuple(_synthetic_transition(observation_width) for _ in trajectory_seed)
        unroll = Unroll(
            transitions,
            actor_id="synthetic-actor",
            sequence_id=1,
            policy_version=1,
        )
        if len(unroll.transitions) != TRAJECTORY_LENGTH:
            raise AssertionError("synthetic trajectory has the wrong length")

    def actor_queue_round_trip() -> None:
        queue.put_nowait(queued_unroll)
        restored = queue.get_nowait()
        if restored is not queued_unroll:
            raise AssertionError("bounded queue changed the queued object")

    warmup_operations = min(10, operations_per_sample)
    measurements = {
        name: measure(
            name,
            operation,
            samples=samples,
            operations_per_sample=operations_per_sample,
            warmup_operations=warmup_operations,
        )
        for name, operation in (
            ("transition_json_round_trip", transition_json_round_trip),
            ("trajectory_build", trajectory_build),
            ("actor_queue_round_trip", actor_queue_round_trip),
        )
    }
    return {
        "schema": REPORT_SCHEMA,
        "backend": json_backend,
        "runtime": safe_runtime_metadata(),
        "workload": {
            "observation_width": observation_width,
            "queue_capacity": queue_capacity,
            "trajectory_length": TRAJECTORY_LENGTH,
        },
        "measurements": {name: value.to_dict() for name, value in measurements.items()},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-backend", choices=("stdlib", "orjson"), default="stdlib")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--operations-per-sample", type=int, default=20)
    parser.add_argument("--observation-width", type=int, default=256)
    parser.add_argument("--queue-capacity", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_suite(
        json_backend=args.json_backend,
        samples=args.samples,
        operations_per_sample=args.operations_per_sample,
        observation_width=args.observation_width,
        queue_capacity=args.queue_capacity,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "BenchmarkMeasurement",
    "RustAdoptionAssessment",
    "assess_rust_candidate",
    "main",
    "measure",
    "run_suite",
    "safe_runtime_metadata",
]
