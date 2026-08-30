from __future__ import annotations

import json

import pytest

from benchmarks.data_plane import (
    BenchmarkMeasurement,
    assess_rust_candidate,
    measure,
    run_suite,
    safe_runtime_metadata,
)


def _measurement(*, throughput: float, p95: float) -> BenchmarkMeasurement:
    return BenchmarkMeasurement(
        name="synthetic",
        samples=10,
        operations=100,
        elapsed_seconds=1.0,
        throughput_ops_s=throughput,
        p50_latency_us=p95 / 2.0,
        p95_latency_us=p95,
    )


def test_measure_uses_warmups_and_reports_positive_per_operation_statistics() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    result = measure(
        "counter",
        operation,
        samples=5,
        operations_per_sample=4,
        warmup_operations=3,
    )

    assert calls == 23
    assert result.name == "counter"
    assert result.samples == 5
    assert result.operations == 20
    assert result.elapsed_seconds > 0.0
    assert result.throughput_ops_s > 0.0
    assert 0.0 < result.p50_latency_us <= result.p95_latency_us


@pytest.mark.parametrize(
    ("candidate", "qualifies", "reason"),
    [
        (_measurement(throughput=200.0, p95=100.0), True, "throughput"),
        (_measurement(throughput=110.0, p95=70.0), True, "p95_latency"),
        (_measurement(throughput=199.0, p95=71.0), False, None),
    ],
)
def test_rust_adoption_assessment_applies_exact_adr_thresholds(
    candidate: BenchmarkMeasurement,
    qualifies: bool,
    reason: str | None,
) -> None:
    baseline = _measurement(throughput=100.0, p95=100.0)

    result = assess_rust_candidate(baseline=baseline, candidate=candidate)

    assert result.qualifies is qualifies
    assert result.reason == reason
    assert result.throughput_ratio == pytest.approx(candidate.throughput_ops_s / 100.0)
    assert result.p95_reduction_percent == pytest.approx(100.0 - candidate.p95_latency_us)


def test_safe_runtime_metadata_is_useful_without_machine_identifiers() -> None:
    metadata = safe_runtime_metadata()

    assert set(metadata) == {"architecture", "platform", "python"}
    assert all(metadata.values())
    serialized = json.dumps(metadata).lower()
    for forbidden in ("hostname", "node", "path", "pid", "user", "\\", "/home/"):
        assert forbidden not in serialized


def test_minimal_suite_uses_only_synthetic_named_workloads() -> None:
    report = run_suite(
        json_backend="stdlib",
        samples=2,
        operations_per_sample=2,
        observation_width=8,
        queue_capacity=4,
    )

    assert report["schema"] == "glr.benchmark.data-plane.v1"
    assert report["backend"] == "stdlib"
    assert set(report["measurements"]) == {
        "actor_queue_round_trip",
        "trajectory_build",
        "transition_json_round_trip",
    }
    serialized = json.dumps(report).lower()
    for forbidden in ("hostname", "private-runtime", "episode_id", "timestamp_ns", "path"):
        assert forbidden not in serialized
