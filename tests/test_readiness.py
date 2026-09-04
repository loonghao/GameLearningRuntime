from __future__ import annotations

from dataclasses import replace

import pytest

from game_learning_runtime import (
    BridgeEnvironment,
    EnvironmentBridgeDriver,
    EnvironmentReadinessError,
    ReadinessMonitor,
    ReadinessResult,
    ReadinessState,
)
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.readiness import READINESS_SCHEMA_VERSION


def test_readiness_result_is_bounded_and_serializable() -> None:
    result = ReadinessResult(ReadinessState.NOT_READY, "display unavailable", checked_at_ns=4)
    assert result.to_mapping()["state"] == "not_ready"
    assert not result.ready
    with pytest.raises(ValueError, match="256"):
        ReadinessResult(ReadinessState.UNAVAILABLE, "x" * 257)


def test_monitor_require_ready_fails_closed_and_remembers_result() -> None:
    monitor = ReadinessMonitor(lambda: ReadinessResult(ReadinessState.UNAVAILABLE, "locked"))
    with pytest.raises(EnvironmentReadinessError, match="locked") as error:
        monitor.require_ready()
    assert error.value.result.state is ReadinessState.UNAVAILABLE
    assert monitor.last_result is error.value.result


def test_bridge_readiness_gate_runs_before_attach() -> None:
    class AttachEnvironment(CounterEnvironment):
        @property
        def spec(self):
            return replace(super().spec, capabilities=super().spec.capabilities | {"live-attach"})

        def attach(self, *, options=None):
            return super().reset(options=options)

    calls: list[str] = []

    def probe() -> ReadinessResult:
        calls.append("probe")
        return ReadinessResult(ReadinessState.NOT_READY, "target not ticking")

    environment = BridgeEnvironment(
        EnvironmentBridgeDriver(AttachEnvironment()),
        readiness_probe=probe,
    )
    with pytest.raises(EnvironmentReadinessError):
        environment.attach()
    assert calls == ["probe"]


def test_readiness_converts_states_and_rejects_bad_values() -> None:
    assert ReadinessResult("ready", checked_at_ns=1).ready  # type: ignore[arg-type]
    assert (
        ReadinessResult(ReadinessState.READY, checked_at_ns=1).to_mapping()["schema_version"]
        == READINESS_SCHEMA_VERSION
    )
    with pytest.raises(ValueError, match="unsupported readiness state"):
        ReadinessResult("unknown", checked_at_ns=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="control"):
        ReadinessResult(ReadinessState.NOT_READY, "bad\nreason", checked_at_ns=1)
    with pytest.raises(ValueError, match="negative"):
        ReadinessResult(ReadinessState.READY, checked_at_ns=-1)
    with pytest.raises(ValueError, match="non-ready"):
        from game_learning_runtime.readiness import EnvironmentReadinessError

        EnvironmentReadinessError(ReadinessResult(ReadinessState.READY, checked_at_ns=1))


def test_readiness_monitor_supports_probe_objects_and_bounded_wait() -> None:
    class Probe:
        def __init__(self) -> None:
            self.calls = 0

        def probe(self) -> ReadinessResult:
            self.calls += 1
            return ReadinessResult(
                ReadinessState.READY if self.calls > 1 else ReadinessState.NOT_READY,
                checked_at_ns=self.calls,
            )

    probe = Probe()
    monitor = ReadinessMonitor(probe)
    with pytest.raises(ValueError, match="timeout_seconds"):
        monitor.wait_until_ready(timeout_seconds=-1, poll_interval_seconds=0.01)
    with pytest.raises(ValueError, match="poll_interval"):
        monitor.wait_until_ready(timeout_seconds=0, poll_interval_seconds=0)
    assert monitor.wait_until_ready(timeout_seconds=1, poll_interval_seconds=0.001).ready
    assert probe.calls == 2

    ready = ReadinessMonitor(lambda: ReadinessResult(ReadinessState.READY, checked_at_ns=1))
    assert ready.require_ready().ready
    unavailable = ReadinessMonitor(
        lambda: ReadinessResult(ReadinessState.UNAVAILABLE, "still locked", checked_at_ns=1)
    )
    with pytest.raises(EnvironmentReadinessError, match="still locked"):
        unavailable.wait_until_ready(timeout_seconds=0, poll_interval_seconds=0.001)

    bad = ReadinessMonitor(lambda: object())
    with pytest.raises(TypeError, match="ReadinessResult"):
        bad.check()
