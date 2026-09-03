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
