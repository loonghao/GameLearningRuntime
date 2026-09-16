from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from game_learning_runtime import (
    BridgeEnvironment,
    EnvironmentBridgeDriver,
    EnvironmentReadinessError,
    ReadinessAttempt,
    ReadinessMonitor,
    ReadinessResult,
    ReadinessState,
    ReadinessWindowOutcome,
    ReadinessWindowVerdict,
    readiness_from_mapping,
    run_readiness_window,
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


def _attempt(
    index: int, exit_code: int, state: ReadinessState | None, reason: str = "host state"
) -> ReadinessAttempt:
    return ReadinessAttempt(
        index=index,
        exit_code=exit_code,
        result=(None if state is None else ReadinessResult(state, reason, checked_at_ns=index)),
    )


def test_readiness_mapping_round_trips_and_rejects_off_schema_values() -> None:
    receipt = ReadinessResult(ReadinessState.NOT_READY, "window absent", checked_at_ns=5)
    assert readiness_from_mapping(receipt.to_mapping()) == receipt
    extra = {**receipt.to_mapping(), "adapter": "ignored"}
    assert readiness_from_mapping(extra).reason == "window absent"
    defaulted = readiness_from_mapping(
        {"schema_version": READINESS_SCHEMA_VERSION, "state": "ready"}
    )
    assert defaulted.checked_at_ns > 0
    for value, error in (
        ({"schema_version": "glr.environment-readiness.v2", "state": "ready"}, "schema_version"),
        ({"schema_version": READINESS_SCHEMA_VERSION, "state": "maybe"}, "state must be one of"),
        ({"schema_version": READINESS_SCHEMA_VERSION}, "state must be a readiness state string"),
        (
            {"schema_version": READINESS_SCHEMA_VERSION, "state": "ready", "reason": 3},
            "reason must be a string",
        ),
        (
            {
                "schema_version": READINESS_SCHEMA_VERSION,
                "state": "ready",
                "checked_at_ns": -1,
            },
            "checked_at_ns",
        ),
        ("ready", "must be an object"),
    ):
        with pytest.raises((TypeError, ValueError), match=error):
            readiness_from_mapping(value)


def test_readiness_window_retries_only_while_the_host_is_parking() -> None:
    observed: list[int] = []

    def attempt(index: int) -> ReadinessAttempt:
        observed.append(index)
        if index > 2:
            return _attempt(index, 0, ReadinessState.READY)
        return _attempt(index, 63, ReadinessState.NOT_READY)

    paused: list[float] = []
    outcome = run_readiness_window(
        timeout_seconds=30,
        poll_interval_seconds=2,
        attempt=attempt,
        sleep_seconds=paused.append,
    )
    assert observed == [1, 2, 3]
    assert paused == [2, 2]
    assert outcome.verdict is ReadinessWindowVerdict.SUCCEEDED
    assert outcome.succeeded and not outcome.exhausted
    assert outcome.last_result is not None and outcome.last_result.ready
    assert outcome.to_mapping() == {
        "verdict": "succeeded",
        "exhausted": False,
        "timeout_seconds": 30,
        "attempts": [
            {
                "index": 1,
                "exit_code": 63,
                "readiness": _attempt(1, 63, ReadinessState.NOT_READY).to_mapping()["readiness"],
            },
            {
                "index": 2,
                "exit_code": 63,
                "readiness": _attempt(2, 63, ReadinessState.NOT_READY).to_mapping()["readiness"],
            },
            {
                "index": 3,
                "exit_code": 0,
                "readiness": _attempt(3, 0, ReadinessState.READY).to_mapping()["readiness"],
            },
        ],
    }


def test_readiness_window_expires_without_calling_a_parked_host_a_failure() -> None:
    def attempt(index: int) -> ReadinessAttempt:
        return _attempt(index, 63, ReadinessState.NOT_READY, "still booting")

    outcome = run_readiness_window(
        timeout_seconds=0.05,
        poll_interval_seconds=0.01,
        attempt=attempt,
        sleep_seconds=lambda seconds: None,
    )
    assert outcome.verdict is ReadinessWindowVerdict.NOT_READY
    assert outcome.exhausted and not outcome.succeeded
    assert outcome.last_result is not None
    assert outcome.last_result.state is ReadinessState.NOT_READY
    assert outcome.to_mapping()["exhausted"] is True
    assert "schema_version" not in outcome.to_mapping()


def _terminal_attempt(
    exit_code: int, state: ReadinessState | None, calls: list[int]
) -> Callable[[int], ReadinessAttempt]:
    def attempt(index: int) -> ReadinessAttempt:
        calls.append(index)
        return _attempt(index, exit_code, state)

    return attempt


def test_readiness_window_is_terminal_on_every_non_retryable_receipt() -> None:
    for state, exit_code, verdict in (
        (None, 17, ReadinessWindowVerdict.UNREPORTED),
        (ReadinessState.UNAVAILABLE, 19, ReadinessWindowVerdict.UNAVAILABLE),
        (ReadinessState.READY, 23, ReadinessWindowVerdict.INCONSISTENT),
    ):
        calls: list[int] = []
        outcome = run_readiness_window(
            timeout_seconds=30,
            poll_interval_seconds=0.01,
            attempt=_terminal_attempt(exit_code, state, calls),
        )
        assert outcome.verdict is verdict
        assert not outcome.exhausted
        assert calls == [1]
        assert (outcome.last_result is None) is (state is None)


def test_readiness_window_rejects_unbounded_bounds_and_bad_attempts() -> None:
    attempt = lambda index: _attempt(index, 0, None)  # noqa: E731
    for timeout, poll, match in ((0, 1, "must be positive"), (1, 0, "must be positive")):
        with pytest.raises(ValueError, match=match):
            run_readiness_window(
                timeout_seconds=timeout, poll_interval_seconds=poll, attempt=attempt
            )
    with pytest.raises(TypeError, match="ReadinessAttempt"):
        run_readiness_window(
            timeout_seconds=1, poll_interval_seconds=1, attempt=lambda index: object()
        )
    with pytest.raises(ValueError, match="positive integer"):
        ReadinessAttempt(index=0, exit_code=0)
    with pytest.raises(ValueError, match="exit_code must be an integer"):
        ReadinessAttempt(index=1, exit_code=True)
    with pytest.raises(TypeError, match="ReadinessResult or None"):
        ReadinessAttempt(index=1, exit_code=0, result=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="verdict"):
        ReadinessWindowOutcome(
            verdict="succeeded",  # type: ignore[arg-type]
            attempts=(_attempt(1, 0, None),),
            timeout_seconds=1,
        )
    with pytest.raises(ValueError, match="at least one attempt"):
        ReadinessWindowOutcome(
            verdict=ReadinessWindowVerdict.SUCCEEDED, attempts=(), timeout_seconds=1
        )
    with pytest.raises(ValueError, match="numbered from one"):
        ReadinessWindowOutcome(
            verdict=ReadinessWindowVerdict.SUCCEEDED,
            attempts=(_attempt(2, 0, None),),
            timeout_seconds=1,
        )
