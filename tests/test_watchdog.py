from __future__ import annotations

from pathlib import Path

import pytest

from game_learning_runtime import (
    WATCHDOG_EXIT_ESCALATED,
    WATCHDOG_EXIT_RECOVERED,
    Heartbeat,
    HeartbeatLog,
    ProcessIdentity,
    ProcessSupervisor,
    SupervisionWatchdog,
    WatchdogAction,
    WatchdogPolicy,
    WatchdogReport,
    WatchdogStateError,
    WatchdogStatus,
    WatchdogTarget,
    watchdog_policy_from_mapping,
)

NS = 1_000_000_000


class _Clock:
    def __init__(self, now: int = 0) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _policy(**overrides: float | int) -> WatchdogPolicy:
    defaults: dict[str, float | int] = {
        "heartbeat_timeout_seconds": 10.0,
        "max_missed_heartbeats": 3,
        "restart_limit": 2,
        "restart_backoff_seconds": 0.0,
        "restart_cooldown_seconds": 0.0,
    }
    defaults.update(overrides)
    return WatchdogPolicy(**defaults)  # type: ignore[arg-type]


def _build(
    *,
    policy: WatchdogPolicy | None = None,
    recovery_command: tuple[str, ...] | None = None,
    supervisor: ProcessSupervisor | None = None,
    recovery_result: bool = True,
):
    """Build a watchdog with one `trainer` target registered at t=0."""

    resolved = policy or _policy()
    calls: list[tuple[list[str], float]] = []
    clock = _Clock()

    def runner(argv: object, *, timeout_seconds: float) -> bool:
        assert isinstance(argv, (list, tuple))
        calls.append((list(argv), timeout_seconds))
        return recovery_result

    watchdog = SupervisionWatchdog(
        policy=resolved,
        clock=clock,
        sleep_fn=lambda _seconds: None,
        recovery_runner=runner,
    )
    watchdog.register(
        WatchdogTarget(
            name="trainer",
            policy=resolved,
            supervisor=supervisor,
            recovery_command=recovery_command,
        )
    )
    return watchdog, calls, clock


def test_policy_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="heartbeat_timeout_seconds"):
        WatchdogPolicy(heartbeat_timeout_seconds=0)
    with pytest.raises(ValueError, match="max_missed_heartbeats"):
        WatchdogPolicy(max_missed_heartbeats=0)
    with pytest.raises(ValueError, match="restart_limit"):
        WatchdogPolicy(restart_limit=-1)


def test_policy_starvation_window_is_timeout_times_missed() -> None:
    policy = _policy(heartbeat_timeout_seconds=5.0, max_missed_heartbeats=4)
    assert policy.starvation_seconds == 20.0


def test_heartbeat_round_trips_through_mapping() -> None:
    heartbeat = Heartbeat("trainer", 7, 42, state="running", detail="ok")
    assert Heartbeat.from_mapping(heartbeat.to_mapping()) == heartbeat


def test_heartbeat_from_mapping_rejects_unknown_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="unexpected fields"):
        Heartbeat.from_mapping({"source": "a", "sequence": 1, "observed_at_ns": 0, "extra": 1})
    with pytest.raises(ValueError, match="missing fields"):
        Heartbeat.from_mapping({"source": "a"})


def test_heartbeat_rejects_blank_source() -> None:
    with pytest.raises(ValueError, match="heartbeat source"):
        Heartbeat("  ", 0, 0)


def test_current_heartbeat_is_healthy() -> None:
    watchdog, _, clock = _build()
    clock.now = 100 * NS
    watchdog.observe(Heartbeat("trainer", 1, 99 * NS))
    decision = watchdog.evaluate("trainer")
    assert decision.status is WatchdogStatus.HEALTHY
    assert decision.action is WatchdogAction.NONE
    assert decision.reason == "heartbeat-current"


def test_late_heartbeat_is_degraded_and_only_notifies() -> None:
    watchdog, _, clock = _build()
    clock.now = 100 * NS
    watchdog.observe(Heartbeat("trainer", 1, 85 * NS))
    decision = watchdog.evaluate("trainer")
    assert decision.status is WatchdogStatus.DEGRADED
    assert decision.action is WatchdogAction.NOTIFY
    assert decision.missed_heartbeats == 1


def test_missing_heartbeat_within_grace_is_degraded() -> None:
    watchdog, _, clock = _build()
    clock.now = 20 * NS
    decision = watchdog.evaluate("trainer")
    assert decision.status is WatchdogStatus.DEGRADED
    assert decision.reason == "no-heartbeat-yet"
    assert decision.action is WatchdogAction.NOTIFY


def test_starved_source_without_recovery_wiring_escalates() -> None:
    watchdog, calls, clock = _build()
    clock.now = 500 * NS
    report = watchdog.tick()
    assert report.decisions[0].status is WatchdogStatus.STARVED
    assert report.decisions[0].action is WatchdogAction.ESCALATE
    assert report.decisions[0].reason == "detect-only-no-recovery-wiring"
    assert calls == []
    assert report.exit_code == WATCHDOG_EXIT_ESCALATED


def test_starved_source_restarts_once_and_reports_recovery() -> None:
    watchdog, calls, clock = _build(recovery_command=("glr", "train"))
    clock.now = 500 * NS
    report = watchdog.tick()
    assert report.decisions[0].action is WatchdogAction.RESTART
    assert report.decisions[0].status is WatchdogStatus.RECOVERING
    assert watchdog.restart_count("trainer") == 1
    assert calls == [(["glr", "train"], 60.0)]
    assert report.exit_code == WATCHDOG_EXIT_RECOVERED


def test_failed_restart_escalates_when_budget_is_exhausted() -> None:
    watchdog, _, clock = _build(
        policy=_policy(restart_limit=1),
        recovery_command=("glr",),
        recovery_result=False,
    )
    clock.now = 500 * NS
    report = watchdog.tick()
    assert report.decisions[0].action is WatchdogAction.ESCALATE
    assert report.decisions[0].status is WatchdogStatus.FAILED
    assert report.exit_code == WATCHDOG_EXIT_ESCALATED


def test_failed_restart_below_budget_keeps_source_starved() -> None:
    watchdog, _, clock = _build(
        policy=_policy(restart_limit=3), recovery_command=("glr",), recovery_result=False
    )
    clock.now = 500 * NS
    report = watchdog.tick()
    assert report.decisions[0].action is WatchdogAction.RESTART
    assert report.decisions[0].status is WatchdogStatus.STARVED
    assert report.decisions[0].reason == "restart-failed"


def test_restart_budget_is_finite_and_then_escalates() -> None:
    watchdog, calls, clock = _build(recovery_command=("glr",))
    clock.now = 500 * NS
    for _ in range(2):
        report = watchdog.tick()
        assert report.decisions[0].action is WatchdogAction.RESTART
    assert len(calls) == 2
    final = watchdog.tick()
    assert final.decisions[0].action is WatchdogAction.ESCALATE
    assert final.decisions[0].reason == "restart-budget-exhausted"
    assert len(calls) == 2
    assert final.exit_code == WATCHDOG_EXIT_ESCALATED


def test_backoff_blocks_immediate_second_restart() -> None:
    watchdog, calls, clock = _build(
        policy=_policy(restart_backoff_seconds=100.0), recovery_command=("glr",)
    )
    clock.now = 500 * NS
    watchdog.tick()
    second = watchdog.tick()
    assert second.decisions[0].action is WatchdogAction.NONE
    assert second.decisions[0].reason == "restart-backoff-active"
    assert len(calls) == 1


def test_cooldown_waits_for_heartbeat_after_restart() -> None:
    watchdog, _, clock = _build(
        policy=_policy(restart_cooldown_seconds=100.0), recovery_command=("glr",)
    )
    clock.now = 500 * NS
    first = watchdog.tick()
    assert first.decisions[0].action is WatchdogAction.RESTART
    watchdog.observe(Heartbeat("trainer", 2, 500 * NS))
    second = watchdog.tick()
    assert second.decisions[0].status is WatchdogStatus.RECOVERING
    assert second.decisions[0].reason == "awaiting-heartbeat-after-restart"


def test_stale_sequence_is_ignored() -> None:
    watchdog, _, clock = _build()
    clock.now = 100 * NS
    watchdog.observe(Heartbeat("trainer", 5, 99 * NS))
    watchdog.observe(Heartbeat("trainer", 4, 100 * NS))
    assert watchdog.evaluate("trainer").age_seconds == pytest.approx(1.0)


def test_unknown_source_heartbeats_are_ignored() -> None:
    watchdog, _, clock = _build()
    clock.now = 20 * NS
    watchdog.observe_all([Heartbeat("other", 1, 19 * NS)])
    assert watchdog.evaluate("trainer").reason == "no-heartbeat-yet"
    assert watchdog.sources == ("trainer",)


def test_duplicate_registration_is_rejected() -> None:
    watchdog, _, _ = _build()
    with pytest.raises(WatchdogStateError):
        watchdog.register(WatchdogTarget(name="trainer", policy=_policy()))


def test_operations_on_unknown_source_raise() -> None:
    watchdog, _, _ = _build()
    with pytest.raises(WatchdogStateError):
        watchdog.evaluate("missing")
    with pytest.raises(WatchdogStateError):
        watchdog.restart_count("missing")


def test_target_requires_non_empty_recovery_command() -> None:
    with pytest.raises(ValueError, match="recovery_command"):
        WatchdogTarget(name="trainer", policy=_policy(), recovery_command=())


def test_run_once_returns_scheduler_exit_code() -> None:
    watchdog, _, clock = _build(recovery_command=("glr",))
    clock.now = 500 * NS
    assert watchdog.run_once() == WATCHDOG_EXIT_RECOVERED


def test_run_loop_stops_on_max_ticks() -> None:
    seen: list[float] = []
    clock = _Clock()
    watchdog = SupervisionWatchdog(
        policy=_policy(restart_limit=100),
        clock=clock,
        sleep_fn=seen.append,
        recovery_runner=lambda argv, timeout_seconds: True,
    )
    watchdog.register(
        WatchdogTarget(name="trainer", policy=_policy(restart_limit=100), recovery_command=("glr",))
    )
    clock.now = 500 * NS
    report = watchdog.run(interval_seconds=1.0, max_ticks=3)
    assert len(seen) == 2
    assert report.decisions[0].restart_count == 3


def test_run_loop_returns_early_on_escalation() -> None:
    seen: list[float] = []
    clock = _Clock()
    watchdog = SupervisionWatchdog(
        policy=_policy(restart_limit=1),
        clock=clock,
        sleep_fn=seen.append,
        recovery_runner=lambda argv, timeout_seconds: False,
    )
    watchdog.register(
        WatchdogTarget(name="trainer", policy=_policy(restart_limit=1), recovery_command=("glr",))
    )
    clock.now = 500 * NS
    report = watchdog.run(interval_seconds=1.0, max_ticks=5)
    assert seen == []
    assert report.exit_code == WATCHDOG_EXIT_ESCALATED


def test_run_rejects_invalid_interval_and_ticks() -> None:
    watchdog, _, _ = _build()
    with pytest.raises(ValueError, match="interval_seconds"):
        watchdog.run(interval_seconds=0)
    with pytest.raises(ValueError, match="max_ticks"):
        watchdog.run(interval_seconds=1.0, max_ticks=0)


def test_report_requires_decision_values() -> None:
    with pytest.raises(ValueError, match="WatchdogDecision"):
        WatchdogReport(("not-a-decision",))  # type: ignore[arg-type]


def test_heartbeat_log_appends_and_reads_latest(tmp_path: Path) -> None:
    log = HeartbeatLog(tmp_path / "nested" / "heartbeats.jsonl")
    assert log.read() == ()
    log.append(Heartbeat("trainer", 1, 10))
    log.append(Heartbeat("trainer", 3, 30))
    log.append(Heartbeat("collector", 1, 10))
    latest = log.latest_by_source()
    assert latest["trainer"].sequence == 3
    assert latest["collector"].sequence == 1


def test_heartbeat_log_skips_unparsable_lines(tmp_path: Path) -> None:
    path = tmp_path / "heartbeats.jsonl"
    path.write_text(
        '{"source":"trainer","sequence":1,"observed_at_ns":5}\nnot-json\n[]\n{"source":"bad"}\n',
        encoding="utf-8",
    )
    assert HeartbeatLog(path).read() == (Heartbeat("trainer", 1, 5),)


def test_heartbeat_log_requires_a_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a Path"):
        HeartbeatLog(str(tmp_path))  # type: ignore[arg-type]


def test_policy_from_mapping_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unexpected fields"):
        watchdog_policy_from_mapping({"unknown": 1})
    assert watchdog_policy_from_mapping({"restart_limit": 5}).restart_limit == 5


class _Probe:
    def __init__(self) -> None:
        self.alive: set[ProcessIdentity] = set()
        self.counter = 0

    def is_alive(self, identity: ProcessIdentity) -> bool:
        return identity in self.alive

    def invoke(self, action: str, identity: ProcessIdentity) -> None:
        self.alive.discard(identity)

    def launch(self) -> ProcessIdentity:
        self.counter += 1
        identity = ProcessIdentity(self.counter, self.counter)
        self.alive.add(identity)
        return identity


def test_watchdog_prefers_attached_supervisor_over_command() -> None:
    probe = _Probe()
    supervisor = ProcessSupervisor(probe, sleep_fn=lambda _seconds: None)
    supervisor.attach(probe.launch())
    watchdog, calls, clock = _build(supervisor=supervisor)
    clock.now = 500 * NS
    report = watchdog.tick()
    assert report.decisions[0].action is WatchdogAction.RESTART
    assert supervisor.restart_count == 1
    assert calls == []


def test_supervisor_restart_failure_is_escalated() -> None:
    class FailingSupervisor:
        def restart(self) -> ProcessIdentity:
            from game_learning_runtime import SupervisionError

            raise SupervisionError("target did not stop")

    watchdog, _, clock = _build(
        policy=_policy(restart_limit=1),
        supervisor=FailingSupervisor(),  # type: ignore[arg-type]
    )
    clock.now = 500 * NS
    report = watchdog.tick()
    assert report.decisions[0].action is WatchdogAction.ESCALATE
    assert report.decisions[0].reason == "restart-failed"
