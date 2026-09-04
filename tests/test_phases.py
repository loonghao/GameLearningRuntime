from __future__ import annotations

import pytest

from game_learning_runtime import (
    EnvironmentPhase,
    PhaseActionError,
    PhaseMonitor,
    PhaseObservationError,
    PhasePolicy,
    PhaseTimeoutError,
    validate_phase,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.phases import PhaseMetrics, PhaseStep


def test_phase_monitor_enforces_policy_and_metrics() -> None:
    monitor = PhaseMonitor(
        {EnvironmentPhase.MENU: PhasePolicy(actions_allowed=True, budget_ms=100)}
    )
    menu = monitor.record_step("menu", now_ns=1_000_000_000)
    assert not menu.training_eligible
    monitor.allow_action("menu")
    with pytest.raises(PhaseActionError, match="loading"):
        monitor.allow_action("loading")
    gameplay = monitor.record_step("gameplay", now_ns=1_050_000_000)
    assert gameplay.training_eligible
    assert monitor.metrics.steps["menu"] == 1


def test_cutscene_allows_missing_observation_and_timeout_is_typed() -> None:
    monitor = PhaseMonitor(
        {EnvironmentPhase.CUTSCENE: PhasePolicy(observations_expected=False, budget_ms=10)}
    )
    step = monitor.record_step("cutscene", observation_present=False, now_ns=1_000_000_000)
    assert not step.training_eligible
    with pytest.raises(PhaseTimeoutError, match="phase_timeout:cutscene"):
        monitor.record_step("cutscene", observation_present=False, now_ns=1_011_000_000)

    other = PhaseMonitor()
    with pytest.raises(PhaseObservationError, match="absent"):
        other.record_step("gameplay", observation_present=False, now_ns=1)


def test_phase_validation_and_policy_edges() -> None:
    assert validate_phase("modal") is EnvironmentPhase.MODAL
    with pytest.raises(ContractViolation, match="unsupported"):
        validate_phase("not-a-phase")
    with pytest.raises(ValueError, match="budget_ms"):
        PhasePolicy(budget_ms=0)
    with pytest.raises(TypeError, match="bool"):
        PhasePolicy(training=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="time totals"):
        PhaseMetrics(time_ms={"menu": -1})
    with pytest.raises(ValueError, match="step totals"):
        PhaseMetrics(steps={"menu": -1})
    assert PhaseStep(EnvironmentPhase.GAMEPLAY, 1, True, True).to_mapping()["phase"] == "gameplay"


def test_phase_monitor_rejects_bad_inputs() -> None:
    monitor = PhaseMonitor()
    with pytest.raises(TypeError, match="observation_present"):
        monitor.record_step("gameplay", observation_present=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now_ns"):
        monitor.record_step("gameplay", now_ns=-1)
    with pytest.raises(ValueError, match="budget_ms"):
        PhasePolicy(budget_ms=0)
