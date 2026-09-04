from __future__ import annotations

import pytest

from game_learning_runtime import (
    EnvironmentFrozenError,
    LivenessMonitor,
    ProgressSignalDeclaration,
    validate_progress_field,
)
from game_learning_runtime.errors import ContractViolation


def test_liveness_reports_age_and_freezes_static_sequence() -> None:
    monitor = LivenessMonitor(freeze_after_ms=100)
    first = monitor.observe(7, now_ns=1_000_000_000)
    frozen = monitor.observe(7, now_ns=1_150_000_000)
    assert first.observation_age_ms == 0
    assert frozen.observation_age_ms == 150
    assert frozen.last_sequence_change_ms == 150
    assert frozen.env_frozen
    with pytest.raises(EnvironmentFrozenError):
        monitor.require_live(frozen)


def test_sequence_change_resets_freshness() -> None:
    monitor = LivenessMonitor(freeze_after_ms=100)
    monitor.observe(1, now_ns=1_000_000_000)
    current = monitor.observe(2, now_ns=1_150_000_000)
    assert current.observation_age_ms == 0
    assert current.last_sequence_change_ms == 0
    assert not current.env_frozen


def test_progress_signal_cannot_alias_liveness_counter() -> None:
    with pytest.raises(ContractViolation, match="liveness counter"):
        ProgressSignalDeclaration("observation_sequence")
    assert validate_progress_field("milestone") == "milestone"
