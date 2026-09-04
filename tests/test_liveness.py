from __future__ import annotations

import pytest

from game_learning_runtime import (
    EnvironmentFrozenError,
    LivenessMonitor,
    ProgressSignalDeclaration,
    validate_progress_field,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.liveness import LivenessSnapshot


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


def test_liveness_validation_and_mapping_edges() -> None:
    with pytest.raises(ValueError, match="observation_sequence"):
        LivenessSnapshot(-1, 0, 0)
    with pytest.raises(ValueError, match="observation_age_ms"):
        LivenessSnapshot(1, -1, 0)
    with pytest.raises(ValueError, match="last_sequence_change_ms"):
        LivenessSnapshot(1, 0, -1)
    with pytest.raises(TypeError, match="env_frozen"):
        LivenessSnapshot(1, 0, 0, env_frozen=1)  # type: ignore[arg-type]
    assert LivenessSnapshot(1, 2, 3, True).to_mapping()["env_frozen"] is True

    monitor = LivenessMonitor()
    with pytest.raises(ValueError, match="freeze_after_ms"):
        LivenessMonitor(freeze_after_ms=0)
    with pytest.raises(ValueError, match="observation_sequence"):
        monitor.observe(-1)
    with pytest.raises(ValueError, match="now_ns"):
        monitor.observe(1, now_ns=-1)
    with pytest.raises(ValueError, match="produced_at_ns"):
        monitor.observe(1, produced_at_ns=-1)
    with pytest.raises(ValueError, match="now_ns"):
        monitor.observe(1, now_ns=1.5)  # type: ignore[arg-type]


def test_liveness_accepts_produced_timestamps_and_validates_errors() -> None:
    monitor = LivenessMonitor()
    snapshot = monitor.observe(1, produced_at_ns=900_000, now_ns=1_000_000)
    assert snapshot.observation_age_ms == 0.1
    updated = monitor.observe(1, produced_at_ns=950_000, now_ns=1_000_000)
    assert updated.observation_age_ms == 0.05
    assert monitor.require_live(snapshot) is snapshot
    with pytest.raises(ValueError, match="frozen"):
        from game_learning_runtime.liveness import EnvironmentFrozenError

        EnvironmentFrozenError(snapshot)


def test_progress_declaration_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        ProgressSignalDeclaration("")
