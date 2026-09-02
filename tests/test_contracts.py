from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from game_learning_runtime import (
    ActionOutcome,
    ActionReceipt,
    ActionReconciliation,
    Event,
    ReconciliationOutcome,
    TimeStep,
    Transition,
    Unroll,
)


def _timestep(*, step_id: int = 0) -> TimeStep:
    return TimeStep(
        observation={"nested": {"value": np.array([1], dtype=np.int64)}},
        reward=np.array([0.5], dtype=np.float32),
        terminated=np.array([False, True], dtype=np.bool_),
        truncated=np.array([False, False], dtype=np.bool_),
        step_id=step_id,
    )


def test_timestep_copies_tensors_and_supports_partial_done() -> None:
    source = np.array([1], dtype=np.int64)
    timestep = TimeStep(
        observation={"value": source},
        reward=np.array([0.0], dtype=np.float32),
        terminated=np.array([False, True], dtype=np.bool_),
        truncated=np.array([False, False], dtype=np.bool_),
    )
    source[0] = 9

    np.testing.assert_array_equal(timestep.observation["value"], np.array([1]))
    assert not timestep.done
    with pytest.raises(ValueError, match="read-only"):
        timestep.reward[0] = 1.0


def test_contract_value_guards() -> None:
    with pytest.raises(ValueError, match="negative"):
        _timestep(step_id=-1)
    with pytest.raises(ValueError, match="empty"):
        Event("")
    with pytest.raises(ValueError, match="at least one"):
        Unroll((), actor_id="actor", sequence_id=0)


def test_transition_and_unroll_metadata_guards() -> None:
    transition = Transition(
        episode_id=uuid4(),
        step_id=0,
        observation={"value": np.array([0], dtype=np.int64)},
        action={"value": np.array([1], dtype=np.int64)},
        reward=np.array([1.0], dtype=np.float32),
        next_observation={"value": np.array([1], dtype=np.int64)},
        terminated=np.array([True], dtype=np.bool_),
        truncated=np.array([False], dtype=np.bool_),
    )
    assert transition.done

    with pytest.raises(ValueError, match="actor_id"):
        Unroll((transition,), actor_id="", sequence_id=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        Unroll((transition,), actor_id="actor", sequence_id=-1)


def test_action_receipt_is_typed_and_counted_without_info_parsing() -> None:
    episode_id = uuid4()
    receipt = ActionReceipt(
        action_id="move-1",
        episode_id=episode_id,
        step_id=1,
        outcome=ActionOutcome.NO_EFFECT,
        issued_timestamp_ns=10,
        observed_timestamp_ns=20,
        progress_delta=0.0,
    )
    transition = Transition(
        episode_id=episode_id,
        step_id=0,
        observation={"value": np.array([0], dtype=np.int64)},
        action={"value": np.array([1], dtype=np.int64)},
        reward=np.array([0.0], dtype=np.float32),
        next_observation={"value": np.array([0], dtype=np.int64)},
        terminated=np.array([False], dtype=np.bool_),
        truncated=np.array([False], dtype=np.bool_),
        action_receipt=receipt,
    )
    unroll = Unroll((transition,), actor_id="actor", sequence_id=0)

    assert unroll.action_outcome_counts == {"no_effect": 1}
    with pytest.raises(ValueError, match="does not match"):
        ActionReceipt(
            action_id="stale",
            episode_id=uuid4(),
            step_id=1,
            outcome=ActionOutcome.ACCEPTED,
            issued_timestamp_ns=1,
            observed_timestamp_ns=2,
        ).validate_against(_timestep(step_id=1))


def test_action_receipt_validation_guards() -> None:
    episode_id = uuid4()

    def make(**overrides: object) -> ActionReceipt:
        values: dict[str, object] = {
            "action_id": "move",
            "episode_id": episode_id,
            "step_id": 1,
            "outcome": ActionOutcome.ACCEPTED,
            "issued_timestamp_ns": 1,
            "observed_timestamp_ns": 2,
        }
        values.update(overrides)
        return ActionReceipt(**values)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="action_id"):
        make(action_id="")
    with pytest.raises(TypeError, match="episode_id"):
        make(episode_id="bad")
    with pytest.raises(ValueError, match="step_id"):
        make(step_id=0)
    assert make(outcome="accepted").outcome is ActionOutcome.ACCEPTED
    with pytest.raises(ValueError, match="unsupported action outcome"):
        make(outcome="invalid")
    with pytest.raises(ValueError, match="issued_timestamp_ns"):
        make(issued_timestamp_ns=-1)
    with pytest.raises(ValueError, match="observed_timestamp_ns"):
        make(observed_timestamp_ns=0)
    with pytest.raises(ValueError, match="postcondition"):
        make(postcondition="")
    with pytest.raises(ValueError, match="progress_delta"):
        make(progress_delta=float("inf"))
    with pytest.raises(ValueError, match="authoritative_observation_sequence"):
        make(authoritative_observation_sequence=-1)
    with pytest.raises(TypeError, match="retryable"):
        make(retryable=1)


def test_action_reconciliation_validates_the_cursor_and_outcome() -> None:
    episode_id = uuid4()
    reconciliation = ActionReconciliation(
        episode_id=episode_id,
        expected_step_id=1,
        outcome="unknown",
        authoritative_step_id=0,
        timestamp_ns=7,
    )
    assert reconciliation.outcome is ReconciliationOutcome.UNKNOWN

    with pytest.raises(TypeError, match="episode_id"):
        ActionReconciliation("not-a-uuid", 1, ReconciliationOutcome.UNKNOWN, 0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected_step_id"):
        ActionReconciliation(episode_id, 0, ReconciliationOutcome.UNKNOWN, 0, 0)
    with pytest.raises(ValueError, match="unsupported reconciliation"):
        ActionReconciliation(episode_id, 1, "invalid", 0, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="authoritative_step_id"):
        ActionReconciliation(episode_id, 1, ReconciliationOutcome.UNKNOWN, -1, 0)
    with pytest.raises(ValueError, match="timestamp_ns"):
        ActionReconciliation(episode_id, 1, ReconciliationOutcome.UNKNOWN, 0, -1)
    with pytest.raises(TypeError, match="retryable"):
        ActionReconciliation(episode_id, 1, ReconciliationOutcome.UNKNOWN, 0, 0, retryable=1)  # type: ignore[arg-type]
