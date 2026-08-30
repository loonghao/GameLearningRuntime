from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from game_learning_runtime import Event, TimeStep, Transition, Unroll


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
