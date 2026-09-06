from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime.contracts import TimeStep, Transition
from game_learning_runtime.training_contract import (
    TrainingContractError,
    assert_transition_provenance,
    transition_provenance,
    validate_timestep,
)


def _step(**info):
    return TimeStep(
        {"state": np.array([1])},
        np.array([0.0]),
        np.array([False]),
        np.array([False]),
        episode_id=UUID(int=1),
        info=info,
    )


def test_provenance_is_bound_to_glr_timestep():
    step = _step(run_id="run-7")
    provenance = transition_provenance(step, segment=2)
    assert provenance["source"] == "glr-timestep"
    assert provenance["run_id"] == "run-7"
    assert provenance["episode_id"] == str(step.episode_id)


def test_conflicting_boundaries_fail_closed():
    step = TimeStep({"state": np.array([1])}, np.array([0.0]), np.array([True]), np.array([True]))
    with pytest.raises(TrainingContractError, match="both"):
        validate_timestep(step)


def test_replay_requires_matching_provenance():
    step = _step()
    transition = Transition(
        step.episode_id,
        0,
        step.observation,
        {"a": 0},
        step.reward,
        step.observation,
        step.terminated,
        step.truncated,
        provenance=transition_provenance(step),
    )
    assert assert_transition_provenance(transition) is transition
    bad = Transition(
        step.episode_id,
        1,
        step.observation,
        {"a": 0},
        step.reward,
        step.observation,
        step.terminated,
        step.truncated,
        provenance=transition_provenance(step),
    )
    with pytest.raises(TrainingContractError, match="step_id"):
        assert_transition_provenance(bad)
