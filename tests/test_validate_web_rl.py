"""Seed-sinking tests for the web RL validation tool.

The tool is the evidence producer behind ADR-0048, so its numbers only mean
something if a run can be repeated. These tests pin the part that makes that
true: every ``reset()`` has to receive an explicit, per-episode world seed. An
adapter that reseeds itself when the caller passes no seed would give every
episode a different world, and the same command would report different numbers
twice -- which is exactly the bug this suite exists to keep fixed.

No browser and no torch are needed here: the module imports cleanly without
either, and the environment is a stub that only records the seeds it is handed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from game_learning_runtime.contracts import TimeStep
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.specs import EnvironmentSpec
from game_learning_runtime.web_game import dodge_environment_spec
from tools.providers.validate_web_rl import (
    EVALUATION_SEED_OFFSET,
    ValidationConfig,
    episode_seed,
    evaluate,
)


class _SeedRecordingEnvironment(GameEnvironment):
    """A `GameEnvironment` that ends every episode immediately and logs seeds."""

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    @property
    def spec(self) -> EnvironmentSpec:
        return dodge_environment_spec("test.seed-sink", max_steps=8)

    def _timestep(self) -> TimeStep:
        return TimeStep(
            observation={"features": np.zeros(4, dtype=np.float32)},
            reward=np.array([0.0], dtype=np.float32),
            terminated=np.array([True], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
            info={"score": 0.0, "web_steps": 0},
        )

    def reset(self, *, seed: int | None = None, options: Any = None) -> TimeStep:
        self.seeds.append(seed)
        return self._timestep()

    def step(self, action: Any) -> TimeStep:
        del action
        return self._timestep()

    def close(self) -> None:
        return None


def _constant_policy(timestep: TimeStep) -> dict[str, Any]:
    del timestep
    return {"choice": np.array([0], dtype=np.int64)}


def test_episode_seed_offsets_one_seed_per_episode() -> None:
    assert [episode_seed(7, index) for index in range(3)] == [7, 8, 9]


def test_evaluate_sinks_a_distinct_seed_into_every_episode() -> None:
    environment = _SeedRecordingEnvironment()

    evaluate(environment, _constant_policy, episodes=4, max_steps=8, seed=100)

    assert environment.seeds == [100, 101, 102, 103]


def test_evaluate_repeats_the_same_seeds_for_a_repeated_run() -> None:
    first = _SeedRecordingEnvironment()
    second = _SeedRecordingEnvironment()

    evaluate(first, _constant_policy, episodes=3, max_steps=8, seed=100)
    evaluate(second, _constant_policy, episodes=3, max_steps=8, seed=100)

    assert first.seeds == second.seeds == [100, 101, 102]


def test_evaluation_seeds_are_held_out_from_the_training_range() -> None:
    config = ValidationConfig(seed=7, train_steps=120_000, evaluation_episodes=30)
    # A 120k-step run completes ~1.1k training episodes; every one of them must
    # fall below the reserved evaluation range, or a policy would be scored on a
    # world it trained on.
    training_seeds = {episode_seed(config.seed, index) for index in range(5_000)}
    evaluation_seeds = {
        episode_seed(config.seed + EVALUATION_SEED_OFFSET, index)
        for index in range(config.evaluation_episodes)
    }

    assert not training_seeds & evaluation_seeds
    assert min(evaluation_seeds) == 1_000_007
