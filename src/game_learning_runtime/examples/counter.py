"""Small complete GLR environment used by the getting-started guide."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import numpy as np

from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.specs import CompositeSpec, EnvironmentSpec, SpaceKind, TensorSpec


class CounterEnvironment(GameEnvironment):
    """Increment a counter until it reaches a target value."""

    def __init__(self, *, target: int = 3, max_steps: int = 5) -> None:
        self._target = target
        self._max_steps = max_steps
        self._position = 0
        self._step_id = 0
        self._episode_id = uuid4()
        self._spec = EnvironmentSpec(
            environment_id="example.counter-v1",
            observation=CompositeSpec(
                {"position": TensorSpec((1,), np.int64, minimum=0, maximum=target)}
            ),
            action=CompositeSpec(
                {
                    "choice": TensorSpec(
                        (1,),
                        np.int64,
                        kind=SpaceKind.DISCRETE,
                        minimum=0,
                        maximum=1,
                        description="0 waits and 1 increments the counter",
                    )
                }
            ),
            action_mask=CompositeSpec(
                {"choice": TensorSpec((2,), np.bool_, kind=SpaceKind.BINARY)}
            ),
            capabilities=frozenset({"action-mask", "deterministic-reset"}),
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        del seed, options
        self._position = 0
        self._step_id = 0
        self._episode_id = uuid4()
        return self._timestep()

    def step(self, action: TensorTree) -> TimeStep:
        choice_value = action["choice"]
        if isinstance(choice_value, Mapping):
            raise TypeError("choice must be a tensor leaf")
        choice = int(choice_value[0])
        if choice == 1:
            self._position = min(self._position + 1, self._target)
        self._step_id += 1
        return self._timestep()

    def _timestep(self) -> TimeStep:
        reached_target = self._position == self._target
        truncated = not reached_target and self._step_id >= self._max_steps
        return TimeStep(
            observation={"position": np.array([self._position], dtype=np.int64)},
            reward=np.array([1.0 if reached_target else -0.01], dtype=np.float32),
            terminated=np.array([reached_target], dtype=np.bool_),
            truncated=np.array([truncated], dtype=np.bool_),
            action_mask={"choice": np.array([True, not reached_target], dtype=np.bool_)},
            episode_id=self._episode_id,
            step_id=self._step_id,
        )


def always_increment(timestep: TimeStep) -> TensorTree:
    del timestep
    return {"choice": np.array([1], dtype=np.int64)}


def make_environment() -> CounterEnvironment:
    return CounterEnvironment()


__all__ = ["CounterEnvironment", "always_increment", "make_environment"]
