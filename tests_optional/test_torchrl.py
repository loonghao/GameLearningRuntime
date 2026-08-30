from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from torchrl.envs.utils import check_env_specs

from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.integrations.torchrl import TorchRLEnvironment, _leaf_spec
from game_learning_runtime.specs import EnvironmentSpec, TensorSpec


class _AttachOnlyCounter(GameEnvironment):
    def __init__(self) -> None:
        self._delegate = CounterEnvironment()

    @property
    def spec(self) -> EnvironmentSpec:
        return replace(
            self._delegate.spec,
            capabilities=self._delegate.spec.capabilities | {"live-attach"},
        )

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        raise AssertionError("attach-mode TorchRL adapter must not call reset")

    def attach(self, *, options: Mapping[str, Any] | None = None) -> TimeStep:
        return self._delegate.reset(options=options)

    def step(self, action: TensorTree) -> TimeStep:
        return self._delegate.step(action)


@pytest.mark.torchrl
def test_torchrl_environment_satisfies_current_envbase_contract() -> None:
    environment = TorchRLEnvironment(CounterEnvironment())

    check_env_specs(environment)
    rollout = environment.rollout(max_steps=3)

    assert rollout.batch_size[-1] == 3
    assert ("next", "reward") in rollout.keys(include_nested=True)


@pytest.mark.torchrl
def test_array_bounds_do_not_create_non_writable_torch_tensors() -> None:
    spec = TensorSpec(
        (2,),
        np.float32,
        minimum=np.array([-1.0, -2.0], dtype=np.float32),
        maximum=np.array([1.0, 2.0], dtype=np.float32),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        converted = _leaf_spec(spec, device=torch.device("cpu"))

    assert tuple(converted.shape) == (2,)


@pytest.mark.torchrl
def test_torchrl_environment_explicitly_attaches_to_a_continuing_runtime() -> None:
    environment = TorchRLEnvironment(_AttachOnlyCounter(), start_mode="attach")

    initial = environment.reset()

    assert ("observation", "position") in initial.keys(include_nested=True)
