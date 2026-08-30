from __future__ import annotations

import pytest
from torchrl.envs.utils import check_env_specs

from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.integrations.torchrl import TorchRLEnvironment


@pytest.mark.torchrl
def test_torchrl_environment_satisfies_current_envbase_contract() -> None:
    environment = TorchRLEnvironment(CounterEnvironment())

    check_env_specs(environment)
    rollout = environment.rollout(max_steps=3)

    assert rollout.batch_size[-1] == 3
    assert ("next", "reward") in rollout.keys(include_nested=True)
