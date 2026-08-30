from __future__ import annotations

import numpy as np
import pytest

from game_learning_runtime import ContractEnvironment, SyncCollector
from game_learning_runtime.examples import CounterEnvironment, always_increment


def test_collector_builds_fixed_length_unroll_across_episodes() -> None:
    collector = SyncCollector(
        ContractEnvironment(CounterEnvironment(target=2)), actor_id="worker-7"
    )

    unroll = collector.collect(always_increment, steps=5, policy_version=11, seed=7)

    assert len(unroll.transitions) == 5
    assert unroll.actor_id == "worker-7"
    assert unroll.sequence_id == 0
    assert unroll.policy_version == 11
    assert sum(transition.done for transition in unroll.transitions) == 2
    np.testing.assert_allclose(unroll.total_reward, np.array([1.97], dtype=np.float32))

    following = collector.collect(always_increment, steps=1, policy_version=12)
    assert following.sequence_id == 1


def test_collector_rejects_invalid_arguments() -> None:
    environment = ContractEnvironment(CounterEnvironment())
    collector = SyncCollector(environment)

    with pytest.raises(ValueError, match="positive"):
        collector.collect(always_increment, steps=0)
    with pytest.raises(ValueError, match="negative"):
        collector.collect(always_increment, steps=1, policy_version=-1)
