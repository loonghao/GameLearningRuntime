from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from game_learning_runtime import ContractEnvironment, GameEnvironment, SyncCollector, TimeStep
from game_learning_runtime.contracts import TensorTree
from game_learning_runtime.examples import CounterEnvironment, always_increment
from game_learning_runtime.specs import EnvironmentSpec


class _AttachOnlyCounter(GameEnvironment):
    def __init__(self) -> None:
        self._delegate = CounterEnvironment(target=2)

    @property
    def spec(self) -> EnvironmentSpec:
        return replace(
            self._delegate.spec,
            capabilities=self._delegate.spec.capabilities | {"live-attach"},
        )

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        raise AssertionError("attach-mode collector must not call reset")

    def attach(self, *, options: Mapping[str, Any] | None = None) -> TimeStep:
        return self._delegate.reset(options=options)

    def step(self, action: TensorTree) -> TimeStep:
        return self._delegate.step(action)


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


def test_collector_explicitly_attaches_to_a_continuing_runtime() -> None:
    collector = SyncCollector(
        ContractEnvironment(_AttachOnlyCounter()),
        actor_id="live-player",
        start_mode="attach",
    )

    unroll = collector.collect(always_increment, steps=3)

    assert len(unroll.transitions) == 3
    assert sum(transition.done for transition in unroll.transitions) == 1


def test_attach_mode_collector_rejects_seeded_initialization() -> None:
    collector = SyncCollector(_AttachOnlyCounter(), start_mode="attach")

    with pytest.raises(ValueError, match="seed is not supported"):
        collector.collect(always_increment, steps=1, seed=7)
