from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from game_learning_runtime import (
    ActorQueueCancelled,
    ActorQueueClosed,
    ActorQueueCommitError,
    ActorQueueFull,
    BoundedActorQueue,
    ContractEnvironment,
    GameEnvironment,
    SyncCollector,
    TimeStep,
)
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


def test_collector_can_stop_at_first_terminal_without_starting_next_episode() -> None:
    collector = SyncCollector(
        ContractEnvironment(CounterEnvironment(target=2)), actor_id="live-player"
    )

    unroll = collector.collect(
        always_increment,
        steps=5,
        policy_version=11,
        seed=7,
        stop_on_done=True,
    )

    assert len(unroll.transitions) == 2
    assert unroll.transitions[-1].done is True
    assert {transition.episode_id for transition in unroll.transitions} == {
        unroll.transitions[0].episode_id
    }

    following = collector.collect(
        always_increment,
        steps=1,
        policy_version=12,
        stop_on_done=True,
    )
    assert following.sequence_id == 1
    assert following.transitions[0].episode_id != unroll.transitions[0].episode_id


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


def _unroll(*, actor_id: str = "actor-0", sequence_id: int = 0, policy_version: int = 0):
    collector = SyncCollector(ContractEnvironment(CounterEnvironment(target=3)), actor_id=actor_id)
    result = collector.collect(always_increment, steps=1, policy_version=policy_version)
    return replace(result, sequence_id=sequence_id)


def test_bounded_actor_queue_drop_and_commit_metrics_are_fenced() -> None:
    queue = BoundedActorQueue(
        1,
        overflow_policy="drop-oldest",
        learner_policy_version=4,
    )
    first = _unroll(sequence_id=0, policy_version=1)
    second = _unroll(sequence_id=1, policy_version=2)
    assert queue.put(first)
    assert queue.put(second)

    metrics = queue.metrics()
    assert metrics.depth == 1
    assert metrics.dropped_unrolls == 1
    assert metrics.uncommitted_unrolls == 1
    assert metrics.max_policy_version_lag == 3
    assert metrics.actor_lag == {"actor-0": 1}

    leased = queue.get()
    assert leased.unroll is second
    queue.commit(leased)
    assert queue.metrics().committed_unrolls == 1
    assert queue.metrics().uncommitted_unrolls == 0
    with pytest.raises(ActorQueueCommitError, match="unknown"):
        queue.commit(leased)


def test_bounded_actor_queue_block_policy_wakes_after_dequeue() -> None:
    queue = BoundedActorQueue(1, overflow_policy="block")
    queue.put(_unroll(sequence_id=0))
    completed = threading.Event()
    errors: list[BaseException] = []

    def producer() -> None:
        try:
            queue.put(_unroll(sequence_id=1), timeout=1)
        except BaseException as error:  # pragma: no cover - assertion below reports it
            errors.append(error)
        finally:
            completed.set()

    thread = threading.Thread(target=producer)
    thread.start()
    time.sleep(0.05)
    leased = queue.get()
    queue.commit(leased)
    assert completed.wait(1)
    thread.join()
    assert errors == []
    assert queue.metrics().blocked_puts >= 1


def test_bounded_actor_queue_cancellation_failure_and_shutdown_are_explicit() -> None:
    queue = BoundedActorQueue(1, overflow_policy="fail")
    queue.put(_unroll(sequence_id=0))
    with pytest.raises(ActorQueueFull, match="full"):
        queue.put(_unroll(sequence_id=1))

    leased = queue.get()
    queue.abort(leased)
    assert queue.metrics().aborted_unrolls == 1
    assert queue.metrics().committed_unrolls == 0

    queue = BoundedActorQueue(1, overflow_policy="block")
    queue.put(_unroll(sequence_id=0))
    cancelled = threading.Event()
    errors: list[BaseException] = []

    def blocked_producer() -> None:
        try:
            queue.put(_unroll(sequence_id=1), cancel_event=cancelled)
        except BaseException as error:  # pragma: no cover - assertion below reports it
            errors.append(error)

    thread = threading.Thread(target=blocked_producer)
    thread.start()
    time.sleep(0.05)
    cancelled.set()
    thread.join(1)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ActorQueueCancelled)

    queue.close()
    leased = queue.get()
    queue.commit(leased)
    with pytest.raises(ActorQueueClosed, match="closed"):
        queue.get(timeout=0)


def test_bounded_actor_queue_rejects_duplicate_actor_sequences() -> None:
    queue = BoundedActorQueue(2)
    item = _unroll(sequence_id=0)
    queue.put(item)
    with pytest.raises(ValueError, match="sequence_id"):
        queue.put(item)
