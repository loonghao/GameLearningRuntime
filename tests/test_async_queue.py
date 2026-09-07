from __future__ import annotations

import threading
from dataclasses import replace
from time import monotonic

import pytest

from game_learning_runtime.collector import (
    ActorQueueCancelled,
    ActorQueueClosed,
    ActorQueueCommitError,
    ActorQueueFull,
    ActorQueueStaleUnroll,
    BoundedActorQueue,
    SyncCollector,
)
from game_learning_runtime.contracts import Unroll
from game_learning_runtime.examples import CounterEnvironment, always_increment


def _unroll(sequence: int = 0, version: int = 0) -> Unroll:
    return replace(
        SyncCollector(CounterEnvironment()).collect(always_increment, steps=1),
        sequence_id=sequence,
        policy_version=version,
    )


def _wait_paused(queue: BoundedActorQueue) -> None:
    deadline = monotonic() + 2
    while not queue.paused and monotonic() < deadline:
        threading.Event().wait(0.001)
    assert queue.paused


@pytest.mark.parametrize("finalize", ["commit", "abort"])
def test_drain_fences_new_leases_and_preserves_queued_work(finalize: str) -> None:
    queue = BoundedActorQueue(2)
    queue.put(_unroll())
    leased = queue.get()
    queue.put(_unroll(1))
    results = []
    errors = []

    def drain() -> None:
        try:
            results.append(queue.drain(timeout=2))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=drain)
    thread.start()
    try:
        _wait_paused(queue)
        assert queue.metrics().in_flight_unrolls == 1
        with pytest.raises(ActorQueueCommitError, match="resume"):
            queue.resume()
        with pytest.raises(ActorQueueCommitError, match="another drain"):
            queue.drain(timeout=0)
        with pytest.raises(ActorQueueFull, match="timed out"):
            queue.get_nowait()
        # Pausing leases does not silently stop an actor or expand capacity.
        queue.put_nowait(_unroll(2))
        with pytest.raises(ActorQueueFull):
            queue.put_nowait(_unroll(3))
        getattr(queue, finalize)(leased)
        thread.join(2)
        assert not thread.is_alive()
        assert not errors
        assert results[0].in_flight_unrolls == 0
        assert results[0].depth == 2
        assert results[0].paused
        assert results[0].drain_count == 1
        assert results[0].drain_latency_ns_total >= 0
        queue.set_learner_policy_version(1)
        assert queue.metrics().carry_over_unrolls == 2
        assert queue.metrics().oldest_pending_age_ns >= 0
        queue.resume()
        for sequence in (1, 2):
            pending = queue.get_nowait()
            assert pending.unroll.sequence_id == sequence
            queue.commit(pending)
        assert queue.metrics().oldest_pending_age_ns == 0
        assert queue.metrics().carry_over_unrolls == 0
    finally:
        queue.close()
        thread.join(3)


def test_timeout_keeps_barrier_and_does_not_report_a_successful_drain() -> None:
    queue = BoundedActorQueue(1)
    queue.put(_unroll())
    leased = queue.get()
    with pytest.raises(ActorQueueFull, match="drain"):
        queue.drain(timeout=0)
    assert queue.paused
    assert queue.metrics().drain_count == 0
    queue.abort(leased)
    assert queue.drain(timeout=0).drain_count == 1
    queue.resume()
    assert not queue.paused


@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_drain_wait_is_interruptible(stop: str) -> None:
    queue = BoundedActorQueue(1)
    queue.put(_unroll())
    leased = queue.get()
    cancel = threading.Event()
    errors = []

    def drain() -> None:
        try:
            queue.drain(cancel_event=cancel)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=drain)
    thread.start()
    try:
        _wait_paused(queue)
        if stop == "cancel":
            cancel.set()
        else:
            queue.close()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ActorQueueCancelled if stop == "cancel" else ActorQueueClosed)
        assert queue.metrics().drain_count == 0
        queue.abort(leased)
    finally:
        queue.close()
        thread.join(2)


def test_closed_queue_releases_paused_readers_and_retains_close_drain_behavior() -> None:
    queue = BoundedActorQueue(1)
    queue.pause()
    queue.put(_unroll())
    queue.close()
    leased = queue.get_nowait()
    queue.commit(leased)
    with pytest.raises(ActorQueueClosed):
        queue.get_nowait()
    with pytest.raises(ActorQueueClosed):
        queue.pause()
    with pytest.raises(ActorQueueClosed):
        queue.drain(timeout=0)


def test_stale_arrival_cannot_evict_valid_work() -> None:
    queue = BoundedActorQueue(
        1, learner_policy_version=4, max_policy_version_lag=1, overflow_policy="drop-oldest"
    )
    queue.put(_unroll(version=3))
    with pytest.raises(ActorQueueStaleUnroll):
        queue.put(_unroll(1, version=2))
    assert queue.metrics().rejected_stale_unrolls == 1
    assert queue.metrics().dropped_unrolls == 0
    leased = queue.get()
    assert leased.unroll.policy_version == 3
    queue.commit(leased)


def test_queued_staleness_is_checked_after_policy_update() -> None:
    queue = BoundedActorQueue(2, max_policy_version_lag=1)
    queue.put(_unroll())
    queue.put(_unroll(1, version=1))
    queue.drain(timeout=0)
    queue.set_learner_policy_version(2)
    queue.resume()
    leased = queue.get_nowait()
    assert leased.unroll.sequence_id == 1
    queue.commit(leased)
    summary = queue.run_summary()["actor_queue"]
    assert summary["stale_dropped_unrolls"] == 1
    assert summary["dropped_unrolls"] == 1
    assert summary["committed_unrolls"] == 1
    assert summary["uncommitted_unrolls"] == 0


def test_all_stale_queue_is_empty_and_cannot_replay_sequence() -> None:
    queue = BoundedActorQueue(1, max_policy_version_lag=0)
    queue.put(_unroll())
    queue.set_learner_policy_version(1)
    with pytest.raises(ActorQueueFull, match="timed out"):
        queue.get_nowait()
    with pytest.raises(ValueError, match="sequence_id"):
        queue.put(_unroll(version=1))
    assert queue.metrics().stale_dropped_unrolls == 1


def test_stale_lease_commit_fails_until_explicit_abort() -> None:
    queue = BoundedActorQueue(1, max_policy_version_lag=0)
    queue.put(_unroll())
    leased = queue.get()
    queue.set_learner_policy_version(1)
    with pytest.raises(ActorQueueStaleUnroll):
        queue.commit(leased)
    assert queue.metrics().in_flight_unrolls == 1
    assert queue.metrics().committed_unrolls == 0
    queue.abort(leased)
    assert queue.metrics().aborted_unrolls == 1


def test_rejected_foreign_lease_does_not_destroy_the_real_lease() -> None:
    queue = BoundedActorQueue(1)
    queue.put(_unroll())
    leased = queue.get()
    for finalize in (queue.commit, queue.abort):
        with pytest.raises(ActorQueueCommitError):
            finalize(replace(leased))
    queue.commit(leased)
    assert queue.metrics().committed_unrolls == 1


@pytest.mark.parametrize("lag", [-1, 0.5, True])
def test_invalid_policy_lag_is_rejected(lag: object) -> None:
    with pytest.raises(ValueError, match="max_policy_version_lag"):
        BoundedActorQueue(1, max_policy_version_lag=lag)


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("operation", ["put", "get", "drain"])
def test_waits_reject_non_finite_or_negative_timeouts(timeout: float, operation: str) -> None:
    queue = BoundedActorQueue(1)
    arguments = (_unroll(),) if operation == "put" else ()
    with pytest.raises(ValueError, match="timeout"):
        getattr(queue, operation)(*arguments, timeout=timeout)


def test_pre_cancelled_drain_leaves_a_safe_barrier() -> None:
    queue = BoundedActorQueue(1)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ActorQueueCancelled):
        queue.drain(cancel_event=cancel)
    assert queue.paused
    assert queue.metrics().drain_count == 0


def test_queue_age_and_drain_latency_use_monotonic_time(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 100
    monkeypatch.setattr("game_learning_runtime.collector.time.monotonic_ns", lambda: now)
    queue = BoundedActorQueue(1)
    queue.put(_unroll())
    now = 140
    assert queue.metrics().oldest_pending_age_ns == 40
    leased = queue.get()
    now = 170
    assert queue.metrics().oldest_pending_age_ns == 70
    queue.commit(leased)
    assert queue.metrics().oldest_pending_age_ns == 0
    ticks = iter((200, 260))
    monkeypatch.setattr("game_learning_runtime.collector.time.monotonic_ns", lambda: next(ticks))
    assert queue.drain().drain_latency_ns_total == 60


@pytest.mark.parametrize("version", [-1, 0.5, True, float("nan"), float("inf"), "1"])
def test_queue_rejects_malformed_policy_versions(version: object) -> None:
    with pytest.raises(ValueError, match="policy_version"):
        BoundedActorQueue(1, learner_policy_version=version)
    queue = BoundedActorQueue(1, max_policy_version_lag=0)
    with pytest.raises(ValueError, match="policy_version"):
        queue.set_learner_policy_version(version)


@pytest.mark.parametrize("version", [0.5, True, float("nan"), float("inf")])
@pytest.mark.parametrize("max_lag", [None, 0])
def test_queue_validates_unroll_versions_at_ingress(version: object, max_lag: int | None) -> None:
    queue = BoundedActorQueue(1, max_policy_version_lag=max_lag)
    with pytest.raises(ValueError, match="policy_version"):
        queue.put(replace(_unroll(), policy_version=version))


def test_policy_update_wakes_a_blocked_producer_that_becomes_stale() -> None:
    queue = BoundedActorQueue(1, max_policy_version_lag=0)
    queue.put(_unroll())
    errors = []

    def produce() -> None:
        try:
            queue.put(_unroll(1))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=produce)
    thread.start()
    try:
        deadline = monotonic() + 2
        while queue.metrics().blocked_puts == 0 and monotonic() < deadline:
            threading.Event().wait(0.001)
        assert queue.metrics().blocked_puts > 0
        queue.set_learner_policy_version(1)
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ActorQueueStaleUnroll)
        assert queue.metrics().rejected_stale_unrolls == 1
    finally:
        queue.close()
        thread.join(2)


def test_competing_blocked_producers_cannot_enqueue_duplicate_sequence() -> None:
    queue = BoundedActorQueue(1)
    queue.put(_unroll())
    errors = []
    duplicate = _unroll(1)

    def produce() -> None:
        try:
            queue.put(duplicate, timeout=2)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=produce) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        deadline = monotonic() + 2
        while queue.metrics().blocked_puts < 2 and monotonic() < deadline:
            threading.Event().wait(0.001)
        assert queue.metrics().blocked_puts >= 2
        queue.commit(queue.get())
        for thread in threads:
            thread.join(2)
            assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert "sequence_id" in str(errors[0])
        queue.commit(queue.get_nowait())
        assert queue.metrics().enqueued_unrolls == 2
    finally:
        queue.close()
        for thread in threads:
            thread.join(2)
