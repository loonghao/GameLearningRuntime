from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import UUID

import pytest

import game_learning_runtime.run_store as run_store_module
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.run_store import (
    RUN_STORE_SCHEMA_VERSION,
    RolloutStatus,
    RunStatus,
    TrainingStore,
)


def _store(tmp_path: Path) -> tuple[TrainingStore, str]:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.game-v1",
        protocol_version="1.0",
        kind="collection",
        started_at_ns=0,
    )
    return store, run.run_id


def test_rollout_lifecycle_survives_reopen_and_has_append_only_event_history(
    tmp_path: Path,
) -> None:
    store, run_id = _store(tmp_path)
    queued = store.create_rollout(run_id, queued_at_ns=1, metadata={"policy_version": 3})
    assert UUID(queued.rollout_id.removeprefix("rollout-")).version == 4
    assert UUID(queued.attempt_id.removeprefix("attempt-")).version == 4
    assert queued.status is RolloutStatus.QUEUING
    assert queued.attempt_index == 1
    assert queued.parent_attempt_id is None
    assert queued.sequence_id == 1
    store.append_event(run_id, kind="policy.loaded", payload={})
    running = store.update_rollout_attempt(
        queued.attempt_id,
        status=RolloutStatus.RUNNING,
        expected_status=RolloutStatus.QUEUING,
        timestamp_ns=2,
    )
    assert running.started_at_ns == 2
    assert running.sequence_id == 3
    succeeded = store.update_rollout_attempt(
        queued.attempt_id,
        status=RolloutStatus.SUCCEEDED,
        expected_status=RolloutStatus.RUNNING,
        timestamp_ns=4,
    )
    reopened = TrainingStore(store.path)
    assert reopened.get_rollout_attempt(queued.attempt_id) == succeeded
    assert succeeded.started_at_ns == 2
    assert succeeded.finished_at_ns == 4
    assert succeeded.metadata == {"policy_version": 3}
    assert reopened.list_rollout_attempts(run_id=run_id) == (succeeded,)
    assert reopened.list_rollout_attempts(rollout_id=queued.rollout_id) == (succeeded,)
    assert reopened.list_rollout_attempts(status=RolloutStatus.QUEUING) == ()
    events = reopened.list_events(run_id)
    assert [event.sequence_id for event in events] == [1, 2, 3, 4]
    assert events[0].payload == queued.to_mapping()
    assert events[2].payload == running.to_mapping()
    assert events[3].payload == succeeded.to_mapping()
    assert events[0].payload["schema_version"] == "glr.rollout.v1"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == RUN_STORE_SCHEMA_VERSION


def test_rollout_retry_preserves_lineage_and_failed_attempt(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    original = store.create_rollout(run_id, queued_at_ns=1, metadata={"seed": 7})
    failed = store.update_rollout_attempt(
        original.attempt_id,
        status=RolloutStatus.FAILED,
        expected_status=RolloutStatus.QUEUING,
        reason="worker unavailable",
        timestamp_ns=2,
    )
    retry = store.retry_rollout(original.attempt_id, reason="worker restarted", queued_at_ns=3)
    assert retry.rollout_id == original.rollout_id
    assert retry.attempt_id != original.attempt_id
    assert retry.parent_attempt_id == original.attempt_id
    assert retry.attempt_index == 2
    assert retry.metadata == {"seed": 7}
    assert retry.retry_reason == "worker restarted"
    assert retry.failure_reason is None
    assert retry.started_at_ns is None
    assert retry.finished_at_ns is None
    assert store.get_rollout_attempt(original.attempt_id) == failed
    assert store.list_rollout_attempts(rollout_id=original.rollout_id) == (failed, retry)
    with pytest.raises(ContractViolation, match="already has a retry"):
        store.retry_rollout(original.attempt_id, reason="duplicate", queued_at_ns=3)
    with pytest.raises(ContractViolation, match="only a failed"):
        store.retry_rollout(retry.attempt_id, reason="still pending", queued_at_ns=4)
    third_rollout = store.create_rollout(run_id, queued_at_ns=4)
    assert third_rollout.rollout_id != original.rollout_id


@pytest.mark.parametrize("terminal", list(RunStatus)[1:])
def test_finishing_parent_atomically_converges_active_attempts_and_blocks_writes(
    tmp_path: Path,
    terminal: RunStatus,
) -> None:
    store, run_id = _store(tmp_path)
    pending = store.create_rollout(run_id, queued_at_ns=1)
    running = store.create_rollout(run_id, queued_at_ns=2)
    store.update_rollout_attempt(
        running.attempt_id,
        status=RolloutStatus.RUNNING,
        expected_status=RolloutStatus.QUEUING,
        timestamp_ns=3,
    )
    complete = store.create_rollout(run_id, queued_at_ns=4)
    failed = store.update_rollout_attempt(
        complete.attempt_id,
        status=RolloutStatus.FAILED,
        expected_status=RolloutStatus.QUEUING,
        reason="already failed",
        timestamp_ns=4,
    )
    store.finish_run(run_id, status=terminal, exit_code=0, finished_at_ns=5)
    for attempt in (pending, running):
        current = store.get_rollout_attempt(attempt.attempt_id)
        assert current.status is RolloutStatus.FAILED
        assert current.finished_at_ns == 5
        assert current.failure_reason == f"parent run {terminal.value}"
    assert store.get_rollout_attempt(complete.attempt_id) == failed
    with pytest.raises(ContractViolation, match="terminal run"):
        store.create_rollout(run_id)
    with pytest.raises(ContractViolation, match="terminal run"):
        store.retry_rollout(pending.attempt_id, reason="closed")
    with pytest.raises(ContractViolation, match="terminal run"):
        store.update_rollout_attempt(
            pending.attempt_id,
            status=RolloutStatus.RUNNING,
            expected_status=RolloutStatus.QUEUING,
        )
    events = store.list_events(run_id)
    assert [event.sequence_id for event in events] == list(range(1, 8))
    assert all(event.kind == "rollout.failed" for event in events[-2:])


@pytest.mark.parametrize(
    ("initial", "destination"),
    [
        (RolloutStatus.QUEUING, RolloutStatus.QUEUING),
        (RolloutStatus.QUEUING, RolloutStatus.SUCCEEDED),
        (RolloutStatus.RUNNING, RolloutStatus.QUEUING),
        (RolloutStatus.RUNNING, RolloutStatus.RUNNING),
        (RolloutStatus.SUCCEEDED, RolloutStatus.RUNNING),
        (RolloutStatus.FAILED, RolloutStatus.RUNNING),
    ],
)
def test_attempt_state_machine_rejects_invalid_transitions(
    tmp_path: Path,
    initial: RolloutStatus,
    destination: RolloutStatus,
) -> None:
    store, run_id = _store(tmp_path)
    attempt = store.create_rollout(run_id, queued_at_ns=1)
    if initial in {RolloutStatus.RUNNING, RolloutStatus.SUCCEEDED}:
        attempt = store.update_rollout_attempt(
            attempt.attempt_id,
            status=RolloutStatus.RUNNING,
            expected_status=RolloutStatus.QUEUING,
            timestamp_ns=2,
        )
    if initial in {RolloutStatus.SUCCEEDED, RolloutStatus.FAILED}:
        attempt = store.update_rollout_attempt(
            attempt.attempt_id,
            status=initial,
            expected_status=attempt.status,
            timestamp_ns=3,
            reason="failed" if initial is RolloutStatus.FAILED else None,
        )
    events = store.list_events(run_id)
    with pytest.raises(ContractViolation, match=r"invalid.*transition"):
        store.update_rollout_attempt(
            attempt.attempt_id,
            status=destination,
            expected_status=initial,
            timestamp_ns=4,
        )
    assert store.get_rollout_attempt(attempt.attempt_id) == attempt
    assert store.list_events(run_id) == events


def test_stale_worker_cannot_finish_attempt_or_create_duplicate_retry(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    attempt = store.create_rollout(run_id, queued_at_ns=1)
    barrier = Barrier(2)

    def start() -> bool:
        barrier.wait()
        try:
            store.update_rollout_attempt(
                attempt.attempt_id,
                status=RolloutStatus.RUNNING,
                expected_status=RolloutStatus.QUEUING,
                timestamp_ns=2,
            )
        except ContractViolation:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: start(), range(2))) == [False, True]
    assert len(store.list_events(run_id)) == 2
    with pytest.raises(ContractViolation, match="expected_status"):
        store.update_rollout_attempt(
            attempt.attempt_id,
            status=RolloutStatus.FAILED,
            expected_status=RolloutStatus.QUEUING,
            reason="stale worker",
            timestamp_ns=3,
        )
    store.update_rollout_attempt(
        attempt.attempt_id,
        status=RolloutStatus.FAILED,
        expected_status=RolloutStatus.RUNNING,
        reason="worker failed",
        timestamp_ns=3,
    )

    def retry() -> bool:
        barrier.wait()
        try:
            store.retry_rollout(attempt.attempt_id, reason="restart", queued_at_ns=4)
        except ContractViolation:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: retry(), range(2))) == [False, True]
    assert len(store.list_rollout_attempts(rollout_id=attempt.rollout_id)) == 2


def test_projection_and_event_roll_back_together_on_write_failure(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    attempt = store.create_rollout(run_id, queued_at_ns=1)
    with patch.object(
        store, "_append_rollout_event", side_effect=RuntimeError("event write failed")
    ):
        with pytest.raises(RuntimeError, match="event write failed"):
            store.update_rollout_attempt(
                attempt.attempt_id,
                status=RolloutStatus.RUNNING,
                expected_status=RolloutStatus.QUEUING,
                timestamp_ns=2,
            )
        with pytest.raises(RuntimeError, match="event write failed"):
            store.finish_run(run_id, status=RunStatus.FAILED, exit_code=1, finished_at_ns=3)
        with pytest.raises(RuntimeError, match="event write failed"):
            store.create_rollout(run_id, queued_at_ns=2)
    assert store.get_run(run_id).status is RunStatus.RUNNING
    assert store.get_rollout_attempt(attempt.attempt_id) == attempt
    assert store.list_rollout_attempts(run_id=run_id) == (attempt,)
    assert len(store.list_events(run_id)) == 1


def test_rollout_inputs_are_bounded_and_fail_without_mutation(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    for metadata in ({"huge": "x" * 65536}, {"nan": float("nan")}):
        with pytest.raises(ValueError):
            store.create_rollout(run_id, metadata=metadata)
    with pytest.raises(TypeError, match="object"):
        store.create_rollout(run_id, metadata=[])  # type: ignore[arg-type]
    for timestamp in (-1, True):
        with pytest.raises(ValueError, match="queued_at_ns"):
            store.create_rollout(run_id, queued_at_ns=timestamp)
    with pytest.raises(KeyError, match="unknown run_id"):
        store.create_rollout("run-missing")
    for operation in (
        lambda: store.get_rollout_attempt("attempt-missing"),
        lambda: store.retry_rollout("attempt-missing", reason="unknown"),
        lambda: store.update_rollout_attempt(
            "attempt-missing",
            status=RolloutStatus.RUNNING,
            expected_status=RolloutStatus.QUEUING,
        ),
    ):
        with pytest.raises(KeyError, match="unknown attempt_id"):
            operation()
    attempt = store.create_rollout(run_id, queued_at_ns=5)
    for reason in (None, "", "bad\nreason", "x" * 257):
        with pytest.raises(ValueError, match="reason"):
            store.update_rollout_attempt(
                attempt.attempt_id,
                status=RolloutStatus.FAILED,
                expected_status=RolloutStatus.QUEUING,
                reason=reason,
            )
    with pytest.raises(ValueError, match="only valid"):
        store.update_rollout_attempt(
            attempt.attempt_id,
            status=RolloutStatus.RUNNING,
            expected_status=RolloutStatus.QUEUING,
            reason="not failure",
        )
    for timestamp in (-1, True, 4):
        with pytest.raises(ValueError, match="timestamp_ns"):
            store.update_rollout_attempt(
                attempt.attempt_id,
                status=RolloutStatus.RUNNING,
                expected_status=RolloutStatus.QUEUING,
                timestamp_ns=timestamp,
            )
    store.update_rollout_attempt(
        attempt.attempt_id,
        status=RolloutStatus.FAILED,
        expected_status=RolloutStatus.QUEUING,
        reason="failure",
        timestamp_ns=6,
    )
    with pytest.raises(ValueError, match="precedes"):
        store.retry_rollout(attempt.attempt_id, reason="retry", queued_at_ns=5)
    with pytest.raises(ValueError, match="queued_at_ns"):
        store.retry_rollout(attempt.attempt_id, reason="retry", queued_at_ns=-1)
    with pytest.raises(ValueError, match="reason"):
        store.retry_rollout(attempt.attempt_id, reason="")
    with pytest.raises(ValueError, match="limit"):
        store.list_rollout_attempts(limit=0)
    with pytest.raises(ValueError, match="rollout_id"):
        store.list_rollout_attempts(rollout_id="invalid/id")
    assert len(store.list_events(run_id)) == 2


def test_existing_python_store_gets_additive_rollout_tables(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE rollout_attempts")
    reopened = TrainingStore(store.path)
    assert reopened.get_run(run_id) == store.get_run(run_id)
    assert reopened.create_rollout(run_id).attempt_index == 1
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == RUN_STORE_SCHEMA_VERSION


@pytest.mark.parametrize("attempt_status", [None, *RolloutStatus])
def test_parent_finish_rejects_backdated_lifecycle_without_mutation(
    tmp_path: Path, attempt_status: RolloutStatus | None
) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.game-v1",
        protocol_version="1.0",
        kind="collection",
        started_at_ns=10,
    )
    latest_timestamp = 10
    if attempt_status is not None:
        attempt = store.create_rollout(run.run_id, queued_at_ns=20)
        latest_timestamp = 20
        if attempt_status in {RolloutStatus.RUNNING, RolloutStatus.SUCCEEDED}:
            attempt = store.update_rollout_attempt(
                attempt.attempt_id,
                status=RolloutStatus.RUNNING,
                expected_status=RolloutStatus.QUEUING,
                timestamp_ns=30,
            )
            latest_timestamp = 30
        if attempt_status in {RolloutStatus.SUCCEEDED, RolloutStatus.FAILED}:
            store.update_rollout_attempt(
                attempt.attempt_id,
                status=attempt_status,
                expected_status=attempt.status,
                timestamp_ns=40,
                reason="worker failed" if attempt_status is RolloutStatus.FAILED else None,
            )
            latest_timestamp = 40
    state = store.run_state(run.run_id, "adapter/progress", schema_version=1)
    state["level"] = 2
    attempts = store.list_rollout_attempts(run_id=run.run_id)
    events = store.list_events(run.run_id)
    with pytest.raises(ValueError, match="finished_at_ns precedes"):
        store.finish_run(
            run.run_id,
            status=RunStatus.INTERRUPTED,
            exit_code=1,
            finished_at_ns=latest_timestamp - 1,
        )
    assert store.get_run(run.run_id) == run
    assert store.list_rollout_attempts(run_id=run.run_id) == attempts
    assert store.list_events(run.run_id) == events
    assert state.snapshot() == {"level": 2}
    finished = store.finish_run(
        run.run_id,
        status=RunStatus.INTERRUPTED,
        exit_code=1,
        finished_at_ns=latest_timestamp,
    )
    assert finished.finished_at_ns == latest_timestamp
    assert all(
        attempt.finished_at_ns is not None and attempt.finished_at_ns <= latest_timestamp
        for attempt in store.list_rollout_attempts(run_id=run.run_id)
    )


def test_parent_finish_processes_bounded_batches_without_skipping_attempts(tmp_path: Path) -> None:
    store, run_id = _store(tmp_path)
    for index in range(7):
        attempt = store.create_rollout(
            run_id, queued_at_ns=index + 1, metadata={"context": "x" * 63000}
        )
        if index % 2:
            store.update_rollout_attempt(
                attempt.attempt_id,
                status=RolloutStatus.RUNNING,
                expected_status=RolloutStatus.QUEUING,
                timestamp_ns=8,
            )
    queries: list[str] = []
    original_connect = store._connect

    @contextmanager
    def traced_connect() -> Iterator[sqlite3.Connection]:
        with original_connect() as connection:
            connection.set_trace_callback(queries.append)
            yield connection

    with (
        patch.object(run_store_module, "_ROLLOUT_FINALIZE_BATCH_SIZE", 2),
        patch.object(store, "_connect", traced_connect),
    ):
        store.finish_run(run_id, status=RunStatus.INTERRUPTED, exit_code=1, finished_at_ns=10)
    batch_queries = [
        query
        for query in queries
        if query.startswith("SELECT * FROM rollout_attempts WHERE run_id")
    ]
    assert len(batch_queries) == 5  # Four bounded batches followed by an empty batch.
    assert all(query.endswith("LIMIT 2") for query in batch_queries)
    attempts = store.list_rollout_attempts(run_id=run_id)
    assert len(attempts) == 7
    assert all(attempt.status is RolloutStatus.FAILED for attempt in attempts)
    assert all(attempt.finished_at_ns == 10 for attempt in attempts)
    failures = [event for event in store.list_events(run_id) if event.kind == "rollout.failed"]
    assert len(failures) == 7
    assert {event.payload["attempt_id"] for event in failures} == {
        attempt.attempt_id for attempt in attempts
    }


def test_later_batch_failure_rolls_back_parent_children_events_and_run_state(
    tmp_path: Path,
) -> None:
    store, run_id = _store(tmp_path)
    for index in range(5):
        store.create_rollout(run_id, queued_at_ns=index + 1)
    run = store.get_run(run_id)
    attempts = store.list_rollout_attempts(run_id=run_id)
    events = store.list_events(run_id)
    state = store.run_state(run_id, "adapter/progress", schema_version=1)
    state["level"] = 2
    original_append = store._append_rollout_event
    append_count = 0

    def fail_third_event(
        connection: sqlite3.Connection, attempt_id: str, timestamp_ns: int
    ) -> run_store_module.RolloutAttempt:
        nonlocal append_count
        append_count += 1
        if append_count == 3:
            raise RuntimeError("later batch event failure")
        return original_append(connection, attempt_id, timestamp_ns)

    with (
        patch.object(run_store_module, "_ROLLOUT_FINALIZE_BATCH_SIZE", 2),
        patch.object(store, "_append_rollout_event", side_effect=fail_third_event),
        pytest.raises(RuntimeError, match="later batch event failure"),
    ):
        store.finish_run(run_id, status=RunStatus.INTERRUPTED, exit_code=1, finished_at_ns=10)
    assert append_count == 3
    assert store.get_run(run_id) == run
    assert store.list_rollout_attempts(run_id=run_id) == attempts
    assert store.list_events(run_id) == events
    assert state.snapshot() == {"level": 2}


def test_rollout_queue_cannot_precede_parent_start(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.game-v1",
        protocol_version="1.0",
        kind="collection",
        started_at_ns=10,
    )
    with pytest.raises(ValueError, match="queued_at_ns precedes the parent run start"):
        store.create_rollout(run.run_id, queued_at_ns=9)
    assert store.list_rollout_attempts(run_id=run.run_id) == ()
    assert store.list_events(run.run_id) == ()
    assert store.create_rollout(run.run_id, queued_at_ns=10).queued_at_ns == 10
