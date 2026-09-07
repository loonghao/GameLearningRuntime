from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from game_learning_runtime import (
    BoundedActorQueue,
    RolloutAttempt,
    RolloutStatus,
    RunStatus,
    SyncCollector,
    TrainingStore,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rollout_control_demo.py"


def test_synthetic_demo_persists_retry_queue_metrics_and_hashed_sidecars(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(process.stdout)
    assert summary == {
        "run_id": summary["run_id"],
        "rollout_id": summary["rollout_id"],
        "status": "succeeded",
        "attempts": 2,
        "committed_unrolls": 2,
        "stale_dropped_unrolls": 1,
        "artifact_count": 2,
    }
    store = TrainingStore(output / "runs.sqlite3")
    run = store.get_run(summary["run_id"])
    assert run.status is RunStatus.SUCCEEDED
    assert run.metadata == {"synthetic": True, "learner_update_performed": False}
    first, retry = store.list_rollout_attempts(rollout_id=summary["rollout_id"])
    assert isinstance(retry, RolloutAttempt)
    assert first.status is RolloutStatus.FAILED
    assert retry.status is RolloutStatus.SUCCEEDED
    assert retry.parent_attempt_id == first.attempt_id
    assert retry.attempt_index == 2
    events = store.list_events(run.run_id)
    assert [event.sequence_id for event in events] == list(range(1, len(events) + 1))
    assert [event.kind for event in events] == [
        "rollout.queuing",
        "rollout.failed",
        "rollout.queuing",
        "rollout.running",
        "rollout.succeeded",
        "queue.drained",
        "queue.policy_changed",
        "queue.completed",
    ]
    assert events[5].payload["paused"] is True
    assert events[5].payload["depth"] == 2
    assert events[5].payload["in_flight_unrolls"] == 0
    assert events[6].payload["actor_queue"]["carry_over_unrolls"] == 2
    assert events[7].payload["stale_dropped_unrolls"] == 1
    assert events[7].payload["uncommitted_unrolls"] == 0
    for artifact in store.list_artifacts(run.run_id):
        content = (output / artifact.path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == artifact.sha256
        sidecar = json.loads(content)
        assert sidecar["schema_version"] == "glr.rollout.v1"
        attempt = store.get_rollout_attempt(sidecar["attempt_id"])
        assert sidecar == attempt.to_mapping()
        assert sidecar == dict(events[attempt.sequence_id - 1].payload)


def test_demo_refuses_to_overwrite_an_existing_output_directory(tmp_path: Path) -> None:
    run_demo = runpy.run_path(str(SCRIPT))["run_demo"]
    with pytest.raises(FileExistsError):
        run_demo(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["collection", "artifact"])
def test_demo_failure_finalizes_parent_and_pending_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    run_demo = runpy.run_path(str(SCRIPT))["run_demo"]
    output = tmp_path / "failed-evidence"
    closed = []
    close = BoundedActorQueue.close

    def record_close(queue: BoundedActorQueue) -> None:
        close(queue)
        closed.append(queue.closed)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"synthetic {failure} error")

    monkeypatch.setattr(BoundedActorQueue, "close", record_close)
    if failure == "collection":
        monkeypatch.setattr(SyncCollector, "collect", fail)
    else:
        monkeypatch.setattr(TrainingStore, "register_artifact", fail)
    with pytest.raises(RuntimeError, match=f"synthetic {failure} error"):
        run_demo(output)
    assert closed == [True]
    store = TrainingStore(output / "runs.sqlite3")
    attempts = store.list_rollout_attempts()
    assert len(attempts) == 2
    assert all(
        attempt.status in {RolloutStatus.FAILED, RolloutStatus.SUCCEEDED} for attempt in attempts
    )
    parent = store.get_run(attempts[0].run_id)
    assert parent.status is RunStatus.FAILED
    assert parent.exit_code == 1
    assert not store.list_artifacts(parent.run_id)
