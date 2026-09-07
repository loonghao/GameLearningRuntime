"""Synthetic rollout lifecycle and learner queue barrier; no game or learner required."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from game_learning_runtime import (
    BoundedActorQueue,
    RolloutAttempt,
    RolloutStatus,
    RunRecord,
    RunStatus,
    SyncCollector,
    TrainingStore,
)
from game_learning_runtime.examples import CounterEnvironment, always_increment


def run_demo(output_dir: Path) -> dict[str, object]:
    """Write a fresh evidence directory; refuse to replace an earlier demo."""

    output_dir.mkdir(parents=True, exist_ok=False)
    store = TrainingStore(output_dir / "runs.sqlite3")
    run = store.create_run(
        environment_id="counter-v1",
        protocol_version="1.0",
        kind="synthetic-rollout-demo",
        metadata={"synthetic": True, "learner_update_performed": False},
    )
    try:
        summary = _collect_demo(store, run, output_dir)
        store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
        return {**summary, "status": RunStatus.SUCCEEDED.value}
    except Exception as error:
        try:
            store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
        except Exception as finalization_error:
            raise error from finalization_error
        raise


def _collect_demo(store: TrainingStore, run: RunRecord, output_dir: Path) -> dict[str, object]:
    first = store.create_rollout(run.run_id, metadata={"actor_id": "actor-0"})
    first = store.update_rollout_attempt(
        first.attempt_id,
        expected_status=RolloutStatus.QUEUING,
        status=RolloutStatus.FAILED,
        reason="synthetic pre-collection failure; no action executed",
    )
    # This explicit retry only creates a ledger entry. It never replays a game action.
    retry = store.retry_rollout(first.attempt_id, reason="retry the synthetic collector")
    retry = store.update_rollout_attempt(
        retry.attempt_id,
        expected_status=RolloutStatus.QUEUING,
        status=RolloutStatus.RUNNING,
    )
    queue = BoundedActorQueue(3, max_policy_version_lag=1)
    collector = SyncCollector(CounterEnvironment())
    try:
        for version in (0, 0, 1):
            queue.put(collector.collect(always_increment, steps=1, policy_version=version))
        retry = store.update_rollout_attempt(
            retry.attempt_id,
            expected_status=RolloutStatus.RUNNING,
            status=RolloutStatus.SUCCEEDED,
        )
        # Acknowledge only after the consumer succeeds. Here the consumer is synthetic;
        # no learner parameters are updated and no game success is inferred.
        queue.commit(queue.get_nowait())
        drained = queue.drain(timeout=1)
        store.append_event(run.run_id, kind="queue.drained", payload=drained.as_dict())
        # In a real learner its owner publishes weights while the lease barrier is held.
        queue.set_learner_policy_version(2)
        store.append_event(run.run_id, kind="queue.policy_changed", payload=queue.run_summary())
        queue.resume()
        # Version 0 is now stale and discarded; version 1 is still within the cutoff.
        queue.commit(queue.get_nowait())
        metrics = queue.metrics()
        store.append_event(run.run_id, kind="queue.completed", payload=metrics.as_dict())
        for name, value in (
            ("queue.committed_unrolls", metrics.committed_unrolls),
            ("queue.stale_dropped_unrolls", metrics.stale_dropped_unrolls),
            ("queue.drain_latency_ns_total", metrics.drain_latency_ns_total),
        ):
            store.record_metric(run.run_id, name=name, value=value)
        for attempt in (first, retry):
            _write_sidecar(store, output_dir, attempt)
    finally:
        queue.close()
    return {
        "run_id": run.run_id,
        "rollout_id": first.rollout_id,
        "attempts": len(store.list_rollout_attempts(run_id=run.run_id)),
        "committed_unrolls": metrics.committed_unrolls,
        "stale_dropped_unrolls": metrics.stale_dropped_unrolls,
        "artifact_count": len(store.list_artifacts(run.run_id)),
    }


def _write_sidecar(store: TrainingStore, output_dir: Path, attempt: RolloutAttempt) -> None:
    path = output_dir / f"{attempt.attempt_id}.json"
    path.write_text(json.dumps(attempt.to_mapping(), indent=2) + "\n", encoding="utf-8")
    store.register_artifact(
        attempt.run_id,
        path=path.name,
        source=path,
        role="rollout-sidecar",
        media_type="application/json",
        metadata={"rollout_id": attempt.rollout_id, "attempt_id": attempt.attempt_id},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new evidence directory")
    arguments = parser.parse_args(argv)
    print(json.dumps(run_demo(arguments.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
