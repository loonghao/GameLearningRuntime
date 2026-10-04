"""Main-version persistence regressions, without game or process execution."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from game_learning_runtime.run_store import RUN_STORE_SCHEMA_VERSION, TrainingStore


def test_schema_one_legacy_data_keeps_config_unknown_after_additive_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as database:
        database.executescript(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, environment_id TEXT NOT NULL, "
            "protocol_version TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, "
            "started_at_ns INTEGER NOT NULL, finished_at_ns INTEGER, exit_code INTEGER, "
            "metadata_json TEXT NOT NULL);"
            "CREATE TABLE metrics (metric_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "run_id TEXT NOT NULL REFERENCES runs(run_id), timestamp_ns INTEGER NOT NULL, "
            "name TEXT NOT NULL, value REAL NOT NULL, step_id INTEGER, "
            "metadata_json TEXT NOT NULL);"
            "INSERT INTO runs VALUES ('run.legacy', 'synthetic.game', '1.0', 'evaluation', "
            "'running', 1, NULL, NULL, '{}');"
            "INSERT INTO metrics VALUES (1, 'run.legacy', 2, 'objective.progress', 1, NULL, '{}');"
            "PRAGMA user_version=1;"
        )
    migrated = TrainingStore(path)
    assert migrated.get_run("run.legacy").environment_config_digest is None
    assert migrated.list_metrics("run.legacy")[0].environment_config_digest is None
    with sqlite3.connect(path) as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == RUN_STORE_SCHEMA_VERSION
        tables = {
            row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"run_state", "rollout_attempts"}.issubset(tables)
    reopened = TrainingStore(path)
    assert reopened.get_run("run.legacy").environment_config_digest is None


def test_config_digest_is_persisted_on_run_and_metric_and_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite3"
    store = TrainingStore(path)
    run = store.create_run(
        environment_id="synthetic.game",
        protocol_version="1.0",
        kind="evaluation",
        environment_config_digest="a" * 64,
        started_at_ns=1,
    )
    recorded = store.record_metric(
        run.run_id, name="objective.progress", value=1, environment_config_digest="a" * 64
    )
    reopened = TrainingStore(path)
    assert reopened.get_run(run.run_id).environment_config_digest == "a" * 64
    assert (
        reopened.list_metrics(run.run_id)[0].environment_config_digest
        == recorded.environment_config_digest
    )


@pytest.mark.parametrize("digest", ["", "A" * 64, "a" * 63, "x" * 64, 0])
def test_config_digest_is_not_a_freeform_relabel(tmp_path: Path, digest: object) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    with pytest.raises(ValueError, match="digest"):
        store.create_run(
            environment_id="synthetic.game",
            protocol_version="1.0",
            kind="evaluation",
            environment_config_digest=digest,  # type: ignore[arg-type]
        )
