"""Exercise the compiled CLI and Python API against the same run store."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
from contextlib import closing
from pathlib import Path

from game_learning_runtime.run_store import RunStatus, TrainingStore

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cli = ROOT / "target/debug" / ("glr.exe" if os.name == "nt" else "glr")
    if not cli.is_file():
        raise RuntimeError("Build the CLI with cargo build -p glr-cli first")
    for python_first in (False, True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bridge").mkdir()
            (root / "glr-project.json").write_text(
                json.dumps(
                    {
                        "schema_version": "glr.project.v1",
                        "environment_id": "synthetic.interop",
                        "environment_family": "synthetic",
                        "protocol_version": "1.0",
                        "data_dir": ".glr",
                        "bridge_path": "bridge",
                        "runtime": {"argv": [str(cli), "--version"]},
                        "trainer": {"argv": [str(cli), "--version"]},
                        "player": {"argv": [str(cli), "--version"]},
                    }
                ),
                encoding="utf-8",
            )

            def command(*argv: str, root: Path = root) -> dict:
                result = subprocess.run(
                    [str(cli), "--project", str(root), "--json", *argv],
                    env={**os.environ, "GLR_NO_UPDATE_CHECK": "1"},
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    raise RuntimeError(result.stderr)
                return json.loads(result.stdout)

            if not python_first:
                command("train")
            path = root / ".glr/runs.sqlite3"
            store = TrainingStore(path)
            record = store.create_run(
                environment_id="synthetic.interop", protocol_version="1.0", kind="training"
            )
            store.record_metric(record.run_id, name="score", value=42.0)
            (root / ".glr/runs" / record.run_id).mkdir(parents=True)
            store.finish_run(record.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
            command("train")
            shown = command("runs", "show", record.run_id)["data"]
            assert shown["metrics"][0]["value"] == 42.0
            command("report", "build", record.run_id)
            assert TrainingStore(path).get_run(record.run_id).run_id == record.run_id
            with closing(sqlite3.connect(path)) as connection:
                assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
                connection.execute("PRAGMA user_version = 99")
            failure = subprocess.run(
                [str(cli), "--project", str(root), "--json", "runs", "list"],
                env={**os.environ, "GLR_NO_UPDATE_CHECK": "1"},
                capture_output=True,
                text=True,
            )
            assert failure.returncode != 0
            assert "upgrade GLR" in failure.stderr
            with closing(sqlite3.connect(path)) as connection:
                assert connection.execute("PRAGMA user_version").fetchone()[0] == 99
    print("CLI/Python shared-store round trips passed in both creation orders")


if __name__ == "__main__":
    main()
