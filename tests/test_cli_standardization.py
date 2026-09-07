from __future__ import annotations

import json
import sys
from pathlib import Path

from game_learning_runtime.cli import main


def _project(root: Path, *, trainer: list[str], game: dict[str, object] | None = None) -> None:
    (root / "bridge").mkdir()
    value: dict[str, object] = {
        "schema_version": "glr.project.v1",
        "environment_id": "example.adventure-v1",
        "environment_family": "action-rpg",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": [sys.executable, "-c", "print('runtime')"]},
        "trainer": {"argv": trainer},
        "player": {"argv": [sys.executable, "-c", "print('player')"]},
        "researcher": None,
        "planner": None,
        "evaluator": None,
        "capture": None,
    }
    if game is not None:
        value["game"] = game
    (root / "glr-project.json").write_text(json.dumps(value), encoding="utf-8")


def test_cli_defaults_to_tables_and_supports_json_format(tmp_path: Path, capsys: object) -> None:
    _project(tmp_path, trainer=[sys.executable, "-c", "print('train')"])

    assert main(["--project", str(tmp_path), "doctor"]) == 0
    table = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "doctor" in table
    assert "| role" in table
    assert "| ready" in table

    assert main(["--project", str(tmp_path), "--format", "json", "doctor"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["schema_version"] == "glr.cli-output.v1"
    assert payload["command"] == "doctor"
    assert payload["data"]["ready"] is True


def test_cli_train_launches_configured_game_before_trainer(tmp_path: Path, capsys: object) -> None:
    game_code = (
        "import os, pathlib, time; "
        "p = pathlib.Path(os.environ['GLR_GAME_INSTANCE_DIR']); "
        "(p / 'ready.json').write_text('ready', encoding='utf-8'); "
        "time.sleep(30)"
    )
    trainer_code = (
        "import json, os, pathlib; "
        "manifest = pathlib.Path(os.environ['GLR_GAME_INSTANCES_MANIFEST']); "
        "pathlib.Path(os.environ['GLR_RUN_DIR'], 'seen.json').write_text("
        "json.dumps(json.loads(manifest.read_text(encoding='utf-8'))), encoding='utf-8')"
    )
    game = {
        "schema_version": "glr.game-launch.v1",
        "game_id": "example.game",
        "command": {"argv": [sys.executable, "-c", game_code, "{instance_dir}"]},
        "instances": 2,
        "parallel": True,
        "max_parallel": 2,
        "readiness": {"kind": "file", "path": "ready.json"},
        "startup_timeout_seconds": 3,
        "shutdown_timeout_seconds": 2,
    }
    _project(tmp_path, trainer=[sys.executable, "-c", trainer_code], game=game)

    assert main(["--project", str(tmp_path), "--format", "json", "train", "--no-capture"]) == 0
    payload = json.loads(capsys.readouterr().out)
    run_dir = tmp_path / ".glr" / "runs" / payload["data"]["run_id"]
    manifest = json.loads((run_dir / "game-instances.json").read_text(encoding="utf-8"))
    assert len(manifest["instances"]) == 2
    assert json.loads((run_dir / "seen.json").read_text(encoding="utf-8"))["game_id"] == (
        "example.game"
    )
