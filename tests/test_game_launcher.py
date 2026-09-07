from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from game_learning_runtime import (
    GameLaunchConfig,
    GameLauncher,
    GameLaunchError,
    LaunchCommand,
    TrainingLauncher,
    load_project_game_launch,
)
from game_learning_runtime.game_launcher import main

GAME_CODE = (
    "import os, pathlib, time; "
    "root = pathlib.Path(os.environ['GLR_GAME_INSTANCE_DIR']); "
    "(root / 'ready.json').write_text(os.environ['GLR_GAME_INSTANCE_ID'], encoding='utf-8'); "
    "time.sleep(30)"
)
TRAINER_CODE = (
    "import json, os, pathlib; "
    "manifest = pathlib.Path(os.environ['GLR_GAME_INSTANCES_MANIFEST']); "
    "data = json.loads(manifest.read_text(encoding='utf-8')); "
    "pathlib.Path(os.environ['GLR_RUN_DIR'], 'trainer-seen.json').write_text("
    "json.dumps(data), encoding='utf-8')"
)


def _config(*, instances: int = 2, max_parallel: int = 2) -> GameLaunchConfig:
    return GameLaunchConfig.from_mapping(
        {
            "schema_version": "glr.game-launch.v1",
            "game_id": "example.game",
            "command": {"argv": [sys.executable, "-c", GAME_CODE]},
            "instances": instances,
            "parallel": instances > 1,
            "max_parallel": max_parallel,
            "readiness": {"kind": "file", "path": "ready.json"},
            "startup_timeout_seconds": 3,
            "shutdown_timeout_seconds": 2,
        }
    )


def test_launcher_starts_bounded_parallel_instances_and_writes_manifest(tmp_path: Path) -> None:
    launcher = GameLauncher(_config(), project_root=tmp_path)
    with launcher.start(tmp_path / ".glr" / "runs" / "one") as running:
        assert len(running.instances) == 2
        assert all(item.process.poll() is None for item in running.instances)
        assert [item.instance_id for item in running.instances] == [
            "example.game-0000",
            "example.game-0001",
        ]
        manifest = json.loads(running.manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "glr.game-instances.v1"
        assert [item["index"] for item in manifest["instances"]] == [0, 1]
        assert all(Path(item["directory"]).is_dir() for item in manifest["instances"])
    assert all(item.process.poll() is not None for item in running.instances)


def test_training_starts_after_game_readiness_and_passes_instance_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / ".glr" / "runs" / "train"
    result = TrainingLauncher(_config(instances=3, max_parallel=2), project_root=tmp_path).run(
        LaunchCommand((sys.executable, "-c", TRAINER_CODE, "{run_dir}")), run_dir=run_dir
    )
    assert result.return_code == 0
    assert result.instance_ids == (
        "example.game-0000",
        "example.game-0001",
        "example.game-0002",
    )
    seen = json.loads((run_dir / "trainer-seen.json").read_text(encoding="utf-8"))
    assert len(seen["instances"]) == 3
    assert not any(item["directory"] == "" for item in seen["instances"])


def test_startup_failure_terminates_already_started_instances(tmp_path: Path) -> None:
    config = GameLaunchConfig.from_mapping(
        {
            "schema_version": "glr.game-launch.v1",
            "game_id": "example.game",
            "command": {
                "argv": [
                    sys.executable,
                    "-c",
                    "import os, pathlib, time; "
                    "p = pathlib.Path(os.environ['GLR_GAME_INSTANCE_DIR']); "
                    "(p / 'ready.json').write_text('ok'); "
                    "time.sleep(30)",
                ]
            },
            "instances": 1,
            "readiness": {"kind": "file", "path": "missing.json"},
            "startup_timeout_seconds": 0.1,
        }
    )
    launcher = GameLauncher(config, project_root=tmp_path)
    with pytest.raises(GameLaunchError, match="did not publish readiness"):
        launcher.start(tmp_path / "run")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("instances", 0, "between 1 and 64"),
        ("max_parallel", 3, "between 1 and game.instances"),
        ("working_dir", "../outside", "project-relative"),
        ("environment", {"GLR_GAME_INSTANCE_ID": "spoof"}, "reserved key"),
    ],
)
def test_launch_config_rejects_unsafe_values(field: str, value: object, message: str) -> None:
    source: dict[str, object] = {
        "schema_version": "glr.game-launch.v1",
        "game_id": "example.game",
        "command": {"argv": ["game.exe"]},
        "instances": 2,
        "parallel": True,
        "max_parallel": 2,
    }
    source[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        GameLaunchConfig.from_mapping(source)


def test_project_loader_reads_game_and_trainer_roles(tmp_path: Path) -> None:
    project = {
        "schema_version": "glr.project.v1",
        "game": _config().to_mapping(),
        "trainer": {"argv": [sys.executable, "train.py"]},
    }
    path = tmp_path / "glr-project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    config, trainer = load_project_game_launch(path)
    assert config.instances == 2
    assert trainer.argv == (sys.executable, "train.py")


def test_command_line_orchestrates_project_and_emits_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = {
        "schema_version": "glr.project.v1",
        "game": _config(instances=1, max_parallel=1).to_mapping(),
        "trainer": {"argv": [sys.executable, "-c", TRAINER_CODE]},
    }
    path = tmp_path / "glr-project.json"
    path.write_text(json.dumps(project), encoding="utf-8")

    assert main(["--project", str(path), "--project-root", str(tmp_path), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == "glr.launch.v1"
    assert output["status"] == "succeeded"
    assert output["instance_ids"] == ["example.game-0000"]
