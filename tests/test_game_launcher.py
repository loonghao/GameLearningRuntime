from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from game_learning_runtime import (
    GameLaunchConfig,
    GameLauncher,
    GameLaunchError,
    LaunchCommand,
    TrainingLauncher,
    load_project_game_launch,
)
from game_learning_runtime.game_launcher import (
    ReadinessConfig,
    RunningGameInstance,
    RunningGameSet,
    load_game_launch_config,
    main,
)

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


def test_startup_failure_terminates_already_started_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    spawned: list[RunningGameInstance] = []
    original_spawn = launcher._spawn

    def record_spawn(**kwargs: object) -> RunningGameInstance:
        instance = original_spawn(**kwargs)
        spawned.append(instance)
        return instance

    monkeypatch.setattr(launcher, "_spawn", record_spawn)
    with pytest.raises(GameLaunchError, match="did not publish readiness"):
        launcher.start(tmp_path / "run")
    assert len(spawned) == 1
    assert spawned[0].process.poll() is not None
    assert spawned[0]._stdout.closed and spawned[0]._stderr.closed
    assert not (tmp_path / "run" / "game-instances.json").exists()


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


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("schema_version", "glr.game-launch.v99", ValueError, "schema version"),
        ("game_id", "../outside", ValueError, "lowercase identifier"),
        ("instances", True, TypeError, "must be an integer"),
        ("instances", 65, ValueError, "between 1 and 64"),
        ("parallel", 1, TypeError, "must be a boolean"),
        ("max_parallel", True, TypeError, "must be an integer"),
        ("max_parallel", 0, ValueError, "between 1 and game.instances"),
        ("command", [], TypeError, "must be an object"),
        ("readiness", [], TypeError, "must be an object"),
        ("environment", [], TypeError, "must be an object"),
        ("environment", {1: "value"}, TypeError, "requires string keys"),
        ("environment", {"bad-key": "value"}, ValueError, "printable strings"),
        ("environment", {"VALID": 1}, ValueError, "printable strings"),
        ("environment", {"VALID": "line\nbreak"}, ValueError, "printable strings"),
        ("working_dir", "", ValueError, "project-relative"),
        ("working_dir", "/outside", ValueError, "project-relative"),
        ("working_dir", "C:/outside", ValueError, "project-relative"),
        ("working_dir", "sub\\directory", ValueError, "project-relative"),
        ("startup_timeout_seconds", True, TypeError, "must be a number"),
        ("startup_timeout_seconds", "30", TypeError, "must be a number"),
        ("startup_timeout_seconds", float("nan"), ValueError, "greater than zero"),
        ("startup_timeout_seconds", float("inf"), ValueError, "greater than zero"),
        ("startup_timeout_seconds", 0, ValueError, "greater than zero"),
        ("startup_timeout_seconds", 86_401, ValueError, "at most 86400"),
        ("shutdown_timeout_seconds", 601, ValueError, "at most 600"),
        ("unknown_option", True, ValueError, "unexpected"),
    ],
)
def test_launch_config_rejects_invalid_contract_values(
    field: str, value: object, error: type[Exception], message: str
) -> None:
    source = _config().to_mapping()
    source[field] = value
    with pytest.raises(error, match=message):
        GameLaunchConfig.from_mapping(source)


@pytest.mark.parametrize("field", ["schema_version", "game_id", "command"])
def test_launch_config_requires_identity_and_command_fields(field: str) -> None:
    source = _config().to_mapping()
    del source[field]
    with pytest.raises(ValueError, match="missing"):
        GameLaunchConfig.from_mapping(source)


def test_direct_launch_config_preserves_validation_and_parallel_defaults() -> None:
    command = LaunchCommand(("game.exe",))
    serial = GameLaunchConfig("example.game", command, instances=3)
    parallel = GameLaunchConfig("example.game", command, instances=3, parallel=True)
    assert serial.max_parallel == 1
    assert parallel.max_parallel == 3
    assert GameLaunchConfig.from_mapping(serial.to_mapping()) == serial
    with pytest.raises(ValueError, match="must be 1 when"):
        replace(serial, max_parallel=2)
    with pytest.raises(TypeError, match="LaunchCommand"):
        replace(serial, command=["game.exe"])
    with pytest.raises(TypeError, match="ReadinessConfig"):
        replace(serial, readiness={"kind": "file"})
    with pytest.raises(TypeError, match="environment must be an object"):
        replace(serial, environment=[])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ((), "cannot be empty"),
        (("game.exe", ""), "non-empty printable"),
        (("game.exe", 1), "non-empty printable"),
        (("game.exe", "bad\x00argument"), "non-empty printable"),
        (("game.exe", "--root={project_root}"), "complete argv entry"),
        (("game.exe", "{unknown}"), "unsupported command placeholder"),
    ],
)
def test_command_rejects_invalid_argv(argv: tuple[object, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LaunchCommand(argv)


@pytest.mark.parametrize("value", ["game.exe --option", b"game.exe", bytearray(b"game"), 42])
def test_command_mapping_requires_an_argv_array(value: object) -> None:
    with pytest.raises(TypeError, match="argv must be an array"):
        LaunchCommand.from_mapping({"argv": value}, path="trainer")


def test_command_requires_argv_and_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="missing"):
        LaunchCommand.from_mapping({}, path="trainer")
    with pytest.raises(ValueError, match="unexpected"):
        LaunchCommand.from_mapping({"argv": ["game.exe"], "shell": True}, path="trainer")


def test_command_expansion_keeps_arguments_literal_and_requires_bound_placeholders() -> None:
    command = LaunchCommand(("game.exe", "{run_dir}", "literal;&&$value"))
    assert command.expand({"run_dir": "a directory/with spaces"}) == (
        "game.exe",
        "a directory/with spaces",
        "literal;&&$value",
    )
    with pytest.raises(GameLaunchError, match="missing command placeholder value: run_dir"):
        command.expand({})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"kind": "port"}, "must be 'process-alive' or 'file'"),
        ({"kind": "process-alive", "path": "ready.json"}, "only valid for file readiness"),
        ({"kind": "file"}, "project-relative"),
        ({"kind": "file", "path": "../ready.json"}, "project-relative"),
        ({"kind": "file", "path": "/ready.json"}, "project-relative"),
    ],
)
def test_readiness_rejects_unsupported_or_escaping_signals(
    value: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ReadinessConfig.from_mapping(value)


def test_direct_readiness_configuration_enforces_the_same_signal_contract() -> None:
    with pytest.raises(ValueError, match="must be 'process-alive' or 'file'"):
        ReadinessConfig(kind="port")
    with pytest.raises(ValueError, match="only valid for file readiness"):
        ReadinessConfig(path="ready.json")
    with pytest.raises(ValueError, match=r"requires game\.readiness\.path"):
        ReadinessConfig(kind="file")


def test_standalone_configuration_roundtrips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "game.json"
    config = _config()
    path.write_text(json.dumps(config.to_mapping()), encoding="utf-8")
    assert load_game_launch_config(path) == config


@pytest.mark.parametrize(
    ("content", "message"),
    [("not-json", "must be UTF-8 JSON"), ("[]", "must be an object"), ("{}", "must contain")],
)
def test_project_loader_rejects_malformed_documents(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "project.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        load_project_game_launch(path)


def test_configuration_loader_rejects_missing_directory_and_oversized_files(
    tmp_path: Path,
) -> None:
    for path in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(FileNotFoundError, match="must be a regular file"):
            load_game_launch_config(path)
    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as stream:
        stream.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="8 MiB limit"):
        load_game_launch_config(oversized)


def _mock_instance(tmp_path: Path, *, index: int = 0) -> RunningGameInstance:
    process = Mock(spec=subprocess.Popen)
    process.pid = 100 + index
    process.poll.return_value = None
    return RunningGameInstance(
        instance_id=f"example.game-{index:04d}",
        index=index,
        directory=tmp_path,
        process=process,
        stdout_path=tmp_path / f"stdout-{index}.log",
        stderr_path=tmp_path / f"stderr-{index}.log",
        _stdout=BytesIO(),
        _stderr=BytesIO(),
    )


def test_shutdown_reaps_unresponsive_games_and_closes_logs_idempotently(tmp_path: Path) -> None:
    stubborn = _mock_instance(tmp_path)
    exited = _mock_instance(tmp_path, index=1)
    stubborn.process.wait.side_effect = [subprocess.TimeoutExpired("game", 0.1), -9]
    exited.process.poll.return_value = 7
    games = RunningGameSet(_config(), tmp_path, (stubborn, exited), tmp_path / "manifest.json")

    games.close()
    games.close()

    stubborn.process.terminate.assert_called_once_with()
    stubborn.process.kill.assert_called_once_with()
    assert stubborn.process.wait.call_count == 2
    exited.process.terminate.assert_not_called()
    exited.process.wait.assert_not_called()
    assert all(item._stdout.closed and item._stderr.closed for item in games.instances)


def test_launcher_rejects_paths_outside_project_before_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn = Mock()
    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(FileNotFoundError, match="project root is not a directory"):
        GameLauncher(_config(), project_root=tmp_path / "missing")
    launcher = GameLauncher(_config(), project_root=tmp_path)
    with pytest.raises(GameLaunchError, match="run_dir must stay inside"):
        launcher.start("../outside")
    spawn.assert_not_called()


def test_spawn_failure_closes_logs_and_does_not_publish_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn = Mock(side_effect=OSError("process creation denied"))
    monkeypatch.setattr(subprocess, "Popen", spawn)
    launcher = GameLauncher(_config(instances=1, max_parallel=1), project_root=tmp_path)
    with pytest.raises(GameLaunchError, match="process creation denied"):
        launcher.start("run")
    assert spawn.call_args.kwargs["shell"] is False
    assert spawn.call_args.kwargs["stdout"].closed
    assert spawn.call_args.kwargs["stderr"].closed
    assert not (tmp_path / "run" / "game-instances.json").exists()


def test_launcher_uses_project_working_directory_and_owned_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "game").mkdir()
    config = GameLaunchConfig(
        "example.game",
        LaunchCommand(("game.exe", "{instance_index}", "{instance_dir}")),
        working_dir="game",
        environment={"GAME_MODE": "train"},
    )
    process = Mock(spec=subprocess.Popen)
    process.pid = 123
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    with GameLauncher(config, project_root=tmp_path).start("run") as games:
        args, kwargs = spawn.call_args
        assert args[0] == ("game.exe", "0", str(games.instances[0].directory))
        assert kwargs["cwd"] == tmp_path / "game"
        assert kwargs["env"]["GAME_MODE"] == "train"
        assert kwargs["env"]["GLR_GAME_PARALLEL"] == "0"
        assert kwargs["env"]["GLR_GAME_INSTANCE_COUNT"] == "1"
        assert kwargs["shell"] is False
    process.terminate.assert_called_once_with()


def test_game_exit_before_readiness_closes_logs_without_publishing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = Mock(spec=subprocess.Popen)
    process.poll.return_value = 17
    spawn = Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(GameLaunchError, match="exited before readiness with code 17"):
        GameLauncher(_config(), project_root=tmp_path).start("run")
    assert all(call.kwargs["stdout"].closed for call in spawn.call_args_list)
    assert all(call.kwargs["stderr"].closed for call in spawn.call_args_list)
    assert not (tmp_path / "run" / "game-instances.json").exists()


def test_training_spawn_failure_always_releases_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = _mock_instance(tmp_path)
    games = RunningGameSet(_config(), tmp_path, (instance,), tmp_path / "manifest.json")
    launcher = TrainingLauncher(_config(), project_root=tmp_path)
    monkeypatch.setattr(launcher.game_launcher, "start", Mock(return_value=games))
    spawn = Mock(side_effect=OSError("trainer not found"))
    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(GameLaunchError, match="failed to start trainer: trainer not found"):
        launcher.run(["trainer.exe"], run_dir=tmp_path)
    instance.process.terminate.assert_called_once_with()
    assert instance._stdout.closed and instance._stderr.closed
    assert spawn.call_args.kwargs["stdout"].closed


@pytest.mark.parametrize("terminates_gracefully", [True, False])
def test_trainer_timeout_reaps_process_and_releases_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminates_gracefully: bool
) -> None:
    instance = _mock_instance(tmp_path)
    games = RunningGameSet(_config(), tmp_path, (instance,), tmp_path / "manifest.json")
    launcher = TrainingLauncher(_config(), project_root=tmp_path)
    monkeypatch.setattr(launcher.game_launcher, "start", Mock(return_value=games))
    trainer = Mock(spec=subprocess.Popen)
    timeout = subprocess.TimeoutExpired("trainer", 0.01)
    trainer.wait.side_effect = (
        [timeout, -15]
        if terminates_gracefully
        else [timeout, subprocess.TimeoutExpired("trainer", 1), -9]
    )
    spawn = Mock(return_value=trainer)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    with pytest.raises(
        GameLaunchError, match="trainer exceeded its configured timeout"
    ) as captured:
        launcher.run(["trainer.exe"], run_dir=tmp_path, timeout_seconds=0.01)
    assert captured.value.__cause__ is timeout
    trainer.terminate.assert_called_once_with()
    if terminates_gracefully:
        trainer.kill.assert_not_called()
        assert trainer.wait.call_count == 2
    else:
        trainer.kill.assert_called_once_with()
        assert trainer.wait.call_count == 3
    instance.process.terminate.assert_called_once_with()
    assert instance._stdout.closed and instance._stderr.closed
    assert spawn.call_args.kwargs["stdout"].closed


def test_invalid_trainer_timeout_never_starts_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = TrainingLauncher(_config(), project_root=tmp_path)
    start = Mock()
    monkeypatch.setattr(launcher.game_launcher, "start", start)
    with pytest.raises(ValueError, match=r"trainer\.timeout_seconds"):
        launcher.run(["trainer.exe"], run_dir=tmp_path, timeout_seconds=0)
    start.assert_not_called()


def test_trainer_environment_reserves_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = _mock_instance(tmp_path)
    games = RunningGameSet(_config(), tmp_path, (instance,), tmp_path / "manifest.json")
    launcher = TrainingLauncher(_config(), project_root=tmp_path)
    monkeypatch.setattr(launcher.game_launcher, "start", Mock(return_value=games))
    trainer = Mock(spec=subprocess.Popen)
    trainer.wait.return_value = 23
    spawn = Mock(return_value=trainer)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    result = launcher.run(
        ["trainer.exe"],
        run_dir=tmp_path,
        environment={"TRAINING_MODE": "evaluate", "GLR_GAME_INSTANCES_MANIFEST": "spoof"},
    )
    assert result.return_code == 23
    assert spawn.call_args.kwargs["env"]["TRAINING_MODE"] == "evaluate"
    assert spawn.call_args.kwargs["env"]["GLR_GAME_INSTANCES_MANIFEST"] == str(games.manifest_path)
    assert spawn.call_args.kwargs["shell"] is False
    instance.process.terminate.assert_called_once_with()


def test_command_line_reports_configuration_errors_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--project", str(tmp_path / "missing.json")]) == 2
    output = capsys.readouterr()
    assert not output.out
    payload = json.loads(output.err)
    assert payload["schema_version"] == "glr.launch.v1"
    assert payload["status"] == "error"
    assert "regular file" in payload["error"]


def test_command_line_reports_missing_trainer_argv_without_launching_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "project.json"
    path.write_text(json.dumps({"game": _config().to_mapping(), "trainer": {}}), encoding="utf-8")
    spawn = Mock()
    monkeypatch.setattr(subprocess, "Popen", spawn)

    assert main(["--project", str(path), "--json"]) == 2

    output = capsys.readouterr()
    assert not output.out
    payload = json.loads(output.err)
    assert payload["status"] == "error"
    assert "project.trainer has missing=['argv']" in payload["error"]
    spawn.assert_not_called()
