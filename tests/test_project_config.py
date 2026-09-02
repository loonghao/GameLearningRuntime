from __future__ import annotations

import json
from pathlib import Path

import pytest

from game_learning_runtime.project import ProjectCommand, find_project, load_project


def _project_value() -> dict[str, object]:
    return {
        "schema_version": "glr.project.v1",
        "environment_id": "example.adventure-v1",
        "environment_family": "action-rpg",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": ["python", "runtime.py"]},
        "trainer": {"argv": ["python", "train.py"]},
        "player": {"argv": ["python", "play.py", "{bundle}"]},
        "researcher": None,
        "planner": None,
        "evaluator": None,
        "capture": None,
    }


def test_project_config_resolves_project_owned_bridge_and_commands(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    config_path = tmp_path / "glr-project.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "glr.project.v1",
                "environment_id": "example.adventure-v1",
                "environment_family": "action-rpg",
                "protocol_version": "1.0",
                "data_dir": ".glr",
                "bridge_path": "bridge",
                "runtime": {"argv": ["vx", "run", "runtime", "{bridge_path}"]},
                "trainer": {"argv": ["vx", "run", "train", "{run_dir}"]},
                "player": {"argv": ["vx", "run", "play", "{bundle}"]},
                "researcher": None,
                "planner": None,
                "evaluator": None,
                "capture": {
                    "argv": ["ffmpeg", "{capture_video}"],
                    "required": True,
                    "stop": "stdin-q",
                    "video_file": "capture.mp4",
                    "index_file": "capture-index.jsonl",
                    "codec": "h264",
                    "frame_rate": 30,
                    "width": 640,
                    "height": 360,
                    "session": {
                        "startup_timeout_seconds": 2,
                        "heartbeat_timeout_seconds": 3,
                        "minimum_frames": 2,
                        "minimum_steps": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    project = load_project(config_path)

    assert project.root == tmp_path
    assert project.bridge_path == bridge
    assert project.data_dir == tmp_path / ".glr"
    assert project.runtime.expand(bridge_path=bridge) == (
        "vx",
        "run",
        "runtime",
        str(bridge),
    )
    assert project.environment_id == "example.adventure-v1"
    assert project.environment_family == "action-rpg"
    assert project.protocol_version == "1.0"
    assert project.capture is not None
    assert project.capture.video_file == "capture.mp4"
    assert project.capture.session is not None
    assert project.capture.session.minimum_frames == 2


def test_project_discovers_parent_and_loads_all_optional_roles(tmp_path: Path) -> None:
    (tmp_path / "bridge").mkdir()
    nested = tmp_path / "nested/deeper"
    nested.mkdir(parents=True)
    value = _project_value()
    value.update(
        researcher={"argv": ["python", "research.py", "{research_path}"]},
        planner={"argv": ["python", "plan.py", "{trial_path}"]},
        evaluator={"argv": ["python", "evaluate.py", "{evaluation_path}"]},
    )
    config = tmp_path / "glr-project.json"
    config.write_text(json.dumps(value), encoding="utf-8")

    assert find_project(nested) == config
    project = load_project(nested / "missing.txt")
    assert project.researcher is not None
    assert project.planner is not None
    assert project.evaluator is not None


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "cannot be empty"),
        ([""], "non-empty"),
        (["python", "prefix-{run_id}"], "complete argv"),
        (["python", "{unknown}"], "unsupported"),
    ],
)
def test_project_command_rejects_unsafe_argv(argv: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProjectCommand(tuple(argv))
    with pytest.raises(TypeError, match="array"):
        ProjectCommand.from_mapping({"argv": "python"}, path="command")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", "wrong", "schema_version"),
        ("environment_id", "INVALID", "must match"),
        ("protocol_version", "", "non-empty"),
        ("bridge_path", "../outside", "project-relative"),
        ("data_dir", "C:/outside", "project-relative"),
        ("runtime", {"argv": []}, "cannot be empty"),
    ],
)
def test_project_rejects_invalid_identity_and_paths(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    (tmp_path / "bridge").mkdir()
    value = _project_value()
    value[field] = replacement
    config = tmp_path / "glr-project.json"
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((FileNotFoundError, TypeError, ValueError), match=message):
        load_project(config)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("required", "yes", "boolean"),
        ("stop", "kill", "stdin-q"),
        ("frame_rate", 0, "positive"),
        ("width", 0, "positive integers"),
        ("codec", "INVALID", "must match"),
        ("argv", "ffmpeg", "array"),
        ("video_file", "../capture.mp4", "project-relative"),
    ],
)
def test_project_rejects_invalid_capture_contract(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    (tmp_path / "bridge").mkdir()
    value = _project_value()
    capture: dict[str, object] = {
        "argv": ["recorder", "{capture_video}"],
        "required": True,
        "stop": "stdin-q",
        "video_file": "capture.mp4",
        "index_file": "capture.jsonl",
        "codec": "h264",
        "frame_rate": 12,
        "width": 640,
        "height": 360,
    }
    capture[field] = replacement
    value["capture"] = capture
    config = tmp_path / "glr-project.json"
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        load_project(config)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("status_file", "../status.jsonl", "project-relative"),
        ("startup_timeout_seconds", 0, "between 0.1 and 60"),
        ("heartbeat_timeout_seconds", 61, "between 0.1 and 60"),
        ("minimum_frames", 0, "positive integers"),
    ],
)
def test_project_rejects_invalid_capture_session(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    (tmp_path / "bridge").mkdir()
    value = _project_value()
    value["capture"] = {
        "argv": ["recorder", "{capture_video}"],
        "required": True,
        "stop": "stdin-q",
        "video_file": "capture.mp4",
        "index_file": "capture.jsonl",
        "codec": "h264",
        "frame_rate": 12,
        "width": 640,
        "height": 360,
        "session": {field: replacement},
    }
    config = tmp_path / "glr-project.json"
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        load_project(config)
