from __future__ import annotations

import json
from pathlib import Path

import pytest

from game_learning_runtime.project import load_project
from game_learning_runtime.run_context import load_run_context


def _project(tmp_path: Path) -> Path:
    (tmp_path / "bridge").mkdir()
    (tmp_path / "config" / "contexts").mkdir(parents=True)
    (tmp_path / "config" / "training.json").write_text(
        '{"schema_version":"glr.training.v1","algorithm":"ppo"}',
        encoding="utf-8",
    )
    (tmp_path / "config" / "contexts" / "native.toml").write_bytes(
        b"""schema_version = "glr.run-context.v1"
context_id = "league-native-100024"
environment_id = "example.context-v1"
protocol_version = "1.0"

[labels]
season = "ranked-2026"
ruleset = "standard"

[[inputs]]
owner = "training"
path = "config/training.json"
schema_version = "glr.training.v1"
""",
    )
    project = {
        "schema_version": "glr.project.v1",
        "environment_id": "example.context-v1",
        "environment_family": "test",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": ["glr", "--version"]},
        "trainer": {"argv": ["glr", "--version"]},
        "player": {"argv": ["glr", "--version"]},
    }
    manifest = tmp_path / "glr-project.json"
    manifest.write_text(json.dumps(project), encoding="utf-8")
    return manifest


def test_python_role_verifies_the_cli_frozen_context(tmp_path: Path) -> None:
    project = load_project(_project(tmp_path))
    context = load_run_context(project, "config/contexts/native.toml")
    assert context.context_sha256 == (
        "8a5f395b6a8bdedf913ee9063c468a69e8027eb2f27b067c1862878272582140"
    )
    inherited = load_run_context(
        project,
        environment={
            "GLR_RUN_CONTEXT": context.to_json(),
            "GLR_RUN_CONTEXT_SHA256": context.context_sha256,
        },
    )
    assert inherited == context
    assert inherited.inputs[0].owner == "training"

    (tmp_path / "config" / "training.json").write_text(
        json.dumps({"schema_version": "glr.training.v1", "algorithm": "changed"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed"):
        load_run_context(
            project,
            environment={
                "GLR_RUN_CONTEXT": context.to_json(),
                "GLR_RUN_CONTEXT_SHA256": context.context_sha256,
            },
        )
