"""Declared per-role environment: declaration, resolution, reporting, and recording."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from game_learning_runtime.cli import main
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.project import load_project
from game_learning_runtime.role_environment import (
    RoleEnvironment,
    is_secret_name,
    parse_environment_table,
    resolve_environment,
)
from game_learning_runtime.run_store import TrainingStore


def _write_project(root: Path, **extra: Any) -> Path:
    (root / "bridge").mkdir()
    manifest = {
        "schema_version": "glr.project.v1",
        "environment_id": "example.adventure-v1",
        "environment_family": "action-rpg",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": ["python", "-c", "print('runtime')"]},
        "trainer": {"argv": ["python", "-c", "print('train')"]},
        "player": {"argv": ["python", "-c", "print('play')", "{bundle}"]},
        "researcher": None,
        "planner": None,
        "evaluator": None,
        "capture": None,
    }
    manifest.update(extra)
    config_path = root / "glr-project.json"
    config_path.write_text(json.dumps(manifest), encoding="utf-8")
    return config_path


# --- 1. declaration: one table for every role, with role-level overrides ---


def test_project_table_applies_to_every_role_and_the_role_table_wins(tmp_path: Path) -> None:
    config_path = _write_project(
        tmp_path,
        environment={"RENDER_DEVICE": "cpu", "DATASET_ROOT": "datasets/synthetic"},
        trainer={
            "argv": ["python", "-c", "print('train')"],
            "environment": {"RENDER_DEVICE": "cuda"},
        },
    )

    project = load_project(config_path)

    assert dict(project.environment) == {
        "RENDER_DEVICE": "cpu",
        "DATASET_ROOT": "datasets/synthetic",
    }
    assert dict(project.role_environments["trainer"]) == {"RENDER_DEVICE": "cuda"}
    # The role table wins key by key; keys it does not name stay project-wide.
    assert dict(project.declared_environment("trainer")) == {
        "RENDER_DEVICE": "cuda",
        "DATASET_ROOT": "datasets/synthetic",
    }
    assert dict(project.declared_environment("runtime")) == {
        "RENDER_DEVICE": "cpu",
        "DATASET_ROOT": "datasets/synthetic",
    }


def test_optional_roles_may_declare_their_own_table(tmp_path: Path) -> None:
    config_path = _write_project(
        tmp_path,
        researcher={
            "argv": ["python", "-c", "print('research')"],
            "environment": {"RESEARCH_DEPTH": "2"},
        },
    )

    project = load_project(config_path)

    assert project.researcher is not None
    assert dict(project.declared_environment("researcher")) == {"RESEARCH_DEPTH": "2"}
    # A role that declared nothing still receives the project table.
    assert dict(project.declared_environment("planner")) == {}


def test_a_role_that_names_no_role_declares_nothing(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path, environment={"RENDER_DEVICE": "cpu"})

    project = load_project(config_path)

    assert dict(project.declared_environment(None)) == {}


# --- 2. resolution: literals, interpolation, and failing closed ---


def test_literals_pass_through_and_references_resolve_from_the_process() -> None:
    declared = {"MODE": "synthetic", "DATASET_ROOT": "${SYNTHETIC_DATASET_ROOT}/v1"}

    resolved = resolve_environment(
        declared, environ={"SYNTHETIC_DATASET_ROOT": "/tmp/synthetic"}, role="trainer"
    )

    assert resolved.ready
    assert resolved.process_environment() == {
        "MODE": "synthetic",
        "DATASET_ROOT": "/tmp/synthetic/v1",
    }
    assert [variable.source for variable in resolved.variables] == ["interpolated", "literal"]


def test_a_reference_to_a_missing_variable_fails_closed_and_names_the_key() -> None:
    declared = {"DATASET_ROOT": "${SYNTHETIC_DATASET_ROOT}", "OTHER": "${ALSO_MISSING}"}

    resolved = resolve_environment(declared, environ={}, role="trainer")

    assert not resolved.ready
    assert [item.name for item in resolved.unresolved] == ["DATASET_ROOT", "OTHER"]
    assert resolved.unresolved[0].missing == ("SYNTHETIC_DATASET_ROOT",)
    assert "DATASET_ROOT" in resolved.refusal()
    assert "${SYNTHETIC_DATASET_ROOT}" in resolved.refusal()
    # A key that cannot be resolved is absent, never an empty string.
    assert resolved.process_environment() == {}


# --- 3. precedence: the real process environment outranks the declared table ---


def test_the_process_environment_outranks_a_declared_literal() -> None:
    declared = {"MODE": "from-manifest"}

    resolved = resolve_environment(declared, environ={"MODE": "from-operator"}, role="trainer")

    assert resolved.process_environment() == {"MODE": "from-operator"}
    assert resolved.variables[0].source == "process"


def test_the_process_environment_outranks_a_declared_reference() -> None:
    declared = {"MODE": "${MISSING_VARIABLE}"}

    resolved = resolve_environment(declared, environ={"MODE": "from-operator"}, role="trainer")

    assert resolved.ready
    assert resolved.process_environment() == {"MODE": "from-operator"}
    assert resolved.variables[0].source == "process"


def test_a_partial_reference_keeps_its_literal_surroundings() -> None:
    declared = {"ENDPOINT": "http://${SYNTHETIC_HOST}:8080/v1"}

    resolved = resolve_environment(declared, environ={"SYNTHETIC_HOST": "127.0.0.1"})

    assert resolved.process_environment() == {"ENDPOINT": "http://127.0.0.1:8080/v1"}


# --- 4. validation reuses the game.environment key shape and reserved names ---


def test_manifest_rejects_a_reserved_glr_key_before_any_process_starts(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path, environment={"GLR_RUN_ID": "forged"})

    with pytest.raises(ValueError, match="GLR_RUN_ID"):
        load_project(config_path)


def test_manifest_rejects_a_role_level_reserved_key(tmp_path: Path) -> None:
    config_path = _write_project(
        tmp_path,
        trainer={
            "argv": ["python", "-c", "print('train')"],
            "environment": {"GLR_MODEL_BUNDLE": "forged"},
        },
    )

    with pytest.raises(ValueError, match="GLR_MODEL_BUNDLE"):
        load_project(config_path)


def test_manifest_rejects_malformed_keys_and_references(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="keys must match"):
        parse_environment_table({"not a key": "value"}, path="project.environment")
    with pytest.raises(ValueError, match="malformed reference"):
        parse_environment_table({"ROOT": "${}"}, path="project.environment")
    with pytest.raises(ValueError, match="malformed reference"):
        parse_environment_table({"ROOT": "${BAD-NAME}"}, path="project.environment")


def test_manifest_rejects_non_string_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="printable string"):
        parse_environment_table({"ROOT": 1}, path="project.environment")


# --- 5. observability: doctor reports per role and records what a run received ---


def _doctor_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[int, dict[str, Any], str]:
    exit_code = main(["--project", str(tmp_path), "--json", "doctor"])
    text = capsys.readouterr().out
    return exit_code, dict(json.loads(text)["data"]), text


def test_doctor_lists_the_resolved_variables_of_every_configured_role(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHETIC_DATASET_ROOT", "/tmp/synthetic")
    _write_project(
        tmp_path,
        environment={"DATASET_ROOT": "${SYNTHETIC_DATASET_ROOT}/v1", "MODE": "synthetic"},
    )

    _, output, _ = _doctor_output(tmp_path, capsys)

    assert output["ready"] is True
    for role in ("runtime", "trainer", "player"):
        entry = next(item for item in output["roles"] if item["role"] == role)
        assert entry["environment"]["ready"] is True
        assert entry["environment"]["variables"] == [
            {
                "name": "DATASET_ROOT",
                "source": "interpolated",
                "secret": False,
                "value": "/tmp/synthetic/v1",
            },
            {"name": "MODE", "source": "literal", "secret": False, "value": "synthetic"},
        ]
    # An unconfigured role never runs, so it reports no environment at all.
    assert "environment" not in next(item for item in output["roles"] if item["role"] == "planner")


def test_doctor_reports_unresolved_variables_and_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SYNTHETIC_DATASET_ROOT", raising=False)
    _write_project(tmp_path, environment={"DATASET_ROOT": "${SYNTHETIC_DATASET_ROOT}"})

    exit_code, output, _ = _doctor_output(tmp_path, capsys)

    assert exit_code == 1
    assert output["ready"] is False
    entry = next(item for item in output["roles"] if item["role"] == "trainer")
    assert entry["environment"]["ready"] is False
    assert entry["environment"]["unresolved"] == [
        {"name": "DATASET_ROOT", "missing": ["SYNTHETIC_DATASET_ROOT"]}
    ]


def test_doctor_never_prints_a_secret_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHETIC_TOKEN", "synthetic-secret-value")
    _write_project(tmp_path, environment={"API_TOKEN": "${SYNTHETIC_TOKEN}"})

    _, output, text = _doctor_output(tmp_path, capsys)

    entry = next(item for item in output["roles"] if item["role"] == "trainer")
    assert entry["environment"]["variables"] == [
        {"name": "API_TOKEN", "source": "interpolated", "secret": True}
    ]
    assert "synthetic-secret-value" not in text


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("API_TOKEN", True),
        ("TRAIN_PASSWORD", True),
        ("DATASET_ROOT", False),
        ("MONKEY_MODE", False),
    ],
)
def test_secret_names_are_detected_lexically(name: str, secret: bool) -> None:
    assert is_secret_name(name) is secret


def test_a_role_run_records_its_resolved_non_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHETIC_HOST", "127.0.0.1")
    monkeypatch.setenv("SYNTHETIC_TOKEN", "synthetic-secret-value")
    _write_project(
        tmp_path,
        environment={
            "ENDPOINT": "http://${SYNTHETIC_HOST}:8080",
            "API_TOKEN": "${SYNTHETIC_TOKEN}",
        },
        runtime={
            "argv": ["python", "-c", "print('runtime')"],
            "environment": {"MODE": "synthetic"},
        },
    )

    assert main(["--project", str(tmp_path), "--json", "runtime", "start"]) == 0

    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = next(iter(store.list_runs(environment_id="example.adventure-v1")))
    recorded = run.metadata["role_environment"]
    assert recorded["role"] == "runtime"
    assert recorded["ready"] is True
    assert {item["name"] for item in recorded["variables"]} == {"ENDPOINT", "API_TOKEN", "MODE"}
    values = {item["name"]: item.get("value") for item in recorded["variables"]}
    assert values["ENDPOINT"] == "http://127.0.0.1:8080"
    assert values["MODE"] == "synthetic"
    # A secret is recorded as received, never as content.
    assert values["API_TOKEN"] is None
    assert "synthetic-secret-value" not in json.dumps(dict(run.metadata))


def test_an_unresolvable_variable_stops_the_run_before_any_role_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SYNTHETIC_DATASET_ROOT", raising=False)
    _write_project(
        tmp_path,
        trainer={
            "argv": ["python", "-c", "print('train')"],
            "environment": {"DATASET_ROOT": "${SYNTHETIC_DATASET_ROOT}"},
        },
    )
    started = tmp_path / "trainer.started"
    (tmp_path / "glr-project.json").write_text(
        json.dumps(
            {
                **json.loads((tmp_path / "glr-project.json").read_text(encoding="utf-8")),
                "trainer": {
                    "argv": [
                        "python",
                        "-c",
                        f"open({str(started)!r}, 'w').close()",
                    ],
                    "environment": {"DATASET_ROOT": "${SYNTHETIC_DATASET_ROOT}"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractViolation, match="DATASET_ROOT"):
        main(["--project", str(tmp_path), "--json", "train"])

    assert not started.exists()


# --- 6. a project that declares nothing behaves exactly as before ---


def test_a_project_without_an_environment_table_declares_nothing(tmp_path: Path) -> None:
    config_path = _write_project(tmp_path)

    project = load_project(config_path)

    assert dict(project.environment) == {}
    assert dict(project.role_environments) == {}
    assert dict(project.declared_environment("trainer")) == {}
    assert resolve_environment(project.declared_environment("trainer"), environ={}).ready
    assert RoleEnvironment(role="trainer", variables=(), unresolved=()).process_environment() == {}


def test_a_run_without_a_declared_environment_records_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_project(tmp_path)

    _, output, _ = _doctor_output(tmp_path, capsys)

    entry = next(item for item in output["roles"] if item["role"] == "trainer")
    assert entry["environment"]["variables"] == []
    assert entry["environment"]["unresolved"] == []


def test_the_declared_environment_never_touches_the_glr_namespace(tmp_path: Path) -> None:
    """The CLI owns GLR_*: a declared value is not one of them, and is preserved."""

    resolved = resolve_environment(
        {"RENDER_DEVICE": "cpu"}, environ=dict(os.environ), role="trainer"
    )

    assert all(not name.startswith("GLR_") for name in resolved.process_environment())
