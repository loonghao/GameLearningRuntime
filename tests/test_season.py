from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from game_learning_runtime.project import load_project
from game_learning_runtime.season import (
    ENVIRONMENT_KEYS,
    initialize_season,
    list_seasons,
    load_season_context,
    require_selection,
    season_path,
    select_season,
)

FIXTURE = Path(__file__).parent / "fixtures/season_project"

MANIFEST = """schema_version = "glr.project.v1"
environment_id = "example.environment-v1"
environment_family = "example-family"
protocol_version = "1.0"
data_dir = ".glr"
bridge_path = "bridge"
[runtime]
argv = ["python", "runtime.py"]
[trainer]
argv = ["python", "train.py"]
[player]
argv = ["python", "play.py"]
[seasons]
config = "config/seasons.toml"
"""


def project_fixture(root: Path) -> None:
    (root / "bridge").mkdir()
    (root / "config").mkdir()
    (root / "glr-project.toml").write_text(MANIFEST, encoding="utf-8")


def test_pending_init_and_selection_are_declarative(tmp_path: Path) -> None:
    project_fixture(tmp_path)
    project = load_project(tmp_path)
    result = initialize_season(project, "example-season", "standard")
    assert result.status == "pending"
    assert select_season(project, "example-season", "standard") == result
    with pytest.raises(ValueError, match="pending"):
        result.require_ready()
    with pytest.raises(FileExistsError):
        initialize_season(project, "example-season", "standard")
    with pytest.raises(ValueError, match="unknown"):
        select_season(project, "other", "standard")
    assert not (tmp_path / ".glr").exists()


def test_context_freezes_references_and_rejects_edits(tmp_path: Path) -> None:
    project_fixture(tmp_path)
    project = load_project(tmp_path)
    context = initialize_season(project, "example-season", "standard")
    declaration = tmp_path / context.declaration.path
    declaration.write_text(declaration.read_text().replace('"pending"', '"ready"'))
    selected = select_season(project, "example-season", "standard")
    assert selected is not None
    selected.require_ready()
    selected.verify(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        context.verify(tmp_path)
    mapping = json.loads(selected.to_json())
    assert mapping["declaration"]["path"].endswith("standard.toml")
    assert len(mapping["context_sha256"]) == 64


@pytest.mark.parametrize("season,ruleset", [("other", None), (None, "standard")])
def test_partial_selection_fails(tmp_path: Path, season: str | None, ruleset: str | None) -> None:
    project_fixture(tmp_path)
    with pytest.raises(ValueError, match="together"):
        select_season(load_project(tmp_path), season, ruleset)


def test_shared_wire_fixture_matches_python_and_survives_relocation(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path / "clone")
    for root in (FIXTURE, tmp_path / "clone"):
        project = load_project(root / "config/seasons")
        context = select_season(project, "example-season", "standard")
        assert context is not None
        assert context.to_mapping() == json.loads((FIXTURE / "context.json").read_text())
        assert list_seasons(project) == [context.to_mapping()]
        assert load_season_context(project, context.environment(root)) == context


@pytest.mark.parametrize("key", ENVIRONMENT_KEYS)
def test_incomplete_or_forged_environment_rejected(key: str) -> None:
    project = load_project(FIXTURE)
    context = select_season(project, "example-season", "standard")
    assert context is not None
    environment = context.environment(FIXTURE)
    environment.pop(key)
    with pytest.raises(ValueError, match="incomplete"):
        load_season_context(project, environment)
    environment = context.environment(FIXTURE)
    environment[key] = "unknown" if key != "GLR_SEASON_CONTEXT" else "{}"
    with pytest.raises(ValueError):
        load_season_context(project, environment)


def test_duplicate_json_and_stale_inputs_rejected(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    project = load_project(tmp_path)
    context = select_season(project, "example-season", "standard")
    assert context is not None
    environment = context.environment(tmp_path)
    environment["GLR_SEASON_CONTEXT"] = context.to_json().replace(
        '"status":"pending"', '"status":"pending","status":"pending"'
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_season_context(project, environment)
    with pytest.raises(TypeError):
        context.extensions["other"] = context.declaration
    (tmp_path / "config/preset.toml").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        context.environment(tmp_path)


@pytest.mark.parametrize(
    "relative", ["", "../file", "/absolute", "a//b", "a/./b", "a\\b", "a:b", "a\nb", "x" * 513]
)
def test_paths_fail_closed(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError):
        season_path(tmp_path, relative)


@pytest.mark.parametrize(
    "replacement",
    [
        ('status = "pending"', 'status = "active"'),
        ('status = "pending"', 'status = "pending"\nunknown = 1'),
        ('ruleset_id = "standard"', 'ruleset_id = "other"'),
        ('config = "config/preset.toml"', 'config = "../outside.toml"'),
        ('config = "config/preset.toml"', 'config = "missing.toml"'),
        ("[extensions.preset]", "[extensions.INVALID]"),
    ],
)
def test_bad_declarations_are_not_ready(tmp_path: Path, replacement: tuple[str, str]) -> None:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "config/seasons/example-season/standard.toml"
    path.write_text(path.read_text().replace(*replacement))
    with pytest.raises(ValueError):
        select_season(load_project(tmp_path), "example-season", "standard")


def test_catalog_duplicates_and_oversized_inputs_fail(tmp_path: Path) -> None:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "config/seasons.toml"
    original = path.read_text()
    path.write_text(original + original[original.index("[[entries]]") :])
    with pytest.raises(ValueError, match="duplicate"):
        list_seasons(load_project(tmp_path))
    path.write_text(original)
    (tmp_path / "config/preset.toml").write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="limit"):
        list_seasons(load_project(tmp_path))


def test_init_rollback_keeps_unowned_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import game_learning_runtime.season as module

    project_fixture(tmp_path)
    sentinel = tmp_path / "config/seasons.toml.new"
    sentinel.write_text("user-owned")
    monkeypatch.setattr(
        module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("synthetic failure"))
    )
    with pytest.raises(OSError, match="synthetic"):
        initialize_season(load_project(tmp_path), "example-season", "standard")
    assert sentinel.read_text() == "user-owned"
    assert not (tmp_path / "config/seasons.toml.lock").exists()
    assert not (tmp_path / "config/seasons/example-season/standard.toml").exists()
    assert sorted(path.name for path in (tmp_path / "config").iterdir()) == [
        "seasons",
        "seasons.toml.new",
    ]


def test_symlink_component_rejected(tmp_path: Path) -> None:
    (tmp_path / "actual").mkdir()
    try:
        (tmp_path / "linked").symlink_to(tmp_path / "actual", target_is_directory=True)
    except OSError:
        pytest.skip("symlink privilege unavailable")
    with pytest.raises(ValueError, match="links"):
        season_path(tmp_path, "linked/config.toml")


def test_python_roles_consume_same_context_and_record_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from game_learning_runtime.cli import main
    from game_learning_runtime.run_store import TrainingStore

    project_fixture(tmp_path)
    script = (
        "import os; from pathlib import Path; "
        "from game_learning_runtime.project import load_project; "
        "p=load_project(); assert p.season_context is not None; "
        "Path(os.environ['GLR_RUN_DIR'], 'received.json').write_text(p.season_context.to_json())"
    )
    path = tmp_path / "glr-project.toml"
    path.write_text(
        MANIFEST.replace(
            '["python", "runtime.py"]', json.dumps([sys.executable, "-c", script])
        ).replace('["python", "train.py"]', json.dumps([sys.executable, "-c", script]))
    )
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "--season",
                "example-season",
                "--ruleset",
                "standard",
                "--json",
                "season",
                "init",
            ]
        )
        == 0
    )
    capsys.readouterr()
    prefix = [
        "--project",
        str(tmp_path),
        "--season",
        "example-season",
        "--ruleset",
        "standard",
        "--json",
    ]
    for args in (["--project", str(tmp_path), "train"], [*prefix, "train"]):
        with pytest.raises(ValueError):
            main(args)
    assert not (tmp_path / ".glr").exists()
    assert main([*prefix, "runtime", "start"]) == 0
    run = json.loads(capsys.readouterr().out)["data"]
    run_dir = tmp_path / ".glr/runs" / run["run_id"]
    assert json.loads((run_dir / "received.json").read_text()) == json.loads(
        (run_dir / "season-context.json").read_text()
    )
    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    assert any(event.kind == "season.selected" for event in store.list_events(run["run_id"]))
    assert any(item.role == "season-context" for item in store.list_artifacts(run["run_id"]))
    declaration = tmp_path / "config/seasons/example-season/standard.toml"
    declaration.write_text(declaration.read_text().replace('"pending"', '"ready"'))
    assert main([*prefix, "train", "--no-capture"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["status"] == "succeeded"


def test_legacy_and_readiness_gates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from game_learning_runtime.cli import main

    project_fixture(tmp_path)
    path = tmp_path / "glr-project.toml"
    path.write_text(MANIFEST.split("[seasons]")[0])
    project = load_project(tmp_path)
    assert select_season(project, None, None) is None
    assert list_seasons(project) == []
    require_selection(project, ready=True)
    with pytest.raises(ValueError, match="no"):
        select_season(project, "example-season", "standard")
    path.write_text(MANIFEST)
    context = initialize_season(load_project(tmp_path), "example-season", "standard")
    require_selection(replace(load_project(tmp_path), season_context=context), ready=False)
    assert main(["--project", str(tmp_path), "--json", "doctor"]) != 0
    result = json.loads(capsys.readouterr().out)["data"]
    assert result["installation_ready"] is True
    assert result["training_config_ready"] is False
    assert result["live_runtime_verified"] is False
    assert not (tmp_path / ".glr").exists()


def test_cli_requires_flags_even_with_inherited_role_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from game_learning_runtime.cli import main

    project_fixture(tmp_path)
    context = initialize_season(load_project(tmp_path), "example-season", "standard")
    for key, value in context.environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    # SDK roles consume the inherited receipt, but a new CLI invocation must select.
    assert load_project(tmp_path).season_context == context
    with pytest.raises(ValueError, match="explicit"):
        main(["--project", str(tmp_path), "runtime", "start"])
    assert not (tmp_path / ".glr").exists()
