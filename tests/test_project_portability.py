from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from game_learning_runtime.project import find_project, load_project, resolve_game_directory

MANIFEST = """schema_version = "glr.project.v1"
environment_id = "example.environment-v1"
environment_family = "example-family"
protocol_version = "1.0"
data_dir = ".glr"
bridge_path = "bridge"
[runtime]
argv = ["python", "runtime.py", "{project_manifest}"]
[trainer]
argv = ["python", "train.py"]
[player]
argv = ["python", "play.py"]
"""


def _manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "bridge").mkdir()
    path = root / "glr-project.toml"
    path.write_text(MANIFEST, encoding="utf-8")
    return path


def test_toml_nearest_root_and_optional_roles(tmp_path: Path) -> None:
    outer = _manifest(tmp_path)
    inner = _manifest(tmp_path / "nested")
    child = inner.parent / "tools"
    child.mkdir()
    project = load_project(child)
    assert find_project(child) == inner
    assert project.root == inner.parent
    assert project.manifest_path == inner
    assert project.data_dir == inner.parent / ".glr"
    assert project.researcher is None
    assert project.capture is None
    assert project.extensions == {}
    assert find_project(outer) == outer


def test_dual_manifests_fail_even_for_explicit_file(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    (tmp_path / "glr-project.json").write_text("{}", encoding="utf-8")
    for start in (tmp_path, path, tmp_path / "glr-project.json"):
        with pytest.raises(ValueError, match="multiple project manifests"):
            load_project(start)


def test_extension_mount_is_root_relative_and_strict(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    config = tmp_path / "config/runtime.toml"
    config.parent.mkdir()
    config.write_text('[game]\ndirectory = "game"\n', encoding="utf-8")
    path.write_text(MANIFEST + '\n[extensions.example]\nconfig = "config/runtime.toml"\n')
    project = load_project(path)
    assert project.extensions == {"example": config}
    with pytest.raises(TypeError):
        project.extensions["other"] = config
    path.write_text(MANIFEST + '\n[extensions.example]\nconfig = "../outside.toml"\n')
    with pytest.raises(ValueError, match="project-relative"):
        load_project(path)
    path.write_text(
        MANIFEST + '\n[extensions.example]\nconfig = "config/runtime.toml"\nextra = 1\n'
    )
    with pytest.raises(ValueError, match="unexpected"):
        load_project(path)


@pytest.mark.parametrize(
    "extra", ["\nunknown = 1\n", '\n[extensions.INVALID]\nconfig = "missing.toml"\n']
)
def test_toml_rejects_unknown_fields(tmp_path: Path, extra: str) -> None:
    path = _manifest(tmp_path)
    path.write_text(MANIFEST + extra)
    with pytest.raises(ValueError):
        load_project(path)


def test_game_directory_accepts_owned_or_explicit_external(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    internal = root / "game"
    internal.mkdir()
    external = tmp_path / "external-runtime"
    external.mkdir()
    assert resolve_game_directory(root, "game") == internal
    assert resolve_game_directory(root, str(external)) == external
    for invalid in ("../external-runtime", "", "missing", "game\n"):
        with pytest.raises((ValueError, FileNotFoundError)):
            resolve_game_directory(root, invalid)


def test_legacy_json_subdirectory_still_works(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    value = tomllib.loads(path.read_text())
    path.unlink()
    (tmp_path / "glr-project.json").write_text(json.dumps(value))
    nested = tmp_path / "src/deep"
    nested.mkdir(parents=True)
    assert load_project(nested).root == tmp_path


def test_roles_receive_selected_manifest_not_an_inherited_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from game_learning_runtime.cli import _command_context, _process_environment

    manifest = _manifest(tmp_path)
    project = load_project(tmp_path)
    monkeypatch.setenv("GLR_PROJECT_MANIFEST", "unrelated-project")
    context = _command_context(project, run_id="synthetic", run_dir=tmp_path / ".glr/run")
    environment = _process_environment(project, run_id="synthetic", run_dir=tmp_path / ".glr/run")
    assert context["project_manifest"] == manifest
    assert environment["GLR_PROJECT_MANIFEST"] == str(manifest)
    assert environment["GLR_PROJECT_ROOT"] == str(tmp_path)


def test_manifest_symlink_cannot_fall_back_to_parent(tmp_path: Path) -> None:
    _manifest(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    try:
        (nested / "glr-project.toml").symlink_to(tmp_path / "glr-project.toml")
    except OSError:
        pytest.skip("symlink privilege is unavailable")
    with pytest.raises(ValueError, match="non-symlink"):
        find_project(nested)
