from __future__ import annotations

import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).parents[1]


def test_python_distribution_does_not_publish_the_glr_cli() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert "scripts" not in project


def test_release_metadata_tracks_package_version() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text(encoding="utf-8"))
    version = re.search(r'^version = "(?P<version>\d+\.\d+\.\d+)"$', pyproject, re.MULTILINE)

    assert version is not None
    assert manifest["."] == version.group("version")


def test_release_please_owns_all_version_surfaces() -> None:
    config = json.loads((ROOT / "release-please-config.json").read_text(encoding="utf-8"))
    package = config["packages"]["."]
    extra_files = {(entry["type"], entry["path"]) for entry in package["extra-files"]}

    assert package["release-type"] == "rust"
    assert package["package-name"] == "game-learning-runtime"
    assert package["changelog-path"] == "CHANGELOG.md"
    assert ("toml", "pyproject.toml") in extra_files
    assert ("toml", "Cargo.toml") in extra_files
    assert ("generic", "README.md") in extra_files
    assert ("generic", "README.zh-CN.md") in extra_files
    assert ("generic", "uv.lock") in extra_files
    assert ("generic", "Cargo.lock") in extra_files


def test_uv_lock_version_matches_manifest_and_has_release_marker() -> None:
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text(encoding="utf-8"))
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    pattern = re.compile(
        r'^name = "game-learning-runtime"\n'
        r'version = "(?P<version>\d+\.\d+\.\d+)" # x-release-please-version$',
        re.MULTILINE,
    )

    match = pattern.search(lock)
    assert match is not None
    assert match.group("version") == manifest["."]


def test_rust_workspace_and_lock_versions_match_manifest() -> None:
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text(encoding="utf-8"))
    workspace = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    workspace_match = re.search(
        r'^version = "(?P<version>\d+\.\d+\.\d+)"$', workspace, re.MULTILINE
    )
    host_lock_match = re.search(
        r'^name = "glr-host"\n'
        r'version = "(?P<version>\d+\.\d+\.\d+)" # x-release-please-version$',
        lock,
        re.MULTILINE,
    )
    cli_lock_match = re.search(
        r'^name = "glr-cli"\n'
        r'version = "(?P<version>\d+\.\d+\.\d+)" # x-release-please-version$',
        lock,
        re.MULTILINE,
    )

    assert workspace_match is not None
    assert host_lock_match is not None
    assert cli_lock_match is not None
    assert workspace_match.group("version") == manifest["."]
    assert host_lock_match.group("version") == manifest["."]
    assert cli_lock_match.group("version") == manifest["."]


def test_bilingual_readme_release_pins_match_manifest() -> None:
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text(encoding="utf-8"))
    expected = manifest["."]
    pattern = re.compile(
        r"reusable-python-ci\.yml@v(?P<version>\d+\.\d+\.\d+)"
        r"\s+# x-release-please-version"
    )

    for readme_name in ("README.md", "README.zh-CN.md"):
        readme = (ROOT / readme_name).read_text(encoding="utf-8")
        match = pattern.search(readme)
        assert match is not None, f"missing release pin marker in {readme_name}"
        assert match.group("version") == expected
