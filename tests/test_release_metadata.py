from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


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

    assert package["release-type"] == "python"
    assert package["package-name"] == "game-learning-runtime"
    assert package["changelog-path"] == "CHANGELOG.md"
    assert ("toml", "pyproject.toml") in extra_files
    assert ("generic", "README.md") in extra_files
    assert ("generic", "README.zh-CN.md") in extra_files


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
