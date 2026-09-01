"""Synchronize the repository skills into the distributable agent plugin.

The plugin is intentionally checked into the repository so compatible agents
can consume one immutable tree.  ``--check`` is safe for CI and reports drift;
``--sync`` is an explicit authoring action that copies the source skills into
the plugin payload.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / ".agents" / "skills"
DEFAULT_PLUGIN = ROOT / "plugins" / "game-learning-runtime-skills"
IGNORED_NAMES = {"__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _iter_files(root: Path) -> set[Path]:
    """Return normalized relative file paths, excluding interpreter caches."""

    files: set[Path] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_NAMES for part in path.parts):
            continue
        if path.suffix in IGNORED_SUFFIXES:
            continue
        files.add(path.relative_to(root))
    return files


def _copy_tree(source: Path, destination: Path) -> None:
    """Copy a skill tree while omitting generated Python caches."""

    for relative in sorted(_iter_files(source)):
        source_path = source / relative
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _skill_sources(source_root: Path) -> list[Path]:
    skills = [
        path
        for path in source_root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name not in IGNORED_NAMES
    ]
    return sorted(skills, key=lambda path: path.name)


def _validate_manifest(plugin_root: Path) -> list[str]:
    manifest_path = plugin_root / ".codex-plugin" / "plugin.json"
    if not manifest_path.is_file():
        return [f"missing plugin manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"invalid plugin manifest {manifest_path}: {error}"]
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["plugin manifest must contain a JSON object"]
    if manifest.get("name") != plugin_root.name:
        errors.append("plugin manifest name must match its directory name")
    if manifest.get("skills") != "./skills/":
        errors.append("plugin manifest skills must resolve to ./skills/")
    for field in ("version", "description", "author", "interface"):
        if not manifest.get(field):
            errors.append(f"plugin manifest is missing {field}")
    return errors


def _compare(source: Path, packaged: Path) -> list[str]:
    errors: list[str] = []
    source_files = _iter_files(source)
    packaged_files = _iter_files(packaged)
    for relative in sorted(source_files - packaged_files):
        errors.append(f"missing packaged file: {relative.as_posix()}")
    for relative in sorted(packaged_files - source_files):
        errors.append(f"stale packaged file: {relative.as_posix()}")
    for relative in sorted(source_files & packaged_files):
        if not filecmp.cmp(source / relative, packaged / relative, shallow=False):
            errors.append(f"packaged file differs: {relative.as_posix()}")
    return errors


def _sync(source_root: Path, plugin_root: Path) -> None:
    skills_root = plugin_root / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    for skill in _skill_sources(source_root):
        destination = skills_root / skill.name
        _copy_tree(skill, destination)

    # Remove files that were previously packaged but are no longer source files.
    # This is limited to the generated skills payload; the manifest remains owned
    # by the plugin author and is never deleted by this command.
    source_files = {
        Path("skills") / skill.name / relative
        for skill in _skill_sources(source_root)
        for relative in _iter_files(skill)
    }
    for relative in sorted(_iter_files(skills_root)):
        packaged_relative = Path("skills") / relative
        if packaged_relative not in source_files:
            (skills_root / relative).unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail when source and package drift")
    mode.add_argument("--sync", action="store_true", help="copy source skills into the plugin")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--plugin", type=Path, default=DEFAULT_PLUGIN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    plugin_root = args.plugin.expanduser().resolve()
    if not source_root.is_dir():
        print(f"source skills directory does not exist: {source_root}", file=sys.stderr)
        return 2
    if not plugin_root.is_dir():
        print(f"plugin directory does not exist: {plugin_root}", file=sys.stderr)
        return 2

    errors = _validate_manifest(plugin_root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    if args.sync:
        _sync(source_root, plugin_root)

    errors = _compare(source_root, plugin_root / "skills")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Agent plugin is synchronized: {plugin_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
