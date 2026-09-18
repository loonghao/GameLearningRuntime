"""Enforce that every repository automation tool is registered and domain-owned.

Run with ``just layout-check``. The check is deliberately mechanical so it can
block a pull request: a new ``.py`` file under ``tools/`` without a matching
``tools/registry.toml`` entry fails, an entry pointing at a deleted file fails,
and a tool whose directory disagrees with its declared domain fails.

Requires Python 3.11 or newer for the standard library ``tomllib``; the pinned
project environment satisfies this. The Python 3.10 CI matrix lane runs
``just ci-core`` and does not invoke this check.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

TOOLS_ROOT = "tools"
REGISTRY_NAME = "registry.toml"
LEGACY_SCRIPTS_DIR = "scripts"
SKIPPED_FILES = frozenset({"__init__.py", "conftest.py"})


def repository_root() -> Path:
    """Resolve the repository root from this file's location."""

    return Path(__file__).resolve().parent.parent.parent


def discover_tools(root: Path) -> tuple[str, ...]:
    """Return every tool path, POSIX-relative to the repository root."""

    tools_dir = root / TOOLS_ROOT
    if not tools_dir.is_dir():
        return ()
    discovered = [
        path.relative_to(root).as_posix()
        for path in sorted(tools_dir.rglob("*.py"))
        if path.name not in SKIPPED_FILES
    ]
    return tuple(discovered)


def load_registry(root: Path) -> list[dict[str, object]]:
    registry_path = root / TOOLS_ROOT / REGISTRY_NAME
    if not registry_path.is_file():
        raise SystemExit(f"missing tool registry: {registry_path.as_posix()}")
    with registry_path.open("rb") as handle:
        payload = tomllib.load(handle)
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise SystemExit("tool registry 'entries' must be an array of tables")
    return [entry for entry in entries if isinstance(entry, dict)]


def domain_of(path: str) -> str:
    """Return the capability domain implied by a tool path."""

    parts = path.split("/")
    if len(parts) < 3 or parts[0] != TOOLS_ROOT:
        return ""
    return parts[1]


def check(root: Path) -> list[str]:
    """Return a sorted list of human-readable registry violations."""

    problems: list[str] = []
    discovered = set(discover_tools(root))
    entries = load_registry(root)

    legacy = root / LEGACY_SCRIPTS_DIR
    if legacy.is_dir():
        stray = sorted(p.name for p in legacy.glob("*.py"))
        if stray:
            problems.append(
                f"{LEGACY_SCRIPTS_DIR}/ must stay empty; move {', '.join(stray)} "
                "into a tools/<domain>/ module and register it"
            )

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for entry in entries:
        identifier = entry.get("id")
        path = entry.get("path")
        domain = entry.get("domain")
        if not isinstance(identifier, str) or not identifier:
            problems.append("registry entry has a missing or non-string id")
            continue
        if identifier in seen_ids:
            problems.append(f"duplicate registry id: {identifier}")
        seen_ids.add(identifier)

        if not isinstance(path, str) or not path:
            problems.append(f"registry entry {identifier} has a missing or non-string path")
            continue
        if path in seen_paths:
            problems.append(f"duplicate registry path: {path}")
        seen_paths.add(path)
        if not (root / path).is_file():
            problems.append(f"registry entry {identifier} points at a missing file: {path}")
            continue
        if path not in discovered:
            problems.append(f"registry entry {identifier} path is outside tools/: {path}")
            continue

        if not isinstance(domain, str) or not domain:
            problems.append(f"registry entry {identifier} has a missing or non-string domain")
        else:
            actual = domain_of(path)
            if actual != domain:
                problems.append(
                    f"registry entry {identifier} declares domain '{domain}' but lives in "
                    f"'{actual or path}'"
                )
        for field in ("purpose", "entrypoints"):
            if field not in entry:
                problems.append(f"registry entry {identifier} is missing required '{field}'")

    for path in sorted(discovered - seen_paths):
        problems.append(
            f"unregistered tool: {path}. Add a [[entries]] block to "
            f"{TOOLS_ROOT}/{REGISTRY_NAME} with its domain, purpose and entry points"
        )
    return sorted(set(problems))


def main() -> int:
    root = repository_root()
    problems = check(root)
    if problems:
        print(f"layout-check failed with {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"layout-check ok: {len(load_registry(root))} registered tools")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
