"""Enforce that every repository automation tool is registered and domain-owned.

Run with ``just layout-check``. The check is deliberately mechanical so it can
block a pull request: a new executable under ``tools/`` without a matching
``tools/registry.toml`` entry fails, an entry pointing at a deleted file fails,
an identifier or domain that does not follow the contract fails, and an entry
point that names a ``just`` recipe or workflow which does not exist fails. The
point is that "where does automation go" is a checked property of the checkout,
not a convention someone has to remember.

``tomllib`` is 3.11+; the 3.10 CI lane falls back to the declared ``tomli``
dependency so this gate can run on every supported interpreter.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found,unused-ignore]

TOOLS_ROOT = "tools"
REGISTRY_NAME = "registry.toml"
LEGACY_SCRIPTS_DIR = "scripts"
SKIPPED_FILES = frozenset({"__init__.py", "conftest.py"})

#: Executable suffixes that count as a tool. A shell or PowerShell helper under
#: ``tools/`` is automation too, so it has to be registered like a Python one.
TOOL_SUFFIXES = (".py", ".sh", ".ps1")

#: Capability domains a tool may declare. The list is explicit on purpose: a new
#: domain is a design decision, so it takes an edit here plus a directory.
ALLOWED_DOMAINS = frozenset(
    {"ci", "demo", "docs", "governance", "packaging", "providers", "release"}
)

#: Top-level directories that may legitimately contain Python sources.
ALLOWED_PY_ROOTS = frozenset(
    {
        "benchmarks",
        "crates",
        "dashboard-ui",
        "docs",
        "sdk",
        "src",
        "tests",
        "tests_optional",
        "tools",
    }
)

#: Skill payloads ship their own ``scripts/`` directories; they are products, not
#: repository automation, so the dumping-ground rule does not apply to them.
SKILL_ROOTS = (".agents", "plugins")

#: Directories never worth walking.
SCAN_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".venv-glr",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
    }
)

JUSTFILE_NAME = "justfile"
WORKFLOW_DIR = ".github/workflows"

#: Entry point for a tool that is run by hand; the only value allowed besides a
#: real ``just`` recipe or workflow.
MANUAL_ENTRYPOINT = "manual"

_ID_SEGMENT = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
#: Recipe headers look like ``name:``, ``name param:`` or ``name: dep``. The
#: ``:(?!=)`` guard keeps ``set shell := [...]`` settings out of the result.
_RECIPE_PATTERN = re.compile(r"^@?(?P<name>[A-Za-z_][A-Za-z0-9_-]*)[^:\n]*:(?!=)", re.MULTILINE)


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
        for path in sorted(tools_dir.rglob("*"))
        if path.is_file()
        and path.suffix in TOOL_SUFFIXES
        and path.name not in SKIPPED_FILES
        and not (set(path.relative_to(tools_dir).parts) & SCAN_EXCLUDED_PARTS)
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


def load_just_recipes(root: Path) -> frozenset[str]:
    """Return the recipe names declared by the repository ``justfile``."""

    justfile = root / JUSTFILE_NAME
    if not justfile.is_file():
        return frozenset()
    return frozenset(
        match.group("name")
        for match in _RECIPE_PATTERN.finditer(justfile.read_text(encoding="utf-8"))
    )


def domain_of(path: str) -> str:
    """Return the capability domain implied by a tool path."""

    parts = path.split("/")
    if len(parts) < 3 or parts[0] != TOOLS_ROOT:
        return ""
    return parts[1]


def _identifier_problems(identifier: str, domain: str) -> list[str]:
    """Return violations of the ``<domain>.<kebab-name>`` identifier contract."""

    problems: list[str] = []
    if "." not in identifier:
        problems.append(f"registry id '{identifier}' must look like '<domain>.<kebab-name>'")
        return problems
    prefix, _, name = identifier.partition(".")
    if not _ID_SEGMENT.match(prefix) or not _ID_SEGMENT.match(name):
        problems.append(f"registry id '{identifier}' must use lower-case kebab-case segments")
    elif domain and prefix != domain:
        problems.append(f"registry id '{identifier}' must start with its domain '{domain}'")
    return problems


def _entrypoint_problems(
    root: Path, identifier: str, path: str, entrypoints: object, recipes: frozenset[str]
) -> list[str]:
    """Return violations of the declared entry points for one registry entry."""

    if not isinstance(entrypoints, list) or not entrypoints:
        return [f"registry entry {identifier} needs a non-empty 'entrypoints' list"]
    problems: list[str] = []
    if not all(isinstance(item, str) and item for item in entrypoints):
        return [f"registry entry {identifier} has non-string entrypoints"]
    for entrypoint in entrypoints:
        if entrypoint == MANUAL_ENTRYPOINT:
            continue
        if entrypoint.startswith("just "):
            recipe = entrypoint.split()[1]
            if recipe not in recipes:
                problems.append(f"registry entry {identifier} names unknown just recipe '{recipe}'")
            continue
        if entrypoint.startswith(WORKFLOW_DIR) and entrypoint.endswith((".yml", ".yaml")):
            workflow = root / entrypoint
            if not workflow.is_file():
                problems.append(
                    f"registry entry {identifier} names a missing workflow: {entrypoint}"
                )
            elif path not in workflow.read_text(encoding="utf-8"):
                problems.append(
                    f"registry entry {identifier} names {entrypoint}, which never invokes {path}"
                )
            continue
        problems.append(
            f"registry entry {identifier} has an unrecognized entrypoint '{entrypoint}'; "
            "expected 'just <recipe>', '.github/workflows/<file>.yml' or 'manual'"
        )
    return problems


def _stray_python_problems(root: Path) -> list[str]:
    """Return Python sources that escaped the module layout."""

    problems: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        parts = relative.parts
        if set(parts) & SCAN_EXCLUDED_PARTS:
            continue
        if any(part.startswith(".") and part != ".github" for part in parts):
            continue
        if parts[0] in SKILL_ROOTS:
            continue
        if parts[0] not in ALLOWED_PY_ROOTS:
            problems.append(
                f"python source outside the module layout: {relative.as_posix()}. "
                f"Automation belongs in {TOOLS_ROOT}/<domain>/ and library code in src/"
            )
            continue
        if LEGACY_SCRIPTS_DIR in parts:
            problems.append(
                f"{LEGACY_SCRIPTS_DIR}/ must stay empty; move {relative.as_posix()} "
                "into a tools/<domain>/ module and register it"
            )
    return problems


def check(root: Path) -> list[str]:
    """Return a sorted list of human-readable registry violations."""

    problems: list[str] = []
    discovered = set(discover_tools(root))
    entries = load_registry(root)
    recipes = load_just_recipes(root)

    legacy = root / LEGACY_SCRIPTS_DIR
    if legacy.is_dir():
        stray = sorted(p.name for p in legacy.glob("*.py"))
        if stray:
            problems.append(
                f"{LEGACY_SCRIPTS_DIR}/ must stay empty; move {', '.join(stray)} "
                "into a tools/<domain>/ module and register it"
            )

    problems.extend(_stray_python_problems(root))

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
        elif domain not in ALLOWED_DOMAINS:
            problems.append(
                f"registry entry {identifier} declares unknown domain '{domain}'; "
                f"allowed: {', '.join(sorted(ALLOWED_DOMAINS))}"
            )
        else:
            actual = domain_of(path)
            if actual != domain:
                problems.append(
                    f"registry entry {identifier} declares domain '{domain}' but lives in "
                    f"'{actual or path}'"
                )
            problems.extend(_identifier_problems(identifier, domain))

        purpose = entry.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            problems.append(f"registry entry {identifier} needs a non-empty 'purpose'")
        problems.extend(
            _entrypoint_problems(root, identifier, path, entry.get("entrypoints"), recipes)
        )

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
