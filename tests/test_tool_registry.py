from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC_PATH = ROOT / "tools" / "governance" / "check_tool_registry.py"
SPEC = importlib.util.spec_from_file_location("check_tool_registry", _SPEC_PATH)
assert SPEC is not None and SPEC.loader is not None
REGISTRY_CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGISTRY_CHECK)


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _registry(entry_id: str, path: str, domain: str) -> str:
    return (
        "[[entries]]\n"
        f'id = "{entry_id}"\n'
        f'path = "{path}"\n'
        f'domain = "{domain}"\n'
        'purpose = "test tool"\n'
        'entrypoints = ["manual"]\n'
    )


def test_repository_tool_registry_is_complete() -> None:
    assert REGISTRY_CHECK.check(REGISTRY_CHECK.repository_root()) == []


def test_repository_root_points_at_the_checkout() -> None:
    assert (REGISTRY_CHECK.repository_root() / "pyproject.toml").is_file()


def test_valid_repo_passes(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    _write(
        tmp_path,
        "tools/registry.toml",
        _registry("demo.example", "tools/demo/example.py", "demo"),
    )
    assert REGISTRY_CHECK.check(tmp_path) == []


def test_unregistered_tool_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    _write(tmp_path, "tools/registry.toml", "")
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("unregistered tool: tools/demo/example.py" in p for p in problems)


def test_stale_entry_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "tools/registry.toml", _registry("gone", "tools/demo/gone.py", "demo"))
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("missing file: tools/demo/gone.py" in p for p in problems)


def test_domain_mismatch_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    _write(
        tmp_path,
        "tools/registry.toml",
        _registry("demo.example", "tools/demo/example.py", "ci"),
    )
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("declares domain 'ci' but lives in 'demo'" in p for p in problems)


def test_duplicate_ids_and_paths_are_reported(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    entry = _registry("demo.example", "tools/demo/example.py", "demo")
    _write(tmp_path, "tools/registry.toml", entry + "\n" + entry)
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("duplicate registry id" in p for p in problems)
    assert any("duplicate registry path" in p for p in problems)


def test_incomplete_entries_are_reported(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    entry = "[[entries]]" + chr(10) + 'id = "demo.example"' + chr(10)
    entry += 'path = "tools/demo/example.py"' + chr(10)
    _write(tmp_path, "tools/registry.toml", entry)
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("missing or non-string domain" in p for p in problems)
    assert any("missing required 'purpose'" in p for p in problems)
    assert any("missing required 'entrypoints'" in p for p in problems)


def test_entry_without_id_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "tools/registry.toml", '[[entries]]\npath = "tools/demo/example.py"\n')
    assert any("missing or non-string id" in p for p in REGISTRY_CHECK.check(tmp_path))


def test_legacy_scripts_directory_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "scripts/stray.py", "print('x')\n")
    _write(tmp_path, "tools/registry.toml", "")
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("scripts/ must stay empty" in p for p in problems)


def test_empty_legacy_scripts_directory_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    _write(tmp_path, "tools/registry.toml", "")
    assert REGISTRY_CHECK.check(tmp_path) == []


def test_missing_registered_file_outside_tools_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/example.py", "print('x')\n")
    _write(tmp_path, "tools/registry.toml", _registry("outside", "src/example.py", "src"))
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("path is outside tools/" in p for p in problems)


def test_discover_skips_helper_files(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/__init__.py", "")
    _write(tmp_path, "tools/demo/conftest.py", "")
    _write(tmp_path, "tools/demo/example.py", "")
    assert REGISTRY_CHECK.discover_tools(tmp_path) == ("tools/demo/example.py",)


def test_discover_without_tools_directory(tmp_path: Path) -> None:
    assert REGISTRY_CHECK.discover_tools(tmp_path) == ()


def test_domain_of_rejects_paths_outside_tools() -> None:
    assert REGISTRY_CHECK.domain_of("src/example.py") == ""
    assert REGISTRY_CHECK.domain_of("tools/demo/example.py") == "demo"
