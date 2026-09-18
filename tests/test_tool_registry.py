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


def _registry(entry_id: str, path: str, domain: str, entrypoints: str = '["manual"]') -> str:
    return (
        "[[entries]]\n"
        f'id = "{entry_id}"\n'
        f'path = "{path}"\n'
        f'domain = "{domain}"\n'
        'purpose = "test tool"\n'
        f"entrypoints = {entrypoints}\n"
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
    assert any("non-empty 'purpose'" in p for p in problems)
    assert any("non-empty 'entrypoints' list" in p for p in problems)


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


def test_unknown_domain_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    _write(
        tmp_path, "tools/registry.toml", _registry("demo.example", "tools/demo/example.py", "misc")
    )
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("declares unknown domain 'misc'" in p for p in problems)


def test_identifier_must_be_domain_prefixed_kebab_case(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    for identifier in ("example", "demo.Example", "demo.example_tool", "ci.example"):
        _write(
            tmp_path,
            "tools/registry.toml",
            _registry(identifier, "tools/demo/example.py", "demo"),
        )
        problems = REGISTRY_CHECK.check(tmp_path)
        assert any(f"registry id '{identifier}'" in p for p in problems), identifier
    _write(
        tmp_path,
        "tools/registry.toml",
        _registry("demo.example-tool", "tools/demo/example.py", "demo"),
    )
    assert REGISTRY_CHECK.check(tmp_path) == []


def test_entrypoints_must_be_a_non_empty_string_list(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    for entrypoints in ("[]", '"just layout-check"', '[""]', "[1]"):
        _write(
            tmp_path,
            "tools/registry.toml",
            _registry("demo.example", "tools/demo/example.py", "demo", entrypoints),
        )
        problems = REGISTRY_CHECK.check(tmp_path)
        assert any("entrypoints" in p for p in problems), entrypoints


def test_unrecognized_entrypoint_form_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    _write(
        tmp_path,
        "tools/registry.toml",
        _registry("demo.example", "tools/demo/example.py", "demo", '["make example"]'),
    )
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("unrecognized entrypoint 'make example'" in p for p in problems)


def test_just_recipe_entrypoint_must_exist(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    _write(
        tmp_path, "justfile", "layout-check:\n    python tools/governance/check_tool_registry.py\n"
    )
    _write(
        tmp_path,
        "tools/registry.toml",
        _registry("demo.example", "tools/demo/example.py", "demo", '["just layout-check"]'),
    )
    assert REGISTRY_CHECK.check(tmp_path) == []
    _write(
        tmp_path,
        "tools/registry.toml",
        _registry("demo.example", "tools/demo/example.py", "demo", '["just no-such-recipe"]'),
    )
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("unknown just recipe 'no-such-recipe'" in p for p in problems)


def test_workflow_entrypoint_must_exist_and_invoke_the_tool(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.py", "print('x')\n")
    workflow = ".github/workflows/ci.yml"
    _write(tmp_path, workflow, "jobs:\n  demo:\n    steps:\n      - run: python other.py\n")
    _write(
        tmp_path,
        "tools/registry.toml",
        _registry("demo.example", "tools/demo/example.py", "demo", f'["{workflow}"]'),
    )
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("never invokes tools/demo/example.py" in p for p in problems)

    _write(
        tmp_path,
        workflow,
        "jobs:\n  demo:\n    steps:\n      - run: python tools/demo/example.py\n",
    )
    assert REGISTRY_CHECK.check(tmp_path) == []

    _write(
        tmp_path,
        "tools/registry.toml",
        _registry("demo.example", "tools/demo/example.py", "demo", '["missing-workflow"]'),
    )
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("unrecognized entrypoint 'missing-workflow'" in p for p in problems)

    _write(
        tmp_path,
        "tools/registry.toml",
        _registry(
            "demo.example",
            "tools/demo/example.py",
            "demo",
            '[".github/workflows/gone.yml"]',
        ),
    )
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("missing workflow: .github/workflows/gone.yml" in p for p in problems)


def test_shell_helpers_under_tools_must_be_registered(tmp_path: Path) -> None:
    _write(tmp_path, "tools/demo/example.sh", "echo x\n")
    _write(tmp_path, "tools/registry.toml", "")
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("unregistered tool: tools/demo/example.sh" in p for p in problems)


def test_stray_python_at_the_repository_root_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "quick_fix.py", "print('x')\n")
    _write(tmp_path, "tools/registry.toml", "")
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("outside the module layout: quick_fix.py" in p for p in problems)


def test_library_and_skill_directories_are_not_stray(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", "")
    _write(tmp_path, "plugins/game-learning-runtime-skills/scripts/build.py", "")
    _write(tmp_path, "sdk/unreal/smoke.py", "")
    _write(tmp_path, "tools/registry.toml", "")
    assert REGISTRY_CHECK.check(tmp_path) == []


def test_nested_scripts_directory_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "docs/scripts/legacy.py", "")
    _write(tmp_path, "tools/registry.toml", "")
    problems = REGISTRY_CHECK.check(tmp_path)
    assert any("scripts/ must stay empty" in p for p in problems)


def test_just_recipe_parsing_covers_parameters_and_dependencies() -> None:
    recipes = REGISTRY_CHECK.load_just_recipes(REGISTRY_CHECK.repository_root())
    assert {"layout-check", "core-check", "ci-core"} <= recipes
    assert "set" not in recipes
