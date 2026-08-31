from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI lane
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_vx_and_just_pin_the_local_and_ci_toolchain() -> None:
    config = tomllib.loads((ROOT / "vx.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "vx.lock").read_text(encoding="utf-8"))
    recipes = (ROOT / "justfile").read_text(encoding="utf-8")

    assert config["tools"] == {
        "actionlint": "1.7.12",
        "dotnet": "10.0.400",
        "python": "3.12.13",
        "uv": "0.12.7",
        "just": "1.58.0",
        "rust": "1.29.0",
    }
    assert config["scripts"]["check"] == "vx just check"
    assert config["scripts"]["ci"] == "vx just ci"
    assert {name: value["version"] for name, value in lock["tools"].items()} == {
        "actionlint": "1.7.12",
        "dotnet": "10.0.400",
        "just": "1.58.0",
        "python": "3.12.13",
        "rust": "1.29.0",
        "uv": "0.12.7",
    }
    assert 'export UV_PROJECT_ENVIRONMENT := ".venv-glr"' in recipes
    assert "vx uv sync --python 3.12.13 --frozen --all-groups" in recipes
    assert "--no-install-project" in recipes
    assert "--no-build-isolation" in recipes
    assert "vx uv run --no-sync python -m build --no-isolation" in recipes
    assert "vx cargo clippy --workspace --all-targets --locked -- -D warnings" in recipes
    assert "dotnet build sdk/csharp/GameLearningRuntime.Provider.Smoke" in recipes
    for recipe in (
        "check:",
        "ci:",
        "ci-core python_version:",
        "ci-runtime-host:",
        "release-check tag:",
    ):
        assert recipe in recipes


def test_internal_github_actions_execute_the_same_just_recipes() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    action = "loonghao/vx@61a604b0dc206c691c33db8e8f9c420ca47c3914"

    assert action in ci
    assert action in release
    assert "loonghao/vx@main" not in ci + release
    assert "name: core / Python ${{ matrix.python-version }}" in ci
    for recipe in (
        "ci-core",
        "ci-torchrl",
        "ci-gymnasium",
        "ci-package",
        "ci-runtime-host",
    ):
        assert f"vx just {recipe}" in ci
    assert "vx just release-check" in release
    assert "Runtime Host and provider SDKs" in ci
    assert "actions/setup-dotnet@a98b56852c35b8e3190ac28c8c2271da59106c68" in ci
    assert "actions/setup-dotnet@a98b56852c35b8e3190ac28c8c2271da59106c68" in release
    for target in (
        "x86_64-unknown-linux-gnu",
        "x86_64-pc-windows-msvc",
        "x86_64-apple-darwin",
        "aarch64-apple-darwin",
    ):
        assert target in release
    assert "SHA256SUMS" in release


def test_public_reusable_python_workflow_stays_uv_compatible() -> None:
    workflow = (ROOT / ".github/workflows/reusable-python-ci.yml").read_text(encoding="utf-8")

    assert "Reusable uv Python CI" in workflow
    assert "astral-sh/setup-uv@" in workflow


def test_bilingual_readmes_show_how_to_give_the_skill_to_an_agent() -> None:
    readmes = [
        (ROOT / "README.md").read_text(encoding="utf-8"),
        (ROOT / "README.zh-CN.md").read_text(encoding="utf-8"),
    ]
    installer = (
        "$skill-installer install "
        "https://github.com/loonghao/GameLearningRuntime/tree/main/"
        ".agents/skills/glr-adapter-builder"
    )

    for readme in readmes:
        assert installer in readme
        assert "$glr-adapter-builder" in readme
        assert "https://developers.openai.com/codex/skills" in readme
