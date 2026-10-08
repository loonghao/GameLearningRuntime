import json
import subprocess
import sys
from pathlib import Path

import pytest

from game_learning_runtime.runtime_integration import load_runtime_integration


@pytest.mark.parametrize(
    "engine,runtime", [("unity", "mono"), ("unity", "il2cpp"), ("unreal", None), ("godot", None)]
)
@pytest.mark.parametrize("access", ["source", "external"])
def test_multi_engine_lanes(tmp_path: Path, engine, runtime, access):
    args = [
        sys.executable,
        ".agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py",
        "--output",
        str(tmp_path / "adapter"),
        "--package",
        "example_adapter",
        "--environment-id",
        "example.runtime-v1",
        "--engine",
        engine,
        "--access",
        access,
    ]
    if runtime:
        args += ["--unity-runtime", runtime]
    subprocess.run(args, check=True, capture_output=True)
    profile = load_runtime_integration(tmp_path / "adapter/runtime-integration.json")
    assert profile.engine_family.value == engine
    selection = json.loads(
        (tmp_path / "adapter/runtime-selection.json").read_text(encoding="utf-8")
    )
    assert selection["live_verified"] is False
    if access == "external":
        assert "manual-step" not in profile.required_capabilities


def test_il2cpp_never_generates_mono_bootstrap(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            ".agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py",
            "--output",
            str(tmp_path / "adapter"),
            "--package",
            "example_adapter",
            "--environment-id",
            "example.runtime-v1",
            "--engine",
            "unity",
            "--unity-runtime",
            "il2cpp",
            "--access",
            "loader",
            "--loader",
            "bepinex",
            "--loader-version",
            "6.0.0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not implemented" in result.stderr
    assert not (tmp_path / "adapter").exists()


def test_scaffold_vx_toml_declares_no_scripts(tmp_path: Path) -> None:
    """The generated vx.toml must not mirror justfile recipes.

    Every entry the template used to emit was a `vx just <recipe>` forward, so
    a scaffolded project violated the duplicate-task rule the moment it was
    generated. The recipes still exist in the generated justfile, so running
    them through `just <recipe>` loses nothing.
    """
    subprocess.run(
        [
            sys.executable,
            ".agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py",
            "--output",
            str(tmp_path / "adapter"),
            "--package",
            "example_adapter",
            "--environment-id",
            "example.runtime-v1",
            "--engine",
            "unreal",
        ],
        check=True,
        capture_output=True,
    )
    vx_toml = (tmp_path / "adapter" / "vx.toml").read_text(encoding="utf-8")
    justfile = (tmp_path / "adapter" / "justfile").read_text(encoding="utf-8")
    assert "[scripts]" not in vx_toml
    # The tasks themselves are still available, just not via [scripts].
    for recipe in ("setup", "check", "ci", "test", "train", "reproduce"):
        assert f"\n{recipe}" in justfile
