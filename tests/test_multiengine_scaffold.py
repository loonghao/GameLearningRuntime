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
    selection = json.loads((tmp_path / "adapter/runtime-selection.json").read_text())
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
