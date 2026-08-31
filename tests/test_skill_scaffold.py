from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from game_learning_runtime import (
    DemonstrationGate,
    EngineFamily,
    IntegrationMode,
    load_demonstration_policy_config,
    load_reward_safety_config,
    load_runtime_integration,
    load_training_config,
    verify_model_bundle,
)
from game_learning_runtime.testing import run_environment_conformance

_SKILL = Path(".agents/skills/glr-adapter-builder")
_SCAFFOLD = _SKILL / "scripts/scaffold_adapter.py"
_VALIDATE_RESEARCH = _SKILL / "scripts/validate_research_manifest.py"


@pytest.mark.parametrize("start_mode", ["reset", "attach"])
def test_skill_scaffold_creates_a_trainable_privacy_safe_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start_mode: str
) -> None:
    output = tmp_path / f"adapter-{start_mode}"

    subprocess.run(
        [
            sys.executable,
            str(_SCAFFOLD),
            "--output",
            str(output),
            "--package",
            f"example_{start_mode}",
            "--environment-id",
            f"example.{start_mode}-v1",
            "--start-mode",
            start_mode,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    config = load_training_config(output / "training.json")
    reward_safety = load_reward_safety_config(output / "reward-safety.json")
    demonstration_gate = DemonstrationGate(
        load_demonstration_policy_config(output / "demonstration-policy.json")
    )
    assert config.lifecycle.start_mode == start_mode
    assert config.knowledge_by_id["guide-research"].authority.value == "advisory"
    assert reward_safety.outcome_signal == "outcome"
    assert demonstration_gate is not None

    manifest_path = output / "knowledge/research-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["environment_id"] == f"example.{start_mode}-v1"
    assert manifest["sources"] == []
    agent_interface = json.loads((output / "agent-interface.json").read_text(encoding="utf-8"))
    assert start_mode in agent_interface["operations"]
    assert ({"reset", "attach"} - {start_mode}).isdisjoint(agent_interface["operations"])
    subprocess.run(
        [sys.executable, str(_VALIDATE_RESEARCH), str(manifest_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    monkeypatch.syspath_prepend(str(output / "src"))
    generated = importlib.import_module(f"example_{start_mode}.environment")
    report = run_environment_conformance(
        generated.create_environment(),
        generated.synthetic_policy,
        steps=3,
        start_mode=start_mode,
    )
    assert report.transition_count == 3


def test_skill_scaffold_refuses_to_overwrite_nonempty_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "owned.txt").write_text("preserve", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCAFFOLD),
            "--output",
            str(output),
            "--package",
            "example_adapter",
            "--environment-id",
            "example.environment-v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "non-empty" in result.stderr
    assert (output / "owned.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("engine", "access", "expected_start", "expected_mode"),
    [
        ("unity", "source", "reset", IntegrationMode.ENGINE_PLUGIN),
        ("unreal", "source", "reset", IntegrationMode.ENGINE_PLUGIN),
        ("unity", "external", "attach", IntegrationMode.EXTERNAL_ATTACH),
        ("unreal", "external", "attach", IntegrationMode.EXTERNAL_ATTACH),
    ],
)
def test_skill_scaffold_emits_truthful_engine_integration_profiles(
    tmp_path: Path,
    engine: str,
    access: str,
    expected_start: str,
    expected_mode: IntegrationMode,
) -> None:
    output = tmp_path / f"{engine}-{access}"

    subprocess.run(
        [
            sys.executable,
            str(_SCAFFOLD),
            "--output",
            str(output),
            "--package",
            f"example_{engine}_{access}",
            "--environment-id",
            f"example.{engine}.{access}-v1",
            "--engine",
            engine,
            "--access",
            access,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    profile = load_runtime_integration(output / "runtime-integration.json")
    config = load_training_config(output / "training.json")
    assert profile.engine_family is EngineFamily(engine)
    assert profile.integration_mode is expected_mode
    assert profile.start_mode == expected_start
    assert config.lifecycle.start_mode == expected_start
    assert profile.required_capabilities == config.bridge.required_capabilities
    justfile = (output / "justfile").read_text(encoding="utf-8")
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "vx uv sync --python 3.12.13 --all-groups" in justfile
    assert "--no-install-project" in justfile
    assert "--no-build-isolation" in justfile
    assert "vx uv lock --check" in justfile
    assert "vx run check" in readme
    assert "package-runtime" in justfile


@pytest.mark.parametrize(
    ("engine", "loader", "version", "expected_host_file"),
    [
        ("unity", "bepinex", "v5.4.23.5", "runtime/bepinex/GlrBridgePlugin.cs"),
        ("unreal", "ue4ss", "v3.0.1", "runtime/ue4ss/GLRBridge/Scripts/main.lua"),
    ],
)
def test_skill_scaffold_generates_authorized_loader_host_and_agent_surface(
    tmp_path: Path,
    engine: str,
    loader: str,
    version: str,
    expected_host_file: str,
) -> None:
    output = tmp_path / f"{engine}-{loader}"

    subprocess.run(
        [
            sys.executable,
            str(_SCAFFOLD),
            "--output",
            str(output),
            "--package",
            f"example_{loader}",
            "--environment-id",
            f"example.{loader}-v1",
            "--engine",
            engine,
            "--access",
            "loader",
            "--loader",
            loader,
            "--loader-version",
            version,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    profile = load_runtime_integration(output / "runtime-integration.json")
    config = load_training_config(output / "training.json")
    deployment = json.loads((output / "deployment/loader.json").read_text(encoding="utf-8"))
    agent_interface = json.loads((output / "agent-interface.json").read_text(encoding="utf-8"))

    assert profile.integration_mode is IntegrationMode.LOADER_PLUGIN
    assert profile.start_mode == "attach"
    assert profile.required_capabilities == config.bridge.required_capabilities
    assert deployment["schema_version"] == "glr.loader-deployment.v1"
    assert deployment["loader_family"] == loader
    assert deployment["upstream_version"] == version
    assert deployment["install_mode"] == "manual-explicit-target"
    assert (output / expected_host_file).is_file()
    assert agent_interface["unknown_operations"] == "deny"
    assert agent_interface["required_mutation_fields"] == [
        "episode_id",
        "expected_step_id",
    ]
    assert agent_interface["action_vocabulary"] == []
    assert "authorized" in (output / "AGENTS.md").read_text(encoding="utf-8").lower()

    host_source = (output / expected_host_file).read_text(encoding="utf-8")
    if loader == "bepinex":
        assert "MaxPendingCommands" in host_source
        assert "TryApply" in host_source
        assert "System.Reflection" not in host_source
    else:
        assert "ExecuteInGameThread" in host_source
        assert "ExecuteAsync" not in host_source
        assert "FindFirstOf" not in host_source


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--engine", "unity", "--access", "loader"), "--loader is required"),
        (
            (
                "--engine",
                "unreal",
                "--access",
                "loader",
                "--loader",
                "bepinex",
                "--loader-version",
                "v5.4.23.5",
            ),
            "BepInEx requires --engine unity",
        ),
        (
            (
                "--engine",
                "unity",
                "--access",
                "loader",
                "--loader",
                "ue4ss",
                "--loader-version",
                "v3.0.1",
            ),
            "UE4SS requires --engine unreal",
        ),
    ],
)
def test_skill_scaffold_rejects_ambiguous_loader_requests(
    tmp_path: Path, arguments: tuple[str, ...], message: str
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_SCAFFOLD),
            "--output",
            str(tmp_path / "adapter"),
            "--package",
            "example_adapter",
            "--environment-id",
            "example.loader-v1",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_generated_training_smoke_produces_a_verifiable_reproduction_bundle(
    tmp_path: Path,
) -> None:
    output = tmp_path / "adapter"
    subprocess.run(
        [
            sys.executable,
            str(_SCAFFOLD),
            "--output",
            str(output),
            "--package",
            "example_training",
            "--environment-id",
            "example.training-v1",
            "--engine",
            "unity",
            "--access",
            "loader",
            "--loader",
            "bepinex",
            "--loader-version",
            "v5.4.23.5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run_dir = output / ".glr-runs/reference-model"
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join([str(output / "src"), str(Path("src").resolve())])
    }

    subprocess.run(
        [
            sys.executable,
            str(output / "scripts/train_reference.py"),
            "--output",
            str(run_dir),
        ],
        cwd=output,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = verify_model_bundle(run_dir)
    assert manifest.environment_id == "example.training-v1"
    assert {entry.path for entry in manifest.inputs} >= {
        "agent-interface.json",
        "demonstration-policy.json",
        "deployment/loader.json",
        "reward-safety.json",
        "training.json",
        "runtime-integration.json",
        "runtime/bepinex/GlrBridgePlugin.cs",
        "src/example_training/environment.py",
    }
    assert {entry.path for entry in manifest.artifacts} == {
        "model.json",
        "metrics.json",
    }


def test_ue4ss_loader_package_is_staged_without_writing_a_game_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "adapter"
    package = tmp_path / "staged-package"
    subprocess.run(
        [
            sys.executable,
            str(_SCAFFOLD),
            "--output",
            str(output),
            "--package",
            "example_ue4ss",
            "--environment-id",
            "example.ue4ss-v1",
            "--engine",
            "unreal",
            "--access",
            "loader",
            "--loader",
            "ue4ss",
            "--loader-version",
            "v3.0.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(output / "scripts/package_runtime.py"),
            "--output",
            str(package),
        ],
        cwd=output,
        check=True,
        capture_output=True,
        text=True,
    )

    staged = package / "payload/Mods/GLRBridge/Scripts/main.lua"
    manifest_text = (package / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert staged.is_file()
    assert manifest["schema_version"] == "glr.loader-package.v1"
    assert manifest["loader_family"] == "ue4ss"
    assert manifest["upstream_version"] == "v3.0.1"
    assert manifest["files"][0]["path"] == "Mods/GLRBridge/Scripts/main.lua"
    assert str(output) not in manifest_text


def test_research_validator_rejects_local_or_credentialed_sources(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "glr.knowledge-research.v1",
        "environment_id": "example.environment-v1",
        "generated_at": "2026-08-31T00:00:00Z",
        "sources": [
            {
                "id": "unsafe",
                "url": "https://user:secret@example.invalid/guide",
                "publisher": "example",
                "source_type": "guide",
                "accessed_at": "2026-08-31T00:00:00Z",
                "summary": "short paraphrase",
                "confidence": "low",
                "volatility": "high",
            }
        ],
        "claims": [],
    }
    path = tmp_path / "research.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_VALIDATE_RESEARCH), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "credentials" in result.stderr
