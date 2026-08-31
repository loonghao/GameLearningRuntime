from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from game_learning_runtime import load_training_config
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
    assert config.lifecycle.start_mode == start_mode
    assert config.knowledge_by_id["guide-research"].authority.value == "advisory"

    manifest_path = output / "knowledge/research-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["environment_id"] == f"example.{start_mode}-v1"
    assert manifest["sources"] == []
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
