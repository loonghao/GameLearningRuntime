from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "game-learning-runtime-skills"
PACKAGE_SCRIPT = ROOT / "scripts" / "package_agent_plugin.py"
MARKETPLACE = ROOT / ".agents" / "plugins" / "marketplace.json"


def test_distributable_plugin_manifest_and_skill_payload_are_valid() -> None:
    manifest_path = PLUGIN / ".codex-plugin" / "plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["name"] == PLUGIN.name
    assert manifest["version"] == "0.6.1"
    assert manifest["skills"] == "./skills/"
    assert manifest["author"] == {
        "name": "loonghao",
        "email": "hal.long@outlook.com",
        "url": "https://github.com/loonghao",
    }
    assert manifest["interface"]["defaultPrompt"]
    assert (PLUGIN / "skills" / "glr-adapter-builder" / "SKILL.md").is_file()
    assert (PLUGIN / "skills" / "glr-cli" / "SKILL.md").is_file()


def test_repo_marketplace_exposes_the_plugin_with_required_policy() -> None:
    marketplace = json.loads(MARKETPLACE.read_text(encoding="utf-8"))
    entry = next(
        item for item in marketplace["plugins"] if item["name"] == "game-learning-runtime-skills"
    )

    assert entry["source"] == {
        "source": "local",
        "path": "./plugins/game-learning-runtime-skills",
    }
    assert entry["policy"] == {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL",
    }
    assert entry["category"] == "Developer Tools"


def test_plugin_sync_repairs_payload_drift(tmp_path: Path) -> None:
    source = tmp_path / "source" / "example-skill"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: example-skill\ndescription: Example\n---\n", encoding="utf-8"
    )
    plugin = tmp_path / "game-learning-runtime-skills"
    (plugin / ".codex-plugin").mkdir(parents=True)
    (plugin / ".codex-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": plugin.name,
                "version": "0.1.0",
                "description": "Example",
                "author": {"name": "Example"},
                "interface": {"displayName": "Example"},
                "skills": "./skills/",
            }
        ),
        encoding="utf-8",
    )
    packaged_skill = plugin / "skills" / "example-skill"
    packaged_skill.mkdir(parents=True)
    (packaged_skill / "SKILL.md").write_text("stale", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_SCRIPT),
            "--sync",
            "--source",
            str(source.parent),
            "--plugin",
            str(plugin),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (packaged_skill / "SKILL.md").read_text(encoding="utf-8") == (
        source / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_plugin_payload_matches_repository_skills() -> None:
    result = subprocess.run(
        [sys.executable, str(PACKAGE_SCRIPT), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "synchronized" in result.stdout
