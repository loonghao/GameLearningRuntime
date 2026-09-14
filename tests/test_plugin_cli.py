from __future__ import annotations

import json
from pathlib import Path

import pytest

from game_learning_runtime.cli import main
from game_learning_runtime.plugins import PLUGIN_SCHEMA_VERSION, PluginError


def _bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "plugin.py").write_text("def create():\n    return None\n", encoding="utf-8")
    (bundle / "glr-plugin.json").write_text(
        json.dumps(
            {
                "schema_version": PLUGIN_SCHEMA_VERSION,
                "id": "cli-plugin",
                "version": "1.0.0",
                "kind": "learner",
                "name": "CLI plugin",
                "description": "A CLI contract fixture.",
                "entrypoint": "plugin:create",
                "capabilities": ["learner.ppo"],
                "permissions": ["read:environment"],
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_plugin_cli_manages_a_profile_without_loading_runtime_roles(
    tmp_path: Path, capsys: object
) -> None:
    bundle = _bundle(tmp_path)
    project = tmp_path / "project"

    assert (
        main(["--project", str(project), "--json", "plugin", "inspect", "--source", str(bundle)])
        == 0
    )
    inspected = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert inspected["command"] == "plugin.inspect"
    digest = inspected["data"]["content_sha256"]

    assert (
        main(
            [
                "--project",
                str(project),
                "--json",
                "plugin",
                "install",
                "--source",
                str(bundle),
                "--sha256",
                digest,
            ]
        )
        == 0
    )
    installed = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert installed["command"] == "plugin.install"

    assert main(["--project", str(project), "--json", "plugin", "list"]) == 0
    assert len(json.loads(capsys.readouterr().out)["data"]) == 1  # type: ignore[attr-defined]

    assert (
        main(
            [
                "--project",
                str(project),
                "--json",
                "plugin",
                "profile",
                "enable",
                "training",
                "cli-plugin",
                "--grant",
                "read:environment",
            ]
        )
        == 0
    )
    enabled = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert enabled["data"]["plugins"][0]["enabled"] is True

    assert (
        main(
            [
                "--project",
                str(project),
                "--json",
                "plugin",
                "profile",
                "resolve",
                "training",
            ]
        )
        == 0
    )
    resolved = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert resolved["command"] == "plugin.profile.resolve"
    assert resolved["data"]["plugins"][0]["id"] == "cli-plugin"

    assert (
        main(
            [
                "--project",
                str(project),
                "--json",
                "plugin",
                "health",
                "--profile",
                "training",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["data"][0]["status"] == "ready"  # type: ignore[attr-defined]

    assert (
        main(
            [
                "--project",
                str(project),
                "--json",
                "plugin",
                "profile",
                "disable",
                "training",
                "cli-plugin",
            ]
        )
        == 0
    )
    disabled = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert disabled["data"]["plugins"][0]["enabled"] is False


def test_plugin_cli_remove_requires_an_explicit_version_when_ambiguous(
    tmp_path: Path, capsys: object
) -> None:
    first = _bundle(tmp_path)
    second = tmp_path / "bundle-v2"
    second.mkdir()
    (second / "plugin.py").write_text("def create():\n    return None\n", encoding="utf-8")
    manifest = json.loads((first / "glr-plugin.json").read_text(encoding="utf-8"))
    manifest["version"] = "2.0.0"
    (second / "glr-plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    project = tmp_path / "project"
    for bundle in (first, second):
        assert (
            main(
                [
                    "--project",
                    str(project),
                    "--json",
                    "plugin",
                    "install",
                    "--source",
                    str(bundle),
                ]
            )
            == 0
        )
        capsys.readouterr()  # type: ignore[attr-defined]

    with pytest.raises(PluginError, match="version is required"):
        main(["--project", str(project), "--json", "plugin", "remove", "cli-plugin"])
