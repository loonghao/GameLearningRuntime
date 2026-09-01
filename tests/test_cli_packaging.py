from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_cli", ROOT / "scripts/package_cli.py")
assert SPEC is not None and SPEC.loader is not None
PACKAGE_CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE_CLI)


def test_unified_cli_archive_contains_host_manifest_and_agent_skills(tmp_path: Path) -> None:
    cli = tmp_path / "glr.exe"
    host = tmp_path / "glr-hostd.exe"
    cli.write_bytes(b"cli")
    host.write_bytes(b"host")

    archive = PACKAGE_CLI.package_cli(
        cli=cli,
        host=host,
        target="x86_64-pc-windows-msvc",
        version="1.2.3",
        output=tmp_path / "dist",
    )

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        root = "glr-1.2.3-x86_64-pc-windows-msvc"
        manifest = json.loads(bundle.read(f"{root}/glr-release.json"))
    assert {
        f"{root}/glr.exe",
        f"{root}/glr-hostd.exe",
        f"{root}/LICENSE",
        f"{root}/install.md",
        f"{root}/skills/glr-cli/SKILL.md",
        f"{root}/skills/glr-adapter-builder/SKILL.md",
    } <= names
    assert manifest["schema_version"] == "glr.release-bundle.v1"
    assert manifest["target"] == "x86_64-pc-windows-msvc"
