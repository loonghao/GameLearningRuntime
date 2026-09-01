from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TARGETS = {
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
}
VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
SKILLS = ("glr-adapter-builder", "glr-cli")


def _entry(name: str, data: bytes, *, executable: bool) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    return info, data


def _regular_binary(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular non-linked file")
    return resolved


def package_cli(*, cli: Path, host: Path, target: str, version: str, output: Path) -> Path:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported Rust target: {target}")
    if not VERSION.fullmatch(version):
        raise ValueError("version must be a semantic version without a v prefix")
    cli = _regular_binary(cli, label="CLI binary")
    host = _regular_binary(host, label="Runtime Host binary")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"glr-{version}-{target}.zip"
    executable_suffix = ".exe" if "windows" in target else ""
    root = f"glr-{version}-{target}"
    manifest = {
        "schema_version": "glr.release-bundle.v1",
        "version": version,
        "target": target,
        "cli": f"glr{executable_suffix}",
        "host": f"glr-hostd{executable_suffix}",
        "skills": [{"name": name, "path": f"skills/{name}"} for name in SKILLS],
    }
    install = f"""# Install GLR {version}

1. Verify this archive against `SHA256SUMS` from the same GitHub Release.
2. Put `glr{executable_suffix}` and `glr-hostd{executable_suffix}` on `PATH`.
3. Run `glr --version` and `glr --project PATH doctor`.
4. Copy `skills/*` into a project's `.agents/skills/`, or let `glr update --yes`
   synchronize them on the next release.

The standalone CLI is the deployment and agent-control entrypoint. Install the
Python `game-learning-runtime` package only in projects that use the Python
learning SDK.
"""
    entries = [
        _entry(f"{root}/glr{executable_suffix}", cli.read_bytes(), executable=True),
        _entry(f"{root}/glr-hostd{executable_suffix}", host.read_bytes(), executable=True),
        _entry(f"{root}/LICENSE", (ROOT / "LICENSE").read_bytes(), executable=False),
        _entry(f"{root}/install.md", install.encode(), executable=False),
        _entry(
            f"{root}/glr-release.json",
            (json.dumps(manifest, indent=2) + "\n").encode(),
            executable=False,
        ),
    ]
    for skill_name in SKILLS:
        skill_root = ROOT / ".agents" / "skills" / skill_name
        if not (skill_root / "SKILL.md").is_file():
            raise ValueError(f"skill is missing SKILL.md: {skill_name}")
        for path in sorted(skill_root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"skill cannot contain symlinks: {path}")
            if path.is_file():
                relative = path.relative_to(skill_root).as_posix()
                entries.append(
                    _entry(
                        f"{root}/skills/{skill_name}/{relative}",
                        path.read_bytes(),
                        executable=False,
                    )
                )
    with zipfile.ZipFile(archive, "w") as bundle:
        for info, data in sorted(entries, key=lambda item: item[0].filename):
            bundle.writestr(info, data)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one deterministic GLR distribution")
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(SUPPORTED_TARGETS), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=Path(".glr-release"))
    arguments = parser.parse_args()
    archive = package_cli(
        cli=arguments.cli,
        host=arguments.host,
        target=arguments.target,
        version=arguments.version,
        output=arguments.output,
    )
    print(archive.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
