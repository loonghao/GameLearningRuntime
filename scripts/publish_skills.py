"""Versioned ClawHub publication with local checksums and validated receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CLI_VERSION = "0.23.1"


@contextmanager
def registry_environment(publish: bool):
    """Give the pinned CLI an isolated credential file, never a token argument."""
    env = os.environ.copy()
    token = env.pop("CLAWHUB_TOKEN", None)
    if not publish:
        yield env
        return
    if not token:
        raise ValueError("configure CLAWHUB_TOKEN in the clawhub GitHub environment")
    with tempfile.TemporaryDirectory(prefix="glr-clawhub-") as directory:
        path = Path(directory) / "config.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"registry": "https://clawhub.ai", "token": token}, stream)
        env["CLAWHUB_CONFIG_PATH"] = str(path)
        yield env


def validate_receipt(value: dict[str, Any], *, slug: str, version: str, dry_run: bool) -> None:
    expected_status = "would-publish" if dry_run else "published"
    if value.get("ok") is not True or value.get("status") != expected_status:
        raise ValueError("registry did not confirm the requested publication state")
    if value.get("slug") != slug or value.get("version") != version:
        raise ValueError("registry receipt identity/version mismatch")
    if re.fullmatch(r"[a-f0-9]{64}", str(value.get("fingerprint"))) is None:
        raise ValueError("registry receipt has no valid fingerprint")
    if type(value.get("fileCount")) is not int or value["fileCount"] < 1:
        raise ValueError("registry receipt has no files")


def bundle_checksums(skill: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(skill.rglob("*")):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("skill bundles must not contain symlinks")
        if path.is_file():
            checksums[path.relative_to(skill).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    if "SKILL.md" not in checksums:
        raise ValueError("skill bundle is missing SKILL.md")
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("dist/clawhub-receipts.json"))
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise ValueError("skills require a stable project semver")
    ref = os.environ.get("GITHUB_REF", "")
    if args.publish:
        if ref != f"refs/tags/v{version}" and not (
            ref == "refs/heads/main" and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        ):
            raise ValueError("publish requires a matching release tag or explicit main dispatch")
        if not os.environ.get("CLAWHUB_TOKEN"):
            raise ValueError("configure CLAWHUB_TOKEN in the clawhub GitHub environment")
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("install Node.js with npx before publishing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    skills = sorted((ROOT / ".agents/skills").iterdir())
    results = []
    for skill in skills:
        if not skill.is_dir() or skill.name.startswith(".") or skill.name == "__pycache__":
            continue
        checksums = bundle_checksums(skill)
        base = [
            npx,
            "--yes",
            f"clawhub@{CLI_VERSION}",
            "--no-input",
            "skill",
            "publish",
            str(skill),
            "--slug",
            skill.name,
            "--owner",
            "loonghao",
            "--version",
            version,
            "--json",
            "--source-repo",
            "loonghao/GameLearningRuntime",
            "--source-commit",
            commit,
            "--source-path",
            skill.relative_to(ROOT).as_posix(),
        ]
        if ref:
            base += ["--source-ref", ref]

        def invoke(*, dry_run: bool, base: list[str] = base, skill: Path = skill) -> dict[str, Any]:
            command = base + (["--dry-run"] if dry_run else [])
            with registry_environment(args.publish) as environment:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=120, env=environment
                )
            if completed.returncode:
                raise RuntimeError(f"ClawHub failed for {skill.name} (exit {completed.returncode})")
            value = json.loads(completed.stdout)
            validate_receipt(value, slug=skill.name, version=version, dry_run=dry_run)
            return dict(value)

        preview = invoke(dry_run=True)
        if preview["fileCount"] != len(checksums):
            raise ValueError("registry file selection differs from the checksummed bundle")
        result = preview
        if args.publish:
            if bundle_checksums(skill) != checksums:
                raise ValueError("bundle changed after preview")
            result = invoke(dry_run=False)
            if any(result[key] != preview[key] for key in ("fingerprint", "fileCount")):
                raise ValueError("published bundle differs from preview")
        results.append(
            {key: result[key] for key in ("status", "slug", "version", "fingerprint", "fileCount")}
            | {"sha256": checksums}
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "version": version,
                    "commit": commit,
                    "public_visibility_verified": False,
                    "skills": results,
                },
                indent=2,
            )
            + "\n"
        )
    if not results:
        raise ValueError("no skill bundles found")
    print(f"Validated {len(results)} skill receipts for {version}")


if __name__ == "__main__":
    main()
