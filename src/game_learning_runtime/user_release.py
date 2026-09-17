"""Prepare and compile an offline Windows installer from a reviewed runtime payload.

This build-time API does not freeze Python, train a model, or execute the payload.
The application owns its frozen launcher and bundled runtime dependencies.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from game_learning_runtime.model_bundle import verify_model_bundle

SCHEMA = "glr.user-release.v1"
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_PATH = re.compile(r"[a-zA-Z0-9_. /-]+")
_REQUIRED = {"glr.exe", "glr-hostd.exe", "LICENSE.txt", "MODEL_CARD.md", "evaluation.json"}


def _regular(path: Path) -> None:
    for part in (path, *path.parents):
        details = part.lstat()
        if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
            raise ValueError("release paths must not contain links or reparse points")
    if not path.is_file():
        raise ValueError("release entry must be a regular file")


def _portable(value: str) -> str:
    if not _PATH.fullmatch(value) or len(value) > 200:
        raise ValueError("release path must be portable")
    for part in value.split("/"):
        stem = part.split(".")[0].upper()
        if (
            part in {"", ".", ".."}
            or part.endswith((".", " "))
            or stem in {"CON", "PRN", "AUX", "NUL"}
            or re.fullmatch(r"(?:COM|LPT)[0-9]", stem)
        ):
            raise ValueError("release path must be portable")
    return value


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or getattr(path.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("release payload must not contain links or reparse points")
        if path.is_dir():
            continue
        _regular(path)
        relative = _portable(path.relative_to(root).as_posix())
        if relative.casefold() in seen:
            raise ValueError("release paths collide on Windows")
        seen.add(relative.casefold())
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files[relative] = {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return files


def _script(release: dict[str, Any]) -> str:
    identity = release["release_id"]
    version = release["version"]
    # Versioned identity permits side-by-side rollback; no previous version is removed.
    lines = [
        "[Setup]",
        f"AppId=glr-{identity}-{version}",
        f"AppName={identity}",
        f"AppVersion={version}",
        f"DefaultDirName={{localappdata}}\\GLR\\{identity}\\{version}",
        "PrivilegesRequired=lowest",
        "ArchitecturesAllowed=x64compatible",
        "ArchitecturesInstallIn64BitMode=x64compatible",
        "UsePreviousAppDir=no",
        "DisableProgramGroupPage=yes",
        "LicenseFile=payload\\LICENSE.txt",
        "OutputDir=installer",
        f"OutputBaseFilename={identity}-{version}-windows-x64-setup",
        "Compression=lzma2",
        "SolidCompression=yes",
        "[Files]",
    ]
    for relative in release["files"]:
        windows_path = relative.replace("/", "\\")
        parent = relative.rpartition("/")[0].replace("/", "\\")
        destination = "{app}" + ("\\" + parent if parent else "")
        lines.append(
            f'Source: "payload\\{windows_path}"; DestDir: "{destination}"; Flags: ignoreversion'
        )
    lines.extend(
        [
            'Source: "release.json"; DestDir: "{app}"; DestName: "glr-user-release.json"',
            "[Icons]",
            f'Name: "{{userprograms}}\\{identity} {version}"; '
            f'Filename: "{{app}}\\{release["launcher"].replace("/", chr(92))}"; '
            'WorkingDir: "{app}"',
        ]
    )
    # No auto-run, PATH mutation, downloads, or recursive uninstall deletion.
    return "\n".join(lines) + "\n"


def _identity(release_id: str, version: str, stage: str, launcher: str) -> None:
    if not _IDENTIFIER.fullmatch(release_id) or not _IDENTIFIER.fullmatch(stage):
        raise ValueError("release_id and stage must be portable identifiers")
    if not _VERSION.fullmatch(version):
        raise ValueError("version must be a numeric major.minor.patch")
    if not _portable(launcher).endswith(".exe"):
        raise ValueError("launcher must be a bundled Windows executable")


def prepare_windows_installer(
    payload: Path,
    output: Path,
    *,
    release_id: str,
    version: str,
    stage: str,
    launcher: str,
) -> Path:
    """Snapshot an explicitly staged payload and emit a checksummed Inno project.

    Payload includes model/, a self-contained frozen-inference launcher, GLR
    executables, runtime dependencies, license, model card and evaluation receipt.
    This validates structure/integrity only, not gameplay quality or installability.
    """
    _identity(release_id, version, stage, launcher)
    if output.exists() or output.is_symlink():
        raise FileExistsError("installer output must be new")
    # Check the original ancestors before resolving paths so links are not hidden.
    for candidate in (payload, output):
        for part in (candidate, *candidate.parents):
            if part.exists() and (
                part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400
            ):
                raise ValueError("release paths must not contain links or reparse points")
    payload = payload.resolve()
    output = output.resolve()
    if output.is_relative_to(payload) or payload.is_relative_to(output):
        raise ValueError("payload and installer output must be separate")
    files = _inventory(payload)
    if not (_REQUIRED | {launcher}).issubset(files):
        raise ValueError("payload is missing launcher, GLR, license, model card or evaluation")
    manifest = verify_model_bundle(payload / "model")
    evaluation = json.loads((payload / "evaluation.json").read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict) or any(
        evaluation.get(key) != value
        for key, value in {
            "stage": stage,
            "environment_id": manifest.environment_id,
            "protocol_version": manifest.protocol_version,
        }.items()
    ):
        raise ValueError("evaluation must identify the same stage, environment and protocol")
    if evaluation.get("evidence_kind") not in {"synthetic", "live"}:
        raise ValueError("evaluation must distinguish synthetic and live evidence")
    release = {
        "schema_version": SCHEMA,
        "release_id": release_id,
        "version": version,
        "stage": stage,
        "target": "windows-x64",
        "mode": "frozen-playback",
        "launcher": launcher,
        "environment_id": manifest.environment_id,
        "protocol_version": manifest.protocol_version,
        "evidence_kind": evaluation["evidence_kind"],
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".glr-release-", dir=output.parent) as temporary:
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        for relative in files:
            destination = staging / "payload" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(payload / relative, destination)
        if _inventory(staging / "payload") != files:
            raise ValueError("payload changed while preparing the installer")
        (staging / "release.json").write_text(
            json.dumps(release, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "setup.iss").write_text(_script(release), encoding="utf-8")
        staging.rename(output)
    return output / "setup.iss"


def compile_windows_installer(prepared: Path, *, compiler: Path) -> Path:
    """Verify the snapshot and invoke an explicitly selected ISCC compiler.

    The caller supplies a trusted compiler. Signing and installed-app acceptance
    are separate release gates; compilation never launches the payload.
    """
    _regular(prepared / "release.json")
    release = json.loads((prepared / "release.json").read_text(encoding="utf-8"))
    if release.get("schema_version") != SCHEMA:
        raise ValueError("unsupported user-release schema")
    _identity(release["release_id"], release["version"], release["stage"], release["launcher"])
    if _inventory(prepared / "payload") != release["files"]:
        raise ValueError("prepared payload failed integrity verification")
    _regular(prepared / "setup.iss")
    if (prepared / "setup.iss").read_text(encoding="utf-8") != _script(release):
        raise ValueError("installer script differs from the release contract")
    _regular(compiler)
    artifact = (
        prepared
        / "installer"
        / (f"{release['release_id']}-{release['version']}-windows-x64-setup.exe")
    )
    if artifact.exists():
        raise FileExistsError("installer artifact already exists")
    subprocess.run([str(compiler.resolve()), str((prepared / "setup.iss").resolve())], check=True)
    _regular(artifact)
    return artifact
