"""Self-contained, checksummed manifests for reproducible trained models."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any

from game_learning_runtime.errors import ContractViolation

MODEL_BUNDLE_SCHEMA_VERSION = "glr.model-bundle.v1"
_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.-")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "environment_id",
        "protocol_version",
        "algorithm",
        "framework",
        "framework_version",
        "seeds",
        "inputs",
        "artifacts",
    }
)


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value[0] not in "abcdefghijklmnopqrstuvwxyz":
        raise ValueError(f"{name} must start with a lowercase letter")
    if any(character not in _IDENTIFIER_CHARS for character in value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must be non-empty printable text")
    return value


def _relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"{name} must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a portable relative path")
    return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BundleFile:
    """One portable file entry stored inside a model bundle."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, name="bundle file path"))
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("bundle file sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("bundle file size_bytes must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BundleFile:
        unexpected = sorted(set(value) - {"path", "sha256", "size_bytes"})
        missing = sorted({"path", "sha256", "size_bytes"} - set(value))
        if unexpected or missing:
            raise ValueError(
                f"bundle file fields differ; missing={missing}, unexpected={unexpected}"
            )
        return cls(path=value["path"], sha256=value["sha256"], size_bytes=value["size_bytes"])


@dataclass(frozen=True, slots=True)
class ModelBundleManifest:
    """Data-only provenance needed to verify and reproduce a trained model."""

    environment_id: str
    protocol_version: str
    algorithm: str
    framework: str
    framework_version: str
    seeds: tuple[int, ...]
    inputs: tuple[BundleFile, ...]
    artifacts: tuple[BundleFile, ...]
    schema_version: str = MODEL_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported model bundle schema: {self.schema_version!r}")
        object.__setattr__(
            self, "environment_id", _identifier(self.environment_id, name="environment_id")
        )
        object.__setattr__(
            self, "protocol_version", _text(self.protocol_version, name="protocol_version")
        )
        object.__setattr__(self, "algorithm", _identifier(self.algorithm, name="algorithm"))
        object.__setattr__(self, "framework", _identifier(self.framework, name="framework"))
        object.__setattr__(
            self, "framework_version", _text(self.framework_version, name="framework_version")
        )
        seeds = tuple(self.seeds)
        if not seeds or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
        ):
            raise ValueError("seeds must contain non-negative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", seeds)
        for group_name, entries in (("inputs", self.inputs), ("artifacts", self.artifacts)):
            if not entries or any(not isinstance(entry, BundleFile) for entry in entries):
                raise ValueError(f"{group_name} must contain BundleFile entries")
            paths = [entry.path for entry in entries]
            if len(set(paths)) != len(paths):
                raise ValueError(f"{group_name} must not contain duplicate paths")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "protocol_version": self.protocol_version,
            "algorithm": self.algorithm,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "seeds": list(self.seeds),
            "inputs": [entry.to_mapping() for entry in self.inputs],
            "artifacts": [entry.to_mapping() for entry in self.artifacts],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelBundleManifest:
        unexpected = sorted(set(value) - _MANIFEST_FIELDS)
        missing = sorted(_MANIFEST_FIELDS - set(value))
        if unexpected or missing:
            raise ValueError(
                f"model bundle has missing fields={missing} and unexpected fields={unexpected}"
            )
        if not isinstance(value["seeds"], Sequence) or isinstance(value["seeds"], str):
            raise TypeError("seeds must be an array")
        return cls(
            schema_version=value["schema_version"],
            environment_id=value["environment_id"],
            protocol_version=value["protocol_version"],
            algorithm=value["algorithm"],
            framework=value["framework"],
            framework_version=value["framework_version"],
            seeds=tuple(value["seeds"]),
            inputs=tuple(BundleFile.from_mapping(item) for item in value["inputs"]),
            artifacts=tuple(BundleFile.from_mapping(item) for item in value["artifacts"]),
        )


def _source_file(value: str | PathLike[str] | Path, *, name: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} must be an existing regular file")
    return path


def _copy_group(
    root: Path,
    group: str,
    sources: Mapping[str, str | PathLike[str] | Path],
) -> tuple[BundleFile, ...]:
    entries: list[BundleFile] = []
    for requested_path, source_value in sorted(sources.items()):
        relative = _relative_path(requested_path, name=f"{group} path")
        source = _source_file(source_value, name=f"{group} source")
        destination = root / group / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entries.append(
            BundleFile(
                path=relative,
                sha256=_sha256(destination),
                size_bytes=destination.stat().st_size,
            )
        )
    return tuple(entries)


def build_model_bundle(
    output: str | PathLike[str] | Path,
    *,
    environment_id: str,
    protocol_version: str,
    algorithm: str,
    framework: str,
    framework_version: str,
    seeds: Sequence[int],
    inputs: Mapping[str, str | PathLike[str] | Path],
    artifacts: Mapping[str, str | PathLike[str] | Path],
) -> ModelBundleManifest:
    """Copy immutable inputs and model artifacts into a new verified bundle."""

    requested_output = Path(output)
    if requested_output.is_symlink():
        raise FileExistsError(f"model bundle output cannot be a symlink: {requested_output}")
    output_path = requested_output.resolve()
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"model bundle output is non-empty: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}-", dir=output_path.parent
    ) as temporary:
        root = Path(temporary)
        manifest = ModelBundleManifest(
            environment_id=environment_id,
            protocol_version=protocol_version,
            algorithm=algorithm,
            framework=framework,
            framework_version=framework_version,
            seeds=tuple(seeds),
            inputs=_copy_group(root, "inputs", inputs),
            artifacts=_copy_group(root, "artifacts", artifacts),
        )
        (root / "manifest.json").write_text(
            json.dumps(manifest.to_mapping(), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        if output_path.exists():
            output_path.rmdir()
        root.replace(output_path)
    return manifest


def load_model_bundle_manifest(path: str | PathLike[str] | Path) -> ModelBundleManifest:
    """Load a strict model bundle manifest."""

    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("model bundle manifest must contain an object")
    return ModelBundleManifest.from_mapping(value)


def verify_model_bundle(path: str | PathLike[str] | Path) -> ModelBundleManifest:
    """Fail closed when any bundled input or artifact changed."""

    root = Path(path)
    manifest = load_model_bundle_manifest(root / "manifest.json")
    for group, entries in (("inputs", manifest.inputs), ("artifacts", manifest.artifacts)):
        for entry in entries:
            candidate = _source_file(root / group / PurePosixPath(entry.path), name="bundle entry")
            if candidate.stat().st_size != entry.size_bytes or _sha256(candidate) != entry.sha256:
                raise ContractViolation(f"sha256 mismatch for {group}/{entry.path}")
    return manifest


__all__ = [
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "BundleFile",
    "ModelBundleManifest",
    "build_model_bundle",
    "load_model_bundle_manifest",
    "verify_model_bundle",
]
