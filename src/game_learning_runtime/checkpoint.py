"""Fail-closed checkpoint contract inspection and explicit migration helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, TypeAlias

from game_learning_runtime.errors import ContractViolation

CHECKPOINT_CONTRACT_SCHEMA_VERSION = "glr.checkpoint-contract.v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "glr.checkpoint-manifest.v1"
_DIGEST_LENGTH = 64
_DIGEST_ALPHABET = frozenset("0123456789abcdef")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "contract",
        "contract_sha256",
        "metadata",
    }
)


def _digest(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in _DIGEST_ALPHABET for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _portable_path(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"{path} must be a portable relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{path} must be a portable relative path")
    return candidate.as_posix()


def _canonical_json(value: Mapping[str, Any], *, path: str) -> bytes:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    try:
        return json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain finite JSON data") from error


@dataclass(frozen=True, slots=True)
class CheckpointContract:
    """Versioned learner contract fingerprints bound to one checkpoint."""

    protocol_version: str
    observation_sha256: str
    action_sha256: str
    reward_sha256: str
    knowledge_sha256: str | None = None
    schema_version: str = CHECKPOINT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint contract schema: {self.schema_version!r}")
        if not isinstance(self.protocol_version, str) or not self.protocol_version:
            raise ValueError("checkpoint contract protocol_version cannot be empty")
        for name in ("observation_sha256", "action_sha256", "reward_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), path=name))
        if self.knowledge_sha256 is not None:
            object.__setattr__(
                self, "knowledge_sha256", _digest(self.knowledge_sha256, path="knowledge_sha256")
            )

    @property
    def digest(self) -> str:
        """Return the canonical digest of all contract fields."""

        payload = _canonical_json(self.to_mapping(), path="checkpoint contract")
        return hashlib.sha256(payload).hexdigest()

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "observation_sha256": self.observation_sha256,
            "action_sha256": self.action_sha256,
            "reward_sha256": self.reward_sha256,
            "knowledge_sha256": self.knowledge_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CheckpointContract:
        expected = {
            "schema_version",
            "protocol_version",
            "observation_sha256",
            "action_sha256",
            "reward_sha256",
            "knowledge_sha256",
        }
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        if missing or unexpected:
            raise ValueError(
                f"checkpoint contract fields differ; missing={missing}, unexpected={unexpected}"
            )
        return cls(
            schema_version=value["schema_version"],
            protocol_version=value["protocol_version"],
            observation_sha256=value["observation_sha256"],
            action_sha256=value["action_sha256"],
            reward_sha256=value["reward_sha256"],
            knowledge_sha256=value["knowledge_sha256"],
        )


@dataclass(frozen=True, slots=True)
class CheckpointContractMismatch:
    """One field-level mismatch and whether the generic migrator can rebind it."""

    field: str
    recorded: str | None
    current: str | None
    migratable: bool


def compare_checkpoint_contract(
    recorded: CheckpointContract,
    current: CheckpointContract,
) -> tuple[CheckpointContractMismatch, ...]:
    """Report field-level differences without changing either contract."""

    mismatches: list[CheckpointContractMismatch] = []
    for field_name in (
        "schema_version",
        "protocol_version",
        "observation_sha256",
        "action_sha256",
        "reward_sha256",
        "knowledge_sha256",
    ):
        recorded_value = getattr(recorded, field_name)
        current_value = getattr(current, field_name)
        if recorded_value != current_value:
            mismatches.append(
                CheckpointContractMismatch(
                    field=field_name,
                    recorded=recorded_value,
                    current=current_value,
                    migratable=field_name in {"reward_sha256", "knowledge_sha256"},
                )
            )
    return tuple(mismatches)


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Portable manifest that binds checkpoint bytes to a contract digest."""

    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    contract: CheckpointContract
    contract_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CHECKPOINT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint manifest schema: {self.schema_version!r}")
        object.__setattr__(
            self, "checkpoint_path", _portable_path(self.checkpoint_path, path="checkpoint_path")
        )
        object.__setattr__(
            self, "checkpoint_sha256", _digest(self.checkpoint_sha256, path="checkpoint_sha256")
        )
        if (
            not isinstance(self.checkpoint_size_bytes, int)
            or isinstance(self.checkpoint_size_bytes, bool)
            or self.checkpoint_size_bytes < 0
        ):
            raise ValueError("checkpoint_size_bytes must be a non-negative integer")
        if not isinstance(self.contract, CheckpointContract):
            raise TypeError("contract must be a CheckpointContract")
        expected_contract_digest = self.contract.digest
        if self.contract_sha256 is not None:
            _digest(self.contract_sha256, path="contract_sha256")
            if self.contract_sha256 != expected_contract_digest:
                raise ValueError("contract_sha256 does not match contract")
        else:
            object.__setattr__(self, "contract_sha256", expected_contract_digest)
        encoded = _canonical_json(self.metadata or {}, path="checkpoint metadata")
        object.__setattr__(self, "metadata", MappingProxyType(json.loads(encoded)))

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "contract": self.contract.to_mapping(),
            "contract_sha256": self.contract_sha256,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CheckpointManifest:
        missing = sorted(_MANIFEST_FIELDS - set(value))
        unexpected = sorted(set(value) - _MANIFEST_FIELDS)
        if missing or unexpected:
            raise ValueError(
                f"checkpoint manifest fields differ; missing={missing}, unexpected={unexpected}"
            )
        if not isinstance(value["contract"], Mapping):
            raise TypeError("checkpoint manifest contract must be an object")
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("checkpoint manifest metadata must be an object")
        return cls(
            schema_version=value["schema_version"],
            checkpoint_path=value["checkpoint_path"],
            checkpoint_sha256=value["checkpoint_sha256"],
            checkpoint_size_bytes=value["checkpoint_size_bytes"],
            contract=CheckpointContract.from_mapping(value["contract"]),
            contract_sha256=value["contract_sha256"],
            metadata=value["metadata"],
        )


@dataclass(frozen=True, slots=True)
class CheckpointMigrationResult:
    """Machine-readable result for dry-run, no-op, or completed migration."""

    changed: bool
    requires_confirmation: bool
    mismatches: tuple[CheckpointContractMismatch, ...]
    manifest: CheckpointManifest
    manifest_backup: Path | None = None
    checkpoint_backup: Path | None = None


CheckpointSaver: TypeAlias = Callable[[Path, Path, CheckpointContract], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_file(manifest_path: Path, relative: str) -> Path:
    root = manifest_path.parent.resolve()
    unresolved = root / PurePosixPath(relative)
    if unresolved.is_symlink():
        raise ContractViolation("checkpoint path cannot be a symlink")
    candidate = unresolved.resolve()
    if candidate.parent != root and root not in candidate.parents:
        raise ContractViolation("checkpoint path escapes the manifest directory")
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError("checkpoint must be an existing regular non-symlink file")
    return candidate


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    """Load and strictly validate one checkpoint manifest."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError("checkpoint manifest must be an existing regular file")
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("checkpoint manifest exceeds the 1 MiB safety limit")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint manifest is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint manifest must contain an object")
    return CheckpointManifest.from_mapping(value)


def verify_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    """Verify both checkpoint bytes and the manifest's contract digest."""

    manifest_path = Path(path)
    manifest = load_checkpoint_manifest(manifest_path)
    checkpoint = _checkpoint_file(manifest_path, manifest.checkpoint_path)
    if (
        checkpoint.stat().st_size != manifest.checkpoint_size_bytes
        or _sha256(checkpoint) != manifest.checkpoint_sha256
    ):
        raise ContractViolation(
            "checkpoint bytes do not match the manifest; restore the backup or regenerate it"
        )
    return manifest


def write_checkpoint_manifest(
    path: str | Path,
    *,
    checkpoint_path: str,
    contract: CheckpointContract,
    metadata: Mapping[str, Any] | None = None,
) -> CheckpointManifest:
    """Create a new manifest for an existing checkpoint without overwriting it."""

    manifest_path = Path(path)
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError("checkpoint manifest already exists")
    checkpoint = _checkpoint_file(
        manifest_path, _portable_path(checkpoint_path, path="checkpoint_path")
    )
    manifest = CheckpointManifest(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256(checkpoint),
        checkpoint_size_bytes=checkpoint.stat().st_size,
        contract=contract,
        metadata=metadata or {},
    )
    _atomic_manifest_write(manifest_path, manifest)
    return verify_checkpoint_manifest(manifest_path)


def _backup(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.bak")
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.bak.{index}")
        index += 1
    shutil.copyfile(path, candidate)
    return candidate


def _atomic_manifest_write(path: Path, manifest: CheckpointManifest) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(manifest.to_mapping(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def migrate_checkpoint_manifest(
    path: str | Path,
    current_contract: CheckpointContract,
    *,
    confirm: bool = False,
    saver: CheckpointSaver | None = None,
) -> CheckpointMigrationResult:
    """Explicitly migrate reward/knowledge contract fields after a dry-run.

    Observation, action, protocol, and schema changes fail closed. A caller may
    provide ``saver(source, destination, contract)`` to deserialize and re-save
    framework weights; without one the generic path copies bytes exactly and
    still verifies the resulting checksum. No file is changed unless
    ``confirm=True``.
    """

    manifest_path = Path(path)
    manifest = verify_checkpoint_manifest(manifest_path)
    mismatches = compare_checkpoint_contract(manifest.contract, current_contract)
    if not mismatches:
        return CheckpointMigrationResult(False, False, (), manifest)
    incompatible = tuple(item for item in mismatches if not item.migratable)
    if incompatible:
        details = ", ".join(
            f"{item.field} recorded={item.recorded!r} current={item.current!r}"
            for item in incompatible
        )
        raise ContractViolation(
            f"checkpoint contract is incompatible ({details}); retrain instead of reshaping weights"
        )
    if not confirm:
        return CheckpointMigrationResult(True, True, mismatches, manifest)

    checkpoint = _checkpoint_file(manifest_path, manifest.checkpoint_path)
    manifest_backup = _backup(manifest_path)
    checkpoint_backup = _backup(checkpoint)
    temporary_checkpoint = checkpoint.with_name(f".{checkpoint.name}.migration-tmp")
    try:
        if temporary_checkpoint.exists():
            temporary_checkpoint.unlink()
        if saver is None:
            shutil.copyfile(checkpoint, temporary_checkpoint)
        else:
            saver(checkpoint, temporary_checkpoint, current_contract)
        if temporary_checkpoint.is_symlink() or not temporary_checkpoint.is_file():
            raise ContractViolation("checkpoint saver did not produce a regular file")
        migrated = replace(
            manifest,
            checkpoint_sha256=_sha256(temporary_checkpoint),
            checkpoint_size_bytes=temporary_checkpoint.stat().st_size,
            contract=current_contract,
            contract_sha256=None,
        )
        os.replace(temporary_checkpoint, checkpoint)
        _atomic_manifest_write(manifest_path, migrated)
        verified = verify_checkpoint_manifest(manifest_path)
    except BaseException:
        if temporary_checkpoint.exists():
            temporary_checkpoint.unlink()
        shutil.copyfile(checkpoint_backup, checkpoint)
        shutil.copyfile(manifest_backup, manifest_path)
        raise
    return CheckpointMigrationResult(
        True,
        False,
        mismatches,
        verified,
        manifest_backup=manifest_backup,
        checkpoint_backup=checkpoint_backup,
    )


__all__ = [
    "CHECKPOINT_CONTRACT_SCHEMA_VERSION",
    "CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "CheckpointContract",
    "CheckpointContractMismatch",
    "CheckpointManifest",
    "CheckpointMigrationResult",
    "compare_checkpoint_contract",
    "load_checkpoint_manifest",
    "migrate_checkpoint_manifest",
    "verify_checkpoint_manifest",
    "write_checkpoint_manifest",
]
