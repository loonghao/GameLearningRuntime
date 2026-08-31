"""Checksummed demonstration artifacts for auditable BC ingestion."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from game_learning_runtime.contracts import Transition
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.serialization import transition_from_record
from game_learning_runtime.training_safety import (
    DemonstrationGate,
    DemonstrationOrigin,
    DemonstrationOutcome,
    DemonstrationProvenance,
)

DEMONSTRATION_ARTIFACT_SCHEMA_VERSION = "glr.demonstration-artifact.v1"
MAX_DEMONSTRATION_BYTES = 256 * 1024 * 1024
_MANIFEST_FIELDS = frozenset(
    {"schema_version", "environment_id", "episode_id", "trajectory", "provenance"}
)


def _strict_fields(value: Mapping[str, Any], *, expected: frozenset[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(f"{path} has missing={missing} and unexpected={unexpected} fields")


def _environment_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise ValueError("environment_id must be non-empty and cannot contain whitespace")
    return value


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("trajectory.path must be a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("trajectory.path must be a portable relative path")
    return path.as_posix()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class DemonstrationArtifactFile:
    """Portable identity of one serialized transition trajectory."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("trajectory.sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("trajectory.size_bytes must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DemonstrationArtifactFile:
        _strict_fields(
            value,
            expected=frozenset({"path", "sha256", "size_bytes"}),
            path="trajectory",
        )
        return cls(
            path=value["path"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
        )


@dataclass(frozen=True, slots=True)
class DemonstrationArtifactManifest:
    """Immutable provenance bound to one ``glr.transition.v1`` episode."""

    environment_id: str
    episode_id: UUID
    trajectory: DemonstrationArtifactFile
    provenance: DemonstrationProvenance
    schema_version: str = DEMONSTRATION_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DEMONSTRATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported demonstration artifact schema: {self.schema_version!r}")
        object.__setattr__(self, "environment_id", _environment_id(self.environment_id))
        if not isinstance(self.episode_id, UUID):
            object.__setattr__(self, "episode_id", UUID(str(self.episode_id)))
        if not isinstance(self.trajectory, DemonstrationArtifactFile):
            raise TypeError("trajectory must be a DemonstrationArtifactFile")
        if not isinstance(self.provenance, DemonstrationProvenance):
            raise TypeError("provenance must be a DemonstrationProvenance")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "episode_id": str(self.episode_id),
            "trajectory": self.trajectory.to_mapping(),
            "provenance": {
                "origin": self.provenance.origin.value,
                "outcome": self.provenance.outcome.value,
                "policy_id": self.provenance.policy_id,
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DemonstrationArtifactManifest:
        _strict_fields(value, expected=_MANIFEST_FIELDS, path="demonstration artifact")
        raw_trajectory = value["trajectory"]
        raw_provenance = value["provenance"]
        if not isinstance(raw_trajectory, Mapping):
            raise TypeError("trajectory must be an object")
        if not isinstance(raw_provenance, Mapping):
            raise TypeError("provenance must be an object")
        _strict_fields(
            raw_provenance,
            expected=frozenset({"origin", "outcome", "policy_id"}),
            path="provenance",
        )
        return cls(
            schema_version=value["schema_version"],
            environment_id=value["environment_id"],
            episode_id=UUID(str(value["episode_id"])),
            trajectory=DemonstrationArtifactFile.from_mapping(raw_trajectory),
            provenance=DemonstrationProvenance(
                origin=DemonstrationOrigin(raw_provenance["origin"]),
                outcome=DemonstrationOutcome(raw_provenance["outcome"]),
                policy_id=raw_provenance["policy_id"],
            ),
        )


@dataclass(frozen=True, slots=True)
class VerifiedDemonstrationArtifact:
    """A byte-verified trajectory admitted by a demonstration policy."""

    manifest: DemonstrationArtifactManifest
    transitions: tuple[Transition, ...]
    sample_weight: float


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be an existing regular non-symlink file")
    return path


def _read_trajectory_bytes(path: Path) -> bytes:
    source = _regular_file(path, label="trajectory")
    size = source.stat().st_size
    if size > MAX_DEMONSTRATION_BYTES:
        raise ValueError("trajectory exceeds the 256 MiB safety limit")
    return source.read_bytes()


def _parse_transitions(data: bytes) -> tuple[Transition, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("trajectory is not valid UTF-8") from error
    transitions: list[Transition] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise TypeError("transition record must be an object")
            transitions.append(transition_from_record(record))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid transition record at line {line_number}") from error
    if not transitions:
        raise ContractViolation("demonstration trajectory must contain at least one transition")
    return tuple(transitions)


def _read_transitions(path: Path) -> tuple[bytes, tuple[Transition, ...]]:
    data = _read_trajectory_bytes(path)
    return data, _parse_transitions(data)


def _validate_episode(
    transitions: tuple[Transition, ...],
    *,
    episode_id: UUID | None = None,
    outcome: DemonstrationOutcome,
) -> UUID:
    observed_episode = transitions[0].episode_id
    if episode_id is not None and observed_episode != episode_id:
        raise ContractViolation("trajectory episode_id does not match the manifest")
    for expected_step, transition in enumerate(transitions):
        if transition.episode_id != observed_episode:
            raise ContractViolation("demonstration trajectory contains multiple episode IDs")
        if transition.step_id != expected_step:
            raise ContractViolation(
                "demonstration trajectory step IDs are not contiguous from zero"
            )
    if (
        outcome in {DemonstrationOutcome.SUCCESS, DemonstrationOutcome.FAILURE}
        and not transitions[-1].done
    ):
        raise ContractViolation(
            "successful or failed demonstration must end at a terminal boundary"
        )
    return observed_episode


def _portable_trajectory_path(manifest_path: Path, trajectory_path: Path) -> str:
    root = manifest_path.parent.resolve()
    resolved = trajectory_path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("trajectory must be inside the manifest directory") from error
    return _relative_path(relative.as_posix())


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise FileExistsError("manifest target must be a regular file or a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def build_demonstration_artifact(
    manifest_path: str | Path,
    *,
    trajectory_path: str | Path,
    environment_id: str,
    provenance: DemonstrationProvenance,
) -> DemonstrationArtifactManifest:
    """Bind a complete transition episode to immutable provenance and bytes."""

    if not isinstance(provenance, DemonstrationProvenance):
        raise TypeError("provenance must be a DemonstrationProvenance")
    manifest_target = Path(manifest_path)
    trajectory_source = _regular_file(Path(trajectory_path), label="trajectory")
    data, transitions = _read_transitions(trajectory_source)
    episode_id = _validate_episode(transitions, outcome=provenance.outcome)
    manifest = DemonstrationArtifactManifest(
        environment_id=environment_id,
        episode_id=episode_id,
        trajectory=DemonstrationArtifactFile(
            path=_portable_trajectory_path(manifest_target, trajectory_source),
            sha256=_sha256(data),
            size_bytes=len(data),
        ),
        provenance=provenance,
    )
    _atomic_json(manifest_target, manifest.to_mapping())
    return manifest


def load_demonstration_artifact_manifest(
    path: str | Path,
) -> DemonstrationArtifactManifest:
    """Load one strict data-only demonstration artifact manifest."""

    source = _regular_file(Path(path), label="demonstration manifest")
    if source.stat().st_size > 64 * 1024:
        raise ValueError("demonstration manifest exceeds the 64 KiB safety limit")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("demonstration manifest is unreadable") from error
    if not isinstance(value, Mapping):
        raise TypeError("demonstration manifest must contain an object")
    return DemonstrationArtifactManifest.from_mapping(value)


def verify_demonstration_artifact(
    manifest_path: str | Path,
    *,
    gate: DemonstrationGate,
    expected_environment_id: str,
) -> VerifiedDemonstrationArtifact:
    """Verify bytes, episode structure, environment identity, and BC policy."""

    if not isinstance(gate, DemonstrationGate):
        raise TypeError("gate must be a DemonstrationGate")
    source = Path(manifest_path)
    manifest = load_demonstration_artifact_manifest(source)
    if manifest.environment_id != _environment_id(expected_environment_id):
        raise ContractViolation("demonstration environment_id does not match the expected adapter")
    trajectory = source.parent / PurePosixPath(manifest.trajectory.path)
    data = _read_trajectory_bytes(trajectory)
    if len(data) != manifest.trajectory.size_bytes:
        raise ContractViolation("demonstration trajectory size does not match the manifest")
    if _sha256(data) != manifest.trajectory.sha256:
        raise ContractViolation("demonstration trajectory sha256 does not match the manifest")
    transitions = _parse_transitions(data)
    _validate_episode(
        transitions,
        episode_id=manifest.episode_id,
        outcome=manifest.provenance.outcome,
    )
    decision = gate.validate(manifest.provenance)
    return VerifiedDemonstrationArtifact(
        manifest=manifest,
        transitions=transitions,
        sample_weight=decision.sample_weight,
    )


__all__ = [
    "DEMONSTRATION_ARTIFACT_SCHEMA_VERSION",
    "DemonstrationArtifactFile",
    "DemonstrationArtifactManifest",
    "VerifiedDemonstrationArtifact",
    "build_demonstration_artifact",
    "load_demonstration_artifact_manifest",
    "verify_demonstration_artifact",
]
