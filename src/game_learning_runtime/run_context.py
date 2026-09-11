"""Verify invocation-scoped configuration selected by the standalone GLR CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from game_learning_runtime.project import GLRProject

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

RUN_CONTEXT_SCHEMA_VERSION = "glr.run-context.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_FILE_BYTES = 1024 * 1024
_MAX_ENV_BYTES = 24 * 1024


@dataclass(frozen=True, slots=True)
class RunContextFile:
    owner: str
    path: str
    schema_version: str
    sha256: str
    size_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "owner": self.owner,
            "path": self.path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class RunContextSource:
    path: str
    sha256: str
    size_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class RunContext:
    context_id: str
    environment_id: str
    protocol_version: str
    labels: Mapping[str, str]
    source: RunContextSource
    inputs: tuple[RunContextFile, ...]
    context_sha256: str
    schema_version: str = RUN_CONTEXT_SCHEMA_VERSION

    def unsigned_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "context_id": self.context_id,
            "environment_id": self.environment_id,
            "protocol_version": self.protocol_version,
            "labels": dict(sorted(self.labels.items())),
            "source": self.source.to_mapping(),
            "inputs": [item.to_mapping() for item in self.inputs],
        }

    def to_mapping(self) -> dict[str, object]:
        return {**self.unsigned_mapping(), "context_sha256": self.context_sha256}

    def to_json(self) -> str:
        return _compact_json(self.to_mapping())

    def verify(self, root: Path) -> None:
        _verify_identity(root, self.source.path, self.source.sha256, self.source.size_bytes)
        for item in self.inputs:
            _verify_identity(root, item.path, item.sha256, item.size_bytes)
        actual = _digest(_compact_json(self.unsigned_mapping()).encode())
        if actual != self.context_sha256:
            raise ValueError("run context identity changed in memory")


def load_run_context(
    project: GLRProject,
    path: str | Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> RunContext:
    """Load a local context deliberately or verify the CLI-inherited receipt."""

    if environment is not None:
        if path is not None:
            raise ValueError("choose a local run context path or inherited environment")
        return _from_environment(project, environment)
    if path is None:
        raise ValueError("run context path is required")
    return _from_file(project, Path(path))


def load_inherited_run_context(
    project: GLRProject, environment: Mapping[str, str] | None = None
) -> RunContext | None:
    """Return no selection when both variables are absent; reject partial receipts."""

    source = os.environ if environment is None else environment
    present = [name in source for name in ("GLR_RUN_CONTEXT", "GLR_RUN_CONTEXT_SHA256")]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("incomplete inherited run context environment")
    return _from_environment(project, source)


def _from_file(project: GLRProject, requested: Path) -> RunContext:
    source_path, source_relative = _project_file(project.root, requested)
    source_bytes = _read(source_path)
    try:
        raw = tomllib.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("run context must be valid UTF-8 TOML") from error
    value = _strict_mapping(
        raw,
        required={"schema_version", "context_id", "environment_id", "protocol_version", "inputs"},
        optional={"labels"},
        path="run context",
    )
    if value["schema_version"] != RUN_CONTEXT_SCHEMA_VERSION:
        raise ValueError(f"run context schema_version must be {RUN_CONTEXT_SCHEMA_VERSION!r}")
    context_id = _identifier(value["context_id"], "context_id")
    environment_id = _identifier(value["environment_id"], "environment_id")
    protocol_version = _text(value["protocol_version"], "protocol_version", 128)
    if environment_id != project.environment_id or protocol_version != project.protocol_version:
        raise ValueError("run context environment or protocol does not match the project")
    labels_value = value.get("labels", {})
    if not isinstance(labels_value, Mapping) or len(labels_value) > 32:
        raise ValueError("run context labels must be an object with at most 32 entries")
    labels = {
        _identifier(name, "label name"): _text(label, "label value", 256)
        for name, label in labels_value.items()
    }
    raw_inputs = value["inputs"]
    if not isinstance(raw_inputs, list) or not 1 <= len(raw_inputs) <= 64:
        raise ValueError("run context must contain 1-64 inputs")
    owners: set[str] = set()
    paths = {source_relative}
    inputs: list[RunContextFile] = []
    for index, raw_input in enumerate(raw_inputs):
        item = _strict_mapping(
            raw_input,
            required={"owner", "path", "schema_version"},
            optional=set(),
            path=f"run context inputs[{index}]",
        )
        owner = _identifier(item["owner"], "input owner")
        if owner in owners:
            raise ValueError(f"duplicate run context input owner: {owner}")
        owners.add(owner)
        expected_schema = _text(item["schema_version"], "input schema_version", 128)
        input_path, relative = _project_file(project.root, Path(item["path"]))
        if relative in paths:
            raise ValueError(f"duplicate run context input path: {relative}")
        paths.add(relative)
        data = _read(input_path)
        _verify_schema(input_path, data, expected_schema)
        inputs.append(RunContextFile(owner, relative, expected_schema, _digest(data), len(data)))
    source = RunContextSource(source_relative, _digest(source_bytes), len(source_bytes))
    frozen_inputs = tuple(inputs)
    frozen_labels = MappingProxyType(dict(sorted(labels.items())))
    provisional = RunContext(
        context_id=context_id,
        environment_id=environment_id,
        protocol_version=protocol_version,
        labels=frozen_labels,
        source=source,
        inputs=frozen_inputs,
        context_sha256="",
    )
    digest = _digest(_compact_json(provisional.unsigned_mapping()).encode())
    context = RunContext(
        context_id=context_id,
        environment_id=environment_id,
        protocol_version=protocol_version,
        labels=frozen_labels,
        source=source,
        inputs=frozen_inputs,
        context_sha256=digest,
    )
    if len(context.to_json().encode()) > _MAX_ENV_BYTES:
        raise ValueError("run context exceeds the 24 KiB role environment limit")
    return context


def _from_environment(project: GLRProject, environment: Mapping[str, str]) -> RunContext:
    try:
        raw_json = environment["GLR_RUN_CONTEXT"]
        expected_digest = environment["GLR_RUN_CONTEXT_SHA256"]
    except KeyError as error:
        raise ValueError("incomplete inherited run context environment") from error
    if len(raw_json.encode()) > _MAX_ENV_BYTES:
        raise ValueError("inherited run context exceeds 24 KiB")
    try:
        raw = json.loads(raw_json, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("inherited run context must be valid JSON") from error
    value = _strict_mapping(
        raw,
        required={
            "schema_version",
            "context_id",
            "environment_id",
            "protocol_version",
            "labels",
            "source",
            "inputs",
            "context_sha256",
        },
        optional=set(),
        path="inherited run context",
    )
    source = _strict_mapping(
        value["source"],
        required={"path", "sha256", "size_bytes"},
        optional=set(),
        path="run context source",
    )
    raw_inputs = value["inputs"]
    if not isinstance(raw_inputs, list):
        raise ValueError("run context inputs must be an array")
    inputs = tuple(_context_file(item, index) for index, item in enumerate(raw_inputs))
    if not 1 <= len(inputs) <= 64:
        raise ValueError("run context must contain 1-64 inputs")
    if len({item.owner for item in inputs}) != len(inputs):
        raise ValueError("duplicate inherited run context input owner")
    if len({item.path for item in inputs}) != len(inputs):
        raise ValueError("duplicate inherited run context input path")
    labels = value["labels"]
    if not isinstance(labels, Mapping) or any(
        not isinstance(name, str) or not isinstance(label, str) for name, label in labels.items()
    ):
        raise ValueError("run context labels must contain string keys and values")
    if len(labels) > 32:
        raise ValueError("run context labels exceed 32 entries")
    context = RunContext(
        schema_version=_text(value["schema_version"], "schema_version", 128),
        context_id=_identifier(value["context_id"], "context_id"),
        environment_id=_identifier(value["environment_id"], "environment_id"),
        protocol_version=_text(value["protocol_version"], "protocol_version", 128),
        labels=MappingProxyType(
            {
                _identifier(name, "label name"): _text(label, "label value", 256)
                for name, label in labels.items()
            }
        ),
        source=RunContextSource(
            path=_text(source["path"], "source path", 512),
            sha256=_sha256(source["sha256"], "source sha256"),
            size_bytes=_size(source["size_bytes"], "source size_bytes"),
        ),
        inputs=inputs,
        context_sha256=str(value["context_sha256"]),
    )
    if context.schema_version != RUN_CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported inherited run context schema")
    if (
        context.environment_id != project.environment_id
        or context.protocol_version != project.protocol_version
    ):
        raise ValueError("inherited run context does not match the project")
    if context.context_sha256 != expected_digest:
        raise ValueError("inherited run context digest does not match its environment receipt")
    context.verify(project.root)
    return context


def _context_file(value: object, index: int) -> RunContextFile:
    item = _strict_mapping(
        value,
        required={"owner", "path", "schema_version", "sha256", "size_bytes"},
        optional=set(),
        path=f"run context inputs[{index}]",
    )
    return RunContextFile(
        owner=_identifier(item["owner"], "input owner"),
        path=_text(item["path"], "input path", 512),
        schema_version=_text(item["schema_version"], "input schema_version", 128),
        sha256=_sha256(item["sha256"], "input sha256"),
        size_bytes=_size(item["size_bytes"], "input size_bytes"),
    )


def _project_file(root: Path, requested: Path) -> tuple[Path, str]:
    portable = PurePosixPath(requested.as_posix())
    if requested.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise ValueError("run context paths must be portable project-relative paths")
    relative = portable.as_posix()
    if len(relative.encode()) > 512:
        raise ValueError("run context path exceeds 512 bytes")
    current = root
    for part in portable.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("run context paths cannot contain links")
    resolved = current.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise FileNotFoundError(f"run context file is missing: {relative}")
    return resolved, relative


def _read(path: Path) -> bytes:
    value = path.read_bytes()
    if len(value) > _MAX_FILE_BYTES:
        raise ValueError(f"run context file exceeds 1 MiB: {path}")
    return value


def _verify_identity(root: Path, relative: str, expected: str, expected_size: int) -> None:
    path, normalized = _project_file(root, Path(relative))
    data = _read(path)
    if normalized != relative or len(data) != expected_size or _digest(data) != expected:
        raise ValueError(f"frozen run context input changed: {relative}")


def _verify_schema(path: Path, data: bytes, expected: str) -> None:
    try:
        if path.suffix == ".json":
            value = json.loads(data)
        elif path.suffix == ".toml":
            value = tomllib.loads(data.decode("utf-8"))
        else:
            raise ValueError("run context inputs must be JSON or TOML")
    except (UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid run context input: {path}") from error
    if not isinstance(value, Mapping) or value.get("schema_version") != expected:
        raise ValueError(f"run context input schema_version must be {expected!r}")


def _strict_mapping(
    value: object, *, required: set[str], optional: set[str], path: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - required - optional)
    if missing or unexpected:
        raise ValueError(f"{path} has missing={missing} unexpected={unexpected} fields")
    return value


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key in inherited run context: {key}")
        result[key] = value
    return result


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must be a portable identifier")
    return value


def _text(value: object, path: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode()) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{path} must be bounded printable text")
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _size(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= _MAX_FILE_BYTES:
        raise ValueError(f"{path} must be a bounded byte size")
    return value


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "RUN_CONTEXT_SCHEMA_VERSION",
    "RunContext",
    "RunContextFile",
    "RunContextSource",
    "load_inherited_run_context",
    "load_run_context",
]
