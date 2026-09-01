"""Strict project-local orchestration configuration for the GLR CLI."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

PROJECT_SCHEMA_VERSION = "glr.project.v1"
PROJECT_FILE_NAME = "glr-project.json"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_PLACEHOLDERS = frozenset(
    {
        "bridge_path",
        "bundle",
        "capture_index",
        "capture_video",
        "evaluation_path",
        "goal_path",
        "previous_evaluation_path",
        "previous_research_path",
        "project_root",
        "research_path",
        "run_dir",
        "run_id",
        "trial_path",
    }
)


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} requires string keys")
    return value


def _reject_unknown(value: Mapping[str, Any], *, allowed: frozenset[str], path: str) -> None:
    missing = sorted(allowed - set(value))
    unexpected = sorted(set(value) - allowed)
    if missing or unexpected:
        raise ValueError(f"{path} has missing={missing} and unexpected={unexpected} fields")


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must match {_IDENTIFIER.pattern!r}")
    return value


def _text(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise ValueError(f"{path} must be non-empty printable text")
    return value


def _portable_relative(value: object, *, path: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"{path} must be a portable project-relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{path} must be a portable project-relative path")
    return candidate


def _inside_project(root: Path, relative: PurePosixPath, *, path: str) -> Path:
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{path} must stay inside the project root")
    return resolved


@dataclass(frozen=True, slots=True)
class ProjectCommand:
    """One explicit argv command executed without a shell."""

    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        values = tuple(self.argv)
        if not values:
            raise ValueError("command.argv cannot be empty")
        for argument in values:
            if (
                not isinstance(argument, str)
                or not argument
                or any(ord(character) < 32 for character in argument)
            ):
                raise ValueError("command.argv entries must be non-empty printable strings")
            if "{" in argument or "}" in argument:
                if not (argument.startswith("{") and argument.endswith("}")):
                    raise ValueError("command placeholders must occupy a complete argv entry")
                placeholder = argument[1:-1]
                if placeholder not in _PLACEHOLDERS:
                    raise ValueError(f"unsupported command placeholder: {argument}")
        object.__setattr__(self, "argv", values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, path: str) -> ProjectCommand:
        _reject_unknown(value, allowed=frozenset({"argv"}), path=path)
        raw = value["argv"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise TypeError(f"{path}.argv must be an array")
        return cls(argv=tuple(raw))

    def expand(self, **values: str | Path) -> tuple[str, ...]:
        """Substitute only the fixed, whole-argument placeholders supplied by the caller."""

        normalized = MappingProxyType({key: str(value) for key, value in values.items()})
        expanded: list[str] = []
        for argument in self.argv:
            if argument.startswith("{") and argument.endswith("}"):
                placeholder = argument[1:-1]
                if placeholder not in normalized:
                    raise ValueError(f"missing command placeholder value: {placeholder}")
                expanded.append(normalized[placeholder])
            else:
                expanded.append(argument)
        return tuple(expanded)


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Project-owned recorder process and the data-grade capture outputs it must produce."""

    command: ProjectCommand
    required: bool
    stop: Literal["stdin-q", "terminate"]
    video_file: str
    index_file: str
    codec: str
    frame_rate: float
    width: int
    height: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CaptureConfig:
        _reject_unknown(
            value,
            allowed=frozenset(
                {
                    "argv",
                    "required",
                    "stop",
                    "video_file",
                    "index_file",
                    "codec",
                    "frame_rate",
                    "width",
                    "height",
                }
            ),
            path="project.capture",
        )
        required = value["required"]
        if not isinstance(required, bool):
            raise TypeError("project.capture.required must be a boolean")
        stop = value["stop"]
        if stop not in {"stdin-q", "terminate"}:
            raise ValueError("project.capture.stop must be 'stdin-q' or 'terminate'")
        frame_rate = float(value["frame_rate"])
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("project.capture.frame_rate must be positive and finite")
        width = value["width"]
        height = value["height"]
        if any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in (width, height)
        ):
            raise ValueError("project.capture width and height must be positive integers")
        codec = _identifier(value["codec"], path="project.capture.codec")
        raw_argv = value["argv"]
        if not isinstance(raw_argv, Sequence) or isinstance(raw_argv, (str, bytes, bytearray)):
            raise TypeError("project.capture.argv must be an array")
        command = ProjectCommand(argv=tuple(raw_argv))
        return cls(
            command=command,
            required=required,
            stop=stop,
            video_file=_portable_relative(
                value["video_file"], path="project.capture.video_file"
            ).as_posix(),
            index_file=_portable_relative(
                value["index_file"], path="project.capture.index_file"
            ).as_posix(),
            codec=codec,
            frame_rate=frame_rate,
            width=width,
            height=height,
        )


@dataclass(frozen=True, slots=True)
class GLRProject:
    """Resolved project contract used by local CLI orchestration."""

    root: Path
    environment_id: str
    environment_family: str
    protocol_version: str
    data_dir: Path
    bridge_path: Path
    runtime: ProjectCommand
    trainer: ProjectCommand
    player: ProjectCommand
    researcher: ProjectCommand | None
    planner: ProjectCommand | None
    evaluator: ProjectCommand | None
    capture: CaptureConfig | None
    schema_version: str = PROJECT_SCHEMA_VERSION


def find_project(start: str | Path = ".") -> Path:
    """Find the nearest project configuration without crossing the filesystem root."""

    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / PROJECT_FILE_NAME
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise FileNotFoundError(f"could not find {PROJECT_FILE_NAME} from {current}")


def load_project(path: str | Path = ".") -> GLRProject:
    """Load and resolve a strict ``glr.project.v1`` configuration."""

    requested = Path(path)
    config_path = (
        requested / PROJECT_FILE_NAME
        if requested.is_dir()
        else requested
        if requested.name == PROJECT_FILE_NAME
        else find_project(requested)
    )
    if config_path.is_symlink() or not config_path.is_file():
        raise FileNotFoundError(f"project config must be a regular non-symlink file: {config_path}")
    root = config_path.parent.resolve()
    try:
        value = _mapping(json.loads(config_path.read_text(encoding="utf-8")), path="project")
    except json.JSONDecodeError as error:
        raise ValueError("project config must be UTF-8 JSON") from error
    _reject_unknown(
        value,
        allowed=frozenset(
            {
                "schema_version",
                "environment_id",
                "environment_family",
                "protocol_version",
                "data_dir",
                "bridge_path",
                "runtime",
                "trainer",
                "player",
                "researcher",
                "planner",
                "evaluator",
                "capture",
            }
        ),
        path="project",
    )
    if value["schema_version"] != PROJECT_SCHEMA_VERSION:
        raise ValueError(f"project.schema_version must be {PROJECT_SCHEMA_VERSION!r}")
    bridge_path = _inside_project(
        root,
        _portable_relative(value["bridge_path"], path="project.bridge_path"),
        path="project.bridge_path",
    )
    if not bridge_path.exists():
        raise FileNotFoundError(f"configured bridge_path does not exist: {bridge_path}")
    data_dir = _inside_project(
        root,
        _portable_relative(value["data_dir"], path="project.data_dir"),
        path="project.data_dir",
    )
    return GLRProject(
        root=root,
        environment_id=_identifier(value["environment_id"], path="project.environment_id"),
        environment_family=_identifier(
            value["environment_family"], path="project.environment_family"
        ),
        protocol_version=_text(value["protocol_version"], path="project.protocol_version"),
        data_dir=data_dir,
        bridge_path=bridge_path,
        runtime=ProjectCommand.from_mapping(
            _mapping(value["runtime"], path="project.runtime"), path="project.runtime"
        ),
        trainer=ProjectCommand.from_mapping(
            _mapping(value["trainer"], path="project.trainer"), path="project.trainer"
        ),
        player=ProjectCommand.from_mapping(
            _mapping(value["player"], path="project.player"), path="project.player"
        ),
        researcher=(
            None
            if value["researcher"] is None
            else ProjectCommand.from_mapping(
                _mapping(value["researcher"], path="project.researcher"),
                path="project.researcher",
            )
        ),
        planner=(
            None
            if value["planner"] is None
            else ProjectCommand.from_mapping(
                _mapping(value["planner"], path="project.planner"), path="project.planner"
            )
        ),
        evaluator=(
            None
            if value["evaluator"] is None
            else ProjectCommand.from_mapping(
                _mapping(value["evaluator"], path="project.evaluator"),
                path="project.evaluator",
            )
        ),
        capture=(
            None
            if value["capture"] is None
            else CaptureConfig.from_mapping(_mapping(value["capture"], path="project.capture"))
        ),
    )


__all__ = [
    "PROJECT_FILE_NAME",
    "PROJECT_SCHEMA_VERSION",
    "CaptureConfig",
    "GLRProject",
    "ProjectCommand",
    "find_project",
    "load_project",
]
