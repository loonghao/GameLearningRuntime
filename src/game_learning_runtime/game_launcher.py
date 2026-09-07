"""Launch configured game instances before starting a project trainer.

The launcher is deliberately a small process boundary.  It does not discover
games, inject into processes, or execute a shell command.  A project supplies
an explicit argv and an optional file-based readiness signal for each instance.
The adapter remains responsible for the authoritative bridge handshake.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

GAME_LAUNCH_SCHEMA_VERSION = "glr.game-launch.v1"
GAME_INSTANCES_SCHEMA_VERSION = "glr.game-instances.v1"
LAUNCH_OUTPUT_SCHEMA_VERSION = "glr.launch.v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDERS = frozenset(
    {"project_root", "run_dir", "instance_id", "instance_index", "instance_dir"}
)
_RESERVED_ENVIRONMENT_KEYS = frozenset(
    {
        "GLR_GAME_INSTANCE_ID",
        "GLR_GAME_INSTANCE_INDEX",
        "GLR_GAME_INSTANCE_DIR",
        "GLR_GAME_INSTANCE_COUNT",
        "GLR_GAME_PARALLEL",
        "GLR_GAME_RUN_DIR",
        "GLR_GAME_INSTANCES_MANIFEST",
    }
)


class GameLaunchError(RuntimeError):
    """A configured game or trainer could not be started safely."""


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} requires string keys")
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    path: str,
    required: frozenset[str] = frozenset(),
) -> None:
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - allowed)
    if missing or unexpected:
        raise ValueError(f"{path} has missing={missing} and unexpected={unexpected} fields")


def _portable_relative(value: object, *, path: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"{path} must be a project-relative POSIX path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{path} must be a project-relative POSIX path")
    return candidate


def _inside(root: Path, relative: PurePosixPath, *, path: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{path} must stay inside the project root")
    return resolved


def _positive_float(value: object, *, path: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0 or result > maximum:
        raise ValueError(f"{path} must be greater than zero and at most {maximum}")
    return result


@dataclass(frozen=True, slots=True)
class LaunchCommand:
    """A fixed argv command.  Shell syntax is never interpreted."""

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
                if argument[1:-1] not in _PLACEHOLDERS:
                    raise ValueError(f"unsupported command placeholder: {argument}")
        object.__setattr__(self, "argv", values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, path: str) -> LaunchCommand:
        _reject_unknown(value, allowed=frozenset({"argv"}), path=path)
        raw = value["argv"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise TypeError(f"{path}.argv must be an array")
        return cls(tuple(raw))

    def expand(self, values: Mapping[str, object]) -> tuple[str, ...]:
        expanded: list[str] = []
        for argument in self.argv:
            if argument.startswith("{") and argument.endswith("}"):
                key = argument[1:-1]
                if key not in values:
                    raise GameLaunchError(f"missing command placeholder value: {key}")
                expanded.append(str(values[key]))
            else:
                expanded.append(argument)
        return tuple(expanded)


@dataclass(frozen=True, slots=True)
class ReadinessConfig:
    """The minimum process-level signal required before training starts."""

    kind: Literal["process-alive", "file"] = "process-alive"
    path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"process-alive", "file"}:
            raise ValueError("game.readiness.kind must be 'process-alive' or 'file'")
        if self.kind == "process-alive" and self.path is not None:
            raise ValueError("game.readiness.path is only valid for file readiness")
        if self.kind == "file" and self.path is None:
            raise ValueError("file readiness requires game.readiness.path")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ReadinessConfig:
        _reject_unknown(value, allowed=frozenset({"kind", "path"}), path="game.readiness")
        kind = value.get("kind", "process-alive")
        if kind not in {"process-alive", "file"}:
            raise ValueError("game.readiness.kind must be 'process-alive' or 'file'")
        path = value.get("path")
        if kind == "file":
            path = _portable_relative(path, path="game.readiness.path").as_posix()
        elif path is not None:
            raise ValueError("game.readiness.path is only valid for file readiness")
        return cls(kind=kind, path=path)


@dataclass(frozen=True, slots=True)
class GameLaunchConfig:
    """Strict, project-owned policy for launching one or more game instances."""

    game_id: str
    command: LaunchCommand
    instances: int = 1
    parallel: bool = False
    max_parallel: int | None = None
    working_dir: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    readiness: ReadinessConfig = field(default_factory=ReadinessConfig)
    startup_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 10.0
    schema_version: str = GAME_LAUNCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.command, LaunchCommand):
            raise TypeError("game.command must be a LaunchCommand")
        if not isinstance(self.readiness, ReadinessConfig):
            raise TypeError("game.readiness must be a ReadinessConfig")
        if self.schema_version != GAME_LAUNCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported game launch schema version: {self.schema_version!r}")
        if _IDENTIFIER.fullmatch(self.game_id) is None:
            raise ValueError("game.game_id must be a lowercase identifier")
        if isinstance(self.instances, bool) or not isinstance(self.instances, int):
            raise TypeError("game.instances must be an integer")
        if self.instances < 1 or self.instances > 64:
            raise ValueError("game.instances must be between 1 and 64")
        if not isinstance(self.parallel, bool):
            raise TypeError("game.parallel must be a boolean")
        max_parallel = self.max_parallel
        if max_parallel is None:
            max_parallel = self.instances if self.parallel else 1
            object.__setattr__(self, "max_parallel", max_parallel)
        if isinstance(max_parallel, bool) or not isinstance(max_parallel, int):
            raise TypeError("game.max_parallel must be an integer")
        if max_parallel < 1 or max_parallel > self.instances:
            raise ValueError("game.max_parallel must be between 1 and game.instances")
        if not self.parallel and max_parallel != 1:
            raise ValueError("game.max_parallel must be 1 when game.parallel is false")
        if self.working_dir is not None:
            _portable_relative(self.working_dir, path="game.working_dir")
        if not isinstance(self.environment, Mapping):
            raise TypeError("game.environment must be an object")
        for key, value in self.environment.items():
            if (
                not isinstance(key, str)
                or _ENVIRONMENT_KEY.fullmatch(key) is None
                or not isinstance(value, str)
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError("game.environment keys and values must be printable strings")
            if key in _RESERVED_ENVIRONMENT_KEYS:
                raise ValueError(f"game.environment cannot override reserved key {key}")
        _positive_float(
            self.startup_timeout_seconds,
            path="game.startup_timeout_seconds",
            maximum=86_400,
        )
        _positive_float(
            self.shutdown_timeout_seconds,
            path="game.shutdown_timeout_seconds",
            maximum=600,
        )
        if self.readiness.kind == "file" and self.readiness.path is None:
            raise ValueError("file readiness requires game.readiness.path")
        if self.readiness.kind == "file" and self.readiness.path is not None:
            _portable_relative(self.readiness.path, path="game.readiness.path")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GameLaunchConfig:
        _reject_unknown(
            value,
            allowed=frozenset(
                {
                    "schema_version",
                    "game_id",
                    "command",
                    "instances",
                    "parallel",
                    "max_parallel",
                    "working_dir",
                    "environment",
                    "readiness",
                    "startup_timeout_seconds",
                    "shutdown_timeout_seconds",
                }
            ),
            path="game",
            required=frozenset({"schema_version", "game_id", "command"}),
        )
        instances = value.get("instances", 1)
        parallel = value.get("parallel", False)
        max_parallel = value.get("max_parallel", instances if parallel else 1)
        environment_value = _mapping(value.get("environment", {}), path="game.environment")
        environment = dict(environment_value)
        readiness_value = value.get("readiness", {"kind": "process-alive"})
        readiness = ReadinessConfig.from_mapping(_mapping(readiness_value, path="game.readiness"))
        return cls(
            schema_version=value.get("schema_version", GAME_LAUNCH_SCHEMA_VERSION),
            game_id=value["game_id"],
            command=LaunchCommand.from_mapping(
                _mapping(value["command"], path="game.command"), path="game.command"
            ),
            instances=instances,
            parallel=parallel,
            max_parallel=max_parallel,
            working_dir=value.get("working_dir"),
            environment=environment,
            readiness=readiness,
            startup_timeout_seconds=value.get("startup_timeout_seconds", 30.0),
            shutdown_timeout_seconds=value.get("shutdown_timeout_seconds", 10.0),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the stable JSON representation used in project files."""

        readiness: dict[str, object] = {"kind": self.readiness.kind}
        if self.readiness.path is not None:
            readiness["path"] = self.readiness.path
        return {
            "schema_version": self.schema_version,
            "game_id": self.game_id,
            "command": {"argv": list(self.command.argv)},
            "instances": self.instances,
            "parallel": self.parallel,
            "max_parallel": self.max_parallel,
            "working_dir": self.working_dir,
            "environment": dict(self.environment),
            "readiness": readiness,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
        }


def _load_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"configuration must be a regular file: {path}")
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("configuration exceeds the 8 MiB limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"configuration must be UTF-8 JSON: {path}") from error
    return _mapping(value, path="game")


def load_game_launch_config(path: str | Path) -> GameLaunchConfig:
    """Load a standalone ``glr.game-launch.v1`` document."""

    return GameLaunchConfig.from_mapping(_load_json(Path(path)))


def load_project_game_launch(path: str | Path) -> tuple[GameLaunchConfig, LaunchCommand]:
    """Load the ``game`` and ``trainer`` roles from a project document.

    The project document is intentionally only checked for the fields needed
    by this launcher.  Existing adapter/runtime fields remain owned by the
    project's normal loader.
    """

    value = _load_json(Path(path))
    if "game" not in value or "trainer" not in value:
        raise ValueError("project must contain game and trainer fields")
    return (
        GameLaunchConfig.from_mapping(_mapping(value["game"], path="project.game")),
        LaunchCommand.from_mapping(
            _mapping(value["trainer"], path="project.trainer"), path="project.trainer"
        ),
    )


@dataclass(slots=True)
class RunningGameInstance:
    """One started game process and its owned logs."""

    instance_id: str
    index: int
    directory: Path
    process: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path
    _stdout: Any = field(repr=False)
    _stderr: Any = field(repr=False)

    @property
    def pid(self) -> int:
        return self.process.pid


@dataclass(slots=True)
class RunningGameSet:
    """A lifecycle fence for all game processes belonging to one run."""

    config: GameLaunchConfig
    run_dir: Path
    instances: tuple[RunningGameInstance, ...]
    manifest_path: Path
    _closed: bool = False
    _lock: RLock = field(default_factory=RLock, repr=False)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": GAME_INSTANCES_SCHEMA_VERSION,
            "game_id": self.config.game_id,
            "run_dir": str(self.run_dir),
            "instances": [
                {
                    "id": item.instance_id,
                    "index": item.index,
                    "pid": item.pid,
                    "directory": str(item.directory),
                    "stdout": str(item.stdout_path),
                    "stderr": str(item.stderr_path),
                }
                for item in self.instances
            ],
        }

    def close(self) -> None:
        """Terminate every owned process, even when one process already exited."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            for item in reversed(self.instances):
                if item.process.poll() is None:
                    with suppress(ProcessLookupError):
                        item.process.terminate()
            deadline = time.monotonic() + self.config.shutdown_timeout_seconds
            for item in reversed(self.instances):
                if item.process.poll() is None:
                    remaining = max(0.0, deadline - time.monotonic())
                    try:
                        item.process.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            item.process.kill()
                        with suppress(subprocess.TimeoutExpired):
                            item.process.wait(timeout=1)
                with suppress(OSError):
                    item._stdout.close()
                with suppress(OSError):
                    item._stderr.close()

    def __enter__(self) -> RunningGameSet:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class GameLauncher:
    """Start configured instances in bounded batches and wait for readiness."""

    def __init__(self, config: GameLaunchConfig, *, project_root: str | Path) -> None:
        self.config = config
        self.project_root = Path(project_root).resolve()
        if not self.project_root.is_dir():
            raise FileNotFoundError(f"project root is not a directory: {self.project_root}")
        if config.working_dir is not None:
            _inside(
                self.project_root,
                _portable_relative(config.working_dir, path="game.working_dir"),
                path="game.working_dir",
            )

    def start(self, run_dir: str | Path) -> RunningGameSet:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = self.project_root / run_path
        run_path = run_path.resolve()
        if not run_path.is_relative_to(self.project_root):
            raise GameLaunchError("run_dir must stay inside the project root")
        run_path.mkdir(parents=True, exist_ok=True)
        games_root = run_path / "games"
        games_root.mkdir(exist_ok=True)
        if games_root.is_symlink() or not games_root.resolve().is_relative_to(self.project_root):
            raise GameLaunchError("run games directory must stay inside the project root")
        parallelism = self.config.max_parallel
        assert parallelism is not None
        started: list[RunningGameInstance] = []
        try:
            for first in range(0, self.config.instances, parallelism):
                batch = range(first, min(first + parallelism, self.config.instances))
                for index in batch:
                    started.append(
                        self._spawn(index=index, run_dir=run_path, games_root=games_root)
                    )
                for item in started[-len(tuple(batch)) :]:
                    self._wait_ready(item)
            manifest_path = run_path / "game-instances.json"
            result = RunningGameSet(
                config=self.config,
                run_dir=run_path,
                instances=tuple(started),
                manifest_path=manifest_path,
            )
            self._write_manifest(result)
            return result
        except Exception as error:
            temporary = RunningGameSet(
                config=self.config,
                run_dir=run_path,
                instances=tuple(started),
                manifest_path=run_path / "game-instances.json",
            )
            temporary.close()
            if isinstance(error, GameLaunchError):
                raise
            raise GameLaunchError(
                f"failed to launch game {self.config.game_id}: {error}"
            ) from error

    def _spawn(self, *, index: int, run_dir: Path, games_root: Path) -> RunningGameInstance:
        instance_id = f"{self.config.game_id}-{index:04d}"
        instance_dir = games_root / instance_id
        instance_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = instance_dir / "stdout.log"
        stderr_path = instance_dir / "stderr.log"
        stdout = stdout_path.open("wb")
        stderr = stderr_path.open("wb")
        values = {
            "project_root": self.project_root,
            "run_dir": run_dir,
            "instance_id": instance_id,
            "instance_index": index,
            "instance_dir": instance_dir,
        }
        argv = self.config.command.expand(values)
        environment = os.environ.copy()
        environment.update(self.config.environment)
        environment.update(
            {
                "GLR_GAME_INSTANCE_ID": instance_id,
                "GLR_GAME_INSTANCE_INDEX": str(index),
                "GLR_GAME_INSTANCE_DIR": str(instance_dir),
                "GLR_GAME_INSTANCE_COUNT": str(self.config.instances),
                "GLR_GAME_PARALLEL": "1" if self.config.parallel else "0",
                "GLR_GAME_RUN_DIR": str(run_dir),
            }
        )
        if self.config.readiness.kind == "file" and self.config.readiness.path is not None:
            readiness_path = instance_dir / Path(*PurePosixPath(self.config.readiness.path).parts)
            if readiness_path.is_symlink() or not readiness_path.resolve().is_relative_to(
                instance_dir.resolve()
            ):
                stdout.close()
                stderr.close()
                raise GameLaunchError(
                    f"readiness path must stay in instance directory: {readiness_path}"
                )
            with suppress(FileNotFoundError):
                readiness_path.unlink()
        try:
            process = subprocess.Popen(
                argv,
                cwd=(
                    _inside(
                        self.project_root,
                        _portable_relative(self.config.working_dir, path="game.working_dir"),
                        path="game.working_dir",
                    )
                    if self.config.working_dir is not None
                    else self.project_root
                ),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
            )
        except Exception:
            stdout.close()
            stderr.close()
            raise
        return RunningGameInstance(
            instance_id=instance_id,
            index=index,
            directory=instance_dir,
            process=process,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            _stdout=stdout,
            _stderr=stderr,
        )

    def _wait_ready(self, item: RunningGameInstance) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        readiness_path = None
        if self.config.readiness.kind == "file":
            readiness_path = item.directory / Path(
                *PurePosixPath(self.config.readiness.path or "").parts
            )
        while True:
            code = item.process.poll()
            if code is not None:
                raise GameLaunchError(
                    f"game instance {item.instance_id} exited before readiness with code {code}"
                )
            if readiness_path is None:
                return
            if not readiness_path.resolve().is_relative_to(item.directory.resolve()):
                raise GameLaunchError(
                    f"readiness path escaped instance directory: {readiness_path}"
                )
            if readiness_path.is_file() and not readiness_path.is_symlink():
                return
            if time.monotonic() >= deadline:
                raise GameLaunchError(
                    f"game instance {item.instance_id} did not publish readiness within "
                    f"{self.config.startup_timeout_seconds:.1f}s"
                )
            time.sleep(0.02)

    @staticmethod
    def _write_manifest(running: RunningGameSet) -> None:
        temporary = running.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(running.manifest(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(running.manifest_path)


@dataclass(frozen=True, slots=True)
class TrainingLaunchResult:
    """Stable result from one game-plus-trainer lifecycle."""

    return_code: int
    run_dir: Path
    manifest_path: Path
    trainer_log: Path
    instance_ids: tuple[str, ...]


class TrainingLauncher:
    """Start games, run the configured trainer, and always release games."""

    def __init__(self, config: GameLaunchConfig, *, project_root: str | Path) -> None:
        self.game_launcher = GameLauncher(config, project_root=project_root)
        self.project_root = self.game_launcher.project_root

    def run(
        self,
        trainer: LaunchCommand | Sequence[str],
        *,
        run_dir: str | Path,
        timeout_seconds: float | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> TrainingLaunchResult:
        command = trainer if isinstance(trainer, LaunchCommand) else LaunchCommand(tuple(trainer))
        if timeout_seconds is not None:
            _positive_float(timeout_seconds, path="trainer.timeout_seconds", maximum=86_400)
        with self.game_launcher.start(run_dir) as games:
            trainer_log = games.run_dir / "trainer.log"
            trainer_environment = os.environ.copy()
            if environment:
                trainer_environment.update(environment)
            trainer_environment.update(
                {
                    "GLR_PROJECT_ROOT": str(self.project_root),
                    "GLR_RUN_DIR": str(games.run_dir),
                    "GLR_GAME_ID": games.config.game_id,
                    "GLR_GAME_INSTANCE_COUNT": str(len(games.instances)),
                    "GLR_GAME_INSTANCE_IDS": ",".join(item.instance_id for item in games.instances),
                    "GLR_GAME_INSTANCES_MANIFEST": str(games.manifest_path),
                }
            )
            trainer_argv = command.expand(
                {
                    "project_root": self.project_root,
                    "run_dir": games.run_dir,
                    "instance_id": games.instances[0].instance_id,
                    "instance_index": 0,
                    "instance_dir": games.instances[0].directory,
                }
            )
            with trainer_log.open("wb") as log:
                try:
                    process = subprocess.Popen(
                        trainer_argv,
                        cwd=self.project_root,
                        env=trainer_environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        shell=False,
                    )
                except OSError as error:
                    raise GameLaunchError(f"failed to start trainer: {error}") from error
                try:
                    return_code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired as error:
                    with suppress(ProcessLookupError):
                        process.terminate()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=1)
                    raise GameLaunchError("trainer exceeded its configured timeout") from error
            return TrainingLaunchResult(
                return_code=return_code,
                run_dir=games.run_dir,
                manifest_path=games.manifest_path,
                trainer_log=trainer_log,
                instance_ids=tuple(item.instance_id for item in games.instances),
            )


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch configured game instances and train")
    parser.add_argument(
        "--project", type=Path, required=True, help="project JSON with game/trainer"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    try:
        config, trainer = load_project_game_launch(args.project)
        project_root = (args.project_root or args.project.parent).resolve()
        run_dir = args.run_dir or project_root / ".glr" / "runs" / f"run-{uuid4().hex}"
        result = TrainingLauncher(config, project_root=project_root).run(
            trainer, run_dir=run_dir, timeout_seconds=args.timeout_seconds
        )
        payload = {
            "schema_version": LAUNCH_OUTPUT_SCHEMA_VERSION,
            "status": "succeeded" if result.return_code == 0 else "failed",
            "return_code": result.return_code,
            "run_dir": str(result.run_dir),
            "manifest": str(result.manifest_path),
            "trainer_log": str(result.trainer_log),
            "instance_ids": list(result.instance_ids),
        }
        stream = sys.stdout
        stream.write(json.dumps(payload, sort_keys=True) if args.json else f"{payload}\n")
        if args.json:
            stream.write("\n")
        return result.return_code
    except (GameLaunchError, FileNotFoundError, TypeError, ValueError, OSError) as error:
        payload = {
            "schema_version": LAUNCH_OUTPUT_SCHEMA_VERSION,
            "status": "error",
            "error": str(error),
        }
        stream = sys.stderr
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        return 2


__all__ = [
    "GAME_INSTANCES_SCHEMA_VERSION",
    "GAME_LAUNCH_SCHEMA_VERSION",
    "LAUNCH_OUTPUT_SCHEMA_VERSION",
    "GameLaunchConfig",
    "GameLaunchError",
    "GameLauncher",
    "LaunchCommand",
    "ReadinessConfig",
    "RunningGameInstance",
    "RunningGameSet",
    "TrainingLaunchResult",
    "TrainingLauncher",
    "load_game_launch_config",
    "load_project_game_launch",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
