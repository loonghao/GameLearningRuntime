"""Agent-first local command line interface for Game Learning Runtime."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import IO, Any

from game_learning_runtime.agent_goal import (
    AgentGoal,
    GoalEvaluation,
    GoalEvidenceBundle,
    ResearchBundle,
    ResearchCategory,
    TrialPlan,
)
from game_learning_runtime.capture import build_capture_manifest
from game_learning_runtime.declared_metrics import summarize_declared_metrics
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.fork_gate import (
    ForkGatePolicy,
    GitRepositoryProbe,
    evaluate_fork_gate,
)
from game_learning_runtime.game_launcher import GameLauncher, GameLaunchError, LaunchCommand
from game_learning_runtime.hook_actions import register_builtin_actions
from game_learning_runtime.hooks import (
    HOOK_SCHEMA_VERSION,
    PREDEFINED_HOOK_EVENTS,
    HookConfigurationError,
    HookDispatchReport,
    HookEvent,
    HookEventStatus,
    HookRegistry,
)
from game_learning_runtime.learnability import (
    LEARNABILITY_BUDGET_SCHEMA_VERSION,
    LearnabilityReport,
)
from game_learning_runtime.model_bundle import verify_model_bundle
from game_learning_runtime.plugins import (
    PluginError,
    PluginManager,
    PluginProfile,
    PluginProfileRef,
)
from game_learning_runtime.project import (
    GLRProject,
    ProjectCommand,
    RuntimeReadinessConfig,
    load_project,
)
from game_learning_runtime.readiness import (
    ReadinessAttempt,
    ReadinessResult,
    ReadinessWindowOutcome,
    readiness_from_mapping,
    run_readiness_window,
)
from game_learning_runtime.run_store import (
    RUN_STORE_SCHEMA_VERSION,
    ArtifactRecord,
    MetricRecord,
    RouteWaypoint,
    RunEvent,
    RunRecord,
    RunStatus,
    SpatialEntity,
    SpatialRoute,
    TrainingStore,
)
from game_learning_runtime.runtime_health import RUNTIME_HEALTH_SCHEMA_VERSION
from game_learning_runtime.spatial_knowledge import SpatialKnowledgeBundle
from game_learning_runtime.supervision import SUPERVISION_SCHEMA_VERSION
from game_learning_runtime.termination import TERMINATION_SCHEMA_VERSION, EpisodeTermination
from game_learning_runtime.watchdog import (
    WATCHDOG_SCHEMA_VERSION,
    Heartbeat,
    HeartbeatLog,
    SupervisionWatchdog,
    WatchdogPolicy,
    WatchdogTarget,
)

CLI_OUTPUT_SCHEMA_VERSION = "glr.cli-output.v1"

_LOGGER = logging.getLogger("game_learning_runtime.cli")

#: Run kinds mapped to the event namespace their lifecycle hooks publish.
ROLE_EVENT_PREFIX: Mapping[str, str] = {"runtime": "runtime", "playback": "play"}

#: Canonical upstream identity enforced by `glr fork-gate`.
CANONICAL_ORIGIN_URL = "https://github.com/loonghao/GameLearningRuntime.git"

#: Canonical wire schema versions a derived checkout must keep aligned. These
#: are pinned literals rather than imports so the gate compares a fixed
#: expectation against whatever this checkout's own modules report; editing a
#: schema constant locally is then detected instead of silently matching.
FORK_GATE_SCHEMA_VERSIONS: Mapping[str, str] = {
    # `RUN_STORE_SCHEMA_VERSION` is a numeric protocol revision, not a string label.
    "run-store": "2",
    "runtime-health": "glr.runtime-health.v1",
    "process-supervision": "glr.process-supervision.v1",
    "watchdog": "glr.watchdog-report.v1",
}

#: Schema versions reported by the modules in this checkout, used as the
#: observed side of the fork gate comparison.
LOCAL_SCHEMA_VERSIONS: Mapping[str, str] = {
    "run-store": str(RUN_STORE_SCHEMA_VERSION),
    "runtime-health": RUNTIME_HEALTH_SCHEMA_VERSION,
    "process-supervision": SUPERVISION_SCHEMA_VERSION,
    "watchdog": WATCHDOG_SCHEMA_VERSION,
}
RUNTIME_NOT_READY_EXIT_CODE = 78
_MAX_READINESS_RECEIPT_BYTES = 64 * 1024


def _run_value(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "environment_id": run.environment_id,
        "protocol_version": run.protocol_version,
        "kind": run.kind,
        "status": run.status.value,
        "started_at_ns": run.started_at_ns,
        "finished_at_ns": run.finished_at_ns,
        "exit_code": run.exit_code,
        "metadata": dict(run.metadata),
    }


def _event_value(event: RunEvent) -> dict[str, Any]:
    return {
        "run_id": event.run_id,
        "sequence_id": event.sequence_id,
        "timestamp_ns": event.timestamp_ns,
        "kind": event.kind,
        "episode_id": event.episode_id,
        "step_id": event.step_id,
        "payload": dict(event.payload),
    }


def _terminations_value(terminations: Sequence[EpisodeTermination]) -> list[dict[str, Any]]:
    return [termination.to_mapping() for termination in terminations]


def _termination_summary(terminations: Sequence[EpisodeTermination]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for termination in terminations:
        reason = termination.reason.value
        counts[reason] = counts.get(reason, 0) + 1
    return {
        "schema_version": TERMINATION_SCHEMA_VERSION,
        "episode_count": len(terminations),
        "reason_counts": counts,
        "goal_reached": sum(1 for item in terminations if item.reached_goal),
        "indeterminate": sum(1 for item in terminations if item.indeterminate),
    }


def _learnability_value(report: LearnabilityReport) -> dict[str, Any]:
    return report.to_mapping()


def _learnability_summary(reports: Sequence[LearnabilityReport]) -> dict[str, Any]:
    """Project the newest verdict so a caller can gate without walking events."""

    if not reports:
        return {
            "schema_version": LEARNABILITY_BUDGET_SCHEMA_VERSION,
            "reported": False,
            "status": None,
            "state_action_cells": None,
            "coverage_ratio": None,
            "projected_steps_to_k_visits": None,
        }
    latest = reports[-1]
    return {
        "schema_version": LEARNABILITY_BUDGET_SCHEMA_VERSION,
        "reported": True,
        "verdict_count": len(reports),
        "status": latest.status.value,
        "state_action_cells": latest.state_action_cells,
        "distinct_cells_visited": latest.distinct_cells_visited,
        "coverage_ratio": latest.coverage_ratio,
        "projected_steps_to_k_visits": latest.projected_steps_to_k_visits,
        "steps_per_second": latest.steps_per_second,
        "budget_steps": latest.budget_steps,
        "min_coverage": latest.min_coverage,
    }


def _metric_value(metric: MetricRecord) -> dict[str, Any]:
    return {
        "run_id": metric.run_id,
        "metric_id": metric.metric_id,
        "timestamp_ns": metric.timestamp_ns,
        "name": metric.name,
        "value": metric.value,
        "step_id": metric.step_id,
        "metadata": dict(metric.metadata),
    }


def _artifact_value(artifact: ArtifactRecord) -> dict[str, Any]:
    return {
        "run_id": artifact.run_id,
        "path": artifact.path,
        "role": artifact.role,
        "media_type": artifact.media_type,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "metadata": dict(artifact.metadata),
    }


def _entity_value(entity: SpatialEntity) -> dict[str, Any]:
    return {
        "environment_id": entity.environment_id,
        "world_id": entity.world_id,
        "entity_id": entity.entity_id,
        "kind": entity.kind,
        "label": entity.label,
        "position": list(entity.position),
        "coordinate_frame": entity.coordinate_frame,
        "authority": entity.authority.value,
        "confidence": entity.confidence,
        "observed_at_ns": entity.observed_at_ns,
        "source_run_id": entity.source_run_id,
        "metadata": dict(entity.metadata),
    }


def _waypoint_value(waypoint: RouteWaypoint) -> dict[str, Any]:
    return {
        "index": waypoint.index,
        "position": list(waypoint.position),
        "tolerance": waypoint.tolerance,
        "label": waypoint.label,
    }


def _route_value(route: SpatialRoute) -> dict[str, Any]:
    return {
        "environment_id": route.environment_id,
        "world_id": route.world_id,
        "route_id": route.route_id,
        "name": route.name,
        "from_entity_id": route.from_entity_id,
        "to_entity_id": route.to_entity_id,
        "coordinate_frame": route.coordinate_frame,
        "confidence": route.confidence,
        "verified_at_ns": route.verified_at_ns,
        "source_run_id": route.source_run_id,
        "waypoints": [_waypoint_value(waypoint) for waypoint in route.waypoints],
        "metadata": dict(route.metadata),
        "advisory": True,
    }


def _evaluation_value(evaluation: GoalEvaluation) -> dict[str, Any]:
    return {
        "goal_id": evaluation.goal_id,
        "satisfied": evaluation.satisfied,
        "criteria": [
            {
                "metric": item.criterion.metric,
                "operator": item.criterion.operator.value,
                "target": item.criterion.target,
                "source": item.criterion.source,
                "observed": item.observed,
                "evidence_run_id": item.evidence_run_id,
                "passed": item.passed,
            }
            for item in evaluation.criteria
        ],
    }


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= 80 else f"{text[:77]}..."


def _table(rows: Sequence[Mapping[str, object]], *, columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "(no rows)"
    names = list(columns or [])
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    values = [[_cell(row.get(name)) for name in names] for row in rows]
    widths = [
        max(len(name), *(len(row[index]) for row in values)) for index, name in enumerate(names)
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header = (
        "| "
        + " | ".join(name.ljust(width) for name, width in zip(names, widths, strict=True))
        + " |"
    )
    lines = [separator, header, separator]
    lines.extend(
        "| "
        + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))
        + " |"
        for row in values
    )
    lines.append(separator)
    return "\n".join(lines)


def _human_output(command: str, data: Any) -> str:
    if isinstance(data, list):
        rows = [item if isinstance(item, Mapping) else {"value": item} for item in data]
        return f"{command}\n{_table(rows)}"
    if isinstance(data, Mapping):
        sections: list[str] = []
        for key, value in data.items():
            if isinstance(value, list):
                rows = [item if isinstance(item, Mapping) else {"value": item} for item in value]
                sections.append(f"{key}\n{_table(rows)}")
            elif isinstance(value, Mapping):
                sections.append(f"{key}\n{_table([value])}")
            else:
                sections.append(
                    _table([{"field": key, "value": value}], columns=("field", "value"))
                )
        return f"{command}\n" + "\n".join(sections)
    return f"{command}: {_cell(data)}"


def _emit(command: str, data: Any, *, as_json: bool) -> None:
    envelope = {
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "command": command,
        "data": data,
    }
    if as_json:
        print(json.dumps(envelope, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    else:
        print(_human_output(command, data))


def _store(project: GLRProject) -> TrainingStore:
    return TrainingStore(project.data_dir / "runs.sqlite3")


def _hooks(project: GLRProject, *, strict: bool) -> HookRegistry:
    """Build the project hook registry on top of the built-in actions.

    ``strict`` is used by inspection verbs, which must surface a broken hook
    configuration. Run verbs use the non-strict path: a broken hook must never
    break a training run, so the configuration is reported and ignored.
    """

    registry = HookRegistry()
    register_builtin_actions(registry, base_dir=project.root)
    try:
        registry.load(project.hooks)
    except (HookConfigurationError, TypeError, ValueError) as error:
        if strict:
            raise
        _LOGGER.warning("ignoring invalid hook configuration: %s", error)
        return HookRegistry()
    return registry


def _emit_hook_event(
    project: GLRProject,
    registry: HookRegistry,
    *,
    store: TrainingStore | None = None,
    run_id: str | None = None,
    name: str,
    status: HookEventStatus | str,
    kind: str,
    stage: str,
    exit_code: int | None = None,
    reason: str | None = None,
    payload: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    strict: bool = False,
) -> HookDispatchReport:
    """Publish one lifecycle event and persist the dispatch outcome.

    Hook bookkeeping is best effort: every failure is logged and reported
    through :class:`HookDispatchReport`, never propagated to the run that
    published the event. ``strict`` re-raises a malformed event instead, which
    is what the inspection verbs owe their caller: a synthetic event with an
    illegal name must fail the verb rather than report a quiet no-op.
    """

    try:
        event = HookEvent(
            name=name,
            status=status,
            environment_id=project.environment_id,
            environment_family=project.environment_family,
            kind=kind,
            stage=stage,
            run_id=run_id,
            exit_code=exit_code,
            reason=reason,
            payload=payload or {},
        )
    except (HookConfigurationError, TypeError, ValueError):
        if strict:
            raise
        _LOGGER.warning("hook event %s was not publishable", name)
        return HookDispatchReport(event=name, dry_run=dry_run)
    try:
        report = registry.dispatch(event, dry_run=dry_run)
        if store is not None and run_id is not None and report.results:
            store.append_event(run_id, kind="hook.dispatched", payload=report.to_mapping())
        return report
    except Exception as error:
        _LOGGER.warning("hook dispatch failed for %s: %s: %s", name, type(error).__name__, error)
        return HookDispatchReport(event=name, dry_run=dry_run)


def _command_available(project: GLRProject, command: ProjectCommand | LaunchCommand) -> bool:
    executable = command.argv[0]
    candidate = Path(executable)
    if candidate.is_absolute():
        return candidate.is_file()
    if len(candidate.parts) > 1:
        return (project.root / candidate).is_file()
    return shutil.which(executable) is not None


def _doctor(project: GLRProject, *, as_json: bool) -> int:
    roles: list[dict[str, object]] = []
    required_ready = True
    for role, command, required in (
        ("runtime", project.runtime, True),
        ("trainer", project.trainer, True),
        ("player", project.player, True),
        ("researcher", project.researcher, False),
        ("planner", project.planner, False),
        ("evaluator", project.evaluator, False),
    ):
        configured = command is not None
        available = command is not None and _command_available(project, command)
        roles.append(
            {
                "role": role,
                "required": required,
                "configured": configured,
                "executable": (None if command is None else command.argv[0]),
                "available": available,
            }
        )
        if required and not available:
            required_ready = False
    game = project.game
    if game is not None:
        available = _command_available(project, game.command)
        roles.append(
            {
                "role": "game",
                "required": True,
                "configured": True,
                "executable": game.command.argv[0],
                "available": available,
                "instances": game.instances,
                "parallel": game.parallel,
                "max_parallel": game.max_parallel,
                "readiness": game.readiness.kind,
            }
        )
        required_ready = required_ready and available
    result = {
        "ready": required_ready,
        "project_root": str(project.root),
        "project_manifest": str(project.manifest_path) if project.manifest_path else None,
        "extensions": {key: str(path) for key, path in project.extensions.items()},
        "environment_id": project.environment_id,
        "environment_family": project.environment_family,
        "roles": roles,
    }
    _emit("doctor", result, as_json=as_json)
    return 0 if required_ready else 1


def _command_context(
    project: GLRProject,
    *,
    run_id: str,
    run_dir: Path,
    bundle: Path | None = None,
    extra: dict[str, str | Path] | None = None,
) -> dict[str, str | Path]:
    return {
        "project_root": project.root,
        **({"project_manifest": project.manifest_path} if project.manifest_path else {}),
        "bridge_path": project.bridge_path,
        "run_id": run_id,
        "run_dir": run_dir,
        "capture_video": run_dir
        / (project.capture.video_file if project.capture else "capture.mp4"),
        "capture_index": run_dir
        / (project.capture.index_file if project.capture else "capture-index.jsonl"),
        **({"bundle": bundle} if bundle is not None else {}),
        **(extra or {}),
    }


def _process_environment(
    project: GLRProject,
    *,
    run_id: str,
    run_dir: Path,
    bundle: Path | None = None,
    extra: dict[str, str | Path] | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("GLR_PROJECT_MANIFEST", None)
    if project.manifest_path is not None:
        environment["GLR_PROJECT_MANIFEST"] = str(project.manifest_path)
    environment.update(
        {
            "GLR_PROJECT_ROOT": str(project.root),
            "GLR_BRIDGE_PATH": str(project.bridge_path),
            "GLR_RUN_ID": run_id,
            "GLR_RUN_DIR": str(run_dir),
            "GLR_STORE_PATH": str(project.data_dir / "runs.sqlite3"),
            "GLR_ENVIRONMENT_ID": project.environment_id,
            "GLR_ENVIRONMENT_FAMILY": project.environment_family,
            "GLR_PROTOCOL_VERSION": project.protocol_version,
            "GLR_CAPTURE_VIDEO": str(
                run_dir / (project.capture.video_file if project.capture else "capture.mp4")
            ),
            "GLR_CAPTURE_INDEX": str(
                run_dir / (project.capture.index_file if project.capture else "capture-index.jsonl")
            ),
        }
    )
    if bundle is not None:
        environment["GLR_MODEL_BUNDLE"] = str(bundle)
    for key, value in (extra or {}).items():
        environment[f"GLR_{key.upper()}"] = str(value)
    return environment


def _run_command(
    command: ProjectCommand,
    *,
    project: GLRProject,
    run_id: str,
    run_dir: Path,
    log_path: Path,
    bundle: Path | None = None,
    extra: dict[str, str | Path] | None = None,
    timeout_seconds: float | None = None,
) -> int:
    argv = command.expand(
        **_command_context(project, run_id=run_id, run_dir=run_dir, bundle=bundle, extra=extra)
    )
    environment = _process_environment(
        project, run_id=run_id, run_dir=run_dir, bundle=bundle, extra=extra
    )
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            argv,
            cwd=project.root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
            raise ContractViolation(f"project command exceeded {timeout_seconds:.1f}s") from error


def _stop_capture(process: subprocess.Popen[str], *, stop: str) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    if stop == "stdin-q" and process.stdin is not None:
        try:
            process.stdin.write("q\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
    else:
        process.terminate()
    try:
        return process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=5.0)


def _start_capture(
    project: GLRProject,
    *,
    run_id: str,
    run_dir: Path,
    extra: dict[str, str | Path] | None = None,
) -> tuple[subprocess.Popen[str], IO[str], Path] | None:
    if project.capture is None:
        return None
    capture_log = run_dir / "capture.log"
    log_stream = capture_log.open("w", encoding="utf-8", newline="\n")
    try:
        process = subprocess.Popen(
            project.capture.command.expand(
                **_command_context(project, run_id=run_id, run_dir=run_dir)
            ),
            cwd=project.root,
            env=_process_environment(project, run_id=run_id, run_dir=run_dir, extra=extra),
            stdin=(subprocess.PIPE if project.capture.stop == "stdin-q" else subprocess.DEVNULL),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )
    except BaseException:
        log_stream.close()
        raise
    return process, log_stream, capture_log


def _finish_capture(
    project: GLRProject,
    *,
    store: TrainingStore,
    run_id: str,
    capture_dir: Path,
    artifact_root: Path,
    session: tuple[subprocess.Popen[str], IO[str], Path],
) -> bool:
    if project.capture is None:
        raise AssertionError("capture configuration disappeared during a run")
    process, log_stream, capture_log = session
    capture_exit = _stop_capture(process, stop=project.capture.stop)
    log_stream.close()
    if capture_log.is_file():
        store.register_artifact(
            run_id,
            path=capture_log.relative_to(artifact_root).as_posix(),
            source=capture_log,
            role="capture-log",
            media_type="text/plain",
        )
    video = capture_dir / project.capture.video_file
    index = capture_dir / project.capture.index_file
    complete = capture_exit == 0 and video.is_file() and index.is_file()
    if not complete:
        return False
    manifest_path = capture_dir / "capture.manifest.json"
    build_capture_manifest(
        manifest_path,
        environment_id=project.environment_id,
        run_id=run_id,
        video_path=video,
        index_path=index,
        codec=project.capture.codec,
        frame_rate=project.capture.frame_rate,
        width=project.capture.width,
        height=project.capture.height,
    )
    for path, role, media_type in (
        (video, "review-video", "video/mp4"),
        (index, "capture-index", "application/x-ndjson"),
        (manifest_path, "capture-manifest", "application/json"),
    ):
        store.register_artifact(
            run_id,
            path=path.relative_to(artifact_root).as_posix(),
            source=path,
            role=role,
            media_type=media_type,
        )
    return True


def _min_coverage(value: float | None) -> float | None:
    """Validate the optional `--min-coverage` coverage floor."""

    if value is None:
        return None
    if not isinstance(value, float) or not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ContractViolation("--min-coverage must be a finite fraction in (0, 1]")
    return value


def _run_training(
    project: GLRProject,
    *,
    as_json: bool,
    capture_enabled: bool,
    min_coverage: float | None = None,
) -> int:
    store = _store(project)
    run = store.create_run(
        environment_id=project.environment_id,
        protocol_version=project.protocol_version,
        kind="training",
        metadata={
            "environment_family": project.environment_family,
            "status_scope": "process_execution",
            "learning_status": "unverified",
            "improvement_status": "unverified",
            "learnability_min_coverage": "unset" if min_coverage is None else f"{min_coverage:.6f}",
        },
    )
    run_dir = project.data_dir / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer_log = run_dir / "trainer.log"
    capture_session: tuple[subprocess.Popen[str], IO[str], Path] | None = None
    capture_complete = True
    trainer_exit = 1
    game_set = None
    game_extra: dict[str, str | Path] = {}
    hooks = _hooks(project, strict=False)
    stage = "train"
    interrupted: BaseException | None = None
    failure_error: BaseException | None = None
    _emit_hook_event(
        project,
        hooks,
        store=store,
        run_id=run.run_id,
        name="train.start",
        status=HookEventStatus.STARTED,
        kind="training",
        stage=stage,
    )
    if min_coverage is not None:
        # The trainer owns collection, so the floor travels as configuration
        # rather than as an assertion the host cannot check on its own.
        game_extra["min_coverage"] = f"{min_coverage:.6f}"

    try:
        if project.game is not None:
            stage = "game-launch"
            game_set = GameLauncher(project.game, project_root=project.root).start(run_dir)
            manifest_path = game_set.manifest_path
            # Merged, not rebound: a coverage floor recorded above has to
            # survive the game launch, or a run that looks gated in its own
            # metadata trains without the gate.
            game_extra.update(
                {
                    "game_id": project.game.game_id,
                    "game_instance_count": str(len(game_set.instances)),
                    "game_instance_ids": ",".join(item.instance_id for item in game_set.instances),
                    "game_instances_manifest": manifest_path,
                }
            )
        if capture_enabled and project.capture is not None:
            stage = "capture"
            capture_session = _start_capture(
                project, run_id=run.run_id, run_dir=run_dir, extra=game_extra
            )
            _emit_hook_event(
                project,
                hooks,
                store=store,
                run_id=run.run_id,
                name="record.start",
                status=HookEventStatus.STARTED,
                kind="record",
                stage="capture",
                payload={"video": project.capture.video_file},
            )
        stage = "trainer"
        trainer_exit = _run_command(
            project.trainer,
            project=project,
            run_id=run.run_id,
            run_dir=run_dir,
            log_path=trainer_log,
            extra=game_extra,
        )
    except KeyboardInterrupt as error:
        interrupted = error
    except BaseException as error:
        failure_error = error
    finally:
        try:
            if capture_session is not None:
                try:
                    capture_complete = _finish_capture(
                        project,
                        store=store,
                        run_id=run.run_id,
                        capture_dir=run_dir,
                        artifact_root=run_dir,
                        session=capture_session,
                    )
                    _emit_hook_event(
                        project,
                        hooks,
                        store=store,
                        run_id=run.run_id,
                        name="record.stop",
                        status=(
                            HookEventStatus.SUCCEEDED
                            if capture_complete
                            else HookEventStatus.FAILED
                        ),
                        kind="record",
                        stage="capture",
                        reason=(
                            None
                            if capture_complete
                            else "recorder did not produce the declared video and index artifacts"
                        ),
                        payload={"complete": capture_complete},
                    )
                except BaseException as error:
                    if failure_error is None and interrupted is None:
                        failure_error = error
        finally:
            if game_set is not None:
                game_set.close()

    # A run only becomes terminal after every lifecycle event has been
    # published: `append_event` refuses a terminal run, so finishing earlier
    # would silently drop the hook results for the failure that ended the run.
    if interrupted is not None:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name="train.failed",
            status=HookEventStatus.INTERRUPTED,
            kind="training",
            stage=stage,
            reason="training was interrupted",
        )
        store.finish_run(run.run_id, status=RunStatus.INTERRUPTED, exit_code=None)
        raise interrupted
    if failure_error is not None:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name="train.failed",
            status=HookEventStatus.FAILED,
            kind="training",
            stage=stage,
            exit_code=1,
            reason=f"{type(failure_error).__name__}: {failure_error}",
        )
        store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
        raise failure_error

    store.register_artifact(
        run.run_id,
        path="trainer.log",
        source=trainer_log,
        role="run-log",
        media_type="text/plain",
    )
    if game_set is not None:
        store.register_artifact(
            run.run_id,
            path="game-instances.json",
            source=game_set.manifest_path,
            role="game-instance-manifest",
            media_type="application/json",
        )
    succeeded = trainer_exit == 0 and (
        capture_complete or project.capture is None or not project.capture.required
    )
    verdicts = store.list_learnability(run.run_id)
    # A trainer that measured its coverage, found it short, and still exits zero
    # has recorded a failed verdict and then ignored it. That must not read
    # green, whatever the process said about itself.
    if succeeded and any(verdict.failed for verdict in verdicts):
        succeeded = False
    exit_code = 0 if succeeded else trainer_exit if trainer_exit != 0 else 1
    failure: dict[str, Any] | None = None
    if succeeded:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name="train.complete",
            status=HookEventStatus.SUCCEEDED,
            kind="training",
            stage="trainer",
            exit_code=exit_code,
        )
    else:
        failure_stage, failure_reason = (
            ("trainer", f"trainer command exited with code {trainer_exit}")
            if trainer_exit != 0 or project.capture is None
            else ("capture", "required capture did not produce complete review artifacts")
        )
        failure = {"stage": failure_stage, "reason": failure_reason, "exit_code": exit_code}
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name="train.failed",
            status=HookEventStatus.FAILED,
            kind="training",
            stage=failure_stage,
            exit_code=exit_code,
            reason=failure_reason,
        )
    finished = store.finish_run(
        run.run_id,
        status=RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED,
        exit_code=exit_code,
    )
    output = _run_value(finished)
    # Every run verb reports `failure` unconditionally so an agent can parse
    # one shape instead of guessing whether the key is present.
    output["failure"] = failure
    output["learnability"] = _learnability_summary(verdicts)
    _emit("train", output, as_json=as_json)
    return exit_code


def _read_readiness_receipt(path: Path) -> ReadinessResult | None:
    """Read one role-published readiness receipt, or None when unusable.

    An absent, oversized, symlinked, unreadable, or off-schema receipt is
    treated as "no receipt", which keeps the start verb fail-closed and never
    turns a role crash into a retryable host transition.
    """

    if path.is_symlink() or not path.is_file():
        return None
    try:
        if path.stat().st_size > _MAX_READINESS_RECEIPT_BYTES:
            return None
        return readiness_from_mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def _readiness_exit_code(outcome: ReadinessWindowOutcome) -> int:
    """Name the window verdict through the documented exit code.

    Only an exhausted window replaces the role's exit code, because "the host
    is still starting" is the one verdict a caller must be able to tell apart
    from a refusal or a crash without parsing logs.
    """

    if outcome.exhausted:
        return RUNTIME_NOT_READY_EXIT_CODE
    return outcome.last_attempt.exit_code


def _run_project_role(
    project: GLRProject,
    *,
    command: ProjectCommand,
    kind: str,
    output_command: str,
    as_json: bool,
    bundle: Path | None = None,
    metadata: dict[str, Any] | None = None,
    readiness: RuntimeReadinessConfig | None = None,
) -> int:
    store = _store(project)
    run = store.create_run(
        environment_id=project.environment_id,
        protocol_version=project.protocol_version,
        kind=kind,
        metadata={"environment_family": project.environment_family, **(metadata or {})},
    )
    run_dir = project.data_dir / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logs: list[Path] = []
    readiness_outcome: ReadinessWindowOutcome | None = None
    prefix = ROLE_EVENT_PREFIX.get(kind, kind)
    hooks = _hooks(project, strict=False)
    _emit_hook_event(
        project,
        hooks,
        store=store,
        run_id=run.run_id,
        name=f"{prefix}.start",
        status=HookEventStatus.STARTED,
        kind=kind,
        stage=kind,
    )

    def invoke(attempt: int) -> int:
        log_path = run_dir / (f"{kind}.log" if attempt == 1 else f"{kind}-attempt{attempt}.log")
        logs.append(log_path)
        extra: dict[str, str | Path] = {}
        if readiness is not None:
            extra = {
                "readiness_path": run_dir / readiness.file,
                "readiness_attempt": str(attempt),
            }
        return _run_command(
            command,
            project=project,
            run_id=run.run_id,
            run_dir=run_dir,
            log_path=log_path,
            bundle=bundle,
            extra=extra,
        )

    def probe(attempt: int) -> ReadinessAttempt:
        if readiness is None:
            raise AssertionError("readiness probing requires a declared window")
        receipt_path = run_dir / readiness.file
        receipt_path.unlink(missing_ok=True)
        exit_code = invoke(attempt)
        result = _read_readiness_receipt(receipt_path)
        record = ReadinessAttempt(index=attempt, exit_code=exit_code, result=result)
        store.append_event(run.run_id, kind="readiness.attempt", payload=record.to_mapping())
        return record

    try:
        if readiness is None:
            exit_code = invoke(1)
        else:
            readiness_outcome = run_readiness_window(
                timeout_seconds=readiness.timeout_seconds,
                poll_interval_seconds=readiness.poll_interval_seconds,
                attempt=probe,
            )
            store.append_event(
                run.run_id, kind="readiness.outcome", payload=readiness_outcome.to_mapping()
            )
            exit_code = _readiness_exit_code(readiness_outcome)
    except KeyboardInterrupt:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name=f"{prefix}.failed",
            status=HookEventStatus.INTERRUPTED,
            kind=kind,
            stage=kind,
            reason=f"{kind} was interrupted",
        )
        store.finish_run(run.run_id, status=RunStatus.INTERRUPTED, exit_code=None)
        raise
    except BaseException as error:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name=f"{prefix}.failed",
            status=HookEventStatus.FAILED,
            kind=kind,
            stage=kind,
            exit_code=1,
            reason=f"{type(error).__name__}: {error}",
        )
        store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
        raise
    for log_path in logs:
        store.register_artifact(
            run.run_id,
            path=log_path.name,
            source=log_path,
            role="run-log",
            media_type="text/plain",
        )
    failure: dict[str, Any] | None = None
    if exit_code == 0:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name=f"{prefix}.complete",
            status=HookEventStatus.SUCCEEDED,
            kind=kind,
            stage=kind,
            exit_code=exit_code,
        )
    else:
        failure = {
            "stage": kind,
            "reason": f"{kind} command exited with code {exit_code}",
            "exit_code": exit_code,
        }
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name=f"{prefix}.failed",
            status=HookEventStatus.FAILED,
            kind=kind,
            stage=kind,
            exit_code=exit_code,
            reason=failure["reason"],
        )
    finished = store.finish_run(
        run.run_id,
        status=RunStatus.SUCCEEDED if exit_code == 0 else RunStatus.FAILED,
        exit_code=exit_code,
    )
    output = _run_value(finished)
    output["failure"] = failure
    if readiness_outcome is not None:
        output["readiness"] = readiness_outcome.to_mapping()
    _emit(output_command, output, as_json=as_json)
    return exit_code


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink JSON file: {path}")
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError(f"{label} exceeds the 8 MiB limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must contain an object with string keys")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise ContractViolation("goal wall-clock budget was exhausted")
    return remaining


def _run_goal_role(
    project: GLRProject,
    *,
    command: ProjectCommand,
    role: str,
    run_id: str,
    role_dir: Path,
    context: dict[str, str | Path],
    deadline: float,
) -> Path:
    log_path = role_dir / f"{role}.log"
    exit_code = _run_command(
        command,
        project=project,
        run_id=run_id,
        run_dir=role_dir,
        log_path=log_path,
        extra=context,
        timeout_seconds=_remaining_seconds(deadline),
    )
    if exit_code != 0:
        raise ContractViolation(f"goal {role} command failed with exit code {exit_code}")
    return log_path


def _require_persisted_goal_evidence(
    store: TrainingStore,
    *,
    run_id: str,
    bundle: GoalEvidenceBundle,
    after_metric_id: int,
) -> None:
    """Bind evaluator claims to metrics persisted by the project runtime/trainer."""

    for evidence in bundle.evidence:
        if evidence.run_id != run_id:
            raise ContractViolation("goal evidence run_id does not match the active goal run")
        matched = store.has_metric_evidence(
            run_id,
            after_metric_id=after_metric_id,
            name=evidence.metric,
            value=evidence.value,
            source=evidence.source,
            authority=evidence.authority,
        )
        if not matched:
            raise ContractViolation(
                f"goal evidence {evidence.metric!r} is not backed by persisted runtime metrics"
            )


def _register_goal_artifact(
    store: TrainingStore,
    *,
    run_id: str,
    run_dir: Path,
    path: Path,
    role: str,
    media_type: str,
) -> None:
    store.register_artifact(
        run_id,
        path=path.relative_to(run_dir).as_posix(),
        source=path,
        role=role,
        media_type=media_type,
    )


def _validate_goal_research(
    goal: AgentGoal,
    research: ResearchBundle,
    *,
    seen_source_ids: set[str],
) -> None:
    all_source_ids = seen_source_ids | {source.source_id for source in research.sources}
    if len(all_source_ids) > goal.budget.max_research_sources:
        raise ContractViolation("research cycles exceed goal max_research_sources")
    disallowed = sorted(
        {
            source.media_type.value
            for source in research.sources
            if source.media_type not in goal.allowed_research_media
        }
    )
    if disallowed:
        raise ContractViolation(f"research bundle uses disallowed media types: {disallowed}")
    seen_source_ids.update(all_source_ids)


def _run_goal(project: GLRProject, *, goal_path: Path, as_json: bool, capture_enabled: bool) -> int:
    if project.researcher is None or project.planner is None or project.evaluator is None:
        raise ContractViolation(
            "goal run requires project researcher, planner, trainer, and evaluator commands"
        )
    goal = AgentGoal.from_mapping(_read_json_mapping(goal_path, label="agent goal"))
    if goal.environment_family != project.environment_family:
        raise ContractViolation("goal environment_family does not match the current GLR project")

    store = _store(project)
    run = store.create_run(
        environment_id=project.environment_id,
        protocol_version=project.protocol_version,
        kind="goal",
        metadata={
            "environment_family": project.environment_family,
            "goal_id": goal.goal_id,
            "objective": goal.objective,
        },
    )
    run_dir = project.data_dir / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    canonical_goal_path = run_dir / "goal.json"
    research_path = run_dir / "research.json"
    _write_json(canonical_goal_path, goal.to_mapping())
    deadline = monotonic() + goal.budget.max_wall_seconds
    total_steps = 0
    trials_completed = 0
    last_evaluation: GoalEvaluation | None = None
    seen_source_ids: set[str] = set()
    known_finding_ids: set[str] = set()
    previous_evaluation_path: Path | None = None
    hooks = _hooks(project, strict=False)
    _emit_hook_event(
        project,
        hooks,
        store=store,
        run_id=run.run_id,
        name="goal.start",
        status=HookEventStatus.STARTED,
        kind="goal",
        stage="goal",
        payload={"goal_id": goal.goal_id},
    )
    try:
        research_log = _run_goal_role(
            project,
            command=project.researcher,
            role="researcher",
            run_id=run.run_id,
            role_dir=run_dir,
            context={"goal_path": canonical_goal_path, "research_path": research_path},
            deadline=deadline,
        )
        research = ResearchBundle.from_mapping(
            _read_json_mapping(research_path, label="research bundle")
        )
        _validate_goal_research(goal, research, seen_source_ids=seen_source_ids)
        store.upsert_research_bundle(research)
        known_finding_ids.update(finding.finding_id for finding in research.findings)
        store.append_event(
            run.run_id,
            kind="research.completed",
            payload={"sources": len(research.sources), "findings": len(research.findings)},
        )
        for path, role, media_type in (
            (canonical_goal_path, "goal", "application/json"),
            (research_path, "research", "application/json"),
            (research_log, "run-log", "text/plain"),
        ):
            _register_goal_artifact(
                store,
                run_id=run.run_id,
                run_dir=run_dir,
                path=path,
                role=role,
                media_type=media_type,
            )

        active_research_path = research_path
        for trial_number in range(1, goal.budget.max_trials + 1):
            _remaining_seconds(deadline)
            trial_id = f"trial-{trial_number}"
            trial_dir = run_dir / "trials" / trial_id
            trial_dir.mkdir(parents=True, exist_ok=False)
            trial_path = trial_dir / "plan.json"
            evaluation_path = trial_dir / "evaluation.json"
            if trial_number > 1:
                refreshed_research_path = trial_dir / "research.json"
                research_context: dict[str, str | Path] = {
                    "goal_path": canonical_goal_path,
                    "research_path": refreshed_research_path,
                    "previous_research_path": active_research_path,
                }
                if previous_evaluation_path is not None:
                    research_context["previous_evaluation_path"] = previous_evaluation_path
                refreshed_log = _run_goal_role(
                    project,
                    command=project.researcher,
                    role="researcher",
                    run_id=run.run_id,
                    role_dir=trial_dir,
                    context=research_context,
                    deadline=deadline,
                )
                research = ResearchBundle.from_mapping(
                    _read_json_mapping(refreshed_research_path, label="research bundle")
                )
                _validate_goal_research(goal, research, seen_source_ids=seen_source_ids)
                store.upsert_research_bundle(research)
                known_finding_ids.update(finding.finding_id for finding in research.findings)
                active_research_path = refreshed_research_path
                store.append_event(
                    run.run_id,
                    kind="research.refreshed",
                    payload={
                        "trial_id": trial_id,
                        "sources_seen": len(seen_source_ids),
                        "findings": len(research.findings),
                    },
                )
                for path, role, media_type in (
                    (refreshed_research_path, "research", "application/json"),
                    (refreshed_log, "run-log", "text/plain"),
                ):
                    _register_goal_artifact(
                        store,
                        run_id=run.run_id,
                        run_dir=run_dir,
                        path=path,
                        role=role,
                        media_type=media_type,
                    )
            context: dict[str, str | Path] = {
                "goal_path": canonical_goal_path,
                "research_path": active_research_path,
                "trial_path": trial_path,
                "evaluation_path": evaluation_path,
                "trial_id": trial_id,
            }
            if previous_evaluation_path is not None:
                context["previous_evaluation_path"] = previous_evaluation_path
            planner_log = _run_goal_role(
                project,
                command=project.planner,
                role="planner",
                run_id=run.run_id,
                role_dir=trial_dir,
                context=context,
                deadline=deadline,
            )
            trial = TrialPlan.from_mapping(_read_json_mapping(trial_path, label="trial plan"))
            if trial.goal_id != goal.goal_id or trial.trial_id != trial_id:
                raise ContractViolation(
                    "trial plan goal_id or trial_id does not match control state"
                )
            if total_steps + trial.max_steps > goal.budget.max_training_steps:
                raise ContractViolation("trial plan exceeds the remaining training-step budget")
            referenced_findings = {
                finding_id for term in trial.reward_terms for finding_id in term.source_finding_ids
            }
            missing_findings = sorted(referenced_findings - known_finding_ids)
            if missing_findings:
                raise ContractViolation(
                    f"trial reward terms reference unknown research findings: {missing_findings}"
                )
            total_steps += trial.max_steps
            store.append_event(
                run.run_id,
                kind="trial.planned",
                payload={
                    "trial_id": trial_id,
                    "max_steps": trial.max_steps,
                    "reward_terms": [term.name for term in trial.reward_terms],
                },
            )
            capture_session = (
                _start_capture(project, run_id=run.run_id, run_dir=trial_dir)
                if capture_enabled and project.capture is not None
                else None
            )
            metric_floor = store.latest_metric_id(run.run_id)
            try:
                trainer_log = _run_goal_role(
                    project,
                    command=project.trainer,
                    role="trainer",
                    run_id=run.run_id,
                    role_dir=trial_dir,
                    context=context,
                    deadline=deadline,
                )
            finally:
                capture_complete = (
                    True
                    if capture_session is None
                    else _finish_capture(
                        project,
                        store=store,
                        run_id=run.run_id,
                        capture_dir=trial_dir,
                        artifact_root=run_dir,
                        session=capture_session,
                    )
                )
            if not capture_complete and project.capture is not None and project.capture.required:
                raise ContractViolation(f"required capture failed for {trial_id}")
            evaluator_log = _run_goal_role(
                project,
                command=project.evaluator,
                role="evaluator",
                run_id=run.run_id,
                role_dir=trial_dir,
                context=context,
                deadline=deadline,
            )
            evidence = GoalEvidenceBundle.from_mapping(
                _read_json_mapping(evaluation_path, label="goal evidence")
            )
            if evidence.goal_id != goal.goal_id or evidence.trial_id != trial_id:
                raise ContractViolation(
                    "goal evidence goal_id or trial_id does not match control state"
                )
            _require_persisted_goal_evidence(
                store,
                run_id=run.run_id,
                bundle=evidence,
                after_metric_id=metric_floor,
            )
            last_evaluation = goal.evaluate(evidence.evidence)
            trials_completed = trial_number
            previous_evaluation_path = evaluation_path
            store.append_event(
                run.run_id,
                kind="trial.evaluated",
                payload={
                    "trial_id": trial_id,
                    "satisfied": last_evaluation.satisfied,
                    "criteria": _evaluation_value(last_evaluation)["criteria"],
                },
            )
            for path, role, media_type in (
                (trial_path, "trial-plan", "application/json"),
                (evaluation_path, "goal-evidence", "application/json"),
                (planner_log, "run-log", "text/plain"),
                (trainer_log, "run-log", "text/plain"),
                (evaluator_log, "run-log", "text/plain"),
            ):
                _register_goal_artifact(
                    store,
                    run_id=run.run_id,
                    run_dir=run_dir,
                    path=path,
                    role=role,
                    media_type=media_type,
                )
            if last_evaluation.satisfied:
                break
    except KeyboardInterrupt:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name="goal.failed",
            status=HookEventStatus.INTERRUPTED,
            kind="goal",
            stage="goal",
            reason="goal run was interrupted",
            payload={"goal_id": goal.goal_id, "trials_completed": trials_completed},
        )
        store.finish_run(run.run_id, status=RunStatus.INTERRUPTED, exit_code=None)
        raise
    except BaseException as error:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name="goal.failed",
            status=HookEventStatus.FAILED,
            kind="goal",
            stage="goal",
            exit_code=1,
            reason=f"{type(error).__name__}: {error}",
            payload={"goal_id": goal.goal_id, "trials_completed": trials_completed},
        )
        store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
        raise

    satisfied = last_evaluation is not None and last_evaluation.satisfied
    exit_code = 0 if satisfied else 3
    failure: dict[str, Any] | None = None
    if satisfied:
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name="goal.complete",
            status=HookEventStatus.SUCCEEDED,
            kind="goal",
            stage="goal",
            exit_code=exit_code,
            payload={"goal_id": goal.goal_id, "trials_completed": trials_completed},
        )
    else:
        failure = {
            "stage": "goal",
            "reason": "goal success criteria were not satisfied within the declared budget",
            "exit_code": exit_code,
        }
        _emit_hook_event(
            project,
            hooks,
            store=store,
            run_id=run.run_id,
            name="goal.failed",
            status=HookEventStatus.FAILED,
            kind="goal",
            stage="goal",
            exit_code=exit_code,
            reason=failure["reason"],
            payload={"goal_id": goal.goal_id, "trials_completed": trials_completed},
        )
    finished = store.finish_run(
        run.run_id,
        status=RunStatus.SUCCEEDED if satisfied else RunStatus.FAILED,
        exit_code=exit_code,
    )
    _emit(
        "goal.run",
        {
            "run": _run_value(finished),
            "goal_id": goal.goal_id,
            "satisfied": satisfied,
            "trials_completed": trials_completed,
            "training_steps_planned": total_steps,
            "evaluation": (None if last_evaluation is None else _evaluation_value(last_evaluation)),
            "failure": failure,
        },
        as_json=as_json,
    )
    return exit_code


def _export_knowledge(project: GLRProject, *, output: Path, as_json: bool) -> int:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"knowledge export output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = SpatialKnowledgeBundle.from_store(
        _store(project),
        environment_id=project.environment_id,
        protocol_version=project.protocol_version,
    )
    _write_json(output, bundle.to_mapping())
    _emit(
        "knowledge.export",
        {
            "path": str(output),
            "environment_id": bundle.environment_id,
            "entities": len(bundle.entities),
            "routes": len(bundle.routes),
        },
        as_json=as_json,
    )
    return 0


def _import_knowledge(project: GLRProject, *, source: Path, as_json: bool) -> int:
    bundle = SpatialKnowledgeBundle.from_mapping(
        _read_json_mapping(source, label="spatial knowledge")
    )
    store = _store(project)
    run = store.create_run(
        environment_id=project.environment_id,
        protocol_version=project.protocol_version,
        kind="knowledge-import",
        metadata={"source": source.name},
    )
    run_dir = project.data_dir / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    imported_path = run_dir / "spatial-knowledge.json"
    try:
        _write_json(imported_path, bundle.to_mapping())
        entities, routes = bundle.import_into(
            store,
            environment_id=project.environment_id,
            protocol_version=project.protocol_version,
            source_run_id=run.run_id,
        )
        store.append_event(
            run.run_id,
            kind="knowledge.imported",
            payload={"entities": entities, "routes": routes, "advisory": True},
        )
        store.register_artifact(
            run.run_id,
            path=imported_path.name,
            source=imported_path,
            role="spatial-knowledge",
            media_type="application/json",
        )
    except BaseException:
        store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
        raise
    finished = store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
    _emit(
        "knowledge.import",
        {
            "run": _run_value(finished),
            "entities": entities,
            "routes": routes,
            "authority": "advisory",
        },
        as_json=as_json,
    )
    return 0


def _plugin_project_root(requested: str | Path) -> Path:
    """Resolve a plugin store root without requiring role executables.

    Plugin management is a control-plane operation.  It must remain usable in
    a newly scaffolded project whose runtime roles are not installed yet, so it
    intentionally does not call :func:`load_project`.
    """

    path = Path(requested).absolute()
    if path.is_file():
        return path.parent
    return path


def _plugin_manager(requested: str | Path) -> PluginManager:
    return PluginManager(_plugin_project_root(requested))


def _plugin_ref_mapping(ref: PluginProfileRef) -> PluginProfileRef:
    """Return a validated copy so profile edits cannot retain mutable state."""

    return PluginProfileRef(
        plugin_id=ref.plugin_id,
        version=ref.version,
        enabled=ref.enabled,
        permissions=ref.permissions,
        config=ref.config,
    )


def _run_plugin_command(arguments: argparse.Namespace) -> int:
    manager = _plugin_manager(arguments.project)
    command = arguments.plugin_command
    if command == "inspect":
        inspection = manager.inspect(Path(arguments.source))
        _emit("plugin.inspect", inspection.to_mapping(), as_json=arguments.json)
        return 0
    if command == "install":
        installation = manager.install(Path(arguments.source), expected_sha256=arguments.sha256)
        _emit("plugin.install", installation.to_mapping(), as_json=arguments.json)
        return 0
    if command == "list":
        _emit(
            "plugin.list",
            [item.to_mapping() for item in manager.list_installed()],
            as_json=arguments.json,
        )
        return 0
    if command == "health":
        health = manager.health(arguments.profile)
        _emit("plugin.health", list(health), as_json=arguments.json)
        return 0
    if command == "remove":
        manager.remove(arguments.plugin_id, arguments.version)
        _emit(
            "plugin.remove",
            {"id": arguments.plugin_id, "version": arguments.version},
            as_json=arguments.json,
        )
        return 0
    if command == "profile":
        profile_command = arguments.profile_command
        if profile_command == "list":
            _emit(
                "plugin.profile.list",
                [profile.to_mapping() for profile in manager.list_profiles()],
                as_json=arguments.json,
            )
            return 0
        if profile_command in {"show", "resolve"}:
            profile = manager.load_profile(arguments.name)
            if profile_command == "show":
                data: Mapping[str, Any] = {
                    **profile.to_mapping(),
                    "digest": profile.digest(),
                }
                output_command = "plugin.profile.show"
            else:
                data = manager.resolve_profile(profile).to_mapping()
                output_command = "plugin.profile.resolve"
            _emit(output_command, data, as_json=arguments.json)
            return 0
        if profile_command in {"enable", "disable"}:
            profile_path = manager.profile_root / f"{arguments.name}.json"
            if not profile_path.exists() and not profile_path.is_symlink():
                if profile_command == "disable":
                    raise FileNotFoundError(profile_path)
                profile = PluginProfile(name=arguments.name)
            else:
                profile = manager.load_profile(arguments.name)
            entries = list(profile.plugins)
            matching = next(
                (
                    index
                    for index, item in enumerate(entries)
                    if item.plugin_id == arguments.plugin_id
                ),
                None,
            )
            if profile_command == "enable":
                if matching is None:
                    entries.append(
                        PluginProfileRef(
                            plugin_id=arguments.plugin_id,
                            version=arguments.version or "*",
                            permissions=tuple(arguments.grant or ()),
                        )
                    )
                else:
                    current = entries[matching]
                    entries[matching] = PluginProfileRef(
                        plugin_id=current.plugin_id,
                        version=arguments.version or current.version,
                        enabled=True,
                        permissions=(
                            tuple(arguments.grant)
                            if arguments.grant is not None
                            else current.permissions
                        ),
                        config=current.config,
                    )
            elif matching is None:
                raise PluginError(
                    f"plugin {arguments.plugin_id!r} is not present in profile {arguments.name!r}"
                )
            else:
                current = entries[matching]
                entries[matching] = _plugin_ref_mapping(
                    PluginProfileRef(
                        plugin_id=current.plugin_id,
                        version=current.version,
                        enabled=False,
                        permissions=current.permissions,
                        config=current.config,
                    )
                )
            updated = PluginProfile(name=profile.name, plugins=tuple(entries))
            manager.save_profile(updated)
            _emit(
                f"plugin.profile.{profile_command}",
                updated.to_mapping(),
                as_json=arguments.json,
            )
            return 0
    raise AssertionError("unreachable plugin command")


def _run_hooks_command(project: GLRProject, arguments: argparse.Namespace, *, as_json: bool) -> int:
    """Inspect or exercise the configured lifecycle hooks."""

    registry = _hooks(project, strict=True)
    if arguments.hooks_command == "list":
        _emit(
            "hooks.list",
            {
                "schema_version": HOOK_SCHEMA_VERSION,
                "enabled": project.hooks.enabled,
                "default_timeout_seconds": project.hooks.default_timeout_seconds,
                "actions": list(registry.action_names),
                "subscriptions": [item.to_mapping() for item in registry.subscriptions],
                "predefined_events": list(PREDEFINED_HOOK_EVENTS),
            },
            as_json=as_json,
        )
        return 0
    if arguments.hooks_command == "emit":
        report = _emit_hook_event(
            project,
            registry,
            name=arguments.event,
            status=arguments.status,
            kind=arguments.kind,
            stage=arguments.stage,
            exit_code=arguments.exit_code,
            reason=arguments.reason,
            dry_run=arguments.dry_run,
            strict=True,
        )
        _emit("hooks.emit", report.to_mapping(), as_json=as_json)
        return 0 if report.ok else 1
    raise AssertionError("unreachable hooks command")


def _fork_gate(arguments: argparse.Namespace, *, as_json: bool) -> int:
    """Evaluate the anti-fork drift gate for one checkout."""

    target = Path(arguments.project).resolve()
    root = target.parent if target.is_file() else target
    policy = ForkGatePolicy(
        expected_origin_url=arguments.origin,
        default_branch=arguments.default_branch,
        max_commits_behind=arguments.max_behind,
        max_commits_ahead=arguments.max_ahead,
        require_origin_match=not arguments.allow_foreign_origin,
        require_version_alignment=not arguments.allow_version_drift,
        require_upstream_ref=not arguments.allow_missing_upstream,
        required_schema_versions=(
            None if arguments.ignore_schema_versions else dict(FORK_GATE_SCHEMA_VERSIONS)
        ),
    )
    probe = GitRepositoryProbe(root, schema_versions=dict(LOCAL_SCHEMA_VERSIONS))
    report = evaluate_fork_gate(probe, policy)
    _emit("fork-gate", report.to_mapping(), as_json=as_json)
    return report.exit_code


def _watchdog_tick(arguments: argparse.Namespace, *, as_json: bool) -> int:
    """Run one scheduler-friendly supervision pass and report the outcome."""

    policy = WatchdogPolicy(
        heartbeat_timeout_seconds=arguments.timeout,
        max_missed_heartbeats=arguments.max_missed,
        restart_attempt_limit=arguments.restart_attempt_limit,
    )
    latest: dict[str, Heartbeat] = {}
    if arguments.heartbeats:
        latest = HeartbeatLog(Path(arguments.heartbeats).resolve()).latest_by_source()
    sources: list[str] = list(arguments.source)
    for name in sorted(latest):
        if name not in sources:
            sources.append(name)
    if not sources:
        raise ContractViolation(
            "watchdog tick requires at least one --source or a readable --heartbeats log"
        )
    recovery = tuple(arguments.recovery_command) if arguments.recovery_command else None
    watchdog = SupervisionWatchdog(policy=policy)
    for name in sources:
        watchdog.register(WatchdogTarget(name=name, policy=policy, recovery_command=recovery))
    watchdog.observe_all(tuple(latest.values()))
    if arguments.interval is None:
        report = watchdog.tick()
    else:
        report = watchdog.run(interval_seconds=arguments.interval, max_ticks=arguments.max_ticks)
    _emit("watchdog.tick", report.to_mapping(), as_json=as_json)
    return report.exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glr", description="Game Learning Runtime control plane")
    parser.add_argument("--project", default=".", help="project root or glr-project.json")
    parser.add_argument("--json", action="store_true", help="emit compact stable JSON")
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="human table output (default) or the stable JSON envelope",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="validate configured roles and game launch readiness")

    fork_gate = commands.add_parser(
        "fork-gate", help="verify this checkout still tracks the canonical upstream"
    )
    fork_gate.add_argument("--origin", default=CANONICAL_ORIGIN_URL, help="canonical origin URL")
    fork_gate.add_argument("--default-branch", default="main", help="canonical default branch")
    fork_gate.add_argument("--max-behind", type=int, default=50, help="allowed commits behind")
    fork_gate.add_argument("--max-ahead", type=int, default=200, help="allowed commits ahead")
    fork_gate.add_argument(
        "--allow-foreign-origin",
        action="store_true",
        help="report a non-canonical origin without failing the gate",
    )
    fork_gate.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="report version misalignment without failing the gate",
    )
    fork_gate.add_argument(
        "--allow-missing-upstream",
        action="store_true",
        help="report an unfetched upstream ref without failing the gate",
    )
    fork_gate.add_argument(
        "--ignore-schema-versions",
        action="store_true",
        help="skip required wire schema version checks",
    )

    watchdog = commands.add_parser(
        "watchdog", help="evaluate supervision heartbeats and recover bounded failures"
    )
    watchdog_commands = watchdog.add_subparsers(dest="watchdog_command", required=True)
    watchdog_tick = watchdog_commands.add_parser(
        "tick", help="run one scheduler-friendly supervision pass"
    )
    watchdog_tick.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME",
        help="supervised source name; repeatable",
    )
    watchdog_tick.add_argument(
        "--heartbeats", help="JSON Lines heartbeat log to read the newest beat per source"
    )
    watchdog_tick.add_argument(
        "--timeout", type=float, default=30.0, help="seconds before a heartbeat is late"
    )
    watchdog_tick.add_argument(
        "--max-missed", type=int, default=3, help="late intervals tolerated before starvation"
    )
    watchdog_tick.add_argument(
        "--restart-attempt-limit",
        type=int,
        default=3,
        help="total automatic recovery attempts allowed per source, success or failure",
    )
    watchdog_tick.add_argument(
        "--recovery-command",
        action="append",
        default=[],
        metavar="TOKEN",
        help="recovery argv tokens, e.g. --recovery-command glr --recovery-command train",
    )
    watchdog_tick.add_argument(
        "--interval", type=float, help="run repeatedly with this many seconds between passes"
    )
    watchdog_tick.add_argument(
        "--max-ticks", type=int, help="stop after this many passes in interval mode"
    )

    runs = commands.add_parser("runs", help="query persisted runtime and training runs")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    run_list = run_commands.add_parser("list")
    run_list.add_argument("--status", choices=[status.value for status in RunStatus])
    run_list.add_argument("--limit", type=int, default=100)
    run_show = run_commands.add_parser("show")
    run_show.add_argument("run_id")

    query = commands.add_parser("query", help="query learned and observed experience")
    query_commands = query.add_subparsers(dest="query_command", required=True)
    entities = query_commands.add_parser("entities")
    entities.add_argument("--world", required=True)
    entities.add_argument("--kind")
    entities.add_argument("--name")
    entities.add_argument("--near", nargs=3, type=float, metavar=("X", "Y", "Z"))
    entities.add_argument("--radius", type=float)
    entities.add_argument("--limit", type=int, default=100)
    routes = query_commands.add_parser("routes")
    routes.add_argument("--world", required=True)
    routes.add_argument("--from-entity")
    routes.add_argument("--to-entity")
    routes.add_argument("--limit", type=int, default=100)
    research = query_commands.add_parser("research")
    research.add_argument("--tag", action="append", default=[])
    research.add_argument("--category", choices=[category.value for category in ResearchCategory])
    research.add_argument("--verified-only", action="store_true")
    research.add_argument("--limit", type=int, default=100)
    train = commands.add_parser("train", help="run the project trainer and persist its evidence")
    train.add_argument("--no-capture", action="store_true")
    train.add_argument(
        "--min-coverage",
        type=float,
        default=None,
        metavar="FRACTION",
        help=(
            "fail the run when its learnability coverage stays below this "
            "fraction of the declared state-action space"
        ),
    )
    runtime = commands.add_parser("runtime", help="start the configured game/runtime bridge")
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_commands.add_parser(
        "start",
        help=(
            "start the runtime role and, when project.runtime.readiness is declared, "
            "re-invoke it until the host reports ready"
        ),
    )
    play = commands.add_parser("play", help="verify and load a trained model bundle")
    play.add_argument("--bundle", required=True)
    goal = commands.add_parser("goal", help="run a bounded agent-first learning objective")
    goal_commands = goal.add_subparsers(dest="goal_command", required=True)
    goal_run = goal_commands.add_parser("run")
    goal_run.add_argument("--goal", required=True)
    goal_run.add_argument("--no-capture", action="store_true")
    knowledge = commands.add_parser("knowledge", help="move exact-environment spatial knowledge")
    knowledge_commands = knowledge.add_subparsers(dest="knowledge_command", required=True)
    knowledge_export = knowledge_commands.add_parser("export")
    knowledge_export.add_argument("--output", required=True)
    knowledge_import = knowledge_commands.add_parser("import")
    knowledge_import.add_argument("--input", required=True)

    hooks = commands.add_parser(
        "hooks", help="inspect and exercise lifecycle hooks without running a job"
    )
    hooks_commands = hooks.add_subparsers(dest="hooks_command", required=True)
    hooks_commands.add_parser("list", help="list registered hook actions and subscriptions")
    hooks_emit = hooks_commands.add_parser(
        "emit", help="publish one lifecycle event to every matching hook"
    )
    hooks_emit.add_argument("--event", required=True, help="event name, e.g. train.failed")
    hooks_emit.add_argument(
        "--status", choices=[item.value for item in HookEventStatus], default="started"
    )
    hooks_emit.add_argument("--kind", default="manual", help="run dimension, e.g. training")
    hooks_emit.add_argument("--stage", default="manual", help="stage, e.g. trainer")
    hooks_emit.add_argument("--exit-code", type=int)
    hooks_emit.add_argument("--reason")
    hooks_emit.add_argument(
        "--dry-run", action="store_true", help="report matches without running actions"
    )

    plugin = commands.add_parser(
        "plugin", help="inspect, install, and compose declarative project plugins"
    )
    plugin_commands = plugin.add_subparsers(dest="plugin_command", required=True)
    plugin_inspect = plugin_commands.add_parser(
        "inspect", help="validate a local plugin bundle without executing it"
    )
    plugin_inspect.add_argument("--source", required=True)
    plugin_install = plugin_commands.add_parser(
        "install", help="atomically copy a validated local bundle into the project store"
    )
    plugin_install.add_argument("--source", required=True)
    plugin_install.add_argument("--sha256", help="expected inspected bundle SHA-256 digest")
    plugin_commands.add_parser("list", help="list installed plugin bundles")
    plugin_health = plugin_commands.add_parser(
        "health", help="report static plugin readiness without starting plugins"
    )
    plugin_health.add_argument("--profile")
    plugin_remove = plugin_commands.add_parser("remove", help="remove a disabled plugin bundle")
    plugin_remove.add_argument("plugin_id")
    plugin_remove.add_argument("--version")
    profiles = plugin_commands.add_parser("profile", help="manage explicit plugin profiles")
    profile_commands = profiles.add_subparsers(dest="profile_command", required=True)
    profile_commands.add_parser("list", help="list saved plugin profiles")
    for profile_action, help_text in (
        ("show", "show one saved profile"),
        ("resolve", "resolve one profile against installed bundles"),
    ):
        profile_parser = profile_commands.add_parser(profile_action, help=help_text)
        profile_parser.add_argument("name")
    profile_enable = profile_commands.add_parser("enable", help="enable or add a profile plugin")
    profile_enable.add_argument("name")
    profile_enable.add_argument("plugin_id")
    profile_enable.add_argument("--version")
    profile_enable.add_argument("--grant", action="append")
    profile_disable = profile_commands.add_parser("disable", help="disable a profile plugin")
    profile_disable.add_argument("name")
    profile_disable.add_argument("plugin_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return a process-compatible exit code."""

    arguments = _parser().parse_args(argv)
    arguments.json = arguments.json or arguments.format == "json"
    if arguments.command == "plugin":
        return _run_plugin_command(arguments)
    if arguments.command == "fork-gate":
        return _fork_gate(arguments, as_json=arguments.json)
    if arguments.command == "watchdog" and arguments.watchdog_command == "tick":
        return _watchdog_tick(arguments, as_json=arguments.json)
    project = load_project(Path(arguments.project))
    if arguments.command == "doctor":
        return _doctor(project, as_json=arguments.json)
    store = _store(project)
    data: Any
    if arguments.command == "hooks":
        return _run_hooks_command(project, arguments, as_json=arguments.json)
    if arguments.command == "train":
        return _run_training(
            project,
            as_json=arguments.json,
            capture_enabled=not arguments.no_capture,
            min_coverage=_min_coverage(arguments.min_coverage),
        )
    if arguments.command == "goal" and arguments.goal_command == "run":
        return _run_goal(
            project,
            goal_path=Path(arguments.goal).resolve(),
            as_json=arguments.json,
            capture_enabled=not arguments.no_capture,
        )
    if arguments.command == "knowledge" and arguments.knowledge_command == "export":
        return _export_knowledge(
            project, output=Path(arguments.output).resolve(), as_json=arguments.json
        )
    if arguments.command == "knowledge" and arguments.knowledge_command == "import":
        return _import_knowledge(
            project, source=Path(arguments.input).resolve(), as_json=arguments.json
        )
    if arguments.command == "runtime" and arguments.runtime_command == "start":
        return _run_project_role(
            project,
            command=project.runtime,
            kind="runtime",
            output_command="runtime.start",
            as_json=arguments.json,
            readiness=project.runtime_readiness,
        )
    if arguments.command == "play":
        bundle = Path(arguments.bundle).resolve()
        manifest = verify_model_bundle(bundle)
        if manifest.environment_id != project.environment_id:
            raise ContractViolation(
                "model bundle environment_id does not match the current GLR project"
            )
        if manifest.protocol_version != project.protocol_version:
            raise ContractViolation(
                "model bundle protocol_version does not match the current GLR project"
            )
        return _run_project_role(
            project,
            command=project.player,
            kind="playback",
            output_command="play",
            as_json=arguments.json,
            bundle=bundle,
            metadata={
                "algorithm": manifest.algorithm,
                "framework": manifest.framework,
                "framework_version": manifest.framework_version,
            },
        )
    if arguments.command == "runs" and arguments.runs_command == "list":
        status = None if arguments.status is None else RunStatus(arguments.status)
        data = [
            _run_value(run)
            for run in store.list_runs(
                environment_id=project.environment_id,
                status=status,
                limit=arguments.limit,
            )
        ]
        _emit("runs.list", data, as_json=arguments.json)
        return 0
    if arguments.command == "runs" and arguments.runs_command == "show":
        run = store.get_run(arguments.run_id)
        audits = store.list_declared_metric_audits(run.run_id)
        terminations = store.list_episode_terminations(run.run_id)
        learnability = store.list_learnability(run.run_id)
        data = {
            "run": _run_value(run),
            "events": [_event_value(event) for event in store.list_events(run.run_id)],
            "metrics": [_metric_value(metric) for metric in store.list_metrics(run.run_id)],
            "artifacts": [
                _artifact_value(artifact) for artifact in store.list_artifacts(run.run_id)
            ],
            # Projected so a scheduler can read all three declared-metric
            # counters without walking events, metrics, or a log.
            "declared_metrics": [audit.to_mapping() for audit in audits],
            "declared_metrics_summary": summarize_declared_metrics(audits),
            # Projected so a scheduler can read why each episode ended without
            # walking events or parsing a log.
            "terminations": _terminations_value(terminations),
            "termination_summary": _termination_summary(terminations),
            # Projected so a scheduler can read both learnability numbers
            # without walking events, metrics, or a log.
            "learnability": [_learnability_value(report) for report in learnability],
            "learnability_summary": _learnability_summary(learnability),
        }
        _emit("runs.show", data, as_json=arguments.json)
        return 0
    if arguments.command == "query" and arguments.query_command == "entities":
        data = [
            _entity_value(entity)
            for entity in store.query_entities(
                environment_id=project.environment_id,
                world_id=arguments.world,
                kind=arguments.kind,
                name=arguments.name,
                near=arguments.near,
                radius=arguments.radius,
                limit=arguments.limit,
            )
        ]
        _emit("query.entities", data, as_json=arguments.json)
        return 0
    if arguments.command == "query" and arguments.query_command == "routes":
        data = [
            _route_value(route)
            for route in store.query_routes(
                environment_id=project.environment_id,
                world_id=arguments.world,
                from_entity_id=arguments.from_entity,
                to_entity_id=arguments.to_entity,
                limit=arguments.limit,
            )
        ]
        _emit("query.routes", data, as_json=arguments.json)
        return 0
    if arguments.command == "query" and arguments.query_command == "research":
        findings = store.query_research(
            environment_id=project.environment_id,
            environment_family=project.environment_family,
            tags=arguments.tag,
            category=(None if arguments.category is None else ResearchCategory(arguments.category)),
            include_unverified=not arguments.verified_only,
            limit=arguments.limit,
        )
        data = [
            {
                **finding.to_mapping(),
                "sources": [source.to_mapping() for source in store.get_research_sources(finding)],
                "action_authority": False,
            }
            for finding in findings
        ]
        _emit("query.research", data, as_json=arguments.json)
        return 0
    raise AssertionError("unreachable CLI command")


def entrypoint() -> None:  # pragma: no cover - exercised by package smoke tests
    try:
        raise SystemExit(main())
    except (
        ContractViolation,
        FileNotFoundError,
        GameLaunchError,
        KeyError,
        PluginError,
        TypeError,
        ValueError,
    ) as error:
        if "--json" in sys.argv[1:] or ("--format" in sys.argv[1:] and "json" in sys.argv[1:]):
            print(
                json.dumps(
                    {
                        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
                        "command": "error",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                file=sys.stderr,
            )
        else:
            print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


__all__ = ["CLI_OUTPUT_SCHEMA_VERSION", "RUNTIME_NOT_READY_EXIT_CODE", "entrypoint", "main"]
