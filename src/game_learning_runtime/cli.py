"""Agent-first local command line interface for Game Learning Runtime."""

from __future__ import annotations

import argparse
import json
import os
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
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.model_bundle import verify_model_bundle
from game_learning_runtime.project import GLRProject, ProjectCommand, load_project
from game_learning_runtime.run_store import (
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
from game_learning_runtime.spatial_knowledge import SpatialKnowledgeBundle

CLI_OUTPUT_SCHEMA_VERSION = "glr.cli-output.v1"


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


def _emit(command: str, data: Any, *, as_json: bool) -> None:
    envelope = {
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "command": command,
        "data": data,
    }
    indent = None if as_json else 2
    print(json.dumps(envelope, ensure_ascii=False, allow_nan=False, indent=indent))


def _store(project: GLRProject) -> TrainingStore:
    return TrainingStore(project.data_dir / "runs.sqlite3")


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
    project: GLRProject, *, run_id: str, run_dir: Path
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
            env=_process_environment(project, run_id=run_id, run_dir=run_dir),
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


def _run_training(project: GLRProject, *, as_json: bool, capture_enabled: bool) -> int:
    store = _store(project)
    run = store.create_run(
        environment_id=project.environment_id,
        protocol_version=project.protocol_version,
        kind="training",
        metadata={"environment_family": project.environment_family},
    )
    run_dir = project.data_dir / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    trainer_log = run_dir / "trainer.log"
    capture_session: tuple[subprocess.Popen[str], IO[str], Path] | None = None
    capture_complete = True
    trainer_exit = 1
    try:
        if capture_enabled and project.capture is not None:
            capture_session = _start_capture(project, run_id=run.run_id, run_dir=run_dir)
        trainer_exit = _run_command(
            project.trainer,
            project=project,
            run_id=run.run_id,
            run_dir=run_dir,
            log_path=trainer_log,
        )
    except KeyboardInterrupt:
        store.finish_run(run.run_id, status=RunStatus.INTERRUPTED, exit_code=None)
        raise
    except BaseException:
        store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
        raise
    finally:
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
            except BaseException:
                if store.get_run(run.run_id).status is RunStatus.RUNNING:
                    store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
                raise

    store.register_artifact(
        run.run_id,
        path="trainer.log",
        source=trainer_log,
        role="run-log",
        media_type="text/plain",
    )
    succeeded = trainer_exit == 0 and (
        capture_complete or project.capture is None or not project.capture.required
    )
    exit_code = 0 if succeeded else trainer_exit if trainer_exit != 0 else 1
    finished = store.finish_run(
        run.run_id,
        status=RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED,
        exit_code=exit_code,
    )
    _emit("train", _run_value(finished), as_json=as_json)
    return exit_code


def _run_project_role(
    project: GLRProject,
    *,
    command: ProjectCommand,
    kind: str,
    output_command: str,
    as_json: bool,
    bundle: Path | None = None,
    metadata: dict[str, Any] | None = None,
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
    log_path = run_dir / f"{kind}.log"
    try:
        exit_code = _run_command(
            command,
            project=project,
            run_id=run.run_id,
            run_dir=run_dir,
            log_path=log_path,
            bundle=bundle,
        )
    except KeyboardInterrupt:
        store.finish_run(run.run_id, status=RunStatus.INTERRUPTED, exit_code=None)
        raise
    except BaseException:
        store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
        raise
    store.register_artifact(
        run.run_id,
        path=log_path.name,
        source=log_path,
        role="run-log",
        media_type="text/plain",
    )
    finished = store.finish_run(
        run.run_id,
        status=RunStatus.SUCCEEDED if exit_code == 0 else RunStatus.FAILED,
        exit_code=exit_code,
    )
    _emit(output_command, _run_value(finished), as_json=as_json)
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
        store.finish_run(run.run_id, status=RunStatus.INTERRUPTED, exit_code=None)
        raise
    except BaseException:
        store.finish_run(run.run_id, status=RunStatus.FAILED, exit_code=1)
        raise

    satisfied = last_evaluation is not None and last_evaluation.satisfied
    exit_code = 0 if satisfied else 3
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glr", description="Game Learning Runtime control plane")
    parser.add_argument("--project", default=".", help="project root or glr-project.json")
    parser.add_argument("--json", action="store_true", help="emit compact stable JSON")
    commands = parser.add_subparsers(dest="command", required=True)

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
    runtime = commands.add_parser("runtime", help="start the configured game/runtime bridge")
    runtime_commands = runtime.add_subparsers(dest="runtime_command", required=True)
    runtime_commands.add_parser("start")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return a process-compatible exit code."""

    arguments = _parser().parse_args(argv)
    project = load_project(Path(arguments.project))
    store = _store(project)
    data: Any
    if arguments.command == "train":
        return _run_training(
            project, as_json=arguments.json, capture_enabled=not arguments.no_capture
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
        data = {
            "run": _run_value(run),
            "events": [_event_value(event) for event in store.list_events(run.run_id)],
            "metrics": [_metric_value(metric) for metric in store.list_metrics(run.run_id)],
            "artifacts": [
                _artifact_value(artifact) for artifact in store.list_artifacts(run.run_id)
            ],
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
    except (ContractViolation, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        if "--json" in sys.argv[1:]:
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


__all__ = ["CLI_OUTPUT_SCHEMA_VERSION", "entrypoint", "main"]
