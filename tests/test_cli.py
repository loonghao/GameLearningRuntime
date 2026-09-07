from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from game_learning_runtime.agent_goal import ResearchBundle
from game_learning_runtime.cli import main
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.model_bundle import build_model_bundle
from game_learning_runtime.run_store import (
    RouteWaypoint,
    RunStatus,
    SpatialEntity,
    SpatialRoute,
    TrainingStore,
)
from game_learning_runtime.training import KnowledgeAuthority


def _project(
    root: Path,
    *,
    trainer_argv: list[str] | None = None,
    player_argv: list[str] | None = None,
    researcher_argv: list[str] | None = None,
    planner_argv: list[str] | None = None,
    evaluator_argv: list[str] | None = None,
    capture_argv: list[str] | None = None,
) -> None:
    (root / "bridge").mkdir()
    (root / "glr-project.json").write_text(
        json.dumps(
            {
                "schema_version": "glr.project.v1",
                "environment_id": "example.adventure-v1",
                "environment_family": "action-rpg",
                "protocol_version": "1.0",
                "data_dir": ".glr",
                "bridge_path": "bridge",
                "runtime": {"argv": ["python", "-c", "print('runtime')"]},
                "trainer": {"argv": trainer_argv or ["python", "-c", "print('train')"]},
                "player": {"argv": player_argv or ["python", "-c", "print('play')", "{bundle}"]},
                "researcher": (None if researcher_argv is None else {"argv": researcher_argv}),
                "planner": None if planner_argv is None else {"argv": planner_argv},
                "evaluator": (None if evaluator_argv is None else {"argv": evaluator_argv}),
                "capture": (
                    None
                    if capture_argv is None
                    else {
                        "argv": capture_argv,
                        "required": True,
                        "stop": "stdin-q",
                        "video_file": "capture.mp4",
                        "index_file": "capture-index.jsonl",
                        "codec": "h264",
                        "frame_rate": 12,
                        "width": 640,
                        "height": 360,
                    }
                ),
            }
        ),
        encoding="utf-8",
    )


def test_cli_lists_training_runs_as_stable_agent_json(tmp_path: Path, capsys: object) -> None:
    _project(tmp_path)
    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        kind="training",
    )
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)

    assert main(["--project", str(tmp_path), "--json", "runs", "list"]) == 0

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["schema_version"] == "glr.cli-output.v1"
    assert output["command"] == "runs.list"
    assert output["data"][0]["run_id"] == run.run_id
    assert output["data"][0]["status"] == "succeeded"


def test_cli_train_runs_project_trainer_and_persists_agent_query_data(
    tmp_path: Path, capsys: object
) -> None:
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        """
import os
from pathlib import Path
from game_learning_runtime.run_store import TrainingStore

store = TrainingStore(os.environ["GLR_STORE_PATH"])
store.append_event(
    os.environ["GLR_RUN_ID"],
    kind="trainer.ready",
    payload={"bridge_path": os.environ["GLR_BRIDGE_PATH"]},
)
Path(os.environ["GLR_RUN_DIR"], "model.bin").write_bytes(b"model")
print("trainer complete")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _project(tmp_path, trainer_argv=[sys.executable, str(trainer)])

    assert main(["--project", str(tmp_path), "--json", "train"]) == 0

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    run_id = output["data"]["run_id"]
    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = store.get_run(run_id)
    assert run.status is RunStatus.SUCCEEDED
    assert store.list_events(run_id)[0].kind == "trainer.ready"
    assert (tmp_path / ".glr/runs" / run_id / "model.bin").read_bytes() == b"model"
    assert store.list_artifacts(run_id)[0].role == "run-log"


def test_cli_play_verifies_compatible_bundle_before_project_player(
    tmp_path: Path, capsys: object
) -> None:
    player = tmp_path / "player.py"
    player.write_text(
        """
import os
from pathlib import Path
from game_learning_runtime.run_store import TrainingStore

assert Path(os.environ["GLR_MODEL_BUNDLE"], "manifest.json").is_file()
TrainingStore(os.environ["GLR_STORE_PATH"]).append_event(
    os.environ["GLR_RUN_ID"], kind="player.loaded", payload={}
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _project(tmp_path, player_argv=[sys.executable, str(player), "{bundle}"])
    source = tmp_path / "source.bin"
    source.write_bytes(b"model")
    bundle = tmp_path / "bundle"
    build_model_bundle(
        bundle,
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        algorithm="bc",
        framework="pytorch",
        framework_version="2.8.0",
        seeds=(7,),
        inputs={"training.json": source},
        artifacts={"model.bin": source},
    )

    assert main(["--project", str(tmp_path), "--json", "play", "--bundle", str(bundle)]) == 0

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    run_id = output["data"]["run_id"]
    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    assert store.get_run(run_id).kind == "playback"
    assert store.list_events(run_id)[0].kind == "player.loaded"


def test_cli_train_records_small_window_video_concurrently_with_step_index(
    tmp_path: Path, capsys: object
) -> None:
    recorder = tmp_path / "recorder.py"
    recorder.write_text(
        """
import json
import os
import sys
from pathlib import Path

Path(os.environ["GLR_CAPTURE_VIDEO"]).write_bytes(b"synthetic-h264")
Path(os.environ["GLR_CAPTURE_INDEX"]).write_text(json.dumps({
    "schema_version": "glr.capture-frame.v1",
    "run_id": os.environ["GLR_RUN_ID"],
    "episode_id": "12345678-1234-5678-1234-567812345678",
    "step_id": 0,
    "frame_index": 0,
    "pts_ns": 0,
    "observation_timestamp_ns": 1
}) + "\\n", encoding="utf-8")
for line in sys.stdin:
    if line.strip() == "q":
        break
""".strip()
        + "\n",
        encoding="utf-8",
    )
    trainer = tmp_path / "capture_trainer.py"
    trainer.write_text(
        """
import os
import time
from pathlib import Path

video = Path(os.environ["GLR_CAPTURE_VIDEO"])
deadline = time.monotonic() + 5
while not video.is_file() and time.monotonic() < deadline:
    time.sleep(0.01)
assert video.is_file(), "recorder did not start concurrently"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _project(
        tmp_path,
        trainer_argv=[sys.executable, str(trainer)],
        capture_argv=[sys.executable, str(recorder)],
    )

    assert main(["--project", str(tmp_path), "--json", "train"]) == 0

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    artifacts = TrainingStore(tmp_path / ".glr/runs.sqlite3").list_artifacts(
        output["data"]["run_id"]
    )
    assert {artifact.role for artifact in artifacts} >= {
        "review-video",
        "capture-index",
        "capture-manifest",
    }


def test_cli_goal_run_researches_plans_trains_and_requires_persisted_authority(
    tmp_path: Path, capsys: object
) -> None:
    researcher = tmp_path / "researcher.py"
    researcher.write_text(
        """
import json
import os
from pathlib import Path

Path(os.environ["GLR_RESEARCH_PATH"]).write_text(json.dumps({
    "schema_version": "glr.research-bundle.v1",
    "sources": [{
        "source_id": "guide.safe-route",
        "media_type": "text-guide",
        "url": "https://example.com/safe-route",
        "publisher": "Example Publisher",
        "title": "Safe route",
        "accessed_at": "2026-09-01T10:00:00Z",
        "updated_at": None,
        "summary": "A route hypothesis for later runtime verification.",
        "confidence": 0.7,
        "volatility": "medium"
    }],
    "findings": [{
        "finding_id": "strategy.safe-route",
        "category": "strategy",
        "status": "unverified",
        "scope": "family",
        "scope_id": "action-rpg",
        "summary": "Prefer landmark-guided routes and re-observe at hazards.",
        "source_ids": ["guide.safe-route"],
        "tags": ["navigation"],
        "confidence": 0.7,
        "locator": "section-2"
    }]
}), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    planner = tmp_path / "planner.py"
    planner.write_text(
        """
import json
import os
from pathlib import Path

Path(os.environ["GLR_TRIAL_PATH"]).write_text(json.dumps({
    "schema_version": "glr.trial-plan.v1",
    "trial_id": os.environ["GLR_TRIAL_ID"],
    "goal_id": "goal.reach-destination",
    "seed": 7,
    "max_steps": 100,
    "reward_terms": [{
        "name": "arrival",
        "metric": "objective.arrived",
        "weight": 10.0,
        "rationale": "Reward authoritative arrival evidence.",
        "source_finding_ids": ["strategy.safe-route"]
    }],
    "notes": "Try the provenance-bound safe route."
}), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    trainer = tmp_path / "goal_trainer.py"
    trainer.write_text(
        """
import os
from game_learning_runtime.run_store import TrainingStore

store = TrainingStore(os.environ["GLR_STORE_PATH"])
value = 1.0 if store.list_metrics(os.environ["GLR_RUN_ID"]) else 0.0
store.record_metric(
    os.environ["GLR_RUN_ID"],
    name="objective.arrived",
    value=value,
    metadata={"authority": "authoritative", "source": "runtime.telemetry"},
)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        """
import json
import os
from pathlib import Path
from game_learning_runtime.run_store import TrainingStore

metrics = TrainingStore(os.environ["GLR_STORE_PATH"]).list_metrics(os.environ["GLR_RUN_ID"])
Path(os.environ["GLR_EVALUATION_PATH"]).write_text(json.dumps({
    "schema_version": "glr.goal-evidence.v1",
    "goal_id": "goal.reach-destination",
    "trial_id": os.environ["GLR_TRIAL_ID"],
    "evidence": [{
        "metric": "objective.arrived",
        "value": metrics[-1].value,
        "source": "runtime.telemetry",
        "authority": "authoritative",
        "run_id": os.environ["GLR_RUN_ID"]
    }]
}), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _project(
        tmp_path,
        trainer_argv=[sys.executable, str(trainer)],
        researcher_argv=[sys.executable, str(researcher)],
        planner_argv=[sys.executable, str(planner)],
        evaluator_argv=[sys.executable, str(evaluator)],
    )
    goal = tmp_path / "goal.json"
    goal.write_text(
        json.dumps(
            {
                "schema_version": "glr.agent-goal.v1",
                "goal_id": "goal.reach-destination",
                "objective": "Reach the destination and verify arrival.",
                "environment_family": "action-rpg",
                "success_criteria": [
                    {
                        "metric": "objective.arrived",
                        "operator": "gte",
                        "target": 1,
                        "source": "runtime.telemetry",
                    }
                ],
                "budget": {
                    "max_trials": 2,
                    "max_training_steps": 1000,
                    "max_wall_seconds": 60,
                    "max_research_sources": 4,
                },
                "allowed_research_media": ["text-guide"],
            }
        ),
        encoding="utf-8",
    )

    assert main(["--project", str(tmp_path), "--json", "goal", "run", "--goal", str(goal)]) == 0

    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["command"] == "goal.run"
    assert output["data"]["satisfied"] is True
    assert output["data"]["trials_completed"] == 2
    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = store.get_run(output["data"]["run"]["run_id"])
    assert run.status is RunStatus.SUCCEEDED
    assert (
        store.query_research(
            environment_id="example.adventure-v1", environment_family="action-rpg"
        )[0].finding_id
        == "strategy.safe-route"
    )


def test_cli_runtime_queries_and_spatial_knowledge_round_trip(
    tmp_path: Path, capsys: object
) -> None:
    _project(tmp_path)
    assert main(["--project", str(tmp_path), "--json", "runtime", "start"]) == 0
    runtime_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert runtime_output["command"] == "runtime.start"

    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1",
        protocol_version="1.0",
        kind="training",
    )
    entity = SpatialEntity(
        environment_id=run.environment_id,
        world_id="forest",
        entity_id="shrine.forest-1",
        kind="shrine",
        label="林外土地庙",
        position=(10.0, 20.0, 3.0),
        coordinate_frame="world",
        authority=KnowledgeAuthority.AUTHORITATIVE,
        confidence=1.0,
        observed_at_ns=10,
        source_run_id=run.run_id,
    )
    store.upsert_entity(entity)
    store.record_route(
        SpatialRoute(
            environment_id=run.environment_id,
            world_id="forest",
            route_id="route.spawn-to-shrine",
            name="出生点到土地庙",
            from_entity_id="spawn.main",
            to_entity_id=entity.entity_id,
            coordinate_frame="world",
            confidence=0.9,
            verified_at_ns=11,
            source_run_id=run.run_id,
            waypoints=(
                RouteWaypoint(index=0, position=(0.0, 0.0, 0.0), tolerance=1.0),
                RouteWaypoint(index=1, position=entity.position, tolerance=2.0),
            ),
        )
    )
    store.upsert_research_bundle(
        ResearchBundle.from_mapping(
            {
                "schema_version": "glr.research-bundle.v1",
                "sources": [
                    {
                        "source_id": "guide.navigation",
                        "media_type": "text-guide",
                        "url": "https://example.com/navigation",
                        "publisher": "Example",
                        "title": "Navigation",
                        "accessed_at": "2026-09-01T00:00:00Z",
                        "updated_at": None,
                        "summary": "A navigation hypothesis.",
                        "confidence": 0.6,
                        "volatility": "medium",
                    }
                ],
                "findings": [
                    {
                        "finding_id": "strategy.landmarks",
                        "category": "navigation",
                        "status": "unverified",
                        "scope": "family",
                        "scope_id": "action-rpg",
                        "summary": "Navigate using stable landmarks.",
                        "source_ids": ["guide.navigation"],
                        "tags": ["navigation"],
                        "confidence": 0.6,
                        "locator": None,
                    }
                ],
            }
        )
    )
    store.append_event(run.run_id, kind="test.ready", payload={})
    store.record_metric(run.run_id, name="reward.total", value=1.0)
    artifact = tmp_path / "run.txt"
    artifact.write_text("run", encoding="utf-8")
    store.register_artifact(
        run.run_id,
        path="run.txt",
        source=artifact,
        role="run-log",
        media_type="text/plain",
    )
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)

    commands = (
        ["runs", "show", run.run_id],
        ["query", "entities", "--world", "forest", "--kind", "shrine"],
        ["query", "routes", "--world", "forest", "--to-entity", entity.entity_id],
        ["query", "research", "--tag", "navigation", "--category", "navigation"],
    )
    for command in commands:
        assert main(["--project", str(tmp_path), "--json", *command]) == 0
        assert json.loads(capsys.readouterr().out)["data"]  # type: ignore[attr-defined]

    snapshot = tmp_path / "spatial.json"
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "--json",
                "knowledge",
                "export",
                "--output",
                str(snapshot),
            ]
        )
        == 0
    )
    capsys.readouterr()  # type: ignore[attr-defined]
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "--json",
                "knowledge",
                "import",
                "--input",
                str(snapshot),
            ]
        )
        == 0
    )
    imported_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert imported_output["data"]["entities"] == 1
    assert imported_output["data"]["authority"] == "advisory"


def test_cli_rejects_missing_roles_incompatible_bundles_and_knowledge(
    tmp_path: Path,
) -> None:
    missing_roles = tmp_path / "missing-roles"
    missing_roles.mkdir()
    _project(missing_roles)
    goal_value = {
        "schema_version": "glr.agent-goal.v1",
        "goal_id": "goal.test",
        "objective": "Reach a test objective.",
        "environment_family": "action-rpg",
        "success_criteria": [
            {"metric": "score", "operator": "gte", "target": 1, "source": "runtime"}
        ],
        "budget": {
            "max_trials": 1,
            "max_training_steps": 1,
            "max_wall_seconds": 1,
            "max_research_sources": 1,
        },
        "allowed_research_media": ["text-guide"],
    }
    goal = missing_roles / "goal.json"
    goal.write_text(json.dumps(goal_value), encoding="utf-8")
    with pytest.raises(ContractViolation, match="requires project"):
        main(["--project", str(missing_roles), "goal", "run", "--goal", str(goal)])

    mismatched_goal = tmp_path / "mismatched-goal"
    mismatched_goal.mkdir()
    noop = [sys.executable, "-c", "pass"]
    _project(
        mismatched_goal,
        researcher_argv=noop,
        planner_argv=noop,
        evaluator_argv=noop,
    )
    bad_goal = mismatched_goal / "goal.json"
    bad_goal.write_text(
        json.dumps({**goal_value, "environment_family": "platformer"}), encoding="utf-8"
    )
    with pytest.raises(ContractViolation, match="environment_family"):
        main(["--project", str(mismatched_goal), "goal", "run", "--goal", str(bad_goal)])

    playback = tmp_path / "playback"
    playback.mkdir()
    _project(playback)
    source = playback / "model.bin"
    source.write_bytes(b"model")
    wrong_environment = playback / "wrong-environment"
    build_model_bundle(
        wrong_environment,
        environment_id="example.other-v1",
        protocol_version="1.0",
        algorithm="bc",
        framework="pytorch",
        framework_version="2.8.0",
        seeds=(1,),
        inputs={"config.json": source},
        artifacts={"model.bin": source},
    )
    with pytest.raises(ContractViolation, match="environment_id"):
        main(["--project", str(playback), "play", "--bundle", str(wrong_environment)])
    wrong_protocol = playback / "wrong-protocol"
    build_model_bundle(
        wrong_protocol,
        environment_id="example.adventure-v1",
        protocol_version="2.0",
        algorithm="bc",
        framework="pytorch",
        framework_version="2.8.0",
        seeds=(1,),
        inputs={"config.json": source},
        artifacts={"model.bin": source},
    )
    with pytest.raises(ContractViolation, match="protocol_version"):
        main(["--project", str(playback), "play", "--bundle", str(wrong_protocol)])

    snapshot = playback / "spatial.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": "glr.spatial-knowledge.v1",
                "environment_id": "example.other-v1",
                "protocol_version": "1.0",
                "exported_at_ns": 1,
                "entities": [],
                "routes": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractViolation, match="environment_id"):
        main(["--project", str(playback), "knowledge", "import", "--input", str(snapshot)])


def test_cli_marks_failed_training_and_refuses_overwriting_export(
    tmp_path: Path, capsys: object
) -> None:
    _project(
        tmp_path,
        trainer_argv=[sys.executable, "-c", "raise SystemExit(5)"],
    )
    assert main(["--project", str(tmp_path), "--json", "train"]) == 5
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["data"]["status"] == "failed"

    snapshot = tmp_path / "spatial.json"
    assert main(["--project", str(tmp_path), "knowledge", "export", "--output", str(snapshot)]) == 0
    capsys.readouterr()  # type: ignore[attr-defined]
    with pytest.raises(FileExistsError, match="already exists"):
        main(["--project", str(tmp_path), "knowledge", "export", "--output", str(snapshot)])
