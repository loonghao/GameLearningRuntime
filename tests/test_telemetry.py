import json

import pytest

from game_learning_runtime.decisions import Candidate, Decision, execute_decision
from game_learning_runtime.run_store import TrainingStore
from game_learning_runtime.telemetry import Telemetry


def telemetry(tmp_path):
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(environment_id="test.telemetry", protocol_version="1.0", kind="training")
    return Telemetry(store, run.run_id)


def test_learner_updates_routes_and_console_are_durable(tmp_path, capsys):
    target = telemetry(tmp_path)
    target.learning_update(step_id=5, metrics={"loss": 0.25, "reward": 2.0})
    target.route_sample([1.0, 2.0, 3.0], step_id=5, episode_id="episode-1")
    reopened = TrainingStore(target.store.path)
    events = reopened.list_events(target.run_id)
    assert [event.kind for event in events] == ["learning.update", "navigation.route_sample"]
    assert events[1].step_id == 5
    assert events[1].episode_id == "episode-1"
    assert dict(events[1].payload)["position"] == [1.0, 2.0, 3.0]
    assert [metric.value for metric in reopened.list_metrics(target.run_id)] == [0.25, 2.0]
    lines = capsys.readouterr().err.splitlines()
    assert json.loads(lines[0])["schema_version"] == "glr.telemetry.v1"


def test_decisions_automatically_bind_to_the_cli_run_and_never_retry(tmp_path, monkeypatch):
    target = telemetry(tmp_path)
    monkeypatch.setenv("GLR_STORE_PATH", str(target.store.path))
    monkeypatch.setenv("GLR_RUN_ID", target.run_id)
    decision = Decision("observation-1", (Candidate("walk", "move"),), "walk", "digest", "train")
    calls = []

    def execute(command, parameters):
        calls.append(command)
        return {"accepted": False, "reason": "blocked"}

    result = execute_decision(decision, execute, step_id=7)
    assert not result["receipt"]["accepted"]
    events = target.store.list_events(target.run_id)
    assert [e.kind for e in events] == ["agent.decision", "agent.execution"]
    assert all(e.step_id == 7 for e in events)
    assert calls == ["move"]

    def fail(command, parameters):
        calls.append(command)
        raise RuntimeError("uncertain execution")

    with pytest.raises(RuntimeError, match="uncertain execution"):
        execute_decision(decision, fail, step_id=8)
    assert calls == ["move", "move"]
    assert target.store.list_events(target.run_id)[-1].kind == "agent.execution_failed"


def test_missing_telemetry_does_not_change_action_semantics(monkeypatch, tmp_path):
    monkeypatch.setenv("GLR_STORE_PATH", str(tmp_path))
    monkeypatch.setenv("GLR_RUN_ID", "run-test")
    decision = Decision("s", (Candidate("walk", "move"),), "walk", "digest", "train")
    calls = []
    result = execute_decision(
        decision, lambda command, _: calls.append(command) or {"accepted": True}
    )
    assert result["receipt"]["accepted"]
    assert calls == ["move"]


def test_no_environment_does_not_create_a_run(monkeypatch):
    monkeypatch.delenv("GLR_RUN_ID", raising=False)
    monkeypatch.delenv("GLR_STORE_PATH", raising=False)
    assert Telemetry.from_env() is None


@pytest.mark.parametrize("position", [[1], [1, 2, 3, 4], [1, float("nan")]])
def test_invalid_routes_are_not_persisted(tmp_path, position):
    target = telemetry(tmp_path)
    with pytest.raises(ValueError):
        target.route_sample(position, step_id=1, episode_id="episode-1")
    assert not target.store.list_events(target.run_id)
