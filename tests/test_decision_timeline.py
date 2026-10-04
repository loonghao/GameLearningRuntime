"""Synthetic diagnostics: safe decision capture and read-only event replay."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime.contracts import ActionOutcome, ActionReceipt, TimeStep, Transition
from game_learning_runtime.correlated_rewards import (
    OBSERVATION_CONTEXT_KEY,
    REWARD_EVIDENCE_KEY,
    CorrelatedRewardGuard,
    CorrelationPolicy,
    LearningConsumerPolicy,
    RewardAttribution,
    ScalarLearningUpdate,
)
from game_learning_runtime.decision_evidence import DecisionEvidence, capture_transition
from game_learning_runtime.decision_timeline import project_events, read_timeline
from game_learning_runtime.knowledge_evidence import RuleIndexBinding
from game_learning_runtime.phases import EnvironmentPhase
from game_learning_runtime.run_store import RunEvent, TrainingStore
from game_learning_runtime.telemetry import Telemetry
from game_learning_runtime.training import RewardSignal, TrainingConfig
from game_learning_runtime.training_safety import RewardSafetyConfig


def _setup(tmp_path: Path):
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="synthetic.timeline",
        protocol_version="1.0",
        kind="train",
        run_id="run-synthetic",
        environment_config_digest="1" * 64,
        metadata={"credential": "PRIVATE-MUST-NOT-EXPORT"},
    )
    receipt = ActionReceipt(
        "action-1",
        UUID(int=1),
        1,
        ActionOutcome.ACCEPTED,
        100,
        110,
        authoritative_observation_sequence=11,
        issued_against_observation_sequence=10,
        target_id="synthetic-target",
    )
    transition = Transition(
        UUID(int=1),
        0,
        {"state": np.array([0], np.float32)},
        {"choice": np.array([1], np.int64)},
        np.array([1.0], np.float32),
        {"state": np.array([1], np.float32)},
        np.array([True]),
        np.array([False]),
        action_receipt=receipt,
        timestamp_ns=110,
        info={"frame": "PRIVATE-RAW-FRAME", "command_args": ["PRIVATE-ARG"]},
    )
    return store, run, transition


def test_capture_preserves_unknown_policy_internals_and_excludes_private_data(tmp_path: Path):
    _store, run, transition = _setup(tmp_path)
    value = capture_transition(transition, run=run, decision_id="decision-1").to_mapping()
    assert value["selection"]["candidates"] is None
    assert value["selection"]["basis"] is None
    assert value["observation"]["confidence"] is None
    assert value["rules"]["consumptions"] is None
    assert value["policy"]["mode"] == "unknown"
    assert value["comparison"]["status"] == "unknown"
    assert value["execution"]["action_id"] == "action-1"
    assert value["selection"]["action_sha256"]
    assert "PRIVATE-" not in json.dumps(value)


def test_local_persistence_and_readonly_cursor_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    store, run, transition = _setup(tmp_path)
    telemetry = Telemetry(store, run.run_id, console=True)
    telemetry.decision_evidence(capture_transition(transition, run=run, decision_id="decision-1"))
    store.append_event(run.run_id, kind="private.log", payload={"command_args": ["SECRET"]})
    telemetry.decision_evidence(capture_transition(transition, run=run, decision_id="decision-2"))
    assert capsys.readouterr().err == ""
    before = hashlib.sha256(store.path.read_bytes()).hexdigest()
    first = read_timeline(store.path, run.run_id, limit=1)
    assert first["cursor"]["events_after"] == 1
    assert first["more"]
    second = read_timeline(store.path, run.run_id, after_sequence=1, limit=1)
    assert second["entries"] == []
    assert second["cursor"]["events_after"] == 2
    third = read_timeline(store.path, run.run_id, after_sequence=2, limit=1)
    assert third["entries"][0]["evidence"]["identity"]["decision_id"] == "decision-2"
    assert not third["more"]
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == before
    assert "SECRET" not in json.dumps(second)


def test_schema_refuses_hidden_rationale_extra_args_and_nonfinite_scores(tmp_path: Path):
    _store, run, transition = _setup(tmp_path)
    original = capture_transition(transition, run=run, decision_id="decision-1").to_mapping()
    for field in ("chain_of_thought", "raw_frame", "command_args", "credentials"):
        value = {**original, field: "never export"}
        with pytest.raises(ValueError):
            DecisionEvidence(value)
    value = capture_transition(transition, run=run, decision_id="decision-1").to_mapping()
    value["selection"]["candidates"] = [
        {
            "id": "candidate-1",
            "legal": True,
            "rejection_reason": None,
            "score": float("nan"),
            "basis": "score_order",
        }
    ]
    with pytest.raises(ValueError):
        DecisionEvidence(value)


def test_missing_reader_database_is_not_created(tmp_path: Path):
    missing = tmp_path / "absent.sqlite3"
    with pytest.raises(FileNotFoundError):
        read_timeline(missing, "run-synthetic")
    assert not missing.exists()


def _strict_receipt(run, transition, *, signal_value: float = 1.0, verified: bool = True):
    training = TrainingConfig.from_mapping(
        {
            "schema_version": "glr.training.v1",
            "lifecycle": {"start_mode": "reset", "stop_on_done": True},
            "bridge": {"required_capabilities": []},
            "knowledge_sources": [
                {"id": "runtime", "authority": "authoritative", "required": True}
            ],
            "reward": {
                "minimum": -2,
                "maximum": 2,
                "terms": [
                    {
                        "name": "progress",
                        "source": "runtime",
                        "weight": 1,
                        "minimum": -1,
                        "maximum": 1,
                        "required": True,
                    },
                    {
                        "name": "outcome",
                        "source": "runtime",
                        "weight": 1,
                        "minimum": -1,
                        "maximum": 1,
                        "required": False,
                    },
                ],
            },
        }
    )
    safety = RewardSafetyConfig.from_mapping(
        {
            "schema_version": "glr.reward-safety.v1",
            "outcome_signal": "outcome",
            "shaping_signals": ["progress"],
            "max_positive_shaping_per_step": 1,
            "max_positive_shaping_per_episode": 1,
            "max_negative_shaping_per_step": 1,
            "max_negative_shaping_per_episode": 1,
            "failure_episode_maximum": 0,
            "require_terminal_outcome": False,
        }
    )
    context = {
        "run_id": run.run_id,
        "environment_id": run.environment_id,
        "protocol_version": "1.0",
        "environment_config_sha256": "1" * 64,
        "target_id": "synthetic-target",
        "episode_id": str(transition.episode_id),
        "step_id": 0,
        "producer_sequence": 10,
        "timestamp_ns": 90,
        "phase": "gameplay",
        "alive": True,
    }
    before = TimeStep(
        transition.observation,
        np.array([0.0], np.float32),
        np.array([False]),
        np.array([False]),
        transition.episode_id,
        0,
        info={OBSERVATION_CONTEXT_KEY: context},
        timestamp_ns=90,
    )
    after = TimeStep(
        transition.next_observation,
        transition.reward,
        transition.terminated,
        transition.truncated,
        transition.episode_id,
        1,
        action_receipt=transition.action_receipt,
        timestamp_ns=110,
        info={
            OBSERVATION_CONTEXT_KEY: {
                **context,
                "step_id": 1,
                "producer_sequence": 11,
                "timestamp_ns": 110,
            },
            REWARD_EVIDENCE_KEY: {
                "signals": [{"name": "progress", "source": "runtime", "value": signal_value}],
                "attributions": [
                    {
                        "signal_name": "progress",
                        "source": "runtime",
                        "action_id": "action-1",
                        "before_sequence": 10,
                        "after_sequence": 11,
                        "effect": "confirmed",
                    }
                ],
            },
        },
    )
    guard = CorrelatedRewardGuard(
        training,
        safety,
        CorrelationPolicy(
            run.run_id, run.environment_id, run.protocol_version, "synthetic-target", "1" * 64
        ),
    )
    guard.reset(transition.episode_id)
    if verified:
        return guard.compose_timestep(before, after, action=transition.action)
    return guard.compose(
        before,
        after,
        [RewardSignal("progress", "runtime", signal_value)],
        [RewardAttribution.from_mapping(after.info[REWARD_EVIDENCE_KEY]["attributions"][0])],
        action=transition.action,
        verify_observed_reward=False,
    )


def _update(receipt):
    return ScalarLearningUpdate(
        "learner-1",
        "table-1",
        0,
        receipt.state_sha256,
        receipt.action_sha256,
        receipt.next_state_sha256,
        receipt.sha256,
        0.0,
        0.0,
        0.9,
        0.5,
        1.0,
        0.5,
    )


def test_actual_strict_receipt_and_reported_update_link_without_claiming_table_learning(
    tmp_path: Path,
):
    store, run, transition = _setup(tmp_path)
    receipt = _strict_receipt(run, transition)
    telemetry = Telemetry(store, run.run_id, console=False)
    telemetry.decision_evidence(
        capture_transition(transition, run=run, decision_id="decision-1", correlated_reward=receipt)
    )
    telemetry.correlated_reward(receipt)
    telemetry.correlated_learning_update(
        receipt, _update(receipt), consumer=LearningConsumerPolicy("learner-1", "table-1", 0)
    )
    page = read_timeline(store.path, run.run_id)
    assert page["warnings"] == []
    assert page["unlinked_update_sequences"] == []
    assert len(page["entries"]) == 2
    for entry in page["entries"]:
        assert entry["authority"] == "diagnostic"
        assert entry["learning_updates"][0]["status"] == "reported_binding_checked"
        assert entry["evidence"]["outcome"]["success"] is None
        assert entry["evidence"]["rules"]["consumptions"] is None


@pytest.mark.parametrize(
    "change", ["episode", "action", "sequence", "next_digest", "receipt_digest", "step"]
)
def test_foreign_or_incomplete_update_never_acquires_a_verified_link(tmp_path: Path, change: str):
    store, run, transition = _setup(tmp_path)
    receipt = _strict_receipt(run, transition)
    telemetry = Telemetry(store, run.run_id, console=False)
    telemetry.correlated_reward(receipt)
    payload = {
        "update": _update(receipt).to_mapping(),
        "action_id": "action-1",
        "before_sequence": 10,
        "after_sequence": 11,
    }
    episode = "episode-" + str(transition.episode_id)
    step = 0
    if change == "episode":
        episode = "episode-" + str(UUID(int=2))
    elif change == "action":
        payload["action_id"] = "action-other"
    elif change == "sequence":
        payload["after_sequence"] = 12
    elif change == "next_digest":
        payload["update"]["next_state_sha256"] = "9" * 64
    elif change == "receipt_digest":
        payload["update"]["reward_receipt_sha256"] = "9" * 64
    else:
        step = 7
    store.append_event(
        run.run_id,
        kind="learning.correlated-update",
        payload=payload,
        episode_id=episode,
        step_id=step,
    )
    page = read_timeline(store.path, run.run_id)
    assert page["entries"][0]["learning_updates"] == []
    assert page["warnings"] or page["unlinked_update_sequences"]


def test_legacy_commands_and_neighbor_execution_do_not_backfill_a_decision(tmp_path: Path):
    store, run, _transition = _setup(tmp_path)
    for kind in ("agent.decision", "agent.execution"):
        store.append_event(
            run.run_id,
            kind=kind,
            step_id=0,
            payload={
                "selected_key": "choice-1",
                "state": "PRIVATE-STATE",
                "command": "PRIVATE-COMMAND",
                "parameters": {"secret": "PRIVATE-ARG"},
                "receipt": {"accepted": True, "reason": "PRIVATE-REASON"},
            },
        )
    page = read_timeline(store.path, run.run_id)
    assert len(page["entries"]) == 2
    assert all(entry["relation"] == "legacy_unverified" for entry in page["entries"])
    assert all(entry["evidence"]["execution"] is None for entry in page["entries"])
    assert all(entry["evidence"]["selection"]["basis"] is None for entry in page["entries"])
    assert "PRIVATE-" not in json.dumps(page)


@pytest.mark.parametrize("change", ["epoch", "poststep"])
def test_capture_rejects_reward_from_another_post_epoch_or_step(tmp_path: Path, change: str):
    _store, run, transition = _setup(tmp_path)
    original = _strict_receipt(run, transition)
    context = (
        replace(original.after, episode_id=UUID(int=2))
        if change == "epoch"
        else replace(original.after, step_id=7)
    )
    with pytest.raises(ValueError):
        capture_transition(transition, run=run, correlated_reward=replace(original, after=context))


@pytest.mark.parametrize(
    "side,alive,phase,expected",
    [
        ("before", None, EnvironmentPhase.GAMEPLAY, "unknown"),
        ("after", None, EnvironmentPhase.GAMEPLAY, "unknown"),
        ("before", True, EnvironmentPhase.LOADING, "loading"),
        ("after", True, EnvironmentPhase.MENU, "menu"),
    ],
)
def test_lifecycle_projection_preserves_explicit_phase_and_unknown_alive(
    tmp_path: Path, side: str, alive: bool | None, phase: EnvironmentPhase, expected: str
):
    _store, run, transition = _setup(tmp_path)
    original = _strict_receipt(run, transition)
    context = replace(getattr(original, side), alive=alive, phase=phase)
    receipt = replace(original, **{side: context})
    record = capture_transition(transition, run=run, correlated_reward=receipt).to_mapping()
    assert record["reward"][f"lifecycle_{side}"] == expected


@pytest.mark.parametrize(
    "baseline,candidate,minimum,status",
    [
        (1.0, 1.0, 0.0, "tie"),
        (1.0, 0.5, 0.0, "baseline_better"),
        (1.0, 1.1, 0.2, "improvement_below_threshold"),
        (1.0, 1.5, 0.2, "candidate_better"),
    ],
)
def test_fixed_comparison_is_diagnostic_and_ties_never_claim_a_winner(
    tmp_path: Path, baseline: float, candidate: float, minimum: float, status: str
):
    _store, run, transition = _setup(tmp_path)
    record = capture_transition(transition, run=run).to_mapping()
    record["comparison"] = {
        "baseline_score": baseline,
        "candidate_score": candidate,
        "minimum_improvement": minimum,
        "status": status,
        "suite_sha256": "2" * 64,
        "evaluator_sha256": "3" * 64,
        "budget_steps": 10,
        "direction": "max",
    }
    assert DecisionEvidence(record).to_mapping()["comparison"]["status"] == status
    if status != "candidate_better":
        record["comparison"]["status"] = "candidate_better"
        with pytest.raises(ValueError):
            DecisionEvidence(record)


class _InjectedMapping(dict):
    def items(self):
        return [*super().items(), ("command_args", ["PRIVATE-INJECTED"])]


@pytest.mark.parametrize("level", ["top", "selection"])
def test_mapping_hooks_cannot_inject_unvalidated_private_exports(tmp_path: Path, level: str):
    _store, run, transition = _setup(tmp_path)
    data = capture_transition(transition, run=run).to_mapping()
    if level == "top":
        data = _InjectedMapping(data)
    else:
        data["selection"] = _InjectedMapping(data["selection"])
    with pytest.raises(ValueError):
        DecisionEvidence(data)


@pytest.mark.parametrize(
    "missing", ["receipt_sha256", "next_state_sha256", "before_sequence", "after_sequence", "total"]
)
def test_unknown_reward_operands_cannot_be_binding_checked(tmp_path: Path, missing: str):
    _store, run, transition = _setup(tmp_path)
    data = capture_transition(
        transition, run=run, correlated_reward=_strict_receipt(run, transition)
    ).to_mapping()
    data["reward"][missing] = None
    if missing in {"before_sequence", "after_sequence"}:
        data["execution"][missing] = None
        data["observation"]["freshness"] = "unknown"
    with pytest.raises(ValueError):
        DecisionEvidence(data)


@pytest.mark.parametrize("timestamp", ["PRIVATE-RAW-FRAME", True, -1, None])
def test_event_metadata_cannot_export_private_text_as_a_timestamp(timestamp):
    event = RunEvent("run-synthetic", 1, timestamp, "agent.decision", None, None, {})
    with pytest.raises(ValueError):
        project_events([event], run_id="run-synthetic")


def test_unknown_source_and_tensor_identity_cannot_be_binding_checked(tmp_path: Path):
    _store, run, transition = _setup(tmp_path)
    receipt = _strict_receipt(run, transition)
    for section, field_name in (
        ("identity", "environment_config_sha256"),
        ("identity", "target_id"),
        ("observation", "state_sha256"),
        ("selection", "action_sha256"),
    ):
        data = capture_transition(transition, run=run, correlated_reward=receipt).to_mapping()
        data[section][field_name] = None
        with pytest.raises(ValueError):
            DecisionEvidence(data)


def test_reward_terminal_marker_must_match_the_captured_transition(tmp_path: Path):
    _store, run, transition = _setup(tmp_path)
    receipt = _strict_receipt(run, transition)
    malformed = replace(receipt, result=replace(receipt.result, terminal=False))
    with pytest.raises(ValueError):
        capture_transition(transition, run=run, correlated_reward=malformed)


@pytest.mark.parametrize(
    ("terminated", "truncated", "terminal"),
    [([True, False], [False, False], False), ([True, False], [False, True], True)],
)
def test_reward_terminal_marker_uses_canonical_all_participant_semantics(
    tmp_path: Path, terminated: list[bool], truncated: list[bool], terminal: bool
):
    _store, run, original = _setup(tmp_path)
    transition = replace(original, terminated=np.array(terminated), truncated=np.array(truncated))
    receipt = _strict_receipt(run, transition)
    assert receipt.result.terminal is terminal
    capture_transition(transition, run=run, correlated_reward=receipt)


@pytest.mark.parametrize(
    "version",
    ["X:/synthetic/trace.txt", "X:synthetic.txt", "private narrative about the next action"],
)
def test_safe_rules_version_rejects_paths_and_free_text_without_changing_old_binding(
    tmp_path: Path, version: str
):
    _store, run, transition = _setup(tmp_path)
    binding = RuleIndexBinding(
        environment_id=run.environment_id,
        protocol_version=run.protocol_version,
        rules_version=version,
        rules_sha256="1" * 64,
        index_sha256="2" * 64,
        index_rules_sha256="1" * 64,
    )
    assert RuleIndexBinding.from_mapping(binding.to_mapping()).rules_version == version
    with pytest.raises(ValueError):
        capture_transition(transition, run=run, rule_binding=binding, consumptions=[])


@pytest.mark.parametrize("version", ["1.0.0", "1.0.0+synthetic.1", "revision-2"])
def test_safe_rules_version_preserves_portable_revision_labels(tmp_path: Path, version: str):
    _store, run, transition = _setup(tmp_path)
    binding = RuleIndexBinding(
        environment_id=run.environment_id,
        protocol_version=run.protocol_version,
        rules_version=version,
        rules_sha256="1" * 64,
        index_sha256="2" * 64,
        index_rules_sha256="1" * 64,
    )
    record = capture_transition(transition, run=run, rule_binding=binding).to_mapping()
    assert record["rules"]["binding"]["rules_version"] == version


@pytest.mark.parametrize(
    "section,field", [("observation", "timestamp_ns"), ("comparison", "baseline_score")]
)
def test_oversized_numeric_payload_skips_bad_row_and_retains_next_decision(
    tmp_path: Path, section: str, field: str
):
    store, run, transition = _setup(tmp_path)
    invalid = capture_transition(transition, run=run).to_mapping()
    invalid[section][field] = 10**1000
    bad = store.append_event(
        run.run_id,
        kind="agent.decision.evidence",
        payload=invalid,
        episode_id="episode-" + str(transition.episode_id),
        step_id=transition.step_id,
    )
    Telemetry(store, run.run_id, console=False).decision_evidence(
        capture_transition(transition, run=run, decision_id="valid-next")
    )
    page = read_timeline(store.path, run.run_id)
    assert page["warnings"] == [{"sequence_id": bad.sequence_id, "code": "invalid_evidence"}]
    assert len(page["entries"]) == 1
    assert page["entries"][0]["evidence"]["identity"]["decision_id"] == "valid-next"
    assert page["cursor"]["events_after"] == 2
    assert not page["more"]


def test_deep_json_payload_skips_bad_row_and_retains_next_decision(tmp_path: Path):
    store, run, transition = _setup(tmp_path)
    bad = store.append_event(
        run.run_id,
        kind="agent.decision.evidence",
        payload={},
        episode_id="episode-" + str(transition.episode_id),
        step_id=transition.step_id,
    )
    deep_json = '{"x":' + "[" * 16000 + "0" + "]" * 16000 + "}"
    assert len(deep_json.encode("utf-8")) < 64 * 1024
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE events SET payload_json=? WHERE run_id=? AND sequence_id=?",
            (deep_json, run.run_id, bad.sequence_id),
        )
    Telemetry(store, run.run_id, console=False).decision_evidence(
        capture_transition(transition, run=run, decision_id="valid-next")
    )
    before = hashlib.sha256(store.path.read_bytes()).hexdigest()
    page = read_timeline(store.path, run.run_id)
    assert page["warnings"] == [{"sequence_id": bad.sequence_id, "code": "invalid_evidence"}]
    assert len(page["entries"]) == 1
    assert page["entries"][0]["evidence"]["identity"]["decision_id"] == "valid-next"
    assert page["cursor"]["events_after"] == 2
    assert not page["more"]
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == before


def test_nontext_payload_skips_bad_row_and_retains_next_decision(tmp_path: Path):
    store, run, transition = _setup(tmp_path)
    bad = store.append_event(
        run.run_id,
        kind="agent.decision.evidence",
        payload={},
        episode_id="episode-" + str(transition.episode_id),
        step_id=transition.step_id,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE events SET payload_json=? WHERE run_id=? AND sequence_id=?",
            (b"synthetic-invalid-blob", run.run_id, bad.sequence_id),
        )
    Telemetry(store, run.run_id, console=False).decision_evidence(
        capture_transition(transition, run=run, decision_id="valid-next")
    )
    page = read_timeline(store.path, run.run_id)
    assert page["warnings"] == [{"sequence_id": bad.sequence_id, "code": "invalid_evidence"}]
    assert len(page["entries"]) == 1
    assert page["entries"][0]["evidence"]["identity"]["decision_id"] == "valid-next"
    assert page["cursor"]["events_after"] == 2
    assert not page["more"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("verified", [True, False])
@pytest.mark.parametrize("change", ["positive", "negative", "nextafter"])
def test_captured_scalar_reward_requires_exact_equality(
    tmp_path: Path, dtype: str, verified: bool, change: str
):
    _store, run, original = _setup(tmp_path)
    transition = replace(original, reward=np.array([1.0], dtype=dtype))
    receipt = _strict_receipt(run, transition, verified=verified)
    if change == "nextafter":
        changed = np.nextafter(transition.reward, np.array([np.inf], dtype=dtype))
    else:
        changed = np.array([1.0 + (5e-7 if change == "positive" else -5e-7)], dtype=dtype)
    with pytest.raises(ValueError):
        capture_transition(replace(transition, reward=changed), run=run, correlated_reward=receipt)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("verified", [True, False])
def test_captured_scalar_reward_preserves_quantization_and_measurement_status(
    tmp_path: Path, dtype: str, verified: bool
):
    _store, run, original = _setup(tmp_path)
    transition = replace(original, reward=np.array([0.1], dtype=dtype))
    receipt = _strict_receipt(run, transition, signal_value=0.1, verified=verified)
    assert (receipt.observed_reward is not None) is verified
    record = capture_transition(transition, run=run, correlated_reward=receipt).to_mapping()
    assert record["reward"]["total"] == float(transition.reward.item())
    assert record["reward"]["correlation"] == ("binding_checked" if verified else "declared")
    assert record["reward"]["receipt_sha256"] == receipt.sha256


@pytest.mark.parametrize("dtype", ["bool", "int64", "complex128", "float16"])
def test_captured_scalar_reward_rejects_unsupported_dtypes(tmp_path: Path, dtype: str):
    _store, run, transition = _setup(tmp_path)
    receipt = _strict_receipt(run, transition)
    invalid = replace(transition, reward=np.array([1], dtype=dtype))
    with pytest.raises(ValueError):
        capture_transition(invalid, run=run, correlated_reward=receipt)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_captured_scalar_reward_rejects_nonfinite_values(tmp_path: Path, value: float):
    _store, run, transition = _setup(tmp_path)
    receipt = _strict_receipt(run, transition)
    invalid = replace(transition, reward=np.array([value], dtype=np.float64))
    with pytest.raises(ValueError):
        capture_transition(invalid, run=run, correlated_reward=receipt)
