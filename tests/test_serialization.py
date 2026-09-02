from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from game_learning_runtime import (
    ActionOutcome,
    ActionReceipt,
    ContractEnvironment,
    JsonlTransitionWriter,
    RefusalReasonClass,
    SyncCollector,
    read_jsonl_transitions,
    transition_from_record,
    transition_to_record,
)
from game_learning_runtime.contracts import Transition
from game_learning_runtime.examples import CounterEnvironment, always_increment


def _transition() -> Transition:
    collector = SyncCollector(ContractEnvironment(CounterEnvironment()))
    return collector.collect(always_increment, steps=1).transitions[0]


def test_transition_record_round_trip_preserves_tensors() -> None:
    original = _transition()

    record = transition_to_record(original)
    restored = transition_from_record(json.loads(json.dumps(record)))

    assert restored.episode_id == original.episode_id
    assert restored.step_id == original.step_id
    np.testing.assert_array_equal(restored.action["choice"], original.action["choice"])
    np.testing.assert_array_equal(restored.reward, original.reward)
    np.testing.assert_array_equal(
        restored.next_observation["position"], original.next_observation["position"]
    )


def test_action_receipt_round_trip_preserves_typed_outcome() -> None:
    original = _transition()
    receipt = ActionReceipt(
        action_id="step-1",
        episode_id=original.episode_id,
        step_id=1,
        outcome=ActionOutcome.BLOCKED,
        issued_timestamp_ns=100,
        observed_timestamp_ns=120,
        postcondition="blocked",
        progress_delta=0.0,
        authoritative_observation_sequence=7,
        target_id="card-1",
        reason_class=RefusalReasonClass.STRUCTURAL,
    )
    original = Transition(
        episode_id=original.episode_id,
        step_id=original.step_id,
        observation=original.observation,
        action=original.action,
        reward=original.reward,
        next_observation=original.next_observation,
        terminated=original.terminated,
        truncated=original.truncated,
        action_mask=original.action_mask,
        next_action_mask=original.next_action_mask,
        action_receipt=receipt,
    )

    restored = transition_from_record(transition_to_record(original))

    assert restored.action_receipt == receipt


def test_transition_provenance_round_trip() -> None:
    original = _transition()
    original = Transition(
        episode_id=original.episode_id,
        step_id=original.step_id,
        observation=original.observation,
        action=original.action,
        reward=original.reward,
        next_observation=original.next_observation,
        terminated=original.terminated,
        truncated=original.truncated,
        provenance={"origin": "policy", "outcome": "neutral", "policy_id": "p1"},
    )
    restored = transition_from_record(transition_to_record(original))
    assert dict(restored.provenance or {}) == dict(original.provenance or {})


def test_jsonl_writer_and_streaming_reader(tmp_path: Path) -> None:
    path = tmp_path / "dataset" / "transitions.jsonl"
    transition = _transition()

    with JsonlTransitionWriter(path) as writer:
        writer.write(transition)
        writer.write(transition)

    restored = list(read_jsonl_transitions(path))
    assert len(restored) == 2
    assert restored[0].episode_id == transition.episode_id


def test_reader_reports_line_number_for_corrupt_record(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\n{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        list(read_jsonl_transitions(path))


def test_writer_requires_context_manager(tmp_path: Path) -> None:
    writer = JsonlTransitionWriter(tmp_path / "unused.jsonl")

    with pytest.raises(RuntimeError, match="context manager"):
        writer.write(_transition())


def test_writer_batch_and_flush_configuration(tmp_path: Path) -> None:
    path = tmp_path / "batch.jsonl"
    transition = _transition()
    with JsonlTransitionWriter(path, flush_every=10) as writer:
        assert writer.write_many([transition, transition]) == 2
    assert len(list(read_jsonl_transitions(path))) == 2
    with pytest.raises(ValueError, match="flush_every"):
        JsonlTransitionWriter(path, flush_every=0)


def test_reader_non_strict_skips_corrupt_and_truncated_tail(tmp_path: Path) -> None:
    path = tmp_path / "crashed.jsonl"
    transition = _transition()
    line = json.dumps(transition_to_record(transition), separators=(",", ":"))
    path.write_text(line + "\n{}\n" + line[:30], encoding="utf-8")
    assert len(list(read_jsonl_transitions(path, strict=False))) == 1
