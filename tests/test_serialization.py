from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from game_learning_runtime import (
    ContractEnvironment,
    JsonlTransitionWriter,
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
