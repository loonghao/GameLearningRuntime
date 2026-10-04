"""Installed-package evidence for finite synthetic nonSIM measured consumption."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from game_learning_runtime.examples.measured_fleet_training import (
    build_synthetic_measured_fixture,
    run_demo,
)
from game_learning_runtime.fleet_measured import verify_measured
from game_learning_runtime.fleet_payload import FleetLimits, canonical, decode_shard
from game_learning_runtime.model_bundle import verify_model_bundle
from game_learning_runtime.serialization import transition_to_record


def test_public_fixture_preserves_original_typed_reward_and_execution() -> None:
    limits = FleetLimits(max_chunk_bytes=512)
    fixture = build_synthetic_measured_fixture(
        runtime_source_commit="a" * 40,
        produced_at_utc_ms=1_700_000_000_000,
        limits=limits,
    )
    verified = verify_measured(
        fixture.envelope,
        decode_shard(fixture.shard.manifest, fixture.shard.payload, limits=limits),
        fixture.authority,
        1_700_000_000_000,
        limits=limits,
    )
    assert verified.grant.evidence_kind == "synthetic_contract_fixture"
    assert fixture.source.simulated is False
    assert len(verified.decoded.unroll.transitions) == 3
    assert verified.steps == fixture.steps
    for original, admitted in zip(
        fixture.unroll.transitions, verified.decoded.unroll.transitions, strict=True
    ):
        assert canonical(transition_to_record(original)) == canonical(
            transition_to_record(admitted)
        )
        assert admitted.action_receipt is not None
        assert admitted.action_receipt.realtime is not None
    assert [item.receipt.observed_reward for item in verified.steps] == [1.0, 3.0, 5.0]
    assert [item.budget_after.action_count for item in verified.steps] == [1, 2, 3]
    assert verified.steps[-1].budget_after.closed is True


def test_measured_demo_requires_explicit_permit_and_records_actual_selected_updates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "demo"
    result = run_demo(output, runtime_source_commit="a" * 40)
    assert result["evidence_kind"] == "synthetic_contract_fixture"
    assert result["source_simulated_flags"] == [False] * 4
    assert result["actual_game_capture"] is False
    assert result["default_callback_calls"] == 0
    assert result["explicit_test_only_callback_calls"] == 2
    assert result["combined_consumed_transition_count"] == 6
    assert result["typed_consumed_transition_count"] == 6
    assert result["consumed_source_ids"] == ["producer-a", "producer-b"]
    assert result["resumed_prefix_chunk_count"] == 1
    assert result["first_shard_chunk_count"] > 1
    assert result["repeated_learning_callback_calls"] == 0
    assert all(item["duplicate"] for item in result["repeat_intake"])
    assert all(item["duplicate"] for item in result["post_consumption_repeat_intake"])
    assert all(item["callback_completed"] for item in result["consumer_receipts"])
    assert result["learner_model_sha256_before"] != result["learner_model_sha256_after"]
    assert result["fixture_training_mse_after"] < result["fixture_training_mse_before"]
    assert result["external_evaluation_result"] == "not_run"
    assert result["game_improvement"] == "not_evaluated"
    assert result["policy_promotion"] == "not_performed"
    suite = json.loads((output / "fixed-evaluation.json").read_text(encoding="utf-8"))
    assert suite["suite"]["evidence_kind"] == "synthetic_contract_fixture"
    assert suite["suite"]["metrics"][0]["require_count"] == 3
    assert suite["training_input"] is False
    assert suite["external_evaluation_result"] == "not_run"
    assert suite["snapshot"]["snapshot_sha256"] == result["fixed_evaluation_snapshot_sha256"]
    model_path = output / "learner-model.npy"
    model = np.load(model_path, allow_pickle=False)
    assert model.shape == (2,)
    assert model.dtype == np.float64
    assert hashlib.sha256(model.tobytes()).hexdigest() == result["learner_model_sha256_after"]
    assert (
        hashlib.sha256(model_path.read_bytes()).hexdigest() == result["learner_model_file_sha256"]
    )
    bundle = verify_model_bundle(output / "learner-bundle")
    assert bundle.schema_version == result["learner_bundle_schema"] == "glr.model-bundle.v1"
    assert bundle.artifacts[0].sha256 == result["learner_model_file_sha256"]
    names = [item.path for item in bundle.inputs]
    assert "runtime-code/fleet_measured.py" in names
    assert any(name.endswith("measured-proof.json") for name in names)
    assert not any("holdout" in name or "other-game" in name for name in names)
    # Independently replay the recorded selected numeric inputs, without calling
    # the example learner or using its declarations of update count.
    recipe = json.loads((output / "learner-recipe.json").read_text(encoding="utf-8"))
    replay = np.array(recipe["initial_parameters"], dtype=np.float64)
    actual_points: list[tuple[float, float]] = []
    for source in recipe["source_order"]:
        published = next((output / "producers" / source / "shards").iterdir())
        payload = b"".join(path.read_bytes() for path in sorted(published.glob("chunk-*.bin")))
        decoded = decode_shard((published / "manifest.json").read_bytes(), payload)
        for transition in decoded.unroll.transitions:
            state = transition.observation["state"]
            assert isinstance(state, np.ndarray)
            features = np.array([float(state[0]), 1.0], dtype=np.float64)
            reward = float(transition.reward[0])
            replay -= (
                recipe["learning_rate"] * (float(np.dot(replay, features)) - reward) * features
            )
            actual_points.append((float(state[0]), reward))
    assert len(actual_points) == len(set(actual_points)) == 6
    np.testing.assert_array_equal(replay, model)


def test_demo_retains_prior_output_and_rejects_ambiguous_runtime_identity(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "retain.txt"
    marker.write_text("prior evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_demo(output, runtime_source_commit="a" * 40)
    assert marker.read_text(encoding="utf-8") == "prior evidence"
    with pytest.raises(ValueError, match="explicit lowercase Git commit"):
        run_demo(tmp_path / "invalid", runtime_source_commit="main")
    assert not (tmp_path / "invalid").exists()
