"""Behavioral acceptance of actual combined data use in the finite SDK example."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from game_learning_runtime.examples.fleet_training import run_demo
from game_learning_runtime.model_bundle import verify_model_bundle
from game_learning_runtime.project import load_project


def test_two_producers_train_one_model_and_preserve_holdout(tmp_path: Path) -> None:
    output = tmp_path / "example"
    result = run_demo(output, runtime_source_commit="a" * 40)
    assert result["mode"] == "SIMULATED"
    assert result["individual_producer_transition_count"] == 3
    assert result["combined_consumed_transition_count"] == 6
    assert result["consumed_source_ids"] == ["producer-a", "producer-b"]
    assert result["consumed_actor_ids"] == ["producer-a:epoch-1", "producer-b:epoch-1"]
    assert result["first_shard_chunk_count"] > 1
    assert result["learner_model_sha256_before"] != result["learner_model_sha256_after"]
    model_path = output / "learner-model.npy"
    model = np.load(model_path, allow_pickle=False)
    assert model.shape == (2,)
    assert model.dtype == np.float64
    assert hashlib.sha256(model.tobytes()).hexdigest() == result["learner_model_sha256_after"]
    assert (
        hashlib.sha256(model_path.read_bytes()).hexdigest() == result["learner_model_file_sha256"]
    )
    assert result["fixture_training_mse_after"] < result["fixture_training_mse_before"]
    assert result["game_improvement"] == "not_evaluated"
    bundle = verify_model_bundle(output / "learner-bundle")
    assert bundle.schema_version == result["learner_bundle_schema"] == "glr.model-bundle.v1"
    assert bundle.algorithm == "synthetic-reward-regression"
    assert bundle.artifacts[0].sha256 == result["learner_model_file_sha256"]
    inputs = [entry.path for entry in bundle.inputs]
    assert "fleet_training.py" in inputs and "learner-recipe.json" in inputs
    assert any(path.startswith("selected-data/producer-a/") for path in inputs)
    assert any(path.startswith("selected-data/producer-b/") for path in inputs)
    assert not any("holdout" in path or "other-game" in path for path in inputs)
    bundle_manifest = output / "learner-bundle/manifest.json"
    assert (
        hashlib.sha256(bundle_manifest.read_bytes()).hexdigest()
        == result["learner_bundle_manifest_sha256"]
    )
    receipts = result["consumer_receipts"]
    assert len(receipts) == 2
    assert all(receipt["status"] == "consumed" for receipt in receipts)
    assert all(receipt["callback_completed"] for receipt in receipts)
    assert sum(receipt["learner_declared_updates"] for receipt in receipts) == 6
    snapshot = json.loads(
        (output / ".glr/fleet/launcher-snapshot.json").read_text(encoding="utf-8")
    )
    project = load_project(output)
    assert project.data_dir == output / ".glr"
    assert project.environment_family == "synthetic"
    machines = {row["source_id"]: row for row in snapshot["machines"]}
    assert machines["unattached"]["heartbeat_received_at_utc"] is None
    assert machines["unattached"]["data_received_at_utc"] is None
    datasets = {row["source_id"]: row for row in snapshot["datasets"]}
    assert datasets["holdout"]["split"] == "evaluation_holdout"
    assert datasets["holdout"]["accepted_transition_count"] == 3
    assert datasets["holdout"]["last_plan_eligible"] is False
    assert datasets["other-game"]["last_plan_eligible"] is False
    assert datasets["other-game"]["last_plan_reason_codes"] == ["compatibility_mismatch"]
    assert sum(row["accepted_transition_count"] for row in datasets.values()) == 12
    assert len(snapshot["consumer_receipts"]) == 2


def test_demo_refuses_to_replace_prior_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "retain.txt"
    marker.write_text("prior evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_demo(output, runtime_source_commit="a" * 40)
    assert marker.read_text(encoding="utf-8") == "prior evidence"


@pytest.mark.parametrize("commit", ["main", "A" * 40, "0" * 39, "0" * 41])
def test_demo_rejects_ambiguous_commit_before_creating_output(tmp_path: Path, commit: str) -> None:
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="explicit lowercase Git commit"):
        run_demo(output, runtime_source_commit=commit)
    assert not output.exists()
