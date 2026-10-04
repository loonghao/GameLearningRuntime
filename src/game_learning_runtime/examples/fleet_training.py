"""Finite SIMULATED data collection and an owner-controlled NumPy learner example."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.collector import BoundedActorQueue
from game_learning_runtime.contracts import Transition, Unroll
from game_learning_runtime.fleet_datahub import FleetHub, write_local_shard
from game_learning_runtime.fleet_learner import (
    ConsumptionTicket,
    FleetConsumer,
    LearnerResult,
    LearnerSelection,
)
from game_learning_runtime.fleet_payload import (
    CompatibilitySpec,
    FleetLimits,
    SourceSpec,
    VectorSpec,
    encode_shard,
)
from game_learning_runtime.model_bundle import build_model_bundle, verify_model_bundle


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unroll(source: SourceSpec, *, start: int, observed_ms: int) -> Unroll:
    """Produce explicitly synthetic vectors without stripping real game evidence."""
    episode = uuid5(NAMESPACE_URL, f"glr-fleet-demo/{source.run_id}")
    transitions = tuple(
        Transition(
            episode_id=episode,
            step_id=step,
            observation={"state": np.array([start + step], dtype=np.float32)},
            action={"choice": np.array([1], dtype=np.float32)},
            reward=np.array([2 * (start + step) + 1], dtype=np.float32),
            next_observation={"state": np.array([start + step + 1], dtype=np.float32)},
            terminated=np.array([step == 2], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
            timestamp_ns=(observed_ms - 3 + step) * 1_000_000,
        )
        for step in range(3)
    )
    return Unroll(
        transitions=transitions,
        actor_id=source.source_id,
        sequence_id=0,
        policy_version=source.policy_version,
        environment_config_digest=source.compatibility.environment_config_sha256,
    )


def run_demo(output_dir: Path, *, runtime_source_commit: str) -> dict[str, object]:
    """Collect two compatible producers and consume their six distinct transitions.

    The callback actually updates a tiny reward predictor. Its training error is
    a fixture diagnostic, never a game evaluation or a policy promotion result.
    No network, background service, external game, accelerator or pretrained model is used.
    """
    if re.fullmatch(r"[0-9a-f]{40}", runtime_source_commit) is None:
        raise ValueError("runtime_source_commit must be an explicit lowercase Git commit")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "bridge").mkdir()
    # The existing read-only observer needs a project manifest. These declared
    # fixture roles are never executed by the example or the observer.
    manifest = {
        "schema_version": "glr.project.v1",
        "environment_id": "synthetic.vector-v1",
        "environment_family": "synthetic",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": ["fixture-not-executed"]},
        "trainer": {"argv": ["fixture-not-executed"]},
        "player": {"argv": ["fixture-not-executed"]},
    }
    (output_dir / "glr-project.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    limits = FleetLimits(max_chunk_bytes=512)
    compatibility = CompatibilitySpec(
        environment_id="synthetic.vector-v1",
        protocol_version="1.0",
        environment_config_sha256=_sha(b"synthetic-vector-config-v1"),
        observation=(VectorSpec("state", "<f4", 1),),
        action=(VectorSpec("choice", "<f4", 1),),
        reward_contract_sha256=_sha(b"SIMULATED reward=2*state+1; no game effect"),
    )
    fixture_sha = _sha(Path(__file__).read_bytes())
    behavior_sha = _sha(b"SIMULATED constant choice=1 policy-v1")
    sources = tuple(
        SourceSpec(
            source_id=name,
            source_epoch="epoch-1",
            machine_id=f"simulated-{name}",
            source_revision="synthetic-vector-v1",
            source_sha256=fixture_sha,
            runtime_source_commit=runtime_source_commit,
            adapter_source_sha256=fixture_sha,
            run_id=f"run-{name}",
            game_id="synthetic-other" if name == "other-game" else "synthetic-vector",
            compatibility=compatibility,
            policy_epoch="policy-1",
            policy_version=0,
            behavior_policy_sha256=behavior_sha,
            assignment_id=f"assignment-{name}",
            split="evaluation_holdout" if name == "holdout" else "train",
            simulated=True,
        )
        for name in ("producer-a", "producer-b", "holdout", "other-game", "unattached")
    )
    hub_root = output_dir / ".glr" / "fleet"
    hub = FleetHub.create(hub_root, limits=limits)
    try:
        for source in sources:
            hub.register_source(source)
        produced_ms = time.time_ns() // 1_000_000
        spools: list[Path] = []
        bundle_inputs = {"fleet_training.py": Path(__file__)}
        first_shard_id = ""
        first_chunk_count = 0
        for index, source in enumerate(sources[:4]):
            spool = output_dir / "producers" / source.source_id
            spool.mkdir(parents=True)
            shard = encode_shard(
                source,
                shard_seq=0,
                unroll=_unroll(source, start=index * 3, observed_ms=produced_ms),
                produced_at_utc_ms=produced_ms,
                limits=limits,
            )
            shard_id = write_local_shard(spool, shard)
            spools.append(spool)
            if index < 2:
                published = spool / "shards" / shard_id
                for name in (
                    "manifest.json",
                    *(f"chunk-{i}.bin" for i in range(len(shard.chunks))),
                ):
                    bundle_inputs[f"selected-data/{source.source_id}/{name}"] = published / name
            hub.receive_heartbeat(
                source.source_id, source.source_epoch, 0, declared_at_utc_ms=produced_ms
            )
            if index == 0:
                first_shard_id = shard_id
                first_chunk_count = len(shard.chunks)
                hub.begin_upload(shard.manifest)
                hub.put_chunk(shard_id, 0, shard.chunks[0])
    finally:
        hub.close()

    # Closing before the remaining chunks models an interrupted sender. The new
    # handle uses the persisted prefix and verifies every remaining chunk.
    hub = FleetHub.open(hub_root)
    try:
        intake = hub.sync_local_spools(tuple(spools), max_shards=8)
        repeat = hub.sync_local_spools(tuple(spools), max_shards=8)
        hub.freeze_evaluation("holdout", "epoch-1")
        model: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        before_sha = _sha(model.tobytes())
        observed_inputs: list[tuple[float, float]] = []
        actors: list[str] = []
        consumed_sources: list[str] = []

        def learn(unroll: Unroll, ticket: ConsumptionTicket) -> LearnerResult:
            actors.append(unroll.actor_id)
            consumed_sources.append(ticket.source_id)
            for transition in unroll.transitions:
                state = transition.observation["state"]
                if not isinstance(state, np.ndarray):
                    raise TypeError("the frozen demo profile requires a vector leaf")
                x, y = float(state[0]), float(transition.reward[0])
                features = np.array([x, 1.0], dtype=np.float64)
                error = float(np.dot(model, features)) - y
                model[:] -= 0.01 * error * features
                observed_inputs.append((x, y))
            return LearnerResult(declared_updates=len(unroll.transitions))

        selection = LearnerSelection(
            compatibility=compatibility,
            policy_sha256=behavior_sha,
            game_id="synthetic-vector",
            runtime_source_commit=runtime_source_commit,
            adapter_source_sha256=fixture_sha,
            allow_simulated=True,
        )
        queue = BoundedActorQueue(2, overflow_policy="fail")
        consumer = FleetConsumer(hub, queue, learner_id="numpy-demo", selection=selection)
        receipts = tuple(consumer.consume_one(consumer.plan(max_shards=1), learn) for _ in range(2))
        if sorted(consumed_sources) != ["producer-a", "producer-b"] or len(observed_inputs) != 6:
            raise RuntimeError("the fixed holdout/compatibility boundary was not preserved")
        after_sha = _sha(model.tobytes())
        if before_sha == after_sha:
            raise RuntimeError("the example learner did not change its model")
        training_error_before = sum(y * y for _, y in observed_inputs) / len(observed_inputs)
        training_error_after = sum(
            (float(np.dot(model, [x, 1.0])) - y) ** 2 for x, y in observed_inputs
        ) / len(observed_inputs)
        model_path = output_dir / "learner-model.npy"
        np.save(model_path, model, allow_pickle=False)
        recipe_path = output_dir / "learner-recipe.json"
        recipe = {
            "mode": "SIMULATED",
            "runtime_source_commit_owner_declared": runtime_source_commit,
            "initial_parameters": [0.0, 0.0],
            "parameter_dtype": "float64",
            "learning_rate": 0.01,
            "update_rule": "theta -= rate * (dot(theta, [state, 1]) - reward) * [state, 1]",
            "seed": 0,
            "random_draws": 0,
            "source_order": consumed_sources,
            "transition_count": len(observed_inputs),
            "game_improvement": "not_evaluated",
        }
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        bundle_inputs["learner-recipe.json"] = recipe_path
        bundle_path = output_dir / "learner-bundle"
        bundle = build_model_bundle(
            bundle_path,
            environment_id=compatibility.environment_id,
            protocol_version=compatibility.protocol_version,
            algorithm="synthetic-reward-regression",
            framework="numpy",
            framework_version=np.__version__,
            seeds=(0,),
            inputs=bundle_inputs,
            artifacts={"learner-model.npy": model_path},
        )
        if verify_model_bundle(bundle_path) != bundle:
            raise RuntimeError("the saved learner inputs and artifact did not verify")
        hub.write_snapshot()
        summary: dict[str, object] = {
            "mode": "SIMULATED",
            "runtime_source_commit_owner_declared": runtime_source_commit,
            "first_interrupted_shard_id": first_shard_id,
            "first_shard_chunk_count": first_chunk_count,
            "intake": asdict(intake),
            "repeat_intake": asdict(repeat),
            "individual_producer_transition_count": 3,
            "combined_consumed_transition_count": len(observed_inputs),
            "consumed_actor_ids": sorted(actors),
            "consumed_source_ids": sorted(consumed_sources),
            "consumer_receipts": [asdict(receipt) for receipt in receipts],
            "learner_model_sha256_before": before_sha,
            "learner_model_sha256_after": after_sha,
            "learner_model_relative_path": "learner-model.npy",
            "learner_model_file_sha256": _sha(model_path.read_bytes()),
            "learner_bundle_schema": bundle.schema_version,
            "learner_bundle_relative_path": "learner-bundle",
            "learner_bundle_manifest_sha256": _sha((bundle_path / "manifest.json").read_bytes()),
            "fixture_training_mse_before": training_error_before,
            "fixture_training_mse_after": training_error_after,
            "game_improvement": "not_evaluated",
            "snapshot_relative_path": ".glr/fleet/launcher-snapshot.json",
        }
        (output_dir / "demo-summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        hub.close()
