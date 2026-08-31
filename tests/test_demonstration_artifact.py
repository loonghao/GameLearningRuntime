from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from game_learning_runtime import (
    ContractEnvironment,
    DemonstrationArtifactFile,
    DemonstrationArtifactManifest,
    DemonstrationGate,
    DemonstrationOrigin,
    DemonstrationOutcome,
    DemonstrationPolicyConfig,
    DemonstrationProvenance,
    JsonlTransitionWriter,
    SyncCollector,
    build_demonstration_artifact,
    load_demonstration_artifact_manifest,
    read_jsonl_transitions,
    verify_demonstration_artifact,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.examples import CounterEnvironment, always_increment


def _gate() -> DemonstrationGate:
    return DemonstrationGate(
        DemonstrationPolicyConfig.from_mapping(
            {
                "schema_version": "glr.demonstration-policy.v1",
                "allowed_origins": ["human", "scripted-expert"],
                "allowed_outcomes": ["success"],
                "origin_weights": {"human": 2, "scripted-expert": 1},
                "outcome_weights": {"success": 1.5},
                "reject_unknown": True,
            }
        )
    )


def _trajectory(path: Path, *, terminal: bool = True) -> None:
    target = 1 if terminal else 3
    collector = SyncCollector(ContractEnvironment(CounterEnvironment(target=target)))
    unroll = collector.collect(always_increment, steps=1, stop_on_done=True)
    with JsonlTransitionWriter(path) as writer:
        for transition in unroll.transitions:
            writer.write(transition)


def test_demonstration_artifact_binds_bytes_episode_and_provenance(tmp_path: Path) -> None:
    trajectory = tmp_path / "episode.jsonl"
    manifest_path = tmp_path / "episode.manifest.json"
    _trajectory(trajectory)

    manifest = build_demonstration_artifact(
        manifest_path,
        trajectory_path=trajectory,
        environment_id="example.counter-v1",
        provenance=DemonstrationProvenance(
            origin=DemonstrationOrigin.HUMAN,
            outcome=DemonstrationOutcome.SUCCESS,
        ),
    )
    verified = verify_demonstration_artifact(
        manifest_path,
        gate=_gate(),
        expected_environment_id="example.counter-v1",
    )

    assert manifest.schema_version == "glr.demonstration-artifact.v1"
    assert manifest.trajectory.path == "episode.jsonl"
    assert len(manifest.trajectory.sha256) == 64
    assert verified.manifest == manifest
    assert verified.sample_weight == 3
    assert len(verified.transitions) == 1
    assert verified.transitions[0].episode_id == manifest.episode_id


def test_demonstration_artifact_rejects_tampering_and_environment_drift(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "episode.jsonl"
    manifest_path = tmp_path / "episode.manifest.json"
    _trajectory(trajectory)
    build_demonstration_artifact(
        manifest_path,
        trajectory_path=trajectory,
        environment_id="example.counter-v1",
        provenance=DemonstrationProvenance(
            origin=DemonstrationOrigin.SCRIPTED_EXPERT,
            outcome=DemonstrationOutcome.SUCCESS,
        ),
    )

    with pytest.raises(ContractViolation, match="environment_id"):
        verify_demonstration_artifact(
            manifest_path,
            gate=_gate(),
            expected_environment_id="example.counter-v2",
        )

    with trajectory.open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ContractViolation, match=r"sha256|size"):
        verify_demonstration_artifact(
            manifest_path,
            gate=_gate(),
            expected_environment_id="example.counter-v1",
        )


def test_demonstration_artifact_requires_terminal_success(tmp_path: Path) -> None:
    trajectory = tmp_path / "unfinished.jsonl"
    _trajectory(trajectory, terminal=False)

    with pytest.raises(ContractViolation, match="terminal"):
        build_demonstration_artifact(
            tmp_path / "unfinished.manifest.json",
            trajectory_path=trajectory,
            environment_id="example.counter-v1",
            provenance=DemonstrationProvenance(
                origin=DemonstrationOrigin.HUMAN,
                outcome=DemonstrationOutcome.SUCCESS,
            ),
        )


def test_demonstration_artifact_rejects_mixed_episode_trajectory(tmp_path: Path) -> None:
    first = (
        SyncCollector(ContractEnvironment(CounterEnvironment(target=1)))
        .collect(always_increment, steps=1, stop_on_done=True)
        .transitions[0]
    )
    second = (
        SyncCollector(ContractEnvironment(CounterEnvironment(target=1)))
        .collect(always_increment, steps=1, stop_on_done=True)
        .transitions[0]
    )
    trajectory = tmp_path / "mixed.jsonl"
    with JsonlTransitionWriter(trajectory) as writer:
        writer.write(first)
        writer.write(second)

    with pytest.raises(ContractViolation, match="multiple episode IDs"):
        build_demonstration_artifact(
            tmp_path / "mixed.manifest.json",
            trajectory_path=trajectory,
            environment_id="example.counter-v1",
            provenance=DemonstrationProvenance(
                origin=DemonstrationOrigin.HUMAN,
                outcome=DemonstrationOutcome.SUCCESS,
            ),
        )


def test_demonstration_artifact_rejects_step_gap_and_empty_file(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    _trajectory(original)
    transition = next(read_jsonl_transitions(original))
    gap = tmp_path / "gap.jsonl"
    with JsonlTransitionWriter(gap) as writer:
        writer.write(replace(transition, step_id=1))

    provenance = DemonstrationProvenance(
        origin=DemonstrationOrigin.HUMAN,
        outcome=DemonstrationOutcome.SUCCESS,
    )
    with pytest.raises(ContractViolation, match="contiguous from zero"):
        build_demonstration_artifact(
            tmp_path / "gap.manifest.json",
            trajectory_path=gap,
            environment_id="example.counter-v1",
            provenance=provenance,
        )

    empty = tmp_path / "empty.jsonl"
    empty.touch()
    with pytest.raises(ContractViolation, match="at least one transition"):
        build_demonstration_artifact(
            tmp_path / "empty.manifest.json",
            trajectory_path=empty,
            environment_id="example.counter-v1",
            provenance=provenance,
        )


def test_demonstration_artifact_rejects_manifest_episode_drift(tmp_path: Path) -> None:
    trajectory = tmp_path / "episode.jsonl"
    manifest_path = tmp_path / "episode.manifest.json"
    _trajectory(trajectory)
    build_demonstration_artifact(
        manifest_path,
        trajectory_path=trajectory,
        environment_id="example.counter-v1",
        provenance=DemonstrationProvenance(
            origin=DemonstrationOrigin.HUMAN,
            outcome=DemonstrationOutcome.SUCCESS,
        ),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode_id"] = str(uuid4())
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ContractViolation, match="episode_id"):
        verify_demonstration_artifact(
            manifest_path,
            gate=_gate(),
            expected_environment_id="example.counter-v1",
        )


def test_demonstration_artifact_requires_trajectory_below_manifest_directory(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "episode.jsonl"
    _trajectory(trajectory)

    with pytest.raises(ValueError, match="inside the manifest directory"):
        build_demonstration_artifact(
            tmp_path / "nested" / "episode.manifest.json",
            trajectory_path=trajectory,
            environment_id="example.counter-v1",
            provenance=DemonstrationProvenance(
                origin=DemonstrationOrigin.HUMAN,
                outcome=DemonstrationOutcome.SUCCESS,
            ),
        )


def test_demonstration_artifact_rejects_policy_data_at_verification(tmp_path: Path) -> None:
    trajectory = tmp_path / "policy.jsonl"
    manifest_path = tmp_path / "policy.manifest.json"
    _trajectory(trajectory)
    build_demonstration_artifact(
        manifest_path,
        trajectory_path=trajectory,
        environment_id="example.counter-v1",
        provenance=DemonstrationProvenance(
            origin=DemonstrationOrigin.POLICY,
            outcome=DemonstrationOutcome.SUCCESS,
            policy_id="candidate-v1",
        ),
    )

    with pytest.raises(ContractViolation, match="origin policy"):
        verify_demonstration_artifact(
            manifest_path,
            gate=_gate(),
            expected_environment_id="example.counter-v1",
        )


def test_demonstration_artifact_manifest_is_strict_and_portable() -> None:
    value = {
        "schema_version": "glr.demonstration-artifact.v1",
        "environment_id": "example.counter-v1",
        "episode_id": "7d444840-9dc0-11d1-b245-5ffdce74fad2",
        "trajectory": {
            "path": "../escape.jsonl",
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
        "provenance": {
            "origin": "human",
            "outcome": "success",
            "policy_id": None,
        },
    }

    with pytest.raises(ValueError, match="portable relative path"):
        DemonstrationArtifactManifest.from_mapping(value)

    value["trajectory"]["path"] = "episode.jsonl"
    value["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        DemonstrationArtifactManifest.from_mapping(json.loads(json.dumps(value)))


def test_demonstration_artifact_rejects_invalid_file_and_api_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        DemonstrationArtifactFile(path="episode.jsonl", sha256="A" * 64, size_bytes=1)
    with pytest.raises(ValueError, match="non-negative integer"):
        DemonstrationArtifactFile(path="episode.jsonl", sha256="a" * 64, size_bytes=-1)
    with pytest.raises(FileNotFoundError, match="demonstration manifest"):
        load_demonstration_artifact_manifest(tmp_path / "missing.json")

    trajectory = tmp_path / "episode.jsonl"
    _trajectory(trajectory)
    with pytest.raises(TypeError, match="provenance"):
        build_demonstration_artifact(
            tmp_path / "episode.manifest.json",
            trajectory_path=trajectory,
            environment_id="example.counter-v1",
            provenance=object(),  # type: ignore[arg-type]
        )
