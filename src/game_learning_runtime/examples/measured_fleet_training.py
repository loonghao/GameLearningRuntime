"""Finite synthetic contract fixtures exercising measured, nonSIM admission.

The publicly known fixture key authorizes only ``synthetic_contract_fixture``.
This example never connects to a game and never provisions production trust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.collector import BoundedActorQueue
from game_learning_runtime.contracts import (
    ActionOutcome,
    ActionReceipt,
    TimeStep,
    Transition,
    Unroll,
    environment_config_digest,
)
from game_learning_runtime.correlated_rewards import (
    OBSERVATION_CONTEXT_KEY,
    REWARD_EVIDENCE_KEY,
    CorrelatedRewardGuard,
    CorrelationPolicy,
    EffectState,
    ObservationContext,
    RewardAttribution,
    tensor_tree_sha256,
)
from game_learning_runtime.fleet_datahub import (
    FleetHub,
    MeasuredDestination,
    MeasuredEvaluationCase,
    MeasuredEvaluationMetric,
    MeasuredEvaluationSuite,
    write_local_shard,
)
from game_learning_runtime.fleet_learner import (
    ConsumptionTicket,
    FleetConsumer,
    LearnerResult,
    LearnerSelection,
    RealTrainingEnablement,
)
from game_learning_runtime.fleet_measured import (
    LegalActionEvidence,
    MeasuredActionBinding,
    MeasuredAuthority,
    MeasuredGrant,
    MeasuredQuality,
    MeasuredStep,
    RewardBudgetState,
    encode_measured_shard,
)
from game_learning_runtime.fleet_payload import (
    CompatibilitySpec,
    EncodedShard,
    FleetError,
    FleetLimits,
    SourceSpec,
    VectorSpec,
    parse_manifest,
)
from game_learning_runtime.model_bundle import build_model_bundle, verify_model_bundle
from game_learning_runtime.phases import EnvironmentPhase
from game_learning_runtime.realtime import RealtimeActionReceipt, RealtimeActionStatus
from game_learning_runtime.training import RewardSignal, TrainingConfig
from game_learning_runtime.training_safety import RewardSafetyConfig

EVIDENCE_KIND = "synthetic_contract_fixture"
_FIXTURE_KEY_ID = "public-synthetic-fixture-key"
_FIXTURE_KEY = b"GLR public SYNTHETIC contract fixture only; never production"
_CONFIG = {"scenario": "synthetic-measured-vector", "revision": "1"}
_COMMANDS = ("wait", "advance")
_BASELINE_COMMIT = "feca029c466ffa9694705634977ca32d9a4befdd"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_commit(runtime_source_commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", runtime_source_commit) is None:
        raise ValueError("runtime_source_commit must be an explicit lowercase Git commit")


@dataclass(frozen=True, slots=True)
class SyntheticMeasuredFixture:
    """A complete public fixture, never a positive claim about a real producer."""

    source: SourceSpec
    grant: MeasuredGrant
    authority: MeasuredAuthority
    destination: MeasuredDestination
    shard: EncodedShard
    envelope: bytes
    unroll: Unroll
    steps: tuple[MeasuredStep, ...]


def build_synthetic_measured_fixture(
    *,
    runtime_source_commit: str,
    produced_at_utc_ms: int,
    source_id: str = "producer-a",
    split: str = "train",
    start: int = 0,
    transition_count: int = 3,
    shard_seq: int = 0,
    limits: FleetLimits | None = None,
    game_id: str = "synthetic-measured-vector",
) -> SyntheticMeasuredFixture:
    """Construct first-generation typed receipts from declared synthetic states.

    UTC and runtime identity are explicit caller facts. ``start`` changes numeric
    inputs, while every independent source starts a fresh episode at step zero.
    A nonzero shard sequence does not invent a preceding episode tail in a hub.
    """
    _validate_commit(runtime_source_commit)
    limits = limits or FleetLimits()
    if type(transition_count) is not int or not 1 <= transition_count <= limits.max_transitions:
        raise ValueError("transition_count must fit the finite shard limit")
    training = TrainingConfig.from_mapping(
        {
            "schema_version": "glr.training.v1",
            "knowledge_sources": [{"id": "fixture", "authority": "authoritative"}],
            "reward": {
                "terms": [
                    {
                        "name": "progress",
                        "source": "fixture",
                        "minimum_authority": "authoritative",
                    },
                    {
                        "name": "outcome",
                        "source": "fixture",
                        "required": False,
                        "minimum_authority": "authoritative",
                    },
                ]
            },
        }
    )
    safety = RewardSafetyConfig.from_mapping(
        {
            "schema_version": "glr.reward-safety.v1",
            "outcome_signal": "outcome",
            "shaping_signals": ["progress"],
            "max_positive_shaping_per_step": 1024.0,
            "max_positive_shaping_per_episode": 4096.0,
            "max_negative_shaping_per_step": 1024.0,
            "max_negative_shaping_per_episode": 4096.0,
            "failure_episode_maximum": 0.0,
            "require_terminal_outcome": True,
        }
    )
    config_sha = environment_config_digest(_CONFIG)
    assert config_sha is not None
    code_sha = _sha(Path(__file__).read_bytes())
    source = SourceSpec(
        source_id=source_id,
        source_epoch="epoch-0",
        machine_id=f"synthetic-{source_id}",
        source_revision="synthetic-measured-vector-v1",
        source_sha256=code_sha,
        runtime_source_commit=runtime_source_commit,
        adapter_source_sha256=code_sha,
        run_id=f"run-{source_id}",
        game_id=game_id,
        compatibility=CompatibilitySpec(
            environment_id="synthetic.measured-vector-v1",
            protocol_version="1.0",
            environment_config_sha256=config_sha,
            observation=(VectorSpec("state", "<f4", 1),),
            action=(VectorSpec("choice", "<i4", 1),),
            masks=(VectorSpec("choice", "|b1", 2),),
            reward_contract_sha256=_sha(b"SYNTHETIC reward=2*state+1; terminal success fixture"),
        ),
        policy_epoch="fixture-policy-0",
        policy_version=0,
        behavior_policy_sha256=_sha(b"SYNTHETIC constant legal advance policy-v1"),
        assignment_id=f"assignment-{source_id}",
        split=split,
        simulated=False,
    )
    grant = MeasuredGrant(
        grant_id=f"grant-{source_id}",
        key_id=_FIXTURE_KEY_ID,
        source=source,
        exporter_source_sha256=code_sha,
        target_id=f"synthetic-target-{source_id}",
        clock_domain="unix-utc-ns",
        training=training,
        safety=safety,
        action_bindings=(MeasuredActionBinding("choice", "choice", _COMMANDS),),
        evaluation_domain_id=f"domain-{game_id}",
        evidence_kind=EVIDENCE_KIND,
        max_age_ms=60_000,
        max_clock_skew_ms=1000,
        expires_at_utc_ms=produced_at_utc_ms + 60_000,
        max_actions_per_episode=transition_count,
    )
    authority = MeasuredAuthority("synthetic-owner", (grant,), {_FIXTURE_KEY_ID: _FIXTURE_KEY})
    destination = MeasuredDestination(
        "synthetic-local-output", _sha(b"synthetic local output only")
    )
    episode = uuid5(NAMESPACE_URL, f"glr-measured-example/{source.run_id}/{shard_seq}")
    guard = CorrelatedRewardGuard(
        training,
        safety,
        CorrelationPolicy(
            source.run_id,
            source.compatibility.environment_id,
            source.compatibility.protocol_version,
            grant.target_id,
            config_sha,
        ),
        max_actions_per_episode=transition_count,
        max_episodes=1,
    )
    guard.reset(episode)
    budget = RewardBudgetState(0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
    transitions: list[Transition] = []
    steps: list[MeasuredStep] = []
    for step_id in range(transition_count):
        before_ns = (produced_at_utc_ms - transition_count + step_id) * 1_000_000
        previous = ObservationContext(
            source.run_id,
            source.compatibility.environment_id,
            source.compatibility.protocol_version,
            grant.target_id,
            config_sha,
            episode,
            step_id,
            10 + step_id,
            before_ns,
            EnvironmentPhase.GAMEPLAY,
            True,
        )
        following = replace(
            previous,
            step_id=step_id + 1,
            producer_sequence=11 + step_id,
            timestamp_ns=before_ns + 1_000_000,
        )
        action_id = f"action.{source_id}.{step_id}"
        execution = ActionReceipt(
            action_id,
            episode,
            step_id + 1,
            ActionOutcome.ACCEPTED,
            before_ns + 100_000,
            before_ns + 900_000,
            postcondition="settled",
            authoritative_observation_sequence=following.producer_sequence,
            target_id=grant.target_id,
            issued_against_observation_sequence=previous.producer_sequence,
            realtime=RealtimeActionReceipt(
                action_id,
                RealtimeActionStatus.CONSUMED,
                800_000,
                100_000,
                before_ns + 100_000,
                before_ns + 200_000,
                before_ns + 700_000,
            ),
        )
        terminal = step_id == transition_count - 1
        mask = {"choice": np.array([True, True], dtype=np.bool_)}
        before = TimeStep(
            {"state": np.array([start + step_id], dtype=np.float32)},
            np.array([0.0], dtype=np.float32),
            np.array([False], dtype=np.bool_),
            np.array([False], dtype=np.bool_),
            episode_id=episode,
            step_id=step_id,
            action_mask=mask,
            info={OBSERVATION_CONTEXT_KEY: previous.to_mapping()},
            timestamp_ns=before_ns,
        )
        observed_reward = float(2 * (start + step_id) + 1)
        after = TimeStep(
            {"state": np.array([start + step_id + 1], dtype=np.float32)},
            np.array([observed_reward], dtype=np.float32),
            np.array([terminal], dtype=np.bool_),
            np.array([False], dtype=np.bool_),
            episode_id=episode,
            step_id=step_id + 1,
            action_mask=mask,
            action_receipt=execution,
            info={OBSERVATION_CONTEXT_KEY: following.to_mapping()},
            timestamp_ns=following.timestamp_ns,
        )
        action = {"choice": np.array([1], dtype=np.int32)}
        signals: tuple[RewardSignal, ...] = (
            RewardSignal("progress", "fixture", 0.0 if terminal else observed_reward),
        )
        signal_name = "progress"
        if terminal:
            signals += (RewardSignal("outcome", "fixture", observed_reward),)
            signal_name = "outcome"
        claims = (
            RewardAttribution(
                signal_name,
                "fixture",
                action_id,
                previous.producer_sequence,
                following.producer_sequence,
                EffectState.CONFIRMED,
            ),
        )
        guard.begin_action(before)
        receipt = guard.compose(
            before, after, signals, claims, action=action, verify_observed_reward=True
        )
        result = receipt.result
        following_budget = RewardBudgetState(
            result.episode_total,
            result.positive_shaping_total,
            result.negative_shaping_total,
            budget.suppressed_positive_shaping_total + result.suppressed_positive_shaping,
            budget.suppressed_negative_shaping_total + result.suppressed_negative_shaping,
            budget.action_count + 1,
            result.terminal,
        )
        transitions.append(
            Transition(
                episode,
                step_id,
                before.observation,
                action,
                after.reward,
                after.observation,
                after.terminated,
                after.truncated,
                action_mask=before.action_mask,
                next_action_mask=after.action_mask,
                action_receipt=execution,
                info={
                    OBSERVATION_CONTEXT_KEY: following.to_mapping(),
                    "observation_sequence": following.producer_sequence,
                    REWARD_EVIDENCE_KEY: {
                        "signals": [asdict(item) for item in signals],
                        "attributions": [item.to_mapping() for item in claims],
                    },
                },
                provenance={
                    "correlated_reward": receipt.to_mapping(),
                    "correlated_reward_sha256": receipt.sha256,
                },
                timestamp_ns=after.timestamp_ns,
            )
        )
        steps.append(
            MeasuredStep(
                receipt,
                f"life-{source_id}-0",
                budget,
                following_budget,
                MeasuredQuality(
                    True, True, True, "authoritative", "authoritative", "authoritative"
                ),
                LegalActionEvidence(
                    ("advance",), previous.producer_sequence, tensor_tree_sha256(mask)
                ),
            )
        )
        budget = following_budget
    unroll = Unroll(
        tuple(transitions),
        actor_id=source.source_id,
        sequence_id=shard_seq,
        policy_version=source.policy_version,
        environment_config_snapshot=_CONFIG,
        environment_config_digest=config_sha,
    )
    packet = encode_measured_shard(
        source,
        unroll,
        grant=grant,
        key=_FIXTURE_KEY,
        shard_seq=shard_seq,
        produced_at_utc_ms=produced_at_utc_ms,
        expires_at_utc_ms=produced_at_utc_ms + 30_000,
        steps=tuple(steps),
        limits=limits,
    )
    return SyntheticMeasuredFixture(
        source, grant, authority, destination, packet.carrier, packet.envelope, unroll, tuple(steps)
    )


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _code_inputs() -> dict[str, Path]:
    """Record ordinary installed module bytes without publishing their paths."""
    result = {"measured_fleet_training.py": Path(__file__)}
    for name in (
        "contracts",
        "collector",
        "correlated_rewards",
        "fleet_datahub",
        "fleet_learner",
        "fleet_measured",
        "fleet_payload",
        "model_bundle",
        "realtime",
        "serialization",
        "training",
        "training_safety",
    ):
        module = import_module(f"game_learning_runtime.{name}")
        if module.__file__ is None:
            raise RuntimeError("the fixture requires installed package module files")
        result[f"runtime-code/{name}.py"] = Path(module.__file__)
    return result


def run_demo(output_dir: Path, *, runtime_source_commit: str) -> dict[str, object]:
    """Exercise a finite measured contract and actual NumPy fixture updates.

    All source declarations are nonSIM, yet every datum remains explicitly
    synthetic. The example supplies a TEST-ONLY local permit, never production
    approval. Frozen evaluator bytes are retained, not executed. No external
    evaluation, physical game claim, policy promotion or background run occurs.
    """
    _validate_commit(runtime_source_commit)
    output_dir.mkdir(parents=True, exist_ok=False)
    now_ms = time.time_ns() // 1_000_000
    limits = FleetLimits(max_chunk_bytes=512)
    fixtures = tuple(
        build_synthetic_measured_fixture(
            runtime_source_commit=runtime_source_commit,
            produced_at_utc_ms=now_ms,
            source_id=source_id,
            split="evaluation_holdout" if source_id == "holdout" else "train",
            start=start,
            limits=limits,
            game_id="synthetic-other-game"
            if source_id == "other-game"
            else "synthetic-measured-vector",
        )
        for source_id, start in (
            ("producer-a", 0),
            ("producer-b", 3),
            ("holdout", 100),
            ("other-game", 200),
        )
    )
    authority = MeasuredAuthority(
        "synthetic-example-owner",
        tuple(item.grant for item in fixtures),
        {_FIXTURE_KEY_ID: _FIXTURE_KEY},
    )
    destination = fixtures[0].destination
    hub_root = output_dir / ".glr" / "fleet"

    def clock() -> int:
        return time.time_ns() // 1_000_000

    hub = FleetHub.create(
        hub_root,
        limits=limits,
        clock_ms=clock,
        measured_authority=authority,
        measured_destination=destination,
    )
    bundle_inputs = _code_inputs()
    module_sha256s = {name: _sha(path.read_bytes()) for name, path in bundle_inputs.items()}
    try:
        for index, fixture in enumerate(fixtures):
            hub.register_source(fixture.source)
            spool = output_dir / "producers" / fixture.source.source_id
            spool.mkdir(parents=True)
            shard_id = write_local_shard(spool, fixture.shard, limits=limits)
            published = spool / "shards" / shard_id
            proof_path = published / "measured-proof.json"
            with proof_path.open("xb") as output:
                output.write(fixture.envelope)
            if index < 2:
                for name in (
                    "manifest.json",
                    "measured-proof.json",
                    *(f"chunk-{chunk}.bin" for chunk in range(len(fixture.shard.chunks))),
                ):
                    bundle_inputs[f"selected-data/{fixture.source.source_id}/{name}"] = (
                        published / name
                    )
            hub.receive_heartbeat(
                fixture.source.source_id,
                fixture.source.source_epoch,
                0,
                declared_at_utc_ms=now_ms,
            )
        first = fixtures[0]
        interrupted = hub.begin_measured_upload(first.shard.manifest, first.envelope)
        hub.put_chunk(interrupted.shard_id, 0, first.shard.chunks[0])
    finally:
        hub.close()

    hub = FleetHub.open(
        hub_root,
        clock_ms=clock,
        measured_authority=authority,
        measured_destination=destination,
    )
    callback_calls = 0
    typed_transition_count = 0
    observed_inputs: list[tuple[float, float]] = []
    consumed_sources: list[str] = []
    consumed_actors: list[str] = []
    model: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
    before_sha = _sha(model.tobytes())

    def learn(unroll: Unroll, ticket: ConsumptionTicket) -> LearnerResult:
        nonlocal callback_calls, typed_transition_count
        callback_calls += 1
        consumed_sources.append(ticket.source_id)
        consumed_actors.append(unroll.actor_id)
        if ticket.measured_proof_sha256 is None or ticket.real_enablement_sha256 is None:
            raise RuntimeError("the example requires both measured proof and explicit permit")
        for transition in unroll.transitions:
            if transition.action_receipt is None or transition.provenance is None:
                raise RuntimeError("original typed producer evidence was lost")
            state = transition.observation["state"]
            if not isinstance(state, np.ndarray):
                raise TypeError("the fixture profile requires an exact vector")
            x, y = float(state[0]), float(transition.reward[0])
            features = np.array([x, 1.0], dtype=np.float64)
            error = float(np.dot(model, features)) - y
            model[:] -= 0.01 * error * features
            observed_inputs.append((x, y))
            typed_transition_count += 1
        return LearnerResult(declared_updates=len(unroll.transitions))

    try:
        intake = []
        resumed_prefix = 0
        for index, fixture in enumerate(fixtures):
            upload = hub.begin_measured_upload(fixture.shard.manifest, fixture.envelope)
            if index == 0:
                resumed_prefix = upload.next_chunk_index
            for chunk in range(upload.next_chunk_index, len(fixture.shard.chunks)):
                hub.put_chunk(upload.shard_id, chunk, fixture.shard.chunks[chunk])
            intake.append(hub.finish_upload(upload.shard_id))
        repeated = tuple(
            hub.ingest_measured(item.shard.manifest, item.shard.payload, item.envelope)
            for item in fixtures
        )
        holdout = fixtures[2]
        holdout_manifest = parse_manifest(holdout.shard.manifest)
        # This is bounded inert definition data, never imported or executed.
        evaluator_artifact = b'{"kind":"SYNTHETIC fixed reward-sum definition","execute":false}'
        suite = MeasuredEvaluationSuite(
            suite_id="synthetic-fixed-suite",
            evaluation_domain_id=holdout.grant.evaluation_domain_id,
            evidence_kind=EVIDENCE_KIND,
            evaluator_source_commit=runtime_source_commit,
            evaluator_artifact=evaluator_artifact,
            cases=(
                MeasuredEvaluationCase(
                    "synthetic-holdout-case",
                    holdout.source.source_id,
                    holdout.source.source_epoch,
                    holdout_manifest.shard_id,
                    holdout_manifest.payload_sha256,
                    _sha(holdout.envelope),
                ),
            ),
            metrics=(MeasuredEvaluationMetric("reward", "sum", "maximize", 3),),
        )
        evaluation = hub.freeze_measured_evaluation(
            "synthetic-fixed-evaluation",
            sources=((holdout.source.source_id, holdout.source.source_epoch),),
            suite=suite,
        )
        _write_json(
            output_dir / "fixed-evaluation.json",
            {
                "suite": suite.to_record(),
                "snapshot": asdict(evaluation),
                "external_evaluation_result": "not_run",
                "training_input": False,
            },
        )
        with (output_dir / "fixed-evaluator-definition.json").open("xb") as output:
            output.write(evaluator_artifact)
        approval = {
            "kind": "TEST-ONLY synthetic local approval configuration",
            "evidence_kind": EVIDENCE_KIND,
            "destination": asdict(destination),
            "authority_sha256": authority.sha256,
            "allowed_source_spec_sha256s": [item.grant.source_spec_sha256 for item in fixtures[:2]],
            "evaluation_suite_sha256": suite.sha256,
            "evaluation_snapshot_sha256": evaluation.snapshot_sha256,
            "max_transitions": 6,
            "max_callback_calls": 2,
        }
        approval_path = output_dir / "test-only-owner-approval.json"
        _write_json(approval_path, approval)
        permit = RealTrainingEnablement(
            destination_id=destination.destination_id,
            destination_sha256=destination.destination_sha256,
            authority_sha256=authority.sha256,
            allowed_source_spec_sha256s=tuple(
                item.grant.source_spec_sha256 for item in fixtures[:2]
            ),
            evaluation_id=evaluation.evaluation_id,
            evaluation_suite_sha256=suite.sha256,
            evaluation_snapshot_sha256=evaluation.snapshot_sha256,
            approval_id="synthetic-test-only-approval",
            approval_sha256=_sha(approval_path.read_bytes()),
            expires_at_utc_ms=now_ms + 20_000,
            evidence_kind=EVIDENCE_KIND,
            max_transitions=6,
            max_callback_calls=2,
        )
        _write_json(output_dir / "test-only-enablement.json", asdict(permit))
        source = fixtures[0].source
        selection = LearnerSelection(
            compatibility=source.compatibility,
            policy_sha256=source.behavior_policy_sha256,
            game_id=source.game_id,
            runtime_source_commit=runtime_source_commit,
            adapter_source_sha256=source.adapter_source_sha256,
        )
        consumer = FleetConsumer(
            hub,
            BoundedActorQueue(2, overflow_policy="fail"),
            learner_id="synthetic-numpy-learner",
            selection=selection,
        )
        default_plan = consumer.plan()
        if default_plan.shard_ids:
            raise RuntimeError("default measured consumer selected data without owner enablement")
        try:
            consumer.consume_one(default_plan, learn)
        except FleetError as error:
            if str(error) != "no_ready_data":
                raise
        else:
            raise RuntimeError("default measured consumer did not remain closed")
        default_callback_calls = callback_calls
        consumer.configure_real(permit)
        receipts = tuple(consumer.consume_one(consumer.plan(), learn) for _ in range(2))
        if sorted(consumed_sources) != ["producer-a", "producer-b"] or len(observed_inputs) != 6:
            raise RuntimeError("the measured cohort or fixed holdout boundary was not preserved")
        after_sha = _sha(model.tobytes())
        if before_sha == after_sha or default_callback_calls != 0 or callback_calls != 2:
            raise RuntimeError("the finite fixture update evidence did not match the contract")
        replay = tuple(
            hub.ingest_measured(item.shard.manifest, item.shard.payload, item.envelope)
            for item in fixtures[:2]
        )
        replay_plan = consumer.plan()
        if replay_plan.shard_ids:
            raise RuntimeError("duplicate measured inputs became available for repeated learning")
        try:
            consumer.consume_one(replay_plan, learn)
        except FleetError as error:
            if str(error) != "no_ready_data":
                raise
        else:
            raise RuntimeError("duplicate measured data reached the learner")
        model_path = output_dir / "learner-model.npy"
        np.save(model_path, model, allow_pickle=False)
        recipe = {
            "evidence_kind": EVIDENCE_KIND,
            "mode": "SYNTHETIC measured contract fixture; nonSIM source declarations",
            "runtime_source_commit_owner_declared": runtime_source_commit,
            "previous_main_commit": _BASELINE_COMMIT,
            "initial_parameters": [0.0, 0.0],
            "parameter_dtype": "float64",
            "learning_rate": 0.01,
            "update_rule": "theta -= rate * (dot(theta, [state, 1]) - reward) * [state, 1]",
            "seed": 0,
            "random_draws": 0,
            "source_order": consumed_sources,
            "transition_count": len(observed_inputs),
            "fixed_evaluation_suite_sha256": suite.sha256,
            "fixed_evaluation_snapshot_sha256": evaluation.snapshot_sha256,
            "external_evaluation_result": "not_run",
            "game_improvement": "not_evaluated",
            "policy_promotion": "not_performed",
        }
        recipe_path = output_dir / "learner-recipe.json"
        _write_json(recipe_path, recipe)
        bundle_inputs["learner-recipe.json"] = recipe_path
        bundle_inputs["test-only-owner-approval.json"] = approval_path
        bundle_path = output_dir / "learner-bundle"
        bundle = build_model_bundle(
            bundle_path,
            environment_id=source.compatibility.environment_id,
            protocol_version=source.compatibility.protocol_version,
            algorithm="synthetic-measured-reward-regression",
            framework="numpy",
            framework_version=np.__version__,
            seeds=(0,),
            inputs=bundle_inputs,
            artifacts={"learner-model.npy": model_path},
        )
        if verify_model_bundle(bundle_path) != bundle:
            raise RuntimeError("the actual selected inputs and model artifact did not verify")
        hub.write_snapshot()
        summary: dict[str, object] = {
            "evidence_kind": EVIDENCE_KIND,
            "source_simulated_flags": [item.source.simulated for item in fixtures],
            "runtime_source_commit_owner_declared": runtime_source_commit,
            "previous_main_commit": _BASELINE_COMMIT,
            "example_module_sha256": _sha(Path(__file__).read_bytes()),
            "installed_module_sha256s": module_sha256s,
            "actual_game_capture": False,
            "default_callback_calls": default_callback_calls,
            "explicit_test_only_callback_calls": callback_calls,
            "first_shard_chunk_count": len(first.shard.chunks),
            "resumed_prefix_chunk_count": resumed_prefix,
            "intake": [asdict(item) for item in intake],
            "repeat_intake": [asdict(item) for item in repeated],
            "post_consumption_repeat_intake": [asdict(item) for item in replay],
            "repeated_learning_callback_calls": callback_calls - 2,
            "individual_producer_transition_count": 3,
            "combined_consumed_transition_count": len(observed_inputs),
            "typed_consumed_transition_count": typed_transition_count,
            "consumed_source_ids": sorted(consumed_sources),
            "consumed_actor_ids": sorted(consumed_actors),
            "consumer_receipts": [asdict(item) for item in receipts],
            "selected_proof_sha256s": [_sha(item.envelope) for item in fixtures[:2]],
            "test_only_permit_sha256": permit.binding_sha256,
            "fixed_evaluation_suite_sha256": suite.sha256,
            "fixed_evaluation_snapshot_sha256": evaluation.snapshot_sha256,
            "fixed_evaluator_artifact_sha256": _sha(evaluator_artifact),
            "fixed_evaluation_relative_path": "fixed-evaluation.json",
            "holdout_transition_count": len(holdout.unroll.transitions),
            "holdout_in_training_bundle": False,
            "learner_model_sha256_before": before_sha,
            "learner_model_sha256_after": after_sha,
            "learner_model_relative_path": "learner-model.npy",
            "learner_model_file_sha256": _sha(model_path.read_bytes()),
            "learner_bundle_schema": bundle.schema_version,
            "learner_bundle_relative_path": "learner-bundle",
            "learner_bundle_manifest_sha256": _sha((bundle_path / "manifest.json").read_bytes()),
            "fixture_training_mse_before": sum(y * y for _, y in observed_inputs) / 6,
            "fixture_training_mse_after": sum(
                (float(np.dot(model, [x, 1.0])) - y) ** 2 for x, y in observed_inputs
            )
            / 6,
            "external_evaluation_result": "not_run",
            "game_improvement": "not_evaluated",
            "policy_promotion": "not_performed",
            "snapshot_relative_path": ".glr/fleet/launcher-snapshot.json",
        }
        _write_json(output_dir / "demo-summary.json", summary)
        return summary
    finally:
        hub.close()


def main() -> int:
    """Run the ordinary installed package from an explicitly new output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-source-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_demo(args.output_dir, runtime_source_commit=args.runtime_source_commit),
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
