"""Independent measured-admission probes; every datum and signing key is synthetic.

NonSIM exercises the reviewed-profile branch, not a claim of physical capture.
The synthetic local owner is trusted; a digest or signature cannot authenticate
an honest game or protect against the owner deliberately approving false facts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
import pytest

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
)
from game_learning_runtime.fleet_learner import (
    FleetConsumer,
    LearnerResult,
    LearnerSelection,
    RealTrainingEnablement,
)
from game_learning_runtime.fleet_measured import (
    EncodedMeasuredShard,
    LegalActionEvidence,
    MeasuredActionBinding,
    MeasuredAuthority,
    MeasuredGrant,
    MeasuredQuality,
    MeasuredStep,
    RewardBudgetState,
    encode_measured_shard,
    sign_measured_body,
    verify_measured,
)
from game_learning_runtime.fleet_payload import (
    CompatibilitySpec,
    FleetError,
    SourceSpec,
    VectorSpec,
    canonical,
    decode_shard,
    parse_manifest,
    sha256,
)
from game_learning_runtime.phases import EnvironmentPhase
from game_learning_runtime.realtime import RealtimeActionReceipt, RealtimeActionStatus
from game_learning_runtime.serialization import transition_to_record
from game_learning_runtime.training import RewardSignal, TrainingConfig
from game_learning_runtime.training_safety import RewardSafetyConfig

NOW_MS = 1_700_000_000_000
ACTOR_COMMIT = "a" * 40
EXPORTER_SHA = hashlib.sha256(b"synthetic-exporter").hexdigest()
KEY = b"synthetic-test-only-key-material!!"
COMMANDS = ("wait", "advance")
CONFIG = {"scenario": "synthetic", "revision": "1"}


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _training() -> TrainingConfig:
    return TrainingConfig.from_mapping(
        {
            "schema_version": "glr.training.v1",
            "knowledge_sources": [{"id": "measured", "authority": "authoritative"}],
            "reward": {
                "terms": [
                    {"name": "progress", "source": "measured"},
                    {"name": "outcome", "source": "measured", "required": False},
                ]
            },
        }
    )


def _safety() -> RewardSafetyConfig:
    return RewardSafetyConfig.from_mapping(
        {
            "schema_version": "glr.reward-safety.v1",
            "outcome_signal": "outcome",
            "shaping_signals": ["progress"],
            "max_positive_shaping_per_step": 1.0,
            "max_positive_shaping_per_episode": 1.0,
            "max_negative_shaping_per_step": 1.0,
            "max_negative_shaping_per_episode": 1.0,
            "failure_episode_maximum": 0.0,
            "require_terminal_outcome": True,
        }
    )


def _source(label: str, *, split: str = "train", **changes: Any) -> SourceSpec:
    config_digest = environment_config_digest(CONFIG)
    assert config_digest is not None
    return replace(
        SourceSpec(
            source_id=label,
            source_epoch="epoch-0",
            machine_id="synthetic-machine",
            source_revision="synthetic-v1",
            source_sha256=_hash("synthetic-producer"),
            runtime_source_commit=ACTOR_COMMIT,
            adapter_source_sha256=_hash("synthetic-adapter"),
            run_id=f"run-{label}",
            game_id="synthetic-game",
            compatibility=CompatibilitySpec(
                environment_id="synthetic.measured",
                protocol_version="1.0",
                environment_config_sha256=config_digest,
                observation=(VectorSpec("state", "<f4", 2),),
                action=(VectorSpec("choice", "<i4", 1),),
                reward_contract_sha256=_hash("reviewed-reward-contract"),
                masks=(VectorSpec("choice", "|b1", 2),),
            ),
            policy_epoch="policy-0",
            policy_version=0,
            behavior_policy_sha256=_hash("synthetic-policy"),
            assignment_id=f"assignment-{label}",
            split=split,
            simulated=False,
        ),
        **changes,
    )


def _grant(
    source: SourceSpec, *, domain: str = "evaluation-domain", **changes: Any
) -> MeasuredGrant:
    return replace(
        MeasuredGrant(
            grant_id=f"grant-{source.source_id}",
            key_id="fixture-key",
            source=source,
            exporter_source_sha256=EXPORTER_SHA,
            target_id="synthetic-target",
            clock_domain="unix-utc-ns",
            training=_training(),
            safety=_safety(),
            action_bindings=(MeasuredActionBinding("choice", "choice", COMMANDS),),
            evaluation_domain_id=domain,
            evidence_kind="synthetic_contract_fixture",
            max_age_ms=60_000,
            max_clock_skew_ms=1_000,
            expires_at_utc_ms=NOW_MS + 60_000,
        ),
        **changes,
    )


ZERO_BUDGET = RewardBudgetState(0.0, 0.0, 0.0, 0.0, 0.0, 0, False)


def _interval(
    grant: MeasuredGrant,
    *,
    base: float = 0.0,
    step: int = 0,
    episode: UUID | None = None,
    guard: CorrelatedRewardGuard | None = None,
    budget_before: RewardBudgetState = ZERO_BUDGET,
    observed_reward: float = 0.75,
) -> tuple[Transition, MeasuredStep]:
    episode = episode or uuid5(NAMESPACE_URL, grant.source.run_id)
    before_ns = NOW_MS * 1_000_000 - 1_000_000 + step * 100_000
    before_context = ObservationContext(
        grant.source.run_id,
        grant.source.compatibility.environment_id,
        grant.source.compatibility.protocol_version,
        grant.target_id,
        grant.source.compatibility.environment_config_sha256,
        episode,
        step,
        10 + step,
        before_ns,
        EnvironmentPhase.GAMEPLAY,
        True,
    )
    after_context = replace(
        before_context,
        step_id=step + 1,
        producer_sequence=11 + step,
        timestamp_ns=before_ns + 100_000,
    )
    action_id = f"action.{grant.source.source_id}.{step}"
    execution = ActionReceipt(
        action_id,
        episode,
        step + 1,
        ActionOutcome.ACCEPTED,
        before_ns + 10_000,
        before_ns + 90_000,
        postcondition="settled",
        target_id=grant.target_id,
        issued_against_observation_sequence=before_context.producer_sequence,
        authoritative_observation_sequence=after_context.producer_sequence,
        realtime=RealtimeActionReceipt(
            action_id,
            RealtimeActionStatus.CONSUMED,
            100_000,
            10_000,
            before_ns + 10_000,
            before_ns + 20_000,
            before_ns + 80_000,
        ),
    )
    mask = {"choice": np.array([True, True])}
    before = TimeStep(
        {"state": np.array([base + step, base + step + 1], dtype=np.float32)},
        np.array([0.0], dtype=np.float32),
        np.array([False]),
        np.array([False]),
        episode_id=episode,
        step_id=step,
        action_mask=mask,
        info={OBSERVATION_CONTEXT_KEY: before_context.to_mapping()},
        timestamp_ns=before_context.timestamp_ns,
    )
    after = TimeStep(
        {"state": np.array([base + step + 1, base + step + 2], dtype=np.float32)},
        np.array([observed_reward], dtype=np.float32),
        np.array([False]),
        np.array([False]),
        episode_id=episode,
        step_id=step + 1,
        action_mask=mask,
        action_receipt=execution,
        timestamp_ns=after_context.timestamp_ns,
        info={OBSERVATION_CONTEXT_KEY: after_context.to_mapping()},
    )
    action = {"choice": np.array([1], dtype=np.int32)}
    signals = (RewardSignal("progress", "measured", 0.75),)
    claims = (
        RewardAttribution(
            "progress", "measured", action_id, 10 + step, 11 + step, EffectState.CONFIRMED
        ),
    )
    if guard is None:
        guard = CorrelatedRewardGuard(
            grant.training,
            grant.safety,
            CorrelationPolicy(
                grant.source.run_id,
                grant.source.compatibility.environment_id,
                grant.source.compatibility.protocol_version,
                grant.target_id,
                grant.source.compatibility.environment_config_sha256,
            ),
        )
        guard.reset(episode)
    proof = guard.compose(
        before, after, signals, claims, action=action, verify_observed_reward=True
    )
    result = proof.result
    budget_after = RewardBudgetState(
        result.episode_total,
        result.positive_shaping_total,
        result.negative_shaping_total,
        budget_before.suppressed_positive_shaping_total + result.suppressed_positive_shaping,
        budget_before.suppressed_negative_shaping_total + result.suppressed_negative_shaping,
        budget_before.action_count + 1,
        result.terminal,
    )
    transition = Transition(
        observation=before.observation,
        action=action,
        reward=after.reward,
        next_observation=after.observation,
        terminated=after.terminated,
        truncated=after.truncated,
        episode_id=episode,
        step_id=step,
        timestamp_ns=after.timestamp_ns,
        action_mask=before.action_mask,
        next_action_mask=after.action_mask,
        action_receipt=execution,
        info={
            OBSERVATION_CONTEXT_KEY: after_context.to_mapping(),
            "observation_sequence": after_context.producer_sequence,
            REWARD_EVIDENCE_KEY: {
                "signals": [
                    {"name": s.name, "source": s.source, "value": s.value} for s in signals
                ],
                "attributions": [claim.to_mapping() for claim in claims],
            },
        },
        provenance={
            "correlated_reward": proof.to_mapping(),
            "correlated_reward_sha256": proof.sha256,
        },
    )
    return transition, MeasuredStep(
        proof,
        "life-0",
        budget_before,
        budget_after,
        MeasuredQuality(True, True, True, "authoritative", "authoritative", "authoritative"),
        LegalActionEvidence(
            ("advance",), before_context.producer_sequence, tensor_tree_sha256(mask)
        ),
    )


def _packet(
    grant: MeasuredGrant,
    sample: tuple[Transition, MeasuredStep] | None = None,
    *,
    shard_seq: int = 0,
) -> EncodedMeasuredShard:
    transition, step = sample or _interval(grant)
    return encode_measured_shard(
        grant.source,
        Unroll(
            (transition,),
            actor_id=grant.source.source_id,
            sequence_id=shard_seq,
            policy_version=grant.source.policy_version,
            environment_config_digest=grant.source.compatibility.environment_config_sha256,
            environment_config_snapshot=CONFIG,
        ),
        grant=grant,
        key=KEY,
        shard_seq=shard_seq,
        produced_at_utc_ms=NOW_MS,
        expires_at_utc_ms=NOW_MS + 30_000,
        steps=(step,),
    )


@dataclass
class _World:
    hub: FleetHub
    authority: MeasuredAuthority
    destination: MeasuredDestination
    grant: MeasuredGrant
    permit: RealTrainingEnablement
    packet: EncodedMeasuredShard

    def consumer(self, *, enabled: bool = True, **changes: Any) -> FleetConsumer:
        selection = replace(
            LearnerSelection(
                self.grant.source.compatibility,
                self.grant.source.behavior_policy_sha256,
                self.grant.source.game_id,
                self.grant.source.runtime_source_commit,
                self.grant.source.adapter_source_sha256,
            ),
            **changes,
        )
        return FleetConsumer(
            self.hub,
            BoundedActorQueue(1, overflow_policy="fail"),
            learner_id="synthetic-learner",
            selection=selection,
            real_enablement=self.permit if enabled else None,
        )


def _world(tmp_path: Path) -> _World:
    grant = _grant(_source("training"))
    holdout = _grant(_source("holdout", split="evaluation_holdout"))
    authority = MeasuredAuthority("synthetic-owner", (grant, holdout), {"fixture-key": KEY})
    destination = MeasuredDestination("synthetic-destination", _hash("synthetic-destination"))
    hub = FleetHub.create(
        tmp_path / "hub",
        clock_ms=lambda: NOW_MS,
        measured_authority=authority,
        measured_destination=destination,
    )
    for item in (grant, holdout):
        hub.register_source(item.source)
    fixed = _packet(holdout, _interval(holdout, base=100.0))
    heldout_receipt = hub.ingest_measured(
        fixed.carrier.manifest, fixed.carrier.payload, fixed.envelope
    )
    assert heldout_receipt.status == "ready"
    description = parse_manifest(fixed.carrier.manifest)
    suite = MeasuredEvaluationSuite(
        suite_id="synthetic-fixed-suite",
        evaluation_domain_id=holdout.evaluation_domain_id,
        evidence_kind="synthetic_contract_fixture",
        evaluator_source_commit="c" * 40,
        evaluator_artifact=b"synthetic inert fixed evaluator artifact; never executed",
        cases=(
            MeasuredEvaluationCase(
                "case-holdout",
                holdout.source.source_id,
                holdout.source.source_epoch,
                description.shard_id,
                description.payload_sha256,
                sha256(fixed.envelope),
            ),
        ),
        metrics=(MeasuredEvaluationMetric("reward", "sum", "maximize", 1),),
    )
    evaluation = hub.freeze_measured_evaluation(
        "fixed-evaluation",
        sources=((holdout.source.source_id, holdout.source.source_epoch),),
        suite=suite,
    )
    packet = _packet(grant)
    verified = verify_measured(
        packet.envelope,
        decode_shard(packet.carrier.manifest, packet.carrier.payload),
        authority,
        NOW_MS,
    )
    permit = RealTrainingEnablement(
        destination_id=destination.destination_id,
        destination_sha256=destination.destination_sha256,
        authority_sha256=verified.authority_sha256,
        allowed_source_spec_sha256s=(sha256(canonical(grant.source.to_record())),),
        evaluation_id="fixed-evaluation",
        evaluation_suite_sha256=suite.sha256,
        evaluation_snapshot_sha256=evaluation.snapshot_sha256,
        approval_id="synthetic-owner-approval",
        approval_sha256=_hash("synthetic-owner-approval"),
        expires_at_utc_ms=NOW_MS + 20_000,
        evidence_kind="synthetic_contract_fixture",
        max_transitions=1,
        max_callback_calls=1,
    )
    receipt = hub.ingest_measured(packet.carrier.manifest, packet.carrier.payload, packet.envelope)
    assert receipt.status == "ready"
    return _World(hub, authority, destination, grant, permit, packet)


def _assert_no_callback(consumer: FleetConsumer) -> None:
    calls: list[str] = []

    def callback(unroll: Unroll, ticket: Any) -> LearnerResult:
        calls.append(ticket.shard_id)
        return LearnerResult(declared_updates=len(unroll.transitions))

    plan = consumer.plan()
    assert plan.shard_ids == ()
    with pytest.raises(FleetError, match=r"^no_ready_data$"):
        consumer.consume_one(plan, callback)
    assert calls == []
    assert consumer.queue.metrics().depth == 0
    assert consumer.queue.metrics().in_flight_unrolls == 0


def test_signed_nonsim_fixture_requires_explicit_consumer_enablement(tmp_path: Path) -> None:
    world = _world(tmp_path)
    assert world.grant.source.simulated is False
    _assert_no_callback(world.consumer(enabled=False, allow_simulated=True))


def test_owner_enabled_signed_fixture_consumes_nonzero_data_once_with_zero_updates(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path)
    consumer = world.consumer()
    calls: list[Any] = []

    def callback(unroll: Unroll, ticket: Any) -> LearnerResult:
        calls.append(ticket)
        assert len(unroll.transitions) == 1
        transition = unroll.transitions[0]
        assert transition.action_receipt is not None
        assert transition.action_receipt.step_id == transition.step_id + 1
        assert transition.provenance is not None
        assert transition.provenance["correlated_reward_sha256"]
        assert ticket.measured_proof_sha256 == sha256(world.packet.envelope)
        return LearnerResult(declared_updates=0)

    plan = consumer.plan()
    assert len(plan.shard_ids) == 1
    receipt = consumer.consume_one(plan, callback)
    assert receipt.status == "consumed"
    assert receipt.transition_count == 1
    assert receipt.callback_completed is True
    assert receipt.learner_declared_updates == 0
    with pytest.raises(FleetError):
        consumer.consume_one(plan, callback)
    assert len(calls) == 1
    _assert_no_callback(consumer)


@pytest.mark.parametrize("component", ["observation", "action", "next_observation"])
def test_changed_tensor_cannot_reuse_original_causal_receipt(component: str) -> None:
    grant = _grant(_source("tensor-mutation"))
    transition, step = _interval(grant)
    if component == "action":
        changed = {"choice": np.array([0], dtype=np.int32)}
    else:
        changed = {"state": np.array([20.0, 21.0], dtype=np.float32)}
    changes: dict[str, Any] = {component: changed}
    changed_transition = replace(transition, **changes)
    with pytest.raises(FleetError):
        _packet(grant, (changed_transition, step))
    assert tensor_tree_sha256(transition.observation) == step.receipt.state_sha256


def test_diagnostic_receipt_without_verified_observed_reward_is_not_admitted() -> None:
    grant = _grant(_source("unobserved-reward"))
    transition, step = _interval(grant)
    unverified = replace(step.receipt, observed_reward=None)
    assert unverified.to_mapping() == step.receipt.to_mapping()
    assert unverified.sha256 == step.receipt.sha256
    with pytest.raises(FleetError):
        _packet(grant, (transition, replace(step, receipt=unverified)))


@pytest.mark.parametrize(
    "change",
    [{"step_id": 2}, {"issued_against_observation_sequence": 9}, {"target_id": "other-target"}],
)
def test_execution_receipt_must_belong_to_this_exact_action_interval(
    change: dict[str, Any],
) -> None:
    grant = _grant(_source("receipt-mutation"))
    transition, step = _interval(grant)
    assert transition.action_receipt is not None
    changed = replace(transition, action_receipt=replace(transition.action_receipt, **change))
    with pytest.raises(FleetError):
        _packet(grant, (changed, step))


def test_selected_action_must_be_legal_in_the_pre_observation_mask() -> None:
    grant = _grant(_source("illegal-action"))
    transition, step = _interval(grant)
    blocked_mask = {"choice": np.array([True, False])}
    changed = replace(transition, action_mask=blocked_mask)
    # Update the declared mask digest too: this probes legality, not a stale hash.
    legal = replace(step.legal_action, action_mask_sha256=tensor_tree_sha256(blocked_mask))
    with pytest.raises(FleetError):
        _packet(grant, (changed, replace(step, legal_action=legal)))


def test_actor_runtime_is_not_replaced_by_exporter_or_receiver_version(tmp_path: Path) -> None:
    world = _world(tmp_path)
    assert world.grant.source.runtime_source_commit == ACTOR_COMMIT
    assert world.grant.exporter_source_sha256 == EXPORTER_SHA
    _assert_no_callback(world.consumer(runtime_source_commit="b" * 40))


def test_callback_failure_is_not_automatically_retried_after_reopening(tmp_path: Path) -> None:
    world = _world(tmp_path)
    consumer = world.consumer()
    effects: list[str] = []

    def callback(unroll: Unroll, ticket: Any) -> LearnerResult:
        effects.append(ticket.shard_id)
        raise RuntimeError("synthetic failure after observable callback effect")

    result = consumer.consume_one(consumer.plan(), callback)
    assert result.status == "unknown_effect"
    assert result.transition_count == 0
    assert len(effects) == 1
    world.hub.close()
    world.hub = FleetHub.open(
        tmp_path / "hub",
        clock_ms=lambda: NOW_MS,
        measured_authority=world.authority,
        measured_destination=world.destination,
    )
    _assert_no_callback(world.consumer())
    assert len(effects) == 1


def test_permission_removed_during_callback_retains_unknown_effect(tmp_path: Path) -> None:
    world = _world(tmp_path)
    consumer = world.consumer()
    effects: list[int] = []

    def callback(unroll: Unroll, ticket: Any) -> LearnerResult:
        effects.append(len(unroll.transitions))
        world.hub.configure_measured(None, destination=world.destination)
        return LearnerResult(declared_updates=0)

    result = consumer.consume_one(consumer.plan(), callback)
    assert effects == [1]
    assert result.status == "unknown_effect"
    assert result.callback_completed is True
    assert result.transition_count == 0
    world.hub.configure_measured(world.authority, destination=world.destination)
    _assert_no_callback(consumer)
    assert effects == [1]


def test_finite_owner_budget_cannot_reset_by_reopen_or_changing_learner_identity(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path)
    consumer = world.consumer()
    result = consumer.consume_one(consumer.plan(), lambda unroll, ticket: LearnerResult(0))
    assert result.status == "consumed"
    episode = uuid5(NAMESPACE_URL, world.grant.source.run_id)
    guard = CorrelatedRewardGuard(
        world.grant.training,
        world.grant.safety,
        CorrelationPolicy(
            world.grant.source.run_id,
            world.grant.source.compatibility.environment_id,
            world.grant.source.compatibility.protocol_version,
            world.grant.target_id,
            world.grant.source.compatibility.environment_config_sha256,
        ),
    )
    guard.reset(episode)
    _, first = _interval(world.grant, episode=episode, guard=guard)
    sample = _interval(
        world.grant,
        step=1,
        episode=episode,
        guard=guard,
        budget_before=first.budget_after,
        observed_reward=0.25,
    )
    packet = _packet(world.grant, sample, shard_seq=1)
    ingested = world.hub.ingest_measured(
        packet.carrier.manifest, packet.carrier.payload, packet.envelope
    )
    assert ingested.status == "ready"
    world.hub.close()
    world.hub = FleetHub.open(
        tmp_path / "hub",
        clock_ms=lambda: NOW_MS,
        measured_authority=world.authority,
        measured_destination=world.destination,
    )
    replacement = FleetConsumer(
        world.hub,
        BoundedActorQueue(1, overflow_policy="fail"),
        learner_id="another-synthetic-learner",
        selection=consumer.selection,
        real_enablement=world.permit,
    )
    _assert_no_callback(replacement)
    replacement.configure_real(replace(world.permit, expires_at_utc_ms=NOW_MS + 25_000))
    _assert_no_callback(replacement)


def test_disabling_real_permission_fences_a_previously_planned_callback(tmp_path: Path) -> None:
    world = _world(tmp_path)
    consumer = world.consumer()
    plan = consumer.plan()
    assert plan.shard_ids
    consumer.configure_real(None)
    calls: list[str] = []

    def callback(unroll: Unroll, ticket: Any) -> LearnerResult:
        calls.append(ticket.shard_id)
        return LearnerResult(0)

    with pytest.raises(FleetError):
        consumer.consume_one(plan, callback)
    assert calls == []


@pytest.mark.parametrize(
    "missing", [OBSERVATION_CONTEXT_KEY, "observation_sequence", REWARD_EVIDENCE_KEY]
)
def test_missing_causal_field_is_rejected_without_mutating_original(missing: str) -> None:
    grant = _grant(_source("missing-field"))
    transition, step = _interval(grant)
    original = dict(transition.info)
    incomplete = {key: value for key, value in original.items() if key != missing}
    with pytest.raises(FleetError):
        _packet(grant, (replace(transition, info=incomplete), step))
    assert dict(transition.info) == original
    assert transition.reward.item() == 0.75


def test_both_terminal_flags_cannot_describe_one_measured_action() -> None:
    grant = _grant(_source("ambiguous-terminal"))
    transition, step = _interval(grant)
    changed = replace(transition, terminated=np.array([True]), truncated=np.array([True]))
    with pytest.raises(FleetError):
        _packet(grant, (changed, step))


def test_correct_digest_does_not_replace_the_registered_signing_key(tmp_path: Path) -> None:
    world = _world(tmp_path)
    wrong_key = MeasuredAuthority(
        "synthetic-owner",
        world.authority.grants,
        {"fixture-key": b"wrong-synthetic-key-material!!!!!"},
    )
    with pytest.raises(FleetError):
        verify_measured(
            world.packet.envelope,
            decode_shard(world.packet.carrier.manifest, world.packet.carrier.payload),
            wrong_key,
            NOW_MS,
        )


def test_fresh_export_timestamp_cannot_make_old_observations_fresh() -> None:
    grant = _grant(_source("stale-observation"), max_age_ms=500)
    packet = _packet(grant)
    manifest = json.loads(packet.carrier.manifest)
    # Resign a later export of exactly the old observed interval. The exporter
    # declaration is fresh, while every actual captured record clock is old.
    manifest["produced_at_utc_ms"] = NOW_MS + 500
    manifest_bytes = canonical(manifest)
    wrapper = json.loads(packet.envelope)
    body = wrapper["body"]
    body["carrier_manifest_sha256"] = sha256(manifest_bytes)
    body["capture"]["produced_at_utc_ms"] = NOW_MS + 500
    wrapper["signature"] = sign_measured_body(body, KEY)
    authority = MeasuredAuthority("synthetic-owner", (grant,), {"fixture-key": KEY})
    with pytest.raises(FleetError, match=r"^measured_stale$"):
        verify_measured(
            canonical(wrapper),
            decode_shard(manifest_bytes, packet.carrier.payload),
            authority,
            NOW_MS + 500,
        )


def test_expired_or_removed_local_permission_never_invokes_callback(tmp_path: Path) -> None:
    world = _world(tmp_path)
    consumer = world.consumer()
    consumer.configure_real(replace(world.permit, expires_at_utc_ms=NOW_MS))
    _assert_no_callback(consumer)
    world.hub.configure_measured(None, destination=world.destination)
    _assert_no_callback(world.consumer())


def test_reward_budget_tail_cannot_be_reset_by_reopening_or_a_new_shard(tmp_path: Path) -> None:
    world = _world(tmp_path)
    episode = uuid5(NAMESPACE_URL, world.grant.source.run_id)
    # A fresh guard can generate a locally consistent 0.75 award at step 1.
    # It has lost the earlier 0.75 episode spend; the durable hub must remember it.
    reset_sample = _interval(
        world.grant, step=1, episode=episode, budget_before=replace(ZERO_BUDGET, action_count=1)
    )
    reset_packet = _packet(world.grant, reset_sample, shard_seq=1)
    world.hub.close()
    world.hub = FleetHub.open(
        tmp_path / "hub",
        clock_ms=lambda: NOW_MS,
        measured_authority=world.authority,
        measured_destination=world.destination,
    )
    with pytest.raises(FleetError):
        world.hub.ingest_measured(
            reset_packet.carrier.manifest, reset_packet.carrier.payload, reset_packet.envelope
        )
    # Rejection of the forged continuation does not consume the valid first shard.
    calls: list[int] = []
    consumer = world.consumer()

    def callback(unroll: Unroll, ticket: Any) -> LearnerResult:
        calls.append(len(unroll.transitions))
        return LearnerResult(0)

    result = consumer.consume_one(consumer.plan(), callback)
    assert result.status == "consumed"
    assert calls == [1]


def test_holdout_copy_stays_excluded_after_all_producer_identity_labels_change(
    tmp_path: Path,
) -> None:
    holdout = _grant(_source("heldout", split="evaluation_holdout"))
    copied = _grant(
        _source(
            "copied",
            runtime_source_commit="b" * 40,
            adapter_source_sha256=_hash("other-adapter"),
            behavior_policy_sha256=_hash("other-policy"),
            source_epoch="other-epoch",
        )
    )
    authority = MeasuredAuthority("synthetic-owner", (holdout, copied), {"fixture-key": KEY})
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS, measured_authority=authority)
    for grant in (holdout, copied):
        hub.register_source(grant.source)
    packets = [_packet(grant, _interval(grant, base=50.0)) for grant in (holdout, copied)]
    original, copy = packets
    first = hub.ingest_measured(
        original.carrier.manifest, original.carrier.payload, original.envelope
    )
    assert first.status == "ready"
    result = hub.ingest_measured(copy.carrier.manifest, copy.carrier.payload, copy.envelope)
    assert result.status != "ready"
    assert result.trust_level != "simulated_declared"


def test_same_game_cannot_escape_holdout_by_declaring_a_different_domain(tmp_path: Path) -> None:
    holdout = _grant(_source("heldout", split="evaluation_holdout"))
    alias = _grant(
        _source("domain-alias", runtime_source_commit="b" * 40),
        domain="invented-evaluation-domain",
    )
    authority = MeasuredAuthority("synthetic-owner", (holdout,), {"fixture-key": KEY})
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS, measured_authority=authority)
    replacement = MeasuredAuthority("synthetic-owner", (holdout, alias), {"fixture-key": KEY})
    with pytest.raises(FleetError):
        hub.configure_measured(replacement)
    assert hub.measured_authority is authority


def test_same_numeric_vectors_in_independent_game_and_domain_are_not_globally_banned(
    tmp_path: Path,
) -> None:
    holdout = _grant(_source("heldout", split="evaluation_holdout"))
    independent = _grant(
        _source("independent", game_id="another-synthetic-game"),
        domain="independent-evaluation-domain",
    )
    authority = MeasuredAuthority("synthetic-owner", (holdout, independent), {"fixture-key": KEY})
    hub = FleetHub.create(tmp_path / "hub", clock_ms=lambda: NOW_MS, measured_authority=authority)
    receipts = []
    for grant in (holdout, independent):
        hub.register_source(grant.source)
        packet = _packet(grant, _interval(grant, base=50.0))
        receipts.append(
            hub.ingest_measured(packet.carrier.manifest, packet.carrier.payload, packet.envelope)
        )
    assert [receipt.status for receipt in receipts] == ["ready", "ready"]


def _wire_change(packet: EncodedMeasuredShard, path: tuple[str | int, ...], value: Any) -> bytes:
    """The fixture owner can sign bad facts; verification must still reject them."""
    wrapper = json.loads(packet.envelope)
    cursor = wrapper["body"]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    wrapper["signature"] = sign_measured_body(wrapper["body"], KEY)
    return canonical(wrapper)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("capture", "steps", 0, "quality", "before_fresh"), "false"),
        (("capture", "steps", 0, "quality", "mask_fresh"), False),
        (("capture", "steps", 0, "receipt", "before", "alive"), "true"),
        (("capture", "steps", 0, "receipt", "after", "producer_sequence"), True),
        (("capture", "steps", 0, "receipt", "observed_reward"), None),
        (("capture", "steps", 0, "receipt", "result", "terminal"), 0),
        (("capture", "unroll", "transitions", 0, "action_receipt", "retryable"), "false"),
        (("capture", "unroll", "transitions", 0, "action_receipt", "step_id"), 2),
        (("key_id",), "unregistered-key"),
        (("grant_id",), "unregistered-grant"),
        (("exporter_source_sha256",), _hash("unapproved-exporter")),
        (("capture", "source", "runtime_source_commit"), "b" * 40),
    ],
)
def test_authenticated_but_invalid_native_wire_facts_are_rejected(
    tmp_path: Path, path: tuple[str | int, ...], value: Any
) -> None:
    world = _world(tmp_path)
    modified = _wire_change(world.packet, path, value)
    with pytest.raises(FleetError):
        verify_measured(
            modified,
            decode_shard(world.packet.carrier.manifest, world.packet.carrier.payload),
            world.authority,
            NOW_MS,
        )


def test_recomputed_unsigned_content_hashes_cannot_authenticate_reward_tampering(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path)
    original_transition, _ = _interval(world.grant)
    altered = replace(
        original_transition,
        reward=np.array([0.5], dtype=np.float32),
        action_receipt=None,
        events=(),
        info={},
        provenance=None,
    )
    payload = canonical(transition_to_record(altered)) + b"\n"
    manifest = json.loads(world.packet.carrier.manifest)
    manifest.update(
        payload_sha256=sha256(payload),
        payload_bytes=len(payload),
        chunks=[{"sha256": sha256(payload), "bytes": len(payload)}],
    )
    manifest_bytes = canonical(manifest)
    wrapper = json.loads(world.packet.envelope)
    body = wrapper["body"]
    body["carrier_manifest_sha256"] = sha256(manifest_bytes)
    body["carrier_payload_sha256"] = sha256(payload)
    body["capture"]["unroll"]["transitions"][0]["reward"] = transition_to_record(altered)["reward"]
    # Preserve the original authentication tag after recomputing every unsigned
    # carrier hash; source/artifact SHA values alone cannot authorize new data.
    forged = canonical(wrapper)
    with pytest.raises(FleetError, match=r"^measured_signature$"):
        verify_measured(forged, decode_shard(manifest_bytes, payload), world.authority, NOW_MS)


@pytest.mark.parametrize("category", ["missing", "extra", "duplicate", "oversized"])
def test_closed_bounded_envelope_parser_never_accepts_partial_or_extra_fields(
    tmp_path: Path, category: str
) -> None:
    world = _world(tmp_path)
    wrapper = json.loads(world.packet.envelope)
    if category == "missing":
        del wrapper["body"]["capture"]["steps"][0]["receipt"]["observed_reward"]
        wrapper["signature"] = sign_measured_body(wrapper["body"], KEY)
        payload = canonical(wrapper)
    elif category == "extra":
        wrapper["body"]["capture"]["steps"][0]["quality"]["extra"] = "synthetic-extra"
        wrapper["signature"] = sign_measured_body(wrapper["body"], KEY)
        payload = canonical(wrapper)
    elif category == "duplicate":
        payload = world.packet.envelope.replace(
            b'{"body":', b'{"signature":"' + b"0" * 64 + b'","body":', 1
        )
    else:
        payload = b" " * 1_048_577
    with pytest.raises(FleetError):
        verify_measured(
            payload,
            decode_shard(world.packet.carrier.manifest, world.packet.carrier.payload),
            world.authority,
            NOW_MS,
        )
