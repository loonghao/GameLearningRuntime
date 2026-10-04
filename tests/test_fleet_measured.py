"""Independent synthetic declarations; these never claim a physical producer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

import numpy as np
import pytest

import game_learning_runtime.fleet_measured as measured
from game_learning_runtime.contracts import (
    ActionOutcome,
    ActionReceipt,
    Event,
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
from game_learning_runtime.fleet_measured import (
    LegalActionEvidence,
    MeasuredActionBinding,
    MeasuredAuthority,
    MeasuredGrant,
    MeasuredQuality,
    MeasuredStep,
    RewardBudgetState,
    encode_measured_shard,
    require_measured_capture,
    sign_measured_body,
    verify_measured,
    verify_measured_manifest,
)
from game_learning_runtime.fleet_payload import (
    CompatibilitySpec,
    FleetError,
    FleetLimits,
    SourceSpec,
    VectorSpec,
    canonical,
    decode_shard,
)
from game_learning_runtime.phases import EnvironmentPhase
from game_learning_runtime.realtime import RealtimeActionReceipt, RealtimeActionStatus
from game_learning_runtime.serialization import transition_to_record
from game_learning_runtime.training import KnowledgeAuthority, RewardSignal, TrainingConfig
from game_learning_runtime.training_safety import RewardSafetyConfig

_TEST_KEY = b"explicit-synthetic-contract-test-key-7"
_CONFIG = {"fixture_mode": "independent-synthetic", "counter_layout": "scalar-v3"}
_CONFIG_SHA = environment_config_digest(_CONFIG)
assert _CONFIG_SHA is not None
_EPISODE = UUID("83142f07-89c3-4714-ad93-a3c5a7b929e0")
_NOW_MS = 1_830_123_400_500


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(frozen=True)
class _Fixture:
    source: SourceSpec
    grant: MeasuredGrant
    authority: MeasuredAuthority
    unroll: Unroll
    steps: tuple[MeasuredStep, ...]

    def encode(self, **changes: Any) -> Any:
        values = {
            "grant": self.grant,
            "key": _TEST_KEY,
            "shard_seq": 0,
            "produced_at_utc_ms": _NOW_MS,
            "expires_at_utc_ms": _NOW_MS + 2_000,
            "steps": self.steps,
            **changes,
        }
        return encode_measured_shard(self.source, self.unroll, **values)


def _fixture(
    *,
    count: int = 1,
    outcome: float | None = None,
    progress: float = 0.375,
    extra_metadata: bool = False,
) -> _Fixture:
    training = TrainingConfig.from_mapping(
        {
            "schema_version": "glr.training.v1",
            "knowledge_sources": [{"id": "exporter", "authority": "authoritative"}],
            "reward": {
                "terms": [
                    {"name": "finish", "source": "exporter", "weight": 3.0, "required": False},
                    {"name": "advance", "source": "exporter", "required": False},
                ]
            },
        }
    )
    safety = RewardSafetyConfig.from_mapping(
        {
            "schema_version": "glr.reward-safety.v1",
            "outcome_signal": "finish",
            "shaping_signals": ["advance"],
            "max_positive_shaping_per_step": 0.5,
            "max_positive_shaping_per_episode": 0.5,
            "max_negative_shaping_per_step": 1.0,
            "max_negative_shaping_per_episode": 2.0,
            "failure_episode_maximum": 0.0,
            "require_terminal_outcome": True,
        }
    )
    source = SourceSpec(
        source_id="contract-source",
        source_epoch="epoch-first",
        machine_id="declared-machine",
        source_revision="fixture-version",
        source_sha256=_hash("synthetic source definition"),
        runtime_source_commit="d" * 40,
        adapter_source_sha256=_hash("synthetic adapter definition"),
        run_id="contract-run",
        game_id="counter-contract",
        policy_epoch="policy-first",
        policy_version=7,
        behavior_policy_sha256=_hash("synthetic policy definition"),
        assignment_id="owner-assignment",
        simulated=False,
        compatibility=CompatibilitySpec(
            "counter-contract",
            "protocol-v3",
            _CONFIG_SHA,
            (VectorSpec("reading", "<f8", 1),),
            (VectorSpec("choice", "<i8", 1),),
            _hash("synthetic reward semantic contract"),
            masks=(VectorSpec("allowed", "|b1", 2),),
        ),
    )
    grant = MeasuredGrant(
        grant_id="contract-grant",
        key_id="contract-key",
        source=source,
        exporter_source_sha256=_hash("synthetic exporter definition"),
        target_id="contract-target",
        clock_domain="unix-utc-ns",
        training=training,
        safety=safety,
        action_bindings=(MeasuredActionBinding("choice", "allowed", ("wait", "advance")),),
        evaluation_domain_id="counter-domain",
        evidence_kind="synthetic_contract_fixture",
        max_age_ms=5_000,
        max_clock_skew_ms=20,
        expires_at_utc_ms=_NOW_MS + 10_000,
        extra_info_keys=("declared_quality",) if extra_metadata else (),
        extra_provenance_keys=("exporter_revision",) if extra_metadata else (),
        event_names=("counter_event",) if extra_metadata else (),
    )
    policy = CorrelationPolicy(
        source.run_id, "counter-contract", "protocol-v3", grant.target_id, _CONFIG_SHA
    )
    guard = CorrelatedRewardGuard(training, safety, policy)
    guard.reset(_EPISODE)
    mask = {"allowed": np.asarray([True, True], dtype=np.bool_)}
    action = {"choice": np.asarray([1], dtype=np.int64)}
    budget = RewardBudgetState(0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
    steps = []
    transitions = []
    base_time = (_NOW_MS - 100) * 1_000_000
    for index in range(count):
        terminal = outcome is not None and index == count - 1
        alive = not (terminal and outcome < 0)
        previous_context = ObservationContext(
            source.run_id,
            "counter-contract",
            "protocol-v3",
            grant.target_id,
            _CONFIG_SHA,
            _EPISODE,
            index,
            101 + index * 7,
            base_time + index * 10_000_000,
            EnvironmentPhase.GAMEPLAY,
            True,
        )
        following_context = replace(
            previous_context,
            step_id=index + 1,
            producer_sequence=previous_context.producer_sequence + 7,
            timestamp_ns=previous_context.timestamp_ns + 10_000_000,
            alive=alive,
        )
        action_id = f"contract-action-{index}"
        issued = previous_context.timestamp_ns + 1_000_000
        action_receipt = ActionReceipt(
            action_id,
            _EPISODE,
            index + 1,
            ActionOutcome.ACCEPTED,
            issued,
            issued + 4_000_000,
            postcondition="settled",
            authoritative_observation_sequence=following_context.producer_sequence,
            issued_against_observation_sequence=previous_context.producer_sequence,
            target_id=grant.target_id,
            realtime=RealtimeActionReceipt(
                action_id,
                RealtimeActionStatus.CONSUMED,
                5_000_000,
                1_000_000,
                issued,
                issued + 1_000_000,
                issued + 3_000_000,
            ),
        )
        if terminal:
            signals = (RewardSignal("finish", "exporter", outcome),)
            total = outcome * 3.0
        else:
            signals = (RewardSignal("advance", "exporter", progress),)
            total = min(progress, max(0.0, 0.5 - budget.positive_shaping_total))
        claims = tuple(
            RewardAttribution(
                signal.name,
                signal.source,
                action_id,
                previous_context.producer_sequence,
                following_context.producer_sequence,
                EffectState.CONFIRMED,
            )
            for signal in signals
        )
        before = TimeStep(
            observation={"reading": np.asarray([index], dtype=np.float64)},
            reward=np.asarray([0.0], dtype=np.float32),
            terminated=np.asarray([False]),
            truncated=np.asarray([False]),
            episode_id=_EPISODE,
            step_id=index,
            action_mask=mask,
            timestamp_ns=previous_context.timestamp_ns,
            info={
                OBSERVATION_CONTEXT_KEY: previous_context.to_mapping(),
                "observation_sequence": previous_context.producer_sequence,
            },
        )
        following = TimeStep(
            observation={"reading": np.asarray([index + 1], dtype=np.float64)},
            reward=np.asarray([total], dtype=np.float32),
            terminated=np.asarray([terminal]),
            truncated=np.asarray([False]),
            episode_id=_EPISODE,
            step_id=index + 1,
            action_mask=mask,
            action_receipt=action_receipt,
            timestamp_ns=following_context.timestamp_ns,
            info={
                OBSERVATION_CONTEXT_KEY: following_context.to_mapping(),
                "observation_sequence": following_context.producer_sequence,
                REWARD_EVIDENCE_KEY: {
                    "signals": [
                        {"name": signal.name, "source": signal.source, "value": signal.value}
                        for signal in signals
                    ],
                    "attributions": [claim.to_mapping() for claim in claims],
                },
            },
        )
        receipt = guard.compose(
            before, following, signals, claims, action=action, verify_observed_reward=True
        )
        new_budget = RewardBudgetState(
            receipt.result.episode_total,
            receipt.result.positive_shaping_total,
            receipt.result.negative_shaping_total,
            budget.suppressed_positive_shaping_total + receipt.result.suppressed_positive_shaping,
            budget.suppressed_negative_shaping_total + receipt.result.suppressed_negative_shaping,
            budget.action_count + 1,
            terminal,
        )
        steps.append(
            MeasuredStep(
                receipt,
                "contract-life",
                budget,
                new_budget,
                MeasuredQuality(
                    True, True, True, "authoritative", "authoritative", "authoritative"
                ),
                LegalActionEvidence(
                    ("advance",), previous_context.producer_sequence, tensor_tree_sha256(mask)
                ),
            )
        )
        info = dict(following.info)
        provenance = {
            "correlated_reward": receipt.to_mapping(),
            "correlated_reward_sha256": receipt.sha256,
        }
        events: tuple[Event, ...] = ()
        if extra_metadata:
            info["declared_quality"] = {"revision": "synthetic-v3", "flags": [True, 4]}
            provenance["exporter_revision"] = "synthetic-v3"
            events = (Event("counter_event", {"counter": index + 1}, following.timestamp_ns),)
        transitions.append(
            Transition(
                _EPISODE,
                index,
                before.observation,
                action,
                following.reward,
                following.observation,
                following.terminated,
                following.truncated,
                action_mask=mask,
                next_action_mask=mask,
                action_receipt=action_receipt,
                info=info,
                provenance=provenance,
                events=events,
                timestamp_ns=following.timestamp_ns,
            )
        )
        budget = new_budget
    return _Fixture(
        source,
        grant,
        MeasuredAuthority("contract-authority", (grant,), {grant.key_id: _TEST_KEY}),
        Unroll(
            tuple(transitions), source.source_id, 0, source.policy_version, _CONFIG, _CONFIG_SHA
        ),
        tuple(steps),
    )


def _verify(fixture: _Fixture, **changes: Any) -> Any:
    encoded = fixture.encode()
    return verify_measured(
        encoded.envelope,
        decode_shard(encoded.carrier.manifest, encoded.carrier.payload),
        changes.get("authority", fixture.authority),
        changes.get("now_ms", _NOW_MS),
    )


def test_original_proof_and_config_survive_roundtrip() -> None:
    fixture = _fixture(extra_metadata=True)
    verified = _verify(fixture)
    assert verified.steps == fixture.steps
    assert verified.decoded.unroll.environment_config_snapshot == _CONFIG
    assert verified.decoded.unroll.actor_id == fixture.source.source_id
    assert verified.carrier_decoded.unroll.actor_id != verified.decoded.unroll.actor_id
    original, restored = fixture.unroll.transitions[0], verified.decoded.unroll.transitions[0]
    assert transition_to_record(restored) == transition_to_record(original)
    assert restored.action_receipt is not None
    assert original.step_id == 0 and restored.action_receipt.step_id == 1
    assert verified.steps[0].before.producer_sequence == 101
    assert verified.steps[0].after.producer_sequence == 108
    for array in (restored.reward, restored.terminated, restored.observation["reading"]):
        assert not array.flags.writeable


def test_episode_shaping_cap_and_suppression_are_preserved() -> None:
    verified = _verify(_fixture(count=2))
    second = verified.steps[1]
    assert second.budget_before == verified.steps[0].budget_after
    assert second.result.total == 0.125
    assert second.result.positive_shaping_total == 0.5
    assert second.result.suppressed_positive_shaping == 0.25
    assert second.budget_after.suppressed_positive_shaping_total == 0.25


def test_terminal_negative_keeps_death_and_outcome() -> None:
    verified = _verify(_fixture(outcome=-0.5))
    transition, step = verified.decoded.unroll.transitions[0], verified.steps[0]
    assert transition.done and step.after.alive is False and step.result.terminal
    assert step.result.contributions == {"finish": -1.5}
    assert float(transition.reward.item()) == -1.5
    assert step.budget_after.closed


def test_confirmed_live_terminal_positive_is_valid() -> None:
    verified = _verify(_fixture(outcome=0.5))
    assert verified.steps[0].result.total == 1.5
    assert verified.steps[0].after.alive is True
    assert verified.decoded.unroll.transitions[0].done


def test_decimal_reward_keeps_actual_float32_value() -> None:
    fixture = _fixture(progress=0.3)
    step = _verify(fixture).steps[0]
    assert step.result.total == 0.3
    assert step.receipt.observed_reward == float(np.float32(0.3))
    assert step.receipt.observed_reward != step.result.total


def test_authority_is_opaque_and_never_exports_keys() -> None:
    fixture = _fixture()
    assert not dataclasses.is_dataclass(fixture.authority)
    assert _TEST_KEY.decode() not in repr(fixture.authority)
    with pytest.raises(TypeError):
        dataclasses.asdict(fixture.authority)
    with pytest.raises(TypeError):
        json.dumps(fixture.authority)
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(fixture.authority)
    with pytest.raises(TypeError, match="immutable"):
        fixture.authority._keys = {}  # type: ignore[misc]


def test_rotated_key_cannot_reuse_proof_despite_same_public_binding() -> None:
    fixture = _fixture()
    other = MeasuredAuthority(
        "contract-authority",
        (fixture.grant,),
        {fixture.grant.key_id: b"other-explicit-synthetic-key-value"},
    )
    assert other.sha256 == fixture.authority.sha256
    with pytest.raises(FleetError, match="measured_signature"):
        _verify(fixture, authority=other)


@pytest.mark.parametrize("invalid", [b"short", "a" * 32, bytearray(b"a" * 32), b"a" * 4097])
def test_key_bounds_and_native_bytes(invalid: object) -> None:
    fixture = _fixture()
    with pytest.raises(FleetError, match="measured_key"):
        MeasuredAuthority("bad-key", (fixture.grant,), {fixture.grant.key_id: invalid})  # type: ignore[dict-item]


def test_no_authority_means_no_admission() -> None:
    with pytest.raises(FleetError, match="measured_authority_required"):
        _verify(_fixture(), authority=None)


@pytest.mark.parametrize(
    "value", [{}, {"step_id": 0, "observation_sequence": 3}, {"schema_version": "unknown"}]
)
def test_missing_capture_rejected_before_registration(value: dict[str, Any]) -> None:
    with pytest.raises(FleetError, match="missing_measured_capture"):
        require_measured_capture(value)


def test_preflight_preserves_full_original_records_without_admitting() -> None:
    fixture = _fixture(extra_metadata=True)
    wrapper = json.loads(fixture.encode().envelope)
    capture = require_measured_capture(wrapper["body"]["capture"])
    assert capture.steps == fixture.steps
    assert transition_to_record(capture.unroll.transitions[0]) == transition_to_record(
        fixture.unroll.transitions[0]
    )


@pytest.mark.parametrize(
    "age_delta,reason",
    [(5_001, "measured_stale"), (-21, "measured_clock_future"), (2_000, "measured_expired")],
)
def test_admission_age_and_expiry_are_separate(age_delta: int, reason: str) -> None:
    fixture = _fixture()
    with pytest.raises(FleetError, match=reason if age_delta != 5_001 else "measured_expired"):
        _verify(fixture, now_ms=_NOW_MS + age_delta)


def test_signature_alone_does_not_validate_corrupted_cumulative_result() -> None:
    fixture = _fixture()
    encoded = fixture.encode()
    wrapper = json.loads(encoded.envelope)
    wrapper["body"]["capture"]["steps"][0]["receipt"]["result"]["episode_total"] += 1.0
    wrapper["signature"] = sign_measured_body(wrapper["body"], _TEST_KEY)
    with pytest.raises(FleetError, match="measured_reward_arithmetic"):
        verify_measured(
            canonical(wrapper),
            decode_shard(encoded.carrier.manifest, encoded.carrier.payload),
            fixture.authority,
            _NOW_MS,
        )


def test_unverified_original_receipt_cannot_be_promoted() -> None:
    fixture = _fixture()
    with pytest.raises(FleetError, match="measured_unverified_reward"):
        replace(fixture.steps[0], receipt=replace(fixture.steps[0].receipt, observed_reward=None))


def test_unsigned_unknown_envelope_field_is_rejected() -> None:
    fixture = _fixture()
    encoded = fixture.encode()
    wrapper = json.loads(encoded.envelope)
    wrapper["untrusted"] = "extra"
    with pytest.raises(FleetError, match="measured_fields"):
        verify_measured_manifest(
            canonical(wrapper), encoded.carrier.manifest, fixture.authority, _NOW_MS
        )


def test_deny_incidental_or_secret_metadata() -> None:
    fixture = _fixture()
    transition = fixture.unroll.transitions[0]
    unroll = replace(
        fixture.unroll,
        transitions=(replace(transition, info={**transition.info, "extra": fixture.authority}),),
    )
    with pytest.raises(FleetError, match="measured_metadata_type"):
        replace(fixture, unroll=unroll).encode()


def test_additional_native_metadata_requires_explicit_owner_allowlist() -> None:
    fixture = _fixture()
    transition = fixture.unroll.transitions[0]
    unroll = replace(
        fixture.unroll,
        transitions=(replace(transition, info={**transition.info, "incidental": 1}),),
    )
    with pytest.raises(FleetError, match="measured_metadata_grant"):
        replace(fixture, unroll=unroll).encode()


def test_missing_snapshot_cannot_be_replaced_with_digest_label() -> None:
    fixture = _fixture()
    with pytest.raises(FleetError, match="measured_environment_snapshot"):
        replace(fixture, unroll=replace(fixture.unroll, environment_config_snapshot=None)).encode()


def test_authority_domain_cannot_change_semantic_compatibility() -> None:
    fixture = _fixture()
    source = replace(
        fixture.source,
        source_id="different-source",
        compatibility=replace(
            fixture.source.compatibility, reward_contract_sha256=_hash("different semantic reward")
        ),
    )
    grant = replace(fixture.grant, grant_id="different-grant", source=source)
    with pytest.raises(FleetError, match="measured_evaluation_domain"):
        MeasuredAuthority("conflicted-domain", (fixture.grant, grant), {grant.key_id: _TEST_KEY})


def test_one_semantic_domain_allows_new_policy_runtime_labels() -> None:
    fixture = _fixture()
    source = replace(
        fixture.source,
        source_id="different-source",
        runtime_source_commit="e" * 40,
        policy_version=8,
    )
    grant = replace(fixture.grant, grant_id="different-grant", source=source)
    authority = MeasuredAuthority(
        "shared-domain", (fixture.grant, grant), {grant.key_id: _TEST_KEY}
    )
    assert len(authority.grants) == 2


def test_measured_proof_obeys_independent_file_limit() -> None:
    fixture = _fixture()
    with pytest.raises(FleetError):
        fixture.encode(limits=FleetLimits(max_shard_bytes=2_048))


def test_global_positive_reward_floor_is_denied_at_registration() -> None:
    fixture = _fixture()
    training = replace(
        fixture.grant.training, reward=replace(fixture.grant.training.reward, minimum=0.1)
    )
    with pytest.raises(FleetError, match="measured_positive_global_minimum"):
        replace(fixture.grant, training=training)


def test_advisory_reward_configuration_cannot_claim_measured_quality() -> None:
    fixture = _fixture()
    training = fixture.grant.training
    advisory = replace(
        training,
        knowledge_sources=tuple(
            replace(item, authority=KnowledgeAuthority.ADVISORY)
            for item in training.knowledge_sources
        ),
        reward=replace(
            training.reward,
            terms=tuple(
                replace(item, minimum_authority=KnowledgeAuthority.ADVISORY)
                for item in training.reward.terms
            ),
        ),
    )
    with pytest.raises(FleetError, match="measured_reward_authority"):
        replace(fixture.grant, training=advisory)


def test_capture_preflight_bounds_raw_mapping_before_codec() -> None:
    fixture = _fixture()
    capture = json.loads(fixture.encode().envelope)["body"]["capture"]
    capture["unroll"]["transitions"][0]["observation"]["reading"]["tensor"]["data"] = (
        "a" * 1_048_577
    )
    with pytest.raises(FleetError, match="measured_envelope_bound"):
        require_measured_capture(capture)


def test_signer_does_not_canonicalize_unbounded_or_object_bodies() -> None:
    with pytest.raises(FleetError, match="measured_envelope_bound"):
        sign_measured_body({"large": "a" * 1_048_577}, _TEST_KEY)
    with pytest.raises(FleetError, match="measured_metadata_type"):
        sign_measured_body({"opaque": object()}, _TEST_KEY)


def test_typed_manifest_cannot_coerce_boolean_sequence() -> None:
    fixture = _fixture()
    encoded = fixture.encode()
    decoded = decode_shard(encoded.carrier.manifest, encoded.carrier.payload)
    malformed = replace(decoded.manifest, shard_seq=False)
    with pytest.raises(FleetError, match="measured_counter"):
        verify_measured_manifest(encoded.envelope, malformed, fixture.authority, _NOW_MS)


def _replace_claims(fixture: _Fixture, claims: tuple[RewardAttribution, ...]) -> _Fixture:
    step = fixture.steps[0]
    receipt = replace(step.receipt, attributions=claims)
    transition = fixture.unroll.transitions[0]
    evidence = {
        **transition.info[REWARD_EVIDENCE_KEY],
        "attributions": [claim.to_mapping() for claim in claims],
    }
    transition = replace(
        transition,
        info={**transition.info, REWARD_EVIDENCE_KEY: evidence},
        provenance={
            "correlated_reward": receipt.to_mapping(),
            "correlated_reward_sha256": receipt.sha256,
        },
    )
    return replace(
        fixture,
        unroll=replace(fixture.unroll, transitions=(transition,)),
        steps=(replace(step, receipt=receipt),),
    )


def test_explicit_unknown_effect_is_denied_even_for_negative_reward() -> None:
    fixture = _fixture(outcome=-0.5)
    claims = (replace(fixture.steps[0].receipt.attributions[0], effect=EffectState.UNKNOWN),)
    with pytest.raises(FleetError, match="measured_unknown_effect"):
        _replace_claims(fixture, claims).encode()


def test_known_no_effect_can_preserve_negative_terminal_data() -> None:
    fixture = _fixture(outcome=-0.5)
    claims = (replace(fixture.steps[0].receipt.attributions[0], effect=EffectState.NO_EFFECT),)
    assert _verify(_replace_claims(fixture, claims)).steps[0].result.total == -1.5


def test_terminal_negative_without_claim_requires_original_dead_life_evidence() -> None:
    fixture = _replace_claims(_fixture(outcome=-0.5), ())
    assert _verify(fixture).steps[0].after.alive is False
    step = fixture.steps[0]
    receipt = replace(step.receipt, after=replace(step.after, alive=True))
    transition = fixture.unroll.transitions[0]
    transition = replace(
        transition,
        info={**transition.info, OBSERVATION_CONTEXT_KEY: receipt.after.to_mapping()},
        provenance={
            "correlated_reward": receipt.to_mapping(),
            "correlated_reward_sha256": receipt.sha256,
        },
    )
    live = replace(
        fixture,
        unroll=replace(fixture.unroll, transitions=(transition,)),
        steps=(replace(step, receipt=receipt),),
    )
    with pytest.raises(FleetError, match="measured_terminal_life_evidence"):
        live.encode()


def test_oversized_live_tensor_is_denied_before_copy_or_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    transition = replace(
        fixture.unroll.transitions[0], observation={"reading": np.zeros(16_384, dtype=np.float64)}
    )
    unroll = replace(fixture.unroll, transitions=(transition,))
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("copy-or-serialization")
        raise AssertionError("oversized data reached copying or serialization")

    monkeypatch.setattr(measured, "replace", forbidden)
    monkeypatch.setattr(measured, "transition_to_record", forbidden)
    with pytest.raises(FleetError, match="measured_tensor_bound"):
        encode_measured_shard(
            fixture.source,
            unroll,
            grant=fixture.grant,
            key=_TEST_KEY,
            shard_seq=0,
            produced_at_utc_ms=_NOW_MS,
            expires_at_utc_ms=_NOW_MS + 2_000,
            steps=fixture.steps,
        )
    assert calls == []


def test_live_tensor_total_is_bounded_before_copying_any_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    source = replace(
        fixture.source,
        compatibility=replace(
            fixture.source.compatibility, observation=(VectorSpec("reading", "<f8", 8192),)
        ),
    )
    grant = replace(fixture.grant, source=source)
    transition = replace(
        fixture.unroll.transitions[0],
        observation={"reading": np.zeros(8192, dtype=np.float64)},
        next_observation={"reading": np.ones(8192, dtype=np.float64)},
    )
    unroll = replace(fixture.unroll, transitions=(transition,) * 9)
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("projection-copy")
        raise AssertionError("aggregate tensor limit was checked after copying")

    monkeypatch.setattr(measured, "replace", forbidden)
    with pytest.raises(FleetError, match="measured_tensor_bound"):
        encode_measured_shard(
            source,
            unroll,
            grant=grant,
            key=_TEST_KEY,
            shard_seq=0,
            produced_at_utc_ms=_NOW_MS,
            expires_at_utc_ms=_NOW_MS + 2_000,
            steps=fixture.steps * 9,
        )
    assert calls == []


def test_public_verify_bounds_constructed_carrier_before_serializing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    encoded = fixture.encode()
    decoded = decode_shard(encoded.carrier.manifest, encoded.carrier.payload)
    oversized = replace(
        decoded.unroll.transitions[0], observation={"reading": np.zeros(16_384, dtype=np.float64)}
    )
    bad_carrier = replace(decoded, unroll=replace(decoded.unroll, transitions=(oversized,)))
    original_serializer = measured.transition_to_record
    calls = []

    def spy(transition: Transition) -> dict[str, Any]:
        if transition is oversized:
            calls.append("oversized-right-serialization")
            raise AssertionError("unvalidated carrier reached serializer")
        return original_serializer(transition)

    monkeypatch.setattr(measured, "transition_to_record", spy)
    with pytest.raises(FleetError, match="measured_tensor_bound"):
        verify_measured(encoded.envelope, bad_carrier, fixture.authority, _NOW_MS)
    assert calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"actor_id": "forged-actor"},
        {"sequence_id": 1},
        {"policy_version": 8},
        {"environment_config_snapshot": _CONFIG},
        {"environment_config_digest": "f" * 64},
    ],
)
def test_public_verify_rejects_forged_carrier_metadata(changes: dict[str, Any]) -> None:
    fixture = _fixture()
    encoded = fixture.encode()
    decoded = decode_shard(encoded.carrier.manifest, encoded.carrier.payload)
    bad_carrier = replace(decoded, unroll=replace(decoded.unroll, **changes))
    with pytest.raises(FleetError, match="measured_carrier_metadata"):
        verify_measured(encoded.envelope, bad_carrier, fixture.authority, _NOW_MS)


def test_public_verify_denies_numeric_carrier_metadata_before_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    encoded = fixture.encode()
    decoded = decode_shard(encoded.carrier.manifest, encoded.carrier.payload)
    contaminated = replace(decoded.unroll.transitions[0], info={"unexpected": "a" * 1_048_577})
    bad_carrier = replace(decoded, unroll=replace(decoded.unroll, transitions=(contaminated,)))
    original_serializer = measured.transition_to_record
    calls = []

    def spy(transition: Transition) -> dict[str, Any]:
        if transition is contaminated:
            calls.append("contaminated-right-serialization")
            raise AssertionError("unvalidated metadata reached serializer")
        return original_serializer(transition)

    monkeypatch.setattr(measured, "transition_to_record", spy)
    with pytest.raises(FleetError, match="measured_carrier_metadata"):
        verify_measured(encoded.envelope, bad_carrier, fixture.authority, _NOW_MS)
    assert calls == []


def test_public_verify_checks_manifest_count_before_finite_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    encoded = fixture.encode()
    decoded = decode_shard(encoded.carrier.manifest, encoded.carrier.payload)
    wrong_count = replace(
        decoded, unroll=replace(decoded.unroll, transitions=decoded.unroll.transitions * 2)
    )
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("finite-scan")
        raise AssertionError("mismatched manifest count reached tensor scans")

    monkeypatch.setattr(measured.np, "isfinite", forbidden)
    with pytest.raises(FleetError, match="measured_proof_count"):
        verify_measured(encoded.envelope, wrong_count, fixture.authority, _NOW_MS)
    assert calls == []


def test_native_source_layout_is_bounded_before_source_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    large_source = replace(
        fixture.source,
        compatibility=replace(
            fixture.source.compatibility,
            observation=tuple(VectorSpec(f"channel{index}", "<f8", 1) for index in range(257)),
        ),
    )
    original = SourceSpec.to_record
    calls = []

    def spy(source: SourceSpec) -> dict[str, Any]:
        if source is large_source:
            calls.append("oversized-source-projection")
            raise AssertionError("unvalidated source layout reached projection")
        return original(source)

    monkeypatch.setattr(SourceSpec, "to_record", spy)
    with pytest.raises(FleetError, match="measured_layout"):
        replace(fixture.grant, source=large_source)
    assert calls == []
