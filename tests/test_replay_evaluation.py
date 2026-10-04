"""Lightweight offline replay evaluation, including deliberately broken exports."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime.continuous_learning import HostAuthority, HostRoleCapability
from game_learning_runtime.contracts import ActionOutcome, ActionReceipt, Event, TimeStep
from game_learning_runtime.correlated_rewards import OBSERVATION_CONTEXT_KEY, REWARD_EVIDENCE_KEY
from game_learning_runtime.environment import ContractEnvironment
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.knowledge_evidence import (
    CapabilityGap,
    CapabilityGapKind,
    ConsumptionState,
    DecisionConsumptionReceipt,
    RuleIndexBinding,
)
from game_learning_runtime.realtime import RealtimeActionReceipt, RealtimeActionStatus
from game_learning_runtime.replay_evaluation import (
    REPLAY_CHECK_METRICS,
    ActionSegment,
    BootstrapAudit,
    CheckCoverage,
    DecisionAudit,
    FixedReplaySuite,
    KnowledgeRequirement,
    MaskIndex,
    ObservationLifecycle,
    ReplayAction,
    ReplayDecision,
    ReplayEnvironment,
    ReplayEpisode,
    ReplayFrame,
    RewardContribution,
    evaluate_replay,
    evaluator_sha256,
)
from game_learning_runtime.specs import CompositeSpec, EnvironmentSpec, SpaceKind, TensorSpec
from game_learning_runtime.training import TrainingConfig
from game_learning_runtime.training_safety import RewardSafetyConfig


@dataclass(frozen=True)
class _TypedEnvelope:
    value: Any


@dataclass(frozen=True)
class _DetailedRealtimeReceipt(RealtimeActionReceipt):
    # Synthetic extension container: main's timing receipt itself has no details field.
    details: Any = field(default_factory=dict)


class _DerivedAuthority(HostAuthority):
    pass


class _DerivedCapability(HostRoleCapability):
    pass


def _flattened_host_fixture(value: Any) -> dict[str, Any]:
    return {
        key: item.hex() if isinstance(item, bytes) else item for key, item in asdict(value).items()
    }


@dataclass(frozen=True)
class _DeclaredHostBinding(RuleIndexBinding):
    private_host: Any = None

    def to_mapping(self) -> dict[str, Any]:
        object.__setattr__(self, "_mapper_calls", getattr(self, "_mapper_calls", 0) + 1)
        return {
            **RuleIndexBinding.to_mapping(self),
            "private_host": _flattened_host_fixture(self.private_host),
        }


class _AttributeHostBinding(RuleIndexBinding):
    def __init__(self, private_host: Any, **base_fields: Any) -> None:
        RuleIndexBinding.__init__(self, **base_fields)
        object.__setattr__(self, "private_host", private_host)

    def to_mapping(self) -> dict[str, Any]:
        object.__setattr__(self, "_mapper_calls", getattr(self, "_mapper_calls", 0) + 1)
        return {
            **RuleIndexBinding.to_mapping(self),
            "private_host": _flattened_host_fixture(self.private_host),
        }


def _extra_export(value: Any, base: dict[str, Any]) -> dict[str, Any]:
    object.__setattr__(value, "_mapper_calls", getattr(value, "_mapper_calls", 0) + 1)
    return {**base, "private_host": _flattened_host_fixture(value.private_host)}


class _ExtraMappedAction(ReplayAction):
    def to_mapping(self) -> dict[str, Any]:
        return _extra_export(self, ReplayAction.to_mapping(self))


class _ExtraMappedFrame(ReplayFrame):
    def to_mapping(self) -> dict[str, Any]:
        return _extra_export(self, ReplayFrame.to_mapping(self))


class _ExtraMappedEpisode(ReplayEpisode):
    def to_mapping(self) -> dict[str, Any]:
        return _extra_export(self, ReplayEpisode.to_mapping(self))


class _ExtraMappedTiming(RealtimeActionReceipt):
    def to_mapping(self) -> dict[str, Any]:
        return _extra_export(self, dict(RealtimeActionReceipt.to_mapping(self)))


class _UncallableSnapshotFrame(ReplayFrame):
    def snapshot(self) -> TimeStep:
        raise AssertionError("subclass snapshot must not define frozen source evidence")


class _UncallableIntegritySuite(FixedReplaySuite):
    def verify_integrity(self) -> None:
        raise AssertionError("subclass verifier must not define fixed integrity")


class _SpoofDigestSuite(FixedReplaySuite):
    @property
    def sha256(self) -> str:
        return "a" * 64


class _SpoofConsumptionReceipt(DecisionConsumptionReceipt):
    def assert_for_decision(self, **_expected: Any) -> None:
        pass

    @property
    def is_used(self) -> bool:
        return True


class _SpoofCapabilityGap(CapabilityGap):
    def verify_missing(self, _available: Any = None) -> None:
        pass


def _privileged(kind: str) -> HostAuthority | HostRoleCapability:
    if kind.startswith("authority"):
        authority_type = _DerivedAuthority if kind.endswith("subclass") else HostAuthority
        return authority_type(
            "host.synthetic", ("evaluator",), ("supervisor",), ("reviewer",), bytes(range(32))
        )
    capability_type = _DerivedCapability if kind.endswith("subclass") else HostRoleCapability
    return capability_type(
        "epoch",
        "1" * 64,
        "campaign.synthetic",
        "trial",
        "token",
        "evaluator",
        "evaluator",
        "2" * 64,
        bytes(range(32)),
    )


def _wrapped_privileged(kind: str, container: str) -> Any:
    value = _privileged(kind)
    if container == "mapping_list":
        return {"outer": [({"inner": value},)]}
    if container == "dataclass":
        return _TypedEnvelope(value)
    if container == "array":
        return np.array([value], dtype=object)
    if container == "enum":
        return Enum("SyntheticEnvelope", {"MEMBER": value}).MEMBER
    if container == "mapping_key":
        return {value: "ordinary"}
    if container == "set":
        return {value}
    return frozenset({value})


def _action(value: int = 0) -> ReplayAction:
    return ReplayAction(
        f"select-{value}",
        {"choice": np.array([value], dtype=np.int64)},
        (MaskIndex("choice", (value,)),),
    )


def _spec() -> EnvironmentSpec:
    return EnvironmentSpec(
        "example.replay",
        CompositeSpec({"state": TensorSpec((1,), np.float32)}),
        CompositeSpec({"choice": TensorSpec((1,), np.int64, SpaceKind.DISCRETE, 0, 1)}),
        action_mask=CompositeSpec({"choice": TensorSpec((2,), np.bool_, SpaceKind.BINARY)}),
        metadata={"environment_config_sha256": "9" * 64, "target_id": "review-target"},
    )


def _binding() -> RuleIndexBinding:
    return RuleIndexBinding("example.replay", "1.0", "rules-v1", "1" * 64, "2" * 64, "1" * 64)


def _reward_contract() -> tuple[TrainingConfig, RewardSafetyConfig]:
    training = TrainingConfig.from_mapping(
        {
            "schema_version": "glr.training.v1",
            "lifecycle": {"start_mode": "reset", "stop_on_done": True},
            "bridge": {"required_capabilities": []},
            "knowledge_sources": [
                {"id": "runtime", "authority": "authoritative", "required": True}
            ],
            "reward": {
                "minimum": -20,
                "maximum": 20,
                "terms": [
                    {
                        "name": "progress",
                        "source": "runtime",
                        "weight": 1,
                        "minimum": -10,
                        "maximum": 10,
                        "required": True,
                    },
                    {
                        "name": "time-cost",
                        "source": "runtime",
                        "weight": 1,
                        "minimum": -10,
                        "maximum": 10,
                        "required": False,
                    },
                    {
                        "name": "outcome",
                        "source": "runtime",
                        "weight": 1,
                        "minimum": -10,
                        "maximum": 10,
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
            "shaping_signals": ["progress", "time-cost"],
            "max_positive_shaping_per_step": 10,
            "max_positive_shaping_per_episode": 100,
            "max_negative_shaping_per_step": 10,
            "max_negative_shaping_per_episode": 100,
            "failure_episode_maximum": 0,
            "require_terminal_outcome": False,
        }
    )
    return training, safety


def _context(index: int) -> dict[str, Any]:
    return {
        "run_id": "synthetic.source-run",
        "environment_id": "example.replay",
        "protocol_version": "1.0",
        "target_id": "review-target",
        "environment_config_sha256": "9" * 64,
        "episode_id": str(UUID(int=1)),
        "step_id": index,
        "producer_sequence": 10 + 2 * index,
        "timestamp_ns": 100 * index + 10,
        "phase": "gameplay",
        "alive": True,
    }


def _reward_evidence(index: int) -> dict[str, Any]:
    return {
        "signals": [{"name": "progress", "source": "runtime", "value": 1.0}],
        "attributions": [
            {
                "signal_name": "progress",
                "source": "runtime",
                "action_id": f"action-{index}",
                "before_sequence": 8 + 2 * index,
                "after_sequence": 10 + 2 * index,
                "effect": "confirmed",
            }
        ],
    }


def _frames() -> tuple[ReplayFrame, ...]:
    frames = []
    for index in range(3):
        receipt = (
            None
            if index == 0
            else ActionReceipt(
                f"action-{index}",
                UUID(int=1),
                index,
                ActionOutcome.ACCEPTED,
                100 * index,
                100 * index + 10,
                authoritative_observation_sequence=10 + 2 * index,
                issued_against_observation_sequence=8 + 2 * index,
                target_id="review-target",
            )
        )
        step = TimeStep(
            {"state": np.array([index], dtype=np.float32)},
            np.array([int(index > 0)], dtype=np.float32),
            np.array([index == 2]),
            np.array([False]),
            UUID(int=1),
            index,
            {"choice": np.array([True, index == 0])},
            receipt,
            info={
                "nested": {"source": [index]},
                OBSERVATION_CONTEXT_KEY: _context(index),
                **({REWARD_EVIDENCE_KEY: _reward_evidence(index)} if index else {}),
            },
            timestamp_ns=100 * index + 10,
        )
        rewards = (
            None if index == 0 else (RewardContribution("progress", 1.0, f"action-{index}", True),)
        )
        interval = (
            None
            if index == 0
            else (
                ActionSegment("first", 8 + 2 * index, 9 + 2 * index),
                ActionSegment("second", 9 + 2 * index, 10 + 2 * index),
            )
        )
        legal = () if index == 2 else ((_action(), _action(1)) if index == 0 else (_action(),))
        frames.append(
            ReplayFrame(
                step,
                10 + 2 * index,
                ObservationLifecycle.ACTIVE,
                legal,
                f"decision-{index}",
                rewards,
                interval,
            )
        )
    return tuple(frames)


def _changed(frame: ReplayFrame, **changes: Any) -> ReplayFrame:
    fields = {
        "timestep": frame.snapshot(),
        "producer_sequence": frame.producer_sequence,
        "lifecycle": frame.lifecycle,
        "legal_actions": frame.legal_actions,
        "decision_id": frame.decision_id,
        "reward_contributions": frame.reward_contributions,
        "action_interval": frame.action_interval,
        "required_findings": frame.required_findings,
    }
    fields.update(changes)
    return ReplayFrame(**fields)


def _suite(frames: tuple[ReplayFrame, ...] | None = None, **changes: Any) -> FixedReplaySuite:
    fields = {
        "suite_id": "fixed-conformance",
        "spec": _spec(),
        "binding": _binding(),
        "reward_training": _reward_contract()[0],
        "reward_safety": _reward_contract()[1],
        "episodes": (
            ReplayEpisode(
                "frozen-export",
                "3" * 64,
                frames or _frames(),
                (_action(), _action()),
                run_id="synthetic.source-run",
            ),
        ),
    }
    fields.update(changes)
    return FixedReplaySuite(**fields)


def _decision(suite: FixedReplaySuite, index: int, **changes: Any) -> ReplayDecision:
    post = suite.episodes[0].frames[index + 1]
    bootstrap = (
        BootstrapAudit({}, None)
        if post.snapshot().done
        else BootstrapAudit({"select-0": 1.0, "select-1": 100.0}, "select-0")
    )
    fields = {
        "action": _action(),
        "audit": DecisionAudit(
            True,
            bootstrap,
            post.reward_contributions,
            post.action_interval,
        ),
    }
    fields.update(changes)
    return ReplayDecision(**fields)


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "inert-reference.json"
    path.write_bytes(b'{"kind":"inert-reference"}')
    return path


def _evaluate(suite: FixedReplaySuite, artifact: Path, policy: Any = None):
    return evaluate_replay(
        suite,
        artifact,
        policy or (lambda current, _legal, _rules: _decision(suite, current.step_id)),
        expected_suite_sha256=suite.sha256,
        expected_candidate_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
    )


def _frame_as(frame_type: type[ReplayFrame], frame: ReplayFrame, **changes: Any) -> ReplayFrame:
    values = {
        "timestep": ReplayFrame.snapshot(frame),
        "producer_sequence": frame.producer_sequence,
        "lifecycle": frame.lifecycle,
        "legal_actions": frame.legal_actions,
        "decision_id": frame.decision_id,
        "reward_contributions": frame.reward_contributions,
        "action_interval": frame.action_interval,
        "required_findings": frame.required_findings,
    }
    values.update(changes)
    return frame_type(**values)


@pytest.mark.parametrize("kind", ["authority", "capability"])
@pytest.mark.parametrize("component", ["action", "frame", "episode", "timing"])
def test_fixed_contract_exports_ignore_subclass_virtual_mappers(kind: str, component: str) -> None:
    baseline = _suite()
    episode = baseline.episodes[0]
    frames = list(episode.frames)
    actions = episode.recorded_actions
    leaf_path: tuple[str | int, ...]
    if component == "action":
        original = actions[0]
        extended: Any = _ExtraMappedAction(
            original.semantic, original.action, original.mask_indices
        )
        object.__setattr__(extended, "private_host", _privileged(kind))
        actions = (extended, actions[1])
        frames[0] = _changed(frames[0], legal_actions=(extended, _action(1)))
        leaf_path = ("episodes", 0, "recorded_actions", 0)
    elif component == "frame":
        extended = _frame_as(_ExtraMappedFrame, frames[0])
        object.__setattr__(extended, "private_host", _privileged(kind))
        frames[0] = extended
        leaf_path = ("episodes", 0, "frames", 0)
    elif component == "episode":
        extended = _ExtraMappedEpisode(
            episode.source_id,
            episode.source_sha256,
            episode.frames,
            episode.recorded_actions,
            run_id=episode.run_id,
        )
        object.__setattr__(extended, "private_host", _privileged(kind))
        leaf_path = ("episodes", 0)
    else:
        timing_values = {
            "action_id": "action-1",
            "status": RealtimeActionStatus.CONSUMED,
            "deadline_ns": 10,
            "quantum_ns": 1,
            "issued_at_ns": 100,
            "consumed_at_ns": 101,
            "settled_at_ns": 102,
        }
        extended = _ExtraMappedTiming(**timing_values)
        object.__setattr__(extended, "private_host", _privileged(kind))
        step = frames[1].snapshot()
        frames[1] = _changed(
            frames[1],
            timestep=replace(step, action_receipt=replace(step.action_receipt, realtime=extended)),
        )
        base_frames = list(episode.frames)
        base_frames[1] = _changed(
            base_frames[1],
            timestep=replace(
                step,
                action_receipt=replace(
                    step.action_receipt, realtime=RealtimeActionReceipt(**timing_values)
                ),
            ),
        )
        baseline = _suite(tuple(base_frames))
        leaf_path = ("episodes", 0, "frames", 1, "timestep", "action_receipt", "realtime")
    rebuilt = (
        extended
        if component == "episode"
        else ReplayEpisode(
            episode.source_id, episode.source_sha256, tuple(frames), actions, run_id=episode.run_id
        )
    )
    suite = _suite(episodes=(rebuilt,))
    actual: Any = suite.to_mapping()
    expected: Any = baseline.to_mapping()
    for key in leaf_path:
        actual, expected = actual[key], expected[key]
    assert frozenset(actual) == frozenset(expected)
    assert suite.sha256 == baseline.sha256
    suite.verify_integrity()
    assert getattr(extended, "_mapper_calls", 0) == 0


def test_fixed_contract_exports_use_base_snapshot_reader(artifact: Path) -> None:
    frames = list(_frames())
    frames[0] = _frame_as(_UncallableSnapshotFrame, frames[0])
    suite = _suite(tuple(frames))
    assert _evaluate(suite, artifact).passed
    assert ReplayEnvironment(suite).reset().step_id == 0


def test_fixed_contract_exports_use_base_integrity_verifier(artifact: Path) -> None:
    original = _suite()
    suite = _UncallableIntegritySuite(
        original.suite_id,
        original.spec,
        original.binding,
        original.episodes,
        reward_training=original.reward_training,
        reward_safety=original.reward_safety,
    )
    assert _evaluate(suite, artifact).passed
    assert ReplayEnvironment(suite).reset().step_id == 0


def test_fixed_contract_exports_ignore_spoofed_digest_in_evaluation_and_provenance(
    artifact: Path,
) -> None:
    original = _suite()
    suite = _SpoofDigestSuite(
        original.suite_id,
        original.spec,
        original.binding,
        original.episodes,
        reward_training=original.reward_training,
        reward_safety=original.reward_safety,
    )
    candidate_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()

    def policy(current: TimeStep, _legal: Any, _rules: Any) -> ReplayDecision:
        return _decision(original, current.step_id)

    with pytest.raises(ContractViolation):
        evaluate_replay(
            suite,
            artifact,
            policy,
            expected_suite_sha256="a" * 64,
            expected_candidate_sha256=candidate_sha,
        )
    result = evaluate_replay(
        suite,
        artifact,
        policy,
        expected_suite_sha256=original.sha256,
        expected_candidate_sha256=candidate_sha,
    )
    assert result.passed
    assert result.suite_sha256 == original.sha256
    assert ReplayEnvironment(suite).reset().info["replay_source"]["suite_sha256"] == original.sha256


@pytest.mark.parametrize("spoof", ["assertion", "used-property", "gap-checker"])
def test_fixed_contract_exports_use_base_knowledge_audit_readers(
    artifact: Path, spoof: str
) -> None:
    frames = list(_frames())
    requirement = KnowledgeRequirement("finding-1", "4" * 64, "scorer", "5" * 64)
    frames[0] = _changed(frames[0], required_findings=(requirement,))
    suite = _suite(tuple(frames))

    def policy(current: TimeStep, _legal: Any, binding: RuleIndexBinding) -> ReplayDecision:
        decision = _decision(suite, current.step_id)
        if current.step_id:
            return decision
        receipt: DecisionConsumptionReceipt = _SpoofConsumptionReceipt(
            "foreign-decision" if spoof == "assertion" else "decision-0",
            "finding-1",
            "4" * 64,
            "scorer",
            binding,
            ConsumptionState.USED,
            "5" * 64,
        )
        if spoof == "used-property":
            # Simulate a modified callback value; fingerprint/primitive projection must
            # retain both raw fields even when its convenience property claims use.
            object.__setattr__(receipt, "state", ConsumptionState.RETRIEVED)
        gaps = ()
        if spoof == "gap-checker":
            gaps = (
                _SpoofCapabilityGap(
                    "decision-0",
                    CapabilityGapKind.MISSING_ACTION,
                    "select-1",
                    ("select-0", "select-1"),
                    "6" * 64,
                ),
            )
        return replace(decision, consumptions=(receipt,), capability_gaps=gaps)

    result = _evaluate(suite, artifact, policy)
    assert not result.passed
    metric = (
        "evaluation.capability_gap_requests"
        if spoof == "gap-checker"
        else "evaluation.rule_consumption_errors"
    )
    assert result.extra_checks[metric] == 1


def test_complete_offline_replay_derives_all_checks_and_objective(artifact: Path) -> None:
    result = _evaluate(_suite(), artifact)
    assert result.passed, result.issues
    assert result.completed_steps == result.expected_steps == 2
    assert set(result.check_counts) == set(REPLAY_CHECK_METRICS)
    assert all(value == 0 for value in result.check_counts.values())
    assert set(result.coverage.values()) == {CheckCoverage.AUDITED}
    assert result.objective["replay.contract_passed"] == 1
    assert len(result.evaluator_sha256) == 64
    assert result.evaluator_sha256 != evaluator_sha256()
    assert result.source_provenance == (("frozen-export", "3" * 64),)
    assert result.to_mapping()["schema"] == "glr.replay-evaluation.v1"
    with pytest.raises(TypeError):
        result.check_counts[REPLAY_CHECK_METRICS[0]] = 4
    with pytest.raises(FrozenInstanceError):
        result.passed = False


def test_reset_allocates_fresh_logical_identity_and_preserves_producer_provenance() -> None:
    suite = _suite()
    replay = ReplayEnvironment(suite)
    environment = ContractEnvironment(replay)
    first = environment.reset()
    post = environment.step(_action().action)
    assert first.episode_id == post.episode_id != UUID(int=1)
    assert post.action_receipt.episode_id == post.episode_id
    assert post.action_receipt.step_id == 1
    assert post.info["replay_source"]["source_step_id"] == 1
    assert post.info["replay_source"]["producer_sequence"] == 12
    assert post.info["replay_source"]["logical_step_id"] == 1
    assert environment.reset().episode_id != first.episode_id
    assert suite.sha256 == _suite().sha256


def test_exposed_snapshot_metadata_and_array_mutation_cannot_change_frozen_suite() -> None:
    suite = _suite()
    digest = suite.sha256
    snapshot = suite.episodes[0].frames[0].snapshot()
    snapshot.info["nested"]["source"].append(999)
    snapshot.observation["state"].setflags(write=True)
    snapshot.observation["state"][0] = 999
    fresh = suite.episodes[0].frames[0].snapshot()
    assert fresh.info["nested"]["source"] == [0]
    assert fresh.observation["state"][0] == 0
    assert suite.sha256 == digest
    with pytest.raises(ValueError):
        _action().action["choice"].setflags(write=True)


def test_replay_enforces_lifecycle_and_never_advances_on_uncaptured_action() -> None:
    replay = ReplayEnvironment(_suite())
    with pytest.raises(ContractViolation, match="reset first"):
        replay.step(_action().action)
    replay.reset()
    with pytest.raises(ContractViolation, match="captured replay branch"):
        replay.step(_action(1).action)
    assert replay.current_frame.snapshot().step_id == 0
    replay.step(_action().action)
    replay.step(_action().action)
    with pytest.raises(ContractViolation, match="terminal"):
        replay.step(_action().action)
    with pytest.raises(ContractViolation, match="reset overrides"):
        replay.reset(seed=42)
    with pytest.raises(ContractViolation):
        replay.attach()
    replay.close()
    with pytest.raises(ContractViolation, match="closed"):
        replay.reset()


def test_parameter_mutation_is_measured_around_candidate_callback(artifact: Path) -> None:
    suite = _suite()

    def mutating(current, _legal, _binding):
        artifact.write_bytes(b"changed")
        return _decision(suite, current.step_id)

    result = _evaluate(suite, artifact, mutating)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[0]] == 1


def test_callback_failure_still_checks_artifact_mutation(artifact: Path) -> None:
    suite = _suite()

    def failing(_current, _legal, _binding):
        artifact.write_bytes(b"changed")
        raise RuntimeError("deliberate offline callback failure")

    result = _evaluate(suite, artifact, failing)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[0]] == 1
    assert result.check_counts[REPLAY_CHECK_METRICS[2]] is None


@pytest.mark.parametrize("field,value", [("step_id", 2), ("episode_id", UUID(int=8))])
def test_source_identity_mismatch_is_derived_not_hidden_by_logical_rebinding(
    artifact: Path,
    field: str,
    value: Any,
) -> None:
    frames = list(_frames())
    frames[1] = _changed(frames[1], timestep=replace(frames[1].snapshot(), **{field: value}))
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[1]] == 1


@pytest.mark.parametrize("sequence", [10, 9, 13])
def test_post_action_sequence_must_be_strictly_new_and_match_producer(
    artifact: Path, sequence: int
) -> None:
    frames = list(_frames())
    post = frames[1].snapshot()
    frames[1] = _changed(
        frames[1],
        timestep=replace(
            post,
            action_receipt=replace(
                post.action_receipt, authoritative_observation_sequence=sequence
            ),
        ),
    )
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[2]] == 1


@pytest.mark.parametrize("lifecycle", [ObservationLifecycle.DEAD, ObservationLifecycle.LOADING])
def test_dead_or_loading_observations_cannot_feed_a_learner_update(
    artifact: Path, lifecycle
) -> None:
    frames = list(_frames())
    frames[1] = _changed(frames[1], lifecycle=lifecycle)
    suite = _suite(tuple(frames))
    bad = _evaluate(suite, artifact)
    assert not bad.passed
    assert bad.check_counts[REPLAY_CHECK_METRICS[3]] == 2
    skipped = _evaluate(
        suite,
        artifact,
        lambda current, _legal, _binding: replace(
            _decision(suite, current.step_id),
            audit=replace(_decision(suite, current.step_id).audit, learner_updated=False),
        ),
    )
    assert not skipped.passed
    assert skipped.check_counts[REPLAY_CHECK_METRICS[3]] == 0
    assert skipped.check_counts[REPLAY_CHECK_METRICS[5]] > 0


def test_bootstrap_ignores_high_value_of_masked_action_and_refuses_selecting_it(
    artifact: Path,
) -> None:
    suite = _suite()
    assert _evaluate(suite, artifact).passed

    def illegal_bootstrap(current, _legal, _binding):
        decision = _decision(suite, current.step_id)
        if current.step_id == 0:
            return replace(
                decision,
                audit=replace(
                    decision.audit,
                    bootstrap=BootstrapAudit(
                        {"select-0": 1.0, "select-1": 100.0},
                        "select-1",
                    ),
                ),
            )
        return decision

    result = _evaluate(suite, artifact, illegal_bootstrap)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[4]] == 1


def test_terminal_observation_never_bootstraps(artifact: Path) -> None:
    suite = _suite()
    result = _evaluate(
        suite,
        artifact,
        lambda current, _legal, _binding: replace(
            _decision(suite, current.step_id),
            audit=replace(
                _decision(suite, current.step_id).audit,
                bootstrap=BootstrapAudit({"select-0": 1.0}, "select-0"),
            ),
        ),
    )
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[4]] == 1


@pytest.mark.parametrize("change", ["wrong_action", "wrong_amount", "rejected"])
def test_reward_attribution_is_checked_against_fixed_terms_and_accepted_receipt(
    artifact: Path, change: str
) -> None:
    frames = list(_frames())
    if change == "rejected":
        step = frames[1].snapshot()
        frames[1] = _changed(
            frames[1],
            timestep=replace(
                step,
                action_receipt=replace(step.action_receipt, outcome=ActionOutcome.REJECTED),
            ),
        )
    suite = _suite(tuple(frames))

    def policy(current, _legal, _binding):
        decision = _decision(suite, current.step_id)
        if current.step_id == 0 and change != "rejected":
            term = decision.audit.reward_contributions[0]
            term = replace(
                term,
                **(
                    {"action_id": "another-action"} if change == "wrong_action" else {"amount": 2.0}
                ),
            )
            return replace(decision, audit=replace(decision.audit, reward_contributions=(term,)))
        return decision

    result = _evaluate(suite, artifact, policy)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] == 1


def test_multisegment_action_requires_full_contiguous_interval_consumption(artifact: Path) -> None:
    suite = _suite()
    result = _evaluate(
        suite,
        artifact,
        lambda current, _legal, _binding: replace(
            _decision(suite, current.step_id),
            audit=replace(
                _decision(suite, current.step_id).audit,
                consumed_interval=_decision(suite, current.step_id).audit.consumed_interval[-1:],
            ),
        ),
    )
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[6]] == 2


@pytest.mark.parametrize(
    "missing",
    ["sequence", "issued_sequence", "lifecycle", "bootstrap", "reward", "interval", "update"],
)
def test_missing_required_audit_information_fails_closed_with_unknown_not_zero(
    artifact: Path, missing: str
) -> None:
    frames = list(_frames())
    if missing == "sequence":
        frames[1] = _changed(frames[1], producer_sequence=None)
    if missing == "issued_sequence":
        post = frames[1].snapshot()
        frames[1] = _changed(
            frames[1],
            timestep=replace(
                post,
                action_receipt=replace(
                    post.action_receipt, issued_against_observation_sequence=None
                ),
            ),
        )
    if missing == "lifecycle":
        frames[1] = _changed(frames[1], lifecycle=ObservationLifecycle.UNKNOWN)
    if missing == "reward":
        frames[1] = _changed(frames[1], reward_contributions=None)
    if missing == "interval":
        frames[1] = _changed(frames[1], action_interval=None)
    suite = _suite(tuple(frames))

    def policy(current, _legal, _binding):
        decision = _decision(suite, current.step_id)
        field = {"bootstrap": "bootstrap", "update": "learner_updated"}.get(missing)
        return (
            decision
            if field is None
            else replace(decision, audit=replace(decision.audit, **{field: None}))
        )

    result = _evaluate(suite, artifact, policy)
    assert not result.passed
    assert CheckCoverage.UNKNOWN in result.coverage.values()
    assert None in result.check_counts.values()
    if missing == "issued_sequence":
        assert result.coverage[REPLAY_CHECK_METRICS[2]] is CheckCoverage.UNKNOWN
        assert result.check_counts[REPLAY_CHECK_METRICS[2]] is None


def test_not_applicable_requires_external_fixed_reason_and_changes_suite_digest(
    artifact: Path,
) -> None:
    frames = list(_frames())
    frames[1] = _changed(frames[1], reward_contributions=None)
    bare = _suite(tuple(frames))
    declared = _suite(
        tuple(frames),
        not_applicable={REPLAY_CHECK_METRICS[5]: "inert interface conformance has no reward model"},
    )
    assert bare.sha256 != declared.sha256
    assert not _evaluate(bare, artifact).passed
    result = _evaluate(declared, artifact)
    assert result.passed, result.issues
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] is None
    assert result.coverage[REPLAY_CHECK_METRICS[5]] is CheckCoverage.NOT_APPLICABLE
    assert result.not_applicable == declared.not_applicable
    with pytest.raises(ValueError):
        _suite(not_applicable={REPLAY_CHECK_METRICS[0]: "skip artifact freeze"})
    with pytest.raises(ValueError):
        _suite(not_applicable={REPLAY_CHECK_METRICS[5]: ""})


@pytest.mark.parametrize("receipt_state", [ConsumptionState.RETRIEVED, ConsumptionState.USED])
def test_rule_consumption_uses_expected_finding_consumer_rules_and_evidence(
    artifact: Path, receipt_state
) -> None:
    frames = list(_frames())
    requirement = KnowledgeRequirement("finding-1", "4" * 64, "scorer", "5" * 64)
    frames[0] = _changed(frames[0], required_findings=(requirement,))
    suite = _suite(tuple(frames))

    def policy(current, _legal, binding):
        decision = _decision(suite, current.step_id)
        if current.step_id != 0:
            return decision
        receipt = DecisionConsumptionReceipt(
            "decision-0",
            "finding-1",
            "4" * 64,
            "scorer",
            binding,
            receipt_state,
            None if receipt_state is ConsumptionState.RETRIEVED else "5" * 64,
        )
        return replace(decision, consumptions=(receipt,))

    result = _evaluate(suite, artifact, policy)
    assert result.passed is (receipt_state is ConsumptionState.USED)
    assert result.extra_checks["evaluation.rule_consumption_errors"] == int(
        receipt_state is ConsumptionState.RETRIEVED
    )


def test_stale_rules_and_missing_required_consumer_never_produce_a_pass(artifact: Path) -> None:
    frames = list(_frames())
    frames[0] = _changed(
        frames[0],
        required_findings=(
            KnowledgeRequirement(
                "finding-1",
                "4" * 64,
                "scorer",
                "5" * 64,
            ),
        ),
    )
    suite = _suite(tuple(frames))
    assert not _evaluate(suite, artifact).passed

    def policy(current, _legal, binding):
        decision = _decision(suite, current.step_id)
        if current.step_id != 0:
            return decision
        old = replace(binding, rules_version="old-rules")
        return replace(
            decision,
            consumptions=(
                DecisionConsumptionReceipt(
                    "decision-0",
                    "finding-1",
                    "4" * 64,
                    "scorer",
                    old,
                    ConsumptionState.USED,
                    "5" * 64,
                ),
            ),
        )

    result = _evaluate(suite, artifact, policy)
    assert not result.passed
    assert result.extra_checks["evaluation.rule_consumption_errors"] == 1


def test_passive_missing_action_gap_cannot_expand_legal_surface(artifact: Path) -> None:
    suite = _suite()

    def policy(current, legal, _binding):
        missing = ReplayAction("revise-options", _action().action, _action().mask_indices)
        gap = CapabilityGap(
            "decision-0",
            CapabilityGapKind.MISSING_ACTION,
            "revise-options",
            tuple(action.semantic for action in legal),
            "6" * 64,
        )
        return replace(_decision(suite, current.step_id), action=missing, capability_gaps=(gap,))

    result = _evaluate(suite, artifact, policy)
    assert not result.passed
    assert result.extra_checks["evaluation.illegal_action_requests"] == 1
    assert result.completed_steps == 0


def test_incomplete_semantic_mask_mapping_is_not_treated_as_legal(artifact: Path) -> None:
    frames = list(_frames())
    unbound = ReplayAction("select-0", _action().action)
    frames[0] = _changed(frames[0], legal_actions=(unbound,))
    episode = ReplayEpisode("frozen-export", "3" * 64, tuple(frames), (unbound, _action()))
    suite = _suite(episodes=(episode,))
    result = _evaluate(
        suite,
        artifact,
        lambda current, legal, _binding: replace(
            _decision(suite, current.step_id),
            action=legal[0],
        ),
    )
    assert not result.passed
    assert any("mask coordinates" in issue for issue in result.issues)


def test_external_suite_and_candidate_hashes_cannot_be_changed_by_caller(artifact: Path) -> None:
    suite = _suite()
    for bad_suite, bad_artifact in [
        ("8" * 64, hashlib.sha256(artifact.read_bytes()).hexdigest()),
        (suite.sha256, "8" * 64),
    ]:
        with pytest.raises(ContractViolation):
            evaluate_replay(
                suite,
                artifact,
                lambda *_: None,
                expected_suite_sha256=bad_suite,
                expected_candidate_sha256=bad_artifact,
            )


@pytest.mark.parametrize("bad_array", [np.array(1), np.array([object()]), np.array([np.nan])])
def test_replay_rejects_scalar_object_and_nonfinite_tensor_records(bad_array) -> None:
    with pytest.raises(ValueError):
        ReplayAction("invalid", {"choice": bad_array})


def test_suite_digest_covers_source_identity_metadata_and_every_audit_record() -> None:
    original = _suite()
    frames = list(_frames())
    source = frames[1].snapshot()
    frames[1] = _changed(frames[1], timestep=replace(source, info={"different": "source"}))
    assert _suite(tuple(frames)).sha256 != original.sha256
    frames = list(_frames())
    frames[1] = _changed(frames[1], producer_sequence=999)
    assert _suite(tuple(frames)).sha256 != original.sha256
    frames = list(_frames())
    frames[1] = _changed(frames[1], action_interval=(ActionSegment("whole", 10, 12),))
    assert _suite(tuple(frames)).sha256 != original.sha256


def test_callback_implementation_and_frozen_captures_are_part_of_evaluator_identity() -> None:
    suite = _suite()

    def first(current, _legal, _binding):
        return _decision(suite, current.step_id)

    def second(current, _legal, _binding):
        return replace(_decision(suite, current.step_id), consumptions=())

    assert evaluator_sha256(first) != evaluator_sha256(second)
    assert evaluator_sha256(first) == evaluator_sha256(first)
    with pytest.raises(ContractViolation, match="static Python function"):
        evaluator_sha256(object())


def test_callback_capture_drift_is_rejected(artifact: Path) -> None:
    suite = _suite()
    captured_state = [0]

    def drifting(current, _legal, _binding):
        captured_state.append(1)
        return _decision(suite, current.step_id)

    original = evaluator_sha256(drifting)
    result = _evaluate(suite, artifact, drifting)
    assert result.evaluator_sha256 == original
    assert not result.passed
    assert result.extra_checks["evaluation.evaluator_mutations"] == 1


def test_callback_cannot_change_frozen_source_records_even_with_object_setattr(
    artifact: Path,
) -> None:
    suite = _suite()

    def tampering(current, _legal, _binding):
        object.__setattr__(suite.episodes[0].frames[1], "producer_sequence", 999)
        return _decision(suite, current.step_id)

    result = _evaluate(suite, artifact, tampering)
    assert not result.passed
    assert any("suite freeze" in message for message in result.issues)


@pytest.mark.parametrize("target", [None, "other-target"])
def test_foreign_or_unbound_action_target_is_refused_at_suite_admission(target: Any) -> None:
    frames = list(_frames())
    source = frames[1].snapshot()
    frames[1] = _changed(
        frames[1],
        timestep=replace(
            source,
            action_receipt=replace(source.action_receipt, target_id=target),
        ),
    )
    with pytest.raises(ContractViolation, match="target"):
        _suite(tuple(frames))


@pytest.mark.parametrize("target", [None, "", "Has spaces", 42])
def test_suite_requires_a_nonempty_portable_target_identifier(target: Any) -> None:
    metadata = dict(_spec().metadata, target_id=target)
    with pytest.raises(ValueError, match="identifier"):
        _suite(spec=replace(_spec(), metadata=metadata))


def test_double_none_target_cannot_be_certified_as_bound() -> None:
    frames = []
    for frame in _frames():
        source = frame.snapshot()
        receipt = source.action_receipt
        frames.append(
            _changed(
                frame,
                timestep=replace(
                    source,
                    action_receipt=None if receipt is None else replace(receipt, target_id=None),
                ),
            )
        )
    metadata = {key: value for key, value in _spec().metadata.items() if key != "target_id"}
    with pytest.raises(ValueError, match="identifier"):
        _suite(tuple(frames), spec=replace(_spec(), metadata=metadata))


@pytest.mark.parametrize(
    "value",
    [
        HostAuthority("host.test", ("evaluator",), ("supervisor",), ("reviewer",), b"s" * 32),
        HostRoleCapability(
            "epoch",
            "1" * 64,
            "campaign.test",
            "trial",
            "token",
            "evaluator",
            "evaluator",
            "2" * 64,
            b"proof",
        ),
        RewardContribution("ordinary-record", 1.0),
        b"ordinary-bytes",
    ],
)
def test_nonstring_spec_metadata_cannot_enter_an_exportable_suite(value: Any) -> None:
    spec = replace(_spec(), metadata={**_spec().metadata, "accidental-object": value})
    with pytest.raises(ValueError, match="metadata"):
        _suite(spec=spec).to_mapping()


@pytest.mark.parametrize(
    "key,value", [(1, "text"), ("", "text"), ("k" * 129, "text"), ("key", "v" * 4097)]
)
def test_spec_metadata_strings_have_explicit_export_bounds(key: Any, value: Any) -> None:
    spec = replace(_spec(), metadata={**_spec().metadata, key: value})
    with pytest.raises(ValueError, match="metadata"):
        _suite(spec=spec).to_mapping()


@pytest.mark.parametrize(
    "kind", ["authority", "capability", "authority_subclass", "capability_subclass"]
)
def test_privileged_dataclasses_never_enter_exported_suite_contract_fields(kind: str) -> None:
    spec = replace(_spec(), capabilities=frozenset({_privileged(kind)}))
    with pytest.raises(ContractViolation, match="privileged host"):
        suite = _suite(spec=spec)
        suite.to_mapping()
        _ = suite.sha256


@pytest.mark.parametrize("kind", ["authority", "capability"])
@pytest.mark.parametrize("storage", ["declared", "attribute"])
def test_real_suite_uses_base_binding_projection_before_subclass_mapper_erases_host_types(
    kind: str,
    storage: str,
) -> None:
    base = _binding()
    base_fields = {
        key: value for key, value in base.to_mapping().items() if key != "schema_version"
    }
    subtype = _DeclaredHostBinding if storage == "declared" else _AttributeHostBinding
    binding = subtype(private_host=_privileged(kind), **base_fields)
    suite = _suite(binding=binding)
    if storage == "declared":
        # The existing typed-object guard still rejects a declared authority field.
        with pytest.raises(ContractViolation, match="privileged host"):
            suite.to_mapping()
    else:
        # Exercise the actual public export; compare keys before any diagnostic can expose values.
        exported = suite.to_mapping()
        observed_fields = frozenset(exported["binding"])
        assert observed_fields == frozenset(base.to_mapping())
    assert suite.sha256 == _suite().sha256
    suite.verify_integrity()
    assert getattr(binding, "_mapper_calls", 0) == 0


@pytest.mark.parametrize("kind", ["authority", "capability"])
@pytest.mark.parametrize("container", ["mapping_list", "dataclass", "array", "enum"])
@pytest.mark.parametrize("location", ["event", "info", "receipt_details"])
def test_privileged_nested_timestep_data_is_rejected_before_full_suite_export(
    kind: str,
    container: str,
    location: str,
) -> None:
    frames = list(_frames())
    source = frames[1].snapshot()
    payload = _wrapped_privileged(kind, container)
    if location == "event":
        source = replace(source, events=(Event("synthetic.event", {"nested": payload}),))
    elif location == "info":
        source = replace(source, info={"nested": payload})
    else:
        timing = _DetailedRealtimeReceipt(
            "action-1",
            RealtimeActionStatus.CONSUMED,
            10,
            1,
            100,
            101,
            102,
            details={"nested": payload},
        )
        source = replace(source, action_receipt=replace(source.action_receipt, realtime=timing))
    with pytest.raises(ContractViolation, match="privileged host"):
        frames[1] = _changed(frames[1], timestep=source)
        suite = _suite(tuple(frames))
        suite.to_mapping()
        _ = suite.sha256


@pytest.mark.parametrize(
    "kind", ["authority", "capability", "authority_subclass", "capability_subclass"]
)
@pytest.mark.parametrize(
    "container", ["mapping_list", "dataclass", "array", "enum", "mapping_key", "set", "frozenset"]
)
def test_callback_typed_containers_cannot_capture_privileged_host_material(
    kind: str,
    container: str,
) -> None:
    captured = _wrapped_privileged(kind, container)

    def policy(_current, _legal, _binding):
        return captured

    with pytest.raises(ContractViolation, match="privileged host"):
        evaluator_sha256(policy)


@pytest.mark.parametrize("kind", ["authority", "capability"])
def test_nested_privileged_action_tree_is_rejected_before_tensor_conversion(kind: str) -> None:
    with pytest.raises(ContractViolation, match="privileged host"):
        ReplayAction("synthetic-action", {"nested": {"value": [_privileged(kind)]}})


@pytest.mark.parametrize("kind", ["authority", "capability"])
def test_result_export_rejects_privileged_objects_in_typed_mapping_fields(
    artifact: Path,
    kind: str,
) -> None:
    result = _evaluate(_suite(), artifact)
    changed = replace(result, extra_checks={"accidental-field": _privileged(kind)})
    with pytest.raises(ContractViolation, match="privileged host"):
        changed.to_mapping()


def test_ordinary_event_timing_receipt_and_dataclass_capture_remain_supported(
    artifact: Path,
) -> None:
    frames = list(_frames())
    source = frames[1].snapshot()
    timing = RealtimeActionReceipt("action-1", RealtimeActionStatus.CONSUMED, 10, 1, 100, 101, 102)
    frames[1] = _changed(
        frames[1],
        timestep=replace(
            source,
            events=(Event("synthetic.event", {"nested": [1, {"value": "ok"}]}),),
            info={**source.info, "nested": [1, {"value": "ok"}]},
            action_receipt=replace(source.action_receipt, realtime=timing),
        ),
    )
    suite = _suite(tuple(frames))
    assert _evaluate(suite, artifact).passed
    assert len(suite.sha256) == 64
    ordinary = _TypedEnvelope(("ordinary", 3))

    def policy(_current, _legal, _binding):
        return ordinary

    assert len(evaluator_sha256(policy)) == 64


def test_target_identity_is_part_of_the_frozen_suite_digest() -> None:
    original = _suite()
    frames = []
    for frame in _frames():
        source = frame.snapshot()
        receipt = source.action_receipt
        frames.append(
            _changed(
                frame,
                timestep=replace(
                    source,
                    action_receipt=None
                    if receipt is None
                    else replace(receipt, target_id="new-target"),
                ),
            )
        )
    changed = _suite(
        tuple(frames),
        spec=replace(_spec(), metadata=dict(_spec().metadata, target_id="new-target")),
    )
    assert changed.sha256 != original.sha256


@pytest.mark.parametrize("invalid", ["", "Has spaces", "a" * 129])
def test_semantic_identifiers_are_bounded_and_portable(invalid: str) -> None:
    with pytest.raises(ValueError):
        ReplayAction(invalid, _action().action)


@pytest.mark.parametrize("amount", [True, "reward", float("inf")])
def test_invalid_reward_evidence_cannot_enter_a_suite(amount: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        RewardContribution("term", amount)


def test_accepted_reward_and_mask_coordinates_require_explicit_typed_identity() -> None:
    with pytest.raises(ValueError):
        RewardContribution("term", 1, requires_accepted=True)
    with pytest.raises(TypeError):
        RewardContribution("term", 1, requires_accepted=1)
    with pytest.raises(ValueError):
        MaskIndex("choice", ())
    with pytest.raises(ValueError):
        MaskIndex("choice", (True,))
    with pytest.raises(ValueError):
        ActionSegment("segment", 10, 10)
    with pytest.raises(ValueError):
        ReplayAction(
            "select-0", _action().action, (MaskIndex("choice", (0,)), MaskIndex("choice", (1,)))
        )


def test_duplicate_source_semantics_terms_and_consumers_are_refused() -> None:
    frame = _frames()[0]
    with pytest.raises(ValueError):
        _changed(frame, legal_actions=(_action(), _action()))
    reward = RewardContribution("term", 0)
    with pytest.raises(ValueError):
        _changed(frame, reward_contributions=(reward, reward))
    requirement = KnowledgeRequirement("finding", "4" * 64, "consumer", "5" * 64)
    with pytest.raises(ValueError):
        _changed(frame, required_findings=(requirement, requirement))
    with pytest.raises(TypeError):
        _changed(frame, legal_actions=[_action()])


def test_empty_or_incoherent_replay_and_unknown_source_hash_are_refused() -> None:
    with pytest.raises(ValueError):
        ReplayEpisode("source", "3" * 64, _frames(), ())
    with pytest.raises(ValueError):
        ReplayEpisode("source", "3" * 64, _frames(), (_action(),))
    with pytest.raises(ValueError):
        ReplayEpisode("source", "WRONG-HASH", _frames(), (_action(), _action()))
    with pytest.raises(ValueError):
        _suite(episodes=())
    episode = _suite().episodes[0]
    with pytest.raises(ValueError):
        _suite(episodes=(episode, episode))
    with pytest.raises(TypeError):
        _suite(spec=None)
    with pytest.raises(TypeError):
        _suite(not_applicable=[])
    with pytest.raises(TypeError):
        ReplayEnvironment(None)


def test_missing_or_undeclared_masks_cannot_be_silently_repaired() -> None:
    frames = list(_frames())
    frames[0] = _changed(frames[0], timestep=replace(frames[0].snapshot(), action_mask=None))
    with pytest.raises(ContractViolation, match="omitted"):
        _suite(tuple(frames))
    with pytest.raises(ContractViolation, match="undeclared"):
        _suite(spec=replace(_spec(), action_mask=None))


def test_scalar_source_timestep_and_oversized_tensors_are_refused() -> None:
    source = replace(_frames()[0].snapshot(), reward=np.array(0, dtype=np.float32))
    with pytest.raises(ValueError, match="non-scalar"):
        _changed(_frames()[0], timestep=source)
    with pytest.raises(ValueError, match="bounds"):
        ReplayAction("large", {"choice": np.zeros((300000,), dtype=np.float32)})
    with pytest.raises(ValueError, match="byte budget"):
        _changed(
            _frames()[0], timestep=replace(_frames()[0].snapshot(), info={"text": "a" * (1 << 20)})
        )
    with pytest.raises(TypeError):
        _changed(_frames()[0], timestep=None)


def test_nested_action_trees_share_a_total_byte_budget() -> None:
    with pytest.raises(ValueError, match="bounds"):
        ReplayAction(
            "large",
            {
                "left": {"choice": np.zeros((150000,), dtype=np.float32)},
                "right": {"choice": np.zeros((150000,), dtype=np.float32)},
            },
        )
    nested = {"choice": np.array([0], dtype=np.int64)}
    for _index in range(10):
        nested = {"child": nested}
    with pytest.raises(ValueError, match="bounds"):
        ReplayAction("deep", nested)
    valid = ReplayAction("nested", {"group": {"choice": np.array([0], dtype=np.int64)}})
    assert valid.to_mapping()["action"]["group"]["choice"]["shape"] == (1,)


def test_masked_and_uncaptured_requests_fail_before_advancing_replay(artifact: Path) -> None:
    suite = _suite()
    result = _evaluate(
        suite,
        artifact,
        lambda current, _legal, _binding: replace(
            _decision(suite, current.step_id),
            action=_action(1),
        ),
    )
    assert not result.passed
    assert result.completed_steps == 0
    assert any("unrecorded branch" in issue for issue in result.issues)
    frames = list(_frames())
    frame = frames[0].snapshot()
    frames[0] = _changed(
        frames[0], timestep=replace(frame, action_mask={"choice": np.array([False, True])})
    )
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert not result.passed
    assert result.extra_checks["evaluation.illegal_action_requests"] == 1


def test_out_of_range_mask_coordinates_fail_closed(artifact: Path) -> None:
    frames = list(_frames())
    invalid = ReplayAction("select-0", _action().action, (MaskIndex("choice", (3,)),))
    frames[0] = _changed(frames[0], legal_actions=(invalid,))
    suite = _suite(
        episodes=(ReplayEpisode("frozen-export", "3" * 64, tuple(frames), (invalid, _action())),)
    )
    result = _evaluate(
        suite,
        artifact,
        lambda current, legal, _binding: replace(
            _decision(suite, current.step_id),
            action=legal[0],
        ),
    )
    assert not result.passed
    assert any("out-of-range" in issue for issue in result.issues)


def test_no_mask_adapter_still_uses_exact_frozen_legal_actions(artifact: Path) -> None:
    frames = []
    for frame in _frames():
        frames.append(
            _changed(
                frame,
                timestep=replace(frame.snapshot(), action_mask=None),
                legal_actions=tuple(
                    ReplayAction(action.semantic, action.action) for action in frame.legal_actions
                ),
            )
        )
    action = ReplayAction("select-0", _action().action)
    suite = _suite(
        spec=replace(_spec(), action_mask=None),
        episodes=(
            ReplayEpisode(
                "frozen-export",
                "3" * 64,
                tuple(frames),
                (action, action),
                run_id="synthetic.source-run",
            ),
        ),
    )
    result = _evaluate(
        suite,
        artifact,
        lambda current, _legal, _binding: replace(
            _decision(suite, current.step_id),
            action=action,
        ),
    )
    assert result.passed, result.issues


def test_absent_receipt_and_bootstrap_values_cannot_mint_zero(artifact: Path) -> None:
    frames = list(_frames())
    frames[1] = _changed(frames[1], timestep=replace(frames[1].snapshot(), action_receipt=None))
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[2]] is None
    assert result.check_counts[REPLAY_CHECK_METRICS[6]] is None
    suite = _suite()
    result = _evaluate(
        suite,
        artifact,
        lambda current, _legal, _binding: replace(
            _decision(suite, current.step_id),
            audit=replace(
                _decision(suite, current.step_id).audit, bootstrap=BootstrapAudit({}, None)
            ),
        ),
    )
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[4]] is None


def test_wrong_typed_callback_output_is_refused(artifact: Path) -> None:
    result = _evaluate(_suite(), artifact, lambda *_args: {"claimed_passed": True})
    assert not result.passed
    assert any("untyped" in issue for issue in result.issues)
    with pytest.raises(TypeError):
        ReplayDecision(None, DecisionAudit(True))
    with pytest.raises(TypeError):
        DecisionAudit(1)
    with pytest.raises(TypeError):
        DecisionAudit(True, bootstrap={})
    with pytest.raises(ValueError):
        BootstrapAudit({"action": float("nan")}, "action")
    with pytest.raises(ValueError):
        BootstrapAudit([], None)


def test_candidate_removal_is_a_measured_artifact_failure(artifact: Path) -> None:
    suite = _suite()

    def removing(current, _legal, _binding):
        if artifact.exists():
            artifact.unlink()
        return _decision(suite, current.step_id)

    result = _evaluate(suite, artifact, removing)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[0]] == 1


def test_unused_finding_and_passive_gap_receipts_are_checked_even_without_claimed_success(
    artifact: Path,
) -> None:
    suite = _suite()

    def policy(current, legal, binding):
        receipt = DecisionConsumptionReceipt(
            f"decision-{current.step_id}",
            "unexpected-finding",
            "4" * 64,
            "scorer",
            binding,
            ConsumptionState.USED,
            "5" * 64,
        )
        gap = CapabilityGap(
            "foreign-decision",
            CapabilityGapKind.MISSING_ACTION,
            "revise-options",
            tuple(action.semantic for action in legal),
            "6" * 64,
        )
        return replace(
            _decision(suite, current.step_id), consumptions=(receipt,), capability_gaps=(gap,)
        )

    result = _evaluate(suite, artifact, policy)
    assert not result.passed
    assert result.extra_checks["evaluation.rule_consumption_errors"] == 2
    assert result.extra_checks["evaluation.capability_gap_requests"] == 2


def test_callback_captures_bytes_and_fixed_default_data_but_refuses_opaque_objects() -> None:
    frozen = b"external-evaluator-config"

    def policy(_current, _legal, _binding, config=1):
        return config, frozen

    assert len(evaluator_sha256(policy)) == 64
    opaque = object()

    def invalid(_current, _legal, _binding):
        return opaque

    with pytest.raises(TypeError):
        evaluator_sha256(invalid)


@pytest.mark.parametrize(
    "outcome", [ActionOutcome.UNKNOWN, ActionOutcome.PARTIAL, ActionOutcome.INDETERMINATE]
)
def test_unresolved_receipt_never_returns_learner_data_or_advances_replay(
    artifact: Path,
    outcome: ActionOutcome,
) -> None:
    frames = list(_frames())
    source = frames[1].snapshot()
    frames[1] = _changed(
        frames[1],
        timestep=replace(
            source,
            action_receipt=replace(
                source.action_receipt,
                outcome=outcome,
            ),
        ),
    )
    suite = _suite(tuple(frames))
    replay = ReplayEnvironment(suite)
    replay.reset()
    with pytest.raises(ContractViolation, match="not learner-facing"):
        replay.step(_action().action)
    assert replay.current_frame.snapshot().step_id == 0
    with pytest.raises(ContractViolation, match="not learner-facing"):
        replay.step(_action().action)
    assert replay.current_frame.snapshot().step_id == 0
    result = _evaluate(suite, artifact)
    assert not result.passed
    assert result.completed_steps == 0
    assert result.extra_checks["evaluation.indeterminate_actions"] == 1


@pytest.mark.parametrize(
    "outcome", [ActionOutcome.REJECTED, ActionOutcome.NO_EFFECT, ActionOutcome.BLOCKED]
)
def test_resolved_nonaccepted_receipt_can_replay_only_independent_reward(
    artifact: Path,
    outcome: ActionOutcome,
) -> None:
    frames = list(_frames())
    post = frames[1].snapshot()
    receipt = replace(post.action_receipt, outcome=outcome)
    frames[1] = _changed(frames[1], timestep=replace(post, action_receipt=receipt))
    accepted_reward = _evaluate(_suite(tuple(frames)), artifact)
    assert not accepted_reward.passed
    assert accepted_reward.completed_steps == 2
    assert accepted_reward.check_counts[REPLAY_CHECK_METRICS[5]] == 1
    assert accepted_reward.extra_checks["evaluation.indeterminate_actions"] == 0

    frames[1] = _changed(
        frames[1],
        timestep=replace(
            post,
            action_receipt=receipt,
            reward=np.array([-0.25], np.float32),
            info={
                **post.info,
                REWARD_EVIDENCE_KEY: {
                    "signals": [
                        {"name": "progress", "source": "runtime", "value": 0.0},
                        {"name": "time-cost", "source": "runtime", "value": -0.25},
                    ],
                    "attributions": [],
                },
            },
        ),
        reward_contributions=(RewardContribution("time-cost", -0.25),),
    )
    suite = _suite(tuple(frames))
    replay = ReplayEnvironment(suite)
    replay.reset()
    following = replay.step(_action().action)
    assert following.action_receipt.outcome is outcome
    assert following.step_id == 1
    assert following.reward[0] == -0.25
    independent_reward = _evaluate(suite, artifact)
    assert independent_reward.passed
    assert independent_reward.completed_steps == 2
    assert independent_reward.check_counts[REPLAY_CHECK_METRICS[5]] == 0
    assert independent_reward.extra_checks["evaluation.indeterminate_actions"] == 0


def test_same_consumption_receipt_cannot_be_reused_through_duplicate_decision_ids() -> None:
    frames = list(_frames())
    frames[1] = _changed(frames[1], decision_id=frames[0].decision_id)
    with pytest.raises(ValueError, match="decision identities"):
        _suite(tuple(frames))


def test_owner_guard_can_reject_between_steps_and_keeps_source_decision_identity(
    artifact: Path,
) -> None:
    suite = _suite()
    replay = ReplayEnvironment(suite)
    assert replay.reset().info["replay_source"]["decision_id"] == "decision-0"
    calls = []

    def guard():
        calls.append(1)
        if len(calls) == 3:
            raise ContractViolation("campaign deadline reached after callback")

    result = evaluate_replay(
        suite,
        artifact,
        lambda current, _legal, _binding: _decision(suite, current.step_id),
        expected_suite_sha256=suite.sha256,
        expected_candidate_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        step_guard=guard,
    )
    assert not result.passed
    assert result.completed_steps == 0
    assert any("deadline" in issue for issue in result.issues)


def test_callback_code_digest_includes_exception_table() -> None:
    def catching(_current, _legal, _binding):
        try:
            raise ValueError("original handler")
        except ValueError:
            return None

    original = evaluator_sha256(catching)
    original_code = catching.__code__
    if not hasattr(original_code, "co_exceptiontable"):
        pytest.skip("this Python version has no exception table")
    catching.__code__ = original_code.replace(co_exceptiontable=b"")
    assert evaluator_sha256(catching) != original


def test_callback_capture_digest_preserves_list_tuple_and_mapping_order() -> None:
    value = [0]

    def callback(_current, _legal, _binding):
        return value

    original = evaluator_sha256(callback)
    callback.__closure__[0].cell_contents = (0,)
    assert evaluator_sha256(callback) != original
    callback.__closure__[0].cell_contents = {"a": 1, "b": 2}
    original = evaluator_sha256(callback)
    callback.__closure__[0].cell_contents = {"b": 2, "a": 1}
    assert evaluator_sha256(callback) != original


@pytest.mark.parametrize("amount", [5.399999999999999, float(np.float32(5.4))])
def test_reward_terms_accept_float_roundtrip_without_relaxing_attribution(
    artifact: Path,
    amount: float,
) -> None:
    frames = list(_frames())
    frames[1] = _changed(
        frames[1],
        timestep=replace(
            frames[1].snapshot(),
            reward=np.array([5.4], dtype=np.float32),
            info={
                **frames[1].snapshot().info,
                REWARD_EVIDENCE_KEY: {
                    **_reward_evidence(1),
                    "signals": [{"name": "progress", "source": "runtime", "value": 5.4}],
                },
            },
        ),
        reward_contributions=(RewardContribution("progress", 5.4, "action-1", True),),
    )
    suite = _suite(tuple(frames))

    def roundtripped(current, _legal, _binding):
        decision = _decision(suite, current.step_id)
        if current.step_id != 0:
            return decision
        return replace(
            decision,
            audit=replace(
                decision.audit,
                reward_contributions=(RewardContribution("progress", amount, "action-1", True),),
            ),
        )

    result = _evaluate(suite, artifact, roundtripped)
    assert result.passed, result.issues
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] == 0


@pytest.mark.parametrize("change", ["term", "action", "accepted", "amount", "order", "missing"])
def test_reward_float_tolerance_preserves_term_identity_order_and_material_difference(
    artifact: Path,
    change: str,
) -> None:
    frames = list(_frames())
    expected = (
        RewardContribution("progress", 5.4, "action-1", True),
        RewardContribution("ambient", 0.0),
    )
    frames[1] = _changed(
        frames[1],
        timestep=replace(
            frames[1].snapshot(),
            reward=np.array([5.4], dtype=np.float32),
            info={
                **frames[1].snapshot().info,
                REWARD_EVIDENCE_KEY: {
                    **_reward_evidence(1),
                    "signals": [{"name": "progress", "source": "runtime", "value": 5.4}],
                },
            },
        ),
        reward_contributions=expected,
    )
    suite = _suite(tuple(frames))

    def mismatched(current, _legal, _binding):
        decision = _decision(suite, current.step_id)
        if current.step_id != 0:
            return decision
        if change == "order":
            observed = expected[::-1]
        elif change == "missing":
            observed = expected[:1]
        else:
            alteration = {
                "term": {"term_id": "different-term"},
                "action": {"action_id": "different-action"},
                "accepted": {"requires_accepted": False},
                "amount": {"amount": 5.4001},
            }[change]
            observed = (replace(expected[0], **alteration), expected[1])
        return replace(decision, audit=replace(decision.audit, reward_contributions=observed))

    result = _evaluate(suite, artifact, mismatched)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] == 1


def test_per_term_relative_tolerance_cannot_hide_a_total_reward_cancellation_error(
    artifact: Path,
) -> None:
    frames = list(_frames())
    frames[1] = _changed(
        frames[1],
        timestep=replace(frames[1].snapshot(), reward=np.array([0.0], dtype=np.float32)),
        reward_contributions=(RewardContribution("gain", 1e6), RewardContribution("loss", -1e6)),
    )
    suite = _suite(tuple(frames))

    def wrong_total(current, _legal, _binding):
        decision = _decision(suite, current.step_id)
        if current.step_id != 0:
            return decision
        return replace(
            decision,
            audit=replace(
                decision.audit,
                reward_contributions=(
                    RewardContribution("gain", 1000000.5),
                    RewardContribution("loss", -999999.5),
                ),
            ),
        )

    result = _evaluate(suite, artifact, wrong_total)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] == 1


@pytest.mark.parametrize(
    "field",
    ["run_id", "environment_id", "protocol_version", "target_id", "environment_config_sha256"],
)
def test_strict_causal_context_identity_is_bound_to_fixed_source(
    field: str, artifact: Path
) -> None:
    frames = list(_frames())
    step = frames[2].snapshot()
    context = dict(step.info[OBSERVATION_CONTEXT_KEY])
    context[field] = "8" * 64 if field == "environment_config_sha256" else "foreign.identity"
    frames[2] = _changed(
        frames[2], timestep=replace(step, info={**step.info, OBSERVATION_CONTEXT_KEY: context})
    )
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] == 1
    assert not result.passed


@pytest.mark.parametrize("missing", ["context", "reward", "run", "contract", "alive"])
def test_missing_strict_causal_measurement_remains_unknown(missing: str, artifact: Path) -> None:
    frames = list(_frames())
    step = frames[1].snapshot()
    info = dict(step.info)
    if missing == "context":
        info.pop(OBSERVATION_CONTEXT_KEY)
    elif missing == "reward":
        info.pop(REWARD_EVIDENCE_KEY)
    elif missing == "alive":
        info[OBSERVATION_CONTEXT_KEY] = {**info[OBSERVATION_CONTEXT_KEY], "alive": None}
    frames[1] = _changed(frames[1], timestep=replace(step, info=info))
    suite = _suite(tuple(frames))
    if missing == "run":
        suite = replace(suite, episodes=(replace(suite.episodes[0], run_id=None),))
    elif missing == "contract":
        suite = replace(suite, reward_training=None, reward_safety=None)
    result = _evaluate(suite, artifact)
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] is None
    assert result.coverage[REPLAY_CHECK_METRICS[5]] is CheckCoverage.UNKNOWN
    assert not result.passed


def test_fixed_reward_contract_changes_suite_identity_before_callback(artifact: Path) -> None:
    original = _suite()
    changed = replace(
        original, reward_safety=replace(original.reward_safety, max_positive_shaping_per_step=1)
    )
    calls = []

    def policy(*_args):
        calls.append(True)
        raise AssertionError("wrong fixed suite must not invoke callback")

    with pytest.raises(ContractViolation, match="suite"):
        evaluate_replay(
            changed,
            artifact,
            policy,
            expected_suite_sha256=original.sha256,
            expected_candidate_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )
    assert not calls


@pytest.mark.parametrize("effect", ["no_effect", "unknown"])
def test_strict_positive_credit_requires_explicit_effect(effect: str, artifact: Path) -> None:
    frames = list(_frames())
    step = frames[1].snapshot()
    evidence = _reward_evidence(1)
    evidence["attributions"][0]["effect"] = effect
    frames[1] = _changed(
        frames[1], timestep=replace(step, info={**step.info, REWARD_EVIDENCE_KEY: evidence})
    )
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] == 1
    assert not result.passed


@pytest.mark.parametrize(
    "run_id,epoch,expected_errors",
    [
        ("synthetic.source-run", UUID(int=1), 1),
        ("synthetic.other-run", UUID(int=1), 0),
        ("synthetic.source-run", UUID(int=2), 0),
    ],
)
def test_source_reset_epochs_are_unique_within_their_run(
    artifact: Path, run_id: str, epoch: UUID, expected_errors: int
) -> None:
    first = _suite().episodes[0]
    other_frames = []
    for frame in first.frames:
        step = frame.snapshot()
        receipt = step.action_receipt
        context = {**step.info[OBSERVATION_CONTEXT_KEY], "run_id": run_id, "episode_id": str(epoch)}
        other_frames.append(
            _changed(
                frame,
                timestep=replace(
                    step,
                    episode_id=epoch,
                    action_receipt=None if receipt is None else replace(receipt, episode_id=epoch),
                    info={**step.info, OBSERVATION_CONTEXT_KEY: context},
                ),
                decision_id=f"other-{frame.decision_id}",
            )
        )
    second = ReplayEpisode(
        "frozen-export-2", "4" * 64, tuple(other_frames), first.recorded_actions, run_id=run_id
    )
    result = _evaluate(_suite(episodes=(first, second)), artifact)
    assert result.check_counts[REPLAY_CHECK_METRICS[1]] == expected_errors
    assert result.passed is (expected_errors == 0), result.issues


def test_same_scalar_cannot_replace_the_composed_reward_term(artifact: Path) -> None:
    frames = list(_frames())
    frames[1] = _changed(
        frames[1],
        reward_contributions=(RewardContribution("unmeasured-bonus", 1.0, "action-1", True),),
    )
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert not result.passed
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] == 1


def test_terminal_dead_penalty_remains_a_valid_captured_transition(artifact: Path) -> None:
    frames = list(_frames())
    step = frames[2].snapshot()
    evidence = {
        "signals": [
            {"name": "progress", "source": "runtime", "value": 0.0},
            {"name": "outcome", "source": "runtime", "value": -1.0},
        ],
        "attributions": [],
    }
    frames[2] = _changed(
        frames[2],
        lifecycle=ObservationLifecycle.DEAD,
        timestep=replace(
            step,
            reward=np.array([-1.0], np.float32),
            info={
                **step.info,
                OBSERVATION_CONTEXT_KEY: {**_context(2), "phase": "gameplay", "alive": False},
                REWARD_EVIDENCE_KEY: evidence,
            },
        ),
        reward_contributions=(RewardContribution("outcome", -1.0),),
    )
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert result.passed, result.issues
    assert result.check_counts[REPLAY_CHECK_METRICS[3]] == 0


def test_evaluator_fingerprint_includes_the_strict_reward_implementation(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from game_learning_runtime import correlated_rewards

    baseline = evaluator_sha256()
    frozen_source = artifact.parent / "synthetic-reward-implementation.py"
    frozen_source.write_text("SYNTHETIC_REVISION = 1\n", encoding="utf-8")
    monkeypatch.setattr(correlated_rewards, "__file__", str(frozen_source))
    assert evaluator_sha256() != baseline


@pytest.mark.parametrize(
    "claim,receipt_outcome,expected_errors",
    [
        ("confirmed", ActionOutcome.ACCEPTED, 0),
        (None, ActionOutcome.ACCEPTED, 1),
        ("unknown", ActionOutcome.ACCEPTED, 1),
        ("confirmed", ActionOutcome.NO_EFFECT, 1),
    ],
)
def test_fixed_positive_outcome_requires_its_own_action_effect_claim(
    artifact: Path, claim: str | None, receipt_outcome: ActionOutcome, expected_errors: int
) -> None:
    frames = list(_frames())
    step = frames[2].snapshot()
    attributions = (
        []
        if claim is None
        else [
            {
                "signal_name": "outcome",
                "source": "runtime",
                "action_id": "action-2",
                "before_sequence": 12,
                "after_sequence": 14,
                "effect": claim,
            }
        ]
    )
    frames[2] = _changed(
        frames[2],
        timestep=replace(
            step,
            reward=np.array([2.0], np.float32),
            action_receipt=replace(step.action_receipt, outcome=receipt_outcome),
            info={
                **step.info,
                REWARD_EVIDENCE_KEY: {
                    "signals": [
                        {"name": "progress", "source": "runtime", "value": 0.0},
                        {"name": "outcome", "source": "runtime", "value": 2.0},
                    ],
                    "attributions": attributions,
                },
            },
        ),
        reward_contributions=(RewardContribution("outcome", 2.0, "action-2", True),),
    )
    result = _evaluate(_suite(tuple(frames)), artifact)
    assert result.check_counts[REPLAY_CHECK_METRICS[5]] == expected_errors
    assert result.passed is (expected_errors == 0), result.issues
