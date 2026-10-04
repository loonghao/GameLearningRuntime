"""Synthetic collector and durable diagnostic integration for strict rewards."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime.collector import SyncCollector
from game_learning_runtime.contracts import (
    ActionOutcome,
    ActionReceipt,
    TensorTree,
    TimeStep,
    environment_config_digest,
)
from game_learning_runtime.correlated_rewards import (
    OBSERVATION_CONTEXT_KEY,
    REWARD_EVIDENCE_KEY,
    CorrelatedRewardGuard,
    CorrelatedRewardReceipt,
    CorrelationPolicy,
    EffectState,
    LearningConsumerPolicy,
    ObservationContext,
    RewardAttribution,
    ScalarLearningUpdate,
    tensor_tree_sha256,
)
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.phases import EnvironmentPhase
from game_learning_runtime.run_store import TrainingStore
from game_learning_runtime.specs import CompositeSpec, EnvironmentSpec, SpaceKind, TensorSpec
from game_learning_runtime.telemetry import Telemetry
from game_learning_runtime.termination import IndeterminateOutcomeError, TerminationReason
from game_learning_runtime.training import TrainingConfig
from game_learning_runtime.training_safety import RewardSafetyConfig

_RUN_ID = "synthetic-collection-run"
_SNAPSHOT = {"scenario": "synthetic-contract", "revision": "one"}
_CONFIG_SHA = environment_config_digest(_SNAPSHOT)
assert _CONFIG_SHA is not None
_POLICY = CorrelationPolicy(
    _RUN_ID, "synthetic-collector", "synthetic.v1", "synthetic-target", _CONFIG_SHA
)
_ACTION = {"move": np.asarray([1], dtype=np.int64)}


def _guard(
    *, guard_type=CorrelatedRewardGuard, max_actions_per_episode=4096, max_episodes=4096
) -> CorrelatedRewardGuard:
    training = TrainingConfig.from_mapping(
        {
            "schema_version": "glr.training.v1",
            "lifecycle": {"start_mode": "reset", "stop_on_done": True},
            "bridge": {"required_capabilities": []},
            "knowledge_sources": [
                {"id": "adapter", "authority": "authoritative", "required": True}
            ],
            "reward": {
                "minimum": -25,
                "maximum": 25,
                "terms": [
                    {
                        "name": "effect",
                        "source": "adapter",
                        "weight": 1,
                        "minimum": -5,
                        "maximum": 5,
                        "required": True,
                    },
                    {
                        "name": "finish",
                        "source": "adapter",
                        "weight": 5,
                        "minimum": -1,
                        "maximum": 1,
                        "required": False,
                    },
                ],
            },
        }
    )
    safety = RewardSafetyConfig.from_mapping(
        {
            "schema_version": "glr.reward-safety.v1",
            "outcome_signal": "finish",
            "shaping_signals": ["effect"],
            "max_positive_shaping_per_step": 2,
            "max_positive_shaping_per_episode": 4,
            "max_negative_shaping_per_step": 3,
            "max_negative_shaping_per_episode": 7,
            "failure_episode_maximum": 0,
            "require_terminal_outcome": True,
        }
    )
    return guard_type(
        training,
        safety,
        _POLICY,
        max_actions_per_episode=max_actions_per_episode,
        max_episodes=max_episodes,
    )


class _SyntheticEnvironment(GameEnvironment):
    """An in-memory adapter whose lifecycle and effect facts are configurable."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.actions: list[TensorTree] = []
        self.omit_start_context = False
        self.start_alive: bool | None = True
        self.omit_post_context = False
        self.post_effect = EffectState.CONFIRMED
        self.post_reward = 1.25
        self.post_target = _POLICY.target_id
        self.post_outcome = ActionOutcome.ACCEPTED
        self.terminal = False
        self.snapshot: dict[str, str] | None = dict(_SNAPSHOT)
        self._step = 0
        self._episode = UUID(int=0x200)
        self._spec = EnvironmentSpec(
            environment_id=_POLICY.environment_id,
            protocol_version=_POLICY.protocol_version,
            observation=CompositeSpec({"sensor": TensorSpec((1,), np.float32)}),
            action=CompositeSpec(
                {"move": TensorSpec((1,), np.int64, SpaceKind.DISCRETE, minimum=0, maximum=1)}
            ),
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def config_snapshot(self) -> Mapping[str, str] | None:
        return self.snapshot

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        del seed, options
        self.reset_calls += 1
        self._step = 0
        self._episode = UUID(int=0x200 + self.reset_calls)
        return self._timestep(start=True)

    def step(self, action: TensorTree) -> TimeStep:
        self.step_calls += 1
        self.actions.append(action)
        self._step += 1
        return self._timestep(start=False)

    def _context(self, *, start: bool) -> ObservationContext:
        return ObservationContext(
            **{name: getattr(_POLICY, name) for name in _POLICY.__dataclass_fields__},
            episode_id=self._episode,
            step_id=self._step,
            producer_sequence=70 + 2 * self._step,
            timestamp_ns=2_000 + 100 * self._step,
            phase=EnvironmentPhase.GAMEPLAY,
            alive=self.start_alive if start else True,
        )

    def _timestep(self, *, start: bool) -> TimeStep:
        context = self._context(start=start)
        info: dict[str, Any] = {"observation_sequence": context.producer_sequence}
        if not (self.omit_start_context if start else self.omit_post_context):
            info[OBSERVATION_CONTEXT_KEY] = context.to_mapping()
        receipt = None
        reward = 0.0 if start else self.post_reward
        if not start:
            action_id = f"synthetic-{self.reset_calls}-{self._step}"
            receipt = ActionReceipt(
                action_id=action_id,
                episode_id=self._episode,
                step_id=self._step,
                outcome=self.post_outcome,
                issued_timestamp_ns=context.timestamp_ns - 90,
                observed_timestamp_ns=context.timestamp_ns - 20,
                target_id=self.post_target,
                issued_against_observation_sequence=context.producer_sequence - 2,
                authoritative_observation_sequence=context.producer_sequence,
            )
            claim = RewardAttribution(
                "effect",
                "adapter",
                action_id,
                context.producer_sequence - 2,
                context.producer_sequence,
                self.post_effect,
            )
            signals = [{"name": "effect", "source": "adapter", "value": 1.25}]
            if self.terminal:
                signals.append({"name": "finish", "source": "adapter", "value": 0.5})
                reward = 3.75
                info["termination_reason"] = TerminationReason.GOAL_REACHED.value
            info[REWARD_EVIDENCE_KEY] = {
                "signals": signals,
                "attributions": [claim.to_mapping()]
                + ([replace(claim, signal_name="finish").to_mapping()] if self.terminal else []),
            }
        return TimeStep(
            observation={"sensor": np.asarray([self._step + 0.25], dtype=np.float32)},
            reward=np.asarray([reward], dtype=np.float32),
            terminated=np.asarray([self.terminal and not start]),
            truncated=np.asarray([False]),
            episode_id=self._episode,
            step_id=self._step,
            timestamp_ns=context.timestamp_ns,
            action_receipt=receipt,
            info=info,
        )


class _ObservedPolicy:
    def __init__(self) -> None:
        self.calls: list[TimeStep] = []

    def __call__(self, timestep: TimeStep) -> TensorTree:
        self.calls.append(timestep)
        return _ACTION


def _store(tmp_path: Path) -> TrainingStore:
    store = TrainingStore(tmp_path / "synthetic.sqlite3")
    store.create_run(
        run_id=_RUN_ID,
        environment_id=_POLICY.environment_id,
        protocol_version=_POLICY.protocol_version,
        kind="training",
        environment_config_snapshot=_SNAPSHOT,
    )
    return store


def test_collector_binds_actual_transition_inputs_and_keeps_pre_step_identity() -> None:
    environment = _SyntheticEnvironment()
    policy = _ObservedPolicy()
    collector = SyncCollector(ContractEnvironment(environment), correlated_rewards=_guard())
    unroll = collector.collect(policy, steps=2, policy_version=3)

    assert len(policy.calls) == environment.step_calls == 2
    assert environment.reset_calls == 1
    assert unroll.environment_config_digest == _CONFIG_SHA
    assert [transition.step_id for transition in unroll.transitions] == [0, 1]
    for transition in unroll.transitions:
        assert transition.action_receipt is not None
        assert transition.action_receipt.step_id == transition.step_id + 1
        assert transition.provenance is not None
        receipt = transition.provenance["correlated_reward"]
        assert receipt["state_sha256"] == tensor_tree_sha256(transition.observation)
        assert receipt["action_sha256"] == tensor_tree_sha256(transition.action)
        assert receipt["next_state_sha256"] == tensor_tree_sha256(transition.next_observation)
        assert receipt["reward"] == float(transition.reward.item())
        assert len(transition.provenance["correlated_reward_sha256"]) == 64


@pytest.mark.parametrize("mode", ["missing", "unknown", "dead"])
def test_invalid_pre_observation_prevents_policy_and_environment_action(mode: str) -> None:
    environment = _SyntheticEnvironment()
    environment.omit_start_context = mode == "missing"
    environment.start_alive = None if mode == "unknown" else mode != "dead"
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard())
    with pytest.raises((ContractViolation, ValueError)):
        collector.collect(policy, steps=1)
    assert policy.calls == []
    assert environment.step_calls == 0


@pytest.mark.parametrize("snapshot", [None, {"scenario": "different"}])
def test_configuration_requires_owner_frozen_snapshot_before_policy(snapshot) -> None:
    environment = _SyntheticEnvironment()
    environment.snapshot = snapshot
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard())
    with pytest.raises(ContractViolation, match="configuration"):
        collector.collect(policy, steps=1)
    assert policy.calls == []
    assert environment.step_calls == 0


@pytest.mark.parametrize("mode", ["effect", "reward", "target", "missing"])
def test_rejected_post_interval_requires_fresh_reset_before_next_action(mode: str) -> None:
    environment = _SyntheticEnvironment()
    environment.post_effect = EffectState.UNKNOWN if mode == "effect" else EffectState.CONFIRMED
    environment.post_reward = 8 if mode == "reward" else 1.25
    environment.post_target = "other-target" if mode == "target" else _POLICY.target_id
    environment.omit_post_context = mode == "missing"
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard())
    with pytest.raises((ContractViolation, ValueError)):
        collector.collect(policy, steps=1)
    rejected_episode = policy.calls[0].episode_id
    assert environment.reset_calls == environment.step_calls == len(policy.calls) == 1

    environment.post_effect = EffectState.CONFIRMED
    environment.post_reward = 1.25
    environment.post_target = _POLICY.target_id
    environment.omit_post_context = False
    unroll = collector.collect(policy, steps=1)
    assert environment.reset_calls == environment.step_calls == len(policy.calls) == 2
    assert unroll.transitions[0].episode_id != rejected_episode
    assert policy.calls[1].step_id == 0
    assert float(unroll.transitions[0].reward.item()) == 1.25


def test_terminal_collection_preserves_completion_reward_and_next_collect_opens_new_episode() -> (
    None
):
    environment = _SyntheticEnvironment()
    environment.terminal = True
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard())
    first = collector.collect(policy, steps=3, stop_on_done=True)
    assert len(first.transitions) == 1
    assert first.transitions[0].done
    assert float(first.transitions[0].reward.item()) == 3.75
    following = collector.collect(policy, steps=1, stop_on_done=True)
    assert following.transitions[0].episode_id != first.transitions[0].episode_id
    assert environment.reset_calls == 2


def test_collector_persists_only_projected_receipt_and_digest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    environment = _SyntheticEnvironment()
    collector = SyncCollector(environment, correlated_rewards=_guard(), store=store, run_id=_RUN_ID)
    transition = collector.collect(_ObservedPolicy(), steps=1).transitions[0]
    reopened = TrainingStore(store.path)
    events = [event for event in reopened.list_events(_RUN_ID) if event.kind == "reward.correlated"]
    assert len(events) == 1
    assert events[0].step_id == transition.step_id == 0
    assert events[0].episode_id == f"episode-{transition.episode_id}"
    assert events[0].payload["authority"] == "diagnostic"
    assert transition.provenance is not None
    assert events[0].payload["receipt_sha256"] == transition.provenance["correlated_reward_sha256"]
    receipt = events[0].payload["receipt"]
    assert receipt["before"]["episode_id"] == str(transition.episode_id)
    assert receipt["after"]["episode_id"] == str(transition.episode_id)
    assert receipt["before"]["alive"] is True
    assert receipt["before"]["phase"] == "gameplay"
    assert receipt["action_sha256"] == tensor_tree_sha256(_ACTION)
    serialized = json.dumps(dict(events[0].payload))
    assert "sensor" not in serialized
    assert 'observation":' not in serialized
    assert reopened.list_metrics(_RUN_ID) == ()


def test_rejected_reward_is_not_persisted_as_correlated_evidence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    environment = _SyntheticEnvironment()
    environment.post_effect = EffectState.UNKNOWN
    collector = SyncCollector(environment, correlated_rewards=_guard(), store=store, run_id=_RUN_ID)
    with pytest.raises(ContractViolation):
        collector.collect(_ObservedPolicy(), steps=1)
    assert not any(event.kind == "reward.correlated" for event in store.list_events(_RUN_ID))


def _receipt():
    environment = _SyntheticEnvironment()
    before = environment.reset()
    after = environment.step(_ACTION)
    guard = _guard()
    guard.reset(before.episode_id)
    return guard.compose_timestep(before, after, action=_ACTION)


def _update(receipt, *, learner_id="synthetic-learner", table_id="synthetic-table"):
    return ScalarLearningUpdate(
        learner_id=learner_id,
        table_id=table_id,
        policy_version=4,
        state_sha256=receipt.state_sha256,
        action_sha256=receipt.action_sha256,
        next_state_sha256=receipt.next_state_sha256,
        reward_receipt_sha256=receipt.sha256,
        previous_value=0.5,
        bootstrap_value=1,
        discount=0.5,
        learning_rate=0.2,
        target=1.75,
        updated_value=0.75,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"learner_id": "other-learner"},
        {"table_id": "other-table"},
        {"policy_version": 5},
        {"state_sha256": sha256(b"other synthetic state").hexdigest()},
        {"action_sha256": sha256(b"other synthetic action").hexdigest()},
        {"next_state_sha256": sha256(b"other synthetic next state").hexdigest()},
        {"reward_receipt_sha256": sha256(b"other synthetic receipt").hexdigest()},
        {"updated_value": 0.85},
        {"target": 1.85},
    ],
)
def test_invalid_learning_update_is_not_persisted(tmp_path: Path, changes: dict[str, Any]) -> None:
    store = _store(tmp_path)
    receipt = _receipt()
    update = _update(receipt)
    consumer = LearningConsumerPolicy(update.learner_id, update.table_id, update.policy_version)
    with pytest.raises(ContractViolation):
        Telemetry(store, _RUN_ID, console=False).correlated_learning_update(
            receipt, replace(update, **changes), consumer=consumer
        )
    assert store.list_events(_RUN_ID) == ()
    assert store.list_metrics(_RUN_ID) == ()


def test_two_consumers_persist_separate_explicit_table_operands(tmp_path: Path) -> None:
    store = _store(tmp_path)
    receipt = _receipt()
    telemetry = Telemetry(store, _RUN_ID, console=False)
    for learner_id, table_id in [("learner-one", "table-one"), ("learner-two", "table-two")]:
        update = _update(receipt, learner_id=learner_id, table_id=table_id)
        telemetry.correlated_learning_update(
            receipt, update, consumer=LearningConsumerPolicy(learner_id, table_id, 4)
        )
    events = TrainingStore(store.path).list_events(_RUN_ID)
    assert [event.kind for event in events] == ["learning.correlated-update"] * 2
    assert [event.payload["update"]["table_id"] for event in events] == ["table-one", "table-two"]
    assert [event.payload["update"]["learner_id"] for event in events] == [
        "learner-one",
        "learner-two",
    ]
    assert all(event.payload["authority"] == "diagnostic" for event in events)
    assert all(event.payload["update"]["updated_value"] == 0.75 for event in events)
    assert all(
        event.payload["update"]["reward_receipt_sha256"] == receipt.sha256 for event in events
    )
    assert store.list_metrics(_RUN_ID) == ()


def test_telemetry_rejects_another_run_before_persistence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    receipt = _receipt()
    update = _update(receipt)
    consumer = LearningConsumerPolicy(update.learner_id, update.table_id, 4)
    telemetry = Telemetry(store, "other-run", console=False)
    with pytest.raises(ContractViolation, match="different run"):
        telemetry.correlated_reward(receipt)
    with pytest.raises(ContractViolation, match="different run"):
        telemetry.correlated_learning_update(receipt, update, consumer=consumer)
    assert store.list_events(_RUN_ID) == ()


def test_strict_partial_error_mode_is_rejected_before_reset_or_policy() -> None:
    environment = _SyntheticEnvironment()
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard())
    with pytest.raises(ValueError, match="on_error"):
        collector.collect(policy, steps=1, on_error="partial")
    assert environment.reset_calls == environment.step_calls == 0
    assert policy.calls == []


def test_second_indeterminate_action_never_returns_partial_learner_unroll() -> None:
    class SecondIndeterminateEnvironment(_SyntheticEnvironment):
        unresolved = True

        def step(self, action: TensorTree) -> TimeStep:
            timestep = super().step(action)
            if self._step == 2 and self.unresolved:
                assert timestep.action_receipt is not None
                return replace(
                    timestep,
                    action_receipt=replace(
                        timestep.action_receipt, outcome=ActionOutcome.INDETERMINATE
                    ),
                )
            return timestep

    environment = SecondIndeterminateEnvironment()
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard())
    with pytest.raises(IndeterminateOutcomeError):
        collector.collect(policy, steps=2)
    assert environment.step_calls == len(policy.calls) == 2
    assert len(collector.terminations) == 1
    old_episode = policy.calls[0].episode_id
    environment.unresolved = False
    with pytest.raises(IndeterminateOutcomeError, match="reattach"):
        collector.collect(policy, steps=1)
    assert environment.reset_calls == 1
    assert environment.step_calls == len(policy.calls) == 2
    collector.reattach()
    unroll = collector.collect(policy, steps=1)
    assert environment.reset_calls == 2
    assert unroll.transitions[0].episode_id != old_episode
    assert unroll.transitions[0].step_id == 0
    assert not bool(unroll.transitions[0].truncated.item())


def test_collector_freezes_action_before_adapter_can_mutate_policy_input() -> None:
    class MutatingEnvironment(_SyntheticEnvironment):
        def step(self, action: TensorTree) -> TimeStep:
            move = action["move"]
            assert isinstance(move, np.ndarray)
            assert move.flags.writeable is False
            with pytest.raises(ValueError):
                move[0] = 0
            with pytest.raises(TypeError):
                action["other"] = np.asarray([1], dtype=np.int64)
            return super().step(action)

    environment = MutatingEnvironment()
    collector = SyncCollector(environment, correlated_rewards=_guard())
    transition = collector.collect(_ObservedPolicy(), steps=1).transitions[0]
    assert _ACTION["move"].flags.writeable is True
    assert int(_ACTION["move"][0]) == 1
    assert transition.provenance is not None
    assert transition.provenance["correlated_reward"]["action_sha256"] == tensor_tree_sha256(
        _ACTION
    )


def test_collector_rejects_guard_subclass_override() -> None:
    class OverriddenGuard(CorrelatedRewardGuard):
        def compose_timestep(self, before, after, *, action):
            raise AssertionError("virtual validation must never be dispatched")

    override = _guard(guard_type=OverriddenGuard)
    with pytest.raises(TypeError):
        SyncCollector(_SyntheticEnvironment(), correlated_rewards=override)


@pytest.mark.parametrize("kind", ["receipt", "update", "consumer"])
def test_telemetry_rejects_virtual_validation_or_export_overrides(
    tmp_path: Path, kind: str
) -> None:
    class OverriddenReceipt(CorrelatedRewardReceipt):
        def to_mapping(self):
            raise AssertionError("virtual export must never be dispatched")

    class OverriddenUpdate(ScalarLearningUpdate):
        def validate_against(self, receipt):
            raise AssertionError("virtual validation must never be dispatched")

    class OverriddenConsumer(LearningConsumerPolicy):
        def validate(self, update):
            raise AssertionError("virtual validation must never be dispatched")

    store = _store(tmp_path)
    receipt = _receipt()
    update = _update(receipt)
    consumer = LearningConsumerPolicy(update.learner_id, update.table_id, update.policy_version)
    if kind == "receipt":
        receipt = OverriddenReceipt(
            **{
                name: getattr(receipt, name)
                for name in CorrelatedRewardReceipt.__dataclass_fields__
            }
        )
    elif kind == "update":
        update = OverriddenUpdate(
            **{name: getattr(update, name) for name in ScalarLearningUpdate.__dataclass_fields__}
        )
    else:
        consumer = OverriddenConsumer(update.learner_id, update.table_id, update.policy_version)
    telemetry = Telemetry(store, _RUN_ID, console=False)
    if kind == "receipt":
        with pytest.raises(TypeError):
            telemetry.correlated_reward(receipt)
    with pytest.raises(TypeError):
        telemetry.correlated_learning_update(receipt, update, consumer=consumer)
    assert store.list_events(_RUN_ID) == ()


@pytest.mark.parametrize(
    "mode", ["unknown-config", "other-config", "other-environment", "other-protocol"]
)
def test_strict_collector_and_telemetry_reject_foreign_store_identity(
    tmp_path: Path, mode: str
) -> None:
    store = TrainingStore(tmp_path / "foreign.sqlite3")
    store.create_run(
        run_id=_RUN_ID,
        environment_id="other-environment"
        if mode == "other-environment"
        else _POLICY.environment_id,
        protocol_version="other.v1" if mode == "other-protocol" else _POLICY.protocol_version,
        kind="training",
        environment_config_snapshot=(
            None
            if mode == "unknown-config"
            else {"other": "configuration"}
            if mode == "other-config"
            else _SNAPSHOT
        ),
    )
    environment = _SyntheticEnvironment()
    with pytest.raises(ContractViolation):
        SyncCollector(environment, correlated_rewards=_guard(), store=store, run_id=_RUN_ID)
    receipt = _receipt()
    update = _update(receipt)
    consumer = LearningConsumerPolicy(update.learner_id, update.table_id, update.policy_version)
    telemetry = Telemetry(store, _RUN_ID, console=False)
    with pytest.raises(ContractViolation):
        telemetry.correlated_reward(receipt)
    with pytest.raises(ContractViolation):
        telemetry.correlated_learning_update(receipt, update, consumer=consumer)
    assert environment.reset_calls == environment.step_calls == 0
    assert store.list_events(_RUN_ID) == ()
    assert store.list_metrics(_RUN_ID) == ()


def test_action_budget_is_checked_before_second_policy_or_dispatch() -> None:
    environment = _SyntheticEnvironment()
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard(max_actions_per_episode=1))
    with pytest.raises(ContractViolation, match="budget"):
        collector.collect(policy, steps=2)
    assert environment.reset_calls == 1
    assert environment.step_calls == len(policy.calls) == 1


def test_episode_budget_is_checked_before_next_reset_after_terminal() -> None:
    environment = _SyntheticEnvironment()
    environment.terminal = True
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard(max_episodes=1))
    collector.collect(policy, steps=1, stop_on_done=True)
    with pytest.raises(ContractViolation, match="budget"):
        collector.collect(policy, steps=1)
    assert environment.reset_calls == environment.step_calls == len(policy.calls) == 1


def test_invalid_reset_attempts_are_bounded_without_policy_or_action() -> None:
    environment = _SyntheticEnvironment()
    environment.omit_start_context = True
    policy = _ObservedPolicy()
    collector = SyncCollector(environment, correlated_rewards=_guard(max_episodes=2))
    for _ in range(2):
        with pytest.raises((ValueError, ContractViolation)):
            collector.collect(policy, steps=1)
    with pytest.raises(ContractViolation, match="budget"):
        collector.collect(policy, steps=1)
    assert environment.reset_calls == 2
    assert environment.step_calls == 0
    assert policy.calls == []
