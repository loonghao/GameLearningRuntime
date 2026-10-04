"""Synthetic adversarial tests for the strict action-to-reward boundary."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime.contracts import ActionOutcome, ActionReceipt, TimeStep
from game_learning_runtime.correlated_rewards import (
    CORRELATED_REWARD_SCHEMA,
    OBSERVATION_CONTEXT_KEY,
    REWARD_EVIDENCE_KEY,
    CorrelatedRewardGuard,
    CorrelationPolicy,
    EffectState,
    LearningConsumerPolicy,
    ObservationContext,
    RewardAttribution,
    ScalarLearningUpdate,
    tensor_tree_sha256,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.phases import EnvironmentPhase
from game_learning_runtime.training import RewardSignal, RewardTermSpec, TrainingConfig
from game_learning_runtime.training_safety import RewardSafetyConfig

_EPISODE = UUID(int=0x101)
_POLICY = CorrelationPolicy(
    run_id="synthetic-run",
    environment_id="synthetic-environment",
    protocol_version="synthetic.v1",
    target_id="synthetic-target",
    environment_config_sha256=sha256(b"independent synthetic configuration").hexdigest(),
)
_REJECTED = (ContractViolation, ValueError, TypeError)
_ACTION = {"move": np.asarray(0, dtype=np.int64)}


def _training(
    *, weight: float = 1.0, term_minimum: float = -6.0, global_minimum: float = -30.0
) -> TrainingConfig:
    return TrainingConfig.from_mapping(
        {
            "schema_version": "glr.training.v1",
            "lifecycle": {"start_mode": "reset", "stop_on_done": True},
            "bridge": {"required_capabilities": []},
            "knowledge_sources": [
                {"id": "adapter", "authority": "authoritative", "required": True}
            ],
            "reward": {
                "minimum": global_minimum,
                "maximum": 30,
                "terms": [
                    {
                        "name": "actuator_gain",
                        "source": "adapter",
                        "weight": weight,
                        "minimum": term_minimum,
                        "maximum": 6,
                        "required": True,
                    },
                    {
                        "name": "completion",
                        "source": "adapter",
                        "weight": 9,
                        "minimum": -1,
                        "maximum": 1,
                        "required": False,
                    },
                ],
            },
        }
    )


def _safety() -> RewardSafetyConfig:
    return RewardSafetyConfig.from_mapping(
        {
            "schema_version": "glr.reward-safety.v1",
            "outcome_signal": "completion",
            "shaping_signals": ["actuator_gain"],
            "max_positive_shaping_per_step": 2,
            "max_positive_shaping_per_episode": 3,
            "max_negative_shaping_per_step": 4,
            "max_negative_shaping_per_episode": 8,
            "failure_episode_maximum": 0,
            "require_terminal_outcome": True,
        }
    )


def _guard(**training_overrides: float) -> CorrelatedRewardGuard:
    guard = CorrelatedRewardGuard(_training(**training_overrides), _safety(), _POLICY)
    guard.reset(_EPISODE)
    return guard


def _context(step: int, *, episode_id: UUID = _EPISODE) -> ObservationContext:
    return ObservationContext(
        **{name: getattr(_POLICY, name) for name in _POLICY.__dataclass_fields__},
        episode_id=episode_id,
        step_id=step,
        producer_sequence=40 + 3 * step,
        timestamp_ns=1_000 + 100 * step,
        phase=EnvironmentPhase.GAMEPLAY,
        alive=True,
    )


def _timestep(
    context: ObservationContext,
    *,
    receipt: ActionReceipt | None = None,
    reward: object = 1.5,
    done: bool = False,
    info: dict[str, Any] | None = None,
) -> TimeStep:
    return TimeStep(
        observation={"sensor": np.asarray([context.step_id], dtype=np.float32)},
        reward=np.asarray(reward, dtype=np.float32),
        terminated=np.asarray([done], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        episode_id=context.episode_id,
        step_id=context.step_id,
        timestamp_ns=context.timestamp_ns,
        action_receipt=receipt,
        info={
            OBSERVATION_CONTEXT_KEY: context.to_mapping(),
            "observation_sequence": context.producer_sequence,
            **(info or {}),
        },
    )


def _interval(
    step: int = 0,
    *,
    action_id: str | None = None,
    outcome: ActionOutcome = ActionOutcome.ACCEPTED,
    before_changes: dict[str, Any] | None = None,
    after_changes: dict[str, Any] | None = None,
    receipt_changes: dict[str, Any] | None = None,
    reward: object = 1.5,
    done: bool = False,
) -> tuple[TimeStep, TimeStep]:
    before_context = replace(_context(step), **(before_changes or {}))
    after_context = replace(_context(step + 1), **(after_changes or {}))
    receipt = ActionReceipt(
        action_id=action_id or f"synthetic-action-{step}",
        episode_id=after_context.episode_id,
        step_id=after_context.step_id,
        outcome=outcome,
        issued_timestamp_ns=before_context.timestamp_ns + 10,
        observed_timestamp_ns=before_context.timestamp_ns + 80,
        target_id=_POLICY.target_id,
        issued_against_observation_sequence=before_context.producer_sequence,
        authoritative_observation_sequence=after_context.producer_sequence,
    )
    receipt = replace(receipt, **(receipt_changes or {}))
    return (
        _timestep(before_context),
        _timestep(after_context, receipt=receipt, reward=reward, done=done),
    )


def _claim(
    before: TimeStep,
    after: TimeStep,
    *,
    effect: EffectState = EffectState.CONFIRMED,
    **changes: Any,
) -> RewardAttribution:
    assert after.action_receipt is not None
    return replace(
        RewardAttribution(
            signal_name="actuator_gain",
            source="adapter",
            action_id=after.action_receipt.action_id,
            before_sequence=ObservationContext.from_timestep(before).producer_sequence,
            after_sequence=ObservationContext.from_timestep(after).producer_sequence,
            effect=effect,
        ),
        **changes,
    )


def _signals(value: float = 1.5, *, completion: float | None = None) -> tuple[RewardSignal, ...]:
    signals = (RewardSignal("actuator_gain", "adapter", value),)
    if completion is not None:
        signals += (RewardSignal("completion", "adapter", completion),)
    return signals


def _with_evidence(before: TimeStep, after: TimeStep, *, value: float = 1.5) -> TimeStep:
    return replace(
        after,
        info={
            **after.info,
            REWARD_EVIDENCE_KEY: {
                "signals": [{"name": "actuator_gain", "source": "adapter", "value": value}],
                "attributions": [_claim(before, after).to_mapping()],
            },
        },
    )


def test_exact_post_step_receipt_composes_and_preserves_audit() -> None:
    before, after = _interval()
    result = _guard().compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)

    assert after.action_receipt is not None
    assert after.action_receipt.step_id == before.step_id + 1
    assert result.result.total == 1.5
    assert result.before.step_id == 0
    assert result.after.step_id == 1
    assert result.action_id == "synthetic-action-0"
    assert result.outcome is ActionOutcome.ACCEPTED
    assert result.to_mapping()["schema_version"] == CORRELATED_REWARD_SCHEMA
    assert result.to_mapping()["reward"] == 1.5
    assert len(result.sha256) == 64
    assert result.sha256 == result.sha256


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("run_id", "other-run"),
        ("environment_id", "other-environment"),
        ("protocol_version", "synthetic.v2"),
        ("target_id", "other-target"),
        ("environment_config_sha256", sha256(b"different synthetic configuration").hexdigest()),
    ],
)
def test_owner_frozen_identity_rejects_both_sides(side: str, name: str, value: str) -> None:
    before, after = _interval(**{f"{side}_changes": {name: value}})
    with pytest.raises(ContractViolation):
        _guard().compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_id": UUID(int=0x102)},
        {"step_id": 2},
        {"producer_sequence": 40},
        {"producer_sequence": 39},
        {"timestamp_ns": 1_070},
    ],
)
def test_reset_stale_skipped_or_early_post_state_is_rejected(changes: dict[str, Any]) -> None:
    before, after = _interval(after_changes=changes)
    with pytest.raises(_REJECTED):
        _guard().compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)


@pytest.mark.parametrize(
    "changes",
    [
        {"episode_id": UUID(int=0x103)},
        {"step_id": 2},
        {"target_id": None},
        {"target_id": "other-target"},
        {"issued_against_observation_sequence": None},
        {"issued_against_observation_sequence": 39},
        {"authoritative_observation_sequence": None},
        {"authoritative_observation_sequence": 42},
        {"issued_timestamp_ns": 999},
        {"observed_timestamp_ns": 1_101},
    ],
)
def test_receipt_must_match_exact_interval(changes: dict[str, Any]) -> None:
    before, after = _interval(receipt_changes=changes)
    with pytest.raises(_REJECTED):
        _guard().compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)


def test_missing_receipt_is_rejected() -> None:
    before, after = _interval()
    with pytest.raises(ContractViolation, match="receipt"):
        _guard().compose(
            before, replace(after, action_receipt=None), _signals(), [], action=_ACTION
        )


@pytest.mark.parametrize(
    "outcome", [ActionOutcome.UNKNOWN, ActionOutcome.PARTIAL, ActionOutcome.INDETERMINATE]
)
def test_unresolved_outcome_rejects_even_negative_rewards(outcome: ActionOutcome) -> None:
    before, after = _interval(outcome=outcome)
    with pytest.raises(ContractViolation, match="unresolved"):
        _guard().compose(before, after, _signals(-1), [], action=_ACTION)


@pytest.mark.parametrize(
    "changes",
    [
        {"alive": False},
        {"alive": None},
        {"phase": EnvironmentPhase.LOADING},
        {"phase": EnvironmentPhase.UNKNOWN},
        {"phase": EnvironmentPhase.MENU},
    ],
)
def test_action_cannot_start_from_dead_or_unsettled_state(changes: dict[str, Any]) -> None:
    before, after = _interval(before_changes=changes)
    with pytest.raises(ContractViolation, match="live gameplay"):
        _guard().compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)


def test_action_cannot_start_from_terminal_state() -> None:
    before, after = _interval()
    with pytest.raises(ContractViolation, match="live gameplay"):
        _guard().compose(
            replace(before, terminated=np.asarray([True])), after, _signals(), [], action=_ACTION
        )


@pytest.mark.parametrize("done", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        {"alive": None},
        {"phase": EnvironmentPhase.LOADING},
        {"phase": EnvironmentPhase.UNKNOWN},
    ],
)
def test_post_state_unknown_or_loading_never_becomes_learner_data(
    changes: dict[str, Any], done: bool
) -> None:
    before, after = _interval(after_changes=changes, done=done)
    with pytest.raises(ContractViolation, match="lifecycle"):
        _guard().compose(
            before, after, _signals(-1, completion=-0.5 if done else None), [], action=_ACTION
        )


def test_dead_post_state_requires_terminal_transition() -> None:
    before, after = _interval(after_changes={"alive": False})
    with pytest.raises(ContractViolation, match="lifecycle"):
        _guard().compose(before, after, _signals(-1), [], action=_ACTION)


def test_dead_terminal_preserves_authoritative_failure_penalty() -> None:
    before, after = _interval(after_changes={"alive": False}, done=True, reward=-7.2)
    result = _guard().compose(before, after, _signals(0, completion=-0.8), [], action=_ACTION)

    assert result.result.terminal is True
    assert result.result.contributions["completion"] == pytest.approx(-7.2)
    assert result.result.total == pytest.approx(-7.2)
    assert result.after.alive is False


def test_dead_terminal_rejects_positive_shaping() -> None:
    before, after = _interval(after_changes={"alive": False}, done=True)
    with pytest.raises(ContractViolation):
        _guard().compose(
            before, after, _signals(completion=-0.8), [_claim(before, after)], action=_ACTION
        )


def test_dead_terminal_cannot_claim_success_outcome() -> None:
    before, after = _interval(after_changes={"alive": False}, done=True)
    with pytest.raises(ContractViolation):
        _guard().compose(before, after, _signals(0, completion=0.8), [], action=_ACTION)


@pytest.mark.parametrize("effect", [EffectState.NO_EFFECT, EffectState.UNKNOWN])
def test_unconfirmed_effect_rejects_positive_contribution(effect: EffectState) -> None:
    before, after = _interval()
    with pytest.raises(ContractViolation, match="confirmed"):
        _guard().compose(
            before, after, _signals(), [_claim(before, after, effect=effect)], action=_ACTION
        )


@pytest.mark.parametrize(
    "outcome", [ActionOutcome.REJECTED, ActionOutcome.BLOCKED, ActionOutcome.NO_EFFECT]
)
def test_known_nonaccepted_action_cannot_claim_positive_effect(outcome: ActionOutcome) -> None:
    before, after = _interval(outcome=outcome)
    with pytest.raises(ContractViolation, match="accepted"):
        _guard().compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)


@pytest.mark.parametrize(
    "outcome", [ActionOutcome.REJECTED, ActionOutcome.BLOCKED, ActionOutcome.NO_EFFECT]
)
def test_settled_nonaccepted_action_can_preserve_negative_cost(outcome: ActionOutcome) -> None:
    before, after = _interval(outcome=outcome)
    result = _guard().compose(before, after, _signals(-1), [], action=_ACTION)
    assert result.result.total == -1
    assert result.outcome is outcome


def test_negative_value_negative_weight_cannot_bypass_effect_requirement() -> None:
    before, after = _interval()
    with pytest.raises(ContractViolation, match="confirmed"):
        _guard(weight=-1).compose(before, after, _signals(-1.5), [], action=_ACTION)


def test_term_minimum_cannot_turn_unattributed_negative_value_into_positive_reward() -> None:
    before, after = _interval()
    with pytest.raises(ContractViolation, match="confirmed"):
        _guard(term_minimum=1).compose(before, after, _signals(-1.5), [], action=_ACTION)


def test_positive_global_minimum_is_rejected_when_constructing_strict_guard() -> None:
    with pytest.raises(ContractViolation):
        _guard(global_minimum=1)


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "another-source"},
        {"action_id": "another-action"},
        {"before_sequence": 39},
        {"after_sequence": 44},
        {"signal_name": "another-signal"},
    ],
)
def test_attribution_cannot_bind_to_nearby_or_other_action(changes: dict[str, Any]) -> None:
    before, after = _interval()
    with pytest.raises(ContractViolation):
        _guard().compose(
            before, after, _signals(), [_claim(before, after, **changes)], action=_ACTION
        )


def test_duplicate_attribution_is_rejected() -> None:
    before, after = _interval()
    claim = _claim(before, after)
    with pytest.raises(ContractViolation, match="duplicate"):
        _guard().compose(before, after, _signals(), [claim, claim], action=_ACTION)


def test_failed_attribution_does_not_consume_budget_action_id_or_last_step() -> None:
    guard = _guard()
    before, after = _interval()
    with pytest.raises(ContractViolation):
        guard.compose(
            before,
            after,
            _signals(2),
            [_claim(before, after, effect=EffectState.UNKNOWN)],
            action=_ACTION,
        )

    accepted = guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    next_before, next_after = _interval(1)
    next_result = guard.compose(
        next_before, next_after, _signals(2), [_claim(next_before, next_after)], action=_ACTION
    )
    assert accepted.result.positive_shaping_total == 1.5
    assert next_result.result.total == 1.5
    assert next_result.result.positive_shaping_total == 3


def test_observed_reward_mismatch_does_not_consume_state() -> None:
    guard = _guard()
    before, after = _interval(reward=8)
    with pytest.raises(ContractViolation, match="observed reward"):
        guard.compose_timestep(before, _with_evidence(before, after), action=_ACTION)

    accepted_after = replace(after, reward=np.asarray([1.5], dtype=np.float32))
    accepted = guard.compose_timestep(
        before, _with_evidence(before, accepted_after), action=_ACTION
    )
    assert accepted.result.total == 1.5
    assert accepted.result.positive_shaping_total == 1.5


@pytest.mark.parametrize("reward", [[1.5, 1.5], [], [np.nan], [np.inf], [-np.inf]])
def test_strict_observed_reward_requires_one_finite_matching_value(reward: object) -> None:
    before, after = _interval(reward=reward)
    with pytest.raises(ContractViolation, match=r"finite.*scalar"):
        _guard().compose_timestep(before, _with_evidence(before, after), action=_ACTION)


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float16, np.complex64, object])
def test_observed_reward_rejects_other_dtypes(dtype: object) -> None:
    before, after = _interval()
    after = replace(after, reward=np.asarray([1], dtype=dtype))
    with pytest.raises(ContractViolation, match="float32 or float64"):
        _guard().compose_timestep(before, _with_evidence(before, after, value=1), action=_ACTION)


def test_budget_exhaustion_keeps_terminal_outcome_reward() -> None:
    guard = _guard()
    for step in range(2):
        before, after = _interval(step)
        guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    before, after = _interval(2, done=True)
    result = guard.compose(
        before,
        after,
        _signals(completion=0.6),
        [_claim(before, after), _claim(before, after, signal_name="completion")],
        action=_ACTION,
    )

    assert result.result.contributions["actuator_gain"] == 0
    assert result.result.contributions["completion"] == pytest.approx(5.4)
    assert result.result.total == pytest.approx(5.4)
    assert result.result.terminal is True
    assert result.result.positive_shaping_total == 3


def test_duplicate_action_and_skipped_step_fail_without_consuming_correct_next_step() -> None:
    guard = _guard()
    before, after = _interval()
    guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    for wrong_before, wrong_after in [_interval(1, action_id="synthetic-action-0"), _interval(2)]:
        with pytest.raises(ContractViolation, match="duplicated or skipped"):
            guard.compose(
                wrong_before,
                wrong_after,
                _signals(),
                [_claim(wrong_before, wrong_after)],
                action=_ACTION,
            )
    before, after = _interval(1)
    assert (
        guard.compose(
            before, after, _signals(), [_claim(before, after)], action=_ACTION
        ).result.total
        == 1.5
    )


def test_guard_requires_reset_and_rejects_reset_with_current_episode() -> None:
    guard = CorrelatedRewardGuard(_training(), _safety(), _POLICY)
    before, after = _interval()
    with pytest.raises(ContractViolation, match="reset"):
        guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    guard.reset(_EPISODE)
    with pytest.raises(ContractViolation, match="fresh episode"):
        guard.reset(_EPISODE)
    assert (
        guard.compose(
            before, after, _signals(), [_claim(before, after)], action=_ACTION
        ).result.total
        == 1.5
    )


def test_terminal_failure_does_not_consume_state_when_outcome_is_missing() -> None:
    guard = _guard()
    before, after = _interval(done=True)
    with pytest.raises(ContractViolation, match="outcome"):
        guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    result = guard.compose(before, after, _signals(0, completion=-0.4), [], action=_ACTION)
    assert result.result.total == pytest.approx(-3.6)


@pytest.mark.parametrize("name", ["step_id", "producer_sequence", "timestamp_ns"])
@pytest.mark.parametrize("value", [True, False, -1, 1.5, None])
def test_context_counters_are_strict_integers(name: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_context(0), **{name: value})


@pytest.mark.parametrize("value", [None, "true", 1, 0])
def test_known_liveness_is_strict_boolean(value: object) -> None:
    if value is None:
        before, after = _interval(before_changes={"alive": value})
        with pytest.raises(ContractViolation):
            _guard().compose(before, after, _signals(), [], action=_ACTION)
    else:
        with pytest.raises(TypeError):
            replace(_context(0), alive=value)


def test_context_mapping_round_trips_and_rejects_null_or_unknown_fields() -> None:
    context = _context(0)
    assert ObservationContext.from_mapping(context.to_mapping()) == context
    for value in [None, {}, {**context.to_mapping(), "unexpected": True}]:
        with pytest.raises(ValueError):
            ObservationContext.from_mapping(value)
    for name in ["episode_id", "phase", "run_id", "environment_config_sha256"]:
        with pytest.raises((TypeError, ValueError)):
            ObservationContext.from_mapping({**context.to_mapping(), name: None})


@pytest.mark.parametrize("effect", [None, "invented", True])
def test_attribution_mapping_rejects_unknown_effect(effect: object) -> None:
    before, after = _interval()
    mapping = _claim(before, after).to_mapping()
    with pytest.raises((TypeError, ValueError)):
        RewardAttribution.from_mapping({**mapping, "effect": effect})


def test_context_must_match_timestep_not_only_receipt() -> None:
    before, after = _interval()
    bad_before = replace(before, timestamp_ns=before.timestamp_ns + 1)
    with pytest.raises(ContractViolation, match="context"):
        _guard().compose(bad_before, after, _signals(), [], action=_ACTION)
    bad_after = replace(after, info={**after.info, "observation_sequence": True})
    with pytest.raises(ContractViolation, match="sequence"):
        _guard().compose(before, bad_after, _signals(), [], action=_ACTION)


@pytest.mark.parametrize("value", [True, False, np.nan, np.inf, -np.inf, None])
def test_signal_values_reject_bool_null_and_nonfinite(value: object) -> None:
    before, after = _interval()
    evidence_after = _with_evidence(before, after)
    evidence = dict(evidence_after.info[REWARD_EVIDENCE_KEY])
    evidence["signals"] = [{"name": "actuator_gain", "source": "adapter", "value": value}]
    with pytest.raises((TypeError, ValueError)):
        _guard().compose_timestep(
            before,
            replace(evidence_after, info={**evidence_after.info, REWARD_EVIDENCE_KEY: evidence}),
            action=_ACTION,
        )


@pytest.mark.parametrize("value", [None, {}, {"signals": [], "attributions": [], "unexpected": 1}])
def test_timestep_evidence_requires_exact_fields(value: object) -> None:
    before, after = _interval()
    with pytest.raises(ValueError):
        _guard().compose_timestep(
            before, replace(after, info={**after.info, REWARD_EVIDENCE_KEY: value}), action=_ACTION
        )


@pytest.mark.parametrize("name", ["signals", "attributions"])
@pytest.mark.parametrize("value", [None, {}, "unknown", [None] * 257])
def test_timestep_evidence_rejects_unbounded_or_wrong_sequence(name: str, value: object) -> None:
    before, after = _interval()
    evidence_after = _with_evidence(before, after)
    evidence = {**evidence_after.info[REWARD_EVIDENCE_KEY], name: value}
    with pytest.raises(ValueError):
        _guard().compose_timestep(
            before,
            replace(evidence_after, info={**evidence_after.info, REWARD_EVIDENCE_KEY: evidence}),
            action=_ACTION,
        )


def test_direct_compose_rejects_more_than_256_signals_or_claims() -> None:
    before, after = _interval()
    claim = _claim(before, after)
    for signals, claims in [(_signals() * 257, [claim]), (_signals(), [claim] * 257)]:
        with pytest.raises(ValueError):
            _guard().compose(before, after, signals, claims, action=_ACTION)


def test_mapping_validation_failure_does_not_consume_budget_or_identity() -> None:
    guard = _guard()
    before, after = _interval()
    valid = _with_evidence(before, after)
    evidence = dict(valid.info[REWARD_EVIDENCE_KEY])
    evidence["attributions"] = [{**evidence["attributions"][0], "unexpected": True}]
    with pytest.raises(ValueError):
        guard.compose_timestep(
            before,
            replace(valid, info={**valid.info, REWARD_EVIDENCE_KEY: evidence}),
            action=_ACTION,
        )
    assert (
        guard.compose_timestep(before, valid, action=_ACTION).result.positive_shaping_total == 1.5
    )


@pytest.mark.parametrize("name", ["max_actions_per_episode", "max_episodes"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 4097, 1.5, None])
def test_collection_budgets_are_bounded_strict_integers(name: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CorrelatedRewardGuard(_training(), _safety(), _POLICY, **{name: value})


def test_action_budget_requires_new_episode_without_reusing_old_identity() -> None:
    guard = CorrelatedRewardGuard(_training(), _safety(), _POLICY, max_actions_per_episode=1)
    guard.reset(_EPISODE)
    before, after = _interval()
    guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    before, after = _interval(1)
    with pytest.raises(ContractViolation, match="action budget"):
        guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    second_episode = UUID(int=0x104)
    guard.reset(second_episode)
    before, after = _interval(
        before_changes={"episode_id": second_episode},
        after_changes={"episode_id": second_episode},
    )
    assert (
        guard.compose(
            before, after, _signals(), [_claim(before, after)], action=_ACTION
        ).result.total
        == 1.5
    )
    with pytest.raises(ContractViolation, match="fresh episode"):
        guard.reset(_EPISODE)


def test_episode_budget_denial_keeps_existing_episode_and_reward_budget() -> None:
    guard = CorrelatedRewardGuard(_training(), _safety(), _POLICY, max_episodes=1)
    guard.reset(_EPISODE)
    with pytest.raises(ContractViolation, match="episode budget"):
        guard.reset(UUID(int=0x105))
    before, after = _interval()
    result = guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    assert result.result.positive_shaping_total == 1.5


@pytest.mark.parametrize("value", [None, "synthetic-episode", 12, True])
def test_reset_requires_uuid(value: object) -> None:
    with pytest.raises(TypeError):
        _guard().reset(value)


def test_signal_iterator_is_consumed_only_up_to_rejection_bound() -> None:
    count = 0

    def oversized_signals():
        nonlocal count
        while True:
            count += 1
            assert count <= 257, "guard must reject before consuming another item"
            yield _signals()[0]

    before, after = _interval()
    with pytest.raises(ValueError, match="too many"):
        _guard().compose(before, after, oversized_signals(), [], action=_ACTION)
    assert count == 257


def test_direct_compose_rejects_untyped_signal_or_attribution_before_state_commit() -> None:
    guard = _guard()
    before, after = _interval()
    with pytest.raises((TypeError, ValueError)):
        guard.compose(before, after, [None], [], action=_ACTION)
    with pytest.raises((TypeError, ValueError)):
        guard.compose(before, after, _signals(), [None], action=_ACTION)
    assert (
        guard.compose(
            before, after, _signals(), [_claim(before, after)], action=_ACTION
        ).result.total
        == 1.5
    )


def test_nonfinite_weighted_contribution_is_denied() -> None:
    before, after = _interval()
    with pytest.raises(ContractViolation, match="finite"):
        _guard(weight=1e308).compose(
            before, after, _signals(6), [_claim(before, after)], action=_ACTION
        )


def test_tensor_digest_binds_values_dtype_shape_and_names_in_stable_key_order() -> None:
    tree = {"left": np.asarray([3], dtype=np.int64), "right": np.asarray([7], dtype=np.int64)}
    digest = tensor_tree_sha256(tree)
    assert digest == tensor_tree_sha256(dict(reversed(list(tree.items()))))
    for alternate in [
        {"left": np.asarray([4], dtype=np.int64), "right": tree["right"]},
        {"left": np.asarray([3], dtype=np.int32), "right": tree["right"]},
        {"left": np.asarray([[3]], dtype=np.int64), "right": tree["right"]},
        {"other": tree["left"], "right": tree["right"]},
    ]:
        assert tensor_tree_sha256(alternate) != digest


@pytest.mark.parametrize(
    "tree",
    [
        {"invalid": np.asarray([object()], dtype=object)},
        {"invalid": np.asarray([(1,)], dtype=[("entry", np.int64)])},
        {"oversized": np.zeros((1 << 20) + 1, dtype=np.uint8)},
        {"invalid": 1},
        {1: np.asarray(1)},
        {"x" * 129: np.asarray(1)},
        {str(index): np.asarray(index) for index in range(257)},
    ],
)
def test_tensor_digest_rejects_unbounded_or_untyped_data(tree: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        tensor_tree_sha256(tree)


def test_invalid_action_digest_does_not_consume_reward_state() -> None:
    guard = _guard()
    before, after = _interval()
    with pytest.raises(ValueError, match="one MiB"):
        guard.compose(
            before,
            after,
            _signals(),
            [_claim(before, after)],
            action={"oversized": np.zeros((1 << 20) + 1, dtype=np.uint8)},
        )
    assert (
        guard.compose(
            before, after, _signals(), [_claim(before, after)], action=_ACTION
        ).result.total
        == 1.5
    )


def _learning_update(receipt, **changes: Any) -> ScalarLearningUpdate:
    return replace(
        ScalarLearningUpdate(
            learner_id="synthetic-learner",
            table_id="synthetic-table",
            policy_version=7,
            state_sha256=receipt.state_sha256,
            action_sha256=receipt.action_sha256,
            next_state_sha256=receipt.next_state_sha256,
            reward_receipt_sha256=receipt.sha256,
            previous_value=0.4,
            bootstrap_value=0.5,
            discount=0.6,
            learning_rate=0.25,
            target=1.8,
            updated_value=0.75,
        ),
        **changes,
    )


def _composed_receipt():
    before, after = _interval()
    return _guard().compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)


def test_scalar_learning_update_validates_reported_operands_without_mutating_any_table() -> None:
    receipt = _composed_receipt()
    update = _learning_update(receipt)
    update.validate_against(receipt)
    assert update.to_mapping()["table_id"] == "synthetic-table"
    assert update.to_mapping()["reward_receipt_sha256"] == receipt.sha256
    assert update.updated_value == 0.75


@pytest.mark.parametrize(
    "name", ["state_sha256", "action_sha256", "next_state_sha256", "reward_receipt_sha256"]
)
def test_scalar_update_rejects_another_transition_operand(name: str) -> None:
    receipt = _composed_receipt()
    update = _learning_update(receipt, **{name: sha256(b"different synthetic operand").hexdigest()})
    with pytest.raises(ContractViolation):
        update.validate_against(receipt)


@pytest.mark.parametrize("changes", [{"target": 1.9}, {"updated_value": 0.8}])
def test_scalar_update_rejects_wrong_reported_arithmetic(changes: dict[str, float]) -> None:
    receipt = _composed_receipt()
    with pytest.raises(ContractViolation, match="arithmetic"):
        _learning_update(receipt, **changes).validate_against(receipt)


@pytest.mark.parametrize(
    "name",
    ["previous_value", "bootstrap_value", "discount", "learning_rate", "target", "updated_value"],
)
@pytest.mark.parametrize("value", [True, None, np.nan, np.inf, -np.inf])
def test_scalar_update_requires_finite_numeric_operands(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        _learning_update(_composed_receipt(), **{name: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"discount": -0.01},
        {"discount": 1.01},
        {"learning_rate": 0},
        {"learning_rate": -0.01},
        {"learning_rate": 1.01},
        {"policy_version": True},
        {"policy_version": -1},
        {"learner_id": ""},
        {"table_id": "space in identifier"},
    ],
)
def test_scalar_update_rejects_unadmitted_bounds_and_identifiers(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _learning_update(_composed_receipt(), **changes)


def test_terminal_scalar_update_requires_zero_bootstrap() -> None:
    before, after = _interval(done=True)
    receipt = _guard().compose(
        before,
        after,
        _signals(0, completion=0.2),
        [_claim(before, after, signal_name="completion")],
        action=_ACTION,
    )
    with pytest.raises(ContractViolation, match="bootstrap"):
        _learning_update(receipt).validate_against(receipt)
    update = _learning_update(receipt, bootstrap_value=0, target=1.8, updated_value=0.75)
    update.validate_against(receipt)


@pytest.mark.parametrize("with_claim", [False, True])
def test_rejected_action_cannot_bypass_correlation_with_positive_terminal_outcome(
    with_claim: bool,
) -> None:
    before, after = _interval(outcome=ActionOutcome.REJECTED, done=True)
    claims = [_claim(before, after, signal_name="completion")] if with_claim else []
    with pytest.raises(ContractViolation, match="confirmed accepted effect"):
        _guard().compose(before, after, _signals(0, completion=0.5), claims, action=_ACTION)


def test_accepted_terminal_outcome_requires_its_own_exact_confirmed_claim() -> None:
    guard = _guard()
    before, after = _interval(done=True)
    with pytest.raises(ContractViolation, match="confirmed accepted effect"):
        guard.compose(before, after, _signals(0, completion=0.5), [], action=_ACTION)
    result = guard.compose(
        before,
        after,
        _signals(0, completion=0.5),
        [_claim(before, after, signal_name="completion")],
        action=_ACTION,
    )
    assert result.result.total == 4.5
    assert result.result.terminal


@pytest.mark.parametrize("outcome", [ActionOutcome.ACCEPTED, ActionOutcome.REJECTED])
@pytest.mark.parametrize("with_claim", [False, True])
def test_positive_unbudgeted_signal_keeps_exact_action_effect_requirement(
    outcome: ActionOutcome, with_claim: bool
) -> None:
    training = _training()
    bonus = RewardTermSpec("bonus", "adapter", minimum=0, maximum=2, required=False)
    training = replace(
        training, reward=replace(training.reward, terms=(*training.reward.terms, bonus))
    )
    guard = CorrelatedRewardGuard(
        training, replace(_safety(), unbudgeted_signals=("bonus",)), _POLICY
    )
    guard.reset(_EPISODE)
    before, after = _interval(outcome=outcome)
    signals = (*_signals(0), RewardSignal("bonus", "adapter", 1))
    claims = [_claim(before, after, signal_name="bonus")] if with_claim else []
    if outcome is ActionOutcome.ACCEPTED and with_claim:
        result = guard.compose(before, after, signals, claims, action=_ACTION)
        assert result.result.total == 1
        assert result.result.positive_shaping_total == 0
    else:
        with pytest.raises(ContractViolation, match="confirmed accepted effect"):
            guard.compose(before, after, signals, claims, action=_ACTION)


def test_tensor_digest_rejects_excessive_nesting() -> None:
    tree = {"entry": np.asarray(1)}
    for _ in range(18):
        tree = {"nested": tree}
    with pytest.raises(ValueError, match="structural bounds"):
        tensor_tree_sha256(tree)


@pytest.mark.parametrize("value", [None, "", "contains space", "x" * 129, "control\n"])
def test_correlation_policy_requires_bounded_identifiers(value: object) -> None:
    with pytest.raises(ValueError):
        replace(_POLICY, run_id=value)


@pytest.mark.parametrize("value", [None, "abc", "A" * 64, "z" * 64])
def test_owner_configuration_requires_lowercase_sha256(value: object) -> None:
    with pytest.raises(ValueError):
        replace(_POLICY, environment_config_sha256=value)
    with pytest.raises(ValueError):
        replace(_context(0), environment_config_sha256=value)


@pytest.mark.parametrize("value", [None, "gameplay", 1])
def test_context_requires_typed_phase(value: object) -> None:
    with pytest.raises(TypeError):
        replace(_context(0), phase=value)


def test_context_requires_typed_episode_identity() -> None:
    with pytest.raises(TypeError):
        replace(_context(0), episode_id=str(_EPISODE))


def test_attribution_requires_typed_effect_and_exact_mapping_fields() -> None:
    before, after = _interval()
    claim = _claim(before, after)
    with pytest.raises(TypeError):
        replace(claim, effect="confirmed")
    for mapping in [None, {}, {**claim.to_mapping(), "unexpected": True}]:
        with pytest.raises(ValueError):
            RewardAttribution.from_mapping(mapping)


@pytest.mark.parametrize("value", [None, "", " ", "x" * 129])
def test_learning_consumer_requires_bounded_table_identifier(value: object) -> None:
    with pytest.raises(ValueError):
        LearningConsumerPolicy("synthetic-learner", value, 2)


@pytest.mark.parametrize("value", [True, -1, 1.5, None])
def test_learning_consumer_requires_strict_policy_version(value: object) -> None:
    with pytest.raises(ValueError):
        LearningConsumerPolicy("synthetic-learner", "synthetic-table", value)


def test_learning_consumer_denies_untyped_update() -> None:
    with pytest.raises(TypeError):
        LearningConsumerPolicy("synthetic-learner", "synthetic-table", 2).validate(None)


@pytest.mark.parametrize(
    "context_changes",
    [{"producer_sequence": 42}, {"timestamp_ns": 1_050}, {"producer_sequence": 47}],
)
def test_next_interval_requires_exact_previous_post_context(context_changes) -> None:
    guard = _guard()
    before, after = _interval()
    guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    wrong_before, wrong_after = _interval(1, before_changes=context_changes)
    with pytest.raises(ContractViolation):
        guard.compose(
            wrong_before,
            wrong_after,
            _signals(),
            [_claim(wrong_before, wrong_after)],
            action=_ACTION,
        )
    before, after = _interval(1)
    assert (
        guard.compose(
            before, after, _signals(), [_claim(before, after)], action=_ACTION
        ).result.total
        == 1.5
    )


def test_next_interval_cannot_substitute_observation_at_same_context() -> None:
    guard = _guard()
    before, after = _interval()
    guard.compose(before, after, _signals(), [_claim(before, after)], action=_ACTION)
    before, after = _interval(1)
    wrong_before = replace(before, observation={"sensor": np.asarray([6], dtype=np.float32)})
    with pytest.raises(ContractViolation):
        guard.compose(wrong_before, after, _signals(), [_claim(before, after)], action=_ACTION)
    assert (
        guard.compose(
            before, after, _signals(), [_claim(before, after)], action=_ACTION
        ).result.total
        == 1.5
    )


def test_observed_reward_must_equal_quantized_composition_without_close_tolerance() -> None:
    guard = _guard()
    before, after = _interval(reward=np.nextafter(np.float32(1.5), np.float32(2)))
    with pytest.raises(ContractViolation, match="observed reward"):
        guard.compose_timestep(before, _with_evidence(before, after), action=_ACTION)
    after = replace(after, reward=np.asarray([1.5], dtype=np.float32))
    receipt = guard.compose_timestep(before, _with_evidence(before, after), action=_ACTION)
    assert receipt.learning_reward == 1.5


def test_scalar_update_uses_actual_quantized_learner_reward() -> None:
    before, after = _interval(reward=1.3)
    receipt = _guard().compose_timestep(
        before, _with_evidence(before, after, value=1.3), action=_ACTION
    )
    actual_reward = float(np.float32(1.3))
    assert receipt.result.total == 1.3
    assert receipt.observed_reward == actual_reward
    assert receipt.learning_reward == actual_reward
    exact_update = _learning_update(
        receipt,
        previous_value=0,
        bootstrap_value=0,
        learning_rate=1,
        target=actual_reward,
        updated_value=actual_reward,
    )
    exact_update.validate_against(receipt)
    with pytest.raises(ContractViolation, match="arithmetic"):
        replace(exact_update, target=1.3, updated_value=1.3).validate_against(receipt)


def test_strict_guard_rejects_policy_subclass_override() -> None:
    class OverriddenPolicy(CorrelationPolicy):
        def validate_interval(self, before, after):
            raise AssertionError("virtual validation must never be dispatched")

    policy = OverriddenPolicy(
        **{name: getattr(_POLICY, name) for name in CorrelationPolicy.__dataclass_fields__}
    )
    with pytest.raises(TypeError):
        CorrelatedRewardGuard(_training(), _safety(), policy)


def test_strict_guard_rejects_action_receipt_subclass_override() -> None:
    class OverriddenReceipt(ActionReceipt):
        def validate_against(self, timestep):
            raise AssertionError("virtual validation must never be dispatched")

    before, after = _interval()
    receipt = after.action_receipt
    assert receipt is not None
    override = OverriddenReceipt(
        **{name: getattr(receipt, name) for name in ActionReceipt.__dataclass_fields__}
    )
    with pytest.raises(ContractViolation, match="action receipt"):
        _guard().compose(
            before, replace(after, action_receipt=override), _signals(), [], action=_ACTION
        )


def test_strict_guard_rejects_reward_signal_and_attribution_subclasses() -> None:
    class OverriddenSignal(RewardSignal):
        pass

    class OverriddenAttribution(RewardAttribution):
        pass

    before, after = _interval()
    claim = _claim(before, after)
    override = OverriddenAttribution(
        **{name: getattr(claim, name) for name in RewardAttribution.__dataclass_fields__}
    )
    with pytest.raises(TypeError):
        _guard().compose(
            before, after, [OverriddenSignal("actuator_gain", "adapter", 1.5)], [], action=_ACTION
        )
    with pytest.raises((TypeError, ValueError)):
        _guard().compose(before, after, _signals(), [override], action=_ACTION)


def test_invalid_action_identity_cannot_consume_negative_budget_or_step() -> None:
    guard = _guard()
    before, after = _interval(action_id=" ")
    with pytest.raises(ValueError):
        guard.compose(before, after, _signals(-6), [], action=_ACTION)
    assert after.action_receipt is not None
    corrected = replace(
        after, action_receipt=replace(after.action_receipt, action_id="valid-action")
    )
    accepted = guard.compose(before, corrected, _signals(-6), [], action=_ACTION)
    next_before, next_after = _interval(1)
    following = guard.compose(next_before, next_after, _signals(-6), [], action=_ACTION)
    assert accepted.result.total == -4
    assert following.result.total == -4
    assert following.result.negative_shaping_total == 8


def test_timestep_subclass_cannot_override_terminal_state_validation() -> None:
    class OverriddenTimeStep(TimeStep):
        @property
        def done(self):
            raise AssertionError("virtual lifecycle must never be dispatched")

    before, after = _interval()
    override = OverriddenTimeStep(
        **{name: getattr(before, name) for name in TimeStep.__dataclass_fields__}
    )
    with pytest.raises((TypeError, ContractViolation)):
        _guard().compose(override, after, _signals(), [], action=_ACTION)
