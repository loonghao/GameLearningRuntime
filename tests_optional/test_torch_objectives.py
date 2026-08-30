from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch

from game_learning_runtime.integrations.torch_objectives import (  # noqa: E402
    behavior_cloning_loss,
    generalized_advantage_estimate,
    impala_loss,
    masked_logits,
    ppo_loss,
    vtrace_targets,
)


def test_masked_logits_preserves_valid_actions_and_rejects_invalid_masks() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    action_mask = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)

    result = masked_logits(logits, action_mask)

    assert torch.equal(result[action_mask], logits[action_mask])
    assert torch.equal(
        result[~action_mask],
        torch.full_like(result[~action_mask], torch.finfo(logits.dtype).min),
    )

    with pytest.raises(TypeError, match="bool"):
        masked_logits(logits, action_mask.to(torch.int64))
    with pytest.raises(ValueError, match="same shape"):
        masked_logits(logits, action_mask[:, :2])
    with pytest.raises(ValueError, match="at least one valid action"):
        masked_logits(logits, torch.zeros_like(action_mask))


def test_objectives_reject_non_finite_float_inputs() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        masked_logits(torch.tensor([[float("nan"), 0.0]]))

    with pytest.raises(ValueError, match="non-finite"):
        vtrace_targets(
            rewards=torch.tensor([[1.0]]),
            values=torch.tensor([[0.0], [0.0]]),
            behavior_log_prob=torch.tensor([[float("inf")]]),
            target_log_prob=torch.tensor([[0.0]]),
            terminated=torch.tensor([[True]]),
        )


def test_objectives_reject_non_finite_hyperparameters() -> None:
    with pytest.raises(ValueError, match="entropy_coefficient"):
        behavior_cloning_loss(
            torch.zeros((1, 2)),
            torch.tensor([0]),
            entropy_coefficient=float("nan"),
        )

    with pytest.raises(ValueError, match="rho_clip"):
        vtrace_targets(
            rewards=torch.tensor([[1.0]]),
            values=torch.tensor([[0.0], [0.0]]),
            behavior_log_prob=torch.tensor([[0.0]]),
            target_log_prob=torch.tensor([[0.0]]),
            terminated=torch.tensor([[True]]),
            rho_clip=float("nan"),
        )


def test_behavior_cloning_loss_reports_stable_components() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    actions = torch.tensor([0, 1])

    result = behavior_cloning_loss(logits, actions)

    assert result.loss.item() == pytest.approx(math.log(2.0))
    assert result.negative_log_likelihood.item() == pytest.approx(math.log(2.0))
    assert result.entropy.item() == pytest.approx(math.log(2.0))
    assert result.accuracy.item() == pytest.approx(0.5)


def test_behavior_cloning_loss_rejects_an_action_excluded_by_its_mask() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    actions = torch.tensor([0, 1])
    action_mask = torch.tensor([[True, False], [True, False]])

    with pytest.raises(ValueError, match="selected action"):
        behavior_cloning_loss(logits, actions, action_mask=action_mask)


def test_behavior_cloning_label_smoothing_uses_only_valid_actions() -> None:
    result = behavior_cloning_loss(
        torch.zeros((1, 3), dtype=torch.float32),
        torch.tensor([0]),
        action_mask=torch.tensor([[True, True, False]]),
        label_smoothing=1.0,
    )

    assert result.loss.item() == pytest.approx(math.log(2.0))
    assert torch.isfinite(result.loss)


def test_gae_bootstraps_truncation_but_stops_cross_episode_recursion() -> None:
    result = generalized_advantage_estimate(
        rewards=torch.tensor([[1.0], [1.0], [1.0]]),
        values=torch.tensor([[0.0], [0.0], [5.0], [7.0]]),
        terminated=torch.tensor([[False], [False], [True]]),
        truncated=torch.tensor([[False], [True], [False]]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(result.advantages, torch.tensor([[7.0], [6.0], [-4.0]]))
    torch.testing.assert_close(result.value_targets, torch.tensor([[7.0], [6.0], [1.0]]))


def test_rollout_targets_reject_conflicting_episode_boundaries() -> None:
    with pytest.raises(ValueError, match="both terminated and truncated"):
        generalized_advantage_estimate(
            rewards=torch.tensor([[1.0]]),
            values=torch.tensor([[0.0], [0.0]]),
            terminated=torch.tensor([[True]]),
            truncated=torch.tensor([[True]]),
        )


def test_ppo_loss_uses_masked_policy_and_explicit_components() -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    actions = torch.tensor([0, 1])
    action_mask = torch.tensor([[True, False], [True, True]])
    old_log_prob = torch.tensor([0.0, -math.log(2.0)])
    values = torch.zeros(2, requires_grad=True)

    result = ppo_loss(
        policy_logits=logits,
        actions=actions,
        old_log_prob=old_log_prob,
        advantages=torch.tensor([2.0, 1.0]),
        values=values,
        value_targets=torch.tensor([1.0, -1.0]),
        action_mask=action_mask,
        normalize_advantage=False,
        value_coefficient=0.5,
        entropy_coefficient=0.0,
    )

    assert result.policy_loss.item() == pytest.approx(-1.5)
    assert result.value_loss.item() == pytest.approx(0.5)
    assert result.loss.item() == pytest.approx(-1.25)
    assert result.approximate_kl.item() == pytest.approx(0.0)
    assert result.clip_fraction.item() == pytest.approx(0.0)
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(values.grad).all()


def test_ppo_metrics_remain_finite_for_extreme_log_probability_drift() -> None:
    result = ppo_loss(
        policy_logits=torch.zeros((1, 2)),
        actions=torch.tensor([0]),
        old_log_prob=torch.tensor([-1_000.0]),
        advantages=torch.tensor([1.0]),
        values=torch.tensor([0.0]),
        value_targets=torch.tensor([0.0]),
        normalize_advantage=False,
    )

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.approximate_kl)


def test_ppo_optionally_clips_value_updates_against_old_predictions() -> None:
    common = {
        "policy_logits": torch.zeros((1, 2)),
        "actions": torch.tensor([0]),
        "old_log_prob": torch.tensor([-math.log(2.0)]),
        "advantages": torch.tensor([0.0]),
        "values": torch.tensor([1.0]),
        "value_targets": torch.tensor([1.0]),
        "normalize_advantage": False,
        "value_coefficient": 1.0,
        "entropy_coefficient": 0.0,
    }

    unclipped = ppo_loss(**common)
    clipped = ppo_loss(
        **common,
        old_values=torch.tensor([0.0]),
        value_clip_epsilon=0.2,
    )

    assert unclipped.value_loss.item() == pytest.approx(0.0)
    assert clipped.value_loss.item() == pytest.approx(0.32)
    with pytest.raises(ValueError, match="provided together"):
        ppo_loss(**common, old_values=torch.tensor([0.0]))


def test_vtrace_matches_on_policy_returns() -> None:
    result = vtrace_targets(
        rewards=torch.tensor([[1.0], [2.0]]),
        values=torch.tensor([[0.0], [0.0], [0.0]]),
        behavior_log_prob=torch.zeros((2, 1)),
        target_log_prob=torch.zeros((2, 1)),
        terminated=torch.tensor([[False], [True]]),
        gamma=1.0,
    )

    torch.testing.assert_close(result.value_targets, torch.tensor([[3.0], [2.0]]))
    torch.testing.assert_close(result.policy_advantages, torch.tensor([[3.0], [2.0]]))
    torch.testing.assert_close(result.importance_weights, torch.ones((2, 1)))


def test_vtrace_reports_but_clips_large_importance_weights() -> None:
    result = vtrace_targets(
        rewards=torch.tensor([[1.0]]),
        values=torch.tensor([[0.0], [0.0]]),
        behavior_log_prob=torch.tensor([[-math.log(10.0)]]),
        target_log_prob=torch.zeros((1, 1)),
        terminated=torch.tensor([[True]]),
        rho_clip=1.0,
    )

    assert result.importance_weights.item() == pytest.approx(10.0)
    assert result.value_targets.item() == pytest.approx(1.0)


def test_vtrace_importance_weights_are_finite_for_float16() -> None:
    result = vtrace_targets(
        rewards=torch.tensor([[1.0]], dtype=torch.float16),
        values=torch.tensor([[0.0], [0.0]], dtype=torch.float16),
        behavior_log_prob=torch.tensor([[0.0]], dtype=torch.float16),
        target_log_prob=torch.tensor([[100.0]], dtype=torch.float16),
        terminated=torch.tensor([[True]]),
    )

    assert torch.isfinite(result.importance_weights).all()


def test_vtrace_bootstraps_truncation_without_crossing_episode_boundary() -> None:
    result = vtrace_targets(
        rewards=torch.tensor([[1.0], [1.0]]),
        values=torch.tensor([[0.0], [5.0], [7.0]]),
        behavior_log_prob=torch.zeros((2, 1)),
        target_log_prob=torch.zeros((2, 1)),
        terminated=torch.tensor([[False], [True]]),
        truncated=torch.tensor([[True], [False]]),
        gamma=1.0,
    )

    torch.testing.assert_close(result.value_targets, torch.tensor([[6.0], [1.0]]))
    torch.testing.assert_close(result.policy_advantages, torch.tensor([[6.0], [-4.0]]))


def test_impala_loss_is_differentiable_but_keeps_vtrace_targets_detached() -> None:
    logits = torch.zeros((2, 1, 2), dtype=torch.float32, requires_grad=True)
    values = torch.zeros((3, 1), dtype=torch.float32, requires_grad=True)

    result = impala_loss(
        policy_logits=logits,
        actions=torch.tensor([[0], [1]]),
        behavior_log_prob=torch.full((2, 1), -math.log(2.0)),
        rewards=torch.tensor([[1.0], [2.0]]),
        values=values,
        terminated=torch.tensor([[False], [True]]),
        gamma=1.0,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    )

    assert result.value_targets.requires_grad is False
    assert result.policy_advantages.requires_grad is False
    assert result.loss.item() == pytest.approx(
        math.log(2.0) * 2.5 + 0.5 * 3.25 - 0.01 * math.log(2.0)
    )
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(values.grad).all()
