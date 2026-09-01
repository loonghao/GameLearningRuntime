"""Reusable PyTorch objectives for custom learning loops.

This module deliberately contains no model, optimizer, collector, or game-specific
semantics.  It accepts time-major tensors produced by any GLR adapter and returns
typed objective components that custom Torch or TorchRL learners can compose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from game_learning_runtime.errors import OptionalDependencyError

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - exercised without the optional extra
    raise OptionalDependencyError(
        "PyTorch objectives require `uv add game-learning-runtime[torch]`"
    ) from error


@dataclass(frozen=True, slots=True)
class BehaviorCloningLoss:
    """Behavior-cloning objective components and detached accuracy metric."""

    loss: Tensor
    negative_log_likelihood: Tensor
    entropy: Tensor
    accuracy: Tensor


@dataclass(frozen=True, slots=True)
class AdvantageTargets:
    """Detached generalized advantages and value-regression targets."""

    advantages: Tensor
    value_targets: Tensor


@dataclass(frozen=True, slots=True)
class PPOLoss:
    """Clipped PPO objective components."""

    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    forced_action_ratio: Tensor
    mean_valid_actions: Tensor


@dataclass(frozen=True, slots=True)
class VTraceTargets:
    """Detached V-trace value targets and policy-gradient advantages."""

    value_targets: Tensor
    policy_advantages: Tensor
    importance_weights: Tensor


@dataclass(frozen=True, slots=True)
class IMPALALoss:
    """IMPALA actor-critic objective components and detached V-trace targets."""

    loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    value_targets: Tensor
    policy_advantages: Tensor


def _require_floating(name: str, value: Tensor) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite values")


def _require_probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_rollout(
    *,
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor | None,
) -> Tensor:
    _require_floating("rewards", rewards)
    _require_floating("values", values)
    if rewards.ndim == 0 or rewards.shape[0] == 0:
        raise ValueError("rewards must have a non-empty leading time dimension")
    if values.shape != (rewards.shape[0] + 1, *rewards.shape[1:]):
        raise ValueError("values must have shape [time + 1, ...] matching rewards")
    if values.device != rewards.device or values.dtype != rewards.dtype:
        raise ValueError("values and rewards must have the same dtype and device")
    if terminated.dtype is not torch.bool:
        raise TypeError("terminated must use dtype torch.bool")
    if terminated.shape != rewards.shape:
        raise ValueError("terminated must have the same shape as rewards")
    if terminated.device != rewards.device:
        raise ValueError("terminated must be on the same device as rewards")
    if truncated is None:
        return torch.zeros_like(terminated)
    if truncated.dtype is not torch.bool:
        raise TypeError("truncated must use dtype torch.bool")
    if truncated.shape != rewards.shape:
        raise ValueError("truncated must have the same shape as rewards")
    if truncated.device != rewards.device:
        raise ValueError("truncated must be on the same device as rewards")
    if torch.logical_and(terminated, truncated).any().item():
        raise ValueError("a transition cannot be both terminated and truncated")
    return truncated


def _validate_actions(logits: Tensor, actions: Tensor, action_mask: Tensor | None) -> None:
    if actions.shape != logits.shape[:-1]:
        raise ValueError("actions must match policy_logits without the action dimension")
    if actions.dtype is torch.bool or torch.is_floating_point(actions):
        raise TypeError("actions must use an integer dtype")
    if actions.device != logits.device:
        raise ValueError("actions and policy_logits must be on the same device")
    if ((actions < 0) | (actions >= logits.shape[-1])).any().item():
        raise ValueError("actions contain an index outside the policy action dimension")
    if action_mask is not None:
        selected_is_valid = action_mask.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        if not selected_is_valid.all().item():
            raise ValueError("selected action is excluded by action_mask")


def masked_logits(policy_logits: Tensor, action_mask: Tensor | None = None) -> Tensor:
    """Replace invalid discrete-action logits with the dtype's finite minimum.

    Masks are intentionally strict: they must exactly match the logits and every
    policy row must retain at least one valid action.
    """

    _require_floating("policy_logits", policy_logits)
    if policy_logits.ndim == 0 or policy_logits.shape[-1] == 0:
        raise ValueError("policy_logits must have a non-empty action dimension")
    if action_mask is None:
        return policy_logits
    if action_mask.dtype is not torch.bool:
        raise TypeError("action_mask must use dtype torch.bool")
    if action_mask.shape != policy_logits.shape:
        raise ValueError("action_mask must have the same shape as policy_logits")
    if action_mask.device != policy_logits.device:
        raise ValueError("action_mask and policy_logits must be on the same device")
    if not action_mask.any(dim=-1).all().item():
        raise ValueError("each policy row must contain at least one valid action")
    invalid_value = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(~action_mask, invalid_value)


def _policy_log_probabilities(
    policy_logits: Tensor, actions: Tensor, action_mask: Tensor | None
) -> Tensor:
    logits = masked_logits(policy_logits, action_mask)
    _validate_actions(logits, actions, action_mask)
    return torch.log_softmax(logits, dim=-1)


def _mean_entropy(log_probabilities: Tensor) -> Tensor:
    return -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()


def behavior_cloning_loss(
    policy_logits: Tensor,
    actions: Tensor,
    *,
    action_mask: Tensor | None = None,
    sample_weight: Tensor | None = None,
    label_smoothing: float = 0.0,
    entropy_coefficient: float = 0.0,
) -> BehaviorCloningLoss:
    """Compute a masked discrete behavior-cloning objective.

    ``sample_weight`` accepts the audited weight emitted by
    :class:`~game_learning_runtime.training_safety.DemonstrationGate` for each
    sample. The weights are normalized by their sum, so their absolute scale
    does not change the optimizer step.
    """

    _require_probability("label_smoothing", label_smoothing)
    _require_non_negative("entropy_coefficient", entropy_coefficient)
    log_probabilities = _policy_log_probabilities(policy_logits, actions, action_mask)
    if sample_weight is not None:
        if sample_weight.shape != actions.shape:
            raise ValueError("sample_weight must have the same batch shape as actions")
        _require_floating("sample_weight", sample_weight)
        if sample_weight.device != policy_logits.device:
            raise ValueError("sample_weight and policy_logits must be on the same device")
        if not torch.isfinite(sample_weight).all().item():
            raise ValueError("sample_weight must contain only finite values")
        if (sample_weight < 0).any().item():
            raise ValueError("sample_weight must be non-negative")
        if not (sample_weight.sum() > 0).item():
            raise ValueError("sample_weight must have a positive total")

    def mean(value: Tensor) -> Tensor:
        if sample_weight is None:
            return value.mean()
        weight = sample_weight.to(dtype=value.dtype)
        return (value * weight).sum() / weight.sum()

    selected_log_probability = log_probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    per_sample_nll = -selected_log_probability
    if label_smoothing:
        valid = action_mask if action_mask is not None else torch.ones_like(policy_logits).bool()
        valid_count = valid.sum(dim=-1)
        smooth_nll = -(log_probabilities.masked_fill(~valid, 0.0).sum(dim=-1) / valid_count)
        per_sample_nll = (1.0 - label_smoothing) * per_sample_nll + (label_smoothing * smooth_nll)
    negative_log_likelihood = mean(per_sample_nll)
    per_sample_entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
    entropy = mean(per_sample_entropy)
    loss = negative_log_likelihood - entropy_coefficient * entropy
    accuracy = mean((log_probabilities.argmax(dim=-1) == actions).to(policy_logits.dtype))
    return BehaviorCloningLoss(
        loss=loss,
        negative_log_likelihood=negative_log_likelihood,
        entropy=entropy,
        accuracy=accuracy.detach(),
    )


def generalized_advantage_estimate(
    *,
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> AdvantageTargets:
    """Compute detached time-major GAE with correct truncation bootstrapping.

    Termination disables bootstrapping. Truncation keeps the next-state bootstrap
    value but stops the recursion so advantages never leak into the next episode.
    """

    _require_probability("gamma", gamma)
    _require_probability("gae_lambda", gae_lambda)
    truncated = _validate_rollout(
        rewards=rewards,
        values=values,
        terminated=terminated,
        truncated=truncated,
    )
    with torch.no_grad():
        terminal = terminated.to(rewards.dtype)
        boundary = torch.logical_or(terminated, truncated).to(rewards.dtype)
        bootstrap_discount = gamma * (1.0 - terminal)
        trace_discount = gamma * gae_lambda * (1.0 - boundary)
        deltas = rewards + bootstrap_discount * values[1:] - values[:-1]
        advantages = torch.empty_like(rewards)
        accumulator = torch.zeros_like(rewards[-1])
        for index in range(rewards.shape[0] - 1, -1, -1):
            accumulator = deltas[index] + trace_discount[index] * accumulator
            advantages[index] = accumulator
        value_targets = advantages + values[:-1]
    return AdvantageTargets(advantages=advantages, value_targets=value_targets)


def ppo_loss(
    *,
    policy_logits: Tensor,
    actions: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    values: Tensor,
    value_targets: Tensor,
    old_values: Tensor | None = None,
    action_mask: Tensor | None = None,
    clip_epsilon: float = 0.2,
    value_clip_epsilon: float | None = None,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    normalize_advantage: bool = True,
) -> PPOLoss:
    """Compute the clipped discrete-action PPO objective."""

    _require_positive("clip_epsilon", clip_epsilon)
    if (old_values is None) != (value_clip_epsilon is None):
        raise ValueError("old_values and value_clip_epsilon must be provided together")
    if value_clip_epsilon is not None:
        _require_positive("value_clip_epsilon", value_clip_epsilon)
    _require_non_negative("value_coefficient", value_coefficient)
    _require_non_negative("entropy_coefficient", entropy_coefficient)
    _require_floating("old_log_prob", old_log_prob)
    _require_floating("advantages", advantages)
    _require_floating("values", values)
    _require_floating("value_targets", value_targets)
    if old_values is not None:
        _require_floating("old_values", old_values)
    log_probabilities = _policy_log_probabilities(policy_logits, actions, action_mask)
    expected_shape = policy_logits.shape[:-1]
    for name, value in (
        ("old_log_prob", old_log_prob),
        ("advantages", advantages),
        ("values", values),
        ("value_targets", value_targets),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must match the policy batch shape")
        if value.device != policy_logits.device:
            raise ValueError(f"{name} and policy_logits must be on the same device")
    if old_values is not None:
        if old_values.shape != expected_shape:
            raise ValueError("old_values must match the policy batch shape")
        if old_values.device != policy_logits.device:
            raise ValueError("old_values and policy_logits must be on the same device")
    prepared_advantages = advantages.detach()
    if normalize_advantage and prepared_advantages.numel() > 1:
        prepared_advantages = (prepared_advantages - prepared_advantages.mean()) / (
            prepared_advantages.std(unbiased=False) + 1e-8
        )
    new_log_prob = log_probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    maximum_log_ratio = min(20.0, math.log(torch.finfo(new_log_prob.dtype).max) - 2.0)
    log_ratio = (new_log_prob - old_log_prob.detach()).clamp(
        min=-maximum_log_ratio, max=maximum_log_ratio
    )
    ratio = log_ratio.exp()
    unclipped = ratio * prepared_advantages
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * prepared_advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_error = torch.square(value_targets.detach() - values)
    if old_values is not None and value_clip_epsilon is not None:
        clipped_values = old_values.detach() + (values - old_values.detach()).clamp(
            min=-value_clip_epsilon, max=value_clip_epsilon
        )
        clipped_value_error = torch.square(value_targets.detach() - clipped_values)
        value_error = torch.maximum(value_error, clipped_value_error)
    value_loss = 0.5 * value_error.mean()
    entropy = _mean_entropy(log_probabilities)
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    approximate_kl = ((ratio - 1.0) - log_ratio).mean().detach()
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).to(policy_logits.dtype).mean().detach()
    if action_mask is None:
        valid_action_counts = torch.full(
            expected_shape,
            policy_logits.shape[-1],
            dtype=policy_logits.dtype,
            device=policy_logits.device,
        )
    else:
        valid_action_counts = action_mask.sum(dim=-1).to(policy_logits.dtype)
    forced_action_ratio = (valid_action_counts < 2).to(policy_logits.dtype).mean().detach()
    mean_valid_actions = valid_action_counts.mean().detach()
    return PPOLoss(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        forced_action_ratio=forced_action_ratio,
        mean_valid_actions=mean_valid_actions,
    )


def vtrace_targets(
    *,
    rewards: Tensor,
    values: Tensor,
    behavior_log_prob: Tensor,
    target_log_prob: Tensor,
    terminated: Tensor,
    truncated: Tensor | None = None,
    gamma: float = 0.99,
    rho_clip: float = 1.0,
    trace_clip: float = 1.0,
    policy_gradient_rho_clip: float = 1.0,
) -> VTraceTargets:
    """Compute detached V-trace targets for a time-major actor trajectory."""

    _require_probability("gamma", gamma)
    for name, clip_limit in (
        ("rho_clip", rho_clip),
        ("trace_clip", trace_clip),
        ("policy_gradient_rho_clip", policy_gradient_rho_clip),
    ):
        _require_positive(name, clip_limit)
    truncated = _validate_rollout(
        rewards=rewards,
        values=values,
        terminated=terminated,
        truncated=truncated,
    )
    for name, log_probability in (
        ("behavior_log_prob", behavior_log_prob),
        ("target_log_prob", target_log_prob),
    ):
        _require_floating(name, log_probability)
        if log_probability.shape != rewards.shape:
            raise ValueError(f"{name} must have the same shape as rewards")
        if log_probability.device != rewards.device:
            raise ValueError(f"{name} and rewards must be on the same device")
    with torch.no_grad():
        log_importance = target_log_prob - behavior_log_prob
        maximum_log_importance = math.log(torch.finfo(log_importance.dtype).max) - 2.0
        importance_weights = torch.exp(log_importance.clamp(max=maximum_log_importance))
        clipped_rho = importance_weights.clamp(max=rho_clip)
        clipped_trace = importance_weights.clamp(max=trace_clip)
        clipped_policy_rho = importance_weights.clamp(max=policy_gradient_rho_clip)
        terminal = terminated.to(rewards.dtype)
        boundary = torch.logical_or(terminated, truncated)
        bootstrap_discount = gamma * (1.0 - terminal)
        trace_discount = gamma * (~boundary).to(rewards.dtype)
        deltas = clipped_rho * (rewards + bootstrap_discount * values[1:] - values[:-1])
        corrections = torch.empty_like(rewards)
        accumulator = torch.zeros_like(rewards[-1])
        for index in range(rewards.shape[0] - 1, -1, -1):
            accumulator = deltas[index] + trace_discount[index] * clipped_trace[index] * accumulator
            corrections[index] = accumulator
        value_targets = values[:-1] + corrections
        corrected_next = torch.cat((value_targets[1:], values[-1:].detach()), dim=0)
        policy_next = torch.where(boundary, values[1:], corrected_next)
        policy_advantages = clipped_policy_rho * (
            rewards + bootstrap_discount * policy_next - values[:-1]
        )
    return VTraceTargets(
        value_targets=value_targets,
        policy_advantages=policy_advantages,
        importance_weights=importance_weights,
    )


def impala_loss(
    *,
    policy_logits: Tensor,
    actions: Tensor,
    behavior_log_prob: Tensor,
    rewards: Tensor,
    values: Tensor,
    terminated: Tensor,
    truncated: Tensor | None = None,
    action_mask: Tensor | None = None,
    gamma: float = 0.99,
    rho_clip: float = 1.0,
    trace_clip: float = 1.0,
    policy_gradient_rho_clip: float = 1.0,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> IMPALALoss:
    """Compute a discrete IMPALA loss from behavior-policy trajectories."""

    _require_non_negative("value_coefficient", value_coefficient)
    _require_non_negative("entropy_coefficient", entropy_coefficient)
    if policy_logits.shape[:-1] != rewards.shape:
        raise ValueError("policy_logits must have shape [time, ..., action]")
    log_probabilities = _policy_log_probabilities(policy_logits, actions, action_mask)
    target_log_prob = log_probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    targets = vtrace_targets(
        rewards=rewards,
        values=values,
        behavior_log_prob=behavior_log_prob,
        target_log_prob=target_log_prob,
        terminated=terminated,
        truncated=truncated,
        gamma=gamma,
        rho_clip=rho_clip,
        trace_clip=trace_clip,
        policy_gradient_rho_clip=policy_gradient_rho_clip,
    )
    policy_loss = -(target_log_prob * targets.policy_advantages).mean()
    value_loss = 0.5 * torch.square(targets.value_targets - values[:-1]).mean()
    entropy = _mean_entropy(log_probabilities)
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    return IMPALALoss(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        value_targets=targets.value_targets,
        policy_advantages=targets.policy_advantages,
    )


__all__ = [
    "AdvantageTargets",
    "BehaviorCloningLoss",
    "IMPALALoss",
    "PPOLoss",
    "VTraceTargets",
    "behavior_cloning_loss",
    "generalized_advantage_estimate",
    "impala_loss",
    "masked_logits",
    "ppo_loss",
    "vtrace_targets",
]
