# Compose custom Torch objectives

The optional objective module removes duplicated algorithm mathematics while
leaving each project in control of its model, optimizer, collector, reward, and
runtime adapter.

## Install

For a custom PyTorch learner:

```powershell
uv add "game-learning-runtime[torch]"
```

For the same objectives plus the `TorchRLEnvironment` adapter:

```powershell
uv add "game-learning-runtime[torchrl]"
```

## Boundary

All rollout functions use time-major tensors. Rewards, terminal flags, and log
probabilities have shape `[time, ...]`; bootstrap values have shape
`[time + 1, ...]`; categorical policy logits add a final action dimension.
Action masks must be boolean, exactly match the logits, and keep at least one
valid action in every row. An expert or sampled action excluded by its mask is
rejected instead of silently trained.

For a structured action contract, inspect the number of legal actions per mask
head before assembling a learner batch:

```python
from game_learning_runtime import mask_valid_counts

mask_spec = environment.spec.action_mask
valid_actions = mask_valid_counts(mask_spec, timestep.action_mask) if mask_spec else {}
```

The helper recursively flattens nested heads and returns paths such as
`{"combat.verb": 2, "combat.target": 1}`. It returns an empty mapping when
the timestep has no mask. An `Unroll` also exposes `mask_freedom`, the fraction
of its masked transitions with more than one legal action. This makes a
masked-policy no-op visible before trusting entropy, KL, or clip diagnostics.

`terminated` and `truncated` are intentionally separate. A terminated state has
no bootstrap value. A truncated state bootstraps from the next-state value but
stops advantage or V-trace recursion at that boundary.

## Behavior cloning

```python
from game_learning_runtime.integrations.torch_objectives import behavior_cloning_loss

terms = behavior_cloning_loss(
    policy_logits,
    expert_actions,
    action_mask=action_mask,
    sample_weight=demonstration_sample_weight,
    label_smoothing=0.05,
)
terms.loss.backward()
```

The result also exposes negative log-likelihood, entropy, and detached accuracy.
Label smoothing distributes probability only across valid actions. Pass the
per-sample weight returned by `DemonstrationGate` to preserve its audited
origin/outcome policy in the optimizer objective.

## PPO and GAE

```python
from game_learning_runtime.integrations.torch_objectives import (
    generalized_advantage_estimate,
    ppo_loss,
)

targets = generalized_advantage_estimate(
    rewards=rewards,
    values=rollout_values,
    terminated=terminated,
    truncated=truncated,
)
terms = ppo_loss(
    policy_logits=policy_logits,
    actions=actions,
    old_log_prob=old_log_prob,
    advantages=targets.advantages,
    values=rollout_values[:-1],
    value_targets=targets.value_targets,
    action_mask=action_mask,
)
```

GAE targets are detached. PPO returns the total loss plus policy loss, value
loss, entropy, approximate KL, and clip fraction. Projects that use value
clipping can pass both `old_values` and `value_clip_epsilon`; omitting both keeps
the ordinary value-regression objective.

For continuous, multi-head, or hybrid policies, construct the distributions in
the project and pass their joint statistics to the distribution-independent
objective:

```python
from game_learning_runtime.integrations.torch_objectives import (
    ppo_loss_from_log_prob,
)

new_log_prob = (
    movement.log_prob(movement_action)
    + skill.log_prob(skill_action)
    + utility.log_prob(utility_action)
)
entropy = movement_entropy + skill.entropy() + utility.entropy()
terms = ppo_loss_from_log_prob(
    new_log_prob=new_log_prob,
    old_log_prob=old_log_prob,
    entropy=entropy,
    advantages=targets.advantages,
    values=values,
    value_targets=targets.value_targets,
    valid_action_counts=minimum_valid_choices,
)
```

When several categorical heads are masked, `valid_action_counts` should be the
minimum number of valid choices across those heads for each sample. This keeps
`forced_action_ratio` meaningful. Omit it when the policy has no masked
categorical head. The helper does not own distributions, models, action masks,
optimizers, or rollout collection.

## IMPALA and V-trace

```python
from game_learning_runtime.integrations.torch_objectives import impala_loss

terms = impala_loss(
    policy_logits=learner_logits,
    actions=actor_actions,
    behavior_log_prob=actor_log_prob,
    rewards=rewards,
    values=learner_values,
    terminated=terminated,
    truncated=truncated,
    action_mask=action_mask,
)
```

The returned V-trace value targets and policy advantages are detached, so the
learner gradient flows through current policy log-probabilities and value
predictions, not through the correction targets.

These functions are compositional building blocks, not a trainer. Keep model
construction, optimizer steps, batching, mixed precision, checkpointing, and
distributed actor queues in the consuming project.

## Migrate an existing project-local stack

Keep the project's encoder and actor-critic module. Replace only the duplicated
mathematics after the model has produced logits and values:

| Existing responsibility | Shared GLR primitive | Remains project-owned |
|---|---|---|
| Mask invalid action logits | `masked_logits` | Action vocabulary and mask construction |
| Expert-action cross-entropy | `behavior_cloning_loss` | Demonstration loading and filtering |
| Advantage recursion | `generalized_advantage_estimate` | Rollout assembly and reward definition |
| PPO minibatch loss | `ppo_loss` / `ppo_loss_from_log_prob` | Distributions, epochs, optimizer, early stopping, checkpoints |
| Off-policy correction | `vtrace_targets` / `impala_loss` | Actor queue, policy-version fencing, backpressure |

When an older batch exposes only `done`, split its source signal before calling
the shared functions. Map a true environment terminal to `terminated`; map a
time limit or externally interrupted episode to `truncated`. Treating every
`done` as terminal loses the bootstrap value, while treating it as truncation
can leak value through a real terminal.
