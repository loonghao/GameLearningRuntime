# Training safety: reward budgets and BC provenance

GLR protects two learner-neutral boundaries: the return emitted for an episode
and the evidence admitted as an expert demonstration. These gates complement
game-specific reward design; they do not infer success from a score or guide.

## Prevent shaping from making failure profitable

Keep named terms in `training.json`, including an optional terminal-only outcome
term with a positive weight. Configure episode guardrails separately:

```json
{
  "schema_version": "glr.reward-safety.v1",
  "outcome_signal": "outcome",
  "shaping_signals": ["progress"],
  "max_positive_shaping_per_step": 1,
  "max_positive_shaping_per_episode": 10,
  "failure_episode_maximum": 0,
  "require_terminal_outcome": true
}
```

Route every step through the guard:

```python
from game_learning_runtime import (
    EpisodeRewardGuard,
    RewardSignal,
    load_reward_safety_config,
    load_training_config,
)

guard = EpisodeRewardGuard(
    load_training_config("training.json"),
    load_reward_safety_config("reward-safety.json"),
)

step = guard.compose([RewardSignal("progress", "runtime", 0.5)])
terminal = guard.compose(
    [
        RewardSignal("progress", "runtime", 0),
        RewardSignal("outcome", "runtime", -1),
    ],
    terminal=True,
)
assert terminal.episode_total <= 0
```

The guard limits only positive shaping. Negative evidence remains intact. If a
terminal failure would still exceed the failure ceiling, the result records a
`guardrail.failure-correction` contribution. Monitor that correction and the
`suppressed_positive_shaping` counter: frequent intervention usually means the
underlying shaping needs redesign or ablation.

Call `reset()` only when a new logical episode starts. An outcome signal before
terminal, a terminal transition without the required outcome, or another step
after terminal fails closed.

## Stop BC policy self-imitation

Every trajectory admitted to BC needs immutable origin and authoritative
episode outcome. The generated default is deliberately strict:

```json
{
  "schema_version": "glr.demonstration-policy.v1",
  "allowed_origins": ["human", "scripted-expert"],
  "allowed_outcomes": ["success"],
  "origin_weights": {"human": 1, "scripted-expert": 1},
  "outcome_weights": {"success": 1},
  "reject_unknown": true
}
```

Validate before dataset insertion:

```python
from game_learning_runtime import (
    DemonstrationGate,
    DemonstrationOrigin,
    DemonstrationOutcome,
    DemonstrationProvenance,
    load_demonstration_policy_config,
)

gate = DemonstrationGate(load_demonstration_policy_config("demonstration-policy.json"))
decision = gate.validate(
    DemonstrationProvenance(
        origin=DemonstrationOrigin.HUMAN,
        outcome=DemonstrationOutcome.SUCCESS,
    )
)
dataset.add(trajectory, weight=decision.sample_weight)
```

Do not infer provenance from the action looking reasonable. Preserve the actor
identity and final outcome at collection time. Policy-produced data belongs in
a separately named distillation or offline-RL policy and requires explicit
allowlisting plus `policy_id`; never relabel it as human or scripted expert.

Include both JSON policies, aggregate acceptance/rejection counts, seeds, and
the trainer source in the model bundle. Do not publish raw proprietary traces,
account identifiers, local paths, or process/window identifiers.
