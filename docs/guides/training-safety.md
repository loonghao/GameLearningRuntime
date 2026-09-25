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

## Bound negative shaping before surviving stops paying

A positive-only budget caps the upside of shaping, not its downside. When a
negative shaping term keeps charging as the episode grows, the return peaks
before the episode ends and the optimal policy becomes dying early — the exact
opposite of a terminal-dominance reward. Add the symmetric budgets when a
shaping term can go negative:

```json
{
  "schema_version": "glr.reward-safety.v1",
  "outcome_signal": "outcome",
  "shaping_signals": ["survival", "damage"],
  "max_positive_shaping_per_step": 5,
  "max_positive_shaping_per_episode": 100,
  "max_negative_shaping_per_step": 1,
  "max_negative_shaping_per_episode": 10,
  "failure_episode_maximum": 0,
  "require_terminal_outcome": true
}
```

Both fields bound the **magnitude** of negative shaping and both are optional;
omitting them leaves negative shaping unbounded, so every existing policy keeps
its current behaviour. `max_negative_shaping_per_step` cannot exceed
`max_negative_shaping_per_episode`. When no episode budget is declared and a
shaping term can contribute a negative value, the guard logs a warning at
construction time naming the terms.

The guard reports `negative_shaping_total` and `suppressed_negative_shaping`
alongside the positive counters, so a budget that fires constantly is visible
before it silently reshapes the objective. Measure the effect before tightening
a default: compare the per-episode regret — the return lost by surviving to the
end instead of stopping at the peak — with and without the budget.

Call `reset()` only when a new logical episode starts. An outcome signal before
terminal, a terminal transition without the required outcome, or another step
after terminal fails closed.

### Classify every reward term or fail closed

Every term in `training.json` must be classified by `reward-safety.json`: the
terminal `outcome_signal`, a member of `shaping_signals`, or an explicit opt-in
member of `unbudgeted_signals`. A declared term that is none of the three makes
`EpisodeRewardGuard` raise at construction time instead of silently escaping the
episode budget:

```python
EpisodeRewardGuard(training, safety)
# ContractViolation: reward terms are neither the outcome signal nor a declared
# shaping signal: ['item_score']; add them to shaping_signals, or to
# unbudgeted_signals if they must stay outside the budget
```

`unbudgeted_signals` defaults to `()`. It exists for terms that must stay
outside the episode budget, such as an externally audited score feed; choosing
it is a decision, not a fallback, so the name has to be spelled out, it must
match a declared reward term, and it cannot overlap `shaping_signals` or the
`outcome_signal`. Unbudgeted terms still count towards the step and episode
return; only the positive-shaping budget ignores them.

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

## Bind provenance to exact transition bytes

Do not keep provenance in an unrelated filename or database row. Build an
adjacent artifact manifest after an authorized collector records one complete
`glr.transition.v1` episode:

```python
from game_learning_runtime import (
    DemonstrationProvenance,
    build_demonstration_artifact,
)

build_demonstration_artifact(
    "episode.manifest.json",
    trajectory_path="episode.jsonl",
    environment_id=environment.spec.environment_id,
    provenance=DemonstrationProvenance(
        origin="scripted-expert",
        outcome="success",
    ),
)
```

Before BC ingestion, verify the expected environment, trajectory SHA-256,
single contiguous episode, terminal outcome, and demonstration policy in one
operation:

```python
from game_learning_runtime import verify_demonstration_artifact

verified = verify_demonstration_artifact(
    "episode.manifest.json",
    gate=gate,
    expected_environment_id=environment.spec.environment_id,
)
dataset.add(verified.transitions, weight=verified.sample_weight)
```

Verification returns the parsed transitions that were hashed. Training code
should consume those objects directly instead of reopening the path.
