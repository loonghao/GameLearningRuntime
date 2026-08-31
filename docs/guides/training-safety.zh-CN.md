# 训练安全：奖励预算与 BC 数据溯源

GLR 在学习器之外保护两个边界：一个 episode 最终交给训练器的回报，以及哪些数据
可以作为专家示范进入 BC。它们用于约束游戏特定的奖励设计，但不会根据分数或攻略
自行推断胜负。

## 避免局部奖励让失败局仍然“赚钱”

在 `training.json` 中保留命名奖励项，并为终局 outcome 配置一个权重为正、平时可缺省
的奖励项。再用独立文件定义 episode 级门禁：

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

所有 step 都必须经过 guard：

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

Guard 只限制正向 shaping，不会抹掉负向证据。如果失败局累计回报仍高于上限，终局
结果会加入可审计的 `guardrail.failure-correction`。应持续观察这个修正项和
`suppressed_positive_shaping`；如果它们频繁触发，说明需要重新设计或消融 shaping，
而不是继续提高终局奖励数值。

只有新逻辑 episode 开始时才调用 `reset()`。非终局提前发送 outcome、终局缺少必需
outcome，或终局后未 reset 继续发送 step，都会失败即关闭。

## 阻止 BC 模仿策略自身

每条进入 BC 的轨迹都必须携带不可变的来源和权威终局结果。脚手架默认配置刻意严格：

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

写入数据集之前先验证：

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

不要因为动作“看起来合理”就推断它是专家样本。采集时就要保存 actor 身份与最终结果。
策略生成的数据只能进入单独命名、明确配置的蒸馏或离线 RL 流程，并携带 `policy_id`；
不得把它改标为人类或脚本专家。

模型包应包含两个 JSON 策略、聚合后的接受/拒绝计数、随机种子和训练器源码。不要发布
专有原始轨迹、账号标识、本地路径或进程/窗口标识。
