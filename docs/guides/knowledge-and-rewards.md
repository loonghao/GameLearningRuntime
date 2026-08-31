# Configure knowledge sources and rewards

Use `glr.training.v1` to keep collection policy and reward composition
reviewable without putting game logic or executable expressions in a config
file.

## Configuration

```json
{
  "schema_version": "glr.training.v1",
  "lifecycle": {"start_mode": "attach", "stop_on_done": true},
  "bridge": {
    "required_capabilities": [
      "authenticated",
      "live-attach",
      "postcondition-verified",
      "target-bound"
    ]
  },
  "knowledge_sources": [
    {
      "id": "runtime",
      "authority": "authoritative",
      "required": true,
      "max_age_seconds": 0,
      "max_payload_bytes": 65536
    },
    {
      "id": "guide-research",
      "authority": "advisory",
      "required": false,
      "max_age_seconds": 604800,
      "max_payload_bytes": 16384
    }
  ],
  "reward": {
    "minimum": -20,
    "maximum": 20,
    "terms": [
      {
        "name": "progress",
        "source": "runtime",
        "weight": 1,
        "minimum": -1,
        "maximum": 1,
        "required": true
      },
      {
        "name": "outcome",
        "source": "runtime",
        "weight": 10,
        "minimum": -1,
        "maximum": 1,
        "required": false
      }
    ]
  }
}
```

Identifiers are deliberately not connection strings. Keep HTTP endpoints,
pipe names, credentials, process/window IDs, and local paths in deployment
configuration outside the public training contract.

Load and compose signals:

```python
from game_learning_runtime import RewardComposer, RewardSignal, load_training_config

config = load_training_config("training.json")
composer = RewardComposer(config)
result = composer.compose([RewardSignal(name="progress", source="runtime", value=1.4)])

assert result.total == 1.0
assert result.contributions == {"progress": 1.0}
```

The adapter computes scalar signals in reviewed code. The composer performs
source matching, required-signal checks, finite-number validation, term
clipping, weighting, total clipping, and immutable breakdown construction.

Per-term clipping is not an episode invariant: repeated local rewards can still
make a failed episode profitable. Route composed signals through
`EpisodeRewardGuard`, configure per-step and per-episode positive shaping
budgets, and require an authoritative terminal outcome. Gate BC demonstrations
separately by immutable origin and result. See [Training safety: reward budgets
and BC provenance](training-safety.md).

## Authority model

- `authoritative`: target-bound runtime state with the postcondition guarantees
  needed by the adapter.
- `advisory`: guides, wikis, strategy priors, caches, model suggestions, or any
  source that cannot prove the runtime state.

Reward terms require `authoritative` by default. To experiment with an advisory
shaping term, set `minimum_authority` to `advisory` on that term, keep its
weight and clip bounded, document the reason, and run an ablation. Advisory
knowledge never expands the action mask or acknowledges an action.

## Gameplay research for new adapters

Use `.agents/skills/glr-adapter-builder` to scaffold a trainable synthetic lane
and a `glr.knowledge-research.v1` manifest. New agents should search current
official rules and patch notes first, then maintained wikis and reputable
guides. Store public URLs and concise paraphrases with access dates,
confidence, volatility, and `unverified`, `runtime-verified`, or `rejected`
status.

Research can propose observation fields, action semantics, masks, and reward
hypotheses. Only bounded authorized runtime evidence can promote a hypothesis
to authoritative reward behavior. Do not copy full guides or store account,
host, process, window, path, token, or proprietary runtime data.
