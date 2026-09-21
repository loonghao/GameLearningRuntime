To address the issue where declared reward terms not present in `shaping_signals` were not being budgeted, we need to ensure that every declared term is included in `shaping_signals`. Here's the solution:

```python
from game_learning_runtime import EpisodeRewar

class EpisodeRewardGuard:
    def __init__(self, terms, safety):
        # Check that all terms in safety.shaping_signals are declared
        missing_shaping = sorted(set(safety.shaping_signals) - terms.keys())
        if missing_shaping:
            raise ContractViolation(f"reward safety references unknown shaping signals: {missing_shaping}")
        
        # Check that all declared terms are in safety.shaping_signals
        missing_declarations = sorted(set(terms.keys()) - set(safety.shaping_signals))
        if missing_declarations:
            raise ContractViolation(f"reward terms not declared in shaping_signals: {missing_declarations}")

    def compose(self, contributions, scale=1.0):
        # Calculate the sum of positive shaping contributions
        positive_shaping = math.fsum(
            value for name, value in contributions.items()
            if name in self._safety.shaping_signals and value > 0
        )
        
        # Apply scaling to each contribution in shaping_signals
        for name in self._safety.shaping_signals:
            contribution = contributions.get(name)
            if contribution is not None and contribution > 0:
                contributions[name] = contribution * scale
        
        return contributions, {
            "positive_shaping": positive_shaping,
            "positive_shaping_total": self._safety.max_positive_shaping_per_episode * self._safety.max_episode,
            "episode_total": sum(contributions.values()) if contributions else 0
        }
```