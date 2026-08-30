# Adapt an existing Gymnasium environment

Install the optional compatibility layer:

```powershell
uv add "game-learning-runtime[gymnasium]"
```

Wrap an existing environment once, then reuse GLR validation, collectors,
recorders, or TorchRL:

```python
from game_learning_runtime import ContractEnvironment
from game_learning_runtime.integrations.gymnasium import GymnasiumEnvironment

source = make_authorized_environment()
adapter = GymnasiumEnvironment(
    source,
    environment_id="example.runtime-v1",
    action_mask_provider=source.action_masks,
)
environment = ContractEnvironment(adapter)
```

Box and Discrete roots are wrapped as `observation` and `action`. Dict spaces
retain their semantic field names. Tuple, MultiDiscrete, and MultiBinary spaces
are also converted to typed GLR trees.

## Metadata and privacy

Gymnasium `info` is empty in GLR by default. This prevents incidental values
such as local paths, account identifiers, window/process details, and bridge
diagnostics from entering replay files or reports.

Export only a stable allowlisted view when a learner truly needs metadata:

```python
adapter = GymnasiumEnvironment(
    source,
    environment_id="example.runtime-v1",
    info_transform=lambda info: {
        "difficulty": str(info["difficulty"]),
        "score": float(info["score"]),
    },
)
```

Do not forward `info` wholesale. Keep observations, reward calculation, action
execution, and authorization inside the game-specific adapter.
