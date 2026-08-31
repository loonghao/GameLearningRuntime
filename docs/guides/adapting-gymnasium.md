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

## Attach to a running game

Gymnasium has no standard live-attachment lifecycle. If an authorized runtime
can observe an already-running world without resetting it, expose that
operation explicitly and pass it as `attach_provider`:

```python
adapter = GymnasiumEnvironment(
    source,
    environment_id="example.live-runtime-v1",
    action_mask_provider=source.action_masks,
    attach_provider=source.attach,
)
environment = ContractEnvironment(adapter)
initial = environment.attach(options={"continuation": "current"})
```

Providing the hook advertises the `live-attach` capability. Omitting it keeps
attachment fail-closed. The hook must return the same `(observation, info)`
shape as Gymnasium `reset()`, but it must not call or impersonate a physical
reset. Each attachment starts a fresh logical GLR episode at step zero.

If the authorized integration layer enforces additional capabilities, declare
only those verified at the same boundary:

```python
adapter = GymnasiumEnvironment(
    source,
    environment_id="example.live-runtime-v1",
    attach_provider=source.attach,
    verified_capabilities={
        "authenticated",
        "postcondition-verified",
        "target-bound",
    },
)
```

These names are assertions by the caller, not capabilities inferred by the
Gymnasium wrapper. Do not declare authentication, target binding, reset, or
postcondition verification unless the underlying runtime and transport fail
closed when that property is absent.

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
