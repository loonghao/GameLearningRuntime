# Bind an invocation run context

Use a run context when one GLR invocation must bind an exact set of
project-owned configuration inputs. Keep setup and orchestration in a VX-backed
`glr.toml` task; use the context only for immutable identity and handoff.

```toml
# config/contexts/ranked.toml
schema_version = "glr.run-context.v1"
context_id = "ranked-2026"
environment_id = "example.context-v1"
protocol_version = "1.0"

[labels]
season = "ranked-2026"
ruleset = "standard"

[[inputs]]
owner = "training"
path = "config/training.json"
schema_version = "glr.training.v1"
```

Inspect or execute with the same selection:

```powershell
glr --project . --context config/contexts/ranked.toml --json doctor
glr --project . --context config/contexts/ranked.toml --json train
```

The CLI validates the environment and protocol, freezes the context file and
every declared JSON/TOML input, and rechecks their bytes before each role. A run
stores `run-context.json`, registers it with role `run-context`, and emits
`context.selected`. Roles receive the same compact JSON as `GLR_RUN_CONTEXT`
and its digest as `GLR_RUN_CONTEXT_SHA256`.

Python roles should fail closed on partial, malformed, or changed receipts:

```python
from game_learning_runtime import load_inherited_run_context

context = load_inherited_run_context(project)
if context is not None:
    training = next(item for item in context.inputs if item.owner == "training")
```

`--context` is supported by `doctor`, `runtime start`, `train`, `goal run`, and
`play`. It is intentionally rejected by task, query, report, transaction, and
maintenance commands. A digest proves input identity, not runtime readiness,
successful training, or gameplay acceptance.
