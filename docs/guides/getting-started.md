# Getting started

## Install for development

```powershell
git clone https://github.com/loonghao/GameLearningRuntime.git
cd GameLearningRuntime
uv sync --frozen --all-groups
uv run pytest -m "not torchrl"
```

## Implement an adapter

1. Declare an immutable `EnvironmentSpec`.
2. Implement `GameEnvironment.reset()` and `step()`.
3. Generate a new episode UUID on every reset and start at step zero.
4. Increment the step ID exactly once for each accepted action.
5. Wrap the adapter with `ContractEnvironment` in tests and production clients.

If the authorized runtime is already running and cannot provide a truthful
physical reset, also advertise the `live-attach` capability and implement
`GameEnvironment.attach()`. Attach must return a fresh logical episode at step
zero without claiming that the world was reset. Use
`SyncCollector(..., start_mode="attach")`; seeded starts are intentionally
rejected in this mode.

See the packaged
[`game_learning_runtime.examples.counter`](../../src/game_learning_runtime/examples/counter.py)
module for a complete reference.

## Collect learner-neutral data

```python
from game_learning_runtime import ContractEnvironment, SyncCollector
from game_learning_runtime.examples import CounterEnvironment, always_increment

env = ContractEnvironment(CounterEnvironment())
unroll = SyncCollector(env).collect(always_increment, steps=128)
```

For a live environment where a transient step failure should preserve the work
already collected, opt into a truncated partial unroll:

```python
partial = SyncCollector(env).collect(
    always_increment,
    steps=128,
    on_error="partial",
)
```

The final valid transition is marked `truncated`, and the next collection
starts a fresh episode because the post-failure environment state is unknown.
The default `on_error="raise"` remains fail-fast.

Each transition has current and next observations, action and next masks,
reward, termination/truncation, events, episode identity, and step identity.
When an adapter has mutable options, implement `config_snapshot()` to return
the active string-valued settings. The collector stores an immutable snapshot
and deterministic `environment_config_digest` on each unroll, so runs can be
grouped by environment configuration rather than mistaken for policy changes.

## Record BC or offline data

```python
from game_learning_runtime import JsonlTransitionWriter

with JsonlTransitionWriter("data/expert.jsonl") as writer:
    for transition in unroll.transitions:
        writer.write(transition)
```

The writer is append-only and flushes each record. Move to chunked/checksummed
storage before treating the format as a high-volume distributed replay system.

## Keep adapter scratch state across trials

Small adapter bookkeeping can survive a trial process restart without entering
the policy checkpoint:

```python
from game_learning_runtime import RunStatus, TrainingStore

store = TrainingStore(".glr/runs.sqlite3")
run = store.create_run(
    environment_id="example.adventure-v1",
    protocol_version="1.0",
    kind="training",
)
state = store.run_state(run.run_id, "adapter/inventory", schema_version=1)
state["unusable_targets"] = ["target.one"]

# A later trial/process opens the same namespace and sees the committed value.
state = TrainingStore(".glr/runs.sqlite3").run_state(
    run.run_id, "adapter/inventory", schema_version=1
)
assert state["unusable_targets"] == ["target.one"]
store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
```

The namespace is adapter-opaque, bounded to a 64 KiB JSON snapshot, and
schema-versioned. Terminal runs delete their scratch state; use a checkpoint
for policy weights or any state that must outlive the run.
