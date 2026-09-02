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
