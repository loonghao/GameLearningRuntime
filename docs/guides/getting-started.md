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

Each transition has current and next observations, action and next masks,
reward, termination/truncation, events, episode identity, and step identity.

## Record BC or offline data

```python
from game_learning_runtime import JsonlTransitionWriter

with JsonlTransitionWriter("data/expert.jsonl") as writer:
    for transition in unroll.transitions:
        writer.write(transition)
```

The writer is append-only and flushes each record. Move to chunked/checksummed
storage before treating the format as a high-volume distributed replay system.
