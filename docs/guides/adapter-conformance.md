# Validate an adapter with the conformance runner

Use the public testing helper in each game-adapter repository. It exercises the
same fail-closed lifecycle and collection path used by learners, then returns a
small report that is safe to attach to CI.

```python
from game_learning_runtime.testing import run_environment_conformance

report = run_environment_conformance(
    make_authorized_environment(),
    legal_test_policy,
    steps=32,
    seed=7,
)

assert report.transition_count == 32
assert report.episode_count >= 1
```

The environment is always closed, including when reset, action selection,
stepping, or report validation fails. Every reset and step passes through
`ContractEnvironment`; collection uses `SyncCollector`. The runner additionally
rejects a participant marked both `terminated` and `truncated` on the same
transition.

## Report privacy

`EnvironmentConformanceReport` contains only:

- transition and episode counts;
- transitions carrying action masks;
- semantic event count;
- transitions carrying termination or truncation signals.

It deliberately excludes environment IDs, observations, actions, rewards,
events and payloads, metadata, timestamps, paths, hostnames, and process/window
identifiers. The consuming project should keep the same rule for surrounding
test logs.

## Representative profiles

The core test suite includes four synthetic profiles. They prove that one GLR
contract can express the structural patterns without importing any real game
code or data:

| Profile | Contract features exercised |
|---|---|
| Turn-based masked | Board tensor, discrete phase and choice, legal-action mask, terminal episode |
| Real-time combat | Nested actor state, nearby entities, continuous movement, masked discrete ability, truncation |
| FPS | Image observation, nested telemetry, continuous movement/look, binary fire action, no mask |
| ARPG | Nested player/inventory state, entity table, hierarchical parameterized command and command mask |

These are conformance fixtures, not reference environments or learner policies.
Game-specific observation encoders, reward shaping, action vocabulary, runtime
binding, authorization, latency, and end-to-end acceptance remain in the adapter
repository.

## Migrate an existing adapter test

Keep the adapter's current legal smoke-test policy and replace repeated manual
reset/step loops with `run_environment_conformance`. Assert only the capabilities
the adapter declares. For example, require every collected transition to carry
a mask only when its `EnvironmentSpec` declares `action_mask`; require semantic
events only when the trace intentionally triggers one.

Do not turn a single legacy `done` flag into both signals. Map a true runtime
terminal to `terminated` and time limits or externally interrupted episodes to
`truncated` before running conformance.
