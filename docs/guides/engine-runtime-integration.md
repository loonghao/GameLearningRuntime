# Connect Unity and Unreal runtimes

[简体中文](engine-runtime-integration.zh-CN.md)

GLR uses one environment contract and three truthful deployment profiles. Choose
the profile from the runtime boundary you are authorized to operate, not from
the learner you plan to use.

## Choose the integration lane

| Boundary | Engine plugin | Loader plugin | External attach |
| --- | --- | --- | --- |
| Typical access | Game source or official extension SDK | Authorized BepInEx/UE4SS mod | Binary-only authorized external seam |
| Lifecycle | Physical `reset` or checkpoint restore | Truthful `attach` | Truthful `attach` by default |
| Clock | Manual step or controlled time scale | Real time | Real time |
| Observation | Semantic engine state; optional rendered sensors | Reviewed semantic engine state | Official telemetry/API first, rendered output second |
| Action | Native gameplay commands | Empty-deny bounded command vocabulary | Official action API or bounded input vocabulary |
| Target safety | Engine instance and session identity | Loader/version, episode, step, game thread | Exact process/window/session binding on every mutation |
| Throughput | Headless builds, time scaling, parallel instances | Limited by real-time game execution | Limited by real-time execution and capture latency |

Both lanes return the same immutable `EnvironmentSpec` and `TimeStep`, use the
same episode/step fencing, and work with the same collectors and learners.

## Scaffold a lane

With source access to a Unity project:

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_unity `
  --package example_unity `
  --environment-id example.unity-v1 `
  --engine unity `
  --access source
```

For an authorized binary-only Unreal runtime:

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_unreal_external `
  --package example_unreal_external `
  --environment-id example.unreal.external-v1 `
  --engine unreal `
  --access external
```

The scaffold emits `runtime-integration.json`, `training.json`, a synthetic
conformance environment, research manifest, tests, `vx.toml`, and `justfile`.
Replace only the synthetic game semantics first; retain its failing and passing
contract tests throughout the adapter implementation.

For authorized BepInEx or UE4SS hosting, use `--access loader` and provide an
exact `--loader` plus `--loader-version`. See the dedicated [loader-plugin
guide](loader-plugin-integration.md); loader code is in-process but cannot claim
source-owned reset, seed, or clock control.

## Source-integrated engine plugin

The engine side owns observation encoding, action application, rewards, masks,
terminal conditions, physical reset, and authoritative post-state. It should:

1. collect semantic state and apply actions on the engine/game thread;
2. expose a stable action vocabulary instead of reflection or script execution;
3. reset or restore a checkpoint before returning GLR step zero;
4. advance a fixed simulation quantum for each accepted action;
5. return the resulting state before acknowledging the action; and
6. support isolated headless or packaged instances when the engine permits it.

For Unity, an adapter can map `CollectObservations`, `ActionBuffers`, discrete
action masks, `OnActionReceived`, and episode callbacks to GLR. ML-Agents can be
an optional compatibility provider, while GLR remains the learner-neutral
boundary. Decision timing may follow fixed physics updates or explicit game
events. Unity's Academy and Agent lifecycle is documented in the official
[learning environment design guide](https://unity-technologies.github.io/ml-agents/Learning-Environment-Design/).

For Unreal, implement the adapter as a plugin component or subsystem and marshal
gameplay state and mutations to the game thread. Learning Agents' Interactor,
Manager, Recorder, training environment, and communicator can be mapped to GLR
where useful. Epic's current API includes local shared-memory and socket
communicators, but GLR only adopts a concrete high-speed transport after a paired
benchmark. See the official [Learning Agents API](https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/LearningAgents)
and [training API](https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/LearningAgentsTraining).

## Binary-only external attachment

Use only a runtime and interaction seam that the operator is authorized to use.
Prefer observations and actions in this order:

1. official game, mod, accessibility, telemetry, replay, or test APIs;
2. documented logs or exported state with bounded freshness;
3. rendered frames and an explicitly bounded input vocabulary.

An external adapter must not expose inaccessible hidden state, arbitrary memory,
reflection, scripts, generic clicks, unrestricted process discovery, or
anti-cheat bypasses. It normally advertises `live-attach`; seeded reset and
manual clock claims are rejected. Every mutating action carries the GLR episode
and expected step, is bound to the selected runtime, holds a releasable input
lease when needed, and returns verified post-state.

If an official API provides both semantic observations and bounded actions,
declare that explicitly:

```python
from game_learning_runtime import (
    ActionMode,
    EngineFamily,
    ObservationMode,
    RuntimeIntegrationProfile,
    TransportMode,
)

profile = RuntimeIntegrationProfile.for_external(
    EngineFamily.UNITY,
    observation_mode=ObservationMode.OFFICIAL_API,
    action_mode=ActionMode.OFFICIAL_API,
    transport_mode=TransportMode.OFFICIAL_API,
)
environment = profile.connect(authorized_driver)
```

## Throughput checklist

- Keep images optional when semantic tensors solve the task.
- Separate decision frequency from render and physics frequency.
- Batch agents or launch isolated game instances instead of adding learner logic
  to the adapter.
- Bound frames, queues, deadlines, and main-thread work.
- Never retry a mutating action after an ambiguous transport failure; reconcile
  through authoritative readback.
- Promote framing, shared-memory rings, or tensor conversion to Rust only after
  the reproducible ADR-0004 benchmark threshold is met.

Run the same quality entrypoint locally and in CI:

```powershell
vx setup
vx just check
```

Synthetic conformance does not prove a real Unity or Unreal game integration.
Live acceptance requires a bounded authorized runtime trace and may publish only
aggregate results.
