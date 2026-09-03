# Runtime Host and provider SDKs

The GLR Runtime Host removes repeated transport and lifecycle code from game
adapters without pretending that one framework can bootstrap every engine.

```text
Python learner / collector
  -> BridgeEnvironment
  -> HostBridgeDriver
  -> glr-hostd (Rust: frame + lifecycle + fencing)
  -> engine provider (C# or C++: semantics + main thread + post-state)
  -> official plugin / BepInEx / UE4SS / official mod SDK
  -> authorized runtime
```

## Run the conformance host

Build and exercise the real subprocess boundary:

```powershell
vx setup
vx just rust-check
vx just host-smoke
```

Or launch it from Python with an explicit downloaded or locally built binary:

```python
from pathlib import Path

from game_learning_runtime import (
    BridgeEnvironment,
    ContractEnvironment,
    HostBridgeDriver,
    HostProcessConfig,
)

config = HostProcessConfig(executable=Path("C:/tools/glr-hostd.exe"))
driver = HostBridgeDriver.from_process(config)
environment = ContractEnvironment(
    BridgeEnvironment(
        driver,
        required_capabilities={"host-stdio", "reset", "step"},
    )
)
```

`HostProcessConfig` rejects a relative or missing executable. It never invokes a
shell or searches for a game. The current host supports only the
`synthetic-counter` provider and serialized stdio. This is a real end-to-end
contract test, not live Unity/Unreal acceptance. A response timeout or malformed
frame fail-closes that child session; callers must start a fresh host rather
than risk pairing a late response with another action.

## Reconnect and reconcile an in-flight action

Providers that can prove durable episode state may advertise
`reconnect-resume-v1` and implement the optional resumable provider contract.
The caller sends the episode ID and its last committed step. The provider
returns the authoritative `ProviderTimeStep` plus an optional
`ActionReconciliation` for the action that was in flight when transport was
lost. `applied`, `not_applied`, and `unknown` are authoritative outcomes;
`retryable` is only true when the provider can prove a retry is safe. A
reconnect result never advances the cursor beyond the returned authoritative
step, and a provider must reject episode or cursor mismatches.

### Bounded realtime control

Providers may advertise `glr.realtime-control.v1` in `realtime_timing` with
minimum/maximum hold, settle deadline, simulation quantum, and clock source.
Each realtime step carries bounded `deadline_ns`, `quantum_ns`, and optional
`hold_ns` values. `lease` operations (`acquire`, `renew`, `release`, and
`preempt`) bind the same session and target identity; stale leases are rejected
before provider dispatch. `cancel` fences an obsolete action. The typed receipt
reports `consumed`, `expired`, `cancelled`, or `rejected` and is never retried
implicitly.

### Runtime identity and health

Providers that participate in launcher upgrades may include a stable
`runtime_identity` in `Describe` and implement the read-only `Health` request.
The identity contains only a public runtime ID and immutable provider version.
The health snapshot carries `starting`, `ready`, `draining`, `unhealthy`, or
`stopped`, the provider timestamp, whether new sessions are accepted, a bounded
active-session count, and optional lease metadata. A launcher should bind each
training session to the identity it observed, require a fresh `ready` snapshot
before switching its `current` selection, and treat an identity mismatch or
ambiguous health response as a failed check. Runtime health does not expose
paths, PIDs, hostnames, or credentials, and it does not replace launcher-owned
drain, process-ownership, or atomic executable replacement logic.

## Implement a Unity provider

Build or download `GameLearningRuntime.Provider`, reference the .NET Standard
2.0 assembly from the authorized Unity plugin, and implement
`IRuntimeProvider`. Keep the official Unity or BepInEx plugin as a thin
bootstrap that:

1. binds the reviewed runtime instance;
2. dispatches `Reset`, `Attach`, and `Step` on the Unity main thread;
3. translates semantic state/actions to copied `TensorBuffer` values;
4. returns authoritative `ProviderTimeStep` post-state; and
5. releases hooks and owned state in `Dispose`.

The provider must reject physical `Reset` if only live `Attach` is truthful.
Do not expose arbitrary reflection, method calls, or C# evaluation.

## Implement an Unreal provider

Include `sdk/cpp/include/glr/provider.hpp` in an Unreal runtime module and
implement `glr::runtime_provider`. The Unreal-facing layer owns Game Thread
marshalling and UObject lifetime. The contract header has no Unreal dependency,
so CI can compile it independently before a licensed engine acceptance run.

An authorized UE4SS mod can remain the bootstrap when source is unavailable,
but the mod policy, exact upstream tag, game version, and game-thread behavior
still need separate review. GLR does not vendor or silently install UE4SS.

## Training and reproduction

Once a live provider transport is available, the learner still consumes the
ordinary `GameEnvironment` contract. Use the existing collector, reward safety,
BC provenance, TorchRL adapter, and model-bundle workflow unchanged. Store the
Runtime Host version, provider SDK version, runtime-integration profile, reward
policy, seeds, lock files, model artifacts, and aggregate metrics in the model
bundle. Never store authentication material or local executable/game paths.

## Current capability boundary

| Capability | Current state |
| --- | --- |
| Rust lifecycle/fencing core | Implemented and tested |
| Bounded stdio client and host | Implemented; serialized, 1 MiB hard bound |
| Synthetic process smoke | Implemented; aggregate-only evidence |
| C# Unity/provider contract | Implemented; .NET Standard 2.0 |
| C++ Unreal/provider contract | Implemented; header-only C++20 |
| Reconnect/resume reconciliation | Implemented as opt-in `reconnect-resume-v1` |
| Runtime identity and read-only health | Implemented as opt-in `runtime-health-v1` |
| Authenticated target-bound local IPC | Not implemented |
| Live external C#/C++ provider connection | Not implemented |
| Shared memory / asynchronous actor queue | Optional standard-library `BoundedActorQueue`; shared memory remains benchmark-gated |
| Universal injection/bootstrap | Intentionally not provided |
