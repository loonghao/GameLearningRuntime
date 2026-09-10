# Multi-engine implementation and acceptance

Engine semantics remain separate from runtime loading, transport and external
input/capture. The shared `GameEnvironment` and `BridgeDriver` interfaces own
the learner-facing reset/attach/step contract. Providers declare capabilities;
selecting an engine never adds a live capability.

| Target | Implemented and locally testable | Still requires runtime integration |
| --- | --- | --- |
| Unity Mono | Real Transform provider using the C# SDK; Unity 2022.3 Editor and standalone Mono Player reset/step/stale-request checks passed | Game-specific semantics and authenticated runtime transport |
| Unity IL2CPP | Same source provider built and executed in a standalone IL2CPP Player after installing the matching official module | Binary-only IL2CPP loader integration and game-specific semantics |
| Unreal | Unreal 5.8.1 Editor commandlet passed Actor, SceneComponent and allowlisted level transition/readback checks | Packaged-game runtime module and game-specific semantics |
| Godot | Godot 4.6.3 Node3D source sample passed reset/step/stale-request checks through the existing Python HostBridgeDriver | Authenticated IPC and game-specific semantics |
| External input/capture | Concrete DccCuaBackend uses persistent exact-window sessions, fenced bounded clicks and PNG/RGB capture; live capture and game counter change observed | Repeatable end-to-end acceptance is currently blocked by Windows input-desktop access denial and D3D11 capture failure |

The reusable `InputCaptureSession` implements coordination; `DccCuaBackend`
provides OS input/capture through the project-owned DCC-CUA runtime.
The engine adapter projects its captured RGB frames into observations and
owns any game-specific reward/terminal extraction. A successful capture does
not prove a game action achieved its intended result. Neither this session nor
an external profile exposes physical reset or exact frame stepping.

There is no universal game support claim. All engine tests used new isolated
source samples, not existing user games. The synthetic scaffold still records
`implementation_status=synthetic-seam` and `live_verified=false`.

See [native sample validation](native-provider-validation.md) for reproducible
commands, engine versions and the limits of these observations.

See the [adapter Skill reference](../../.agents/skills/glr-adapter-builder/references/multi-engine.md)
for the commands and contract boundaries, and [skills publication](skills-publication.md)
for CI distribution.

Installation markers are conservative hints based on the official
[Unity Windows IL2CPP layout](https://docs.unity3d.com/ja/2023.2/Manual/WindowsPlayerIL2CPPScriptingBackend.html)
and [Godot pack documentation](https://docs.godotengine.org/en/stable/tutorials/export/exporting_pcks.html).
Absence of a marker is not proof of another runtime.
