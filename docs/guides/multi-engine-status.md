# Multi-engine implementation and acceptance

Engine semantics remain separate from runtime loading, transport and external
input/capture. The shared `GameEnvironment` and `BridgeDriver` interfaces own
the learner-facing reset/attach/step contract. Providers declare capabilities;
selecting an engine never adds a live capability.

| Target | Implemented and locally testable | Still requires runtime integration |
| --- | --- | --- |
| Unity Mono | Installation hints, source/external scaffolds, existing BepInEx bootstrap | Concrete semantic provider and authenticated target-bound bridge |
| Unity IL2CPP | Installation hints, explicit source/external selection, rejection of Mono loader substitution | IL2CPP bootstrap/provider and live trace |
| Unreal | Project hints, source/external scaffolds, existing UE4SS bootstrap and C++ SDK | Concrete game provider and live trace |
| Godot | Project/PCK hints, source/external profiles and synthetic scaffold | Godot node/extension provider and live trace |
| External input/capture | Tested command allowlist, bounded hold, sequence fencing, fresh readback, cleanup and failure closure | Concrete input/capture backend with target binding, deadlines and watchdog |

The reusable `InputCaptureSession` implements coordination, not OS input.
The engine adapter projects its captured RGB frames into observations and
owns any game-specific reward/terminal extraction. A successful capture does
not prove a game action achieved its intended result. Neither this session nor
an external profile exposes physical reset or exact frame stepping.

There is no four-engine live support claim. No game target was selected or
modified for these tests. The synthetic scaffold explicitly records
`implementation_status=synthetic-seam` and `live_verified=false`.

See the [adapter Skill reference](../../.agents/skills/glr-adapter-builder/references/multi-engine.md)
for the commands and contract boundaries, and [skills publication](skills-publication.md)
for CI distribution.

Installation markers are conservative hints based on the official
[Unity Windows IL2CPP layout](https://docs.unity3d.com/ja/2023.2/Manual/WindowsPlayerIL2CPPScriptingBackend.html)
and [Godot pack documentation](https://docs.godotengine.org/en/stable/tutorials/export/exporting_pcks.html).
Absence of a marker is not proof of another runtime.
