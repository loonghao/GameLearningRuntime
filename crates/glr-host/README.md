# GLR Runtime Host

`glr-hostd` is the first transport and lifecycle host for Game Learning Runtime.
It provides a bounded, serialized `glr.host.v1` JSON-lines session and enforces
episode/step fencing before a semantic action reaches a provider.

This release contains only the explicit `synthetic-counter` conformance
provider:

```powershell
glr-hostd --provider synthetic-counter --transport stdio
```

It is not a universal game loader, does not discover processes or game
installations, and does not claim authenticated or target-bound IPC. Unity and
Unreal integrations still need an authorized official plugin, BepInEx, UE4SS,
or another reviewed bootstrap to expose an engine-specific provider. See the
repository documentation for the C# and C++ provider contracts and current
capability boundary.
