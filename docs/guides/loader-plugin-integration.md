# Connect authorized BepInEx and UE4SS runtimes

[简体中文](loader-plugin-integration.zh-CN.md)

Loader plugins are GLR's middle lane: code runs inside an authorized game, but
the adapter does not own the source project. They can provide semantic state
and main-thread action dispatch while retaining truthful live-attach and
real-time lifecycle claims.

## Supported first templates

| Loader | Generated host | Upstream status to verify |
| --- | --- | --- |
| BepInEx | Unity Mono C# plugin for BepInEx 5 LTS | Check the exact game/runtime and official release |
| UE4SS | Unreal Lua mod for UE4SS 3.x | Check the exact Unreal/game build and official release |

GLR does not vendor either loader. BepInEx IL2CPP and UE4SS C++ templates are
not generated yet. Never substitute those variants silently.

## Scaffold

Unity Mono with an explicitly selected BepInEx release:

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_bepinex `
  --package example_bepinex `
  --environment-id example.bepinex-v1 `
  --engine unity `
  --access loader `
  --loader bepinex `
  --loader-version v5.4.23.5
```

Unreal with an explicitly selected UE4SS release:

```powershell
vx python .agents/skills/glr-adapter-builder/scripts/scaffold_adapter.py `
  --output adapters/example_ue4ss `
  --package example_ue4ss `
  --environment-id example.ue4ss-v1 `
  --engine unreal `
  --access loader `
  --loader ue4ss `
  --loader-version v3.0.1
```

These versions are reproducible examples, not compatibility promises. Refresh
official upstream releases and validate the target game before choosing one.

## Agent operation surface

The generated `agent-interface.json` exposes only `describe`, `attach`, `step`,
and `close`. Unknown operations are denied. `step` requires
the current episode and expected step identity, and `action_vocabulary` starts
empty. The adjacent `AGENTS.md` tells a new Agent which files to inspect, which
commands to run, and which actions are forbidden.

Implement a small game-semantic action table; do not expose reflection, object
search/dumps, arbitrary Lua/C# execution, generic function calls, or raw input.
The BepInEx host has a fixed queue bound and `TryApply` post-state contract. The
UE4SS host uses `ExecuteInGameThread` and accepts no action until a reviewed Lua
handler is added.

## Build and stage deployment

For BepInEx, build the generated project with operator-provided reference
locations. Keep those locations out of Git:

```powershell
dotnet build runtime/bepinex/GlrBridge.csproj -c Release `
  -p:BEPINEX_ROOT="$env:BEPINEX_ROOT" `
  -p:GAME_MANAGED_ROOT="$env:GAME_MANAGED_ROOT"
```

UE4SS Lua needs no compilation. Stage either loader using the declared relative
layout:

```powershell
vx run package-runtime
```

The result under `.glr-dist/loader-package` contains `payload/` plus a
checksummed manifest. Packaging never scans for a game or installs anything.
An operator must select the exact authorized target and approve the copy.

## Train and reproduce

```powershell
vx run check
vx run train
vx run reproduce
```

The initial trainer is a deterministic synthetic behavior-cloning smoke test.
It creates a self-contained `glr.model-bundle.v1`; replace the learner while
preserving the same bundle gate. See [reproducible model
bundles](reproducible-model-bundles.md).

Synthetic checks do not prove loader startup or live game behavior. Record a
separate bounded authorized acceptance trace and publish only aggregate
conformance evidence.

Official upstream references:

- [BepInEx](https://github.com/BepInEx/BepInEx)
- [BepInEx plugin development](https://docs.bepinex.dev/master/articles/dev_guide/plugin_tutorial/index.html)
- [UE4SS](https://github.com/UE4SS-RE/RE-UE4SS)
- [UE4SS Lua mods](https://docs.ue4ss.com/dev/guides/creating-a-lua-mod.html)
- [UE4SS game-thread dispatch](https://docs.ue4ss.com/dev/lua-api/global-functions/executeingamethread.html)
