# GameLearningRuntime.Provider

Engine-neutral C# provider contracts for the Game Learning Runtime (GLR)
Runtime Host.

The package targets .NET Standard 2.0 so an authorized Unity plugin or
BepInEx 5 Unity Mono bootstrap can implement `IRuntimeProvider` while keeping
training algorithms outside the game process. The engine bootstrap remains
responsible for exact runtime binding, main-thread dispatch, semantic action
validation, authoritative post-state, and cleanup.

This package is an SDK contract, not an injector or a live transport. See the
[Runtime Host guide](https://github.com/loonghao/GameLearningRuntime/blob/main/docs/guides/runtime-host-and-provider-sdks.md)
for the current capability boundary and integration workflow.
