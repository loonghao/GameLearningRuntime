# ADR-0010: Add authorized loader plugins and reproducible model bundles

## Status

Accepted

## Context

ADR-0009 separated source engine plugins from binary-only external attachment.
Authorized mod loaders create a third boundary: adapter code runs inside a game
without owning its source project. BepInEx provides this boundary for supported
Unity/.NET runtimes and UE4SS provides it for supported Unreal runtimes. These
loaders can expose semantic state and game-thread callbacks, but they do not
prove physical reset, deterministic seeding, universal game compatibility, or
permission to instrument a particular game.

Developers also need a fast Agent-facing workflow from scaffold to deployment
and training. A model artifact alone cannot reproduce a result when its runtime
profile, reward configuration, source, dependency lock, seeds, or framework
version are missing.

## Decision

Introduce `glr.runtime-integration.v2` with a backward-compatible v1 reader and
three integration modes:

- `engine-plugin` for source or official extension SDKs;
- `loader-plugin` for an explicitly authorized BepInEx or UE4SS host; and
- `external-attach` for an out-of-process authorized seam.

A loader profile always uses truthful `attach`, real-time clock ownership,
semantic engine-state observations, a reviewed bounded-command vocabulary,
authenticated target-bound local IPC, main-thread dispatch, bounded queues, and
verified post-state. BepInEx maps only to Unity and UE4SS maps only to Unreal.
The first generated hosts are BepInEx 5 LTS Unity Mono and UE4SS 3.x Lua.
IL2CPP and UE4SS C++ remain future, separately validated templates.

The adapter scaffold requires an exact upstream loader tag and emits:

- `agent-interface.json` with unknown operations denied and an empty action
  vocabulary;
- `AGENTS.md` with the authorized operating boundary;
- bounded BepInEx C# or UE4SS Lua host source;
- `deployment/loader.json` with upstream license, version, and relative layout;
- a packager that stages a checksummed payload but never discovers or writes a
  game directory; and
- a synthetic training smoke test plus `glr.model-bundle.v1` output.

`glr.model-bundle.v1` copies portable reproduction inputs and model artifacts
into one directory. Its manifest records environment/protocol identity,
algorithm, framework/version, seeds, relative paths, sizes, and SHA-256
digests. Verification fails closed on missing, linked, resized, or modified
files. Bundles contain no originating absolute paths.

## Non-functional requirements

- **Correctness:** v1 profiles remain readable; loader profiles cannot claim
  reset, manual clock, arbitrary actions, or an incompatible engine.
- **Security:** unknown operations/actions are denied, queue depth is bounded,
  mutation identity is fenced, and installation needs an explicit operator
  target.
- **Reproducibility:** model bundles carry config, source snapshots, lock files,
  seeds, versions, metrics, and content digests.
- **Portability:** manifests use portable relative paths and contain no host,
  process, account, credential, endpoint, or game-installation identifier.
- **Performance:** loaders dispatch on the game thread but transports and Rust
  acceleration remain benchmark gated by ADR-0004.
- **Operability:** vx/just expose the same check, train, reproduce, and package
  commands to developers and Agents.

## Failure modes and mitigation

- **Wrong loader/game build:** require and preserve an exact upstream tag, then
  gate live acceptance on the selected runtime.
- **Queue overload or stale actions:** reject before dispatch; never replay a
  mutation after an ambiguous response.
- **Failed postcondition:** retain the current step and reconcile through
  authoritative observation.
- **Artifact or environment drift:** model-bundle verification reports the
  changed relative entry before use.
- **Unsafe automated installation:** the packager stages only; an operator must
  select and approve the target separately.

## Consequences

### Positive

- No-source authorized adapters gain semantic engine access without forking the
  learner contract.
- New Agents receive one machine-readable operation surface and deterministic
  scaffold workflow.
- Loader packages and trained models can be reviewed, moved, and verified
  without leaking the source workstation.

### Negative

- Generated host code still needs game-specific semantic handlers and live
  acceptance.
- BepInEx and UE4SS compatibility remains game/version dependent.
- Reproduction captures software inputs but cannot guarantee identical GPU,
  driver, platform, or nondeterministic kernel behavior.

### Neutral

- GLR links to but does not vendor BepInEx or UE4SS binaries.
- The reference behavior-cloning smoke model proves plumbing, not useful game
  policy quality.

## Alternatives considered

**Treat loaders as external attachment.** Rejected because it hides semantic
in-process state and game-thread dispatch requirements.

**Treat loaders as source plugins.** Rejected because loaders cannot truthfully
claim source-owned reset, seed, clock, or build lifecycle.

**Automatically locate and install into games.** Rejected because discovery and
mutation would expand authority, risk the wrong target, and leak local paths.

**Store only model weights.** Rejected because weights without exact inputs,
versions, and integrity metadata are not a reproduction environment.

## References

- <https://github.com/BepInEx/BepInEx>
- <https://github.com/UE4SS-RE/RE-UE4SS>
- <https://docs.bepinex.dev/master/articles/dev_guide/plugin_tutorial/index.html>
- <https://docs.ue4ss.com/dev/guides/creating-a-lua-mod.html>
- ADR-0004, ADR-0006, ADR-0007, and ADR-0009
