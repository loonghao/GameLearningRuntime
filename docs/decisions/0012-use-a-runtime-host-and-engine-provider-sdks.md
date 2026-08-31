# ADR-0012: Use a Runtime Host and engine provider SDKs

## Status

Accepted

## Context

Unity and Unreal adapters currently repeat lifecycle fencing, framing, queue
limits, training-client conversion, and deployment logic. Source-owned plugins,
authorized BepInEx plugins, and authorized UE4SS mods still need different
engine bootstraps. A framework-owned universal injector would have to reproduce
engine version detection, managed/native runtime startup, main-thread and
object-lifetime rules, reload cleanup, platform hardening, and mod-policy
compatibility. It would also create a larger security and maintenance boundary
than GLR's learner-neutral environment contract requires.

The shared layer needs to serve Python/TorchRL and custom learners without
moving game semantics or authorization into Rust. It must remain useful to
Unity C# and Unreal C++ projects, and it must not claim authenticated or
target-bound IPC before those controls are implemented and verified.

## Decision

Build a framework-owned `GLR Runtime Host + Engine Provider SDK` boundary:

- `glr-hostd` is a Rust process whose domain layer owns `glr.host.v1` envelopes,
  strict tensor bytes, bounded frames, serialized request handling, provider
  negotiation, lifecycle state, and episode/step fencing.
- The first concrete transport is one explicitly launched JSON-lines stdio
  session. It has a 1 MiB hard frame bound and no automatic mutating retry. It
  advertises `host-stdio`, but not `authenticated` or `target-bound`.
- The first built-in provider is `synthetic-counter`, used only for
  cross-language conformance, packaging, and training-plumbing smoke tests.
- Python `HostBridgeDriver` maps the host to the existing `BridgeDriver` port.
  `HostProcessConfig` requires an existing absolute executable path, starts no
  shell, bounds both directions, enforces a response deadline, and owns child
  cleanup.
- The C# `GameLearningRuntime.Provider` contract targets .NET Standard 2.0 so a
  reviewed Unity plugin or BepInEx 5 Unity Mono bootstrap can implement it.
- The header-only C++20 provider contract gives an Unreal runtime module or
  reviewed native bootstrap the same descriptor, tensor, event, reset/attach,
  step, and close vocabulary.
- Engine bootstraps remain separate and minimal. Source projects use official
  Unity/Unreal plugin mechanisms. Authorized no-source projects may use
  BepInEx, UE4SS, or an official mod SDK. GLR does not discover, inject into, or
  patch a game process.
- Reward composition, `EpisodeRewardGuard`, `DemonstrationGate`, collectors,
  TorchRL, PPO/IMPALA/BC objectives, and model bundles stay in Python outside
  the Runtime Host.

An authenticated, target-bound named-pipe/Unix-socket transport and external
engine-provider connection remain a later threat-modeled increment. Shared
memory and an in-process Rust library remain benchmark-gated by ADR-0004.

## Non-functional requirements

- **Correctness:** unknown operations and fields fail closed; reset/attach must
  return a fresh step-zero episode; steps are checked before and after provider
  execution; termination and truncation cannot both hold for one participant.
- **Security:** no process scanning, arbitrary library path, arbitrary script,
  reflection, injection, network listener, or game-directory mutation enters
  the host. The stdio client launches only an explicit absolute executable.
- **Reliability:** mutating requests are never retried; child process cleanup is
  bounded; provider post-state must match the active episode and next step.
- **Performance:** the current transport is deliberately serialized. Promotion
  to shared memory, asynchronous queues, or in-process FFI requires ADR-0004's
  paired benchmark and parity fixtures.
- **Portability:** Rust 1.98.0, rustup 1.29.0, .NET SDK 10.0.400, C# .NET Standard
  2.0, and C++20 checks are pinned or declared through the project toolchain.
- **Privacy:** the conformance provider and release smoke emit aggregate counts
  only; public artifacts contain no machine paths, host/process identity,
  accounts, observations from real games, or proprietary data.
- **Operability:** `vx just rust-check`, `vx just provider-sdk-check`, and
  `vx just host-smoke` are the local and CI gates. Release Please publishes
  checksummed native conformance-host archives and the C# contract package.

## Failure modes and mitigation

- **Stale or duplicate action:** reject episode/step mismatch before calling the
  provider; never replay an ambiguous mutation.
- **Malformed or oversized tensor:** validate dtype, shape product, base64, byte
  length, and bool representation within the frame bound.
- **Provider contract drift:** reject stale post-state and run Rust, Python, C#,
  and C++ smoke gates against `glr.host.v1`.
- **Host hang or crash:** the Python channel returns a deadline/EOF error,
  fail-closes the ambiguous session, and terminates only its owned child.
- **Bootstrap mismatch:** keep official/BepInEx/UE4SS version compatibility in
  the adapter package and require a separate authorized live acceptance gate.
- **Capability overclaim:** the stdio descriptor intentionally lacks
  `authenticated` and `target-bound`, so loader/source profiles requiring them
  cannot connect through this transport yet.

## Consequences

### Positive

- Python learners, Rust lifecycle code, Unity C#, and Unreal C++ share one
  provider vocabulary without importing an RL algorithm into the runtime.
- Common fencing, framing, packaging, and client conversion stop being copied
  into each game adapter.
- A synthetic end-to-end process proves training plumbing while live engine
  acceptance remains an explicit separate boundary.

### Negative

- The first host cannot yet connect to a live external C#/C++ provider.
- Real integrations still need a small authorized engine bootstrap and
  game-semantic provider implementation.
- Multiple release artifacts and SDK checks add CI and packaging cost.

### Neutral

- Existing BepInEx and UE4SS templates remain supported lanes; they become
  bootstrap choices rather than the GLR learner protocol.
- `glr-hostd` release binaries are conformance assets until a target-bound local
  provider transport is implemented.

## Alternatives considered

**Replace BepInEx and UE4SS with one universal loader.** Rejected because it
would duplicate engine startup/injection compatibility, expand the security
boundary, and still require separate Unity and Unreal semantic shims.

**Embed Rust directly into every game.** Deferred because crash isolation,
unload behavior, engine ABI compatibility, and FFI copies need separate live
evidence and performance measurements.

**Keep independent C#, C++, and Python bridges.** Rejected because lifecycle,
fencing, packaging, and training-side conversion would continue to drift.

**Start with authenticated network RPC.** Rejected for the first slice because
network exposure and remote authorization require a dedicated threat model;
stdio provides a smaller truthful conformance boundary.

## References

- ADR-0004, ADR-0007, ADR-0009, ADR-0010, and ADR-0011
- `crates/glr-host`
- `src/game_learning_runtime/host.py`
- `sdk/csharp/GameLearningRuntime.Provider`
- `sdk/cpp/include/glr/provider.hpp`
