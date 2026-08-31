# ADR-0009: Profile engine-plugin and external-attach integrations

## Status

Accepted

## Context

Unity and Unreal games reach GLR through two materially different integration
boundaries. A project with source access can host a native engine adapter,
observe semantic state on the engine thread, control the simulation clock, and
implement a physical reset. A binary-only integration normally observes an
already-running process through an official API, telemetry, replay data, or a
rendered output and applies actions through an authorized bounded interface.

Treating both boundaries as a generic remote environment hides important
differences in lifecycle truth, observation authority, target identity, action
ownership, and attainable training throughput. Creating separate environment
or learner APIs for each engine would instead duplicate GLR's tensor, episode,
collector, and bridge contracts.

The design must remain learner-neutral, NumPy-only at the core, safe for public
examples, and usable from C#, C++, Rust, or Python. Source possession does not
grant runtime authorization, and lack of source does not authorize hidden-state
access, injection, or anti-cheat bypasses.

## Decision

Keep one `GameEnvironment`, `EnvironmentSpec`, `TimeStep`, and `glr.v1` bridge
contract. Add `glr.runtime-integration.v1` as a strict deployment profile with:

- engine family;
- `engine-plugin` or `external-attach` integration mode;
- `reset` or `attach` start mode;
- manual-step, time-scaled, or real-time clock ownership;
- engine-state, official-API, or rendered observation authority;
- native, official-API, or bounded-input action authority;
- in-process, local-IPC, or official-API deployment shape; and
- a truthful deterministic-seed claim.

`RuntimeIntegrationProfile` derives the capabilities a bridge must prove before
connection. The recommended source profile uses an engine plugin, deterministic
reset, manual stepping, semantic engine state, native actions, main-thread
dispatch, and authenticated target-bound local IPC. The recommended binary-only
profile uses a live attachment, real-time clock, rendered observations, bounded
input with an owned input lease, exact target binding, and verified post-state.
An external profile may use official observations and actions instead when a
documented API provides them.

Unity and Unreal adapter packages remain outside the core distribution. They
map engine callbacks and optional ML integrations onto the same GLR contract.
Concrete transports also remain separate; the profile describes required
properties rather than choosing a gRPC, socket, pipe, or shared-memory library.

Use the repository-owned scaffold to generate the profile, training policy,
synthetic conformance seam, and a pinned vx/just development entrypoint. A
synthetic seam proves contract behavior only, never live-game acceptance.

## Non-functional requirements

- **Correctness:** reject false reset, clock, state-authority, or action claims
  before training starts.
- **Performance:** source integrations should support manual stepping and
  multiple isolated game instances; transport acceleration remains benchmark
  gated.
- **Security:** external control requires exact target binding, bounded actions,
  input lease cleanup, and post-action readback.
- **Portability:** profiles contain no endpoint, path, PID, window identifier,
  credential, account, or proprietary game value.
- **Maintainability:** engine APIs and transports remain optional adapters while
  learner and dataset code depend only on GLR.

## Consequences

### Positive

- Unity and Unreal projects share one learner-facing runtime contract.
- New adapters get an explicit, machine-checkable source or binary-only lane.
- Unsafe lifecycle or authority mismatches fail during connection instead of
  silently degrading training data.
- Local development and CI can execute the same vx/just recipes.

### Negative

- Native C# and C++ adapter SDKs still need to be generated and implemented.
- Binary-only environments usually have lower observability, weaker reset
  semantics, and real-time throughput limits.
- Existing bridges must add explicit capability claims before using a profile.

### Neutral

- Unity ML-Agents and Unreal Learning Agents can accelerate integration but do
  not become required learner or runtime dependencies.
- A game with an official binary extension SDK may choose the engine-plugin
  lane even when the full game source is unavailable, provided its capabilities
  are truthful.

## Alternatives Considered

**Create separate Unity and Unreal environment APIs.** Rejected because engine
callbacks differ but observation, action, lifecycle, and collection semantics do
not need to fork.

**Treat binary-only attachment as reset.** Rejected because menu navigation,
respawn, or save loading does not prove a deterministic physical reset.

**Require pixels for every binary-only integration.** Rejected because official
telemetry and control APIs provide more stable and authoritative seams when
available.

**Adopt shared memory or Rust for every source adapter immediately.** Rejected
until a paired benchmark meets ADR-0004's promotion threshold.

## References

- [Unity: Designing a Learning Environment](https://unity-technologies.github.io/ml-agents/Learning-Environment-Design/)
- [Unity: Agents](https://unity-technologies.github.io/ml-agents/Learning-Environment-Design-Agents/)
- [Unreal Engine 5.8: Learning Agents](https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/LearningAgents)
- [Unreal Engine 5.8: Learning Agents Training](https://dev.epicgames.com/documentation/unreal-engine/API/Plugins/LearningAgentsTraining)
- ADR-0004, ADR-0006, and ADR-0007
