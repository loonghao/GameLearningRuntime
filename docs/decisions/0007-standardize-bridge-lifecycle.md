# ADR-0007: Standardize bridge lifecycle, not game transports

## Status

Accepted

## Context

Existing authorized game integrations use several valid local bridge shapes:
loopback HTTP with a runtime-main-thread queue, length-prefixed loopback
sockets, process-bound named pipes, read-only observer feeds, and direct
in-process adapters. They independently repeat protocol negotiation, reset or
attach handling, action sequence checks, post-state receipts, close behavior,
and metadata filtering.

The shared layer must support turn-based and real-time environments without
embedding engine APIs, game vocabulary, input automation, or learning
algorithms. For live control it must fail closed on stale actions, transport
ambiguity, target drift, and missing capabilities. It must remain NumPy-only;
adding an async or native runtime dependency requires measured data-plane value.

## Decision

Provide two transport-neutral Python boundaries:

- `BridgeEnvironment` adapts a `BridgeDriver` to `GameEnvironment`. It verifies
  the protocol version and caller-required capabilities, denies remote metadata
  unless explicitly allowlisted, copies reset/action requests, sends the
  current episode and expected next step, rejects stale receipts, and never
  retries a mutating action.
- `EnvironmentBridgeDriver` is the server-side lifecycle kernel for a local
  `GameEnvironment`. It serializes calls, reuses `ContractEnvironment`, and
  rejects mismatched episode or step identities before an action reaches the
  adapter.
- `BridgeEnvironment.resume` and the optional `reconnect-resume-v1` capability
  provide a read-only reconnect handshake. Providers return an authoritative
  cursor and may include an `ActionReconciliation` verdict for a lost in-flight
  action. Resume never retries a mutation and rejects stale episode/cursor
  results.

Concrete transports remain separate integrations. They must map the existing
`glr.v1` Describe/Reset/Step contract and own loopback or local-IPC restriction,
authentication, exact target binding, payload/deadline/queue bounds, structured
errors, and runtime-main-thread dispatch. Live consumers should require
capabilities such as `authenticated`, `target-bound`, and
`postcondition-verified` according to their threat model.

Game adapters continue to own observation encoding, action vocabulary and
masks, reward semantics, reset versus live-attach policy, authoritative
postconditions, and engine-specific execution. Read-only auxiliary observers
may enrich an observation but cannot authorize or acknowledge an action.

No arbitrary reflection, script evaluation, generic click, stealth injection,
or anti-cheat bypass enters the shared contract.

## Consequences

### Positive

- HTTP, sockets, pipes, gRPC, tests, and future native drivers share one
  learner-facing API and one episode/step fencing model.
- Game packages remove repeated lifecycle code without moving proprietary data
  or engine dependencies into GLR.
- Lost mutating responses cannot be hidden by an unsafe automatic retry.
- Security and privacy requirements are explicit at the driver boundary.

### Negative

- Existing adapters still need a small driver that converts their raw payloads
  to `EnvironmentSpec` and `TimeStep`.
- Native C#/C++/Rust runtimes cannot reuse Python code directly; they implement
  the equivalent versioned protocol until generated SDKs are available.
- Authenticated target-bound reconnect transports remain a future decision;
  the current resume contract is transport-neutral and opt-in.

### Neutral

- Reset and live attach remain separate operations and capabilities; neither
  operation may silently stand in for the other.
- Concrete transports can evolve independently without changing learner code.

## Alternatives Considered

**Choose loopback HTTP as the universal transport.** Rejected because native
pipes and framed sockets provide useful identity, deployment, and performance
properties, while some official APIs are already in process.

**Copy one existing bridge implementation into core.** Rejected because it
would import game-specific assumptions and preserve unsafe or obsolete seams.

**Expose arbitrary reflection or script execution for flexibility.** Rejected
because it defeats capability allowlisting, stable schemas, and authorization.

**Rewrite all transports in Rust immediately.** Rejected because current
benchmarks have not met ADR-0004's adoption threshold. Framing, shared memory,
or actor queues remain Rust candidates only after paired measurement.

## References

- `src/game_learning_runtime/protocol/glr/v1/runtime.proto`
- ADR-0001, ADR-0002, and ADR-0004
