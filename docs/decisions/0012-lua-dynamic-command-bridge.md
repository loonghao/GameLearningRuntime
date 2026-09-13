# 0012: Engine-neutral Lua dynamic command bridge

## Status

Accepted for implementation.

## Decision

GameLearningRuntime owns a versioned Lua command registry. Unity and Unreal
adapters embed a small engine host which exposes only reviewed primitives
(`observe`, game-thread dispatch, bounded object lookup, and typed receipts).
Lua modules register commands at runtime with a name, schema, preconditions,
capability class, and receipt schema. The GLR runtime publishes the resulting
registry as part of the negotiated adapter capabilities and reloads modules
atomically when their content hash changes.

The registry is data-driven and learner-neutral. A command is not available
until the engine host acknowledges its implementation and the runtime has
validated its schema. Requests carry a registry generation and state sequence;
stale generations are rejected and the caller must renegotiate.

## Lifecycle

1. The adapter host starts with an empty deny-by-default registry.
2. Lua modules are loaded from the adapter's signed command directory.
3. Each module registers typed commands; duplicate names or invalid schemas
   reject the complete reload and leave the previous generation active.
4. The host publishes `lua_registry_generation`, module hashes, and command
   capabilities through the normal GLR handshake.
5. A reload swaps the immutable registry between game-thread ticks. In-flight
   requests finish against their captured generation.

## Safety boundary

Lua cannot evaluate arbitrary source from a request, invoke reflection, or
write raw UE/Unity properties. Commands dispatch through host primitives and
must return bounded, auditable receipts. Authentication, PID binding, timeout,
monotonic request IDs, and terminal fencing remain in the native/C# control
transport. Hot reload changes command definitions only; it never changes the
transport security contract.

## Operational effect

Adding a shrine menu action, shop operation, chest interaction, or collection
command becomes a Lua package update and registry reload. Recompilation and a
game restart remain necessary only when the engine host ABI or transport
implementation changes.
