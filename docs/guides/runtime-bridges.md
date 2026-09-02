# Build a reusable runtime bridge

Use the bridge ports when the authorized runtime and the learner live in
different processes, languages, or thread-affinity domains. The shared core
standardizes lifecycle and identity; it does not select HTTP, pipes, gRPC, or a
game engine.

## Client side

Implement `BridgeDriver` around an authenticated local transport, then compose
it with the normal contract wrapper:

```python
from game_learning_runtime import BridgeEnvironment, ContractEnvironment

driver = make_project_driver()
environment = ContractEnvironment(
    BridgeEnvironment(
        driver,
        required_capabilities={
            "authenticated",
            "target-bound",
            "postcondition-verified",
        },
    )
)
```

A driver implements five typed methods: `describe`, `reset`, `attach`, `step`,
and `close`. Drivers that advertise realtime control additionally expose
`lease` and `cancel`; `describe` carries an optional
`glr.realtime-control.v1` timing descriptor. `describe` authenticates and returns `EnvironmentSpec`; reset,
attach, and step return `TimeStep`. The bridge supplies immutable request
objects and includes the current `episode_id` plus expected next step with
every action. A driver must not implement attach by claiming a physical reset.

Realtime steps use bounded `deadline_ns`, `quantum_ns`, and optional `hold_ns`
values. A typed receipt distinguishes `consumed`, `expired`, `cancelled`, and
`rejected`; expired or cancelled actions are never retried implicitly. Input
lease `acquire`, `renew`, `release`, and `preempt` operations carry the same
`session_id` and `target_id` as the step. A stale or preempted token is rejected
before provider mutation.

Remote metadata is empty by default. Export only stable, non-sensitive keys:

```python
BridgeEnvironment(driver, metadata_allowlist={"runtime_family"})
```

Do not allowlist local paths, hostnames, account IDs, process/window IDs,
tokens, build installations, or raw diagnostic payloads.

## Server side

A Python transport server can delegate decoded calls to the shared kernel:

```python
from game_learning_runtime import EnvironmentBridgeDriver

kernel = EnvironmentBridgeDriver(make_authorized_environment())
descriptor = kernel.describe()
initial = kernel.reset(decoded_reset_request)
following = kernel.step(decoded_step_request)
```

The kernel serializes calls, validates the underlying environment, and rejects
an incorrect episode or step before execution. Native runtimes implement the
same `glr.v1` fields and perform engine API work on the engine-owned main
thread.

## Transport responsibilities

The concrete driver/server pair must provide all of the following for live
control:

- loopback-only or OS-local IPC plus an authenticated handshake;
- exact runtime binding that is revalidated before mutation;
- bounded frame/body size, request deadline, pending queue, and work per frame;
- typed capability allowlisting and structured, sanitized errors;
- action acknowledgement from an authoritative post-state;
- release of owned input or leases when the client closes or its heartbeat
  expires.

Never automatically retry a timed-out action. The action may have committed
even when its response was lost. Reconcile it through authoritative state and
the episode/step cursor, or truncate/fail the episode.

Read-only observers, caches, telemetry, and advisory tools belong on a separate
path. They can enrich an observation but cannot expand an action mask, grant a
capability, prove a mutation, or replace the authoritative runtime receipt.

## What stays in each adapter

Keep these concerns outside GLR:

- engine/plugin loading and main-thread scheduling;
- observation encoding and bounded entity selection;
- semantic action names, parameters, masks, and postconditions;
- authoritative reward signals and terminal/truncation interpretation;
- reset, snapshot, or live-attach policy;
- deployment authorization and exact-build acceptance.

This division lets multiple game categories reuse the same bridge contract
without publishing proprietary state, local machine details, or a universal
unsafe control surface.

## Rust boundary

Rust is a good candidate for framing, checksums, shared-memory rings, bounded
MPMC actor queues, and native language SDKs. Adopt it only when a representative
paired benchmark reaches ADR-0004's threshold: at least 2x throughput or at
least 30% lower p95 latency. Game semantics and learner objectives remain out
of the native transport layer.

Configure source authority and reward composition separately using
[`glr.training.v1`](knowledge-and-rewards.md). Web research and cached strategy
knowledge can guide the policy, but cannot acknowledge a bridge action or
satisfy an authoritative reward term.

### Optional action receipts

Realtime providers that advertise `action-receipt-v1` may attach an
`action_receipt` to each post-action time step. The receipt is bound to the
returned `episode_id` and `step_id` and reports one of `accepted`, `rejected`,
`unknown`, `no_effect`, `partial`, or `blocked`, with bounded timestamps and an
optional progress delta. `unknown` is preserved across transport failures and
is never retried implicitly. Collectors expose typed counts through
`Unroll.action_outcome_counts`; adapters that do not advertise the capability
remain wire-compatible and return no receipt.

### Command refusal funnel

Adapters should use one `RefusalFunnel` for both refusal forms: a command may
raise `CommandRefusal`, or it may return a `TimeStep` whose receipt is
`rejected`/`blocked`. The funnel preserves the action identity, provider
`target_id`, and `transient`/`structural` reason class, then invokes the
configured handler once per action. Exceptions are still raised after they are
reported, so callers cannot accidentally treat a refused command as success.
The refusal fields are optional on receipts to keep older providers
wire-compatible; new refusals should always populate them.

Durable multi-step workflows must make resume explicit. The CLI transaction
commands persist the step cursor and refusal history:

```text
glr --json transaction begin --run-id RUN --transaction-id TX --steps steps.json
glr --json transaction resume --transaction-id TX --refusal refusal.json
```

Structural refusals never advance the cursor and consume a bounded resume
budget. When the budget is exhausted the transaction becomes `abandoned`; the
`transaction.resume` envelope reports that terminal outcome in
`glr.cli-output.v1` and the CLI exits with code `77`. There is no implicit
retry. A resume without a refusal is an explicit acknowledgement that advances
one step (or completes the transaction).
