# Protocol and data flow

## Environment handshake

1. A client calls `Describe` and obtains the environment ID, protocol version,
   tensor specs, masks, capabilities, and metadata.
2. The client decides whether it supports that exact contract.
3. `Reset` starts an episode and returns `step_id = 0`.
4. Every action carries the current `episode_id` and expected next step ID.
5. A time step returns observation, reward, terminated/truncated tensors,
   masks, events, and metadata.

```text
Client                         Runtime adapter
  │──── Describe ────────────────────▶│
  │◀─── EnvironmentDescriptor ────────│
  │──── Reset(seed/options) ─────────▶│
  │◀─── TimeStep(episode, step=0) ────│
  │──── Step(episode, expected=1) ───▶│
  │◀─── TimeStep(step=1) ─────────────│
```

`Interact` provides the same ordered step contract over a bidirectional stream
for future high-throughput actor adapters. It does not weaken identity or
ordering rules.

An adapter advertising `live-attach` may use `Attach` instead of `Reset`:

```text
Client                         Already-running runtime
  │──── Describe ────────────────────▶│
  │◀─── capabilities: live-attach ────│
  │──── Attach(options) ─────────────▶│
  │◀─── TimeStep(logical episode, 0) ─│
```

Attach establishes ordering and a fresh logical episode only. It does not
assert a physical reset, deterministic checkpoint, or seeded initial state.

The Python bridge ports implement the same unary lifecycle independently of a
specific transport:

```text
ContractEnvironment
  └─ BridgeEnvironment
       └─ BridgeDriver (HTTP / framed socket / named pipe / gRPC / native)
            └─ transport server
                 └─ EnvironmentBridgeDriver
                      └─ authorized runtime adapter / main-thread dispatcher
```

The implemented Runtime Host path is one concrete serialized driver:

```text
ContractEnvironment
  └─ BridgeEnvironment
       └─ HostBridgeDriver
            └─ glr-hostd stdio (glr.host.v1, <= 1 MiB)
                 └─ RuntimeProvider
                      └─ synthetic-counter (implemented conformance provider)
```

The C# and C++ provider SDKs define the next engine-facing port. They do not yet
connect a live provider to `glr-hostd`, so `host-stdio` cannot satisfy runtime
profiles that require authenticated and exact target-bound local IPC.

The client does not retry a failed `Step`. A lost mutating response can mean
the action happened, so a concrete driver must reconcile through an
authoritative readback or report an unknown outcome. Read-only health and
observation requests may use a separate retry policy.

## Knowledge and reward flow

```text
Public rules / guides ── paraphrased, cited ──▶ advisory research manifest
                                                     │
                                 versioned snapshot ──┼──▶ KnowledgeInjector
                                                     │        │
                                                     │        ▼
                                                     │   bounded KnowledgeContext
                                                     │        │ learner-owned encoding
Authoritative runtime telemetry ─────────────────────┼──▶ adapter signals
                                                     │
glr.training.v1 ── source authority / weights / clips┘
                                                     ▼
                                               RewardComposer
                                                     ▼
                                      total + immutable breakdown
```

Research suggests observations, actions, masks, and reward hypotheses. It does
not become runtime truth. The adapter emits named scalar signals from reviewed
code; the composer rejects undeclared, missing, non-finite, or wrong-source
signals. Reward terms require an authoritative source unless configuration
deliberately lowers that individual term to advisory authority.

The knowledge branch is separate from reward and action authority. A learner
queries `acquire`, `engage`, `upgrade`, and `avoid` items by stage and tags,
then encodes the immutable context alongside observations. Snapshot provenance,
freshness, payload bounds, and digest are checked before selection. The context
cannot alter adapter masks or acknowledge runtime effects.

## Agent-first goal flow

```text
glr.agent-goal.v1 + hard budgets
        │
        ├──▶ project researcher ──▶ cited research bundle (advisory)
        ├──▶ project planner    ──▶ bounded trial/reward plan
        ├──▶ project trainer    ──▶ transitions + persisted metrics
        │                              └── concurrent MP4 + step/frame index
        └──▶ project evaluator  ──▶ evidence matched to persisted metrics
                                             │
                  authoritative criteria met? ── yes: stop
                                             └── no: refresh/adjust within budget
```

The roles are fixed project commands and remain outside the learner-neutral
core. A research source can inform a plan but cannot acknowledge an action or
prove success. Evaluator claims must match the active run's stored metric name,
value, source, and authority. A trial, training-step, wall-clock, or unique
source budget cannot be expanded by a role.

The run database stores query projections and artifact hashes, not observation
tensors. Exact-environment entity/route snapshots can move to a fresh instance
only when environment and protocol match, and imports become advisory. Similar
games can reuse family-scoped research findings, never coordinates, action
authority, or model compatibility.

## Learning paths

- PPO consumes fixed-length `Unroll` values; the optional PyTorch integration
  provides masked clipped loss and truncation-aware GAE primitives.
- IMPALA actors attach `actor_id`, `sequence_id`, and `policy_version`. The
  optional `BoundedActorQueue` transports those unrolls through a bounded,
  learner-neutral lease/commit boundary. Its block, drop-oldest, and fail
  policies expose queue depth, actor lag, policy-version lag, and dropped or
  uncommitted unrolls without importing an async runtime. The synchronous
  collector remains the compatibility path.
- A live bridge may opt into `SyncCollector.collect(on_error="partial")` when a
  transient environment step fails. The returned unroll keeps all validated
  transitions gathered before the failure, marks its final transition as
  truncated for learner bootstrapping, and forces the next collection to reset;
  the default `on_error="raise"` behavior is unchanged.
- BC writes expert actions as ordinary transitions, preserving masks and next
  observations for later DAgger or offline RL; masked cross-entropy is reusable
  across project-specific policies.
- TorchRL maps GLR trees to `TensorDict` only at the integration boundary.

## Compatibility

The Protobuf package is `glr.v1` and the JSONL record schema is
`glr.transition.v1`. Additive fields may remain within v1. Removing fields,
changing meaning, or changing tensor encoding requires a new major schema.
Training policy uses `glr.training.v1`; the Runtime Host envelope uses
`glr.host.v1`. Unknown fields fail closed so spelling
mistakes do not silently change reward behavior. Gameplay research uses a
separate `glr.knowledge-research.v1` design manifest and is never loaded as an
executable runtime configuration. Agent orchestration adds strict
`glr.project.v1`, `glr.agent-goal.v1`, `glr.research-bundle.v1`,
`glr.trial-plan.v1`, `glr.goal-evidence.v1`, `glr.capture.v1`, and
`glr.spatial-knowledge.v1` and the optional `glr.spatial-knowledge.v2` directed graph contracts
without changing the runtime protocol.
