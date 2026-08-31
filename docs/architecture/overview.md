# Architecture overview

## Requirements

GLR must serve reinforcement learning, imitation learning, supervised/offline
learning, agents, evaluation, and QA without coupling a game adapter to any of
them. It must span turn-based, real-time, discrete, continuous, hybrid,
hierarchical, and eventually multi-agent environments.

Non-functional priorities are correctness before throughput, deterministic
identity and ordering, optional heavy dependencies, portable records, semantic
versioning, and low adoption cost for local uv projects.

## Components

```text
┌─────────────────────────────────────────────────────────────┐
│ Learners and consumers                                      │
│ TorchRL │ custom PPO/IMPALA │ BC/offline │ evaluation/QA   │
└───────────────────────────┬─────────────────────────────────┘
                            │ Tensor trees / Unroll / JSONL
┌───────────────────────────▼─────────────────────────────────┐
│ GLR application layer                                       │
│ SyncCollector │ recorder/replay │ framework adapters         │
│ optional BC/PPO/GAE/V-trace objective primitives            │
│ TrainingConfig │ RewardComposer │ knowledge source policy     │
│ privacy-safe adapter conformance runner                      │
│ BridgeEnvironment (client) │ EnvironmentBridgeDriver (server)│
├─────────────────────────────────────────────────────────────┤
│ GLR domain and ports                                         │
│ Specs │ TimeStep │ Transition │ GameEnvironment │ BridgeDriver│
│ ContractEnvironment (fail-closed lifecycle validation)       │
├─────────────────────────────────────────────────────────────┤
│ Protocol                                                     │
│ glr.v1 Protobuf: Describe │ Reset │ Step │ Interact stream  │
└───────────────────────────┬─────────────────────────────────┘
                            │ Adapter-specific transport
┌───────────────────────────▼─────────────────────────────────┐
│ Authorized runtime adapters                                 │
│ Unity/C# │ Unreal/C++ │ Native/Rust/C++ │ official APIs    │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
                          Game
```

The domain has no Torch, gRPC runtime, game-engine, or algorithm dependency.
Transport adapters implement `GameEnvironment`; learning integrations consume
it. Composite tensor trees express hybrid and hierarchical spaces while leaves
remain easy to map to NumPy, Torch, C#, C++, or Rust tensors.

`BridgeEnvironment` and `EnvironmentBridgeDriver` standardize both sides of a
transport without choosing its framing or runtime. The client verifies
protocol/capability negotiation, copies requests, fences episode/step identity,
and denies metadata by default. The server serializes requests and reuses
`ContractEnvironment`. A concrete driver still owns authentication, exact
runtime binding, deadlines, bounded frames/queues, error mapping, and the
engine's main-thread dispatcher.

Existing Gymnasium environments enter through a deny-by-default compatibility
adapter. Native Rust is reserved for benchmark-proven protocol, storage, and
actor data-plane work; it does not create a second learner or game-semantics
layer. Optional PyTorch objectives consume learner-side tensors but contain no
model, optimizer, collector, reward shaping, or game-specific state.

`TrainingConfig` is a versioned data-only policy, not a dependency-injection
container. It assigns knowledge sources an advisory or authoritative role,
bounds freshness and payload size, selects reset versus attach collection, and
declares named reward contributions. Adapters still own data acquisition and
game semantics. `RewardComposer` accepts scalar signals from reviewed code,
validates their declared sources, applies clipping and weights, and returns an
immutable breakdown without evaluating expressions or importing callbacks.

The testing integration composes the same contract wrapper and collector used
by production code. It returns aggregate counts only; observations, actions,
environment IDs, metadata, paths, and timestamps never enter its report.

## Failure modes

| Failure | Current behavior | Next scaling step |
|---|---|---|
| Wrong dtype/shape/bounds | Contract wrapper rejects the boundary | Structured remote error codes |
| Missing/extra action key | Contract wrapper rejects the action | Schema negotiation tooling |
| Step before reset/after terminal | Contract and bridge wrappers fail closed | Attach/reset policy profiles |
| Stale episode or skipped step | Client and server bridge fencing reject it | Explicit resume protocol |
| Dataset corruption | Reader reports the exact record line | Checksummed chunk containers |
| Slow actor/backpressure | Fixed synchronous unroll only | Bounded async queues and metrics |
| Transport loss during action | No implicit retry; driver reports/reconciles outcome | Reconnect/resume protocol ADR |
| Stale or conflicting guide claim | Advisory only; preserve provenance and verify at runtime | Automated source refresh policy |
| Missing/wrong reward source | Composer fails closed before returning a reward | Multi-agent/vector reward policy |

## Security boundary

GLR transports observations and actions; it does not authorize access to a
game. Adapters must enforce the operator's legal and technical authorization,
bind to the intended runtime, avoid secret/proprietary data in records, and
default to local authenticated transports. Network exposure and remote
authorization are deferred until a threat model and dedicated ADR exist.
Public gameplay research contains paraphrased claims and public URLs only. It
cannot grant runtime authority, disclose local connection details, or replace
post-action verification.
