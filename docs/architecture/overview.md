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
│ SyncCollector │ recorder/replay │ Gymnasium/TorchRL adapters│
├─────────────────────────────────────────────────────────────┤
│ GLR domain and ports                                         │
│ Specs │ TimeStep │ Transition │ GameEnvironment             │
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

Existing Gymnasium environments enter through a deny-by-default compatibility
adapter. Native Rust is reserved for benchmark-proven protocol, storage, and
actor data-plane work; it does not create a second learner or game-semantics
layer.

## Failure modes

| Failure | Current behavior | Next scaling step |
|---|---|---|
| Wrong dtype/shape/bounds | Contract wrapper rejects the boundary | Structured remote error codes |
| Missing/extra action key | Contract wrapper rejects the action | Schema negotiation tooling |
| Step before reset/after terminal | Contract wrapper fails closed | Remote session fencing |
| Stale episode or skipped step | Episode/step identity is rejected | Lease and retry semantics |
| Dataset corruption | Reader reports the exact record line | Checksummed chunk containers |
| Slow actor/backpressure | Fixed synchronous unroll only | Bounded async queues and metrics |
| Transport loss | Adapter owns recovery | Reconnect/resume protocol ADR |

## Security boundary

GLR transports observations and actions; it does not authorize access to a
game. Adapters must enforce the operator's legal and technical authorization,
bind to the intended runtime, avoid secret/proprietary data in records, and
default to local authenticated transports. Network exposure and remote
authorization are deferred until a threat model and dedicated ADR exist.
