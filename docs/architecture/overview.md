# Architecture overview

## Requirements

GLR must serve reinforcement learning, imitation learning, supervised/offline
learning, agents, evaluation, and QA without coupling a game adapter to any of
them. It must span turn-based, real-time, discrete, continuous, hybrid,
hierarchical, and eventually multi-agent environments.

Non-functional priorities are correctness before throughput, deterministic
identity and ordering, optional heavy dependencies, portable records, semantic
versioning, and low adoption cost for local vx/uv projects.

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
│ RuntimeIntegrationProfile: engine │ loader │ external         │
│ ModelBundleManifest: inputs │ seeds │ artifacts │ SHA-256       │
│ Agent CLI │ goal loop │ run store │ indexed review capture    │
│ privacy-safe adapter conformance runner                      │
│ BridgeEnvironment (client) │ EnvironmentBridgeDriver (server)│
├─────────────────────────────────────────────────────────────┤
│ GLR domain and ports                                         │
│ Specs │ TimeStep │ Transition │ GameEnvironment │ BridgeDriver│
│ ContractEnvironment (fail-closed lifecycle validation)       │
├─────────────────────────────────────────────────────────────┤
│ Protocol                                                     │
│ glr.v1 environment │ glr.host.v1 bounded host envelope       │
├─────────────────────────────────────────────────────────────┤
│ Runtime Host and provider SDKs                               │
│ Rust hostd │ Python driver │ Unity C# │ Unreal C++           │
└───────────────────────────┬─────────────────────────────────┘
                            │ Adapter-specific transport
┌───────────────────────────▼─────────────────────────────────┐
│ Authorized runtime adapters                                 │
│ Engine plugin       │ Loader plugin         │ External attach │
│ Unity/C# │ Unreal/C++ │ BepInEx │ UE4SS     │ API │ rendered  │
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

The first concrete Runtime Host composes that client boundary as
`BridgeEnvironment -> HostBridgeDriver -> glr-hostd`. Rust owns strict
`glr.host.v1` JSON-lines framing, a 1 MiB hard bound, serialized lifecycle, and
episode/step fencing. The built-in `synthetic-counter` provider proves the
cross-process training seam. C# .NET Standard 2.0 and header-only C++20
contracts define the engine-provider vocabulary without importing Unity,
Unreal, BepInEx, or UE4SS into core. The first stdio transport truthfully omits
`authenticated` and `target-bound`; live engine-provider IPC is still future
work.

`RuntimeIntegrationProfile` selects a truthful deployment boundary before the
bridge connects. Source-integrated Unity and Unreal plugins normally prove
physical reset, controllable time, semantic state, native actions, and
main-thread dispatch. Authorized loader plugins prove live attach, semantic
state, bounded commands, loader/version provenance, main-thread dispatch, and
post-action readback without claiming source-owned reset or clock control.
External adapters normally prove live attach, exact target binding, real-time
operation, bounded input ownership, and post-action readback. All profiles
reuse the same learner-facing environment contract.

`ModelBundleManifest` packages model artifacts with copied training/runtime
configuration, source and lock inputs, seeds, algorithm/framework versions,
and per-file SHA-256. It contains portable relative paths only and fails closed
on linked, missing, resized, or modified entries. It captures a reproduction
environment without claiming hardware-level determinism or model quality.

The local `glr` control plane composes project-owned roles without importing
their implementations. `glr.project.v1` binds fixed argv commands to one
project-relative bridge and exact environment/protocol identity. The SQLite run
store is a query projection for lifecycle, scalar metrics, artifacts, spatial
knowledge, and cited research; tensors, transitions, videos, and model bytes
remain checksummed artifacts. The bounded goal loop can refresh research and
adjust declarative reward terms between trials, but only persisted authoritative
runtime metrics can satisfy its machine-readable criteria.

Concurrent capture is also a port: a project recorder owns OS/window capture
and emits H.264 plus a step/frame index. GLR validates and hashes those files;
it does not choose a platform capture API or label policy output as expert data.
Spatial imports require exact environment/protocol identity and become advisory
until re-observed. Only cited family-scoped findings can cross game boundaries.

Existing Gymnasium environments enter through a deny-by-default compatibility
adapter. Native Rust now owns the bounded Runtime Host lifecycle/framing slice
and remains reserved for benchmark-proven storage and actor data-plane work; it
does not create a second learner or game-semantics layer. Optional PyTorch
objectives consume learner-side tensors but contain no
model, optimizer, collector, reward shaping, or game-specific state.

`TrainingConfig` is a versioned data-only policy, not a dependency-injection
container. It assigns knowledge sources an advisory or authoritative role,
bounds freshness and payload size, selects reset versus attach collection, and
declares named reward contributions. Adapters still own data acquisition and
game semantics. `RewardComposer` accepts scalar signals from reviewed code,
validates their declared sources, applies clipping and weights, and returns an
immutable breakdown without evaluating expressions or importing callbacks.
`EpisodeRewardGuard` then enforces per-step and per-episode positive shaping
budgets plus terminal failure dominance. `DemonstrationGate` rejects
unapproved origin/outcome pairs before BC ingestion. A
`DemonstrationArtifactManifest` binds the accepted provenance to the exact
`glr.transition.v1` bytes, environment, and episode. These policies remain
learner-neutral JSON inputs and are included in model bundles.

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
| Dense shaping overwhelms failure | Episode budget plus terminal failed-return ceiling | Game-specific reward ablation |
| Policy output relabelled as expert | BC provenance/outcome allowlist fails closed | Deliberate distillation policy |
| Loader mismatch or unknown action | Exact loader tag plus empty-deny vocabulary | Version-specific live acceptance matrix |
| Model/config drift | Bundle verifier identifies the changed relative entry | Signed model and dataset attestations |
| Guide or video claim treated as success | Goal evaluator requires matching persisted authoritative metrics | Signed runtime evidence receipts |
| Stale imported world position | Import downgrades it to advisory and preserves source run | Runtime refresh/expiry policy |
| Capture video lacks step alignment | Required capture fails without a valid frame index | Hardware timestamp calibration |
| Oversized/stale Host request | Reject before provider execution; never retry mutation | Authenticated target-bound local IPC |
| Host/provider SDK schema drift | Rust, Python, C#, and C++ contract gates | Generated SDK codecs after schema stabilizes |

## Security boundary

GLR transports observations and actions; it does not authorize access to a
game. Adapters must enforce the operator's legal and technical authorization,
bind to the intended runtime, avoid secret/proprietary data in records, and
default to local authenticated transports. Network exposure and remote
authorization are deferred until a threat model and dedicated ADR exist.
Public gameplay research contains paraphrased claims and public URLs only. It
cannot grant runtime authority, disclose local connection details, or replace
post-action verification.

`glr-hostd` does not discover or inject into processes, accept arbitrary
provider library paths, listen on a network, or select a game installation. Its
published binaries currently contain only the synthetic conformance provider.
