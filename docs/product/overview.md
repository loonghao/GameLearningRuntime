# Product overview

## Problem

Game-learning projects repeatedly rebuild runtime observation, action, reset,
recording, and training bridges. Those bridges often couple one game, one
language, one transport, and one algorithm, making data and infrastructure hard
to reuse.

## Users

### Runtime adapter developer

Builds an authorized Unity, Unreal, native, emulator, or official-API adapter.
Source-integrated engine plugins and binary-only external attachments declare
separate machine-checkable profiles while sharing one environment contract.
They need one precise lifecycle and tensor contract, independent of training
code.

### Learning engineer

Builds TorchRL or custom PPO, IMPALA, BC, DAgger, offline learning, or world
model pipelines. They need stable tensors, masks, events, and replay data
without game-engine dependencies.

### Evaluation and QA engineer

Uses the same adapter for deterministic scenarios, regression metrics, dataset
capture, and automated gameplay checks without importing learner internals.

## Current requirements

- Express nested observations and hybrid/hierarchical actions.
- Make action masks and terminal/truncation semantics explicit.
- Fail closed on dtype, shape, bound, identity, and lifecycle violations.
- Collect algorithm-neutral transitions and fixed-length actor unrolls.
- Store portable versioned transition records.
- Package a versioned cross-language runtime protocol.
- Offer TorchRL as an optional integration, not a core dependency.
- Provide version-pinned Git and GitHub Release reuse.

## Non-goals for v0.1

- Shipping production game-specific adapters.
- Choosing or implementing a canonical PPO, IMPALA, or BC learner.
- Hiding instrumentation or bypassing anti-cheat/security controls.
- Claiming distributed transport, shared memory, or generated SDKs are ready.

## Success criteria

- Core tests pass on Python 3.10 through 3.13.
- The packaged Protobuf schema compiles.
- TorchRL's `check_env_specs` accepts the reference adapter.
- A local project can install a release tag and call the reusable CI workflow.
