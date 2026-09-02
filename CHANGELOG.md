# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.10.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.9.0...v0.10.0) (2026-09-02)


### Features

* add action mask diagnostics ([24f122f](https://github.com/loonghao/GameLearningRuntime/commit/24f122f965a6e99807a9906f24d0804d0bf79ac6))
* add bounded actor queue telemetry ([#70](https://github.com/loonghao/GameLearningRuntime/issues/70)) ([46f2abd](https://github.com/loonghao/GameLearningRuntime/commit/46f2abdde710f357bc2ea6d43a568894828a35f8))
* add bounded realtime control contracts ([#68](https://github.com/loonghao/GameLearningRuntime/issues/68)) ([7979137](https://github.com/loonghao/GameLearningRuntime/commit/797913745c2a8d59296760836ca08960cdc3433c))
* add capture session lifecycle gates ([#69](https://github.com/loonghao/GameLearningRuntime/issues/69)) ([023ecbe](https://github.com/loonghao/GameLearningRuntime/commit/023ecbe2057688fb16fa795cb98e12bfb0c7125c))
* add checkpoint contract migration ([6e92a55](https://github.com/loonghao/GameLearningRuntime/commit/6e92a554a711ad5d9eb67412cb745d067f9393a1))
* add checkpoint promotion gate ([#52](https://github.com/loonghao/GameLearningRuntime/issues/52)) ([b8b9f21](https://github.com/loonghao/GameLearningRuntime/commit/b8b9f219fd7bd1b524cae58c2670d21ca5e269bb))
* add directed spatial traversability graph ([#67](https://github.com/loonghao/GameLearningRuntime/issues/67)) ([041fa70](https://github.com/loonghao/GameLearningRuntime/commit/041fa702de1e6b8ed9a6c004323b54a41bfbf514))
* add optional DeepSeek harness provider ([bd31f16](https://github.com/loonghao/GameLearningRuntime/commit/bd31f169ab0c7ec5910f3f9fed4a9e3e0fa2ea33))
* add realtime action receipts ([e22d61c](https://github.com/loonghao/GameLearningRuntime/commit/e22d61c1344fcf4644a4598cc6d1b366e68f656b))
* add reconnect resume reconciliation ([#66](https://github.com/loonghao/GameLearningRuntime/issues/66)) ([d6b002f](https://github.com/loonghao/GameLearningRuntime/commit/d6b002fa6c983c8231aa3368dd6e3d037a4397e0))
* add run-scoped adapter state ([#76](https://github.com/loonghao/GameLearningRuntime/issues/76)) ([629a1d5](https://github.com/loonghao/GameLearningRuntime/commit/629a1d573509160b3bc8e98f0e147ba14ba583f1))
* align checkpoint contract projections ([#78](https://github.com/loonghao/GameLearningRuntime/issues/78)) ([644ab35](https://github.com/loonghao/GameLearningRuntime/commit/644ab350b73e9511c73bad101119a118808dfd3b))
* detect stalled goal progress ([7e58c04](https://github.com/loonghao/GameLearningRuntime/commit/7e58c04fee604446a0801ec9a3a3afd195fbf5af))
* fingerprint environment configuration ([793fc97](https://github.com/loonghao/GameLearningRuntime/commit/793fc970d0a2d1fe38b01575c0fcc0cb95934484))
* route command refusals safely ([#77](https://github.com/loonghao/GameLearningRuntime/issues/77)) ([f5bb8ef](https://github.com/loonghao/GameLearningRuntime/commit/f5bb8ef0d408ed12da5c5d2f58525b9b02d8c0c3))


### Bug Fixes

* distinguish trainer no-data results ([b2d89ce](https://github.com/loonghao/GameLearningRuntime/commit/b2d89ce2cee56ad39f4db14b505f356052811b32))
* persist structured trainer metrics ([11655d2](https://github.com/loonghao/GameLearningRuntime/commit/11655d2c948c9433a0f290d5426ec1a8a75df1ee))
* preserve partial unrolls after step failures ([#71](https://github.com/loonghao/GameLearningRuntime/issues/71)) ([992cd0f](https://github.com/loonghao/GameLearningRuntime/commit/992cd0ff04a80ca301de6d875eb6241ae1fec388))
* satisfy trainer status clippy gates ([da23809](https://github.com/loonghao/GameLearningRuntime/commit/da238091fa09b66b7bef2c27d828eb45ca9154bd))

## [0.9.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.8.0...v0.9.0) (2026-09-01)


### Features

* add offline interactive run reports ([3c7a32e](https://github.com/loonghao/GameLearningRuntime/commit/3c7a32ea68a6500ea4e3bf3d6b84e8eebf967f33))
* add runtime evidence contracts ([313f677](https://github.com/loonghao/GameLearningRuntime/commit/313f677be595c47261a05359c57f6cd3ffe9c6a0))


### Bug Fixes

* harden transition persistence and batching ([4a85190](https://github.com/loonghao/GameLearningRuntime/commit/4a85190703820733b77f9fd23d6cf5c3b2d9a0cf))
* report forced actions in PPO metrics ([6c62985](https://github.com/loonghao/GameLearningRuntime/commit/6c62985550cc2881d321d5400ef16990aeeef958)), closes [#41](https://github.com/loonghao/GameLearningRuntime/issues/41)
* stabilize repeated run reports ([ace373c](https://github.com/loonghao/GameLearningRuntime/commit/ace373c8d52b4a66f14abeeedb547277cf59672a))

## [0.8.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.7.0...v0.8.0) (2026-09-01)


### Features

* trace knowledge injection queries ([e6c067d](https://github.com/loonghao/GameLearningRuntime/commit/e6c067d24c4ce89c49d2909875c4d21242a5646f))

## [0.7.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.6.1...v0.7.0) (2026-09-01)


### Features

* distribute skills as agent plugin ([af70610](https://github.com/loonghao/GameLearningRuntime/commit/af7061092994a4f4a35094b5a6d3157bd4081606))

## [0.6.1](https://github.com/loonghao/GameLearningRuntime/compare/v0.6.0...v0.6.1) (2026-09-01)


### Bug Fixes

* upload hidden release archives ([68a1d12](https://github.com/loonghao/GameLearningRuntime/commit/68a1d12dde856e3524d3938facc846b1ec25a67a))

## [0.6.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.5.0...v0.6.0) (2026-09-01)


### Features

* add agent-first training control plane ([240a02e](https://github.com/loonghao/GameLearningRuntime/commit/240a02e3e25f5575b710884891b28652cafa993d))
* make rust cli the primary entrypoint ([b3da49e](https://github.com/loonghao/GameLearningRuntime/commit/b3da49ea44054ac96da0ca8336c764fb1ef7c26c))


### Bug Fixes

* support virtual workspace releases ([9f87d76](https://github.com/loonghao/GameLearningRuntime/commit/9f87d7649a9acd7866c0ca17c1caa59c054035ea))

## [0.5.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.4.0...v0.5.0) (2026-08-31)


### Features

* add bounded knowledge injection ([08ca15f](https://github.com/loonghao/GameLearningRuntime/commit/08ca15fb2fd2e2dd74783300d16f69a633c3a4a4))
* bind demonstration provenance to trajectory bytes ([fcb3e6e](https://github.com/loonghao/GameLearningRuntime/commit/fcb3e6ef6dfbab4bd5416d41a57794a7ea26f695))

## [0.4.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.3.0...v0.4.0) (2026-08-31)


### Features

* **gymnasium:** expose verified capabilities ([8fa57dd](https://github.com/loonghao/GameLearningRuntime/commit/8fa57dd115188836c4ba24f672a0fb32b15bac90))
* **torch:** support demonstration sample weights ([01c053e](https://github.com/loonghao/GameLearningRuntime/commit/01c053e9a5002fce935298e8f291c389cdd8be59))

## [0.3.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.2.0...v0.3.0) (2026-08-31)


### Features

* add engine runtime integration profiles ([955489b](https://github.com/loonghao/GameLearningRuntime/commit/955489b2f9dcdf0e8e72344db1bcabef1627f53e))
* add loader and training safety toolkit ([#18](https://github.com/loonghao/GameLearningRuntime/issues/18)) ([c22cbb2](https://github.com/loonghao/GameLearningRuntime/commit/c22cbb2a0fbb60c39ac5e81e07aa7ba228067908))
* add runtime host and provider SDKs ([e1bb13f](https://github.com/loonghao/GameLearningRuntime/commit/e1bb13f307e13674eed7f49a432fe60541528d16))


### Bug Fixes

* **ci:** harden release checksum glob ([9e01292](https://github.com/loonghao/GameLearningRuntime/commit/9e01292aaabb8d7f658070eeca5f87283801f3e0))

## [0.2.0](https://github.com/loonghao/GameLearningRuntime/compare/v0.1.0...v0.2.0) (2026-08-31)

### Added

- Optional terminal-bounded synchronous collection with `stop_on_done=True`,
  preserving the default fixed-length cross-episode behavior.
- Capability-gated live attach across the core environment, synchronous
  collector, TorchRL adapter, and additive `glr.v1` protocol RPC, without
  aliasing attachment to deterministic reset.
- Explicit Gymnasium `attach_provider` forwarding for authorized running games,
  preserving the distinction between logical attachment and physical reset.
- Optional deny-by-default Gymnasium compatibility adapter for Box, Discrete,
  MultiDiscrete, MultiBinary, Dict, and Tuple spaces.
- Dedicated Gymnasium conformance CI and architecture decisions for adapter
  reuse, metadata privacy, and benchmark-gated Rust data-plane work.
- Optional model-neutral PyTorch objectives for masked behavior cloning,
  PPO/GAE, and IMPALA/V-trace, including explicit termination and truncation
  semantics.
- A privacy-safe synthetic data-plane benchmark for transition JSON round
  trips, trajectory construction, bounded actor-queue handoff, and mechanical
  application of the Rust adoption threshold.
- A reusable environment conformance runner and four synthetic contract
  profiles covering turn-based masks, real-time hybrid actions, FPS controls,
  and nested ARPG actions.
- Transport-neutral client and server bridge ports with protocol/capability
  negotiation, immutable reset/step requests, episode/step fencing, metadata
  deny-by-default, and no implicit action retry.
- Strict `glr.training.v1` knowledge-source, lifecycle, bridge-capability, and
  reward configuration with source authority checks and immutable reward
  contribution breakdowns.
- A project-owned adapter-builder Agent Skill with a synthetic reset/attach
  scaffold, provenance-first gameplay research workflow, and a privacy-safe
  research-manifest validator.

### Fixed

- Copy immutable NumPy bounds before creating TorchRL specs, avoiding
  non-writable tensor warnings and undefined write behavior.

## [0.1.0](https://github.com/loonghao/GameLearningRuntime/releases/tag/v0.1.0) (2026-08-31)

### Added

- Framework-neutral environment, tensor-tree, time-step, transition, and
  unroll contracts.
- Fail-closed runtime contract validation.
- Versioned Protobuf runtime service and JSONL transition records.
- Optional TorchRL environment adapter.
- Counter environment example, documentation, reusable CI, and release CD.
