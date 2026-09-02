# Reproduce trained models

[简体中文](reproducible-model-bundles.zh-CN.md)

`glr.model-bundle.v1` keeps a trained artifact together with the software inputs
needed to inspect and rerun its training. A bundle contains:

- environment and protocol identity;
- algorithm, framework, and framework version;
- learner/environment seeds;
- copied training/runtime configuration, reward-safety and demonstration
  policies, source snapshots, and lock files;
- copied model artifacts and aggregate metrics; and
- relative path, byte size, and SHA-256 for every file.

For an agent-first run, include the exact goal, research bundle, trial plans,
training/safety configuration, selected transition manifests, and capture
manifests as inputs. You may also include a `glr.spatial-knowledge.v1` export
for a fresh instance of the same environment and protocol. Import that snapshot
separately with `glr knowledge import`; GLR downgrades it to advisory until the
new runtime observes it again.

Build a bundle from an adapter trainer:

```python
from game_learning_runtime import build_model_bundle

manifest = build_model_bundle(
    "dist/model-bundle",
    environment_id="example.environment-v1",
    protocol_version="1.0",
    algorithm="ppo",
    framework="pytorch",
    framework_version="2.8.0",
    seeds=(7,),
    inputs={
        "training.json": "training.json",
        "reward-safety.json": "reward-safety.json",
        "demonstration-policy.json": "demonstration-policy.json",
        "runtime-integration.json": "runtime-integration.json",
        "uv.lock": "uv.lock",
    },
    artifacts={"weights/model.pt": "runs/model.pt"},
)
```

Verification fails on a missing, linked, resized, or changed entry:

```python
from game_learning_runtime import verify_model_bundle

manifest = verify_model_bundle("dist/model-bundle")
```

The builder refuses to overwrite a non-empty directory and never writes source
absolute paths into the manifest. Publish licenses and model cards alongside a
bundle when redistribution permits it. Integrity and captured inputs improve
reproduction; they do not guarantee identical nondeterministic GPU kernels,
driver behavior, live-game state, or model quality.

## Checkpoint contract migration

Long-running trainers can bind an opaque checkpoint to the exact learner
contract without putting framework code in GLR:

```python
from game_learning_runtime import (
    CheckpointContract,
    migrate_checkpoint_manifest,
    write_checkpoint_manifest,
)

contract = CheckpointContract(
    protocol_version="glr.v1",
    observation_sha256=observation_digest,
    action_sha256=action_digest,
    reward_sha256=reward_digest,
    knowledge_sha256=knowledge_digest,
)
write_checkpoint_manifest(
    "checkpoint.manifest.json",
    checkpoint_path="policy.ckpt",
    contract=contract,
)
```

Call `migrate_checkpoint_manifest(..., confirm=False)` first. It returns
field-level diagnostics and marks only reward/knowledge changes as generic
migrations; protocol, observation, and action changes fail closed. Passing
`confirm=True` creates adjacent backups and atomically rewrites the manifest.
Projects that need framework serialization can provide a `saver(source,
destination, contract)` callback; the default path copies bytes exactly and
re-verifies the resulting checksum. No checkpoint is changed without explicit
confirmation.

The standalone CLI exposes the same preflight as a machine-readable command:

```shell
glr --json checkpoint migrate \
  --manifest checkpoints/policy.manifest.json \
  --contract contracts/live-checkpoint-contract.json
```

An unchanged contract exits `0`; a compatible change is reported with exit
`3` until `--force` is supplied, and an incompatible observation/action or
protocol change exits `4` without touching either file. Successful migration
keeps the checkpoint bytes unchanged, writes adjacent `.bak` files, and
re-verifies the updated manifest.

The provider SDK projections expose the same `glr.checkpoint-contract.v1` and
`glr.checkpoint-manifest.v1` fields for native adapters. They are validation
types only: migration remains an explicit Python/CLI operation, and neither
the C# nor C++ provider contract silently rewrites learner state.
