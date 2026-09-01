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
