from __future__ import annotations

import json
from pathlib import Path

import pytest

from game_learning_runtime import (
    MODEL_BUNDLE_SCHEMA_VERSION,
    BundleFile,
    ModelBundleManifest,
    build_model_bundle,
    load_model_bundle_manifest,
    verify_model_bundle,
)
from game_learning_runtime.errors import ContractViolation


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_model_bundle_copies_reproduction_inputs_and_verifies_artifacts(tmp_path: Path) -> None:
    training = _write(tmp_path / "workspace/training.json", '{"seed":7}\n')
    lock = _write(tmp_path / "workspace/uv.lock", "version = 1\n")
    model = _write(tmp_path / "workspace/model.json", '{"action":1}\n')
    bundle = tmp_path / "bundle"

    manifest = build_model_bundle(
        bundle,
        environment_id="example.environment-v1",
        protocol_version="1.0",
        algorithm="reference-fixed-policy",
        framework="glr-reference",
        framework_version="0.2.0",
        seeds=(7,),
        inputs={"training.json": training, "uv.lock": lock},
        artifacts={"model.json": model},
    )

    assert manifest.schema_version == MODEL_BUNDLE_SCHEMA_VERSION
    assert manifest.seeds == (7,)
    assert (bundle / "inputs/training.json").read_text(encoding="utf-8") == '{"seed":7}\n'
    assert (bundle / "artifacts/model.json").read_text(encoding="utf-8") == '{"action":1}\n'
    assert verify_model_bundle(bundle) == manifest
    serialized = (bundle / "manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert load_model_bundle_manifest(bundle / "manifest.json") == manifest


def test_model_bundle_detects_tampering_and_refuses_overwrite(tmp_path: Path) -> None:
    source = _write(tmp_path / "source/model.bin", "original")
    bundle = tmp_path / "bundle"
    build_model_bundle(
        bundle,
        environment_id="example.environment-v1",
        protocol_version="1.0",
        algorithm="bc",
        framework="pytorch",
        framework_version="2.8.0",
        seeds=(1, 2),
        inputs={"training.json": source},
        artifacts={"weights/model.bin": source},
    )
    (bundle / "artifacts/weights/model.bin").write_text("changed", encoding="utf-8")

    with pytest.raises(ContractViolation, match="sha256 mismatch"):
        verify_model_bundle(bundle)
    with pytest.raises(FileExistsError, match="non-empty"):
        build_model_bundle(
            bundle,
            environment_id="example.environment-v1",
            protocol_version="1.0",
            algorithm="bc",
            framework="pytorch",
            framework_version="2.8.0",
            seeds=(1,),
            inputs={"training.json": source},
            artifacts={"model.bin": source},
        )


@pytest.mark.parametrize(
    ("inputs", "artifacts", "message"),
    [
        ({"../secret": "source"}, {"model.bin": "source"}, "relative"),
        ({"training.json": "missing"}, {"model.bin": "source"}, "regular file"),
        ({"training.json": "source"}, {"/model.bin": "source"}, "relative"),
    ],
)
def test_model_bundle_rejects_unsafe_or_missing_files(
    tmp_path: Path,
    inputs: dict[str, str],
    artifacts: dict[str, str],
    message: str,
) -> None:
    source = _write(tmp_path / "source", "data")
    resolved_inputs = {
        name: source if value == "source" else tmp_path / value for name, value in inputs.items()
    }
    resolved_artifacts = {
        name: source if value == "source" else tmp_path / value for name, value in artifacts.items()
    }

    with pytest.raises((FileNotFoundError, ValueError), match=message):
        build_model_bundle(
            tmp_path / "bundle",
            environment_id="example.environment-v1",
            protocol_version="1.0",
            algorithm="bc",
            framework="pytorch",
            framework_version="2.8.0",
            seeds=(1,),
            inputs=resolved_inputs,
            artifacts=resolved_artifacts,
        )


def test_model_bundle_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": MODEL_BUNDLE_SCHEMA_VERSION}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        load_model_bundle_manifest(path)


def test_model_bundle_manifest_rejects_non_object_and_malformed_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="must contain an object"):
        load_model_bundle_manifest(path)

    digest = "0" * 64
    entry = BundleFile(path="model.bin", sha256=digest, size_bytes=1)
    with pytest.raises(ValueError, match="start with a lowercase"):
        ModelBundleManifest(
            environment_id="Invalid",
            protocol_version="1.0",
            algorithm="bc",
            framework="pytorch",
            framework_version="2.8.0",
            seeds=(1,),
            inputs=(entry,),
            artifacts=(entry,),
        )
    with pytest.raises(ValueError, match="duplicates"):
        ModelBundleManifest(
            environment_id="example.environment-v1",
            protocol_version="1.0",
            algorithm="bc",
            framework="pytorch",
            framework_version="2.8.0",
            seeds=(1, 1),
            inputs=(entry,),
            artifacts=(entry,),
        )


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"path": "model.bin", "sha256": "bad", "size_bytes": 1}, "SHA-256"),
        ({"path": "model.bin", "sha256": "0" * 64, "size_bytes": -1}, "non-negative"),
        ({"path": "model.bin", "sha256": "0" * 64}, "fields differ"),
    ],
)
def test_bundle_file_rejects_invalid_integrity_metadata(
    entry: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BundleFile.from_mapping(entry)
