from __future__ import annotations

import json
from pathlib import Path

import pytest

from game_learning_runtime import (
    CheckpointContract,
    ContractViolation,
    compare_checkpoint_contract,
    migrate_checkpoint_manifest,
    verify_checkpoint_manifest,
    write_checkpoint_manifest,
)


def _contract(*, reward: str = "4" * 64, action: str = "2" * 64) -> CheckpointContract:
    return CheckpointContract(
        protocol_version="glr.v1",
        observation_sha256="1" * 64,
        action_sha256=action,
        reward_sha256=reward,
        knowledge_sha256="3" * 64,
    )


def test_checkpoint_manifest_verifies_and_reports_field_level_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    checkpoint.write_bytes(b"weights")
    manifest_path = tmp_path / "checkpoint.manifest.json"
    recorded = _contract()
    manifest = write_checkpoint_manifest(
        manifest_path,
        checkpoint_path="policy.ckpt",
        contract=recorded,
    )
    assert verify_checkpoint_manifest(manifest_path) == manifest

    mismatches = compare_checkpoint_contract(recorded, _contract(reward="5" * 64))
    assert [(item.field, item.migratable) for item in mismatches] == [("reward_sha256", True)]


def test_checkpoint_migration_requires_confirmation_then_keeps_backups(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    checkpoint.write_bytes(b"weights")
    manifest_path = tmp_path / "checkpoint.manifest.json"
    write_checkpoint_manifest(manifest_path, checkpoint_path="policy.ckpt", contract=_contract())
    current = _contract(reward="5" * 64)

    dry_run = migrate_checkpoint_manifest(manifest_path, current)
    assert dry_run.requires_confirmation
    assert dry_run.changed
    assert verify_checkpoint_manifest(manifest_path).contract == _contract()

    result = migrate_checkpoint_manifest(manifest_path, current, confirm=True)
    assert result.changed
    assert not result.requires_confirmation
    assert result.manifest.contract == current
    assert result.manifest_backup is not None and result.manifest_backup.is_file()
    assert result.checkpoint_backup is not None and result.checkpoint_backup.is_file()
    assert checkpoint.read_bytes() == b"weights"
    assert verify_checkpoint_manifest(manifest_path).contract == current


def test_checkpoint_migration_rejects_incompatible_action_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    checkpoint.write_bytes(b"weights")
    manifest_path = tmp_path / "checkpoint.manifest.json"
    write_checkpoint_manifest(manifest_path, checkpoint_path="policy.ckpt", contract=_contract())

    with pytest.raises(ContractViolation, match=r"action_sha256.*retrain"):
        migrate_checkpoint_manifest(manifest_path, _contract(action="6" * 64), confirm=True)


def test_checkpoint_migration_rolls_back_when_saver_fails(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.ckpt"
    checkpoint.write_bytes(b"weights")
    manifest_path = tmp_path / "checkpoint.manifest.json"
    original = write_checkpoint_manifest(
        manifest_path, checkpoint_path="policy.ckpt", contract=_contract()
    )

    def failing_saver(source: Path, destination: Path, contract: CheckpointContract) -> None:
        del source, contract
        destination.write_bytes(b"partial")
        raise RuntimeError("serializer failed")

    with pytest.raises(RuntimeError, match="serializer failed"):
        migrate_checkpoint_manifest(
            manifest_path,
            _contract(reward="5" * 64),
            confirm=True,
            saver=failing_saver,
        )
    assert verify_checkpoint_manifest(manifest_path) == original
    assert checkpoint.read_bytes() == b"weights"


def test_checkpoint_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.manifest.json"
    path.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing=.*checkpoint_path"):
        verify_checkpoint_manifest(path)
