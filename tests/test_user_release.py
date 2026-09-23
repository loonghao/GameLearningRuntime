from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from game_learning_runtime.model_bundle import build_model_bundle
from game_learning_runtime.user_release import (
    compile_windows_installer,
    prepare_windows_installer,
)


@pytest.fixture
def payload(tmp_path: Path) -> Path:
    root = tmp_path / "payload"
    root.mkdir()
    for name in ("app.exe", "glr.exe", "glr-hostd.exe", "LICENSE.txt", "MODEL_CARD.md"):
        (root / name).write_text("synthetic build fixture only", encoding="utf-8")
    evaluation = {
        "stage": "stage-1",
        "environment_id": "example.environment-v1",
        "protocol_version": "1.0",
        "evidence_kind": "synthetic",
    }
    (root / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    source = tmp_path / "model.json"
    source.write_text('{"action":1}', encoding="utf-8")
    build_model_bundle(
        root / "model",
        environment_id="example.environment-v1",
        protocol_version="1.0",
        algorithm="synthetic",
        framework="reference",
        framework_version="1.0",
        seeds=(7,),
        inputs={"config.json": source},
        artifacts={"model.json": source},
    )
    return root


def prepare(payload: Path, output: Path, **overrides: str) -> Path:
    values = dict(release_id="example-app", version="1.0.0", stage="stage-1", launcher="app.exe")
    values.update(overrides)
    return prepare_windows_installer(payload, output, **values)


def test_release_snapshots_a_model_and_never_autoruns_it(payload: Path, tmp_path: Path) -> None:
    prepared = tmp_path / "installer project"
    script = prepare(payload, prepared)
    manifest = json.loads((prepared / "release.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "frozen-playback"
    assert manifest["evidence_kind"] == "synthetic"
    assert manifest["files"]["app.exe"]["size_bytes"] > 0
    assert str(payload) not in (prepared / "release.json").read_text(encoding="utf-8")
    text = script.read_text(encoding="utf-8")
    assert "PrivilegesRequired=lowest" in text
    assert "AppId=glr-example-app-1.0.0" in text
    assert "[Run]" not in text
    assert "[UninstallDelete]" not in text
    (payload / "app.exe").write_text("modified original", encoding="utf-8")
    assert (prepared / "payload/app.exe").read_text(
        encoding="utf-8"
    ) == "synthetic build fixture only"
    with pytest.raises(FileExistsError):
        prepare(payload, prepared)


@pytest.mark.parametrize(
    "overrides",
    [
        {"release_id": "unsafe\n[Run]"},
        {"version": "1.0"},
        {"stage": "../stage"},
        {"launcher": "app.py"},
        {"launcher": "../app.exe"},
        {"launcher": "C:/app.exe"},
        {"launcher": "CON.exe"},
        {"launcher": "bin/app.exe."},
        {"launcher": "bin//app.exe"},
    ],
)
def test_release_rejects_unsafe_metadata(payload: Path, tmp_path: Path, overrides: dict) -> None:
    with pytest.raises(ValueError):
        prepare(payload, tmp_path / "output", **overrides)


def test_release_rejects_incomplete_or_mismatched_payload(payload: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate"):
        prepare(payload, payload / "output")
    with pytest.raises(ValueError, match="same stage"):
        prepare(payload, tmp_path / "output", stage="stage-2")
    (payload / "evaluation.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="same stage"):
        prepare(payload, tmp_path / "output")
    (payload / "app.exe").unlink()
    with pytest.raises(ValueError, match="missing"):
        prepare(payload, tmp_path / "output")


def test_release_rejects_unlabelled_evidence(payload: Path, tmp_path: Path) -> None:
    evaluation = payload / "evaluation.json"
    value = json.loads(evaluation.read_text(encoding="utf-8"))
    value["evidence_kind"] = "assumed"
    evaluation.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="distinguish"):
        prepare(payload, tmp_path / "output")


def test_release_checks_compiler_output_and_integrity(
    payload: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = tmp_path / "prepared"
    prepare(payload, prepared)
    compiler = tmp_path / "ISCC.exe"
    compiler.write_bytes(b"compiler test double")
    artifact = prepared / "installer/example-app-1.0.0-windows-x64-setup.exe"

    def run(command: list[str], *, check: bool) -> None:
        assert command == [str(compiler.resolve()), str((prepared / "setup.iss").resolve())]
        assert check is True
        artifact.parent.mkdir()
        artifact.write_bytes(b"compiled test double")

    monkeypatch.setattr(subprocess, "run", run)
    assert compile_windows_installer(prepared, compiler=compiler) == artifact
    with pytest.raises(FileExistsError):
        compile_windows_installer(prepared, compiler=compiler)
    (prepared / "payload/app.exe").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        compile_windows_installer(prepared, compiler=compiler)


def test_release_rejects_modified_compiler_script(payload: Path, tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    script = prepare(payload, prepared)
    script.write_text(script.read_text(encoding="utf-8") + "[Run]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="script differs"):
        compile_windows_installer(prepared, compiler=tmp_path / "unused")
    manifest = prepared / "release.json"
    manifest.write_text('{"schema_version":"unknown"}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        compile_windows_installer(prepared, compiler=tmp_path / "unused")


@pytest.mark.skipif(not os.environ.get("GLR_INNO_COMPILER"), reason="requires native Inno compiler")
def test_native_installer_round_trip_preserves_user_data(payload: Path, tmp_path: Path) -> None:
    """Exercise installer mechanics, not inference: payload executables are fixtures."""
    compiler = Path(os.environ["GLR_INNO_COMPILER"])
    assert compiler.is_file(), "Configured native compiler must be installed"
    prepared = tmp_path / "prepared"
    prepare(payload, prepared)
    installer = compile_windows_installer(prepared, compiler=compiler)
    destination = tmp_path / "installed app 用户"
    subprocess.run(
        [
            str(installer),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/CURRENTUSER",
            "/NOICONS",
            f"/DIR={destination}",
        ],
        check=True,
        timeout=120,
    )
    try:
        assert (destination / "app.exe").read_bytes() == (payload / "app.exe").read_bytes()
        assert (destination / "model/manifest.json").is_file()
        assert (destination / "glr-user-release.json").is_file()
        user_data = destination / "user-progress.json"
        user_data.write_text('{"progress":42}', encoding="utf-8")
    finally:
        subprocess.run(
            [str(destination / "unins000.exe"), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
            check=True,
            timeout=120,
        )
    assert not (destination / "app.exe").exists()
    assert user_data.read_text(encoding="utf-8") == '{"progress":42}'
