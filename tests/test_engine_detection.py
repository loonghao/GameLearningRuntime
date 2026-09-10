from pathlib import Path

import pytest

from game_learning_runtime.engine_detection import RuntimeVariant, inspect_installation


@pytest.mark.parametrize(
    ("markers", "variant"),
    [
        (["Demo_Data/Managed/Assembly-CSharp.dll"], RuntimeVariant.UNITY_MONO),
        (
            ["GameAssembly.dll", "Demo_Data/il2cpp_data/Metadata/global-metadata.dat"],
            RuntimeVariant.UNITY_IL2CPP,
        ),
        (["Demo.uproject"], RuntimeVariant.UNREAL),
        (["project.godot"], RuntimeVariant.GODOT),
        ([], RuntimeVariant.UNKNOWN),
    ],
)
def test_explicit_installation_markers(tmp_path: Path, markers: list[str], variant: RuntimeVariant):
    for marker in markers:
        path = tmp_path / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert inspect_installation(tmp_path).variant is variant


def test_conflicting_markers_do_not_select_a_backend(tmp_path: Path):
    (tmp_path / "project.godot").touch()
    (tmp_path / "Demo.uproject").touch()
    result = inspect_installation(tmp_path)
    assert result.variant is RuntimeVariant.UNKNOWN
    assert result.ambiguous


def test_pck_extension_alone_is_not_engine_evidence(tmp_path: Path):
    (tmp_path / "unrelated.pck").write_bytes(b"not a Godot pack")
    assert inspect_installation(tmp_path).variant is RuntimeVariant.UNKNOWN


def test_only_explicit_directory_is_accepted(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        inspect_installation(tmp_path / "missing")


def test_godot_pack_magic_and_engine_families(tmp_path: Path):
    (tmp_path / "demo.pck").write_bytes(b"GDPC")
    assert inspect_installation(tmp_path).variant is RuntimeVariant.GODOT
    assert RuntimeVariant.UNITY_MONO.engine.value == "unity"
    assert RuntimeVariant.UNITY_IL2CPP.engine.value == "unity"
    assert RuntimeVariant.UNREAL.engine.value == "unreal"
    assert RuntimeVariant.GODOT.engine.value == "godot"
    assert RuntimeVariant.UNKNOWN.engine.value == "other"
    with pytest.raises(NotADirectoryError):
        inspect_installation(tmp_path / "demo.pck")
