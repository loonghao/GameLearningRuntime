"""Read-only installation hints. Detection never grants runtime capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from game_learning_runtime.runtime_integration import EngineFamily


class RuntimeVariant(str, Enum):
    UNITY_MONO = "unity-mono"
    UNITY_IL2CPP = "unity-il2cpp"
    UNREAL = "unreal"
    GODOT = "godot"
    UNKNOWN = "unknown"

    @property
    def engine(self) -> EngineFamily:
        if self in {self.UNITY_MONO, self.UNITY_IL2CPP}:
            return EngineFamily.UNITY
        return {
            self.UNREAL: EngineFamily.UNREAL,
            self.GODOT: EngineFamily.GODOT,
            self.UNKNOWN: EngineFamily.OTHER,
        }[self]


@dataclass(frozen=True, slots=True)
class InstallationHint:
    variant: RuntimeVariant
    candidates: frozenset[RuntimeVariant]

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


def inspect_installation(directory: Path) -> InstallationHint:
    """Inspect fixed, shallow markers in one operator-selected directory.

    Windows Unity layouts, Unreal source projects and Godot source/PCK files
    are hints only. Embedded/stripped exports may remain unknown. No executable
    is loaded, no process is scanned and no directory outside this root is read.
    """
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    def files(pattern: str) -> list[Path]:
        return [p for p in root.glob(pattern) if p.resolve().is_relative_to(root) and p.is_file()]

    candidates: set[RuntimeVariant] = set()
    if files("*_Data/Managed/Assembly-CSharp.dll"):
        candidates.add(RuntimeVariant.UNITY_MONO)
    if files("GameAssembly.dll") and files("*_Data/il2cpp_data/Metadata/global-metadata.dat"):
        candidates.add(RuntimeVariant.UNITY_IL2CPP)
    if files("*.uproject") or files("Engine/Binaries/Win64/Unreal*.exe"):
        candidates.add(RuntimeVariant.UNREAL)
    if files("project.godot"):
        candidates.add(RuntimeVariant.GODOT)
    for pack in files("*.pck"):
        with pack.open("rb") as stream:
            if stream.read(4) == b"GDPC":
                candidates.add(RuntimeVariant.GODOT)
    variant = next(iter(candidates)) if len(candidates) == 1 else RuntimeVariant.UNKNOWN
    return InstallationHint(variant, frozenset(candidates))
