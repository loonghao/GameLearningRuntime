"""Check only distributions for the version declared by this checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    python_artifacts = [
        ROOT / "dist" / f"game_learning_runtime-{version}-py3-none-any.whl",
        ROOT / "dist" / f"game_learning_runtime-{version}.tar.gz",
    ]
    executable_suffix = ".exe" if os.name == "nt" else ""
    native_artifacts = [
        ROOT / "target" / "release" / f"glr{executable_suffix}",
        ROOT / "target" / "release" / f"glr-hostd{executable_suffix}",
    ]
    artifacts = [*python_artifacts, *native_artifacts]
    missing = [path.name for path in artifacts if not path.is_file()]
    if missing:
        raise SystemExit(f"missing distributions for version {version}: {missing}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "twine",
            "check",
            *(str(path) for path in python_artifacts),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
