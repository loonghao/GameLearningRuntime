"""Check only distributions for the version declared by this checkout."""

from __future__ import annotations

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
    artifacts = [
        ROOT / "dist" / f"game_learning_runtime-{version}-py3-none-any.whl",
        ROOT / "dist" / f"game_learning_runtime-{version}.tar.gz",
    ]
    missing = [path.name for path in artifacts if not path.is_file()]
    if missing:
        raise SystemExit(f"missing distributions for version {version}: {missing}")
    subprocess.run(
        [sys.executable, "-m", "twine", "check", *(str(path) for path in artifacts)],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
