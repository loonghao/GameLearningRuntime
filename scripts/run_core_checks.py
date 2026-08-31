"""Run the core quality gates inside one selected uv environment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import game_learning_runtime

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    (sys.executable, "-m", "ruff", "check", "."),
    (sys.executable, "-m", "ruff", "format", "--check", "."),
    (sys.executable, "-m", "mypy"),
    (
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not torchrl",
        "--cov=game_learning_runtime",
        "--cov-report=term-missing",
    ),
)


def _verify_project_origin() -> None:
    expected = (ROOT / "src/game_learning_runtime/__init__.py").resolve()
    actual = Path(game_learning_runtime.__file__).resolve()
    if actual != expected:
        raise RuntimeError(
            f"game_learning_runtime import does not originate from this checkout: {actual}"
        )


def main() -> int:
    _verify_project_origin()
    for command in COMMANDS:
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
