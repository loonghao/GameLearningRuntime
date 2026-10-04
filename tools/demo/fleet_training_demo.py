"""Run the installed SDK's finite local SIMULATED fleet example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game_learning_runtime.examples.fleet_training import run_demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-source-commit", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_demo(args.output_dir, runtime_source_commit=args.runtime_source_commit),
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
