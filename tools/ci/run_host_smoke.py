from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from game_learning_runtime import (
    BridgeEnvironment,
    ContractEnvironment,
    HostBridgeDriver,
    HostProcessConfig,
)


def main() -> int:
    if len(sys.argv) > 2:
        raise SystemExit("usage: run_host_smoke.py [ABSOLUTE_GLR_HOSTD_PATH]")
    executable = (
        Path(sys.argv[1])
        if len(sys.argv) == 2
        else Path.cwd() / "target" / "debug" / ("glr-hostd.exe" if os.name == "nt" else "glr-hostd")
    )
    config = HostProcessConfig(executable=executable.resolve())
    driver = HostBridgeDriver.from_process(config)
    step_count = 0
    with ContractEnvironment(
        BridgeEnvironment(
            driver,
            required_capabilities={"host-stdio", "reset", "step"},
        )
    ) as environment:
        timestep = environment.reset(seed=7)
        while not timestep.done:
            timestep = environment.step({"choice": np.array([1], dtype=np.int64)})
            step_count += 1
            if step_count > 8:
                raise RuntimeError("synthetic Runtime Host did not terminate within its bound")
    print(
        json.dumps(
            {
                "schema": "glr.host-smoke.v1",
                "environment_count": 1,
                "episode_count": 1,
                "step_count": step_count,
                "terminal_count": 1,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
