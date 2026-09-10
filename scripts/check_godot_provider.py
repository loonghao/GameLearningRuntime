"""Exercise the real Godot Node3D sample through the existing GLR Host driver."""

import argparse
import json
from pathlib import Path

import numpy as np

from game_learning_runtime import (
    BridgeResetRequest,
    BridgeStepRequest,
    HostBridgeDriver,
    HostProcessConfig,
    HostRemoteError,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1] / "sdk/godot"
    driver = HostBridgeDriver.from_process(
        HostProcessConfig(
            executable=args.godot.resolve(),
            arguments=(
                "--headless",
                "--no-header",
                "--path",
                str(project),
                "--script",
                "res://provider.gd",
            ),
            request_timeout_seconds=20,
        )
    )
    try:
        initial = driver.reset(BridgeResetRequest(seed=7))
        try:
            driver.step(
                BridgeStepRequest(initial.episode_id, 1, {"move": np.array([9], dtype=np.int32)})
            )
        except HostRemoteError:
            pass
        else:
            raise AssertionError("Out-of-range action accepted")
        action = {"move": np.array([1], dtype=np.int32)}
        for step in range(1, 4):
            request = BridgeStepRequest(initial.episode_id, step, action)
            result = driver.step(request)
            assert float(result.observation["position"][0]) == step
            try:
                driver.step(request)
            except HostRemoteError:
                pass
            else:
                raise AssertionError("Stale request accepted")
        assert result.done
        reset = driver.reset(BridgeResetRequest(seed=7))
        assert float(reset.observation["position"][0]) == 0
        assert reset.episode_id != initial.episode_id
        print(json.dumps({"engine": "godot", "steps": 3, "reset": True, "stale_rejected": True}))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
