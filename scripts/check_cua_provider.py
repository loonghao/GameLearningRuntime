"""Verify one DCC-CUA click against the isolated Unity external-input sample."""

import argparse
import json
import time
from pathlib import Path

from game_learning_runtime.cua_backend import DccCuaBackend
from game_learning_runtime.external_runtime import InputCaptureSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, type=Path)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--hwnd", required=True, type=int)
    parser.add_argument("--x", required=True, type=int)
    parser.add_argument("--y", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--activate-before-capture", action="store_true")
    args = parser.parse_args()
    before_count = json.loads(args.state_file.read_text())["count"]
    print(f"provider=dcc-cua runtime=1.8.1 pid={args.pid} hwnd={args.hwnd}", flush=True)
    backend = DccCuaBackend(
        args.cli.resolve(),
        process_id=args.pid,
        window_handle=args.hwnd,
        clicks={"advance": (args.x, args.y)},
        activate_before_capture=args.activate_before_capture,
    )
    session = InputCaptureSession(backend, commands={"advance"})
    try:
        before = session.observe()
        if before.pixels.shape[:2] != (args.height, args.width):
            raise ValueError(
                "Window geometry changed; inspect a fresh frame before selecting coordinates"
            )
        after = session.act("advance", expected_sequence=before.sequence, hold_ms=50)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            count = json.loads(args.state_file.read_text())["count"]
            if count == before_count + 1:
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Input acknowledgement did not produce the game postcondition")
        print(
            json.dumps(
                {
                    "provider": "dcc-cua",
                    "count_delta": 1,
                    "fresh_capture": after.sequence > before.sequence,
                }
            )
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
