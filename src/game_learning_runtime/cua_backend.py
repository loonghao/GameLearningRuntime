"""Project-owned DCC-CUA input/capture backend; no generic GUI fallback."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import numpy as np

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.external_runtime import CapturedFrame
from game_learning_runtime.host import HostProcessConfig, JsonLineHostChannel


class DccCuaBackend:
    """Bounded click commands on one persistent exact-window Host session.

    Callers must attest provider/runtime/PID/HWND before constructing this UI
    backend. Commands map to pixel coordinates in fresh exact-window captures.
    There are no persistent key-down operations; each click releases its input.
    DCC-CUA owns OS identity validation, observation fencing and disconnect cleanup.
    """

    def __init__(
        self,
        executable: Path,
        *,
        process_id: int,
        window_handle: int,
        clicks: Mapping[str, tuple[int, int]],
        activate_before_capture: bool = False,
    ) -> None:
        if type(process_id) is not int or process_id <= 0:
            raise ValueError("process_id must be positive")
        if type(window_handle) is not int or window_handle <= 0:
            raise ValueError("window_handle must be positive")
        if not isinstance(activate_before_capture, bool):
            raise TypeError("activate_before_capture must be a bool")
        if any(
            not name or len(point) != 2 or any(type(v) is not int or v < 0 for v in point)
            for name, point in clicks.items()
        ):
            raise ValueError("click commands require nonnegative integer pixel coordinates")
        self._clicks = dict(clicks)
        self._target = (process_id, window_handle)
        self._activate_before_capture = activate_before_capture
        self._directory = TemporaryDirectory(prefix="glr-cua-")
        self._closed = False
        self._sequence = 0
        self._last_observation_id: str | None = None
        self._snapshot: dict[str, Any] | None = None
        self._channel = JsonLineHostChannel.open(
            HostProcessConfig(
                executable=executable,
                arguments=("host-jsonl", "--output-dir", self._directory.name),
                request_timeout_seconds=15,
            )
        )
        self._session = "glr-" + uuid4().hex
        self._binding: dict[str, Any] = {
            "session_id": self._session,
            "task_grant_id": self._session,
        }
        try:
            pong = self._call("ping", {})
            if pong.get("host_version") != "1.8.1":
                raise ContractViolation("DCC-CUA backend is validated against runtime 1.8.1")
            health = self._call("doctor", {})
            if not health.get("routes", {}).get("visual", {}).get("ready", False):
                raise ContractViolation("DCC-CUA visual capture is unavailable")
            if (
                not health.get("checks", {})
                .get("interactive_desktop", {})
                .get("input_ready", False)
            ):
                raise ContractViolation("DCC-CUA interactive input desktop is unavailable")
            opened = self._call(
                "open_session",
                {
                    "session_id": self._session,
                    "grant": {
                        "task_grant_id": self._session,
                        "application_label": "GLR runtime",
                        "process_id": process_id,
                        "window_handle": window_handle,
                        "allow_raw_input": True,
                    },
                },
            )
            self._binding["window_capability"] = opened["window_capability"]
        except BaseException:
            self.close()
            raise

    def _call(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise ContractViolation("DCC-CUA backend is closed")
        response = dict(self._channel.exchange({"method": method, "params": dict(params)}))
        if response.get("type") == "error":
            raise ContractViolation(f"DCC-CUA refused {method}: {response.get('code')}")
        return response

    def validate_target(self) -> None:
        # The host validates the session capability against its bound OS identity.
        self._call("get_window_state", self._binding)

    def capture(self) -> CapturedFrame:
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("DCC-CUA capture requires the 'cua' optional dependency") from error
        snapshot = self._call(
            "snapshot",
            self._binding
            | {
                "max_depth": 0,
                "max_nodes": 0,
                "activate_before": self._activate_before_capture,
            },
        )
        observation = snapshot["observation"]
        if (observation["process_id"], observation["window_handle"]) != self._target:
            raise ContractViolation("DCC-CUA capture target changed")
        if snapshot["observation_id"] == self._last_observation_id:
            raise ContractViolation("DCC-CUA returned a stale observation")
        path = Path(snapshot["_dcc_cua_binary_output"]).resolve()
        if not path.is_relative_to(Path(self._directory.name).resolve()):
            raise ContractViolation("DCC-CUA capture artifact escaped its directory")
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ContractViolation("DCC-CUA capture exceeds size bound")
        with Image.open(path) as image:
            if image.width * image.height * 3 > 32 * 1024 * 1024:
                raise ContractViolation("DCC-CUA decoded capture exceeds size bound")
            pixels = np.array(image.convert("RGB"), dtype=np.uint8)
        path.unlink()
        self._snapshot = snapshot
        self._last_observation_id = snapshot["observation_id"]
        self._sequence += 1
        return CapturedFrame(self._sequence, pixels)

    def apply(self, command: str, *, hold_ms: int) -> None:
        if command not in self._clicks or self._snapshot is None:
            raise ContractViolation("Click requires an allowlisted command and fresh capture")
        if type(hold_ms) is not int or not 1 <= hold_ms <= 1000:
            raise ValueError("Click duration must be in [1, 1000] ms")
        x, y = self._clicks[command]
        observation = self._snapshot["observation"]
        if x >= observation["width"] or y >= observation["height"]:
            raise ContractViolation("Click is outside the captured window")
        snapshot, self._snapshot = self._snapshot, None
        self._call(
            "execute_action",
            self._binding
            | {
                "observation_id": snapshot["observation_id"],
                "accessibility_state_id": snapshot["accessibility_state_id"],
                "action": {
                    "action": "click",
                    "input_kind": "raw_input",
                    "intent": command,
                    "x": x,
                    "y": y,
                    "button": "left",
                    "duration_ms": hold_ms,
                    "delivery_mode": "background",
                },
            },
        )

    def release(self) -> None:
        # Only complete bounded clicks are exposed, never persistent key holds.
        # Host disconnection also releases input after an uncertain response.
        return None

    def close(self) -> None:
        if not self._closed:
            with suppress(Exception):
                self._call("stop_session", {"session_id": self._session})
            self._closed = True
            self._channel.close()
            self._directory.cleanup()
