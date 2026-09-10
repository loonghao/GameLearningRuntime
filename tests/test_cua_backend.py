import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from game_learning_runtime.cua_backend import DccCuaBackend
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.external_runtime import InputCaptureSession


class Channel:
    def __init__(self, config):
        self.directory = Path(config.arguments[-1])
        self.requests = []
        self.cursor = 0
        self.closed = False
        self.pid = 12

    def exchange(self, request):
        self.requests.append(request)
        method = request["method"]
        if method == "ping":
            return {"host_version": "1.8.1"}
        if method == "doctor":
            return {
                "routes": {"visual": {"ready": True}},
                "checks": {"interactive_desktop": {"input_ready": True}},
            }
        if method == "open_session":
            return {"window_capability": "test-capability"}
        if method == "snapshot":
            self.cursor += 1
            path = self.directory / "capture.bin"
            path.write_bytes(b"test decoder fixture")
            return {
                "observation_id": str(self.cursor),
                "accessibility_state_id": "ax",
                "observation": {
                    "process_id": self.pid,
                    "window_handle": 34,
                    "width": 4,
                    "height": 3,
                },
                "_dcc_cua_binary_output": str(path),
            }
        return {"type": "ok"}

    def close(self):
        self.closed = True


class DecodedImage:
    width, height = 4, 3

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def convert(self, mode):
        return np.zeros((3, 4, 3), dtype=np.uint8)


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr("game_learning_runtime.cua_backend.JsonLineHostChannel.open", Channel)
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        SimpleNamespace(Image=SimpleNamespace(open=lambda _: DecodedImage())),
    )
    value = DccCuaBackend(
        Path(sys.executable), process_id=12, window_handle=34, clicks={"advance": (2, 1)}
    )
    yield value
    value.close()


def test_persistent_binding_and_fenced_input(backend):
    session = InputCaptureSession(backend, commands={"advance"})
    first = session.observe()
    second = session.act("advance", expected_sequence=first.sequence, hold_ms=20)
    assert second.sequence == first.sequence + 1
    actions = [r for r in backend._channel.requests if r["method"] == "execute_action"]
    assert actions[0]["params"]["observation_id"] == "1"
    assert actions[0]["params"]["window_capability"] == "test-capability"
    session.close()
    assert backend._channel.closed


def test_capture_target_change_fails_closed(backend):
    backend._channel.pid = 99
    session = InputCaptureSession(backend, commands={"advance"})
    with pytest.raises(ContractViolation, match="target changed"):
        session.observe()
    assert backend._channel.closed


def test_direct_backend_rejects_unknown_and_out_of_frame_actions(backend):
    with pytest.raises(ContractViolation):
        backend.apply("advance", hold_ms=20)
    backend.capture()
    with pytest.raises(ValueError):
        backend.apply("advance", hold_ms=1001)
    backend._clicks["advance"] = (20, 20)
    with pytest.raises(ContractViolation, match="outside"):
        backend.apply("advance", hold_ms=20)
