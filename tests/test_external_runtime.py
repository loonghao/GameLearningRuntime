import numpy as np
import pytest

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.external_runtime import CapturedFrame, InputCaptureSession


class Backend:
    def __init__(self):
        self.sequence = 0
        self.valid = True
        self.released = 0
        self.actions = 0
        self.fail = False

    def validate_target(self):
        if not self.valid:
            raise ContractViolation("target changed")

    def capture(self):
        if self.fail:
            raise TimeoutError("capture")
        self.sequence += 1
        return CapturedFrame(self.sequence, np.zeros((2, 2, 3), dtype=np.uint8))

    def apply(self, command, *, hold_ms):
        self.actions += 1

    def release(self):
        self.released += 1

    def close(self):
        pass


def test_session_fences_actions_and_releases_input():
    backend = Backend()
    session = InputCaptureSession(backend, commands={"left"}, max_hold_ms=20)
    first = session.observe()
    second = session.act("left", expected_sequence=first.sequence, hold_ms=10)
    assert second.sequence > first.sequence
    assert backend.released == 1
    with pytest.raises(ContractViolation, match="stale"):
        session.act("left", expected_sequence=first.sequence, hold_ms=10)
    assert backend.actions == 1


def test_failure_after_action_closes_session_without_retry():
    backend = Backend()
    session = InputCaptureSession(backend, commands={"left"})
    frame = session.observe()
    backend.fail = True
    with pytest.raises(TimeoutError):
        session.act("left", expected_sequence=frame.sequence, hold_ms=10)
    backend.fail = False
    with pytest.raises(ContractViolation, match="closed"):
        session.observe()
    assert backend.released >= 1
    assert backend.actions == 1


@pytest.mark.parametrize(("command", "hold"), [("unknown", 1), ("left", 1001), ("left", True)])
def test_invalid_command_never_reaches_backend(command, hold):
    backend = Backend()
    session = InputCaptureSession(backend, commands={"left"})
    frame = session.observe()
    with pytest.raises((ValueError, ContractViolation)):
        session.act(command, expected_sequence=frame.sequence, hold_ms=hold)
    assert backend.actions == 0


def test_changed_target_and_stale_capture_fail_closed():
    backend = Backend()
    session = InputCaptureSession(backend, commands={"left"})
    frame = session.observe()
    backend.valid = False
    with pytest.raises(ContractViolation, match="target"):
        session.act("left", expected_sequence=frame.sequence, hold_ms=1)
    assert backend.actions == 0


def test_frame_owns_immutable_pixels():
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    frame = CapturedFrame(1, pixels)
    pixels[:] = 255
    assert not frame.pixels.any()
    assert not frame.pixels.flags.writeable


def test_stale_capture_closes_observation_session():
    backend = Backend()
    session = InputCaptureSession(backend, commands=set())
    session.observe()
    backend.sequence = 0
    with pytest.raises(ContractViolation, match="stale capture"):
        session.observe()
    with pytest.raises(ContractViolation, match="closed"):
        session.observe()


@pytest.mark.parametrize(
    "sequence,pixels",
    [
        (-1, np.zeros((2, 2, 3), dtype=np.uint8)),
        (True, np.zeros((2, 2, 3), dtype=np.uint8)),
        (0, np.zeros((2, 2), dtype=np.uint8)),
        (0, np.zeros((0, 2, 3), dtype=np.uint8)),
    ],
)
def test_invalid_frames(sequence, pixels):
    with pytest.raises(ValueError):
        CapturedFrame(sequence, pixels)


def test_session_configuration_and_close():
    with pytest.raises(ValueError):
        InputCaptureSession(Backend(), commands=set(), max_hold_ms=0)
    with pytest.raises(ValueError):
        InputCaptureSession(Backend(), commands={""})
    backend = Backend()
    session = InputCaptureSession(backend, commands=set())
    session.close()
    session.close()
    assert backend.released == 1
