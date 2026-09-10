"""Reusable real-time input/capture orchestration, independent of game semantics.

A concrete backend owns exact target binding, bounded capture deadlines and
an OS input watchdog. This module does not synthesize reward, reset a game or
claim frame-accurate stepping. Game adapters project frames into their existing
GameEnvironment/BridgeDriver observation and action contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.errors import ContractViolation


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """A backend-owned monotonic capture cursor and RGB pixels."""

    sequence: int
    pixels: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("capture sequence must be a non-negative integer")
        pixels = np.asarray(self.pixels)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("capture must be uint8 HWC RGB")
        if not pixels.shape[0] or not pixels.shape[1] or pixels.nbytes > 32 * 1024 * 1024:
            raise ValueError("capture must be nonempty and at most 32 MiB")
        owned = pixels.copy()
        owned.flags.writeable = False
        object.__setattr__(self, "pixels", owned)


class InputCaptureBackend(Protocol):
    """Port for a reviewed target-bound input/capture provider.

    validate_target must reject recycled process/window identities. apply must
    stop holding input within hold_ms even if the client dies. capture must
    return a newly acquired frame with a strictly increasing sequence, under a
    backend-enforced deadline. release is idempotent and releases only owned
    input. These requirements need concrete backend acceptance tests.
    """

    def validate_target(self) -> None: ...
    def capture(self) -> CapturedFrame: ...
    def apply(self, command: str, *, hold_ms: int) -> None: ...
    def release(self) -> None: ...
    def close(self) -> None: ...


class InputCaptureSession:
    """Serialize allowlisted actions with fresh readback and no mutating retry."""

    def __init__(
        self, backend: InputCaptureBackend, *, commands: set[str], max_hold_ms: int = 1000
    ) -> None:
        if type(max_hold_ms) is not int or not 1 <= max_hold_ms <= 1000:
            raise ValueError("max_hold_ms must be an integer in [1, 1000]")
        if any(not isinstance(c, str) or not c or len(c) > 128 for c in commands):
            raise ValueError("commands must contain bounded nonempty names")
        self._backend = backend
        self._commands = frozenset(commands)
        self._max_hold_ms = max_hold_ms
        self._sequence = -1
        self._closed = False
        self._lock = RLock()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractViolation("input/capture session is closed")

    def _capture(self) -> CapturedFrame:
        self._backend.validate_target()
        frame = self._backend.capture()
        self._backend.validate_target()
        if frame.sequence <= self._sequence:
            raise ContractViolation("stale capture sequence")
        self._sequence = frame.sequence
        return frame

    def observe(self) -> CapturedFrame:
        with self._lock:
            self._ensure_open()
            try:
                return self._capture()
            except BaseException:
                self.close()
                raise

    def act(self, command: str, *, expected_sequence: int, hold_ms: int) -> CapturedFrame:
        with self._lock:
            self._ensure_open()
            if command not in self._commands:
                raise ContractViolation("command is not allowlisted")
            if type(hold_ms) is not int or not 1 <= hold_ms <= self._max_hold_ms:
                raise ValueError("hold_ms exceeds the session bound")
            if (
                type(expected_sequence) is not int
                or self._sequence < 0
                or expected_sequence != self._sequence
            ):
                raise ContractViolation("stale observation sequence")
            try:
                self._backend.validate_target()
                try:
                    self._backend.apply(command, hold_ms=hold_ms)
                finally:
                    self._backend.release()
                return self._capture()
            except BaseException:
                # An action may already have happened. Never allow a blind retry.
                self.close()
                raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._backend.release()
                finally:
                    self._backend.close()
