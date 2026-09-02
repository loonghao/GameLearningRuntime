"""Client boundary for the versioned GLR Runtime Host.

The first transport is an explicitly launched, serialized JSON-lines stdio
session. It deliberately does not claim authentication or exact target binding.
Those capabilities require a future local-IPC transport and OS identity proof.
"""

from __future__ import annotations

import base64
import binascii
import json
import queue
import subprocess
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Protocol, cast
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.bridge import (
    BridgeAttachRequest,
    BridgeResetRequest,
    BridgeResumeRequest,
    BridgeResumeResult,
    BridgeStepRequest,
)
from game_learning_runtime.contracts import (
    ActionOutcome,
    ActionReceipt,
    ActionReconciliation,
    Event,
    ReconciliationOutcome,
    TensorTree,
    TimeStep,
)
from game_learning_runtime.errors import HostProtocolError, HostRemoteError
from game_learning_runtime.specs import CompositeSpec, EnvironmentSpec, SpaceKind, TensorSpec

HOST_SCHEMA = "glr.host.v1"
DEFAULT_MAX_FRAME_BYTES = 1_048_576
HARD_MAX_FRAME_BYTES = 1_048_576

_WIRE_TO_DTYPE: Mapping[str, np.dtype[Any]] = {
    "bool": np.dtype(np.bool_),
    "uint8": np.dtype(np.uint8),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}
_DTYPE_TO_WIRE = {dtype: name for name, dtype in _WIRE_TO_DTYPE.items()}


class HostChannel(Protocol):
    """One ordered request/response channel to a single Runtime Host."""

    def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Exchange one bounded request without retrying it."""
        ...

    def close(self) -> None:
        """Release channel resources and its owned host process, if any."""
        ...


@dataclass(frozen=True, slots=True)
class HostProcessConfig:
    """Explicit process launch policy for the first stdio Runtime Host."""

    executable: Path
    arguments: tuple[str, ...] = (
        "--provider",
        "synthetic-counter",
        "--transport",
        "stdio",
    )
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    request_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        executable = Path(self.executable)
        if not executable.is_absolute():
            raise ValueError("Runtime Host executable must be an explicit absolute path")
        if not executable.is_file():
            raise ValueError("Runtime Host executable must name an existing file")
        arguments = tuple(self.arguments)
        if any(not argument or "\x00" in argument or "\n" in argument for argument in arguments):
            raise ValueError("Runtime Host arguments must be non-empty single-line strings")
        if not 1 <= self.max_frame_bytes <= HARD_MAX_FRAME_BYTES:
            raise ValueError(f"max_frame_bytes must be between 1 and {HARD_MAX_FRAME_BYTES}")
        if not 0 < self.request_timeout_seconds <= 300:
            raise ValueError("request_timeout_seconds must be in the range (0, 300]")
        object.__setattr__(self, "executable", executable)
        object.__setattr__(self, "arguments", arguments)


_ReaderItem = bytes | BaseException | None


class JsonLineHostChannel:
    """Bounded, deadline-aware JSON-lines channel over an owned subprocess."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        max_frame_bytes: int,
        request_timeout_seconds: float,
    ) -> None:
        if process.stdin is None or process.stdout is None:
            raise ValueError("Runtime Host process requires piped stdin and stdout")
        self._process = process
        self._stdin: IO[bytes] = process.stdin
        self._stdout: IO[bytes] = process.stdout
        self._max_frame_bytes = max_frame_bytes
        self._request_timeout_seconds = request_timeout_seconds
        self._responses: queue.Queue[_ReaderItem] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_responses,
            name="glr-host-stdio-reader",
            daemon=True,
        )
        self._reader.start()

    @classmethod
    def open(cls, config: HostProcessConfig) -> JsonLineHostChannel:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                [str(config.executable), *config.arguments],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=False,
                env={},
                creationflags=creation_flags,
            )
        except OSError as error:
            raise HostProtocolError("failed to start the configured Runtime Host") from error
        return cls(
            process,
            max_frame_bytes=config.max_frame_bytes,
            request_timeout_seconds=config.request_timeout_seconds,
        )

    def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
        with self._lock:
            if self._closed:
                raise HostProtocolError("Runtime Host channel is closed")
            try:
                frame = json.dumps(dict(request), separators=(",", ":"), allow_nan=False).encode(
                    "utf-8"
                )
            except (TypeError, ValueError) as error:
                raise HostProtocolError("Runtime Host request is not strict JSON") from error
            if len(frame) > self._max_frame_bytes:
                raise HostProtocolError(
                    f"Runtime Host request exceeds {self._max_frame_bytes} bytes"
                )
            try:
                self._stdin.write(frame + b"\n")
                self._stdin.flush()
            except OSError as error:
                self._shutdown_locked(force=True)
                raise HostProtocolError("Runtime Host request write failed") from error
            try:
                item = self._responses.get(timeout=self._request_timeout_seconds)
            except queue.Empty as error:
                self._shutdown_locked(force=True)
                raise HostProtocolError("Runtime Host response deadline expired") from error
            if item is None:
                self._shutdown_locked(force=True)
                raise HostProtocolError("Runtime Host closed before returning a response")
            if isinstance(item, BaseException):
                self._shutdown_locked(force=True)
                raise HostProtocolError("Runtime Host response stream failed") from item
            try:
                response = json.loads(item)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                self._shutdown_locked(force=True)
                raise HostProtocolError("Runtime Host response is not valid UTF-8 JSON") from error
            try:
                return _mapping(response, path="response")
            except HostProtocolError:
                self._shutdown_locked(force=True)
                raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._shutdown_locked(force=False)

    def _shutdown_locked(self, *, force: bool) -> None:
        self._closed = True
        with suppress(OSError):
            self._stdin.close()
        if force and self._process.poll() is None:
            with suppress(OSError):
                self._process.terminate()
        try:
            self._process.wait(timeout=self._request_timeout_seconds)
        except subprocess.TimeoutExpired:
            self._process.kill()
            # A request deadline can be only a few milliseconds. Windows process
            # teardown is not a protocol request and needs enough time to reap the
            # killed child under test/coverage load.
            self._process.wait(timeout=max(self._request_timeout_seconds, 1.0))

    def _read_responses(self) -> None:
        try:
            while True:
                line = self._stdout.readline(self._max_frame_bytes + 2)
                if not line:
                    self._put_reader_item(None)
                    return
                if len(line) > self._max_frame_bytes + 1 or not line.endswith(b"\n"):
                    self._put_reader_item(
                        HostProtocolError(
                            f"Runtime Host response exceeds {self._max_frame_bytes} bytes"
                        )
                    )
                    return
                self._put_reader_item(line[:-1].removesuffix(b"\r"))
        except BaseException as error:  # pragma: no cover - OS pipe failures are platform-specific
            self._put_reader_item(error)

    def _put_reader_item(self, item: _ReaderItem) -> None:
        try:
            self._responses.put_nowait(item)
        except queue.Full:  # pragma: no cover - the channel permits one in-flight request
            return


class HostBridgeDriver:
    """Map one Runtime Host session to the transport-neutral BridgeDriver port."""

    def __init__(self, channel: HostChannel) -> None:
        self._channel = channel
        self._request_sequence = 0
        self._closed = False
        try:
            result = self._request("describe", {})
            self._spec = _environment_spec_from_wire(result)
        except Exception:
            self._closed = True
            channel.close()
            raise

    @classmethod
    def from_process(cls, config: HostProcessConfig) -> HostBridgeDriver:
        return cls(JsonLineHostChannel.open(config))

    def describe(self) -> EnvironmentSpec:
        self._ensure_open()
        return self._spec

    def reset(self, request: BridgeResetRequest) -> TimeStep:
        payload: dict[str, object] = {"options": dict(request.options)}
        if request.seed is not None:
            payload["seed"] = request.seed
        return self._time_step(self._request("reset", payload))

    def attach(self, request: BridgeAttachRequest) -> TimeStep:
        return self._time_step(self._request("attach", {"options": dict(request.options)}))

    def step(self, request: BridgeStepRequest) -> TimeStep:
        return self._time_step(
            self._request(
                "step",
                {
                    "episode_id": str(request.episode_id),
                    "expected_step_id": request.expected_step_id,
                    "action": _tree_to_wire(request.action),
                },
            )
        )

    def resume(self, request: BridgeResumeRequest) -> BridgeResumeResult:
        payload: dict[str, object] = {
            "episode_id": str(request.episode_id),
            "last_committed_step_id": request.last_committed_step_id,
        }
        if request.target_id is not None:
            payload["target_id"] = request.target_id
        return self._resume_result(self._request("resume", payload))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._request("close", {})
        finally:
            self._closed = True
            self._channel.close()

    def _request(self, operation: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        self._ensure_open()
        self._request_sequence += 1
        request_id = f"py-{self._request_sequence}"
        response = self._channel.exchange(
            {
                "schema": HOST_SCHEMA,
                "request_id": request_id,
                "operation": operation,
                "payload": dict(payload),
            }
        )
        if response.get("schema") != HOST_SCHEMA:
            raise HostProtocolError("Runtime Host response schema does not match glr.host.v1")
        if response.get("request_id") != request_id:
            raise HostProtocolError("Runtime Host response request_id does not match")
        ok = response.get("ok")
        if not isinstance(ok, bool):
            raise HostProtocolError("Runtime Host response ok must be bool")
        if not ok:
            error = _mapping(response.get("error"), path="response.error")
            code = _string(error.get("code"), path="response.error.code")
            message = _string(error.get("message"), path="response.error.message")
            retryable = error.get("retryable")
            if not isinstance(retryable, bool):
                raise HostProtocolError("response.error.retryable must be bool")
            raise HostRemoteError(code=code, message=message, retryable=retryable)
        return _mapping(response.get("result"), path="response.result")

    def _time_step(self, value: Mapping[str, object]) -> TimeStep:
        action_mask = _tree_from_wire(
            _mapping(value.get("action_mask", {}), path="timestep.action_mask")
        )
        if self._spec.action_mask is None:
            parsed_action_mask: TensorTree | None = action_mask if action_mask else None
        else:
            parsed_action_mask = action_mask
        events_raw = value.get("events", [])
        if not isinstance(events_raw, list):
            raise HostProtocolError("timestep.events must be a list")
        events: list[Event] = []
        for index, raw_event in enumerate(events_raw):
            event = _mapping(raw_event, path=f"timestep.events[{index}]")
            payload = _mapping(event.get("payload", {}), path=f"timestep.events[{index}].payload")
            events.append(
                Event(
                    name=_string(event.get("name"), path=f"timestep.events[{index}].name"),
                    timestamp_ns=_non_negative_int(
                        event.get("timestamp_ns"),
                        path=f"timestep.events[{index}].timestamp_ns",
                    ),
                    payload=payload,
                )
            )
        episode_id = UUID(_string(value.get("episode_id"), path="timestep.episode_id"))
        step_id = _non_negative_int(value.get("step_id"), path="timestep.step_id")
        action_receipt = _action_receipt_from_wire(
            value.get("action_receipt"), episode_id=episode_id, step_id=step_id
        )
        return TimeStep(
            episode_id=episode_id,
            step_id=step_id,
            timestamp_ns=_non_negative_int(value.get("timestamp_ns"), path="timestep.timestamp_ns"),
            observation=_tree_from_wire(
                _mapping(value.get("observation"), path="timestep.observation")
            ),
            reward=_tensor_from_wire(value.get("reward"), path="timestep.reward"),
            terminated=_tensor_from_wire(value.get("terminated"), path="timestep.terminated"),
            truncated=_tensor_from_wire(value.get("truncated"), path="timestep.truncated"),
            action_mask=parsed_action_mask,
            action_receipt=action_receipt,
            events=tuple(events),
            info=_mapping(value.get("info", {}), path="timestep.info"),
        )

    def _resume_result(self, value: Mapping[str, object]) -> BridgeResumeResult:
        timestep = self._time_step(_mapping(value.get("timestep"), path="resume.timestep"))
        committed_step_id = _non_negative_int(
            value.get("committed_step_id"), path="resume.committed_step_id"
        )
        if timestep.step_id != committed_step_id:
            raise HostProtocolError("resume committed_step_id does not match timestep.step_id")
        raw_reconciliation = value.get("reconciliation")
        reconciliation = None
        if raw_reconciliation is not None:
            item = _mapping(raw_reconciliation, path="resume.reconciliation")
            try:
                retryable = item.get("retryable", False)
                if not isinstance(retryable, bool):
                    raise HostProtocolError("resume.reconciliation.retryable must be bool")
                reconciliation = ActionReconciliation(
                    episode_id=UUID(
                        _string(
                            item.get("episode_id"),
                            path="resume.reconciliation.episode_id",
                        )
                    ),
                    expected_step_id=_positive_int(
                        item.get("expected_step_id"),
                        path="resume.reconciliation.expected_step_id",
                    ),
                    outcome=ReconciliationOutcome(
                        _string(item.get("outcome"), path="resume.reconciliation.outcome")
                    ),
                    authoritative_step_id=_non_negative_int(
                        item.get("authoritative_step_id"),
                        path="resume.reconciliation.authoritative_step_id",
                    ),
                    timestamp_ns=_non_negative_int(
                        item.get("timestamp_ns"),
                        path="resume.reconciliation.timestamp_ns",
                    ),
                    retryable=retryable,
                )
            except (TypeError, ValueError) as error:
                raise HostProtocolError(f"invalid resume.reconciliation: {error}") from error
            if reconciliation.episode_id != timestep.episode_id:
                raise HostProtocolError("resume reconciliation episode_id does not match timestep")
            if reconciliation.authoritative_step_id != committed_step_id:
                raise HostProtocolError(
                    "resume reconciliation authoritative_step_id does not match committed_step_id"
                )
        return BridgeResumeResult(
            timestep=timestep,
            committed_step_id=committed_step_id,
            reconciliation=reconciliation,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise HostProtocolError("Runtime Host driver is closed")


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise HostProtocolError(f"{path} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise HostProtocolError(f"{path} must be a non-empty string")
    return value


def _non_negative_int(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise HostProtocolError(f"{path} must be a non-negative integer")
    return value


def _action_receipt_from_wire(
    value: object, *, episode_id: UUID, step_id: int
) -> ActionReceipt | None:
    if value is None:
        return None
    receipt = _mapping(value, path="timestep.action_receipt")
    try:
        receipt_episode_id = UUID(
            _string(receipt.get("episode_id"), path="timestep.action_receipt.episode_id")
        )
        progress_delta = receipt.get("progress_delta")
        if progress_delta is not None and (
            isinstance(progress_delta, bool) or not isinstance(progress_delta, (int, float))
        ):
            raise HostProtocolError(
                "timestep.action_receipt.progress_delta must be numeric or null"
            )
        retryable = receipt.get("retryable", False)
        if not isinstance(retryable, bool):
            raise HostProtocolError("timestep.action_receipt.retryable must be bool")
        parsed = ActionReceipt(
            action_id=_string(receipt.get("action_id"), path="timestep.action_receipt.action_id"),
            episode_id=receipt_episode_id,
            step_id=_positive_int(receipt.get("step_id"), path="timestep.action_receipt.step_id"),
            outcome=ActionOutcome(
                _string(receipt.get("outcome"), path="timestep.action_receipt.outcome")
            ),
            issued_timestamp_ns=_non_negative_int(
                receipt.get("issued_timestamp_ns"),
                path="timestep.action_receipt.issued_timestamp_ns",
            ),
            observed_timestamp_ns=_non_negative_int(
                receipt.get("observed_timestamp_ns"),
                path="timestep.action_receipt.observed_timestamp_ns",
            ),
            postcondition=_string(
                receipt.get("postcondition", "unknown"),
                path="timestep.action_receipt.postcondition",
            ),
            progress_delta=progress_delta,
            authoritative_observation_sequence=(
                _non_negative_int(
                    receipt["authoritative_observation_sequence"],
                    path="timestep.action_receipt.authoritative_observation_sequence",
                )
                if receipt.get("authoritative_observation_sequence") is not None
                else None
            ),
            retryable=retryable,
        )
    except (TypeError, ValueError) as error:
        raise HostProtocolError(f"invalid timestep.action_receipt: {error}") from error
    if parsed.episode_id != episode_id or parsed.step_id != step_id:
        raise HostProtocolError("timestep.action_receipt does not match the timestep identity")
    return parsed


def _positive_int(value: object, *, path: str) -> int:
    result = _non_negative_int(value, path=path)
    if result == 0:
        raise HostProtocolError(f"{path} must be positive")
    return result


def _tensor_from_wire(value: object, *, path: str) -> NDArray[Any]:
    tensor = _mapping(value, path=path)
    expected_keys = {"shape", "dtype", "data"}
    if set(tensor) != expected_keys:
        raise HostProtocolError(f"{path} fields must be {sorted(expected_keys)}")
    shape_raw = tensor["shape"]
    if not isinstance(shape_raw, list):
        raise HostProtocolError(f"{path}.shape must be a list")
    shape = tuple(
        _non_negative_int(dimension, path=f"{path}.shape[{index}]")
        for index, dimension in enumerate(shape_raw)
    )
    dtype_name = _string(tensor["dtype"], path=f"{path}.dtype")
    try:
        dtype = _WIRE_TO_DTYPE[dtype_name]
    except KeyError as error:
        raise HostProtocolError(f"{path}.dtype is unsupported: {dtype_name}") from error
    data = _string(tensor["data"], path=f"{path}.data")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HostProtocolError(f"{path}.data is not valid base64") from error
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise HostProtocolError(f"{path} has {len(raw)} bytes; expected {expected_bytes}")
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if dtype == np.dtype(np.bool_) and np.any(np.frombuffer(raw, dtype=np.uint8) > 1):
        raise HostProtocolError(f"{path} bool data must contain only zero or one")
    return array


def _tree_from_wire(value: Mapping[str, object]) -> TensorTree:
    root: dict[str, Any] = {}
    for path, tensor in value.items():
        parts = path.split(".")
        if any(not part for part in parts):
            raise HostProtocolError("tensor paths require non-empty dot-separated fields")
        current = root
        for part in parts[:-1]:
            existing = current.setdefault(part, {})
            if not isinstance(existing, dict):
                raise HostProtocolError(f"tensor path conflicts at {path}")
            current = existing
        if parts[-1] in current:
            raise HostProtocolError(f"duplicate tensor path: {path}")
        current[parts[-1]] = _tensor_from_wire(tensor, path=f"tensor[{path}]")
    return root


def _tree_to_wire(tree: TensorTree) -> dict[str, object]:
    flattened: dict[str, object] = {}

    def visit(value: Mapping[str, Any], prefix: str) -> None:
        for name, item in value.items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(item, Mapping):
                visit(item, path)
                continue
            array = np.asarray(item)
            try:
                wire_dtype = _DTYPE_TO_WIRE[array.dtype]
            except KeyError as error:
                raise HostProtocolError(
                    f"action tensor {path} has unsupported dtype {array.dtype}"
                ) from error
            wire_array = np.ascontiguousarray(array, dtype=_WIRE_TO_DTYPE[wire_dtype])
            flattened[path] = {
                "shape": list(wire_array.shape),
                "dtype": wire_dtype,
                "data": base64.b64encode(wire_array.tobytes()).decode("ascii"),
            }

    visit(tree, "")
    return flattened


def _tensor_spec_from_wire(value: object, *, path: str) -> tuple[str, TensorSpec]:
    spec = _mapping(value, path=path)
    field_path = _string(spec.get("path"), path=f"{path}.path")
    shape_raw = spec.get("shape")
    if not isinstance(shape_raw, list):
        raise HostProtocolError(f"{path}.shape must be a list")
    shape: list[int | None] = []
    for index, dimension in enumerate(shape_raw):
        if dimension == -1:
            shape.append(None)
        else:
            shape.append(_non_negative_int(dimension, path=f"{path}.shape[{index}]"))
    dtype_name = _string(spec.get("dtype"), path=f"{path}.dtype")
    try:
        dtype = _WIRE_TO_DTYPE[dtype_name]
    except KeyError as error:
        raise HostProtocolError(f"{path}.dtype is unsupported: {dtype_name}") from error
    try:
        kind = SpaceKind(_string(spec.get("kind"), path=f"{path}.kind"))
    except ValueError as error:
        raise HostProtocolError(f"{path}.kind is unsupported") from error
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    if minimum is not None and (not isinstance(minimum, (int, float)) or isinstance(minimum, bool)):
        raise HostProtocolError(f"{path}.minimum must be numeric or null")
    if maximum is not None and (not isinstance(maximum, (int, float)) or isinstance(maximum, bool)):
        raise HostProtocolError(f"{path}.maximum must be numeric or null")
    description = spec.get("description", "")
    if not isinstance(description, str):
        raise HostProtocolError(f"{path}.description must be a string")
    return field_path, TensorSpec(
        shape=tuple(shape),
        dtype=dtype,
        kind=kind,
        minimum=minimum,
        maximum=maximum,
        description=description,
    )


def _composite_spec_from_wire(value: object, *, path: str) -> CompositeSpec:
    if not isinstance(value, list) or not value:
        raise HostProtocolError(f"{path} must be a non-empty list")
    root: dict[str, Any] = {}
    for index, item in enumerate(value):
        field_path, spec = _tensor_spec_from_wire(item, path=f"{path}[{index}]")
        parts = field_path.split(".")
        if any(not part for part in parts):
            raise HostProtocolError(f"{path}[{index}].path is invalid")
        current = root
        for part in parts[:-1]:
            existing = current.setdefault(part, {})
            if not isinstance(existing, dict):
                raise HostProtocolError(f"spec path conflicts at {field_path}")
            current = existing
        if parts[-1] in current:
            raise HostProtocolError(f"duplicate spec path: {field_path}")
        current[parts[-1]] = spec

    def build(node: Mapping[str, Any]) -> CompositeSpec:
        return CompositeSpec(
            {
                name: build(child) if isinstance(child, Mapping) else child
                for name, child in node.items()
            }
        )

    return build(root)


def _environment_spec_from_wire(value: Mapping[str, object]) -> EnvironmentSpec:
    action_masks = value.get("action_masks", [])
    if not isinstance(action_masks, list):
        raise HostProtocolError("descriptor.action_masks must be a list")
    capabilities_raw = value.get("capabilities", [])
    if not isinstance(capabilities_raw, list) or any(
        not isinstance(capability, str) or not capability for capability in capabilities_raw
    ):
        raise HostProtocolError("descriptor.capabilities must be a list of strings")
    metadata = _mapping(value.get("metadata", {}), path="descriptor.metadata")
    if any(not isinstance(item, str) for item in metadata.values()):
        raise HostProtocolError("descriptor.metadata values must be strings")
    _, reward = _tensor_spec_from_wire(value.get("reward"), path="descriptor.reward")
    _, done = _tensor_spec_from_wire(value.get("done"), path="descriptor.done")
    return EnvironmentSpec(
        environment_id=_string(value.get("environment_id"), path="descriptor.environment_id"),
        protocol_version=_string(value.get("protocol_version"), path="descriptor.protocol_version"),
        observation=_composite_spec_from_wire(
            value.get("observations"), path="descriptor.observations"
        ),
        action=_composite_spec_from_wire(value.get("actions"), path="descriptor.actions"),
        action_mask=(
            _composite_spec_from_wire(action_masks, path="descriptor.action_masks")
            if action_masks
            else None
        ),
        reward=reward,
        done=done,
        capabilities=frozenset(cast(Sequence[str], capabilities_raw)),
        metadata=cast(Mapping[str, str], metadata),
    )


__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "HARD_MAX_FRAME_BYTES",
    "HOST_SCHEMA",
    "HostBridgeDriver",
    "HostChannel",
    "HostProcessConfig",
    "JsonLineHostChannel",
]
