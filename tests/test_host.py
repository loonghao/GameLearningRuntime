from __future__ import annotations

import base64
import copy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime import (
    BridgeAttachRequest,
    BridgeEnvironment,
    BridgeResetRequest,
    BridgeResumeRequest,
    BridgeStepRequest,
    ContractEnvironment,
    HostBridgeDriver,
    HostProcessConfig,
    HostProtocolError,
    HostRemoteError,
    JsonLineHostChannel,
)
from game_learning_runtime.host import HOST_SCHEMA, HostChannel


def _tensor(value: np.ndarray[Any, Any]) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    dtype_names = {
        np.dtype(np.bool_): "bool",
        np.dtype(np.uint8): "uint8",
        np.dtype(np.int32): "int32",
        np.dtype(np.int64): "int64",
        np.dtype(np.float32): "float32",
        np.dtype(np.float64): "float64",
    }
    return {
        "shape": list(contiguous.shape),
        "dtype": dtype_names[contiguous.dtype],
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _descriptor() -> dict[str, object]:
    return {
        "environment_id": "glr.synthetic.host-client-v1",
        "protocol_version": "1.0",
        "observations": [
            {
                "path": "state.value",
                "shape": [1],
                "dtype": "int64",
                "kind": "discrete",
                "minimum": 0,
                "maximum": 2,
                "description": "Synthetic value",
            }
        ],
        "actions": [
            {
                "path": "choice",
                "shape": [1],
                "dtype": "int64",
                "kind": "discrete",
                "minimum": 0,
                "maximum": 1,
                "description": "Increment choice",
            }
        ],
        "action_masks": [
            {
                "path": "choice",
                "shape": [2],
                "dtype": "bool",
                "kind": "binary",
                "description": "Available choices",
            }
        ],
        "reward": {
            "path": "reward",
            "shape": [1],
            "dtype": "float32",
            "kind": "continuous",
            "minimum": 0,
            "maximum": 1,
        },
        "done": {
            "path": "done",
            "shape": [1],
            "dtype": "bool",
            "kind": "binary",
        },
        "capabilities": ["host-stdio", "reset", "step"],
        "metadata": {"private_origin": "filtered by BridgeEnvironment"},
    }


def _timestep(episode_id: UUID, step_id: int, *, terminal: bool) -> dict[str, object]:
    return {
        "episode_id": str(episode_id),
        "step_id": step_id,
        "timestamp_ns": 10 + step_id,
        "observation": {"state.value": _tensor(np.array([step_id], dtype=np.int64))},
        "reward": _tensor(np.array([float(terminal)], dtype=np.float32)),
        "terminated": _tensor(np.array([terminal], dtype=np.bool_)),
        "truncated": _tensor(np.array([False], dtype=np.bool_)),
        "action_mask": {"choice": _tensor(np.array([True, True], dtype=np.bool_))},
        "events": [],
        "info": {},
    }


class _ScriptedChannel(HostChannel):
    def __init__(self) -> None:
        self.episode_id = UUID("11111111-1111-4111-8111-111111111111")
        self.requests: list[Mapping[str, object]] = []
        self.closed = False
        self.remote_error: tuple[str, str] | None = None

    def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
        self.requests.append(request)
        request_id = request["request_id"]
        operation = request["operation"]
        if self.remote_error is not None:
            code, message = self.remote_error
            return {
                "schema": HOST_SCHEMA,
                "request_id": request_id,
                "ok": False,
                "error": {"code": code, "message": message, "retryable": False},
            }
        if operation == "describe":
            result = _descriptor()
        elif operation == "reset" or operation == "attach":
            result = _timestep(self.episode_id, 0, terminal=False)
        elif operation == "step":
            result = _timestep(self.episode_id, 1, terminal=True)
        elif operation == "close":
            result = {"closed": True}
        else:  # pragma: no cover - the production driver owns the operation vocabulary
            raise AssertionError(f"unexpected operation {operation}")
        return {
            "schema": HOST_SCHEMA,
            "request_id": request_id,
            "ok": True,
            "result": result,
        }

    def close(self) -> None:
        self.closed = True


def test_host_driver_maps_the_wire_contract_to_the_standard_environment() -> None:
    channel = _ScriptedChannel()
    driver = HostBridgeDriver(channel)
    environment = ContractEnvironment(BridgeEnvironment(driver))

    initial = environment.reset(seed=7, options={"profile": "synthetic"})
    terminal = environment.step({"choice": np.array([1], dtype=np.int64)})
    environment.close()

    assert environment.spec.environment_id == "glr.synthetic.host-client-v1"
    assert environment.spec.metadata == {}
    np.testing.assert_array_equal(initial.observation["state"]["value"], [0])
    np.testing.assert_array_equal(terminal.observation["state"]["value"], [1])
    assert terminal.done
    step_request = channel.requests[2]
    assert step_request["payload"] == {
        "episode_id": str(channel.episode_id),
        "expected_step_id": 1,
        "action": {"choice": _tensor(np.array([1], dtype=np.int64))},
    }
    assert channel.closed


def test_host_driver_parses_and_binds_action_receipt() -> None:
    class _ReceiptChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = copy.deepcopy(dict(super().exchange(request)))
            if request["operation"] == "step":
                result = response["result"]
                assert isinstance(result, dict)
                result["action_receipt"] = {
                    "action_id": "move-1",
                    "episode_id": str(self.episode_id),
                    "step_id": 1,
                    "outcome": "no_effect",
                    "issued_timestamp_ns": 10,
                    "observed_timestamp_ns": 20,
                    "postcondition": "blocked",
                    "progress_delta": 0.0,
                    "authoritative_observation_sequence": 3,
                    "retryable": False,
                }
            return response

    driver = HostBridgeDriver(_ReceiptChannel())
    initial = driver.reset(BridgeResetRequest())
    result = driver.step(
        BridgeStepRequest(
            episode_id=initial.episode_id,
            expected_step_id=1,
            action={"choice": np.array([1], dtype=np.int64)},
        )
    )

    assert result.action_receipt is not None
    assert result.action_receipt.outcome.value == "no_effect"
    assert result.action_receipt.authoritative_observation_sequence == 3
    driver.close()


def test_host_driver_round_trips_resume_reconciliation() -> None:
    class _ResumeChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            if request["operation"] != "resume":
                return super().exchange(request)
            self.requests.append(request)
            request_id = request["request_id"]
            return {
                "schema": HOST_SCHEMA,
                "request_id": request_id,
                "ok": True,
                "result": {
                    "timestep": _timestep(self.episode_id, 0, terminal=False),
                    "committed_step_id": 0,
                    "reconciliation": {
                        "episode_id": str(self.episode_id),
                        "expected_step_id": 1,
                        "outcome": "unknown",
                        "authoritative_step_id": 0,
                        "timestamp_ns": 42,
                        "retryable": False,
                    },
                },
            }

    channel = _ResumeChannel()
    driver = HostBridgeDriver(channel)
    initial = driver.reset(BridgeResetRequest())

    result = driver.resume(
        BridgeResumeRequest(
            episode_id=initial.episode_id,
            last_committed_step_id=0,
            target_id="runtime-1",
        )
    )

    assert result.timestep.episode_id == initial.episode_id
    assert result.timestep.step_id == initial.step_id
    assert result.reconciliation is not None
    assert result.reconciliation.outcome.value == "unknown"
    assert channel.requests[-1]["payload"] == {
        "episode_id": str(initial.episode_id),
        "last_committed_step_id": 0,
        "target_id": "runtime-1",
    }
    driver.close()


def test_host_driver_rejects_resume_reconciliation_cursor_mismatch() -> None:
    class _InvalidResumeChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            if request["operation"] == "resume":
                self.requests.append(request)
                return {
                    "schema": HOST_SCHEMA,
                    "request_id": request["request_id"],
                    "ok": True,
                    "result": {
                        "timestep": _timestep(self.episode_id, 0, terminal=False),
                        "committed_step_id": 0,
                        "reconciliation": {
                            "episode_id": str(self.episode_id),
                            "expected_step_id": 1,
                            "outcome": "applied",
                            "authoritative_step_id": 1,
                            "timestamp_ns": 42,
                        },
                    },
                }
            return super().exchange(request)

    channel = _InvalidResumeChannel()
    driver = HostBridgeDriver(channel)
    initial = driver.reset(BridgeResetRequest())

    with pytest.raises(HostProtocolError, match="authoritative_step_id"):
        driver.resume(
            BridgeResumeRequest(
                episode_id=initial.episode_id,
                last_committed_step_id=0,
            )
        )

    driver.close()


def test_host_driver_surfaces_structured_remote_errors_without_retry() -> None:
    channel = _ScriptedChannel()
    driver = HostBridgeDriver(channel)
    channel.remote_error = ("lifecycle_violation", "stale step")

    with pytest.raises(HostRemoteError, match="stale step") as captured:
        driver.reset(BridgeResetRequest())

    assert captured.value.code == "lifecycle_violation"
    assert not captured.value.retryable
    assert len(channel.requests) == 2
    channel.remote_error = None
    driver.close()


def test_host_driver_closes_a_channel_after_handshake_mismatch() -> None:
    class _WrongSchemaChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = dict(super().exchange(request))
            response["schema"] = "glr.host.v2"
            return response

    channel = _WrongSchemaChannel()

    with pytest.raises(HostProtocolError, match="schema"):
        HostBridgeDriver(channel)

    assert channel.closed


def test_process_config_requires_an_explicit_absolute_executable(tmp_path: Path) -> None:
    executable = tmp_path / "glr-hostd"
    executable.write_bytes(b"fixture")

    config = HostProcessConfig(executable=executable.resolve())

    assert config.arguments == ("--provider", "synthetic-counter", "--transport", "stdio")
    assert config.max_frame_bytes == 1_048_576
    with pytest.raises(ValueError, match="absolute"):
        HostProcessConfig(executable=Path("glr-hostd"))
    with pytest.raises(ValueError, match="max_frame_bytes"):
        HostProcessConfig(executable=executable.resolve(), max_frame_bytes=0)
    with pytest.raises(ValueError, match="existing file"):
        HostProcessConfig(executable=(tmp_path / "missing").resolve())
    with pytest.raises(ValueError, match="arguments"):
        HostProcessConfig(executable=executable.resolve(), arguments=("bad\nargument",))
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        HostProcessConfig(executable=executable.resolve(), request_timeout_seconds=0)


def test_host_driver_rejects_corrupt_tensor_lengths() -> None:
    class _CorruptChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = dict(super().exchange(request))
            if request["operation"] == "reset":
                result = dict(response["result"])  # type: ignore[arg-type]
                observation = dict(result["observation"])  # type: ignore[arg-type]
                tensor = dict(observation["state.value"])  # type: ignore[arg-type]
                tensor["data"] = "AA=="
                observation["state.value"] = tensor
                result["observation"] = observation
                response["result"] = result
            return response

    driver = HostBridgeDriver(_CorruptChannel())

    with pytest.raises(HostProtocolError, match="expected 8"):
        driver.reset(BridgeResetRequest())

    driver.close()


def test_json_line_channel_uses_an_owned_explicit_process() -> None:
    script = (
        "import json,sys;"
        "r=json.loads(sys.stdin.readline());"
        "print(json.dumps({'schema':'glr.host.v1','request_id':r['request_id'],"
        "'ok':True,'result':{'echo':True}}),flush=True)"
    )
    config = HostProcessConfig(
        executable=Path(sys.executable).resolve(),
        arguments=("-u", "-c", script),
    )
    channel = JsonLineHostChannel.open(config)

    response = channel.exchange(
        {
            "schema": HOST_SCHEMA,
            "request_id": "process-1",
            "operation": "describe",
            "payload": {},
        }
    )
    channel.close()

    assert response["request_id"] == "process-1"
    assert response["result"] == {"echo": True}


def test_json_line_channel_fails_on_a_response_deadline_without_retry() -> None:
    config = HostProcessConfig(
        executable=Path(sys.executable).resolve(),
        arguments=("-u", "-c", "import time;time.sleep(1)"),
        request_timeout_seconds=0.05,
    )
    channel = JsonLineHostChannel.open(config)

    with pytest.raises(HostProtocolError, match="deadline"):
        channel.exchange(
            {
                "schema": HOST_SCHEMA,
                "request_id": "timeout-1",
                "operation": "describe",
                "payload": {},
            }
        )

    with pytest.raises(HostProtocolError, match="closed"):
        channel.exchange({"value": "must-not-follow-an-ambiguous-request"})

    channel.close()


@pytest.mark.parametrize(
    ("script", "max_frame_bytes", "message"),
    [
        ("print('not-json',flush=True)", 1_048_576, "UTF-8 JSON"),
        ("print('x'*512,flush=True)", 256, "response stream"),
        ("pass", 1_048_576, "closed before"),
    ],
)
def test_json_line_channel_rejects_invalid_or_missing_responses(
    script: str, max_frame_bytes: int, message: str
) -> None:
    config = HostProcessConfig(
        executable=Path(sys.executable).resolve(),
        arguments=("-u", "-c", script),
        max_frame_bytes=max_frame_bytes,
    )
    channel = JsonLineHostChannel.open(config)

    with pytest.raises(HostProtocolError, match=message):
        channel.exchange(
            {
                "schema": HOST_SCHEMA,
                "request_id": "invalid-response-1",
                "operation": "describe",
                "payload": {},
            }
        )

    channel.close()
    channel.close()


def test_json_line_channel_rejects_non_json_and_oversized_requests() -> None:
    config = HostProcessConfig(
        executable=Path(sys.executable).resolve(),
        arguments=("-u", "-c", "import time;time.sleep(1)"),
        max_frame_bytes=128,
        request_timeout_seconds=0.05,
    )
    channel = JsonLineHostChannel.open(config)

    with pytest.raises(HostProtocolError, match="strict JSON"):
        channel.exchange({"value": float("nan")})
    with pytest.raises(HostProtocolError, match="exceeds"):
        channel.exchange({"value": "x" * 256})

    channel.close()
    with pytest.raises(HostProtocolError, match="closed"):
        channel.exchange({"value": "after-close"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("action_masks", {}, "action_masks"),
        ("capabilities", [""], "capabilities"),
        ("metadata", {"key": 1}, "metadata values"),
        ("observations", [], "non-empty list"),
    ],
)
def test_host_driver_rejects_invalid_descriptors(field: str, value: object, message: str) -> None:
    class _InvalidDescriptorChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = copy.deepcopy(dict(super().exchange(request)))
            if request["operation"] == "describe":
                result = response["result"]
                assert isinstance(result, dict)
                result[field] = value
            return response

    channel = _InvalidDescriptorChannel()

    with pytest.raises(HostProtocolError, match=message):
        HostBridgeDriver(channel)

    assert channel.closed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", "bad", "shape"),
        ("dtype", "complex64", "dtype is unsupported"),
        ("kind", "unknown", "kind is unsupported"),
        ("minimum", "zero", "minimum"),
        ("maximum", "one", "maximum"),
        ("description", 1, "description"),
        ("path", "state..value", "path is invalid"),
    ],
)
def test_host_driver_rejects_invalid_tensor_specs(field: str, value: object, message: str) -> None:
    class _InvalidSpecChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = copy.deepcopy(dict(super().exchange(request)))
            if request["operation"] == "describe":
                result = response["result"]
                assert isinstance(result, dict)
                observations = result["observations"]
                assert isinstance(observations, list)
                spec = observations[0]
                assert isinstance(spec, dict)
                spec[field] = value
            return response

    with pytest.raises(HostProtocolError, match=message):
        HostBridgeDriver(_InvalidSpecChannel())


def test_host_driver_accepts_dynamic_spec_and_attach_then_rejects_after_close() -> None:
    class _DynamicSpecChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = copy.deepcopy(dict(super().exchange(request)))
            if request["operation"] == "describe":
                result = response["result"]
                assert isinstance(result, dict)
                observations = result["observations"]
                assert isinstance(observations, list)
                spec = observations[0]
                assert isinstance(spec, dict)
                spec["shape"] = [-1]
            return response

    driver = HostBridgeDriver(_DynamicSpecChannel())

    assert driver.describe().observation.fields["state"].is_dynamic
    attached = driver.attach(BridgeAttachRequest())
    assert attached.step_id == 0
    driver.close()
    driver.close()
    with pytest.raises(HostProtocolError, match="closed"):
        driver.describe()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", "bad", "shape must be a list"),
        ("dtype", "complex64", "dtype is unsupported"),
        ("data", "not-base64", "valid base64"),
    ],
)
def test_host_driver_rejects_invalid_wire_tensor_fields(
    field: str, value: object, message: str
) -> None:
    class _InvalidTensorChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = copy.deepcopy(dict(super().exchange(request)))
            if request["operation"] == "reset":
                result = response["result"]
                assert isinstance(result, dict)
                observation = result["observation"]
                assert isinstance(observation, dict)
                tensor = observation["state.value"]
                assert isinstance(tensor, dict)
                tensor[field] = value
            return response

    driver = HostBridgeDriver(_InvalidTensorChannel())

    with pytest.raises(HostProtocolError, match=message):
        driver.reset(BridgeResetRequest())

    driver.close()


def test_host_driver_rejects_invalid_bool_and_conflicting_tensor_paths() -> None:
    class _InvalidTreeChannel(_ScriptedChannel):
        invalid_bool = True

        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = copy.deepcopy(dict(super().exchange(request)))
            if request["operation"] == "reset":
                result = response["result"]
                assert isinstance(result, dict)
                if self.invalid_bool:
                    terminated = result["terminated"]
                    assert isinstance(terminated, dict)
                    terminated["data"] = base64.b64encode(b"\x02").decode("ascii")
                else:
                    observation = result["observation"]
                    assert isinstance(observation, dict)
                    result["observation"] = {
                        "state": _tensor(np.array([0], dtype=np.int64)),
                        **observation,
                    }
            return response

    channel = _InvalidTreeChannel()
    driver = HostBridgeDriver(channel)
    with pytest.raises(HostProtocolError, match="zero or one"):
        driver.reset(BridgeResetRequest())
    driver.close()

    channel = _InvalidTreeChannel()
    channel.invalid_bool = False
    driver = HostBridgeDriver(channel)
    with pytest.raises(HostProtocolError, match="conflicts"):
        driver.reset(BridgeResetRequest())
    driver.close()


def test_host_driver_rejects_unsupported_action_dtype_before_exchange() -> None:
    channel = _ScriptedChannel()
    driver = HostBridgeDriver(channel)
    initial = driver.reset(BridgeResetRequest())

    with pytest.raises(HostProtocolError, match="unsupported dtype"):
        driver.step(
            BridgeStepRequest(
                episode_id=initial.episode_id,
                expected_step_id=1,
                action={"choice": np.array([1 + 0j], dtype=np.complex64)},
            )
        )

    assert len(channel.requests) == 2
    driver.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_id", "wrong", "request_id"),
        ("ok", "yes", "ok must be bool"),
    ],
)
def test_host_driver_rejects_invalid_response_envelopes(
    field: str, value: object, message: str
) -> None:
    class _InvalidEnvelopeChannel(_ScriptedChannel):
        def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
            response = dict(super().exchange(request))
            response[field] = value
            return response

    with pytest.raises(HostProtocolError, match=message):
        HostBridgeDriver(_InvalidEnvelopeChannel())
