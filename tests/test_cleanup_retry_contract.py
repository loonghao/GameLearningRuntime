"""Synthetic cleanup faults; child handles belong exclusively to these tests."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import game_learning_runtime.host as host_module
from game_learning_runtime.bridge import (
    BridgeEnvironment,
    BridgeResetRequest,
    EnvironmentBridgeDriver,
)
from game_learning_runtime.environment import ContractEnvironment
from game_learning_runtime.errors import CleanupPendingError, ContractViolation, HostProtocolError
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.host import (
    HOST_SCHEMA,
    HostBridgeDriver,
    HostProcessConfig,
    JsonLineHostChannel,
)

_REAL_POPEN = subprocess.Popen


def _descriptor() -> dict[str, object]:
    return {
        "environment_id": "synthetic.cleanup-v1",
        "protocol_version": "1.0",
        "observations": [{"path": "value", "shape": [1], "dtype": "int64", "kind": "discrete"}],
        "actions": [{"path": "choice", "shape": [1], "dtype": "int64", "kind": "discrete"}],
        "reward": {"path": "reward", "shape": [1], "dtype": "float32", "kind": "continuous"},
        "done": {"path": "done", "shape": [1], "dtype": "bool", "kind": "binary"},
        "capabilities": ["reset", "step"],
    }


def _config(tmp_path: Path, *, descriptor: dict[str, object] | None = None) -> HostProcessConfig:
    script = tmp_path / "owned_child.py"
    if descriptor is None:
        code = "import time\ntime.sleep(30)\n"
    else:
        operations = tmp_path / "owned_operations.txt"
        code = (
            "import json,sys,time\nfrom pathlib import Path\n"
            f"descriptor=json.loads({json.dumps(descriptor)!r})\n"
            "for line in sys.stdin:\n"
            " request=json.loads(line)\n"
            f" with Path({str(operations)!r}).open('a',encoding='utf-8') as out:\n"
            "  out.write(request['operation']+'\\n')\n"
            " result=descriptor if request['operation']=='describe' else {'closed':True}\n"
            " print(json.dumps({'schema':'glr.host.v1','request_id':request['request_id'],"
            "'ok':True,'result':result}),flush=True)\n"
            " if request['operation']=='close': time.sleep(30)\n"
        )
    script.write_text(code, encoding="utf-8")
    return HostProcessConfig(
        executable=Path(sys.executable).resolve(),
        arguments=("-I", "-B", "-u", str(script)),
        request_timeout_seconds=0.05 if descriptor is None else 5.0,
    )


def _finish_owned(process: subprocess.Popen[bytes], reader: threading.Thread | None = None) -> None:
    # Always retain the exact Popen created here; never discover or adopt a PID.
    if process.poll() is None:
        _REAL_POPEN.kill(process)
    _REAL_POPEN.wait(process, timeout=5.0)
    if reader is not None and reader.ident is not None:
        reader.join(timeout=5.0)
        assert not reader.is_alive()
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            with suppress(OSError):
                stream.close()
    assert process.poll() is not None


@pytest.fixture
def owned_channel(tmp_path: Path) -> Iterator[JsonLineHostChannel]:
    channel = JsonLineHostChannel.open(_config(tmp_path))
    try:
        yield channel
    finally:
        _finish_owned(channel._process, channel._reader)


def _delay_waits(
    process: subprocess.Popen[bytes], monkeypatch: pytest.MonkeyPatch, *, failures: int = 2
) -> list[float | None]:
    waits: list[float | None] = []
    real_wait = process.wait

    def uncertain_wait(timeout: float | None = None) -> int:
        waits.append(timeout)
        if len(waits) <= failures:
            raise subprocess.TimeoutExpired(process.args, timeout)
        return real_wait(timeout=timeout)

    monkeypatch.setattr(process, "wait", uncertain_wait)
    return waits


def _retry_owned_pending(
    pending: CleanupPendingError,
    process: subprocess.Popen[bytes],
    reader: threading.Thread,
) -> None:
    try:
        pending.retry_cleanup()
    except CleanupPendingError as reader_pending:
        # Scheduling can leave the Windows pipe reader alive after a confirmed
        # process wait. The SDK must report that state, rather than fake success.
        assert not pending.cleanup_complete
        assert not reader_pending.cleanup_complete
        assert process.poll() is not None
        assert process.stdout is not None and not process.stdout.closed
        # It may exit between the SDK's alive check and this assertion; neither
        # state may turn the pending callback into an implicit success.
        reader.join(timeout=5.0)  # Independent test-owner teardown, not an SDK deadline.
        assert not reader.is_alive()
        pending.retry_cleanup()
    assert pending.cleanup_complete


class _UncertainAdapter(CounterEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.closes = 0
        self.resets = 0

    def reset(self, **kwargs: Any):
        self.resets += 1
        return super().reset(**kwargs)

    def close(self) -> None:
        self.closes += 1
        if self.closes == 1:
            raise RuntimeError("synthetic adapter cleanup failure")


class _UncertainChannel:
    def __init__(self) -> None:
        self.operations: list[str] = []
        self.closes = 0

    def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
        operation = str(request["operation"])
        self.operations.append(operation)
        return {
            "schema": HOST_SCHEMA,
            "request_id": request["request_id"],
            "ok": True,
            "result": _descriptor() if operation == "describe" else {"closed": True},
        }

    def close(self) -> None:
        self.closes += 1
        if self.closes == 1:
            raise RuntimeError("synthetic channel cleanup failure")


class _RemoteCloseFailureChannel(_UncertainChannel):
    def exchange(self, request: Mapping[str, object]) -> Mapping[str, object]:
        if request["operation"] != "close":
            return super().exchange(request)
        self.operations.append("close")
        return {
            "schema": HOST_SCHEMA,
            "request_id": request["request_id"],
            "ok": False,
            "error": {
                "code": "CLOSE_FAILED",
                "message": "synthetic remote close failure",
                "retryable": False,
            },
        }


def test_remote_close_error_survives_pending_cleanup_without_resending_close() -> None:
    channel = _RemoteCloseFailureChannel()
    driver = HostBridgeDriver(channel)
    with pytest.raises(
        CleanupPendingError, match=r"remote close failure.*cleanup.*unconfirmed"
    ) as pending:
        driver.close()
    pending.value.retry_cleanup()
    driver.close()
    assert channel.operations == ["describe", "close"]
    assert channel.closes == 2
    assert pending.value.cleanup_complete


def test_pending_callback_failure_stays_unconfirmed_and_success_is_idempotent() -> None:
    calls = 0

    def cleanup() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic cleanup failure")

    error = CleanupPendingError("cleanup pending", retry_cleanup=cleanup)
    with pytest.raises(RuntimeError):
        error.retry_cleanup()
    assert not error.cleanup_complete
    error.retry_cleanup()
    error.retry_cleanup()
    assert error.cleanup_complete
    assert calls == 2
    assert error.args == ("cleanup pending",)


def test_contract_environment_fences_then_retries_original_adapter() -> None:
    adapter = _UncertainAdapter()
    environment = ContractEnvironment(adapter)
    with pytest.raises(CleanupPendingError) as pending:
        environment.close()
    with pytest.raises(ContractViolation, match="closed"):
        environment.reset()
    assert adapter.resets == 0
    pending.value.retry_cleanup()
    environment.close()
    assert pending.value.cleanup_complete
    assert adapter.closes == 2


def test_environment_driver_preserves_logical_state_until_cleanup_succeeds() -> None:
    adapter = _UncertainAdapter()
    driver = EnvironmentBridgeDriver(adapter)
    driver.reset(BridgeResetRequest())
    original_book, original_current = driver._lease_book, driver._current
    with pytest.raises(CleanupPendingError) as pending:
        driver.close()
    assert driver._lease_book is original_book
    assert driver._current is original_current
    with pytest.raises(ContractViolation, match="closed"):
        driver.reset(BridgeResetRequest())
    pending.value.retry_cleanup()
    driver.close()
    assert adapter.closes == 2
    assert adapter.resets == 1
    assert driver._current is None


def test_bridge_environment_retry_updates_its_own_cleanup_state() -> None:
    adapter = _UncertainAdapter()
    environment = BridgeEnvironment(EnvironmentBridgeDriver(adapter))
    with pytest.raises(CleanupPendingError) as pending:
        environment.close()
    with pytest.raises(ContractViolation, match="closed"):
        environment.reset()
    pending.value.retry_cleanup()
    environment.close()
    assert adapter.closes == 2
    assert adapter.resets == 0


def test_bridge_failed_validation_exposes_same_owner_cleanup_without_describe_retry() -> None:
    class WrongVersion(EnvironmentBridgeDriver):
        descriptions = 0

        def describe(self):
            self.descriptions += 1
            return replace(super().describe(), protocol_version="2.0")

    adapter = _UncertainAdapter()
    driver = WrongVersion(adapter)
    with pytest.raises(CleanupPendingError, match=r"protocol version.*unconfirmed") as pending:
        BridgeEnvironment(driver)
    pending.value.retry_cleanup()
    pending.value.retry_cleanup()
    assert pending.value.cleanup_complete
    assert adapter.closes == 2
    assert driver.descriptions == 1
    assert adapter.resets == 0


def test_host_driver_cleanup_retry_never_resends_remote_close() -> None:
    channel = _UncertainChannel()
    driver = HostBridgeDriver(channel)
    with pytest.raises(CleanupPendingError) as pending:
        driver.close()
    with pytest.raises(HostProtocolError, match="closed"):
        driver.describe()
    pending.value.retry_cleanup()
    driver.close()
    assert channel.operations == ["describe", "close"]
    assert channel.closes == 2


def test_owned_process_post_kill_wait_failure_can_be_reaped_later(
    owned_channel: JsonLineHostChannel, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = owned_channel._process
    waits = _delay_waits(process, monkeypatch)
    with pytest.raises(CleanupPendingError, match="cleanup is unconfirmed") as pending:
        owned_channel.close()
    assert not pending.value.cleanup_complete
    with pytest.raises(HostProtocolError, match="closed"):
        owned_channel.exchange({"value": "fenced"})
    _retry_owned_pending(pending.value, process, owned_channel._reader)
    owned_channel.close()
    assert pending.value.cleanup_complete
    assert waits[:3] == [0.05, 5.0, 0.05]
    assert waits[3:] in ([], [0.05])
    assert process.poll() is not None
    assert owned_channel._process is process
    assert process.stdin.closed and process.stdout.closed
    assert not owned_channel._reader.is_alive()


def test_owned_process_deadline_keeps_reason_and_pending_cleanup(
    owned_channel: JsonLineHostChannel, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = owned_channel._process
    waits = _delay_waits(process, monkeypatch)
    with pytest.raises(
        CleanupPendingError, match=r"deadline expired.*cleanup.*unconfirmed"
    ) as pending:
        owned_channel.exchange({"operation": "describe"})
    with pytest.raises(HostProtocolError, match="closed"):
        owned_channel.exchange({"operation": "step"})
    _retry_owned_pending(pending.value, process, owned_channel._reader)
    assert process.poll() is not None
    assert waits[:3] == [0.05, 5.0, 0.05]
    assert waits[3:] in ([], [0.05])


def test_owned_process_kill_failure_is_explicit_and_recoverable(
    owned_channel: JsonLineHostChannel, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_kill() -> None:
        raise OSError("synthetic owned kill failure")

    with monkeypatch.context() as faults:
        faults.setattr(owned_channel._process, "kill", fail_kill)
        with pytest.raises(CleanupPendingError) as pending:
            owned_channel.close()
        with pytest.raises(HostProtocolError, match="closed"):
            owned_channel.exchange({"value": "fenced"})
    _retry_owned_pending(pending.value, owned_channel._process, owned_channel._reader)
    assert owned_channel._process.poll() is not None


def test_reader_must_exit_before_stdout_close_or_cleanup_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()

    def blocked_reader(self: JsonLineHostChannel) -> None:
        entered.set()
        release.wait(timeout=10.0)

    monkeypatch.setattr(JsonLineHostChannel, "_read_responses", blocked_reader)
    channel = JsonLineHostChannel.open(_config(tmp_path))
    try:
        assert entered.wait(timeout=2.0)
        with pytest.raises(CleanupPendingError, match="reader cleanup is unconfirmed") as pending:
            channel.close()
        assert channel._process.poll() is not None
        assert not channel._stdout.closed
        assert channel._reader.is_alive()
        release.set()
        pending.value.retry_cleanup()
        assert not channel._reader.is_alive()
        assert channel._stdout.closed
    finally:
        release.set()
        _finish_owned(channel._process, channel._reader)


def _capture_factory(
    monkeypatch: pytest.MonkeyPatch, *, fail_start: bool = False
) -> tuple[list[subprocess.Popen[bytes]], list[threading.Thread]]:
    owned, readers = [], []
    real_start = threading.Thread.start

    def spawn(*args: Any, **kwargs: Any):
        process = _REAL_POPEN(*args, **kwargs)
        owned.append(process)
        _delay_waits(process, monkeypatch)
        return process

    def start(reader: threading.Thread) -> None:
        if reader.name == "glr-host-stdio-reader":
            readers.append(reader)
            if fail_start:
                raise RuntimeError("synthetic reader startup failure")
        real_start(reader)

    monkeypatch.setattr(host_module.subprocess, "Popen", spawn)
    monkeypatch.setattr(threading.Thread, "start", start)
    return owned, readers


def test_host_factory_retains_recovery_when_handshake_and_cleanup_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned, readers = _capture_factory(monkeypatch)
    try:
        with pytest.raises(CleanupPendingError, match="cleanup is unconfirmed") as pending:
            HostBridgeDriver.from_process(_config(tmp_path, descriptor={}))
        assert len(owned) == 1
        _retry_owned_pending(pending.value, owned[0], readers[0])
        pending.value.retry_cleanup()
        assert pending.value.cleanup_complete
        assert owned[0].poll() is not None
        assert (tmp_path / "owned_operations.txt").read_text(encoding="utf-8").splitlines() == [
            "describe"
        ]
        assert len(owned) == 1
    finally:
        for process in owned:
            _finish_owned(process, readers[0] if readers else None)


def test_reader_start_failure_retains_only_original_process_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned, readers = _capture_factory(monkeypatch, fail_start=True)
    try:
        with pytest.raises(
            CleanupPendingError, match=r"reader startup failed.*unconfirmed"
        ) as pending:
            JsonLineHostChannel.open(_config(tmp_path))
        pending.value.retry_cleanup()
        pending.value.retry_cleanup()
        assert pending.value.cleanup_complete
        assert len(owned) == len(readers) == 1
        assert readers[0].ident is None
        assert owned[0].poll() is not None
        assert owned[0].stdin.closed and owned[0].stdout.closed
    finally:
        for process in owned:
            _finish_owned(process, readers[0] if readers else None)


def test_nested_contract_host_chain_recovers_without_new_wire_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel = JsonLineHostChannel.open(_config(tmp_path, descriptor=_descriptor()))
    try:
        driver = HostBridgeDriver(channel)
        environment = ContractEnvironment(BridgeEnvironment(driver))
        _delay_waits(channel._process, monkeypatch)
        with pytest.raises(CleanupPendingError) as pending:
            environment.close()
        with pytest.raises(ContractViolation, match="closed"):
            environment.reset()
        _retry_owned_pending(pending.value, channel._process, channel._reader)
        environment.close()
        assert pending.value.cleanup_complete
        assert channel._process.poll() is not None
        assert channel._stdin.closed and channel._stdout.closed
        assert not channel._reader.is_alive()
        assert (tmp_path / "owned_operations.txt").read_text(encoding="utf-8").splitlines() == [
            "describe",
            "close",
        ]
    finally:
        _finish_owned(channel._process, channel._reader)


def test_normal_owned_child_closes_idempotently(tmp_path: Path) -> None:
    channel = JsonLineHostChannel.open(_config(tmp_path, descriptor=_descriptor()))
    try:
        driver = HostBridgeDriver(channel)
        driver.close()
        driver.close()
        assert channel._process.poll() is not None
        assert channel._stdin.closed and channel._stdout.closed
        assert not channel._reader.is_alive()
    finally:
        _finish_owned(channel._process, channel._reader)
