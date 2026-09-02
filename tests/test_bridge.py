from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime import (
    BridgeAttachRequest,
    BridgeEnvironment,
    BridgeResetRequest,
    BridgeResumeRequest,
    BridgeResumeResult,
    BridgeStepRequest,
    ContractEnvironment,
    ContractViolation,
    EnvironmentBridgeDriver,
    InputLeaseOperation,
    InputLeaseReceipt,
    InputLeaseRequest,
    InputLeaseStatus,
    InputLeaseToken,
    RealtimeTimingContract,
)
from game_learning_runtime.contracts import ActionReconciliation, ReconciliationOutcome, TimeStep
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.specs import EnvironmentSpec


class _ScriptedDriver:
    def __init__(self, *, protocol_version: str = "1.0") -> None:
        source = CounterEnvironment(target=1).spec
        self._spec = EnvironmentSpec(
            environment_id="example.remote-v1",
            observation=source.observation,
            action=source.action,
            reward=source.reward,
            done=source.done,
            action_mask=source.action_mask,
            protocol_version=protocol_version,
            capabilities=frozenset({"reset", "step"}),
            metadata={"private_origin": "must-not-cross-the-default-boundary"},
        )
        self._delegate = CounterEnvironment(target=1)
        self.reset_requests: list[BridgeResetRequest] = []
        self.step_requests: list[BridgeStepRequest] = []
        self.resume_requests: list[BridgeResumeRequest] = []
        self._current: TimeStep | None = None
        self.close_count = 0

    def describe(self) -> EnvironmentSpec:
        return self._spec

    def reset(self, request: BridgeResetRequest) -> TimeStep:
        self.reset_requests.append(request)
        self._current = self._delegate.reset(seed=request.seed, options=request.options)
        return self._current

    def attach(self, request: BridgeAttachRequest) -> TimeStep:
        self._current = self._delegate.reset(options=request.options)
        return self._current

    def step(self, request: BridgeStepRequest) -> TimeStep:
        self.step_requests.append(request)
        self._current = self._delegate.step(request.action)
        return self._current

    def resume(self, request: BridgeResumeRequest) -> BridgeResumeResult:
        self.resume_requests.append(request)
        if self._current is None:
            raise ContractViolation("resume requires reset first")
        return BridgeResumeResult(self._current, self._current.step_id)

    def close(self) -> None:
        self.close_count += 1
        self._delegate.close()


def _action() -> Mapping[str, np.ndarray[Any, Any]]:
    return {"choice": np.array([1], dtype=np.int64)}


def test_bridge_environment_reuses_driver_without_leaking_metadata() -> None:
    driver = _ScriptedDriver()
    environment = ContractEnvironment(BridgeEnvironment(driver))

    initial = environment.reset(seed=7, options={"profile": "synthetic"})
    terminal = environment.step(_action())
    environment.close()

    assert environment.spec.environment_id == "example.remote-v1"
    assert environment.spec.metadata == {}
    assert driver.reset_requests == [BridgeResetRequest(seed=7, options={"profile": "synthetic"})]
    request = driver.step_requests[0]
    assert request.episode_id == initial.episode_id
    assert request.expected_step_id == 1
    np.testing.assert_array_equal(request.action["choice"], np.array([1], dtype=np.int64))
    assert terminal.done
    assert driver.close_count == 1


def test_bridge_environment_rejects_protocol_mismatch_and_closes_driver() -> None:
    driver = _ScriptedDriver(protocol_version="2.0")

    with pytest.raises(ContractViolation, match="protocol version"):
        BridgeEnvironment(driver)

    assert driver.close_count == 1


def test_bridge_environment_requires_declared_live_capabilities() -> None:
    driver = _ScriptedDriver()

    with pytest.raises(ContractViolation, match="missing required capabilities"):
        BridgeEnvironment(
            driver,
            required_capabilities={"authenticated", "target-bound"},
        )

    assert driver.close_count == 1


def test_bridge_requests_are_immutable_copies() -> None:
    options: dict[str, str] = {"profile": "synthetic"}
    reset = BridgeResetRequest(seed=None, options=options)
    options["profile"] = "changed"

    action = np.array([1], dtype=np.int64)
    step = BridgeStepRequest(
        episode_id=UUID(int=1),
        expected_step_id=1,
        action={"choice": action},
    )
    action[0] = 0

    assert reset.options == {"profile": "synthetic"}
    np.testing.assert_array_equal(step.action["choice"], np.array([1], dtype=np.int64))
    assert not step.action["choice"].flags.writeable


def test_bridge_environment_requires_reset_before_remote_step() -> None:
    driver = _ScriptedDriver()
    environment = BridgeEnvironment(driver)

    with pytest.raises(ContractViolation, match="reset first"):
        environment.step(_action())

    assert driver.step_requests == []
    environment.close()


def test_bridge_request_boundaries_match_the_wire_contract() -> None:
    with pytest.raises(ValueError, match="seed"):
        BridgeResetRequest(seed=-1)
    with pytest.raises(TypeError, match="string values"):
        BridgeResetRequest(options={"profile": 1})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="string keys"):
        BridgeResetRequest(options={1: "synthetic"})  # type: ignore[dict-item]
    attach_options: dict[str, str] = {"mode": "continue"}
    attach = BridgeAttachRequest(options=attach_options)
    attach_options["mode"] = "changed"
    assert attach.options == {"mode": "continue"}
    with pytest.raises(ValueError, match="expected_step_id"):
        BridgeStepRequest(
            episode_id=UUID(int=1),
            expected_step_id=0,
            action=_action(),
        )
    with pytest.raises(TypeError, match="episode_id"):
        BridgeStepRequest("episode.one", 1, _action())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="action_id"):
        BridgeStepRequest(UUID(int=1), 1, _action(), action_id="")
    with pytest.raises(ValueError, match="issued_at_ns"):
        BridgeStepRequest(UUID(int=1), 1, _action(), issued_at_ns=-1)
    with pytest.raises(ValueError, match="provided together"):
        BridgeStepRequest(UUID(int=1), 1, _action(), deadline_ns=1)
    with pytest.raises(TypeError, match="InputLeaseToken"):
        BridgeStepRequest(UUID(int=1), 1, _action(), lease="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cancellation_token"):
        BridgeStepRequest(UUID(int=1), 1, _action(), cancellation_token="")


def test_bridge_environment_rejects_a_stale_remote_receipt() -> None:
    class _StaleDriver(_ScriptedDriver):
        def step(self, request: BridgeStepRequest) -> TimeStep:
            result = super().step(request)
            return TimeStep(
                observation=result.observation,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                episode_id=result.episode_id,
                step_id=request.expected_step_id + 1,
            )

    driver = _StaleDriver()
    environment = BridgeEnvironment(driver)
    environment.reset()

    with pytest.raises(ContractViolation, match="expected step 1"):
        environment.step(_action())

    assert len(driver.step_requests) == 1
    environment.close()


def test_bridge_close_is_idempotent_and_prevents_new_requests() -> None:
    driver = _ScriptedDriver()
    environment = BridgeEnvironment(driver)

    environment.close()
    environment.close()

    assert driver.close_count == 1
    with pytest.raises(ContractViolation, match="closed"):
        environment.reset()


def test_environment_bridge_driver_round_trips_the_standard_contract() -> None:
    source = CounterEnvironment(target=1)
    driver = EnvironmentBridgeDriver(source)
    environment = ContractEnvironment(BridgeEnvironment(driver))

    initial = environment.reset(seed=11)
    terminal = environment.step(_action())
    environment.close()

    assert initial.step_id == 0
    assert terminal.step_id == 1
    assert terminal.episode_id == initial.episode_id
    assert terminal.done


def test_environment_bridge_driver_rejects_stale_request_before_action() -> None:
    source = CounterEnvironment(target=2)
    driver = EnvironmentBridgeDriver(source)
    initial = driver.reset(BridgeResetRequest())

    with pytest.raises(ContractViolation, match="episode_id"):
        driver.step(
            BridgeStepRequest(
                episode_id=UUID(int=0),
                expected_step_id=1,
                action=_action(),
            )
        )
    with pytest.raises(ContractViolation, match="expected_step_id"):
        driver.step(
            BridgeStepRequest(
                episode_id=initial.episode_id,
                expected_step_id=2,
                action=_action(),
            )
        )

    current = driver.step(
        BridgeStepRequest(
            episode_id=initial.episode_id,
            expected_step_id=1,
            action=_action(),
        )
    )
    assert current.step_id == 1
    driver.close()


def test_environment_bridge_driver_fences_realtime_deadline_cancel_and_lease() -> None:
    source = CounterEnvironment(target=3)
    base = source.spec
    source._spec = EnvironmentSpec(
        environment_id=base.environment_id,
        observation=base.observation,
        action=base.action,
        reward=base.reward,
        done=base.done,
        action_mask=base.action_mask,
        capabilities=base.capabilities | frozenset({"realtime", "input-lease"}),
        realtime_timing=RealtimeTimingContract(1, 100, 200, 20),
    )
    driver = EnvironmentBridgeDriver(source)
    initial = driver.reset(BridgeResetRequest())
    lease = driver.lease(
        InputLeaseRequest(
            InputLeaseOperation.ACQUIRE,
            "session.one",
            "target.game",
            expires_at_ns=10**20,
        )
    )
    assert lease.status is InputLeaseStatus.ACQUIRED
    assert lease.token is not None

    driver.cancel("action.cancelled")
    with pytest.raises(ContractViolation, match="cancelled"):
        driver.step(
            BridgeStepRequest(
                initial.episode_id,
                1,
                _action(),
                action_id="action.cancelled",
            )
        )
    with pytest.raises(ContractViolation, match="expired"):
        driver.step(
            BridgeStepRequest(
                initial.episode_id,
                1,
                _action(),
                action_id="action.expired",
                issued_at_ns=0,
                deadline_ns=1,
                quantum_ns=1,
            )
        )
    with pytest.raises(ContractViolation, match="lease"):
        driver.step(
            BridgeStepRequest(
                initial.episode_id,
                1,
                _action(),
                action_id="action.stale",
                issued_at_ns=10**20,
                deadline_ns=100,
                quantum_ns=1,
                lease=InputLeaseToken("session.one.lease", "session.one", "target.other"),
            )
        )
    current = driver.step(
        BridgeStepRequest(
            initial.episode_id,
            1,
            _action(),
            action_id="action.valid",
            issued_at_ns=0,
        )
    )
    assert current.step_id == 1
    driver.close()


def test_environment_bridge_driver_rejects_unsupported_and_invalid_timing() -> None:
    plain = EnvironmentBridgeDriver(CounterEnvironment(target=2))
    initial = plain.reset(BridgeResetRequest())
    with pytest.raises(ContractViolation, match="does not advertise"):
        plain.step(BridgeStepRequest(initial.episode_id, 1, _action(), deadline_ns=1, quantum_ns=1))
    with pytest.raises(ValueError, match="action_id"):
        plain.cancel("")
    plain.close()
    plain.close()
    with pytest.raises(ContractViolation, match="closed"):
        plain.describe()

    source = CounterEnvironment(target=2)
    base = source.spec
    source._spec = EnvironmentSpec(
        environment_id=base.environment_id,
        observation=base.observation,
        action=base.action,
        reward=base.reward,
        done=base.done,
        action_mask=base.action_mask,
        realtime_timing=RealtimeTimingContract(1, 10, 20, 5),
    )
    driver = EnvironmentBridgeDriver(source)
    initial = driver.reset(BridgeResetRequest())
    with pytest.raises(ContractViolation, match="exceeds realtime timing"):
        driver.step(
            BridgeStepRequest(
                initial.episode_id,
                1,
                _action(),
                action_id="action.bad",
                issued_at_ns=10**20,
                deadline_ns=21,
                quantum_ns=1,
            )
        )
    driver.close()


def test_bridge_environment_exposes_realtime_contract_and_optional_ports() -> None:
    class _RealtimeDriver(_ScriptedDriver):
        def __init__(self) -> None:
            super().__init__()
            source = self._spec
            self._spec = EnvironmentSpec(
                environment_id=source.environment_id,
                observation=source.observation,
                action=source.action,
                reward=source.reward,
                done=source.done,
                action_mask=source.action_mask,
                protocol_version=source.protocol_version,
                capabilities=source.capabilities | frozenset({"realtime"}),
                realtime_timing=RealtimeTimingContract(1, 100, 200, 20),
            )
            self.cancelled: list[str] = []

        def lease(self, request: InputLeaseRequest):
            return InputLeaseReceipt(
                InputLeaseStatus.REJECTED,
                None,
                1,
                reason=f"unsupported {request.operation.value}",
            )

        def cancel(self, action_id: str) -> None:
            self.cancelled.append(action_id)

    driver = _RealtimeDriver()
    environment = BridgeEnvironment(driver)
    environment.reset()
    result = environment.step_realtime(
        _action(), deadline_ns=100, quantum_ns=10, action_id="action.one", issued_at_ns=1
    )
    assert result.step_id == 1
    assert driver.step_requests[-1].action_id == "action.one"
    assert driver.step_requests[-1].issued_at_ns == 1
    environment.cancel("action.two")
    assert driver.cancelled == ["action.two"]
    environment.close()

    unsupported = BridgeEnvironment(_ScriptedDriver())
    unsupported.reset()
    with pytest.raises(ContractViolation, match="does not advertise"):
        unsupported.step_realtime(_action(), deadline_ns=1, quantum_ns=1)
    with pytest.raises(ContractViolation, match="does not support input leases"):
        unsupported.lease(
            InputLeaseRequest(InputLeaseOperation.ACQUIRE, "session.one", "target.game")
        )
    with pytest.raises(ContractViolation, match="does not support realtime cancellation"):
        unsupported.cancel("action.one")
    unsupported.close()


def test_bridge_environment_reconciles_an_in_flight_action() -> None:
    class _ResumableDriver(_ScriptedDriver):
        def __init__(self) -> None:
            super().__init__()
            source = self._spec
            self._spec = EnvironmentSpec(
                environment_id=source.environment_id,
                observation=source.observation,
                action=source.action,
                reward=source.reward,
                done=source.done,
                action_mask=source.action_mask,
                protocol_version=source.protocol_version,
                capabilities=source.capabilities | frozenset({"reconnect-resume-v1"}),
            )

        def resume(self, request: BridgeResumeRequest) -> BridgeResumeResult:
            self.resume_requests.append(request)
            assert self._current is not None
            return BridgeResumeResult(
                timestep=self._current,
                committed_step_id=self._current.step_id,
                reconciliation=ActionReconciliation(
                    episode_id=request.episode_id,
                    expected_step_id=request.last_committed_step_id + 1,
                    outcome=ReconciliationOutcome.UNKNOWN,
                    authoritative_step_id=self._current.step_id,
                    timestamp_ns=99,
                ),
            )

    driver = _ResumableDriver()
    environment = BridgeEnvironment(driver)
    initial = environment.reset()

    result = environment.resume(
        episode_id=initial.episode_id,
        last_committed_step_id=0,
        target_id="runtime-1",
    )

    assert result.timestep == initial
    assert result.reconciliation is not None
    assert result.reconciliation.outcome is ReconciliationOutcome.UNKNOWN
    assert driver.resume_requests == [
        BridgeResumeRequest(
            episode_id=initial.episode_id,
            last_committed_step_id=0,
            target_id="runtime-1",
        )
    ]
    environment.close()


def test_bridge_environment_requires_resume_capability() -> None:
    driver = _ScriptedDriver()
    environment = BridgeEnvironment(driver)
    initial = environment.reset()

    with pytest.raises(ContractViolation, match="reconnect-resume-v1"):
        environment.resume(episode_id=initial.episode_id, last_committed_step_id=0)

    assert driver.resume_requests == []
    environment.close()


def test_bridge_resume_result_validates_authoritative_cursor() -> None:
    timestep = CounterEnvironment(target=1).reset()
    with pytest.raises(TypeError, match="timestep"):
        BridgeResumeResult(timestep=object(), committed_step_id=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        BridgeResumeResult(timestep=timestep, committed_step_id=-1)
    with pytest.raises(ValueError, match="match timestep"):
        BridgeResumeResult(timestep=timestep, committed_step_id=1)
    reconciliation = ActionReconciliation(
        episode_id=UUID(int=2),
        expected_step_id=1,
        outcome=ReconciliationOutcome.UNKNOWN,
        authoritative_step_id=0,
        timestamp_ns=0,
    )
    with pytest.raises(ValueError, match="episode_id"):
        BridgeResumeResult(
            timestep=timestep,
            committed_step_id=0,
            reconciliation=reconciliation,
        )

    current = replace(timestep, step_id=1)
    assert BridgeResumeResult(current, committed_step_id=1).committed_step_id == 1


def test_bridge_round_trips_declared_live_attach_without_claiming_reset() -> None:
    class _AttachEnvironment(CounterEnvironment):
        def __init__(self) -> None:
            super().__init__(target=1)
            source = self._spec
            self._spec = EnvironmentSpec(
                environment_id=source.environment_id,
                observation=source.observation,
                action=source.action,
                reward=source.reward,
                done=source.done,
                action_mask=source.action_mask,
                capabilities=frozenset({"live-attach"}),
            )

        def reset(
            self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
        ) -> TimeStep:
            del seed, options
            raise ContractViolation("physical reset is unavailable")

        def attach(self, *, options: Mapping[str, Any] | None = None) -> TimeStep:
            return super().reset(options=options)

    driver = EnvironmentBridgeDriver(_AttachEnvironment())
    environment = ContractEnvironment(BridgeEnvironment(driver))

    initial = environment.attach(options={"mode": "continue"})
    terminal = environment.step(_action())
    environment.close()

    assert initial.step_id == 0
    assert terminal.done
