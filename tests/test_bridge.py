from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime import (
    BridgeAttachRequest,
    BridgeEnvironment,
    BridgeResetRequest,
    BridgeStepRequest,
    ContractEnvironment,
    ContractViolation,
    EnvironmentBridgeDriver,
)
from game_learning_runtime.contracts import TimeStep
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
        self.close_count = 0

    def describe(self) -> EnvironmentSpec:
        return self._spec

    def reset(self, request: BridgeResetRequest) -> TimeStep:
        self.reset_requests.append(request)
        return self._delegate.reset(seed=request.seed, options=request.options)

    def attach(self, request: BridgeAttachRequest) -> TimeStep:
        return self._delegate.reset(options=request.options)

    def step(self, request: BridgeStepRequest) -> TimeStep:
        self.step_requests.append(request)
        return self._delegate.step(request.action)

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
