from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from game_learning_runtime import (
    ActionMode,
    BridgeAttachRequest,
    BridgeResetRequest,
    BridgeStepRequest,
    ClockMode,
    ContractViolation,
    EngineFamily,
    IntegrationMode,
    ObservationMode,
    RuntimeIntegrationProfile,
    TransportMode,
    load_runtime_integration,
)
from game_learning_runtime.contracts import TimeStep
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.specs import EnvironmentSpec


class _ProfileDriver:
    def __init__(self, capabilities: frozenset[str]) -> None:
        self._delegate = CounterEnvironment(target=1)
        source = self._delegate.spec
        self._spec = EnvironmentSpec(
            environment_id="example.engine-runtime-v1",
            observation=source.observation,
            action=source.action,
            reward=source.reward,
            done=source.done,
            action_mask=source.action_mask,
            protocol_version=source.protocol_version,
            capabilities=capabilities,
            metadata={"local_target": "must-not-cross"},
        )
        self.close_count = 0

    def describe(self) -> EnvironmentSpec:
        return self._spec

    def reset(self, request: BridgeResetRequest) -> TimeStep:
        return self._delegate.reset(seed=request.seed, options=request.options)

    def attach(self, request: BridgeAttachRequest) -> TimeStep:
        return self._delegate.reset(options=request.options)

    def step(self, request: BridgeStepRequest) -> TimeStep:
        return self._delegate.step(request.action)

    def close(self) -> None:
        self.close_count += 1
        self._delegate.close()


def _action() -> Mapping[str, np.ndarray[Any, Any]]:
    return {"choice": np.array([1], dtype=np.int64)}


@pytest.mark.parametrize("engine", [EngineFamily.UNITY, EngineFamily.UNREAL])
def test_source_profile_declares_a_fast_engine_native_contract(engine: EngineFamily) -> None:
    profile = RuntimeIntegrationProfile.for_source(engine)

    assert profile.integration_mode is IntegrationMode.ENGINE_PLUGIN
    assert profile.start_mode == "reset"
    assert profile.clock_mode is ClockMode.MANUAL_STEP
    assert profile.observation_mode is ObservationMode.ENGINE_STATE
    assert profile.action_mode is ActionMode.NATIVE
    assert profile.seedable
    assert {
        "authenticated",
        "deterministic-reset",
        "main-thread-dispatch",
        "manual-step",
        "native-action",
        "postcondition-verified",
        "reset",
        "semantic-observation",
        "step",
        "target-bound",
    } <= profile.required_capabilities


@pytest.mark.parametrize("engine", [EngineFamily.UNITY, EngineFamily.UNREAL])
def test_binary_only_profile_is_truthful_live_attach(engine: EngineFamily) -> None:
    profile = RuntimeIntegrationProfile.for_external(engine)

    assert profile.integration_mode is IntegrationMode.EXTERNAL_ATTACH
    assert profile.start_mode == "attach"
    assert profile.clock_mode is ClockMode.REALTIME
    assert profile.observation_mode is ObservationMode.RENDERED
    assert profile.action_mode is ActionMode.BOUNDED_INPUT
    assert not profile.seedable
    assert {
        "authenticated",
        "bounded-input",
        "input-lease",
        "live-attach",
        "postcondition-verified",
        "rendered-observation",
        "step",
        "target-bound",
    } <= profile.required_capabilities


def test_external_profile_rejects_source_only_claims() -> None:
    with pytest.raises(ValueError, match="external attach requires realtime"):
        RuntimeIntegrationProfile(
            engine_family=EngineFamily.UNITY,
            integration_mode=IntegrationMode.EXTERNAL_ATTACH,
            start_mode="attach",
            clock_mode=ClockMode.MANUAL_STEP,
            observation_mode=ObservationMode.RENDERED,
            action_mode=ActionMode.BOUNDED_INPUT,
            transport_mode=TransportMode.LOCAL_IPC,
        )
    with pytest.raises(ValueError, match="cannot claim engine-state observations"):
        RuntimeIntegrationProfile(
            engine_family=EngineFamily.UNREAL,
            integration_mode=IntegrationMode.EXTERNAL_ATTACH,
            start_mode="attach",
            clock_mode=ClockMode.REALTIME,
            observation_mode=ObservationMode.ENGINE_STATE,
            action_mode=ActionMode.BOUNDED_INPUT,
            transport_mode=TransportMode.LOCAL_IPC,
        )
    with pytest.raises(ValueError, match="cannot claim native actions"):
        RuntimeIntegrationProfile(
            engine_family=EngineFamily.UNREAL,
            integration_mode=IntegrationMode.EXTERNAL_ATTACH,
            start_mode="attach",
            clock_mode=ClockMode.REALTIME,
            observation_mode=ObservationMode.RENDERED,
            action_mode=ActionMode.NATIVE,
            transport_mode=TransportMode.LOCAL_IPC,
        )


def test_profile_connects_only_when_the_bridge_proves_its_capabilities() -> None:
    profile = RuntimeIntegrationProfile.for_source(EngineFamily.UNITY)
    driver = _ProfileDriver(profile.required_capabilities)

    environment = profile.connect(driver)
    initial = environment.reset(seed=7)
    terminal = environment.step(_action())
    environment.close()

    assert initial.step_id == 0
    assert terminal.done
    assert environment.spec.metadata == {}
    assert driver.close_count == 1


def test_profile_connection_fails_closed_on_a_missing_runtime_capability() -> None:
    profile = RuntimeIntegrationProfile.for_external(EngineFamily.UNREAL)
    driver = _ProfileDriver(profile.required_capabilities - {"target-bound"})

    with pytest.raises(ContractViolation, match="target-bound"):
        profile.connect(driver)

    assert driver.close_count == 1


def test_profile_validation_uses_the_shared_contract_error() -> None:
    profile = RuntimeIntegrationProfile.for_source(EngineFamily.UNREAL)
    spec = _ProfileDriver(profile.required_capabilities - {"manual-step"}).describe()

    with pytest.raises(ContractViolation, match="manual-step"):
        profile.validate_environment_spec(spec)


def test_runtime_integration_profile_round_trips_strict_json(tmp_path: Any) -> None:
    expected = RuntimeIntegrationProfile.for_external(
        EngineFamily.UNITY,
        observation_mode=ObservationMode.OFFICIAL_API,
        action_mode=ActionMode.OFFICIAL_API,
        transport_mode=TransportMode.OFFICIAL_API,
    )
    path = tmp_path / "runtime-integration.json"
    path.write_text(json.dumps(expected.to_mapping()), encoding="utf-8")

    assert load_runtime_integration(path) == expected

    invalid = expected.to_mapping() | {"endpoint": "private-local-value"}
    with pytest.raises(ValueError, match="unexpected fields"):
        RuntimeIntegrationProfile.from_mapping(invalid)
