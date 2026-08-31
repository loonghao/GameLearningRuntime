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
    LoaderFamily,
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


@pytest.mark.parametrize(
    ("engine", "loader"),
    [
        (EngineFamily.UNITY, LoaderFamily.BEPINEX),
        (EngineFamily.UNREAL, LoaderFamily.UE4SS),
    ],
)
def test_loader_profile_is_in_process_but_truthful_live_attach(
    engine: EngineFamily, loader: LoaderFamily
) -> None:
    profile = RuntimeIntegrationProfile.for_loader(engine, loader_family=loader)

    assert profile.integration_mode is IntegrationMode.LOADER_PLUGIN
    assert profile.loader_family is loader
    assert profile.start_mode == "attach"
    assert profile.clock_mode is ClockMode.REALTIME
    assert profile.observation_mode is ObservationMode.ENGINE_STATE
    assert profile.action_mode is ActionMode.BOUNDED_COMMAND
    assert profile.transport_mode is TransportMode.LOCAL_IPC
    assert not profile.seedable
    assert {
        "authenticated",
        "bounded-command",
        "live-attach",
        "loader-plugin",
        "main-thread-dispatch",
        "postcondition-verified",
        "realtime",
        "semantic-observation",
        "step",
        "target-bound",
    } <= profile.required_capabilities


def test_loader_profile_rejects_incompatible_or_false_capabilities() -> None:
    with pytest.raises(ValueError, match="BepInEx loader profiles require Unity"):
        RuntimeIntegrationProfile.for_loader(
            EngineFamily.UNREAL, loader_family=LoaderFamily.BEPINEX
        )
    with pytest.raises(ValueError, match="UE4SS loader profiles require Unreal"):
        RuntimeIntegrationProfile.for_loader(EngineFamily.UNITY, loader_family=LoaderFamily.UE4SS)
    with pytest.raises(ValueError, match="loader plugins require start_mode='attach'"):
        RuntimeIntegrationProfile(
            engine_family=EngineFamily.UNITY,
            integration_mode=IntegrationMode.LOADER_PLUGIN,
            loader_family=LoaderFamily.BEPINEX,
            start_mode="reset",
            clock_mode=ClockMode.REALTIME,
            observation_mode=ObservationMode.ENGINE_STATE,
            action_mode=ActionMode.BOUNDED_COMMAND,
            transport_mode=TransportMode.LOCAL_IPC,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"loader_family": None}, "require loader_family"),
        ({"clock_mode": ClockMode.MANUAL_STEP}, "require realtime"),
        ({"observation_mode": ObservationMode.RENDERED}, "engine-state"),
        ({"action_mode": ActionMode.NATIVE}, "bounded-command"),
        ({"transport_mode": TransportMode.IN_PROCESS}, "local-ipc"),
    ],
)
def test_loader_profile_rejects_each_unsupported_boundary_claim(
    override: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "engine_family": EngineFamily.UNITY,
        "integration_mode": IntegrationMode.LOADER_PLUGIN,
        "loader_family": LoaderFamily.BEPINEX,
        "start_mode": "attach",
        "clock_mode": ClockMode.REALTIME,
        "observation_mode": ObservationMode.ENGINE_STATE,
        "action_mode": ActionMode.BOUNDED_COMMAND,
        "transport_mode": TransportMode.LOCAL_IPC,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        RuntimeIntegrationProfile(**values)  # type: ignore[arg-type]


def test_profiles_reject_loader_identity_outside_the_loader_lane() -> None:
    with pytest.raises(ValueError, match="external attach cannot declare"):
        RuntimeIntegrationProfile(
            engine_family=EngineFamily.UNITY,
            integration_mode=IntegrationMode.EXTERNAL_ATTACH,
            loader_family=LoaderFamily.BEPINEX,
            start_mode="attach",
            clock_mode=ClockMode.REALTIME,
            observation_mode=ObservationMode.RENDERED,
            action_mode=ActionMode.BOUNDED_INPUT,
            transport_mode=TransportMode.LOCAL_IPC,
        )
    with pytest.raises(ValueError, match="engine plugins cannot declare"):
        RuntimeIntegrationProfile(
            engine_family=EngineFamily.UNITY,
            integration_mode=IntegrationMode.ENGINE_PLUGIN,
            loader_family=LoaderFamily.BEPINEX,
            start_mode="reset",
            clock_mode=ClockMode.MANUAL_STEP,
            observation_mode=ObservationMode.ENGINE_STATE,
            action_mode=ActionMode.NATIVE,
            transport_mode=TransportMode.LOCAL_IPC,
        )


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


def test_loader_profile_round_trips_v2_and_v1_remains_loadable(tmp_path: Any) -> None:
    expected = RuntimeIntegrationProfile.for_loader(
        EngineFamily.UNREAL, loader_family=LoaderFamily.UE4SS
    )
    mapping = expected.to_mapping()

    assert mapping["schema_version"] == "glr.runtime-integration.v2"
    assert mapping["loader_family"] == "ue4ss"
    assert RuntimeIntegrationProfile.from_mapping(mapping) == expected

    legacy = {
        "schema_version": "glr.runtime-integration.v1",
        "engine_family": "unity",
        "integration_mode": "engine-plugin",
        "start_mode": "reset",
        "clock_mode": "manual-step",
        "observation_mode": "engine-state",
        "action_mode": "native",
        "transport_mode": "local-ipc",
        "seedable": True,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = load_runtime_integration(path)
    assert loaded.schema_version == "glr.runtime-integration.v1"
    assert loaded.loader_family is None

    with pytest.raises(ValueError, match="cannot describe loader plugins"):
        RuntimeIntegrationProfile.from_mapping(legacy | {"integration_mode": "loader-plugin"})
