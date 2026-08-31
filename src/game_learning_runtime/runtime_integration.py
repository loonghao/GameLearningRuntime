"""Truthful engine integration profiles for source, loader, and external runtimes."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Literal, TypeVar

from game_learning_runtime.bridge import BridgeDriver, BridgeEnvironment
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.specs import EnvironmentSpec

RUNTIME_INTEGRATION_SCHEMA_VERSION = "glr.runtime-integration.v2"
LEGACY_RUNTIME_INTEGRATION_SCHEMA_VERSION = "glr.runtime-integration.v1"


class EngineFamily(str, Enum):
    """Engine family used only to select an integration template."""

    UNITY = "unity"
    UNREAL = "unreal"
    OTHER = "other"


class IntegrationMode(str, Enum):
    """Where the authorized GLR adapter executes relative to the game."""

    ENGINE_PLUGIN = "engine-plugin"
    LOADER_PLUGIN = "loader-plugin"
    EXTERNAL_ATTACH = "external-attach"


class LoaderFamily(str, Enum):
    """Authorized third-party loader hosting an in-process adapter."""

    BEPINEX = "bepinex"
    UE4SS = "ue4ss"


class ClockMode(str, Enum):
    """How learner decisions advance the game clock."""

    MANUAL_STEP = "manual-step"
    TIME_SCALED = "time-scaled"
    REALTIME = "realtime"


class ObservationMode(str, Enum):
    """Authority boundary for observations returned by the adapter."""

    ENGINE_STATE = "engine-state"
    OFFICIAL_API = "official-api"
    RENDERED = "rendered"


class ActionMode(str, Enum):
    """Authority boundary for actions accepted by the adapter."""

    NATIVE = "native"
    OFFICIAL_API = "official-api"
    BOUNDED_COMMAND = "bounded-command"
    BOUNDED_INPUT = "bounded-input"


class TransportMode(str, Enum):
    """Deployment shape without selecting a concrete transport library."""

    IN_PROCESS = "in-process"
    LOCAL_IPC = "local-ipc"
    OFFICIAL_API = "official-api"


_PROFILE_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "engine_family",
        "integration_mode",
        "start_mode",
        "clock_mode",
        "observation_mode",
        "action_mode",
        "transport_mode",
        "seedable",
    }
)
_PROFILE_FIELDS_V2 = _PROFILE_FIELDS_V1 | {"loader_family"}


_EnumValue = TypeVar("_EnumValue", bound=Enum)


def _enum_value(enum_type: type[_EnumValue], value: object, *, field_name: str) -> _EnumValue:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = sorted(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of {choices}") from error


@dataclass(frozen=True, slots=True)
class RuntimeIntegrationProfile:
    """Machine-checkable deployment profile for one authorized runtime bridge.

    The profile does not describe game semantics, endpoints, processes, or
    credentials. It derives the minimum bridge capabilities needed for a
    source-integrated engine plugin, authorized loader, or external attachment.
    """

    engine_family: EngineFamily
    integration_mode: IntegrationMode
    start_mode: Literal["reset", "attach"]
    clock_mode: ClockMode
    observation_mode: ObservationMode
    action_mode: ActionMode
    transport_mode: TransportMode
    loader_family: LoaderFamily | None = None
    seedable: bool = False
    schema_version: str = RUNTIME_INTEGRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        enum_fields = {
            "engine_family": (self.engine_family, EngineFamily),
            "integration_mode": (self.integration_mode, IntegrationMode),
            "clock_mode": (self.clock_mode, ClockMode),
            "observation_mode": (self.observation_mode, ObservationMode),
            "action_mode": (self.action_mode, ActionMode),
            "transport_mode": (self.transport_mode, TransportMode),
        }
        for field_name, (value, expected_type) in enum_fields.items():
            if not isinstance(value, expected_type):
                raise TypeError(f"{field_name} must be a {expected_type.__name__}")
        if self.loader_family is not None and not isinstance(self.loader_family, LoaderFamily):
            raise TypeError("loader_family must be a LoaderFamily or None")
        if self.schema_version not in {
            LEGACY_RUNTIME_INTEGRATION_SCHEMA_VERSION,
            RUNTIME_INTEGRATION_SCHEMA_VERSION,
        }:
            raise ValueError(
                f"unsupported runtime integration schema version: {self.schema_version!r}"
            )
        if self.schema_version == LEGACY_RUNTIME_INTEGRATION_SCHEMA_VERSION and (
            self.integration_mode is IntegrationMode.LOADER_PLUGIN or self.loader_family is not None
        ):
            raise ValueError("glr.runtime-integration.v1 cannot describe loader plugins")
        if self.start_mode not in {"reset", "attach"}:
            raise ValueError("start_mode must be 'reset' or 'attach'")
        if not isinstance(self.seedable, bool):
            raise TypeError("seedable must be a bool")
        if self.seedable and self.start_mode != "reset":
            raise ValueError("seedable profiles require reset start mode")

        if self.integration_mode is IntegrationMode.EXTERNAL_ATTACH:
            if self.loader_family is not None:
                raise ValueError("external attach cannot declare a loader family")
            if self.start_mode != "attach":
                raise ValueError("external attach requires start_mode='attach'")
            if self.clock_mode is not ClockMode.REALTIME:
                raise ValueError("external attach requires realtime clock mode")
            if self.observation_mode is ObservationMode.ENGINE_STATE:
                raise ValueError("external attach cannot claim engine-state observations")
            if self.action_mode is ActionMode.NATIVE:
                raise ValueError("external attach cannot claim native actions")
            if self.transport_mode is TransportMode.IN_PROCESS:
                raise ValueError("external attach cannot use in-process transport")
        elif self.integration_mode is IntegrationMode.LOADER_PLUGIN:
            if self.loader_family is None:
                raise ValueError("loader plugins require loader_family")
            if self.start_mode != "attach":
                raise ValueError("loader plugins require start_mode='attach'")
            if self.clock_mode is not ClockMode.REALTIME:
                raise ValueError("loader plugins require realtime clock mode")
            if self.observation_mode is not ObservationMode.ENGINE_STATE:
                raise ValueError("loader plugins require semantic engine-state observations")
            if self.action_mode is not ActionMode.BOUNDED_COMMAND:
                raise ValueError("loader plugins require bounded-command actions")
            if self.transport_mode is not TransportMode.LOCAL_IPC:
                raise ValueError("loader plugins require authenticated local-ipc transport")
            if (
                self.loader_family is LoaderFamily.BEPINEX
                and self.engine_family is not EngineFamily.UNITY
            ):
                raise ValueError("BepInEx loader profiles require Unity")
            if (
                self.loader_family is LoaderFamily.UE4SS
                and self.engine_family is not EngineFamily.UNREAL
            ):
                raise ValueError("UE4SS loader profiles require Unreal")
        elif self.transport_mode is TransportMode.OFFICIAL_API:
            raise ValueError("engine plugins cannot use official-api transport")
        elif self.loader_family is not None:
            raise ValueError("engine plugins cannot declare a loader family")

    @classmethod
    def for_source(
        cls,
        engine_family: EngineFamily,
        *,
        clock_mode: ClockMode = ClockMode.MANUAL_STEP,
        transport_mode: TransportMode = TransportMode.LOCAL_IPC,
        seedable: bool = True,
    ) -> RuntimeIntegrationProfile:
        """Create the recommended source-integrated Unity or Unreal profile."""

        return cls(
            engine_family=engine_family,
            integration_mode=IntegrationMode.ENGINE_PLUGIN,
            start_mode="reset",
            clock_mode=clock_mode,
            observation_mode=ObservationMode.ENGINE_STATE,
            action_mode=ActionMode.NATIVE,
            transport_mode=transport_mode,
            seedable=seedable,
        )

    @classmethod
    def for_external(
        cls,
        engine_family: EngineFamily,
        *,
        observation_mode: ObservationMode = ObservationMode.RENDERED,
        action_mode: ActionMode = ActionMode.BOUNDED_INPUT,
        transport_mode: TransportMode = TransportMode.LOCAL_IPC,
    ) -> RuntimeIntegrationProfile:
        """Create a binary-only profile that makes a truthful live attachment."""

        return cls(
            engine_family=engine_family,
            integration_mode=IntegrationMode.EXTERNAL_ATTACH,
            start_mode="attach",
            clock_mode=ClockMode.REALTIME,
            observation_mode=observation_mode,
            action_mode=action_mode,
            transport_mode=transport_mode,
            seedable=False,
        )

    @classmethod
    def for_loader(
        cls,
        engine_family: EngineFamily,
        *,
        loader_family: LoaderFamily,
    ) -> RuntimeIntegrationProfile:
        """Create an authorized in-process loader with truthful attach semantics."""

        return cls(
            engine_family=engine_family,
            integration_mode=IntegrationMode.LOADER_PLUGIN,
            loader_family=loader_family,
            start_mode="attach",
            clock_mode=ClockMode.REALTIME,
            observation_mode=ObservationMode.ENGINE_STATE,
            action_mode=ActionMode.BOUNDED_COMMAND,
            transport_mode=TransportMode.LOCAL_IPC,
            seedable=False,
        )

    @property
    def required_capabilities(self) -> frozenset[str]:
        """Derive the bridge claims that must be proven before connection."""

        capabilities = {"postcondition-verified", "step"}
        if self.start_mode == "reset":
            capabilities.add("reset")
        else:
            capabilities.add("live-attach")
        if self.seedable:
            capabilities.add("deterministic-reset")

        capabilities.add(
            {
                ClockMode.MANUAL_STEP: "manual-step",
                ClockMode.TIME_SCALED: "time-scale-control",
                ClockMode.REALTIME: "realtime",
            }[self.clock_mode]
        )
        capabilities.add(
            {
                ObservationMode.ENGINE_STATE: "semantic-observation",
                ObservationMode.OFFICIAL_API: "official-observation",
                ObservationMode.RENDERED: "rendered-observation",
            }[self.observation_mode]
        )
        capabilities.add(
            {
                ActionMode.NATIVE: "native-action",
                ActionMode.OFFICIAL_API: "official-action",
                ActionMode.BOUNDED_COMMAND: "bounded-command",
                ActionMode.BOUNDED_INPUT: "bounded-input",
            }[self.action_mode]
        )

        if self.integration_mode in {
            IntegrationMode.ENGINE_PLUGIN,
            IntegrationMode.LOADER_PLUGIN,
        }:
            capabilities.add("main-thread-dispatch")
        else:
            capabilities.add("target-bound")
        if self.integration_mode is IntegrationMode.LOADER_PLUGIN:
            capabilities.update({"loader-plugin", "target-bound"})
        if self.transport_mode in {TransportMode.LOCAL_IPC, TransportMode.OFFICIAL_API}:
            capabilities.update({"authenticated", "target-bound"})
        if self.action_mode is ActionMode.BOUNDED_INPUT:
            capabilities.add("input-lease")
        return frozenset(capabilities)

    def validate_environment_spec(self, spec: EnvironmentSpec) -> None:
        """Reject a runtime descriptor that cannot prove this profile."""

        missing = sorted(self.required_capabilities - spec.capabilities)
        if missing:
            raise ContractViolation(f"environment is missing integration capabilities: {missing}")

    def connect(
        self,
        driver: BridgeDriver,
        *,
        protocol_version: str = "1.0",
        metadata_allowlist: Iterable[str] = (),
    ) -> BridgeEnvironment:
        """Connect only after the bridge proves the derived capability set."""

        return BridgeEnvironment(
            driver,
            protocol_version=protocol_version,
            metadata_allowlist=metadata_allowlist,
            required_capabilities=self.required_capabilities,
        )

    def to_mapping(self) -> dict[str, object]:
        """Return the stable JSON representation without runtime-local data."""

        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "engine_family": self.engine_family.value,
            "integration_mode": self.integration_mode.value,
            "start_mode": self.start_mode,
            "clock_mode": self.clock_mode.value,
            "observation_mode": self.observation_mode.value,
            "action_mode": self.action_mode.value,
            "transport_mode": self.transport_mode.value,
            "seedable": self.seedable,
        }
        if self.schema_version == RUNTIME_INTEGRATION_SCHEMA_VERSION:
            value["loader_family"] = (
                None if self.loader_family is None else self.loader_family.value
            )
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RuntimeIntegrationProfile:
        """Parse a strict v1 or v2 runtime integration data contract."""

        schema_version = value.get("schema_version")
        if schema_version == LEGACY_RUNTIME_INTEGRATION_SCHEMA_VERSION:
            fields = _PROFILE_FIELDS_V1
        elif schema_version == RUNTIME_INTEGRATION_SCHEMA_VERSION:
            fields = _PROFILE_FIELDS_V2
        else:
            raise ValueError(f"unsupported runtime integration schema version: {schema_version!r}")
        unexpected = sorted(set(value) - fields)
        if unexpected:
            raise ValueError(f"runtime integration has unexpected fields: {unexpected}")
        missing = sorted(fields - set(value))
        if missing:
            raise ValueError(f"runtime integration is missing fields: {missing}")
        seedable = value["seedable"]
        if not isinstance(seedable, bool):
            raise TypeError("seedable must be a bool")
        if not isinstance(schema_version, str):
            raise TypeError("schema_version must be a string")
        start_mode = value["start_mode"]
        if start_mode not in {"reset", "attach"}:
            raise ValueError("start_mode must be 'reset' or 'attach'")
        loader_value = value.get("loader_family")
        if loader_value is None:
            loader_family = None
        else:
            loader_family = _enum_value(LoaderFamily, loader_value, field_name="loader_family")
        return cls(
            engine_family=_enum_value(
                EngineFamily, value["engine_family"], field_name="engine_family"
            ),
            integration_mode=_enum_value(
                IntegrationMode, value["integration_mode"], field_name="integration_mode"
            ),
            start_mode=start_mode,
            clock_mode=_enum_value(ClockMode, value["clock_mode"], field_name="clock_mode"),
            observation_mode=_enum_value(
                ObservationMode, value["observation_mode"], field_name="observation_mode"
            ),
            action_mode=_enum_value(ActionMode, value["action_mode"], field_name="action_mode"),
            transport_mode=_enum_value(
                TransportMode, value["transport_mode"], field_name="transport_mode"
            ),
            loader_family=loader_family,
            seedable=seedable,
            schema_version=schema_version,
        )


def load_runtime_integration(
    path: str | PathLike[str] | Path,
) -> RuntimeIntegrationProfile:
    """Load a strict runtime integration profile from JSON."""

    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("runtime integration document must contain a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("runtime integration fields must be strings")
    return RuntimeIntegrationProfile.from_mapping(value)


__all__ = [
    "LEGACY_RUNTIME_INTEGRATION_SCHEMA_VERSION",
    "RUNTIME_INTEGRATION_SCHEMA_VERSION",
    "ActionMode",
    "ClockMode",
    "EngineFamily",
    "IntegrationMode",
    "LoaderFamily",
    "ObservationMode",
    "RuntimeIntegrationProfile",
    "TransportMode",
    "load_runtime_integration",
]
