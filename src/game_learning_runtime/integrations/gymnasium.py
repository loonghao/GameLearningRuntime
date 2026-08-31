"""Adapt Gymnasium environments to the learner-neutral GLR contract."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.errors import ContractViolation, OptionalDependencyError
from game_learning_runtime.specs import (
    CompositeSpec,
    EnvironmentSpec,
    SpaceKind,
    TensorSpec,
)

try:
    import gymnasium
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - exercised without the optional extra
    raise OptionalDependencyError(
        "Gymnasium support requires `uv add game-learning-runtime[gymnasium]`"
    ) from error

InfoTransform: TypeAlias = Callable[[Mapping[str, Any]], Mapping[str, Any]]
ActionMaskProvider: TypeAlias = Callable[[], Any]
SpecNode: TypeAlias = TensorSpec | CompositeSpec


class AttachProvider(Protocol):
    """Explicit live-attachment hook for a Gymnasium-backed runtime."""

    def __call__(
        self, *, options: Mapping[str, Any] | None = None
    ) -> tuple[Any, Mapping[str, Any]]: ...


def _unsupported(space: spaces.Space[Any]) -> ValueError:
    return ValueError(f"unsupported Gymnasium space: {type(space).__name__}")


def _space_to_spec(space: spaces.Space[Any]) -> SpecNode:
    if isinstance(space, spaces.Box):
        return TensorSpec(
            shape=tuple(int(dimension) for dimension in space.shape),
            dtype=np.dtype(space.dtype),
            kind=SpaceKind.CONTINUOUS,
            minimum=np.array(space.low, copy=True),
            maximum=np.array(space.high, copy=True),
        )
    if isinstance(space, spaces.Discrete):
        discrete_start = int(space.start)
        return TensorSpec(
            shape=(1,),
            dtype=np.dtype(space.dtype),
            kind=SpaceKind.DISCRETE,
            minimum=discrete_start,
            maximum=discrete_start + int(space.n) - 1,
        )
    if isinstance(space, spaces.MultiDiscrete):
        multi_start = np.array(space.start, copy=True)
        return TensorSpec(
            shape=tuple(int(dimension) for dimension in space.shape),
            dtype=np.dtype(space.dtype),
            kind=SpaceKind.MULTI_DISCRETE,
            minimum=multi_start,
            maximum=multi_start + np.array(space.nvec, copy=True) - 1,
        )
    if isinstance(space, spaces.MultiBinary):
        return TensorSpec(
            shape=tuple(int(dimension) for dimension in space.shape),
            dtype=np.bool_,
            kind=SpaceKind.BINARY,
        )
    if isinstance(space, spaces.Dict):
        fields: dict[str, SpecNode] = {}
        for key, child in space.spaces.items():
            if not isinstance(key, str):
                raise ValueError("Gymnasium Dict space keys must be strings")
            fields[key] = _space_to_spec(child)
        return CompositeSpec(fields)
    if isinstance(space, spaces.Tuple):
        return CompositeSpec(
            {f"item_{index}": _space_to_spec(child) for index, child in enumerate(space.spaces)}
        )
    raise _unsupported(space)


def _root_spec(space: spaces.Space[Any], *, leaf_name: str) -> CompositeSpec:
    converted = _space_to_spec(space)
    if isinstance(converted, CompositeSpec):
        return converted
    return CompositeSpec({leaf_name: converted})


def _ensure_contained(space: spaces.Space[Any], value: Any, *, path: str) -> None:
    if not space.contains(value):
        raise ContractViolation(f"{path} is outside the declared Gymnasium space")


def _encode_node(space: spaces.Space[Any], value: Any, *, path: str) -> Any:
    if isinstance(space, spaces.Box):
        array = np.asarray(value)
        _ensure_contained(space, array, path=path)
        return np.array(array, copy=True)
    if isinstance(space, spaces.Discrete):
        _ensure_contained(space, value, path=path)
        return np.array([int(value)], dtype=space.dtype)
    if isinstance(space, spaces.MultiDiscrete):
        array = np.asarray(value)
        _ensure_contained(space, array, path=path)
        return np.array(array, copy=True)
    if isinstance(space, spaces.MultiBinary):
        array = np.asarray(value)
        _ensure_contained(space, array, path=path)
        return np.array(array, dtype=np.bool_, copy=True)
    if isinstance(space, spaces.Dict):
        if not isinstance(value, Mapping):
            raise ContractViolation(f"{path} must be a mapping")
        _ensure_contained(space, value, path=path)
        return {
            key: _encode_node(child, value[key], path=f"{path}.{key}")
            for key, child in space.spaces.items()
        }
    if isinstance(space, spaces.Tuple):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ContractViolation(f"{path} must be a sequence")
        _ensure_contained(space, value, path=path)
        return {
            f"item_{index}": _encode_node(child, value[index], path=f"{path}.item_{index}")
            for index, child in enumerate(space.spaces)
        }
    raise _unsupported(space)


def _encode_root(space: spaces.Space[Any], value: Any, *, leaf_name: str) -> TensorTree:
    encoded = _encode_node(space, value, path=leaf_name)
    if isinstance(space, (spaces.Dict, spaces.Tuple)):
        return cast(TensorTree, encoded)
    return {leaf_name: encoded}


def _leaf_array(value: Any, *, path: str) -> NDArray[Any]:
    if isinstance(value, Mapping):
        raise ContractViolation(f"{path} must be a tensor leaf")
    return np.asarray(value)


def _decode_node(space: spaces.Space[Any], value: Any, *, path: str) -> Any:
    if isinstance(space, spaces.Box):
        array = _leaf_array(value, path=path)
        _ensure_contained(space, array, path=path)
        return np.array(array, copy=True)
    if isinstance(space, spaces.Discrete):
        array = _leaf_array(value, path=path)
        if array.shape != (1,):
            raise ContractViolation(f"{path} must have shape (1,)")
        decoded = int(array[0])
        _ensure_contained(space, decoded, path=path)
        return decoded
    if isinstance(space, spaces.MultiDiscrete):
        array = _leaf_array(value, path=path)
        _ensure_contained(space, array, path=path)
        return np.array(array, copy=True)
    if isinstance(space, spaces.MultiBinary):
        array = _leaf_array(value, path=path)
        binary_decoded = np.array(array, dtype=space.dtype, copy=True)
        _ensure_contained(space, binary_decoded, path=path)
        return binary_decoded
    if isinstance(space, spaces.Dict):
        if not isinstance(value, Mapping):
            raise ContractViolation(f"{path} must be a mapping")
        return {
            key: _decode_node(child, value[key], path=f"{path}.{key}")
            for key, child in space.spaces.items()
        }
    if isinstance(space, spaces.Tuple):
        if not isinstance(value, Mapping):
            raise ContractViolation(f"{path} must be a mapping")
        return tuple(
            _decode_node(child, value[f"item_{index}"], path=f"{path}.item_{index}")
            for index, child in enumerate(space.spaces)
        )
    raise _unsupported(space)


def _decode_root(space: spaces.Space[Any], value: TensorTree, *, leaf_name: str) -> Any:
    if isinstance(space, (spaces.Dict, spaces.Tuple)):
        return _decode_node(space, value, path=leaf_name)
    if leaf_name not in value:
        raise ContractViolation(f"action is missing {leaf_name!r}")
    return _decode_node(space, value[leaf_name], path=leaf_name)


def _action_mask_spec(space: spaces.Space[Any], *, leaf_name: str) -> CompositeSpec:
    if not isinstance(space, spaces.Discrete):
        raise ValueError("action masks currently require a root Discrete action space")
    return CompositeSpec(
        {
            leaf_name: TensorSpec(
                (int(space.n),),
                np.bool_,
                kind=SpaceKind.BINARY,
                description="True marks a currently legal discrete action",
            )
        }
    )


class GymnasiumEnvironment(GameEnvironment):
    """Wrap one Gymnasium environment without exporting incidental metadata.

    ``info`` is discarded by default. Callers must provide an explicit
    ``info_transform`` to export stable, non-sensitive metadata.
    """

    def __init__(
        self,
        environment: gymnasium.Env[Any, Any],
        *,
        environment_id: str,
        action_mask_provider: ActionMaskProvider | None = None,
        attach_provider: AttachProvider | None = None,
        info_transform: InfoTransform | None = None,
        observation_key: str = "observation",
        action_key: str = "action",
        metadata: Mapping[str, str] | None = None,
        verified_capabilities: Iterable[str] = (),
    ) -> None:
        self._environment = environment
        self._observation_key = observation_key
        self._action_key = action_key
        self._action_mask_provider = action_mask_provider
        self._attach_provider = attach_provider
        self._info_transform = info_transform
        self._episode_id: UUID | None = None
        self._step_id = 0
        action_mask = (
            _action_mask_spec(environment.action_space, leaf_name=action_key)
            if action_mask_provider is not None
            else None
        )
        integration_capabilities = frozenset(verified_capabilities)
        if any(
            not isinstance(capability, str)
            or not capability
            or any(character.isspace() for character in capability)
            for capability in integration_capabilities
        ):
            raise ValueError("verified capabilities must be non-empty strings without whitespace")
        capabilities = {
            "gymnasium-adapter",
            "metadata-deny-by-default",
            *integration_capabilities,
        }
        if action_mask is not None:
            capabilities.add("action-mask")
        if attach_provider is not None:
            capabilities.add("live-attach")
        self._spec = EnvironmentSpec(
            environment_id=environment_id,
            observation=_root_spec(environment.observation_space, leaf_name=observation_key),
            action=_root_spec(environment.action_space, leaf_name=action_key),
            action_mask=action_mask,
            capabilities=frozenset(capabilities),
            metadata=MappingProxyType(dict(metadata or {})),
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        observation, info = self._environment.reset(
            seed=seed, options=None if options is None else dict(options)
        )
        return self._start_timestep(observation, info)

    def attach(self, *, options: Mapping[str, Any] | None = None) -> TimeStep:
        if self._attach_provider is None:
            raise ContractViolation("Gymnasium adapter does not support live attach")
        observation, info = self._attach_provider(options=options)
        return self._start_timestep(observation, info)

    def _start_timestep(self, observation: Any, info: Mapping[str, Any]) -> TimeStep:
        self._episode_id = uuid4()
        self._step_id = 0
        return self._timestep(
            observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )

    def step(self, action: TensorTree) -> TimeStep:
        if self._episode_id is None:
            raise ContractViolation("step requires reset first")
        gym_action = _decode_root(
            self._environment.action_space, action, leaf_name=self._action_key
        )
        observation, reward, terminated, truncated, info = self._environment.step(gym_action)
        reward_value = float(reward)
        if not math.isfinite(reward_value):
            raise ContractViolation("reward must be finite")
        self._step_id += 1
        return self._timestep(
            observation,
            reward=reward_value,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def close(self) -> None:
        self._environment.close()

    def _timestep(
        self,
        observation: Any,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> TimeStep:
        if self._episode_id is None:
            raise ContractViolation("environment has no active episode")
        exported_info = self._info_transform(info) if self._info_transform is not None else {}
        if not isinstance(exported_info, Mapping):
            raise ContractViolation("info_transform must return a mapping")
        return TimeStep(
            observation=_encode_root(
                self._environment.observation_space,
                observation,
                leaf_name=self._observation_key,
            ),
            reward=np.array([reward], dtype=np.float32),
            terminated=np.array([terminated], dtype=np.bool_),
            truncated=np.array([truncated], dtype=np.bool_),
            action_mask=self._action_mask(),
            episode_id=self._episode_id,
            step_id=self._step_id,
            info=dict(exported_info),
        )

    def _action_mask(self) -> TensorTree | None:
        if self._action_mask_provider is None:
            return None
        return {self._action_key: np.array(self._action_mask_provider(), copy=True)}


__all__ = [
    "ActionMaskProvider",
    "AttachProvider",
    "GymnasiumEnvironment",
    "InfoTransform",
]
