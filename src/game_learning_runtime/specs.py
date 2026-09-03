"""Framework-neutral tensor and environment specifications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.realtime import RealtimeTimingContract
from game_learning_runtime.runtime_health import RuntimeIdentity


class SpaceKind(str, Enum):
    """Semantic meaning of a tensor leaf in an observation or action tree."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MULTI_DISCRETE = "multi_discrete"
    BINARY = "binary"


Shape = tuple[int | None, ...]
Limit: TypeAlias = int | float | NDArray[Any]


def _immutable_limit(value: Limit | None) -> Limit | None:
    if value is None or isinstance(value, (int, float)):
        return value
    array = np.array(value, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Describes and validates one tensor leaf.

    A ``None`` dimension is dynamic. Composite trees provide hybrid,
    parameterized, and hierarchical spaces without adding algorithm concepts to
    the runtime contract.
    """

    shape: Shape
    dtype: np.dtype[Any] | str | type[Any]
    kind: SpaceKind = SpaceKind.CONTINUOUS
    minimum: Limit | None = None
    maximum: Limit | None = None
    description: str = ""

    def __post_init__(self) -> None:
        normalized_shape = tuple(self.shape)
        if any(dimension is not None and dimension < 0 for dimension in normalized_shape):
            raise ValueError("tensor dimensions must be non-negative or None")
        normalized_dtype = np.dtype(self.dtype)
        if self.kind in {SpaceKind.DISCRETE, SpaceKind.MULTI_DISCRETE} and not np.issubdtype(
            normalized_dtype, np.integer
        ):
            raise ValueError(f"{self.kind.value} tensors require an integer dtype")
        if self.kind is SpaceKind.BINARY and normalized_dtype != np.dtype(np.bool_):
            raise ValueError("binary tensors require the bool dtype")
        if (
            self.minimum is not None
            and self.maximum is not None
            and np.any(np.asarray(self.minimum) > np.asarray(self.maximum))
        ):
            raise ValueError("minimum cannot exceed maximum")
        object.__setattr__(self, "shape", normalized_shape)
        object.__setattr__(self, "dtype", normalized_dtype)
        object.__setattr__(self, "minimum", _immutable_limit(self.minimum))
        object.__setattr__(self, "maximum", _immutable_limit(self.maximum))

    @property
    def is_dynamic(self) -> bool:
        return any(dimension is None for dimension in self.shape)

    def validate(self, value: Any, *, path: str = "tensor") -> NDArray[Any]:
        array = np.asarray(value)
        if array.dtype != self.dtype:
            raise ContractViolation(f"{path} has dtype {array.dtype}; expected {self.dtype}")
        if array.ndim != len(self.shape):
            raise ContractViolation(f"{path} has shape {array.shape}; expected {self.shape}")
        for actual, expected in zip(array.shape, self.shape, strict=True):
            if expected is not None and actual != expected:
                raise ContractViolation(f"{path} has shape {array.shape}; expected {self.shape}")
        if self.minimum is not None and np.any(array < self.minimum):
            raise ContractViolation(f"{path} contains a value below {self.minimum}")
        if self.maximum is not None and np.any(array > self.maximum):
            raise ContractViolation(f"{path} contains a value above {self.maximum}")
        return array


SpecNode: TypeAlias = "TensorSpec | CompositeSpec"


@dataclass(frozen=True, slots=True)
class CompositeSpec:
    """A recursively nested tensor tree specification."""

    fields: Mapping[str, SpecNode]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a composite spec requires at least one field")
        copied: dict[str, SpecNode] = {}
        for name, spec in self.fields.items():
            if not name or "." in name:
                raise ValueError("field names must be non-empty and cannot contain dots")
            if not isinstance(spec, (TensorSpec, CompositeSpec)):
                raise TypeError(f"field {name!r} is not a TensorSpec or CompositeSpec")
            copied[name] = spec
        object.__setattr__(self, "fields", MappingProxyType(copied))

    @property
    def is_dynamic(self) -> bool:
        return any(spec.is_dynamic for spec in self.fields.values())

    def validate(self, value: Mapping[str, Any], *, path: str = "tree") -> None:
        if not isinstance(value, Mapping):
            raise ContractViolation(f"{path} must be a mapping")
        actual_keys = set(value)
        expected_keys = set(self.fields)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise ContractViolation(
                f"{path} keys differ from the contract; missing={missing}, unexpected={unexpected}"
            )
        for name, spec in self.fields.items():
            child_path = f"{path}.{name}"
            child = value[name]
            if isinstance(spec, CompositeSpec):
                spec.validate(child, path=child_path)
            else:
                spec.validate(child, path=child_path)

    def flatten(self, *, prefix: str = "") -> dict[str, TensorSpec]:
        result: dict[str, TensorSpec] = {}
        for name, spec in self.fields.items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(spec, CompositeSpec):
                result.update(spec.flatten(prefix=path))
            else:
                result[path] = spec
        return result


def mask_valid_counts(
    spec: CompositeSpec,
    mask: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Return the number of legal actions for every action-mask leaf.

    ``None`` means that no mask was supplied and therefore returns an empty
    mapping. A supplied mask is validated against the recursive composite
    contract before counting flattened leaves, avoiding the common mistake of
    counting action-head keys instead of boolean entries.
    """

    if mask is None:
        return {}
    spec.validate(mask, path="action_mask")
    counts: dict[str, int] = {}
    for path, leaf_spec in spec.flatten().items():
        if leaf_spec.kind is not SpaceKind.BINARY:
            raise ValueError(f"{path} is not a binary action-mask leaf")
        value: Any = mask
        for name in path.split("."):
            value = value[name]
        counts[path] = int(np.count_nonzero(np.asarray(value).reshape(-1)))
    return counts


def _default_reward_spec() -> TensorSpec:
    return TensorSpec(shape=(1,), dtype=np.float32)


def _default_done_spec() -> TensorSpec:
    return TensorSpec(shape=(1,), dtype=np.bool_, kind=SpaceKind.BINARY)


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Complete machine-readable contract exposed by a game adapter."""

    environment_id: str
    observation: CompositeSpec
    action: CompositeSpec
    reward: TensorSpec = field(default_factory=_default_reward_spec)
    done: TensorSpec = field(default_factory=_default_done_spec)
    action_mask: CompositeSpec | None = None
    protocol_version: str = "1.0"
    capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, str] = field(default_factory=dict)
    realtime_timing: RealtimeTimingContract | None = None
    runtime_identity: RuntimeIdentity | None = None

    def __post_init__(self) -> None:
        if not self.environment_id or any(character.isspace() for character in self.environment_id):
            raise ValueError("environment_id must be non-empty and cannot contain whitespace")
        if self.reward.kind is not SpaceKind.CONTINUOUS:
            raise ValueError("reward spec must be continuous")
        if self.done.kind is not SpaceKind.BINARY:
            raise ValueError("done spec must be binary")
        if self.realtime_timing is not None and not isinstance(
            self.realtime_timing, RealtimeTimingContract
        ):
            raise TypeError("realtime_timing must be a RealtimeTimingContract or None")
        if self.runtime_identity is not None and not isinstance(
            self.runtime_identity, RuntimeIdentity
        ):
            raise TypeError("runtime_identity must be a RuntimeIdentity or None")
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
