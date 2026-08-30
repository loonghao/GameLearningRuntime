"""Game Learning Runtime public API."""

from importlib.metadata import PackageNotFoundError, version

from game_learning_runtime.collector import Policy, SyncCollector
from game_learning_runtime.contracts import Event, TimeStep, Transition, Unroll
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.errors import ContractViolation, GLRError, OptionalDependencyError
from game_learning_runtime.protocol import protocol_path
from game_learning_runtime.serialization import (
    JsonlTransitionWriter,
    read_jsonl_transitions,
    transition_from_record,
    transition_to_record,
)
from game_learning_runtime.specs import CompositeSpec, EnvironmentSpec, SpaceKind, TensorSpec

try:
    __version__ = version("game-learning-runtime")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0+local"

__all__ = [
    "CompositeSpec",
    "ContractEnvironment",
    "ContractViolation",
    "EnvironmentSpec",
    "Event",
    "GLRError",
    "GameEnvironment",
    "JsonlTransitionWriter",
    "OptionalDependencyError",
    "Policy",
    "SpaceKind",
    "SyncCollector",
    "TensorSpec",
    "TimeStep",
    "Transition",
    "Unroll",
    "__version__",
    "protocol_path",
    "read_jsonl_transitions",
    "transition_from_record",
    "transition_to_record",
]
