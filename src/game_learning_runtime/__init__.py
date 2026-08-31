"""Game Learning Runtime public API."""

from importlib.metadata import PackageNotFoundError, version

from game_learning_runtime.bridge import (
    BridgeAttachRequest,
    BridgeDriver,
    BridgeEnvironment,
    BridgeResetRequest,
    BridgeStepRequest,
    EnvironmentBridgeDriver,
)
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
from game_learning_runtime.training import (
    TRAINING_SCHEMA_VERSION,
    BridgeConfig,
    KnowledgeAuthority,
    KnowledgeSourceSpec,
    LifecycleConfig,
    RewardComposer,
    RewardConfig,
    RewardResult,
    RewardSignal,
    RewardTermSpec,
    TrainingConfig,
    load_training_config,
)

try:
    __version__ = version("game-learning-runtime")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0+local"

__all__ = [
    "TRAINING_SCHEMA_VERSION",
    "BridgeAttachRequest",
    "BridgeConfig",
    "BridgeDriver",
    "BridgeEnvironment",
    "BridgeResetRequest",
    "BridgeStepRequest",
    "CompositeSpec",
    "ContractEnvironment",
    "ContractViolation",
    "EnvironmentBridgeDriver",
    "EnvironmentSpec",
    "Event",
    "GLRError",
    "GameEnvironment",
    "JsonlTransitionWriter",
    "KnowledgeAuthority",
    "KnowledgeSourceSpec",
    "LifecycleConfig",
    "OptionalDependencyError",
    "Policy",
    "RewardComposer",
    "RewardConfig",
    "RewardResult",
    "RewardSignal",
    "RewardTermSpec",
    "SpaceKind",
    "SyncCollector",
    "TensorSpec",
    "TimeStep",
    "TrainingConfig",
    "Transition",
    "Unroll",
    "__version__",
    "load_training_config",
    "protocol_path",
    "read_jsonl_transitions",
    "transition_from_record",
    "transition_to_record",
]
