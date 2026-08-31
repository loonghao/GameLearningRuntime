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
from game_learning_runtime.runtime_integration import (
    RUNTIME_INTEGRATION_SCHEMA_VERSION,
    ActionMode,
    ClockMode,
    EngineFamily,
    IntegrationMode,
    ObservationMode,
    RuntimeIntegrationProfile,
    TransportMode,
    load_runtime_integration,
)
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
    "RUNTIME_INTEGRATION_SCHEMA_VERSION",
    "TRAINING_SCHEMA_VERSION",
    "ActionMode",
    "BridgeAttachRequest",
    "BridgeConfig",
    "BridgeDriver",
    "BridgeEnvironment",
    "BridgeResetRequest",
    "BridgeStepRequest",
    "ClockMode",
    "CompositeSpec",
    "ContractEnvironment",
    "ContractViolation",
    "EngineFamily",
    "EnvironmentBridgeDriver",
    "EnvironmentSpec",
    "Event",
    "GLRError",
    "GameEnvironment",
    "IntegrationMode",
    "JsonlTransitionWriter",
    "KnowledgeAuthority",
    "KnowledgeSourceSpec",
    "LifecycleConfig",
    "ObservationMode",
    "OptionalDependencyError",
    "Policy",
    "RewardComposer",
    "RewardConfig",
    "RewardResult",
    "RewardSignal",
    "RewardTermSpec",
    "RuntimeIntegrationProfile",
    "SpaceKind",
    "SyncCollector",
    "TensorSpec",
    "TimeStep",
    "TrainingConfig",
    "Transition",
    "TransportMode",
    "Unroll",
    "__version__",
    "load_runtime_integration",
    "load_training_config",
    "protocol_path",
    "read_jsonl_transitions",
    "transition_from_record",
    "transition_to_record",
]
