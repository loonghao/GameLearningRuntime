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
from game_learning_runtime.errors import (
    ContractViolation,
    GLRError,
    HostProtocolError,
    HostRemoteError,
    OptionalDependencyError,
)
from game_learning_runtime.host import (
    HostBridgeDriver,
    HostProcessConfig,
    JsonLineHostChannel,
)
from game_learning_runtime.model_bundle import (
    MODEL_BUNDLE_SCHEMA_VERSION,
    BundleFile,
    ModelBundleManifest,
    build_model_bundle,
    load_model_bundle_manifest,
    verify_model_bundle,
)
from game_learning_runtime.protocol import protocol_path
from game_learning_runtime.runtime_integration import (
    RUNTIME_INTEGRATION_SCHEMA_VERSION,
    ActionMode,
    ClockMode,
    EngineFamily,
    IntegrationMode,
    LoaderFamily,
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
from game_learning_runtime.training_safety import (
    DEMONSTRATION_POLICY_SCHEMA_VERSION,
    REWARD_SAFETY_SCHEMA_VERSION,
    DemonstrationDecision,
    DemonstrationGate,
    DemonstrationOrigin,
    DemonstrationOutcome,
    DemonstrationPolicyConfig,
    DemonstrationProvenance,
    EpisodeRewardGuard,
    GuardedRewardResult,
    RewardSafetyConfig,
    load_demonstration_policy_config,
    load_reward_safety_config,
)

try:
    __version__ = version("game-learning-runtime")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0.0.0+local"

__all__ = [
    "DEMONSTRATION_POLICY_SCHEMA_VERSION",
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "REWARD_SAFETY_SCHEMA_VERSION",
    "RUNTIME_INTEGRATION_SCHEMA_VERSION",
    "TRAINING_SCHEMA_VERSION",
    "ActionMode",
    "BridgeAttachRequest",
    "BridgeConfig",
    "BridgeDriver",
    "BridgeEnvironment",
    "BridgeResetRequest",
    "BridgeStepRequest",
    "BundleFile",
    "ClockMode",
    "CompositeSpec",
    "ContractEnvironment",
    "ContractViolation",
    "DemonstrationDecision",
    "DemonstrationGate",
    "DemonstrationOrigin",
    "DemonstrationOutcome",
    "DemonstrationPolicyConfig",
    "DemonstrationProvenance",
    "EngineFamily",
    "EnvironmentBridgeDriver",
    "EnvironmentSpec",
    "EpisodeRewardGuard",
    "Event",
    "GLRError",
    "GameEnvironment",
    "GuardedRewardResult",
    "HostBridgeDriver",
    "HostProcessConfig",
    "HostProtocolError",
    "HostRemoteError",
    "IntegrationMode",
    "JsonLineHostChannel",
    "JsonlTransitionWriter",
    "KnowledgeAuthority",
    "KnowledgeSourceSpec",
    "LifecycleConfig",
    "LoaderFamily",
    "ModelBundleManifest",
    "ObservationMode",
    "OptionalDependencyError",
    "Policy",
    "RewardComposer",
    "RewardConfig",
    "RewardResult",
    "RewardSafetyConfig",
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
    "build_model_bundle",
    "load_demonstration_policy_config",
    "load_model_bundle_manifest",
    "load_reward_safety_config",
    "load_runtime_integration",
    "load_training_config",
    "protocol_path",
    "read_jsonl_transitions",
    "transition_from_record",
    "transition_to_record",
    "verify_model_bundle",
]
