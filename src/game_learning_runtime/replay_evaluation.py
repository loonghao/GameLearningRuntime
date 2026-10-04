"""Bounded, read-only adapter replay and externally frozen contract evaluation.

The policy is a trusted caller-supplied evaluation seam, not a sandbox. This
module never imports or executes a candidate artifact. It hashes inert artifact
bytes and callback implementation/captures around evaluation and only replays
captured actions and observations. The strict reward modules are fingerprinted;
callback globals and other imported dependencies remain part of the trusted
owner's environment. This is not code attestation.
Passing means offline contract conformance, never live task success or learning.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field, fields, is_dataclass, replace
from enum import Enum
from itertools import pairwise
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType
from typing import Any, TypeAlias, cast
from uuid import UUID, uuid4

import numpy as np

from game_learning_runtime import correlated_rewards, training, training_safety
from game_learning_runtime.contracts import (
    ActionOutcome,
    ActionReceipt,
    TensorTree,
    TimeStep,
    Transition,
)
from game_learning_runtime.correlated_rewards import (
    OBSERVATION_CONTEXT_KEY,
    REWARD_EVIDENCE_KEY,
    CorrelatedRewardGuard,
    CorrelationPolicy,
)
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.knowledge_evidence import (
    CapabilityGap,
    ConsumptionState,
    DecisionConsumptionReceipt,
    RuleIndexBinding,
)
from game_learning_runtime.realtime import RealtimeActionReceipt
from game_learning_runtime.serialization import transition_from_record, transition_to_record
from game_learning_runtime.specs import EnvironmentSpec
from game_learning_runtime.training import TrainingConfig
from game_learning_runtime.training_safety import RewardSafetyConfig

REPLAY_SUITE_SCHEMA = "glr.fixed-replay-suite.v1"
REPLAY_RESULT_SCHEMA = "glr.replay-evaluation.v1"
REPLAY_CHECK_METRICS = (
    "evaluation.parameter_mutations",
    "evaluation.reset_identity_mismatches",
    "evaluation.stale_observation_updates",
    "evaluation.dead_or_loading_updates",
    "evaluation.illegal_action_bootstraps",
    "evaluation.reward_attribution_errors",
    "evaluation.action_interval_errors",
)
_ID = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_HASH = re.compile(r"[0-9a-f]{64}")
_MAX_RECORD_BYTES = 1 << 20
_MAX_SUITE_BYTES = 16 << 20
_MAX_ARTIFACT_BYTES = 64 << 20
_MAX_STEPS = 4096
_REWARD_REL_TOLERANCE = 1e-6
_REWARD_ABS_TOLERANCE = 1e-6
_UNRESOLVED_ACTION_OUTCOMES = frozenset(
    {ActionOutcome.UNKNOWN, ActionOutcome.PARTIAL, ActionOutcome.INDETERMINATE}
)


def _id(value: object) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("expected a bounded portable identifier")


def _digest(value: object) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("expected a lowercase SHA-256 digest")


def _sequence(value: object, *, maximum: int = _MAX_STEPS) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError("expected a bounded nonnegative integer")


def _tuple(value: object, item_type: type[Any], *, maximum: int = 256) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) > maximum
        or any(not isinstance(item, item_type) for item in value)
    ):
        raise TypeError("expected a bounded typed tuple")


def _json(value: object, *, maximum: int = _MAX_RECORD_BYTES) -> str:
    _reject_privileged(value)
    result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(result.encode("utf-8")) > maximum:
        raise ValueError("replay evidence exceeds its byte budget")
    return result


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_privileged(value: object, *, recursive: bool = True) -> None:
    """Reject typed host authority before any container loses its type identity.

    This is not a scanner for arbitrary secrets or already flattened mappings.
    Traverse dataclass fields directly, never via ``asdict`` or ``to_mapping``.
    """
    # Lazy import keeps the replay and campaign contracts independently importable.
    from game_learning_runtime.continuous_learning import HostAuthority, HostRoleCapability

    if isinstance(value, (HostAuthority, HostRoleCapability)):
        raise ContractViolation("privileged host material cannot be serialized as replay evidence")
    if not recursive:
        return
    visited: set[int] = set()

    def visit(item: Any) -> None:
        if isinstance(item, (HostAuthority, HostRoleCapability)):
            raise ContractViolation(
                "privileged host material cannot be serialized as replay evidence"
            )
        if not (
            isinstance(item, (Mapping, tuple, list, set, frozenset, np.ndarray, np.void, Enum))
            or (is_dataclass(item) and not isinstance(item, type))
        ):
            return
        identity = id(item)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (tuple, list, set, frozenset)):
            for child in item:
                visit(child)
        elif isinstance(item, (np.ndarray, np.void)):
            if item.dtype.hasobject:
                if item.dtype.names:
                    for name in item.dtype.names:
                        visit(item[name])
                elif isinstance(item, np.ndarray):
                    for child in item.flat:
                        visit(child)
        elif isinstance(item, Enum):
            visit(item.value)
        else:
            for definition in fields(item):
                visit(getattr(item, definition.name))

    visit(value)


def _primitive(value: Any, *, _checked: bool = False) -> Any:
    _reject_privileged(value, recursive=not _checked)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": value.shape, "hex": value.tobytes().hex()}
    if isinstance(value, np.dtype):
        return value.str
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _primitive(item, _checked=True) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_primitive(item, _checked=True) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_primitive(item, _checked=True) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _primitive(getattr(value, item.name), _checked=True)
            for item in fields(value)
        }
    return value


def _tree(tree: TensorTree, *, depth: int = 0, budget: list[int] | None = None) -> TensorTree:
    if depth == 0:
        _reject_privileged(tree)
    if not isinstance(tree, Mapping) or depth > 8 or len(tree) > 256:
        raise ValueError("tensor tree exceeds replay bounds")
    result: dict[str, Any] = {}
    if budget is None:
        budget = [0, 0]
    for key, value in tree.items():
        _id(key)
        if isinstance(value, Mapping):
            result[key] = _tree(value, depth=depth + 1, budget=budget)
        else:
            array = np.asarray(value)
            if array.ndim == 0 or array.dtype.kind not in "biuf":
                raise ValueError("replay requires non-scalar numeric tensors")
            budget[0] += array.nbytes
            budget[1] += 1
            if budget[0] > _MAX_RECORD_BYTES or budget[1] > 256 or not np.all(np.isfinite(array)):
                raise ValueError("tensor is nonfinite or exceeds replay bounds")
            # Bytes-backed leaves cannot be made writable with setflags.
            result[key] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    return MappingProxyType(result)


def _step_record(step: TimeStep) -> str:
    _reject_privileged(step)
    if not isinstance(step, TimeStep) or not isinstance(step.episode_id, UUID):
        raise TypeError("replay requires a TimeStep with UUID identity")
    _sequence(step.step_id)
    for array in (step.reward, step.terminated, step.truncated):
        _tree({"value": array})
    _tree(step.observation)
    if step.action_mask is not None:
        _tree(step.action_mask)
    receipt = step.action_receipt
    if receipt is not None:
        # The core serializer is shared with legacy providers. Project only its
        # declared contract fields here, retaining explicit None/unknown values.
        values = {item.name: getattr(receipt, item.name) for item in fields(ActionReceipt)}
        if receipt.realtime is not None:
            values["realtime"] = RealtimeActionReceipt(
                **{
                    item.name: getattr(receipt.realtime, item.name)
                    for item in fields(RealtimeActionReceipt)
                }
            )
        receipt = ActionReceipt(**values)
    record = transition_to_record(
        Transition(
            episode_id=step.episode_id,
            step_id=step.step_id,
            observation=step.observation,
            action={},
            reward=step.reward,
            next_observation=step.observation,
            terminated=step.terminated,
            truncated=step.truncated,
            action_mask=step.action_mask,
            next_action_mask=step.action_mask,
            action_receipt=receipt,
            events=step.events,
            info=step.info,
            timestamp_ns=step.timestamp_ns,
        )
    )
    return _json(record)


def _decode_step(record: str) -> TimeStep:
    transition = transition_from_record(json.loads(record))
    return TimeStep(
        observation=transition.observation,
        reward=transition.reward,
        terminated=transition.terminated,
        truncated=transition.truncated,
        episode_id=transition.episode_id,
        step_id=transition.step_id,
        action_mask=transition.action_mask,
        action_receipt=transition.action_receipt,
        events=transition.events,
        info=transition.info,
        timestamp_ns=transition.timestamp_ns,
    )


class ObservationLifecycle(str, Enum):
    ACTIVE = "active"
    DEAD = "dead"
    LOADING = "loading"
    UNKNOWN = "unknown"


class CheckCoverage(str, Enum):
    AUDITED = "audited"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not-applicable"


@dataclass(frozen=True, slots=True)
class MaskIndex:
    """Adapter-exported coordinates connecting a semantic action to each mask head."""

    path: str
    index: tuple[int, ...]

    def __post_init__(self) -> None:
        _id(self.path)
        if not isinstance(self.index, tuple) or not 1 <= len(self.index) <= 8:
            raise ValueError("mask index must be a bounded coordinate tuple")
        for item in self.index:
            _sequence(item, maximum=_MAX_RECORD_BYTES)


@dataclass(frozen=True, slots=True)
class ReplayAction:
    semantic: str
    action: TensorTree
    mask_indices: tuple[MaskIndex, ...] = ()

    def __post_init__(self) -> None:
        _id(self.semantic)
        object.__setattr__(self, "action", _tree(self.action))
        _tuple(self.mask_indices, MaskIndex)
        if len({item.path for item in self.mask_indices}) != len(self.mask_indices):
            raise ValueError("mask paths must be unique")

    def to_mapping(self) -> dict[str, Any]:
        _reject_privileged(self)
        return {
            "semantic": self.semantic,
            "action": _primitive(self.action),
            "mask_indices": _primitive(self.mask_indices),
        }


@dataclass(frozen=True, slots=True)
class RewardContribution:
    """Adapter-exported attribution; runtime does not infer game-specific rewards."""

    term_id: str
    amount: float
    action_id: str | None = None
    requires_accepted: bool = False

    def __post_init__(self) -> None:
        _id(self.term_id)
        if isinstance(self.amount, bool) or not isinstance(self.amount, (int, float)):
            raise TypeError("reward contribution must be numeric")
        if not math.isfinite(self.amount):
            raise ValueError("reward contribution must be finite")
        if self.action_id is not None:
            _id(self.action_id)
        if not isinstance(self.requires_accepted, bool):
            raise TypeError("requires_accepted must be boolean")
        if self.requires_accepted and self.action_id is None:
            raise ValueError("accepted-action reward requires action identity")


@dataclass(frozen=True, slots=True)
class ActionSegment:
    segment_id: str
    start_sequence: int
    end_sequence: int

    def __post_init__(self) -> None:
        _id(self.segment_id)
        _sequence(self.start_sequence, maximum=2**63 - 1)
        _sequence(self.end_sequence, maximum=2**63 - 1)
        if self.end_sequence <= self.start_sequence:
            raise ValueError("action segment must advance its sequence")


@dataclass(frozen=True, slots=True)
class KnowledgeRequirement:
    finding_id: str
    finding_sha256: str
    consumer_id: str
    use_evidence_sha256: str

    def __post_init__(self) -> None:
        _id(self.finding_id)
        _id(self.consumer_id)
        _digest(self.finding_sha256)
        _digest(self.use_evidence_sha256)


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """One detached TimeStep snapshot plus explicit producer-side evidence."""

    timestep: InitVar[TimeStep]
    producer_sequence: int | None
    lifecycle: ObservationLifecycle
    legal_actions: tuple[ReplayAction, ...]
    decision_id: str
    reward_contributions: tuple[RewardContribution, ...] | None = None
    action_interval: tuple[ActionSegment, ...] | None = None
    required_findings: tuple[KnowledgeRequirement, ...] = ()
    _record: str = field(init=False, repr=False)

    def __post_init__(self, timestep: TimeStep) -> None:
        if self.producer_sequence is not None:
            _sequence(self.producer_sequence, maximum=2**63 - 1)
        object.__setattr__(self, "lifecycle", ObservationLifecycle(self.lifecycle))
        _id(self.decision_id)
        _tuple(self.legal_actions, ReplayAction)
        if len({action.semantic for action in self.legal_actions}) != len(self.legal_actions):
            raise ValueError("legal action semantics must be unique")
        if self.reward_contributions is not None:
            _tuple(self.reward_contributions, RewardContribution)
            if len({item.term_id for item in self.reward_contributions}) != len(
                self.reward_contributions
            ):
                raise ValueError("reward terms must be unique")
        if self.action_interval is not None:
            _tuple(self.action_interval, ActionSegment)
        _tuple(self.required_findings, KnowledgeRequirement)
        if len({(item.finding_id, item.consumer_id) for item in self.required_findings}) != len(
            self.required_findings
        ):
            raise ValueError("knowledge requirements must be unique")
        object.__setattr__(self, "_record", _step_record(timestep))

    def snapshot(self) -> TimeStep:
        """Return a detached copy; mutation cannot change frozen source evidence."""
        return _decode_step(self._record)

    def to_mapping(self) -> dict[str, Any]:
        _reject_privileged(self)
        return {
            "timestep": json.loads(self._record),
            "producer_sequence": self.producer_sequence,
            "lifecycle": self.lifecycle.value,
            "decision_id": self.decision_id,
            "legal_actions": [ReplayAction.to_mapping(item) for item in self.legal_actions],
            "reward_contributions": _primitive(self.reward_contributions),
            "action_interval": _primitive(self.action_interval),
            "required_findings": _primitive(self.required_findings),
        }


@dataclass(frozen=True, slots=True)
class ReplayEpisode:
    source_id: str
    source_sha256: str
    frames: tuple[ReplayFrame, ...]
    recorded_actions: tuple[ReplayAction, ...]
    run_id: str | None = None

    def __post_init__(self) -> None:
        _id(self.source_id)
        if self.run_id is not None:
            _id(self.run_id)
        _digest(self.source_sha256)
        _tuple(self.frames, ReplayFrame, maximum=_MAX_STEPS + 1)
        _tuple(self.recorded_actions, ReplayAction, maximum=_MAX_STEPS)
        if not self.recorded_actions or len(self.frames) != len(self.recorded_actions) + 1:
            raise ValueError("replay requires a reset frame and one post-frame per recorded action")

    def to_mapping(self) -> dict[str, Any]:
        _reject_privileged(self)
        return {
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "run_id": self.run_id,
            "frames": [ReplayFrame.to_mapping(frame) for frame in self.frames],
            "recorded_actions": [
                ReplayAction.to_mapping(action) for action in self.recorded_actions
            ],
        }


@dataclass(frozen=True, slots=True)
class FixedReplaySuite:
    """External fixture contract frozen before candidate execution.

    Every suite explicitly names a target, including synthetic offline suites.
    Missing target identities never count as matching evidence.
    """

    suite_id: str
    spec: EnvironmentSpec
    binding: RuleIndexBinding
    episodes: tuple[ReplayEpisode, ...]
    not_applicable: Mapping[str, str] = field(default_factory=dict)
    reward_training: TrainingConfig | None = None
    reward_safety: RewardSafetyConfig | None = None
    _spec_record: str = field(init=False, repr=False)
    _record: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _id(self.suite_id)
        if not isinstance(self.spec, EnvironmentSpec) or not isinstance(
            self.binding, RuleIndexBinding
        ):
            raise TypeError("suite requires EnvironmentSpec and RuleIndexBinding")
        if (self.reward_training is None) != (self.reward_safety is None):
            raise ValueError("a fixed reward contract requires both training and safety")
        if self.reward_training is not None and (
            not isinstance(self.reward_training, TrainingConfig)
            or not isinstance(self.reward_safety, RewardSafetyConfig)
        ):
            raise TypeError("fixed rewards require typed training and safety contracts")
        target = self.spec.metadata.get("target_id")
        _id(target)
        if (
            not isinstance(self.spec.metadata, Mapping)
            or len(self.spec.metadata) > 256
            or any(
                not isinstance(key, str)
                or not 1 <= len(key) <= 128
                or not isinstance(value, str)
                or len(value) > 4096
                for key, value in self.spec.metadata.items()
            )
        ):
            raise ValueError("replay spec metadata requires a bounded string-to-string mapping")
        RuleIndexBinding.assert_current(
            self.binding,
            environment_id=self.spec.environment_id,
            protocol_version=self.spec.protocol_version,
            rules_version=self.binding.rules_version,
            rules_sha256=self.binding.rules_sha256,
        )
        _tuple(self.episodes, ReplayEpisode, maximum=64)
        if (
            not self.episodes
            or sum(len(item.recorded_actions) for item in self.episodes) > _MAX_STEPS
        ):
            raise ValueError("suite requires a bounded nonempty replay")
        if len({item.source_id for item in self.episodes}) != len(self.episodes):
            raise ValueError("suite source identifiers must be unique")
        decisions = [
            frame.decision_id for episode in self.episodes for frame in episode.frames[:-1]
        ]
        if len(set(decisions)) != len(decisions):
            raise ValueError("frozen decision identities must be unique across the suite")
        if not isinstance(self.not_applicable, Mapping):
            raise TypeError("not_applicable must be a mapping")
        for metric, reason in self.not_applicable.items():
            if metric not in REPLAY_CHECK_METRICS[2:]:
                raise ValueError("artifact and reset identity checks cannot be skipped")
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
                raise ValueError("not-applicable checks require a bounded explicit reason")
        object.__setattr__(self, "not_applicable", MappingProxyType(dict(self.not_applicable)))
        for episode in self.episodes:
            for frame in episode.frames:
                step = ReplayFrame.snapshot(frame)
                if step.action_receipt is not None and step.action_receipt.target_id != target:
                    raise ContractViolation(
                        "record action receipt differs from frozen target binding"
                    )
                self.spec.observation.validate(step.observation)
                self.spec.reward.validate(step.reward)
                self.spec.done.validate(step.terminated)
                self.spec.done.validate(step.truncated)
                if self.spec.action_mask is not None:
                    if step.action_mask is None:
                        raise ContractViolation("record omitted declared action mask")
                    self.spec.action_mask.validate(step.action_mask)
                elif step.action_mask is not None:
                    raise ContractViolation("record contains undeclared action mask")
                for action in frame.legal_actions:
                    self.spec.action.validate(action.action)
            for action in episode.recorded_actions:
                self.spec.action.validate(action.action)
        object.__setattr__(self, "_spec_record", _json(_primitive(self.spec)))
        record = {
            "schema": REPLAY_SUITE_SCHEMA,
            "suite_id": self.suite_id,
            "spec": json.loads(self._spec_record),
            "binding": RuleIndexBinding.to_mapping(self.binding),
            "episodes": [ReplayEpisode.to_mapping(episode) for episode in self.episodes],
            "not_applicable": dict(self.not_applicable),
            "reward_training": _primitive(self.reward_training),
            "reward_safety": _primitive(self.reward_safety),
        }
        object.__setattr__(self, "_record", _json(record, maximum=_MAX_SUITE_BYTES))

    @property
    def sha256(self) -> str:
        return _fixed_suite_sha256(self)

    def to_mapping(self) -> dict[str, Any]:
        _reject_privileged(self)
        return cast(dict[str, Any], json.loads(self._record))

    def verify_integrity(self) -> None:
        actual = {
            "schema": REPLAY_SUITE_SCHEMA,
            "suite_id": self.suite_id,
            "spec": _primitive(self.spec),
            "binding": RuleIndexBinding.to_mapping(self.binding),
            "episodes": [ReplayEpisode.to_mapping(episode) for episode in self.episodes],
            "not_applicable": dict(self.not_applicable),
            "reward_training": _primitive(self.reward_training),
            "reward_safety": _primitive(self.reward_safety),
        }
        if _json(actual, maximum=_MAX_SUITE_BYTES) != self._record:
            raise ContractViolation("replay evidence changed after suite freeze")


def _fixed_suite_sha256(suite: FixedReplaySuite) -> str:
    """Read the frozen base digest without invoking a subclass descriptor."""
    return _sha(suite._record)


def _same_action(left: ReplayAction, right: ReplayAction) -> bool:
    return _json(ReplayAction.to_mapping(left)) == _json(ReplayAction.to_mapping(right))


def _mask_allows(spec: EnvironmentSpec, frame: ReplayFrame, action: ReplayAction) -> bool:
    if spec.action_mask is None:
        return not action.mask_indices
    mask = ReplayFrame.snapshot(frame).action_mask
    if mask is None or set(spec.action_mask.flatten()) != {
        item.path for item in action.mask_indices
    }:
        raise ContractViolation("semantic action lacks complete authoritative mask coordinates")
    for coordinate in action.mask_indices:
        value: Any = mask
        for key in coordinate.path.split("."):
            value = value[key]
        array = np.asarray(value)
        if len(coordinate.index) != array.ndim or any(
            index >= size for index, size in zip(coordinate.index, array.shape, strict=True)
        ):
            raise ContractViolation("semantic action has out-of-range mask coordinates")
        if not bool(array[coordinate.index]):
            return False
    return True


class ReplayEnvironment(GameEnvironment):
    """Replay captured branches, stopping before any unresolved action result.

    Unknown, partial, and indeterminate outcomes require reconciliation. A
    complete rejected, blocked, or no-effect post-state remains replayable;
    reward terms requiring acceptance still require an accepted receipt.
    """

    def __init__(self, suite: FixedReplaySuite) -> None:
        if not isinstance(suite, FixedReplaySuite):
            raise TypeError("replay requires a FixedReplaySuite")
        FixedReplaySuite.verify_integrity(suite)
        self._suite = suite
        self._episode_index = -1
        self._position = 0
        self._logical_id: UUID | None = None
        self._closed = False

    @property
    def spec(self) -> EnvironmentSpec:
        return replace(self._suite.spec, capabilities=frozenset({"offline-replay"}))

    @property
    def current_frame(self) -> ReplayFrame:
        self._ensure_open()
        if self._logical_id is None:
            raise ContractViolation("replay requires reset first")
        return self._suite.episodes[self._episode_index].frames[self._position]

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        self._ensure_open()
        if seed is not None or options:
            raise ContractViolation("frozen replay does not accept reset overrides")
        self._episode_index = (self._episode_index + 1) % len(self._suite.episodes)
        self._position = 0
        self._logical_id = uuid4()
        source = ReplayFrame.snapshot(self.current_frame)
        if source.step_id != 0 or source.done:
            raise ContractViolation("recorded reset must be nonterminal source step zero")
        return self._logical_step()

    def step(self, action: TensorTree) -> TimeStep:
        frame = self.current_frame
        episode = self._suite.episodes[self._episode_index]
        if ReplayFrame.snapshot(frame).done or self._position >= len(episode.recorded_actions):
            raise ContractViolation("replay is terminal or exhausted; reset first")
        current_receipt = ReplayFrame.snapshot(frame).action_receipt
        if current_receipt is not None and current_receipt.outcome in _UNRESOLVED_ACTION_OUTCOMES:
            raise ContractViolation(
                "unresolved action outcome cannot be followed by another action"
            )
        self.spec.action.validate(action)
        recorded = episode.recorded_actions[self._position]
        requested = ReplayAction(recorded.semantic, action, recorded.mask_indices)
        if not _same_action(requested, recorded):
            raise ContractViolation("action has no captured replay branch")
        if not any(_same_action(recorded, item) for item in frame.legal_actions):
            raise ContractViolation("recorded action is not in the frozen legal surface")
        if not _mask_allows(self.spec, frame, recorded):
            raise ContractViolation("recorded action is masked illegal")
        following_receipt = ReplayFrame.snapshot(episode.frames[self._position + 1]).action_receipt
        if (
            following_receipt is not None
            and following_receipt.outcome in _UNRESOLVED_ACTION_OUTCOMES
        ):
            raise ContractViolation("unresolved outcome is not learner-facing replay data")
        self._position += 1
        return self._logical_step()

    def _logical_step(self) -> TimeStep:
        episode = self._suite.episodes[self._episode_index]
        source = ReplayFrame.snapshot(self.current_frame)
        receipt = source.action_receipt
        if receipt is not None:
            receipt = replace(
                receipt, episode_id=cast(UUID, self._logical_id), step_id=self._position
            )
        provenance = {
            "source_id": episode.source_id,
            "source_sha256": episode.source_sha256,
            "source_episode_id": str(source.episode_id),
            "source_step_id": source.step_id,
            "producer_sequence": self.current_frame.producer_sequence,
            "logical_step_id": self._position,
            "decision_id": self.current_frame.decision_id,
            "suite_sha256": _fixed_suite_sha256(self._suite),
        }
        return replace(
            source,
            episode_id=cast(UUID, self._logical_id),
            step_id=self._position,
            action_receipt=receipt,
            info={**source.info, "replay_source": provenance},
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractViolation("replay environment is closed")
        FixedReplaySuite.verify_integrity(self._suite)

    def close(self) -> None:
        self._closed = True


@dataclass(frozen=True, slots=True)
class BootstrapAudit:
    values: Mapping[str, float]
    selected_semantic: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping) or len(self.values) > 256:
            raise ValueError("bootstrap values must be a bounded mapping")
        for semantic, value in self.values.items():
            _id(semantic)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError("bootstrap values must be finite")
        if self.selected_semantic is not None:
            _id(self.selected_semantic)
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class DecisionAudit:
    learner_updated: bool | None
    bootstrap: BootstrapAudit | None = None
    reward_contributions: tuple[RewardContribution, ...] | None = None
    consumed_interval: tuple[ActionSegment, ...] | None = None

    def __post_init__(self) -> None:
        if self.learner_updated is not None and not isinstance(self.learner_updated, bool):
            raise TypeError("learner_updated must be boolean or unknown")
        if self.bootstrap is not None and not isinstance(self.bootstrap, BootstrapAudit):
            raise TypeError("bootstrap must be a BootstrapAudit")
        if self.reward_contributions is not None:
            _tuple(self.reward_contributions, RewardContribution)
        if self.consumed_interval is not None:
            _tuple(self.consumed_interval, ActionSegment)


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    action: ReplayAction
    audit: DecisionAudit
    consumptions: tuple[DecisionConsumptionReceipt, ...] = ()
    capability_gaps: tuple[CapabilityGap, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.action, ReplayAction) or not isinstance(self.audit, DecisionAudit):
            raise TypeError("decision requires a ReplayAction and DecisionAudit")
        _tuple(self.consumptions, DecisionConsumptionReceipt)
        _tuple(self.capability_gaps, CapabilityGap)


ReplayPolicy: TypeAlias = Callable[
    [TimeStep, tuple[ReplayAction, ...], RuleIndexBinding], ReplayDecision
]


@dataclass(frozen=True, slots=True)
class ReplayEvaluationResult:
    evaluator_sha256: str
    suite_sha256: str
    candidate_sha256: str
    check_counts: Mapping[str, int | None]
    coverage: Mapping[str, CheckCoverage]
    not_applicable: Mapping[str, str]
    extra_checks: Mapping[str, int]
    passed: bool
    completed_steps: int
    expected_steps: int
    issues: tuple[str, ...]
    source_provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for digest in (self.evaluator_sha256, self.suite_sha256, self.candidate_sha256):
            _digest(digest)
        if set(self.check_counts) != set(REPLAY_CHECK_METRICS) or set(self.coverage) != set(
            REPLAY_CHECK_METRICS
        ):
            raise ValueError("result must account for every contract check")
        for name in ("check_counts", "coverage", "not_applicable", "extra_checks"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def objective(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                "replay.contract_passed": float(self.passed),
                "replay.completed_steps": float(self.completed_steps),
                "replay.expected_steps": float(self.expected_steps),
                "replay.completion_ratio": self.completed_steps / self.expected_steps,
            }
        )

    def to_mapping(self) -> dict[str, Any]:
        _reject_privileged(self)
        return {
            "schema": REPLAY_RESULT_SCHEMA,
            "evaluator_sha256": self.evaluator_sha256,
            "suite_sha256": self.suite_sha256,
            "candidate_sha256": self.candidate_sha256,
            "check_counts": dict(self.check_counts),
            "coverage": {key: value.value for key, value in self.coverage.items()},
            "not_applicable": dict(self.not_applicable),
            "extra_checks": dict(self.extra_checks),
            "passed": self.passed,
            "completed_steps": self.completed_steps,
            "expected_steps": self.expected_steps,
            "objective": dict(self.objective),
            "issues": list(self.issues),
            "source_provenance": [
                {"source_id": source, "source_sha256": digest}
                for source, digest in self.source_provenance
            ],
        }


def _code_record(code: CodeType) -> dict[str, Any]:
    return {
        "bytecode": code.co_code.hex(),
        "constants": [
            _code_record(item) if isinstance(item, CodeType) else _capture_record(item)
            for item in code.co_consts
        ],
        "names": code.co_names,
        "arguments": (code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount),
        "variables": code.co_varnames,
        "freevars": code.co_freevars,
        "cellvars": code.co_cellvars,
        "locals": code.co_nlocals,
        "stack_size": code.co_stacksize,
        "exception_table": getattr(code, "co_exceptiontable", b"").hex(),
        "flags": code.co_flags,
    }


def _capture_record(value: Any, *, depth: int = 0) -> dict[str, Any]:
    """Preserve data types and mapping iteration order in callback captures."""
    _reject_privileged(value, recursive=depth == 0)
    if depth > 32:
        raise ValueError("callback capture nesting exceeds replay bounds")
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    payload: Any
    if value is None or isinstance(value, (str, bool, int, float)):
        payload = value
    elif isinstance(value, (bytes, Path, UUID, np.dtype, Enum)):
        payload = _primitive(value)
    elif isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or value.nbytes > _MAX_RECORD_BYTES:
            raise ValueError("callback tensor exceeds explicit numeric data bounds")
        payload = {
            **_primitive(value),
            "strides": value.strides,
            "writeable": value.flags.writeable,
        }
    elif isinstance(value, Mapping):
        if len(value) > _MAX_STEPS + 1:
            raise ValueError("callback mapping exceeds replay bounds")
        payload = [
            [_capture_record(key, depth=depth + 1), _capture_record(item, depth=depth + 1)]
            for key, item in value.items()
        ]
    elif isinstance(value, (tuple, list, set, frozenset)):
        if len(value) > _MAX_STEPS + 1:
            raise ValueError("callback sequence exceeds replay bounds")
        payload = [_capture_record(item, depth=depth + 1) for item in value]
        if isinstance(value, (set, frozenset)):
            payload = sorted(payload, key=_json)
    elif is_dataclass(value) and not isinstance(value, type):
        payload = {
            item.name: _capture_record(getattr(value, item.name), depth=depth + 1)
            for item in fields(value)
        }
    else:
        raise TypeError("callback captures must use explicit data contracts")
    return {"type": kind, "value": payload}


def evaluator_sha256(policy: ReplayPolicy | None = None) -> str:
    """Bind installed implementation and a static Python callback with frozen captures.

    With no policy this reports the replay and strict reward implementation digest. Campaign
    acceptance must use the callback-bound variant. Opaque callable objects,
    dynamic source and captures outside bounded JSON/data contracts are refused.
    """
    module_digest = _sha(
        _json(
            {
                "replay": _artifact_hash(Path(__file__)),
                "correlated_rewards": _artifact_hash(Path(correlated_rewards.__file__)),
                "training": _artifact_hash(Path(training.__file__)),
                "training_safety": _artifact_hash(Path(training_safety.__file__)),
            }
        )
    )
    if policy is None:
        return module_digest
    if not isinstance(policy, FunctionType):
        raise ContractViolation("replay evaluator callback must be a static Python function")
    source_path = Path(policy.__code__.co_filename)
    source_digest = _artifact_hash(source_path)
    record = {
        "module_sha256": module_digest,
        "callback_source_sha256": source_digest,
        "qualname": policy.__qualname__,
        "code": _code_record(policy.__code__),
        "defaults": _capture_record(policy.__defaults__),
        "keyword_defaults": _capture_record(policy.__kwdefaults__),
        "captures": [_capture_record(cell.cell_contents) for cell in policy.__closure__ or ()],
    }
    return _sha(_json(record, maximum=_MAX_SUITE_BYTES))


def _artifact_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ContractViolation("candidate must be a bounded regular inert artifact")
    data = path.read_bytes()
    if len(data) > _MAX_ARTIFACT_BYTES:
        raise ContractViolation("candidate artifact exceeds its byte budget")
    return hashlib.sha256(data).hexdigest()


def evaluate_replay(
    suite: FixedReplaySuite,
    candidate_path: str | Path,
    policy: ReplayPolicy,
    *,
    expected_suite_sha256: str,
    expected_candidate_sha256: str,
    step_guard: Callable[[], None] | None = None,
) -> ReplayEvaluationResult:
    """Derive checks; optional owner guard runs between steps, not as a callback sandbox.

    A blocked callback requires an external owned supervisor. This function
    cannot interrupt a callback or promise a wall-clock termination deadline.
    """
    _digest(expected_suite_sha256)
    _digest(expected_candidate_sha256)
    FixedReplaySuite.verify_integrity(suite)
    if _fixed_suite_sha256(suite) != expected_suite_sha256:
        raise ContractViolation(
            "external replay suite does not match the frozen evaluator admission"
        )
    path = Path(candidate_path)
    before = _artifact_hash(path)
    if before != expected_candidate_sha256:
        raise ContractViolation("candidate differs from the admitted artifact")
    fixed_evaluator = evaluator_sha256(policy)
    counts: dict[str, int | None] = {metric: 0 for metric in REPLAY_CHECK_METRICS}
    coverage = {metric: CheckCoverage.AUDITED for metric in REPLAY_CHECK_METRICS}
    for metric in suite.not_applicable:
        counts[metric] = None
        coverage[metric] = CheckCoverage.NOT_APPLICABLE
    extra = {
        "evaluation.rule_consumption_errors": 0,
        "evaluation.illegal_action_requests": 0,
        "evaluation.capability_gap_requests": 0,
        "evaluation.target_binding_errors": 0,
        "evaluation.evaluator_mutations": 0,
        "evaluation.indeterminate_actions": 0,
    }
    issues: list[str] = []

    def issue(message: str) -> None:
        if len(issues) < 128:
            issues.append(message[:512])

    def record(metric: str, error: bool | None) -> None:
        if metric in suite.not_applicable:
            return
        if error is None:
            counts[metric] = None
            coverage[metric] = CheckCoverage.UNKNOWN
            issue(f"missing required evidence: {metric}")
        elif counts[metric] is not None:
            counts[metric] = cast(int, counts[metric]) + int(error)

    completed = 0
    logical_ids: set[UUID] = set()
    source_epochs: set[tuple[str, UUID]] = set()
    replay = ReplayEnvironment(suite)
    environment = ContractEnvironment(replay)
    try:
        for episode in suite.episodes:
            source_initial = ReplayFrame.snapshot(episode.frames[0])
            causal_guard = _fixed_reward_guard(suite, episode)
            if causal_guard is not None:
                causal_guard.reset(source_initial.episode_id)
            identity_bad = source_initial.step_id != 0 or source_initial.done
            if episode.run_id is not None:
                source_epoch = (episode.run_id, source_initial.episode_id)
                identity_bad |= source_epoch in source_epochs
                source_epochs.add(source_epoch)
            for index, frame in enumerate(episode.frames):
                source = ReplayFrame.snapshot(frame)
                identity_bad |= (
                    source.episode_id != source_initial.episode_id or source.step_id != index
                )
                if source.action_receipt is not None:
                    if source.action_receipt.outcome in _UNRESOLVED_ACTION_OUTCOMES:
                        # Keep the stable metric key; count every outcome needing reconciliation.
                        extra["evaluation.indeterminate_actions"] += 1
                        issue("unresolved action cannot produce a learner-facing sample")
                    target = suite.spec.metadata.get("target_id")
                    if source.action_receipt.target_id != target:
                        extra["evaluation.target_binding_errors"] += 1
                        issue("action receipt lacks or differs from the frozen target binding")
                    try:
                        source.action_receipt.validate_against(source)
                    except ValueError:
                        identity_bad = True
            record(REPLAY_CHECK_METRICS[1], identity_bad if episode.run_id is not None else None)
            if step_guard is not None:
                step_guard()
            current = environment.reset()
            record(
                REPLAY_CHECK_METRICS[1], current.episode_id in logical_ids or current.step_id != 0
            )
            logical_ids.add(current.episode_id)
            for index, captured in enumerate(episode.recorded_actions):
                pre = episode.frames[index]
                post = episode.frames[index + 1]
                if step_guard is not None:
                    step_guard()
                decision = policy(current, pre.legal_actions, suite.binding)
                if step_guard is not None:
                    step_guard()
                FixedReplaySuite.verify_integrity(suite)
                if not isinstance(decision, ReplayDecision):
                    raise ContractViolation("policy returned an untyped replay decision")
                legal = any(_same_action(decision.action, action) for action in pre.legal_actions)
                if not legal or not _mask_allows(suite.spec, pre, decision.action):
                    extra["evaluation.illegal_action_requests"] += 1
                    issue("candidate requested an action outside the frozen legal surface")
                    break
                if not _same_action(decision.action, captured):
                    issue("candidate chose an unrecorded branch; no counterfactual evidence")
                    break
                _audit_knowledge(suite, pre, decision, extra, issue)
                following = environment.step(decision.action.action)
                causal_error = _strict_reward_error(causal_guard, pre, post, captured.action, issue)
                _audit_step(suite, pre, post, decision.audit, record, causal_error=causal_error)
                current = following
                completed += 1
    except Exception as error:
        issue(f"replay refused: {error}")
    finally:
        environment.close()
    try:
        after = _artifact_hash(path)
        record(REPLAY_CHECK_METRICS[0], before != after)
        FixedReplaySuite.verify_integrity(suite)
        if _fixed_suite_sha256(suite) != expected_suite_sha256:
            raise ContractViolation("replay suite admission changed during evaluation")
    except (ContractViolation, ValueError, OSError) as error:
        record(REPLAY_CHECK_METRICS[0], True)
        issue(f"evaluation artifact or fixture changed: {error}")
    try:
        if evaluator_sha256(policy) != fixed_evaluator:
            extra["evaluation.evaluator_mutations"] += 1
            issue("external evaluator callback or captured data changed")
    except (ContractViolation, ValueError, TypeError, OSError) as error:
        extra["evaluation.evaluator_mutations"] += 1
        issue(f"external evaluator callback cannot be reverified: {error}")
    expected_steps = sum(len(episode.recorded_actions) for episode in suite.episodes)
    if completed != expected_steps:
        issue("replay did not consume all frozen steps")
        # Unexecuted checks have no observations; they must not masquerade as audited zero.
        for metric in REPLAY_CHECK_METRICS[2:]:
            if metric not in suite.not_applicable and counts[metric] == 0:
                record(metric, None)
    passed = (
        not issues
        and all(value in (None, 0) for value in counts.values())
        and not any(extra.values())
    )
    return ReplayEvaluationResult(
        evaluator_sha256=fixed_evaluator,
        suite_sha256=_fixed_suite_sha256(suite),
        candidate_sha256=before,
        check_counts=counts,
        coverage=coverage,
        not_applicable=suite.not_applicable,
        extra_checks=extra,
        passed=passed,
        completed_steps=completed,
        expected_steps=expected_steps,
        issues=tuple(issues),
        source_provenance=tuple(
            (episode.source_id, episode.source_sha256) for episode in suite.episodes
        ),
    )


def _audit_knowledge(
    suite: FixedReplaySuite,
    frame: ReplayFrame,
    decision: ReplayDecision,
    counts: dict[str, int],
    issue: Callable[[str], None],
) -> None:
    requirements = {(item.finding_id, item.consumer_id): item for item in frame.required_findings}
    receipts = {(item.finding_id, item.consumer_id): item for item in decision.consumptions}
    if len(receipts) != len(decision.consumptions) or set(receipts) != set(requirements):
        counts["evaluation.rule_consumption_errors"] += 1
        issue("required finding consumption is missing, duplicated or unexpected")
    for key, receipt in receipts.items():
        expected = requirements.get(key)
        if expected is None:
            continue
        try:
            DecisionConsumptionReceipt.assert_for_decision(
                receipt,
                decision_id=frame.decision_id,
                finding_id=expected.finding_id,
                finding_sha256=expected.finding_sha256,
                consumer_id=expected.consumer_id,
                binding=suite.binding,
            )
            if (
                receipt.state is not ConsumptionState.USED
                or receipt.use_evidence_sha256 != expected.use_evidence_sha256
            ):
                raise ContractViolation("retrieval is not the required consumer-linked use")
        except (ContractViolation, ValueError) as error:
            counts["evaluation.rule_consumption_errors"] += 1
            issue(str(error))
    semantics = tuple(action.semantic for action in frame.legal_actions)
    for gap in decision.capability_gaps:
        if gap.decision_id != frame.decision_id:
            counts["evaluation.capability_gap_requests"] += 1
            issue("capability gap belongs to another decision")
        # Passive gaps cannot authorize even a missing action; actual requests were checked above.
        if gap.required_semantic == decision.action.semantic:
            counts["evaluation.capability_gap_requests"] += 1
            issue("capability gap cannot authorize the requested action")
        if gap.kind.value == "missing_action":
            try:
                CapabilityGap.verify_missing(gap, semantics)
            except ContractViolation as error:
                counts["evaluation.capability_gap_requests"] += 1
                issue(str(error))


def _fixed_reward_guard(
    suite: FixedReplaySuite, episode: ReplayEpisode
) -> CorrelatedRewardGuard | None:
    """Use the frozen source run and evaluator-owned composition, never learner config."""
    if suite.reward_training is None or suite.reward_safety is None or episode.run_id is None:
        return None
    config = suite.spec.metadata.get("environment_config_sha256")
    if config is None:
        return None
    return CorrelatedRewardGuard(
        suite.reward_training,
        suite.reward_safety,
        CorrelationPolicy(
            episode.run_id,
            suite.spec.environment_id,
            suite.spec.protocol_version,
            suite.spec.metadata["target_id"],
            config,
        ),
    )


def _strict_reward_error(
    guard: CorrelatedRewardGuard | None,
    pre: ReplayFrame,
    post: ReplayFrame,
    action: TensorTree,
    issue: Callable[[str], None],
) -> bool | None:
    """Consume captured strict evidence. Missing coverage stays unknown."""
    before, after = ReplayFrame.snapshot(pre), ReplayFrame.snapshot(post)
    if guard is None or after.info.get(REWARD_EVIDENCE_KEY) is None:
        return None
    for frame, step in ((pre, before), (post, after)):
        context = step.info.get(OBSERVATION_CONTEXT_KEY)
        if context is None:
            return None
        if isinstance(context, Mapping):
            if context.get("alive") is None or context.get("phase") == "unknown":
                return None
            expected = (
                ObservationLifecycle.LOADING
                if context.get("phase") == "loading"
                else (
                    ObservationLifecycle.ACTIVE
                    if context.get("alive") is True and context.get("phase") == "gameplay"
                    else ObservationLifecycle.DEAD
                )
            )
            if frame.lifecycle is ObservationLifecycle.UNKNOWN:
                return None
            if frame.lifecycle is not expected:
                return True
            if frame.producer_sequence != context.get("producer_sequence"):
                return True
    try:
        # The action belongs to the captured source branch, not a candidate rewrite.
        receipt = guard.compose_timestep(before, after, action=action)
        if post.reward_contributions is None:
            return None
        composed = {
            name: value for name, value in receipt.result.contributions.items() if value != 0
        }
        captured = {
            item.term_id: item.amount for item in post.reward_contributions if item.amount != 0
        }
        if composed.keys() != captured.keys() or any(
            not math.isclose(
                value, captured[name], rel_tol=_REWARD_REL_TOLERANCE, abs_tol=_REWARD_ABS_TOLERANCE
            )
            for name, value in composed.items()
        ):
            issue("captured reward terms differ from the fixed composition")
            return True
    except (ContractViolation, ValueError, TypeError) as error:
        issue(f"strict reward evidence refused: {error}")
        return True
    return False


def _audit_step(
    suite: FixedReplaySuite,
    pre: ReplayFrame,
    post: ReplayFrame,
    audit: DecisionAudit,
    record: Callable[[str, bool | None], None],
    *,
    causal_error: bool | None,
) -> None:
    step = ReplayFrame.snapshot(post)
    receipt = step.action_receipt
    issued = None if receipt is None else receipt.issued_against_observation_sequence
    observed = None if receipt is None else receipt.authoritative_observation_sequence
    if (
        audit.learner_updated is None
        or issued is None
        or observed is None
        or post.producer_sequence is None
        or pre.producer_sequence is None
    ):
        record(REPLAY_CHECK_METRICS[2], None)
    else:
        fresh = (
            observed > issued
            and issued == pre.producer_sequence
            and observed == post.producer_sequence
        )
        record(REPLAY_CHECK_METRICS[2], audit.learner_updated and not fresh)
    if (
        audit.learner_updated is None
        or post.lifecycle is ObservationLifecycle.UNKNOWN
        or pre.lifecycle is ObservationLifecycle.UNKNOWN
    ):
        record(REPLAY_CHECK_METRICS[3], None)
    else:
        inactive = pre.lifecycle is not ObservationLifecycle.ACTIVE or (
            post.lifecycle is not ObservationLifecycle.ACTIVE
            and not (step.done and post.lifecycle is ObservationLifecycle.DEAD)
        )
        record(REPLAY_CHECK_METRICS[3], audit.learner_updated and inactive)

    if audit.bootstrap is None:
        record(REPLAY_CHECK_METRICS[4], None)
    elif step.done:
        record(REPLAY_CHECK_METRICS[4], audit.bootstrap.selected_semantic is not None)
    else:
        legal = {
            action.semantic
            for action in post.legal_actions
            if _mask_allows(suite.spec, post, action)
        }
        values = audit.bootstrap.values
        if not legal or not legal.issubset(values):
            record(REPLAY_CHECK_METRICS[4], None)
        else:
            best = max(values[semantic] for semantic in legal)
            selected = audit.bootstrap.selected_semantic
            record(
                REPLAY_CHECK_METRICS[4],
                selected not in legal or values[selected] != best,
            )

    contributions = post.reward_contributions
    if contributions is None or audit.reward_contributions is None or step.reward.size != 1:
        record(REPLAY_CHECK_METRICS[5], None)
    else:
        expected_sum = math.fsum(item.amount for item in contributions)
        bad = not math.isclose(
            float(step.reward.reshape(-1)[0]),
            expected_sum,
            rel_tol=_REWARD_REL_TOLERANCE,
            abs_tol=_REWARD_ABS_TOLERANCE,
        )
        observed_sum = math.fsum(item.amount for item in audit.reward_contributions)
        bad |= not math.isclose(
            float(step.reward.reshape(-1)[0]),
            observed_sum,
            rel_tol=_REWARD_REL_TOLERANCE,
            abs_tol=_REWARD_ABS_TOLERANCE,
        )
        bad |= len(contributions) != len(audit.reward_contributions)
        if len(contributions) == len(audit.reward_contributions):
            bad |= any(
                expected.term_id != observed.term_id
                or expected.action_id != observed.action_id
                or expected.requires_accepted != observed.requires_accepted
                or not math.isclose(
                    expected.amount,
                    observed.amount,
                    rel_tol=_REWARD_REL_TOLERANCE,
                    abs_tol=_REWARD_ABS_TOLERANCE,
                )
                for expected, observed in zip(
                    contributions, audit.reward_contributions, strict=True
                )
            )
        for item in contributions:
            if item.action_id is not None:
                bad |= receipt is None or item.action_id != receipt.action_id
            if item.requires_accepted:
                bad |= receipt is None or receipt.outcome is not ActionOutcome.ACCEPTED
        record(REPLAY_CHECK_METRICS[5], None if causal_error is None else bad or causal_error)
    interval = post.action_interval
    if interval is None or audit.consumed_interval is None or issued is None or observed is None:
        record(REPLAY_CHECK_METRICS[6], None)
    else:
        bad = not interval or _primitive(interval) != _primitive(audit.consumed_interval)
        if interval:
            bad |= interval[0].start_sequence != issued or interval[-1].end_sequence != observed
            bad |= len({part.segment_id for part in interval}) != len(interval)
            bad |= any(
                left.end_sequence != right.start_sequence for left, right in pairwise(interval)
            )
        record(REPLAY_CHECK_METRICS[6], bool(bad))


__all__ = [
    "REPLAY_CHECK_METRICS",
    "REPLAY_RESULT_SCHEMA",
    "REPLAY_SUITE_SCHEMA",
    "ActionSegment",
    "BootstrapAudit",
    "CheckCoverage",
    "DecisionAudit",
    "FixedReplaySuite",
    "KnowledgeRequirement",
    "MaskIndex",
    "ObservationLifecycle",
    "ReplayAction",
    "ReplayDecision",
    "ReplayEnvironment",
    "ReplayEpisode",
    "ReplayEvaluationResult",
    "ReplayFrame",
    "ReplayPolicy",
    "RewardContribution",
    "evaluate_replay",
    "evaluator_sha256",
]
