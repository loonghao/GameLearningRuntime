"""Strict adapter evidence linking rewards to one settled action interval.

The adapter owns lifecycle and effect facts. These contracts validate identity,
freshness and composition; they do not authenticate a game or infer causality
from nearby timestamps. Missing evidence is a contract failure, never zero.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from typing import Any
from uuid import UUID

import numpy as np

from game_learning_runtime.contracts import ActionOutcome, ActionReceipt, TensorTree, TimeStep
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.phases import EnvironmentPhase
from game_learning_runtime.training import RewardComposer, RewardSignal, TrainingConfig
from game_learning_runtime.training_safety import (
    EpisodeRewardGuard,
    GuardedRewardResult,
    RewardSafetyConfig,
)

OBSERVATION_CONTEXT_KEY = "glr.observation-context"
REWARD_EVIDENCE_KEY = "glr.reward-evidence"
CORRELATED_REWARD_SCHEMA = "glr.correlated-reward.v1"
_SHA = re.compile(r"[0-9a-f]{64}")


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in value
        )
    ):
        raise ValueError(f"{name} must be bounded non-whitespace text")
    return value


def _sequence(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= (1 << 63) - 1:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _fields(value: object, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or len(value) != len(expected) or set(value) != expected:
        raise ValueError("evidence has missing or unexpected fields")
    return value


def tensor_tree_sha256(tree: TensorTree) -> str:
    """Hash bounded tensor values, names, dtypes and shapes without logging them."""
    digest = hashlib.sha256()
    budget = [1 << 20]

    def append(payload: bytes) -> None:
        budget[0] -= len(payload)
        if budget[0] < 0:
            raise ValueError("tensor evidence exceeds one MiB")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(value: TensorTree, depth: int) -> None:
        if depth > 16 or not isinstance(value, Mapping) or len(value) > 256:
            raise ValueError("tensor evidence exceeds structural bounds")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("tensor keys must be strings")
        append(b"mapping")
        append(str(len(value)).encode("ascii"))
        for key in sorted(value):
            if len(key) > 128:
                raise ValueError("tensor keys must be bounded")
            append(key.encode("utf-8"))
            item = value[key]
            if isinstance(item, Mapping):
                visit(item, depth + 1)
            else:
                if not isinstance(item, np.ndarray) or item.dtype.hasobject or item.dtype.fields:
                    raise TypeError("tensor evidence requires arrays without object dtype")
                if item.nbytes > budget[0]:
                    raise ValueError("tensor evidence exceeds one MiB")
                append(b"array")
                append(item.dtype.str.encode("ascii"))
                append(str(item.shape).encode("ascii"))
                append(item.tobytes(order="C"))

    visit(tree, 0)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """Adapter facts for an exact producer state, including lifecycle."""

    run_id: str
    environment_id: str
    protocol_version: str
    target_id: str
    environment_config_sha256: str
    episode_id: UUID
    step_id: int
    producer_sequence: int
    timestamp_ns: int
    phase: EnvironmentPhase
    alive: bool | None

    def __post_init__(self) -> None:
        for name in ("run_id", "environment_id", "protocol_version", "target_id"):
            _text(getattr(self, name), name)
        if (
            not isinstance(self.environment_config_sha256, str)
            or _SHA.fullmatch(self.environment_config_sha256) is None
        ):
            raise ValueError("environment_config_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        for name in ("step_id", "producer_sequence", "timestamp_ns"):
            _sequence(getattr(self, name), name)
        if not isinstance(self.phase, EnvironmentPhase):
            raise TypeError("phase must be an EnvironmentPhase")
        if self.alive is not None and not isinstance(self.alive, bool):
            raise TypeError("alive must be bool or None")

    def to_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "environment_id": self.environment_id,
            "protocol_version": self.protocol_version,
            "target_id": self.target_id,
            "environment_config_sha256": self.environment_config_sha256,
            "episode_id": str(self.episode_id),
            "step_id": self.step_id,
            "producer_sequence": self.producer_sequence,
            "timestamp_ns": self.timestamp_ns,
            "phase": self.phase.value,
            "alive": self.alive,
        }

    @classmethod
    def from_mapping(cls, value: object) -> ObservationContext:
        data = _fields(value, set(cls.__dataclass_fields__))
        if not isinstance(data["episode_id"], str) or not isinstance(data["phase"], str):
            raise TypeError("episode_id and phase must be strings")
        return cls(
            **{
                **data,
                "episode_id": UUID(data["episode_id"]),
                "phase": EnvironmentPhase(data["phase"]),
            }
        )

    @classmethod
    def from_timestep(cls, timestep: TimeStep) -> ObservationContext:
        if type(timestep) is not TimeStep:
            raise TypeError("strict observation evidence requires a base TimeStep")
        _sequence(timestep.step_id, "timestep.step_id")
        _sequence(timestep.timestamp_ns, "timestep.timestamp_ns")
        context = cls.from_mapping(timestep.info.get(OBSERVATION_CONTEXT_KEY))
        if (
            context.episode_id != timestep.episode_id
            or context.step_id != timestep.step_id
            or context.timestamp_ns != timestep.timestamp_ns
        ):
            raise ContractViolation("observation context does not match its timestep")
        sequence = timestep.info.get("observation_sequence")
        if sequence is not None and (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence != context.producer_sequence
        ):
            raise ContractViolation("producer sequence disagrees with timestep observation")
        return context


class EffectState(str, Enum):
    CONFIRMED = "confirmed"
    NO_EFFECT = "no_effect"
    UNKNOWN = "unknown"


class CorrelationBudgetError(ContractViolation):
    """Strict collector admission refused before another adapter operation."""


@dataclass(frozen=True, slots=True)
class RewardAttribution:
    """Reviewed adapter claim about one signal and one exact action."""

    signal_name: str
    source: str
    action_id: str
    before_sequence: int
    after_sequence: int
    effect: EffectState

    def __post_init__(self) -> None:
        for name in ("signal_name", "source", "action_id"):
            _text(getattr(self, name), name)
        _sequence(self.before_sequence, "before_sequence")
        _sequence(self.after_sequence, "after_sequence")
        if not isinstance(self.effect, EffectState):
            raise TypeError("effect must be an EffectState")

    def to_mapping(self) -> dict[str, object]:
        return {
            "signal_name": self.signal_name,
            "source": self.source,
            "action_id": self.action_id,
            "before_sequence": self.before_sequence,
            "after_sequence": self.after_sequence,
            "effect": self.effect.value,
        }

    @classmethod
    def from_mapping(cls, value: object) -> RewardAttribution:
        data = _fields(value, set(cls.__dataclass_fields__))
        if not isinstance(data["effect"], str):
            raise TypeError("effect must be a string")
        return cls(**{**data, "effect": EffectState(data["effect"])})


@dataclass(frozen=True, slots=True)
class CorrelationPolicy:
    """Owner-frozen identity required by the strict collection path."""

    run_id: str
    environment_id: str
    protocol_version: str
    target_id: str
    environment_config_sha256: str

    def __post_init__(self) -> None:
        for name in ("run_id", "environment_id", "protocol_version", "target_id"):
            _text(getattr(self, name), name)
        if (
            not isinstance(self.environment_config_sha256, str)
            or _SHA.fullmatch(self.environment_config_sha256) is None
        ):
            raise ValueError("environment_config_sha256 must be a lowercase SHA-256 digest")

    def validate_context(self, context: ObservationContext) -> None:
        for name in self.__dataclass_fields__:
            if getattr(context, name) != getattr(self, name):
                raise ContractViolation(f"observation context has different {name}")

    def validate_before(self, timestep: TimeStep) -> ObservationContext:
        context = ObservationContext.from_timestep(timestep)
        self.validate_context(context)
        if (
            timestep.done
            or context.phase is not EnvironmentPhase.GAMEPLAY
            or context.alive is not True
        ):
            raise ContractViolation("action requires an authoritative live gameplay observation")
        return context

    def validate_interval(
        self, before: TimeStep, after: TimeStep
    ) -> tuple[ObservationContext, ObservationContext]:
        previous = self.validate_before(before)
        following = ObservationContext.from_timestep(after)
        self.validate_context(following)
        if following.episode_id != previous.episode_id or following.step_id != previous.step_id + 1:
            raise ContractViolation("action interval crossed an episode or step boundary")
        if following.producer_sequence <= previous.producer_sequence:
            raise ContractViolation("action interval has no fresh authoritative post-state")
        if following.phase is not EnvironmentPhase.GAMEPLAY or (
            following.alive is not True and not (after.done and following.alive is False)
        ):
            raise ContractViolation("post-state lifecycle is unknown or outside gameplay")
        receipt = after.action_receipt
        if receipt is None or type(receipt) is not ActionReceipt:
            raise ContractViolation("strict reward attribution requires an action receipt")
        _text(receipt.action_id, "action_id")
        receipt.validate_against(after)
        if (
            receipt.target_id != self.target_id
            or receipt.issued_against_observation_sequence != previous.producer_sequence
            or receipt.authoritative_observation_sequence != following.producer_sequence
            or not previous.timestamp_ns
            <= receipt.issued_timestamp_ns
            <= receipt.observed_timestamp_ns
            <= following.timestamp_ns
        ):
            raise ContractViolation("action receipt does not match the observation interval")
        if receipt.outcome in {
            ActionOutcome.UNKNOWN,
            ActionOutcome.PARTIAL,
            ActionOutcome.INDETERMINATE,
        }:
            raise ContractViolation("unresolved action outcome cannot produce learner data")
        return previous, following


@dataclass(frozen=True, slots=True)
class CorrelatedRewardReceipt:
    """Bounded diagnostic proof of validated composition, never authority."""

    before: ObservationContext
    after: ObservationContext
    action_id: str
    outcome: ActionOutcome
    result: GuardedRewardResult
    attributions: tuple[RewardAttribution, ...]
    state_sha256: str
    action_sha256: str
    next_state_sha256: str
    observed_reward: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.before) is not ObservationContext
            or type(self.after) is not ObservationContext
        ):
            raise TypeError("reward receipt requires base observation contexts")
        if type(self.result) is not GuardedRewardResult or type(self.outcome) is not ActionOutcome:
            raise TypeError("reward receipt requires base result and outcome contracts")
        if type(self.result.terminal) is not bool or len(self.result.contributions) > 257:
            raise ValueError("reward result requires a bounded contribution map and bool terminal")
        for name, value in self.result.contributions.items():
            _text(name, "contribution name")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("reward contributions must be finite numbers")
        for name in (
            "total",
            "episode_total",
            "positive_shaping_total",
            "negative_shaping_total",
            "suppressed_positive_shaping",
            "suppressed_negative_shaping",
        ):
            value = getattr(self.result, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("reward result scalars must be finite numbers")
        _text(self.action_id, "action_id")
        if (
            not isinstance(self.attributions, tuple)
            or len(self.attributions) > 256
            or any(type(item) is not RewardAttribution for item in self.attributions)
        ):
            raise TypeError("reward receipt requires bounded base attribution contracts")
        for name in ("state_sha256", "action_sha256", "next_state_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.observed_reward is not None and (
            isinstance(self.observed_reward, bool)
            or not isinstance(self.observed_reward, (int, float))
            or not math.isfinite(self.observed_reward)
        ):
            raise ValueError("observed_reward must be finite or None")

    @property
    def learning_reward(self) -> float:
        return self.result.total if self.observed_reward is None else self.observed_reward

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": CORRELATED_REWARD_SCHEMA,
            "before": self.before.to_mapping(),
            "after": self.after.to_mapping(),
            "action_id": self.action_id,
            "outcome": self.outcome.value,
            "reward": self.learning_reward,
            "composed_reward": self.result.total,
            "contributions": dict(self.result.contributions),
            "terminal": self.result.terminal,
            "attributions": [item.to_mapping() for item in self.attributions],
            "state_sha256": self.state_sha256,
            "action_sha256": self.action_sha256,
            "next_state_sha256": self.next_state_sha256,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CorrelatedRewardGuard:
    """Validate causal evidence before consuming reward budgets exactly once."""

    def __init__(
        self,
        training: TrainingConfig,
        safety: RewardSafetyConfig,
        policy: CorrelationPolicy,
        *,
        max_actions_per_episode: int = 4096,
        max_episodes: int = 4096,
    ) -> None:
        if type(policy) is not CorrelationPolicy:
            raise TypeError("policy must be a CorrelationPolicy")
        self._policy = policy
        if training.reward.minimum is not None and training.reward.minimum > 0:
            raise ContractViolation(
                "strict rewards cannot create positive credit through a total floor"
            )
        self._composer = RewardComposer(training)
        self._guard = EpisodeRewardGuard(training, safety)
        self._outcome_signal = safety.outcome_signal
        self._episode_id: UUID | None = None
        self._last_step: int | None = None
        self._last_context: ObservationContext | None = None
        self._last_state_sha256: str | None = None
        self._action_ids: set[str] = set()
        self._episode_ids: set[UUID] = set()
        self._action_attempts = 0
        self._reset_attempts = 0
        for limit in (max_actions_per_episode, max_episodes):
            if not 1 <= _sequence(limit, "collection budget") <= 4096:
                raise ValueError("collection budgets must be between 1 and 4096")
        self._max_actions = max_actions_per_episode
        self._max_episodes = max_episodes

    @property
    def policy(self) -> CorrelationPolicy:
        return self._policy

    def reset(self, episode_id: UUID) -> None:
        if not isinstance(episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        if episode_id in self._episode_ids:
            raise ContractViolation("reward reset requires a fresh episode identity")
        if len(self._episode_ids) >= self._max_episodes:
            raise ContractViolation("strict collection episode budget is exhausted")
        self._guard.reset()
        self._episode_id = episode_id
        self._last_step = None
        self._last_context = None
        self._last_state_sha256 = None
        self._action_attempts = 0
        self._action_ids.clear()
        self._episode_ids.add(episode_id)

    def begin_reset(self) -> None:
        """Charge one collector reset/attach attempt before calling the adapter."""
        if (
            self._reset_attempts >= self._max_episodes
            or len(self._episode_ids) >= self._max_episodes
        ):
            raise CorrelationBudgetError("strict collection reset attempt budget is exhausted")
        self._reset_attempts += 1

    def check_action(self, timestep: TimeStep) -> ObservationContext:
        """Read-only admission, before invoking a policy or dispatching an action."""
        context = self.policy.validate_before(timestep)
        if context.episode_id != self._episode_id:
            raise ContractViolation("call reset with the authoritative fresh episode identity")
        if self._action_attempts >= self._max_actions or len(self._action_ids) >= self._max_actions:
            raise CorrelationBudgetError("strict collection action attempt budget is exhausted")
        if self._last_context is not None and context != self._last_context:
            raise ContractViolation(
                "reward pre-state differs from the previous validated post-state"
            )
        if (
            self._last_state_sha256 is not None
            and tensor_tree_sha256(timestep.observation) != self._last_state_sha256
        ):
            raise ContractViolation(
                "reward pre-state tensor differs from the previous validated post-state"
            )
        return context

    def begin_action(self, timestep: TimeStep) -> None:
        """Charge a collector dispatch attempt; a failed attempt is not refunded."""
        self.check_action(timestep)
        self._action_attempts += 1

    def compose(
        self,
        before: TimeStep,
        after: TimeStep,
        signals: Iterable[RewardSignal],
        attributions: Iterable[RewardAttribution],
        *,
        action: TensorTree,
        verify_observed_reward: bool = False,
    ) -> CorrelatedRewardReceipt:
        previous, following = self.policy.validate_interval(before, after)
        if self._episode_id != before.episode_id:
            raise ContractViolation("call reset with the authoritative fresh episode identity")
        receipt = after.action_receipt
        assert receipt is not None
        if (
            self._last_step is not None and before.step_id != self._last_step
        ) or receipt.action_id in self._action_ids:
            raise ContractViolation("reward action was duplicated or skipped")
        if self._last_context is not None and previous != self._last_context:
            raise ContractViolation(
                "reward pre-state differs from the previous validated post-state"
            )
        if len(self._action_ids) >= self._max_actions:
            raise ContractViolation("strict collection action budget is exhausted")
        received = tuple(islice(signals, 257))
        if len(received) > 256:
            raise ValueError("too many reward signals")
        if any(type(signal) is not RewardSignal for signal in received):
            raise TypeError("reward signals must be RewardSignal instances")
        claims = tuple(islice(attributions, 257))
        if len(claims) > 256 or any(type(item) is not RewardAttribution for item in claims):
            raise ValueError("reward attributions must be a bounded typed sequence")
        by_name = {item.signal_name: item for item in claims}
        signals_by_name = {signal.name: signal for signal in received}
        if len(by_name) != len(claims) or set(by_name) - set(signals_by_name):
            raise ContractViolation("duplicate or unexpected reward attributions")
        composed = self._composer.compose(received)
        if not all(math.isfinite(value) for value in composed.contributions.values()):
            raise ContractViolation("reward contributions must be finite")
        terminal_signal = signals_by_name.get(self._outcome_signal)
        if following.alive is False and (
            any(value > 0 for value in composed.contributions.values())
            or (terminal_signal is not None and terminal_signal.value > 0)
        ):
            raise ContractViolation("a dead terminal state cannot claim positive outcome credit")
        for name, claim in by_name.items():
            if (
                claim.source != signals_by_name[name].source
                or claim.action_id != receipt.action_id
                or claim.before_sequence != previous.producer_sequence
                or claim.after_sequence != following.producer_sequence
            ):
                raise ContractViolation(
                    "reward attribution has a different action or producer interval"
                )
        for name, contribution in composed.contributions.items():
            if contribution > 0:
                positive_claim = by_name.get(name)
                if (
                    receipt.outcome is not ActionOutcome.ACCEPTED
                    or following.alive is not True
                    or positive_claim is None
                    or positive_claim.effect is not EffectState.CONFIRMED
                ):
                    raise ContractViolation("positive reward requires a confirmed accepted effect")
        state_digest = tensor_tree_sha256(before.observation)
        action_digest = tensor_tree_sha256(action)
        next_state_digest = tensor_tree_sha256(after.observation)
        if self._last_state_sha256 is not None and state_digest != self._last_state_sha256:
            raise ContractViolation(
                "reward pre-state tensor differs from the previous validated post-state"
            )
        preview = copy.copy(self._guard).compose(received, terminal=after.done)
        observed_reward = None
        if verify_observed_reward:
            if (
                after.reward.size != 1
                or after.reward.dtype.kind != "f"
                or after.reward.dtype.itemsize not in (4, 8)
                or not np.isfinite(after.reward).all()
            ):
                raise ContractViolation("strict reward requires a finite float32 or float64 scalar")
            expected_reward = float(np.asarray(preview.total, dtype=after.reward.dtype).item())
            observed_reward = float(after.reward.item())
            if observed_reward != expected_reward:
                raise ContractViolation(
                    "observed reward differs from the validated bounded composition"
                )
        validated_receipt = CorrelatedRewardReceipt(
            previous,
            following,
            receipt.action_id,
            receipt.outcome,
            preview,
            claims,
            state_digest,
            action_digest,
            next_state_digest,
            observed_reward,
        )
        self._guard.compose(received, terminal=after.done)
        self._last_step = after.step_id
        self._last_context = following
        self._last_state_sha256 = next_state_digest
        self._action_ids.add(receipt.action_id)
        return validated_receipt

    def compose_timestep(
        self, before: TimeStep, after: TimeStep, *, action: TensorTree
    ) -> CorrelatedRewardReceipt:
        data = _fields(after.info.get(REWARD_EVIDENCE_KEY), {"signals", "attributions"})
        for name in ("signals", "attributions"):
            if not isinstance(data[name], (list, tuple)) or len(data[name]) > 256:
                raise ValueError("reward evidence must contain bounded sequences")
        signals = tuple(
            RewardSignal(**_fields(item, {"name", "source", "value"})) for item in data["signals"]
        )
        claims = tuple(RewardAttribution.from_mapping(item) for item in data["attributions"])
        return self.compose(
            before, after, signals, claims, action=action, verify_observed_reward=True
        )


@dataclass(frozen=True, slots=True)
class ScalarLearningUpdate:
    """Learner-reported scalar TD operands, bound to one validated transition.

    This checks a consumer's reported arithmetic. It does not read or mutate a
    learner table and is not evidence that an uninstrumented learner updated.
    ``bootstrap_value`` is already selected under the learner's legal mask;
    the runtime never invents a legal next action or estimates a missing value.
    """

    learner_id: str
    table_id: str
    policy_version: int
    state_sha256: str
    action_sha256: str
    next_state_sha256: str
    reward_receipt_sha256: str
    previous_value: float
    bootstrap_value: float
    discount: float
    learning_rate: float
    target: float
    updated_value: float

    def __post_init__(self) -> None:
        _text(self.learner_id, "learner_id")
        _text(self.table_id, "table_id")
        _sequence(self.policy_version, "policy_version")
        for name in ("state_sha256", "action_sha256", "next_state_sha256", "reward_receipt_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "previous_value",
            "bootstrap_value",
            "discount",
            "learning_rate",
            "target",
            "updated_value",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.discount <= 1 or not 0 < self.learning_rate <= 1:
            raise ValueError("discount and learning rate must be within their admitted bounds")

    def validate_against(self, receipt: CorrelatedRewardReceipt) -> None:
        if type(receipt) is not CorrelatedRewardReceipt:
            raise TypeError("receipt must be a base CorrelatedRewardReceipt")
        if self.reward_receipt_sha256 != receipt.sha256:
            raise ContractViolation("learning update refers to a different reward receipt")
        if (
            self.state_sha256 != receipt.state_sha256
            or self.action_sha256 != receipt.action_sha256
            or self.next_state_sha256 != receipt.next_state_sha256
        ):
            raise ContractViolation("learning update refers to different state or action operands")
        if receipt.result.terminal and self.bootstrap_value != 0:
            raise ContractViolation("terminal updates must not bootstrap another state")
        target = receipt.learning_reward + self.discount * self.bootstrap_value
        updated = self.previous_value + self.learning_rate * (target - self.previous_value)
        if not math.isclose(self.target, target, rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(
            self.updated_value, updated, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ContractViolation(
                "learning update arithmetic does not match the correlated reward"
            )

    def to_mapping(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in ScalarLearningUpdate.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class LearningConsumerPolicy:
    """Owner-selected consumer identity; telemetry cannot choose another table."""

    learner_id: str
    table_id: str
    policy_version: int

    def __post_init__(self) -> None:
        _text(self.learner_id, "learner_id")
        _text(self.table_id, "table_id")
        _sequence(self.policy_version, "policy_version")

    def validate(self, update: ScalarLearningUpdate) -> None:
        if type(update) is not ScalarLearningUpdate:
            raise TypeError("update must be a ScalarLearningUpdate")
        if any(getattr(update, name) != getattr(self, name) for name in self.__dataclass_fields__):
            raise ContractViolation(
                "learning update belongs to a different consumer or policy version"
            )
