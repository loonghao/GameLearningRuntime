"""Runtime data contracts shared by adapters, collectors, and learners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from time import time_ns
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.realtime import RealtimeActionReceipt

TensorTree: TypeAlias = Mapping[str, "NDArray[Any] | TensorTree"]


class ActionOutcome(str, Enum):
    """Portable outcome of one mutating realtime action."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    NO_EFFECT = "no_effect"
    PARTIAL = "partial"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    """Bounded, authoritative receipt for a realtime action attempt."""

    action_id: str
    episode_id: UUID
    step_id: int
    outcome: ActionOutcome
    issued_timestamp_ns: int
    observed_timestamp_ns: int
    postcondition: str = "unknown"
    progress_delta: float | None = None
    authoritative_observation_sequence: int | None = None
    retryable: bool = False
    realtime: RealtimeActionReceipt | None = None

    def __post_init__(self) -> None:
        if not self.action_id or len(self.action_id) > 128:
            raise ValueError("action_id must contain 1-128 characters")
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        if not isinstance(self.step_id, int) or isinstance(self.step_id, bool) or self.step_id <= 0:
            raise ValueError("step_id must be a positive integer")
        if not isinstance(self.outcome, ActionOutcome):
            try:
                object.__setattr__(self, "outcome", ActionOutcome(self.outcome))
            except ValueError as error:
                raise ValueError(f"unsupported action outcome: {self.outcome!r}") from error
        for name in ("issued_timestamp_ns", "observed_timestamp_ns"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.observed_timestamp_ns < self.issued_timestamp_ns:
            raise ValueError("observed_timestamp_ns cannot precede issued_timestamp_ns")
        if not self.postcondition or len(self.postcondition) > 128:
            raise ValueError("postcondition must contain 1-128 characters")
        if self.progress_delta is not None and (
            isinstance(self.progress_delta, bool)
            or not isinstance(self.progress_delta, (int, float))
            or not isfinite(float(self.progress_delta))
        ):
            raise ValueError("progress_delta must be a finite number or None")
        if self.authoritative_observation_sequence is not None and (
            not isinstance(self.authoritative_observation_sequence, int)
            or isinstance(self.authoritative_observation_sequence, bool)
            or self.authoritative_observation_sequence < 0
        ):
            raise ValueError(
                "authoritative_observation_sequence must be a non-negative integer or None"
            )
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be bool")
        if self.realtime is not None and not isinstance(self.realtime, RealtimeActionReceipt):
            raise TypeError("realtime must be a RealtimeActionReceipt or None")
        if self.realtime is not None and self.realtime.action_id != self.action_id:
            raise ValueError("realtime receipt action_id must match action_id")

    def validate_against(self, timestep: TimeStep) -> None:
        """Ensure the receipt belongs to the authoritative post-state."""

        if self.episode_id != timestep.episode_id or self.step_id != timestep.step_id:
            raise ValueError("action receipt does not match the authoritative timestep")


class ReconciliationOutcome(str, Enum):
    """Provider verdict for an action whose response was lost in transit."""

    APPLIED = "applied"
    NOT_APPLIED = "not_applied"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ActionReconciliation:
    """One in-flight action verdict returned by a reconnect handshake."""

    episode_id: UUID
    expected_step_id: int
    outcome: ReconciliationOutcome
    authoritative_step_id: int
    timestamp_ns: int
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        if (
            not isinstance(self.expected_step_id, int)
            or isinstance(self.expected_step_id, bool)
            or self.expected_step_id <= 0
        ):
            raise ValueError("expected_step_id must be a positive integer")
        if not isinstance(self.outcome, ReconciliationOutcome):
            try:
                object.__setattr__(self, "outcome", ReconciliationOutcome(self.outcome))
            except ValueError as error:
                raise ValueError(f"unsupported reconciliation outcome: {self.outcome!r}") from error
        if (
            not isinstance(self.authoritative_step_id, int)
            or isinstance(self.authoritative_step_id, bool)
            or self.authoritative_step_id < 0
        ):
            raise ValueError("authoritative_step_id must be a non-negative integer")
        if (
            not isinstance(self.timestamp_ns, int)
            or isinstance(self.timestamp_ns, bool)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be bool")


def freeze_tree(value: Mapping[str, Any]) -> TensorTree:
    """Copy a nested tensor mapping and make each array read-only."""

    frozen: dict[str, NDArray[Any] | TensorTree] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            frozen[key] = freeze_tree(item)
        else:
            array = np.array(item, copy=True)
            array.flags.writeable = False
            frozen[key] = array
    return MappingProxyType(frozen)


def freeze_array(value: Any) -> NDArray[Any]:
    array = np.array(value, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class Event:
    """A timestamped semantic event emitted by a game runtime."""

    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=time_ns)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("event name cannot be empty")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


@dataclass(frozen=True, slots=True)
class TimeStep:
    """Observation and environment signals after reset or an action."""

    observation: TensorTree
    reward: NDArray[Any]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    episode_id: UUID = field(default_factory=uuid4)
    step_id: int = 0
    action_mask: TensorTree | None = None
    action_receipt: ActionReceipt | None = None
    events: tuple[Event, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ns: int = field(default_factory=time_ns)

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id cannot be negative")
        object.__setattr__(self, "observation", freeze_tree(self.observation))
        object.__setattr__(self, "reward", freeze_array(self.reward))
        object.__setattr__(self, "terminated", freeze_array(self.terminated))
        object.__setattr__(self, "truncated", freeze_array(self.truncated))
        if self.action_mask is not None:
            object.__setattr__(self, "action_mask", freeze_tree(self.action_mask))
        if self.action_receipt is not None and not isinstance(self.action_receipt, ActionReceipt):
            raise TypeError("action_receipt must be an ActionReceipt or None")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))

    @property
    def done(self) -> bool:
        """Whether every participant represented by the done tensors has ended."""

        return bool(np.all(np.logical_or(self.terminated, self.truncated)))


@dataclass(frozen=True, slots=True)
class Transition:
    """One learner-neutral transition suitable for RL, BC, or offline data."""

    episode_id: UUID
    step_id: int
    observation: TensorTree
    action: TensorTree
    reward: NDArray[Any]
    next_observation: TensorTree
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    action_mask: TensorTree | None = None
    next_action_mask: TensorTree | None = None
    action_receipt: ActionReceipt | None = None
    events: tuple[Event, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] | None = None
    timestamp_ns: int = field(default_factory=time_ns)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", freeze_tree(self.observation))
        object.__setattr__(self, "action", freeze_tree(self.action))
        object.__setattr__(self, "reward", freeze_array(self.reward))
        object.__setattr__(self, "next_observation", freeze_tree(self.next_observation))
        object.__setattr__(self, "terminated", freeze_array(self.terminated))
        object.__setattr__(self, "truncated", freeze_array(self.truncated))
        if self.action_mask is not None:
            object.__setattr__(self, "action_mask", freeze_tree(self.action_mask))
        if self.next_action_mask is not None:
            object.__setattr__(self, "next_action_mask", freeze_tree(self.next_action_mask))
        if self.action_receipt is not None and not isinstance(self.action_receipt, ActionReceipt):
            raise TypeError("action_receipt must be an ActionReceipt or None")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))
        if self.provenance is not None:
            object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def done(self) -> bool:
        return bool(np.all(np.logical_or(self.terminated, self.truncated)))


@dataclass(frozen=True, slots=True)
class Unroll:
    """A fixed-length actor unroll for PPO/IMPALA-style learners."""

    transitions: tuple[Transition, ...]
    actor_id: str
    sequence_id: int
    policy_version: int = 0

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("an unroll requires at least one transition")
        if not self.actor_id:
            raise ValueError("actor_id cannot be empty")
        if self.sequence_id < 0 or self.policy_version < 0:
            raise ValueError("sequence_id and policy_version cannot be negative")
        object.__setattr__(self, "transitions", tuple(self.transitions))

    @property
    def total_reward(self) -> NDArray[Any]:
        total = np.zeros_like(self.transitions[0].reward)
        for transition in self.transitions:
            total = total + transition.reward
        return total

    @property
    def action_outcome_counts(self) -> Mapping[str, int]:
        """Count typed action outcomes without inspecting adapter ``info``."""

        counts: dict[str, int] = {}
        for transition in self.transitions:
            receipt = transition.action_receipt
            if receipt is not None:
                counts[receipt.outcome.value] = counts.get(receipt.outcome.value, 0) + 1
        return MappingProxyType(counts)

    @property
    def mask_freedom(self) -> float:
        """Fraction of masked transitions with more than one legal action.

        The denominator contains only transitions carrying an action mask;
        unmasked transitions have no mask-local count and are therefore not
        classified as forced or free by this property.
        """

        masked = [
            transition.action_mask for transition in self.transitions if transition.action_mask
        ]
        if not masked:
            return 0.0
        free = sum(
            any(np.count_nonzero(np.asarray(leaf).reshape(-1)) > 1 for leaf in _tree_leaves(mask))
            for mask in masked
        )
        return free / len(masked)


def _tree_leaves(value: TensorTree) -> tuple[NDArray[Any], ...]:
    leaves: list[NDArray[Any]] = []
    for child in value.values():
        if isinstance(child, Mapping):
            leaves.extend(_tree_leaves(child))
        else:
            leaves.append(np.asarray(child))
    return tuple(leaves)
