"""Runtime data contracts shared by adapters, collectors, and learners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import time_ns
from types import MappingProxyType
from typing import Any, TypeAlias
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

TensorTree: TypeAlias = Mapping[str, "NDArray[Any] | TensorTree"]


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
    events: tuple[Event, ...] = ()
    info: Mapping[str, Any] = field(default_factory=dict)
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
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "info", MappingProxyType(dict(self.info)))

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
