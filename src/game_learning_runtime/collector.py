"""Framework-neutral synchronous collection primitives."""

from __future__ import annotations

from typing import Protocol

from game_learning_runtime.contracts import TensorTree, TimeStep, Transition, Unroll
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment


class Policy(Protocol):
    """Minimal policy port shared by custom PPO, IMPALA, BC, and evaluation."""

    def __call__(self, timestep: TimeStep) -> TensorTree:
        """Choose a structured action from a time step."""
        ...


class SyncCollector:
    """Collect fixed-length unrolls without coupling to a learner framework."""

    def __init__(self, environment: GameEnvironment, *, actor_id: str = "actor-0") -> None:
        if not actor_id:
            raise ValueError("actor_id cannot be empty")
        self._environment = (
            environment
            if isinstance(environment, ContractEnvironment)
            else ContractEnvironment(environment)
        )
        self._actor_id = actor_id
        self._current: TimeStep | None = None
        self._sequence_id = 0

    def collect(
        self,
        policy: Policy,
        *,
        steps: int,
        policy_version: int = 0,
        seed: int | None = None,
    ) -> Unroll:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if policy_version < 0:
            raise ValueError("policy_version cannot be negative")
        if self._current is None or self._current.done:
            self._current = self._environment.reset(seed=seed)

        transitions: list[Transition] = []
        for _ in range(steps):
            current = self._current
            action = policy(current)
            following = self._environment.step(action)
            transitions.append(
                Transition(
                    episode_id=current.episode_id,
                    step_id=current.step_id,
                    observation=current.observation,
                    action=action,
                    action_mask=current.action_mask,
                    reward=following.reward,
                    next_observation=following.observation,
                    next_action_mask=following.action_mask,
                    terminated=following.terminated,
                    truncated=following.truncated,
                    events=following.events,
                    info=following.info,
                    timestamp_ns=following.timestamp_ns,
                )
            )
            self._current = following
            if following.done and len(transitions) < steps:
                self._current = self._environment.reset()

        unroll = Unroll(
            transitions=tuple(transitions),
            actor_id=self._actor_id,
            sequence_id=self._sequence_id,
            policy_version=policy_version,
        )
        self._sequence_id += 1
        return unroll
