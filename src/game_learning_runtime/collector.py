"""Framework-neutral synchronous collection primitives."""

from __future__ import annotations

from typing import Literal, Protocol

from game_learning_runtime.contracts import TensorTree, TimeStep, Transition, Unroll
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment


class Policy(Protocol):
    """Minimal policy port shared by custom PPO, IMPALA, BC, and evaluation."""

    def __call__(self, timestep: TimeStep) -> TensorTree:
        """Choose a structured action from a time step."""
        ...


class SyncCollector:
    """Collect fixed-length unrolls without coupling to a learner framework."""

    def __init__(
        self,
        environment: GameEnvironment,
        *,
        actor_id: str = "actor-0",
        start_mode: Literal["reset", "attach"] = "reset",
    ) -> None:
        if not actor_id:
            raise ValueError("actor_id cannot be empty")
        if start_mode not in {"reset", "attach"}:
            raise ValueError("start_mode must be 'reset' or 'attach'")
        self._environment = (
            environment
            if isinstance(environment, ContractEnvironment)
            else ContractEnvironment(environment)
        )
        self._actor_id = actor_id
        self._start_mode = start_mode
        self._current: TimeStep | None = None
        self._sequence_id = 0

    def _start(self, *, seed: int | None = None) -> TimeStep:
        if self._start_mode == "attach":
            if seed is not None:
                raise ValueError("seed is not supported when start_mode='attach'")
            return self._environment.attach()
        return self._environment.reset(seed=seed)

    def collect(
        self,
        policy: Policy,
        *,
        steps: int,
        policy_version: int = 0,
        seed: int | None = None,
        stop_on_done: bool = False,
    ) -> Unroll:
        """Collect up to ``steps`` transitions.

        By default a terminal transition starts a fresh episode so the result
        remains fixed length. Set ``stop_on_done`` for long-running live games
        where an unroll must never mix progression from multiple episodes.
        """

        if steps <= 0:
            raise ValueError("steps must be positive")
        if policy_version < 0:
            raise ValueError("policy_version cannot be negative")
        if self._current is None or self._current.done:
            self._current = self._start(seed=seed)

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
                    action_receipt=following.action_receipt,
                    terminated=following.terminated,
                    truncated=following.truncated,
                    events=following.events,
                    info=following.info,
                    timestamp_ns=following.timestamp_ns,
                )
            )
            self._current = following
            if following.done:
                if stop_on_done:
                    break
                if len(transitions) < steps:
                    self._current = self._start()

        unroll = Unroll(
            transitions=tuple(transitions),
            actor_id=self._actor_id,
            sequence_id=self._sequence_id,
            policy_version=policy_version,
        )
        self._sequence_id += 1
        return unroll
