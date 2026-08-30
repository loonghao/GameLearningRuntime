"""Environment port and a fail-closed runtime contract wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import TracebackType
from typing import Any
from uuid import UUID

from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.specs import EnvironmentSpec


class GameEnvironment(ABC):
    """Port implemented by in-process, RPC, shared-memory, or replay adapters."""

    @property
    @abstractmethod
    def spec(self) -> EnvironmentSpec:
        """Return the immutable environment contract."""

    @abstractmethod
    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        """Start a new episode and return step zero."""

    @abstractmethod
    def step(self, action: TensorTree) -> TimeStep:
        """Apply one structured action and return the resulting time step."""

    def close(self) -> None:
        """Release runtime resources. Implementations may override this method."""
        return None

    def __enter__(self) -> GameEnvironment:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ContractEnvironment(GameEnvironment):
    """Validates an adapter at every state transition and fails closed."""

    def __init__(self, environment: GameEnvironment) -> None:
        self._environment = environment
        self._current: TimeStep | None = None
        self._previous_episode_id: UUID | None = None
        self._closed = False

    @property
    def spec(self) -> EnvironmentSpec:
        return self._environment.spec

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        self._ensure_open()
        timestep = self._environment.reset(seed=seed, options=options)
        self._validate_timestep(timestep)
        if timestep.step_id != 0:
            raise ContractViolation(f"reset returned step_id={timestep.step_id}; expected 0")
        if timestep.done:
            raise ContractViolation("reset returned a terminal time step")
        if self._previous_episode_id == timestep.episode_id:
            raise ContractViolation("reset reused the previous episode_id")
        self._current = timestep
        self._previous_episode_id = timestep.episode_id
        return timestep

    def step(self, action: TensorTree) -> TimeStep:
        self._ensure_open()
        if self._current is None:
            raise ContractViolation("step requires reset first")
        if self._current.done:
            raise ContractViolation("step cannot follow a terminal time step; reset first")
        self.spec.action.validate(action, path="action")
        timestep = self._environment.step(action)
        self._validate_timestep(timestep)
        if timestep.episode_id != self._current.episode_id:
            raise ContractViolation("step changed episode_id without reset")
        expected_step_id = self._current.step_id + 1
        if timestep.step_id != expected_step_id:
            raise ContractViolation(
                f"step returned step_id={timestep.step_id}; expected {expected_step_id}"
            )
        self._current = timestep
        return timestep

    def close(self) -> None:
        if not self._closed:
            self._environment.close()
            self._closed = True

    def _validate_timestep(self, timestep: TimeStep) -> None:
        self.spec.observation.validate(timestep.observation, path="observation")
        self.spec.reward.validate(timestep.reward, path="reward")
        self.spec.done.validate(timestep.terminated, path="terminated")
        self.spec.done.validate(timestep.truncated, path="truncated")
        if self.spec.action_mask is None:
            if timestep.action_mask is not None:
                raise ContractViolation("adapter returned an undeclared action_mask")
        elif timestep.action_mask is None:
            raise ContractViolation("adapter omitted the declared action_mask")
        else:
            self.spec.action_mask.validate(timestep.action_mask, path="action_mask")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractViolation("environment is closed")
