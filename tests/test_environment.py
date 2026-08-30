from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from uuid import UUID

import numpy as np
import pytest

from game_learning_runtime import ContractEnvironment, ContractViolation, GameEnvironment, TimeStep
from game_learning_runtime.contracts import TensorTree
from game_learning_runtime.examples import CounterEnvironment
from game_learning_runtime.specs import EnvironmentSpec


def _action(value: int = 1) -> TensorTree:
    return {"choice": np.array([value], dtype=np.int64)}


def test_contract_environment_enforces_lifecycle() -> None:
    environment = ContractEnvironment(CounterEnvironment(target=1))

    with pytest.raises(ContractViolation, match="reset first"):
        environment.step(_action())
    first = environment.reset(seed=42)
    terminal = environment.step(_action())
    assert first.step_id == 0
    assert terminal.done
    with pytest.raises(ContractViolation, match="terminal"):
        environment.step(_action())

    second = environment.reset()
    assert second.episode_id != first.episode_id


def test_contract_environment_rejects_invalid_action() -> None:
    environment = ContractEnvironment(CounterEnvironment())
    environment.reset()

    with pytest.raises(ContractViolation, match="dtype"):
        environment.step({"choice": np.array([1.0], dtype=np.float32)})


class _WrongStepEnvironment(GameEnvironment):
    def __init__(self) -> None:
        self._delegate = CounterEnvironment()
        self._episode_id: UUID | None = None

    @property
    def spec(self) -> EnvironmentSpec:
        return self._delegate.spec

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        result = self._delegate.reset(seed=seed, options=options)
        self._episode_id = result.episode_id
        return result

    def step(self, action: TensorTree) -> TimeStep:
        result = self._delegate.step(action)
        return TimeStep(
            observation=result.observation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            action_mask=result.action_mask,
            episode_id=result.episode_id,
            step_id=result.step_id + 1,
        )


class _LiveAttachEnvironment(GameEnvironment):
    def __init__(self) -> None:
        self._delegate = CounterEnvironment(target=1)

    @property
    def spec(self) -> EnvironmentSpec:
        return replace(
            self._delegate.spec,
            capabilities=self._delegate.spec.capabilities | {"live-attach"},
        )

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        raise AssertionError("live attach must not call reset")

    def attach(self, *, options: Mapping[str, Any] | None = None) -> TimeStep:
        return self._delegate.reset(options=options)

    def step(self, action: TensorTree) -> TimeStep:
        return self._delegate.step(action)


class _ReusedAttachEnvironment(_LiveAttachEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self._attached: TimeStep | None = None

    def attach(self, *, options: Mapping[str, Any] | None = None) -> TimeStep:
        if self._attached is None:
            self._attached = super().attach(options=options)
        return self._attached


class _DeclaredButMissingAttachEnvironment(CounterEnvironment):
    @property
    def spec(self) -> EnvironmentSpec:
        return replace(
            super().spec,
            capabilities=super().spec.capabilities | {"live-attach"},
        )


def test_contract_environment_can_attach_without_claiming_a_physical_reset() -> None:
    environment = ContractEnvironment(_LiveAttachEnvironment())

    attached = environment.attach(options={"session": "already-running"})
    terminal = environment.step(_action())

    assert attached.step_id == 0
    assert not attached.done
    assert terminal.done


def test_contract_environment_denies_undeclared_live_attach() -> None:
    environment = ContractEnvironment(CounterEnvironment())

    with pytest.raises(ContractViolation, match="does not declare live-attach"):
        environment.attach()


def test_contract_environment_rejects_reused_attach_episode_identity() -> None:
    environment = ContractEnvironment(_ReusedAttachEnvironment())
    environment.attach()

    with pytest.raises(ContractViolation, match="attach reused"):
        environment.attach()


def test_declaring_live_attach_without_implementing_it_fails_closed() -> None:
    environment = ContractEnvironment(_DeclaredButMissingAttachEnvironment())

    with pytest.raises(ContractViolation, match="does not support live attach"):
        environment.attach()


def test_contract_environment_rejects_non_monotonic_step_id() -> None:
    environment = ContractEnvironment(_WrongStepEnvironment())
    environment.reset()

    with pytest.raises(ContractViolation, match="expected 1"):
        environment.step(_action())


def test_close_is_idempotent_and_prevents_reuse() -> None:
    environment = ContractEnvironment(CounterEnvironment())
    environment.close()
    environment.close()

    with pytest.raises(ContractViolation, match="closed"):
        environment.reset()
