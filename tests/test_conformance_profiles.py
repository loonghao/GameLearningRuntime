from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np
import pytest

from game_learning_runtime import (
    CompositeSpec,
    ContractViolation,
    EnvironmentSpec,
    Event,
    GameEnvironment,
    SpaceKind,
    TensorSpec,
    TimeStep,
)
from game_learning_runtime.contracts import TensorTree
from game_learning_runtime.testing import run_environment_conformance

Boundary = Literal["terminated", "truncated", "conflict"]


@dataclass(frozen=True)
class ProfileFixture:
    profile: str
    spec: EnvironmentSpec
    observation: Callable[[int], TensorTree]
    action: TensorTree
    action_mask: TensorTree | None
    boundary: Boundary
    emits_events: bool


class ScriptedProfileEnvironment(GameEnvironment):
    def __init__(self, fixture: ProfileFixture, *, episode_length: int = 2) -> None:
        self._fixture = fixture
        self._episode_length = episode_length
        self._episode_id = UUID(int=0)
        self._step_id = 0
        self.closed = False

    @property
    def spec(self) -> EnvironmentSpec:
        return self._fixture.spec

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        del seed, options
        self._episode_id = uuid4()
        self._step_id = 0
        return self._timestep()

    def step(self, action: TensorTree) -> TimeStep:
        del action
        self._step_id += 1
        return self._timestep()

    def close(self) -> None:
        self.closed = True

    def _timestep(self) -> TimeStep:
        boundary = self._step_id >= self._episode_length
        terminated = boundary and self._fixture.boundary in {"terminated", "conflict"}
        truncated = boundary and self._fixture.boundary in {"truncated", "conflict"}
        events = (
            (Event("synthetic-event", timestamp_ns=self._step_id),)
            if self._fixture.emits_events and self._step_id > 0
            else ()
        )
        return TimeStep(
            observation=self._fixture.observation(self._step_id),
            reward=np.array([0.25], dtype=np.float32),
            terminated=np.array([terminated], dtype=np.bool_),
            truncated=np.array([truncated], dtype=np.bool_),
            episode_id=self._episode_id,
            step_id=self._step_id,
            action_mask=self._fixture.action_mask,
            events=events,
            timestamp_ns=self._step_id,
        )


class AttachOnlyProfileEnvironment(ScriptedProfileEnvironment):
    @property
    def spec(self) -> EnvironmentSpec:
        return replace(
            super().spec,
            capabilities=super().spec.capabilities | {"live-attach"},
        )

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        raise AssertionError("attach conformance must not call reset")

    def attach(self, *, options: Mapping[str, Any] | None = None) -> TimeStep:
        return super().reset(options=options)


def _turn_based_fixture() -> ProfileFixture:
    return ProfileFixture(
        profile="turn-based-masked",
        spec=EnvironmentSpec(
            environment_id="conformance.turn-based-v1",
            observation=CompositeSpec(
                {
                    "board": TensorSpec((4, 4), np.float32),
                    "phase": TensorSpec(
                        (1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=2
                    ),
                }
            ),
            action=CompositeSpec(
                {
                    "choice": TensorSpec(
                        (1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=3
                    )
                }
            ),
            action_mask=CompositeSpec(
                {"choice": TensorSpec((4,), np.bool_, kind=SpaceKind.BINARY)}
            ),
        ),
        observation=lambda step: {
            "board": np.full((4, 4), step, dtype=np.float32),
            "phase": np.array([step % 3], dtype=np.int64),
        },
        action={"choice": np.array([0], dtype=np.int64)},
        action_mask={"choice": np.array([True, True, False, False], dtype=np.bool_)},
        boundary="terminated",
        emits_events=True,
    )


def _real_time_fixture() -> ProfileFixture:
    return ProfileFixture(
        profile="real-time-combat",
        spec=EnvironmentSpec(
            environment_id="conformance.real-time-v1",
            observation=CompositeSpec(
                {
                    "actor": CompositeSpec(
                        {
                            "kinematics": TensorSpec((4,), np.float32),
                            "resources": TensorSpec((2,), np.float32, minimum=0.0),
                        }
                    ),
                    "nearby": TensorSpec((8, 3), np.float32),
                }
            ),
            action=CompositeSpec(
                {
                    "movement": TensorSpec((2,), np.float32, minimum=-1.0, maximum=1.0),
                    "ability": TensorSpec(
                        (1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=2
                    ),
                }
            ),
            action_mask=CompositeSpec(
                {"ability": TensorSpec((3,), np.bool_, kind=SpaceKind.BINARY)}
            ),
        ),
        observation=lambda step: {
            "actor": {
                "kinematics": np.full(4, step, dtype=np.float32),
                "resources": np.array([1.0, 0.5], dtype=np.float32),
            },
            "nearby": np.zeros((8, 3), dtype=np.float32),
        },
        action={
            "movement": np.array([0.25, -0.25], dtype=np.float32),
            "ability": np.array([1], dtype=np.int64),
        },
        action_mask={"ability": np.array([True, True, False], dtype=np.bool_)},
        boundary="truncated",
        emits_events=True,
    )


def _fps_fixture() -> ProfileFixture:
    return ProfileFixture(
        profile="fps",
        spec=EnvironmentSpec(
            environment_id="conformance.fps-v1",
            observation=CompositeSpec(
                {
                    "frame": TensorSpec((8, 8, 3), np.uint8, minimum=0, maximum=255),
                    "telemetry": CompositeSpec(
                        {
                            "position": TensorSpec((3,), np.float32),
                            "ammo": TensorSpec((1,), np.int64, minimum=0, maximum=100),
                        }
                    ),
                }
            ),
            action=CompositeSpec(
                {
                    "move": TensorSpec((2,), np.float32, minimum=-1.0, maximum=1.0),
                    "look": TensorSpec((2,), np.float32, minimum=-1.0, maximum=1.0),
                    "fire": TensorSpec((1,), np.bool_, kind=SpaceKind.BINARY),
                }
            ),
        ),
        observation=lambda step: {
            "frame": np.full((8, 8, 3), step, dtype=np.uint8),
            "telemetry": {
                "position": np.array([step, 0.0, 0.0], dtype=np.float32),
                "ammo": np.array([100 - step], dtype=np.int64),
            },
        },
        action={
            "move": np.array([1.0, 0.0], dtype=np.float32),
            "look": np.array([0.1, -0.1], dtype=np.float32),
            "fire": np.array([False], dtype=np.bool_),
        },
        action_mask=None,
        boundary="terminated",
        emits_events=False,
    )


def _arpg_fixture() -> ProfileFixture:
    return ProfileFixture(
        profile="arpg",
        spec=EnvironmentSpec(
            environment_id="conformance.arpg-v1",
            observation=CompositeSpec(
                {
                    "player": CompositeSpec(
                        {
                            "stats": TensorSpec((6,), np.float32),
                            "inventory_ids": TensorSpec(
                                (4,), np.int64, kind=SpaceKind.MULTI_DISCRETE, minimum=0
                            ),
                        }
                    ),
                    "threats": TensorSpec((6, 4), np.float32),
                }
            ),
            action=CompositeSpec(
                {
                    "command": CompositeSpec(
                        {
                            "kind": TensorSpec(
                                (1,),
                                np.int64,
                                kind=SpaceKind.DISCRETE,
                                minimum=0,
                                maximum=3,
                            ),
                            "target": TensorSpec(
                                (1,),
                                np.int64,
                                kind=SpaceKind.DISCRETE,
                                minimum=0,
                                maximum=5,
                            ),
                            "point": TensorSpec((2,), np.float32, minimum=-1.0, maximum=1.0),
                        }
                    )
                }
            ),
            action_mask=CompositeSpec(
                {
                    "command": CompositeSpec(
                        {"kind": TensorSpec((4,), np.bool_, kind=SpaceKind.BINARY)}
                    )
                }
            ),
        ),
        observation=lambda step: {
            "player": {
                "stats": np.full(6, step, dtype=np.float32),
                "inventory_ids": np.array([1, 2, 0, 0], dtype=np.int64),
            },
            "threats": np.zeros((6, 4), dtype=np.float32),
        },
        action={
            "command": {
                "kind": np.array([1], dtype=np.int64),
                "target": np.array([0], dtype=np.int64),
                "point": np.array([0.0, 0.0], dtype=np.float32),
            }
        },
        action_mask={"command": {"kind": np.array([True, True, False, False], dtype=np.bool_)}},
        boundary="terminated",
        emits_events=True,
    )


PROFILES = (
    _turn_based_fixture(),
    _real_time_fixture(),
    _fps_fixture(),
    _arpg_fixture(),
)


@pytest.mark.parametrize("fixture", PROFILES, ids=lambda fixture: fixture.profile)
def test_cross_game_profile_conforms(fixture: ProfileFixture) -> None:
    environment = ScriptedProfileEnvironment(fixture)

    report = run_environment_conformance(
        environment,
        lambda timestep: fixture.action,
        steps=4,
        seed=7,
    )

    assert report.transition_count == 4
    assert report.episode_count == 2
    assert report.masked_transition_count == (4 if fixture.action_mask is not None else 0)
    assert report.event_count == (4 if fixture.emits_events else 0)
    assert report.terminated_transition_count == (2 if fixture.boundary == "terminated" else 0)
    assert report.truncated_transition_count == (2 if fixture.boundary == "truncated" else 0)
    assert environment.closed
    assert "environment" not in asdict(report)


def test_live_attach_profile_uses_the_same_privacy_safe_conformance_path() -> None:
    fixture = _real_time_fixture()
    environment = AttachOnlyProfileEnvironment(fixture)

    report = run_environment_conformance(
        environment,
        lambda timestep: fixture.action,
        steps=3,
        start_mode="attach",
    )

    assert report.transition_count == 3
    assert report.episode_count == 2
    assert environment.closed


def test_conformance_rejects_conflicting_boundaries_and_closes_environment() -> None:
    base = _turn_based_fixture()
    fixture = ProfileFixture(
        profile="conflicting-boundary",
        spec=base.spec,
        observation=base.observation,
        action=base.action,
        action_mask=base.action_mask,
        boundary="conflict",
        emits_events=False,
    )
    environment = ScriptedProfileEnvironment(fixture, episode_length=1)

    with pytest.raises(ContractViolation, match="both terminated and truncated"):
        run_environment_conformance(
            environment,
            lambda timestep: fixture.action,
            steps=1,
        )

    assert environment.closed


def test_conformance_rejects_invalid_step_count_before_reset() -> None:
    fixture = _fps_fixture()
    environment = ScriptedProfileEnvironment(fixture)

    with pytest.raises(ValueError, match="steps must be positive"):
        run_environment_conformance(
            environment,
            lambda timestep: fixture.action,
            steps=0,
        )

    assert environment.closed
