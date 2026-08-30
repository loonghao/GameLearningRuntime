from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")
spaces = gymnasium.spaces

from game_learning_runtime import ContractEnvironment, ContractViolation  # noqa: E402
from game_learning_runtime.integrations.gymnasium import GymnasiumEnvironment  # noqa: E402

pytestmark = pytest.mark.gymnasium


class _DiscreteEnvironment(gymnasium.Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def __init__(self, *, mask: np.ndarray | None = None) -> None:
        self.mask = mask if mask is not None else np.array([True, False, True])
        self.last_action: int | None = None
        self.closed = False

    def action_masks(self) -> np.ndarray:
        return self.mask

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del seed, options
        return np.array([0.25, -0.25], dtype=np.float32), {
            "score": 7,
            "local_path": "/private/runtime/session.json",
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.last_action = action
        return (
            np.array([0.5, -0.5], dtype=np.float32),
            1.5,
            action == 2,
            False,
            {"score": 8, "local_path": "/private/runtime/session.json"},
        )

    def close(self) -> None:
        self.closed = True


def test_discrete_environment_converts_lifecycle_and_masks_without_leaking_info() -> None:
    source = _DiscreteEnvironment()
    environment = ContractEnvironment(
        GymnasiumEnvironment(
            source,
            environment_id="example.discrete-v1",
            action_mask_provider=source.action_masks,
        )
    )

    initial = environment.reset(seed=42)
    following = environment.step({"action": np.array([2], dtype=np.int64)})

    assert set(initial.observation) == {"observation"}
    np.testing.assert_array_equal(initial.action_mask["action"], np.array([True, False, True]))
    assert initial.info == {}
    assert following.info == {}
    assert source.last_action == 2
    assert following.step_id == 1
    assert following.done
    np.testing.assert_array_equal(following.reward, np.array([1.5], dtype=np.float32))

    environment.close()
    assert source.closed


class _AttachableDiscreteEnvironment(_DiscreteEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.reset_calls = 0
        self.attach_calls = 0
        self.attach_options: Mapping[str, Any] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.reset_calls += 1
        return super().reset(seed=seed, options=options)

    def attach(
        self, *, options: Mapping[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.attach_calls += 1
        self.attach_options = options
        return np.array([0.75, -0.75], dtype=np.float32), {
            "score": 11,
            "local_path": "/private/runtime/live.json",
        }


def test_live_attach_provider_is_explicit_and_does_not_alias_reset() -> None:
    source = _AttachableDiscreteEnvironment()
    adapter = GymnasiumEnvironment(
        source,
        environment_id="example.live-v1",
        action_mask_provider=source.action_masks,
        attach_provider=source.attach,
        info_transform=lambda value: {"score": int(value["score"])},
    )
    environment = ContractEnvironment(adapter)

    initial = environment.attach(options={"continuation": "current"})

    assert "live-attach" in adapter.spec.capabilities
    assert source.attach_calls == 1
    assert source.reset_calls == 0
    assert source.attach_options == {"continuation": "current"}
    assert initial.step_id == 0
    assert initial.info == {"score": 11}
    np.testing.assert_array_equal(
        initial.observation["observation"],
        np.array([0.75, -0.75], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        initial.action_mask["action"],
        np.array([True, False, True]),
    )


def test_gymnasium_adapter_without_provider_rejects_live_attach() -> None:
    adapter = GymnasiumEnvironment(
        _DiscreteEnvironment(),
        environment_id="example.reset-only-v1",
    )
    environment = ContractEnvironment(adapter)

    assert "live-attach" not in adapter.spec.capabilities
    with pytest.raises(ContractViolation, match="does not declare live-attach"):
        environment.attach()


def test_info_transform_is_the_only_metadata_export_path() -> None:
    source = _DiscreteEnvironment()
    environment = ContractEnvironment(
        GymnasiumEnvironment(
            source,
            environment_id="example.filtered-info-v1",
            info_transform=lambda value: {"score": int(value["score"])},
        )
    )

    initial = environment.reset()
    following = environment.step({"action": np.array([0], dtype=np.int64)})

    assert initial.info == {"score": 7}
    assert following.info == {"score": 8}
    assert "local_path" not in initial.info
    assert "local_path" not in following.info


class _NestedEnvironment(gymnasium.Env):
    observation_space = spaces.Dict(
        {
            "phase": spaces.Discrete(4),
            "position": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
        }
    )
    action_space = spaces.Dict(
        {
            "ability": spaces.Discrete(3),
            "target": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        }
    )

    def __init__(self) -> None:
        self.last_action: Mapping[str, Any] | None = None

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        del kwargs
        return {
            "phase": 1,
            "position": np.array([2.0, 3.0], dtype=np.float32),
        }, {}

    def step(
        self, action: Mapping[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.last_action = action
        return (
            {
                "phase": 2,
                "position": np.array([3.0, 4.0], dtype=np.float32),
            },
            0.25,
            False,
            False,
            {},
        )


def test_nested_dict_spaces_preserve_semantic_action_and_observation_names() -> None:
    source = _NestedEnvironment()
    environment = ContractEnvironment(
        GymnasiumEnvironment(source, environment_id="example.hybrid-v1")
    )

    initial = environment.reset()
    environment.step(
        {
            "ability": np.array([2], dtype=np.int64),
            "target": np.array([0.5, -0.5], dtype=np.float32),
        }
    )

    assert int(initial.observation["phase"][0]) == 1
    np.testing.assert_array_equal(
        initial.observation["position"], np.array([2.0, 3.0], dtype=np.float32)
    )
    assert source.last_action is not None
    assert source.last_action["ability"] == 2
    np.testing.assert_array_equal(
        source.last_action["target"], np.array([0.5, -0.5], dtype=np.float32)
    )


def test_invalid_action_mask_fails_at_the_glr_contract_boundary() -> None:
    source = _DiscreteEnvironment(mask=np.array([True, False]))
    environment = ContractEnvironment(
        GymnasiumEnvironment(
            source,
            environment_id="example.invalid-mask-v1",
            action_mask_provider=source.action_masks,
        )
    )

    with pytest.raises(ContractViolation, match=r"action_mask\.action has shape"):
        environment.reset()


def test_non_boolean_action_mask_is_not_silently_coerced() -> None:
    source = _DiscreteEnvironment(mask=np.array([1, 0, 1], dtype=np.int8))
    environment = ContractEnvironment(
        GymnasiumEnvironment(
            source,
            environment_id="example.invalid-mask-dtype-v1",
            action_mask_provider=source.action_masks,
        )
    )

    with pytest.raises(ContractViolation, match=r"action_mask\.action has dtype"):
        environment.reset()


class _TupleEnvironment(gymnasium.Env):
    observation_space = spaces.Tuple((spaces.MultiDiscrete([2, 3]), spaces.MultiBinary(2)))
    action_space = spaces.Tuple((spaces.MultiDiscrete([3, 2]), spaces.MultiBinary(2)))

    def __init__(self) -> None:
        self.last_action: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self, **kwargs: Any) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        del kwargs
        return (
            np.array([1, 2], dtype=np.int64),
            np.array([1, 0], dtype=np.int8),
        ), {}

    def step(
        self, action: tuple[np.ndarray, np.ndarray]
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.last_action = action
        return (
            (
                np.array([0, 1], dtype=np.int64),
                np.array([0, 1], dtype=np.int8),
            ),
            0.0,
            False,
            False,
            {},
        )


def test_tuple_multidiscrete_and_multibinary_spaces_round_trip() -> None:
    source = _TupleEnvironment()
    environment = ContractEnvironment(
        GymnasiumEnvironment(source, environment_id="example.tuple-v1")
    )

    initial = environment.reset()
    environment.step(
        {
            "item_0": np.array([2, 1], dtype=np.int64),
            "item_1": np.array([True, False], dtype=np.bool_),
        }
    )

    np.testing.assert_array_equal(initial.observation["item_0"], np.array([1, 2]))
    np.testing.assert_array_equal(initial.observation["item_1"], np.array([True, False]))
    assert source.last_action is not None
    np.testing.assert_array_equal(source.last_action[0], np.array([2, 1]))
    np.testing.assert_array_equal(source.last_action[1], np.array([1, 0]))
