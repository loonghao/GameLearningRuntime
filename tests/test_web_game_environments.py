"""Contract and adapter tests for the browser/web-game environments.

These tests never launch a browser. :class:`ScriptedBrowserBridge` stands in for
the page, so the suite proves the adapter contract, the observation encoding,
the reward shaping, and the episode boundaries without a Chromium download.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from game_learning_runtime.contracts import TimeStep
from game_learning_runtime.environment import ContractEnvironment
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.web_game import (
    DODGE_ACTIONS,
    DODGE_FEATURES,
    BlackBoxWebGameEnvironment,
    InstrumentedWebGameEnvironment,
    ScriptedBrowserBridge,
    dodge_action_mask_spec,
    dodge_action_spec,
    dodge_environment_spec,
    dodge_page_spec,
    dodge_reward,
    normalize_features,
)

# A scripted page that drifts one threat toward a player parked at x = 0.
_ALIVE_STATE: dict[str, Any] = {
    "features": {"player_x": 0.0, "threat_dx": 0.0, "threat_dy": 0.9, "neighbor_dx": 0.0},
    "alive": True,
    "score": 0.0,
    "steps": 0,
}


class _FakePage:
    """Minimal in-page model: threats close in, a collision ends the episode."""

    def __init__(self, *, crash_at_step: int = 5) -> None:
        self.crash_at_step = crash_at_step
        self.reset_calls: list[int] = []
        self.step_calls: list[int] = []
        self.steps = 0
        self.player_x = 0.0

    def handle(self, expression: str) -> Any:
        if ".reset(" in expression:
            seed = int(expression.rsplit("(", 1)[1].rstrip(")"))
            self.reset_calls.append(seed)
            self.steps = 0
            self.player_x = 0.0
            return True
        if ".step(" in expression:
            action = int(expression.rsplit("(", 1)[1].rstrip(")"))
            self.step_calls.append(action)
            self.player_x += {0: -0.1, 1: 0.0, 2: 0.1}[action]
            self.steps += 1
            return self.steps < self.crash_at_step
        if ".state()" in expression:
            return {
                "features": {
                    "player_x": self.player_x,
                    "threat_dx": -0.5,
                    "threat_dy": max(0.0, 0.9 - 0.1 * self.steps),
                    "neighbor_dx": 0.4,
                },
                "alive": self.steps < self.crash_at_step,
                "score": float(self.steps),
                "steps": self.steps,
            }
        raise AssertionError(f"unexpected expression: {expression}")


def _make_environment(crash_at_step: int = 5, **kwargs: Any) -> InstrumentedWebGameEnvironment:
    page = _FakePage(crash_at_step=crash_at_step)
    bridge = ScriptedBrowserBridge(page.handle)
    return InstrumentedWebGameEnvironment(bridge, **kwargs)


# --- specs -----------------------------------------------------------------


def test_action_and_mask_specs_agree_on_the_action_count() -> None:
    action = dodge_action_spec()
    mask = dodge_action_mask_spec()
    assert action.flatten()["choice"].maximum == len(DODGE_ACTIONS) - 1
    assert mask.flatten()["choice"].shape == (len(DODGE_ACTIONS),)


def test_observation_spec_declares_one_feature_per_name() -> None:
    spec = dodge_environment_spec("test.env", max_steps=8)
    leaf = spec.observation.flatten()["features"]
    assert leaf.shape == (len(DODGE_FEATURES),)
    assert leaf.dtype == np.float32


def test_page_spec_requires_a_url() -> None:
    with pytest.raises(ValueError, match="url cannot be empty"):
        dodge_page_spec("")


# --- observation encoding ---------------------------------------------------


def test_normalize_features_clips_out_of_range_values() -> None:
    result = normalize_features([4.0, -4.0, 0.5], expected=3)
    assert result.tolist() == [1.0, -1.0, 0.5]
    assert result.dtype == np.float32


def test_normalize_features_rejects_a_wrong_width() -> None:
    with pytest.raises(ContractViolation, match="expected 4"):
        normalize_features([0.1, 0.2], expected=4)


def test_normalize_features_rejects_non_finite_values() -> None:
    with pytest.raises(ContractViolation, match="finite"):
        normalize_features([float("nan"), 0.0, 0.0, 0.0], expected=4)


# --- reward shaping ---------------------------------------------------------


def test_crash_returns_the_penalty_and_nothing_else() -> None:
    reward = dodge_reward(
        survived=False,
        alive_reward=0.01,
        dodge_bonus=0.05,
        crash_penalty=-1.0,
        threat_dx=0.0,
        threat_dy=0.0,
    )
    assert reward == -1.0


def test_a_clear_dodge_earns_more_than_a_held_position() -> None:
    kwargs = {
        "survived": True,
        "alive_reward": 0.01,
        "dodge_bonus": 0.05,
        "crash_penalty": -1.0,
        "threat_dy": 0.0,
    }
    clear = dodge_reward(threat_dx=1.0, **kwargs)
    centered = dodge_reward(threat_dx=0.0, **kwargs)
    assert clear > centered
    # The bonus is capped, so a surviving step can never beat the crash penalty.
    assert clear <= 0.01 + 0.05 + 1e-9


def test_a_distant_threat_earns_no_dodge_bonus() -> None:
    reward = dodge_reward(
        survived=True,
        alive_reward=0.01,
        dodge_bonus=0.05,
        crash_penalty=-1.0,
        threat_dx=1.0,
        threat_dy=1.0,
    )
    assert reward == pytest.approx(0.01)


# --- instrumented adapter ---------------------------------------------------


def test_reset_reseeds_the_page_and_returns_step_zero() -> None:
    environment = _make_environment()
    timestep = environment.reset(seed=11)
    assert environment.spec.environment_id == "web.dodge-instrumented-v1"
    assert timestep.step_id == 0
    assert not timestep.done
    assert isinstance(environment, InstrumentedWebGameEnvironment)


def test_step_reaches_the_page_and_advances_the_contract() -> None:
    environment = _make_environment(crash_at_step=99)
    environment.reset(seed=0)
    timestep = environment.step({"choice": np.array([2], dtype=np.int64)})
    assert timestep.step_id == 1
    assert timestep.observation["features"].shape == (len(DODGE_FEATURES),)
    assert timestep.info["web_steps"] == 1


def test_frames_per_step_advances_the_page_repeatedly() -> None:
    environment = _make_environment(crash_at_step=99, frames_per_step=3)
    environment.reset(seed=0)
    timestep = environment.step({"choice": np.array([1], dtype=np.int64)})
    # One contract step, three page ticks, so the page is the faster clock.
    assert timestep.step_id == 1
    assert timestep.info["web_steps"] == 3


def test_a_crash_terminates_and_is_never_also_a_truncation() -> None:
    environment = _make_environment(crash_at_step=2, max_steps=50)
    timestep = environment.reset(seed=0)
    while not timestep.done:
        timestep = environment.step({"choice": np.array([1], dtype=np.int64)})
        assert not (timestep.terminated[0] and timestep.truncated[0])
    assert timestep.terminated[0]
    assert not timestep.truncated[0]


def test_exhausting_the_step_budget_truncates_without_a_termination() -> None:
    environment = _make_environment(crash_at_step=999, max_steps=3)
    timestep = environment.reset(seed=0)
    for _ in range(3):
        timestep = environment.step({"choice": np.array([1], dtype=np.int64)})
    assert timestep.truncated[0]
    assert not timestep.terminated[0]


def test_each_reset_uses_a_different_seed_when_none_is_given() -> None:
    page = _FakePage()
    environment = InstrumentedWebGameEnvironment(ScriptedBrowserBridge(page.handle))
    environment.reset()
    environment.reset()
    assert len(page.reset_calls) == 2
    assert page.reset_calls[0] != page.reset_calls[1]


def test_a_supplied_seed_is_passed_through_verbatim() -> None:
    page = _FakePage()
    environment = InstrumentedWebGameEnvironment(ScriptedBrowserBridge(page.handle))
    environment.reset(seed=4242)
    assert page.reset_calls == [4242]


def test_an_out_of_range_action_is_refused() -> None:
    environment = _make_environment()
    environment.reset(seed=0)
    with pytest.raises(ContractViolation, match="outside the declared action space"):
        environment.step({"choice": np.array([7], dtype=np.int64)})


def test_a_nested_action_leaf_is_refused() -> None:
    environment = _make_environment()
    environment.reset(seed=0)
    with pytest.raises(ContractViolation, match="tensor leaf"):
        environment.step({"choice": {"nested": np.array([0], dtype=np.int64)}})


def test_a_malformed_page_state_fails_closed() -> None:
    environment = InstrumentedWebGameEnvironment(
        ScriptedBrowserBridge(lambda _expression: "not-an-object")
    )
    with pytest.raises(ContractViolation, match="must be a JSON object"):
        environment.reset(seed=0)


def test_a_wrong_feature_count_fails_closed() -> None:
    state = {"features": {"player_x": 0.0}, "alive": True, "score": 0.0, "steps": 0}
    environment = InstrumentedWebGameEnvironment(ScriptedBrowserBridge(lambda _e: state))
    with pytest.raises(ContractViolation, match="expected 4"):
        environment.reset(seed=0)


def test_a_non_boolean_alive_flag_fails_closed() -> None:
    state = {
        "features": dict.fromkeys(DODGE_FEATURES, 0.0),
        "alive": "yes",
        "score": 0.0,
        "steps": 0,
    }
    environment = InstrumentedWebGameEnvironment(ScriptedBrowserBridge(lambda _e: state))
    with pytest.raises(ContractViolation, match=r"state.alive must be a boolean"):
        environment.reset(seed=0)


def test_contract_environment_validates_every_transition() -> None:
    environment = ContractEnvironment(_make_environment(crash_at_step=99))
    timestep = environment.reset(seed=0)
    assert timestep.action_mask is not None
    for _ in range(4):
        timestep = environment.step({"choice": np.array([0], dtype=np.int64)})
    assert timestep.step_id == 4


def test_contract_environment_refuses_a_step_before_reset() -> None:
    environment = ContractEnvironment(_make_environment())
    with pytest.raises(ContractViolation, match="step requires reset first"):
        environment.step({"choice": np.array([0], dtype=np.int64)})


def test_a_closed_bridge_is_reported_not_swallowed() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: _ALIVE_STATE)
    environment = InstrumentedWebGameEnvironment(bridge)
    environment.reset(seed=0)
    bridge.close()
    with pytest.raises(ContractViolation, match="bridge is closed"):
        environment.step({"choice": np.array([0], dtype=np.int64)})


def test_closing_the_environment_leaves_a_borrowed_bridge_open() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: _ALIVE_STATE)
    environment = InstrumentedWebGameEnvironment(bridge)
    environment.close()
    assert not bridge.closed


def test_closing_the_environment_closes_an_owned_bridge() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: _ALIVE_STATE)
    environment = InstrumentedWebGameEnvironment(bridge, close_bridge=True)
    environment.close()
    assert bridge.closed


# --- black-box adapter ------------------------------------------------------


def test_black_box_adapter_presses_the_declared_keys() -> None:
    pressed: list[str] = []
    state = {
        "features": {"player_x": 0.0, "threat_dx": 0.1, "threat_dy": 0.2, "neighbor_dx": 0.3},
        "alive": True,
        "score": 1.0,
        "steps": 1,
    }
    bridge = ScriptedBrowserBridge(lambda _expression: state, on_press=pressed.append)
    environment = BlackBoxWebGameEnvironment(bridge, state_expression="window.score")
    environment.reset()
    environment.step({"choice": np.array([0], dtype=np.int64)})
    environment.step({"choice": np.array([2], dtype=np.int64)})
    assert pressed == ["ArrowLeft", "ArrowRight"]


def test_black_box_adapter_accepts_a_list_of_features() -> None:
    state = {"features": [0.1, 0.2, 0.3, 0.4], "alive": True, "score": 2.0, "steps": 2}
    environment = BlackBoxWebGameEnvironment(
        ScriptedBrowserBridge(lambda _expression: state),
        state_expression="window.score",
    )
    timestep = environment.reset()
    assert timestep.observation["features"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_black_box_adapter_presses_a_reset_key_when_configured() -> None:
    pressed: list[str] = []
    state = {"features": [0.0, 0.0, 0.0, 0.0], "alive": True, "score": 0.0, "steps": 0}
    bridge = ScriptedBrowserBridge(lambda _expression: state, on_press=pressed.append)
    environment = BlackBoxWebGameEnvironment(
        bridge, state_expression="window.score", reset_key="KeyR"
    )
    environment.reset()
    assert pressed == ["KeyR"]


def test_black_box_adapter_rejects_an_empty_key_set() -> None:
    with pytest.raises(ValueError, match="action_keys cannot be empty"):
        BlackBoxWebGameEnvironment(
            ScriptedBrowserBridge(lambda _e: {}),
            state_expression="window.score",
            action_keys=(),
        )


def test_black_box_adapter_rejects_an_empty_state_expression() -> None:
    with pytest.raises(ValueError, match="state_expression cannot be empty"):
        BlackBoxWebGameEnvironment(ScriptedBrowserBridge(lambda _e: {}), state_expression="")


def test_black_box_adapter_rejects_scalar_features() -> None:
    state = {"features": 1.5, "alive": True, "score": 0.0, "steps": 0}
    environment = BlackBoxWebGameEnvironment(
        ScriptedBrowserBridge(lambda _expression: state), state_expression="window.score"
    )
    with pytest.raises(ContractViolation, match="features must be a list or an object"):
        environment.reset()


def test_black_box_adapter_reports_its_own_environment_id() -> None:
    state = {"features": [0.0, 0.0, 0.0, 0.0], "alive": True, "score": 0.0, "steps": 0}
    environment = BlackBoxWebGameEnvironment(
        ScriptedBrowserBridge(lambda _expression: state),
        state_expression="window.score",
        environment_id="web.external-v1",
    )
    assert environment.spec.environment_id == "web.external-v1"


# --- constructors -----------------------------------------------------------


def test_adapters_refuse_a_non_bridge_transport() -> None:
    with pytest.raises(TypeError, match="BrowserBridge protocol"):
        InstrumentedWebGameEnvironment(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BrowserBridge protocol"):
        BlackBoxWebGameEnvironment(object(), state_expression="x")  # type: ignore[arg-type]


def test_instrumented_adapter_rejects_an_empty_hook() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: _ALIVE_STATE)
    with pytest.raises(ValueError, match="hook cannot be empty"):
        InstrumentedWebGameEnvironment(bridge, hook="")


def test_adapters_reject_a_non_positive_step_budget() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: _ALIVE_STATE)
    with pytest.raises(ValueError, match="max_steps"):
        InstrumentedWebGameEnvironment(bridge, max_steps=0)
    with pytest.raises(ValueError, match="max_steps"):
        BlackBoxWebGameEnvironment(bridge, state_expression="x", max_steps=0)


# --- cross-path parity ------------------------------------------------------
# The two adapters decode the same state payload. They must not drift: a page
# author who follows the documented hook and emits `features` as an array must
# get the same behaviour through either adapter, and a malformed payload must
# raise the same typed error from both.


def _instrumented_for_payload(state: Any) -> InstrumentedWebGameEnvironment:
    return InstrumentedWebGameEnvironment(ScriptedBrowserBridge(lambda _expression: state))


def _blackbox_for_payload(state: Any) -> BlackBoxWebGameEnvironment:
    return BlackBoxWebGameEnvironment(
        ScriptedBrowserBridge(lambda _expression: state), state_expression="window.state"
    )


@pytest.mark.parametrize(
    "features",
    [
        pytest.param([0.1, 0.2, 0.3, 0.4], id="array"),
        pytest.param((0.1, 0.2, 0.3, 0.4), id="tuple"),
        pytest.param(
            {"player_x": 0.1, "threat_dx": 0.2, "threat_dy": 0.3, "neighbor_dx": 0.4},
            id="object",
        ),
    ],
)
def test_both_paths_decode_the_same_payload_identically(features: Any) -> None:
    state = {"features": features, "alive": True, "score": 5.0, "steps": 5}
    instrumented = _instrumented_for_payload(state).reset(seed=0)
    blackbox = _blackbox_for_payload(state).reset()
    assert instrumented.observation["features"].tolist() == pytest.approx(
        blackbox.observation["features"].tolist()
    )
    assert float(instrumented.reward[0]) == pytest.approx(float(blackbox.reward[0]))


def test_both_paths_reject_a_scalar_features_payload_alike() -> None:
    state = {"features": 1.5, "alive": True, "score": 0.0, "steps": 0}
    with pytest.raises(ContractViolation, match="features must be a list or an object"):
        _instrumented_for_payload(state).reset(seed=0)
    with pytest.raises(ContractViolation, match="features must be a list or an object"):
        _blackbox_for_payload(state).reset()


def test_both_paths_reject_a_non_numeric_feature_alike() -> None:
    state = {"features": [0.1, "high", 0.3, 0.4], "alive": True, "score": 0.0, "steps": 0}
    for environment in (
        _instrumented_for_payload(state),
        _blackbox_for_payload(state),
    ):
        with pytest.raises(ContractViolation, match="must be a number"):
            environment.reset(seed=0)


def test_both_paths_reject_a_wrong_feature_count_alike() -> None:
    state = {"features": [0.1, 0.2], "alive": True, "score": 0.0, "steps": 0}
    for environment in (
        _instrumented_for_payload(state),
        _blackbox_for_payload(state),
    ):
        with pytest.raises(ContractViolation, match="expected 4"):
            environment.reset(seed=0)


def test_both_paths_reject_a_boolean_feature_alike() -> None:
    # A bool is an int to Python but is never a feature value a page meant.
    state = {"features": [True, 0.2, 0.3, 0.4], "alive": True, "score": 0.0, "steps": 0}
    for environment in (
        _instrumented_for_payload(state),
        _blackbox_for_payload(state),
    ):
        with pytest.raises(ContractViolation, match="must be a number"):
            environment.reset(seed=0)


def test_a_timestep_carries_the_score_through_info() -> None:
    environment = _make_environment(crash_at_step=99)
    environment.reset(seed=0)
    timestep = environment.step({"choice": np.array([1], dtype=np.int64)})
    assert timestep.info["score"] == 1.0
    assert isinstance(timestep, TimeStep)
