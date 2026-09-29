"""Contract environments for self-hosted browser and Three.js web games.

GLR reaches a web game the same way it reaches a desktop title: an adapter
implements :class:`~game_learning_runtime.environment.GameEnvironment` and is
wrapped by :class:`~game_learning_runtime.environment.ContractEnvironment`, so
observation, action, mask, reward, and episode boundaries are validated on every
transition. The only browser-specific part is the
:class:`~game_learning_runtime.web_game.bridge.BrowserBridge` transport, which
evaluates a script in one page and presses keys.

Two adapters ship here:

:class:`InstrumentedWebGameEnvironment`
    The page exposes a state hook (``window.__glr`` by default) that the adapter
    owns. This is the path for a game you wrote or were authorized to
    instrument: the adapter reads structured state and calls a structured step
    function, so no pixel decoding and no input synthesis is needed.

:class:`BlackBoxWebGameEnvironment`
    The page is unmodified. The adapter reads a caller-declared JavaScript
    expression and drives the game with real keyboard events. This is the path
    for an external game: it needs no cooperation from the page beyond a
    readable score and a keyboard binding, which is exactly the boundary the
    ADR records.

Both adapters refuse to drive a page that stops answering: a missing or
malformed state payload raises
:class:`~game_learning_runtime.errors.ContractViolation` rather than being
coerced into a default observation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.specs import (
    CompositeSpec,
    EnvironmentSpec,
    SpaceKind,
    TensorSpec,
)
from game_learning_runtime.termination import EpisodeCaps
from game_learning_runtime.web_game.bridge import (
    BrowserBridge,
    PageSpec,
    require_bool,
    require_mapping,
    require_number,
)

#: Action names shared by both adapters, in contract order.
DODGE_ACTIONS: tuple[str, ...] = ("left", "hold", "right")

#: Keyboard binding used for the black-box path.
DODGE_KEYS: tuple[str, ...] = ("ArrowLeft", "Space", "ArrowRight")

#: Observation feature names, in contract order.
DODGE_FEATURES: tuple[str, ...] = ("player_x", "threat_dx", "threat_dy", "neighbor_dx")

#: Normalized horizontal offset at which a threat counts as fully cleared.
#:
#: The bundled page collides when the horizontal offset drops below roughly 0.8
#: world units out of an 8-unit-wide world, i.e. ~0.1 normalized. Clearing at
#: 0.25 would make the term saturate almost every step and shape nothing, so
#: the constant tracks the actual collision radius rather than a round number.
DODGE_CLEARANCE: float = 0.15

#: Normalized vertical band in which a threat counts as imminent.
DODGE_IMMINENT_BAND: float = 0.2


def dodge_observation_spec(feature_count: int = len(DODGE_FEATURES)) -> CompositeSpec:
    """Return the observation contract for the dodge task."""

    return CompositeSpec(
        {
            "features": TensorSpec(
                (feature_count,),
                np.float32,
                minimum=-1.0,
                maximum=1.0,
                description=("normalized player_x, threat_dx, threat_dy, neighbor_dx in [-1, 1]"),
            )
        }
    )


def dodge_action_spec() -> CompositeSpec:
    """Return the three-way discrete action contract for the dodge task."""

    return CompositeSpec(
        {
            "choice": TensorSpec(
                (1,),
                np.int64,
                kind=SpaceKind.DISCRETE,
                minimum=0,
                maximum=len(DODGE_ACTIONS) - 1,
                description="0=left, 1=hold, 2=right",
            )
        }
    )


def dodge_action_mask_spec() -> CompositeSpec:
    """Return the action-mask contract for the dodge task."""

    return CompositeSpec(
        {"choice": TensorSpec((len(DODGE_ACTIONS),), np.bool_, kind=SpaceKind.BINARY)}
    )


def dodge_environment_spec(
    environment_id: str,
    *,
    max_steps: int,
    feature_count: int = len(DODGE_FEATURES),
    capabilities: frozenset[str] = frozenset({"action-mask", "deterministic-reset"}),
) -> EnvironmentSpec:
    """Return the full contract for one dodge-task adapter instance."""

    return EnvironmentSpec(
        environment_id=environment_id,
        observation=dodge_observation_spec(feature_count),
        action=dodge_action_spec(),
        action_mask=dodge_action_mask_spec(),
        capabilities=capabilities,
        episode_caps=EpisodeCaps(max_steps=max_steps, death_cap=1),
    )


def normalize_features(features: Any, *, expected: int) -> NDArray[Any]:
    """Validate and clip one observation feature vector."""

    values = np.asarray(features, dtype=np.float64).reshape(-1)
    if values.shape[0] != expected:
        raise ContractViolation(f"observation has {values.shape[0]} features; expected {expected}")
    if not np.all(np.isfinite(values)):
        raise ContractViolation("observation features must be finite")
    return np.clip(values, -1.0, 1.0).astype(np.float32)


def dodge_reward(
    *,
    survived: bool,
    alive_reward: float,
    dodge_bonus: float,
    crash_penalty: float,
    threat_dx: float,
    threat_dy: float,
    clearance: float = DODGE_CLEARANCE,
    imminent_band: float = DODGE_IMMINENT_BAND,
) -> float:
    """Shape one dodge reward.

    Staying alive pays a small constant and crashing pays a penalty, so episode
    length dominates the return. The bonus pays for a *good* dodge: it is the
    product of how imminent the nearest threat is and how much horizontal
    clearance the player kept. A player that never moves collects the alive
    reward but no bonus and dies early; a player that dodges into a second
    threat loses more than the bonus can pay.

    The shaping depends only on the two threat features of the current
    observation, so it cannot be farmed by stalling: a stalled player faces an
    imminent threat with no clearance and forfeits the bonus before it dies.
    """

    if not survived:
        return crash_penalty
    imminence = max(0.0, 1.0 - abs(threat_dy) / imminent_band)
    horizontal = min(1.0, abs(threat_dx) / clearance)
    return alive_reward + dodge_bonus * imminence * horizontal


@dataclass(frozen=True, slots=True)
class InstrumentedStepResult:
    """One decoded step from an instrumented page."""

    features: tuple[float, ...]
    alive: bool
    score: float
    steps: int


class _DodgeEnvironmentBase(GameEnvironment):
    """Shared dodge-task behaviour: state decoding, reward, episode boundary.

    Subclasses own exactly one thing: how a step reaches the page and how a
    fresh episode starts. Everything that makes the result a GLR contract --
    validation, reward shaping, termination attribution -- lives here once.
    """

    _action_count: int

    def __init__(
        self,
        bridge: BrowserBridge,
        *,
        environment_id: str,
        max_steps: int,
        feature_count: int,
        alive_reward: float,
        dodge_bonus: float,
        crash_penalty: float,
        close_bridge: bool,
        capabilities: frozenset[str] = frozenset({"action-mask", "deterministic-reset"}),
    ) -> None:
        if not isinstance(bridge, BrowserBridge):
            raise TypeError("bridge must implement the BrowserBridge protocol")
        if max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if feature_count < 1:
            raise ValueError("feature_count must be a positive integer")
        self._bridge = bridge
        self._max_steps = max_steps
        self._feature_count = feature_count
        self._alive_reward = alive_reward
        self._dodge_bonus = dodge_bonus
        self._crash_penalty = crash_penalty
        self._close_bridge = close_bridge
        self._episode_id = uuid4()
        self._step_id = 0
        self._last: InstrumentedStepResult | None = None
        self._spec = dodge_environment_spec(
            environment_id,
            max_steps=max_steps,
            feature_count=feature_count,
            capabilities=capabilities,
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        self._start_episode(seed=seed, options=options)
        self._episode_id = uuid4()
        self._step_id = 0
        self._last = self._read_state()
        return self._timestep()

    def step(self, action: TensorTree) -> TimeStep:
        self._apply_action(self._decode_choice(action))
        self._step_id += 1
        self._last = self._read_state()
        return self._timestep()

    def close(self) -> None:
        if self._close_bridge:
            self._bridge.close()

    def _start_episode(self, *, seed: int | None, options: Mapping[str, Any] | None) -> None:
        raise NotImplementedError

    def _apply_action(self, choice: int) -> None:
        raise NotImplementedError

    def _read_state(self) -> InstrumentedStepResult:
        raise NotImplementedError

    def _decode_choice(self, action: TensorTree) -> int:
        choice = action["choice"]
        if isinstance(choice, Mapping):
            raise ContractViolation("choice must be a tensor leaf")
        index = int(np.asarray(choice).reshape(-1)[0])
        if not 0 <= index < self._action_count:
            raise ContractViolation(f"choice {index} is outside the declared action space")
        return index

    def _decode_features(self, payload: Mapping[str, Any]) -> tuple[float, ...]:
        """Decode one observation payload into a feature tuple.

        This lives on the shared base on purpose. Both adapters accept the same
        state payload, so they must accept the same feature encodings and raise
        the same error for the same malformed input. Decoding it twice is how
        the two paths drift: a downstream integrator who wrote the documented
        array form would have been accepted by one adapter and rejected by the
        other, with a bare ``ValueError`` instead of a typed refusal.

        Both a JSON array and a JSON object are accepted, because both are
        natural for a page to emit and neither is ambiguous. Values are checked
        with :func:`require_number` so a non-numeric entry is a
        ``ContractViolation`` rather than a ``TypeError`` from deep in NumPy.
        """

        features = payload.get("features")
        if isinstance(features, Mapping):
            values = tuple(
                require_number(value, path=f"state.features.{key}")
                for key, value in features.items()
            )
        elif isinstance(features, (list, tuple)):
            values = tuple(
                require_number(value, path=f"state.features[{index}]")
                for index, value in enumerate(features)
            )
        else:
            raise ContractViolation("state.features must be a list or an object")
        if len(values) != self._feature_count:
            raise ContractViolation(
                f"state.features has {len(values)} entries; expected {self._feature_count}"
            )
        return values

    def _timestep(self) -> TimeStep:
        state = self._last
        if state is None:  # pragma: no cover - reset always sets it first
            raise ContractViolation("step requires reset first")
        # Terminated and truncated are disjoint by contract, but the underlying
        # facts are not: a player can crash on the very step that exhausts the
        # budget, and the page does report `alive: false`. Both are true, so the
        # adapter resolves them by precedence rather than by pretending one did
        # not happen.
        #
        # Truncation wins, because the budget cap is what stopped the episode
        # and it is the signal the learner acts on: a truncated transition keeps
        # the bootstrap value in the advantage recursion, while a terminated one
        # does not. Reporting a termination here would tell the learner the
        # trajectory ended in a real crash and suppress the bootstrap, when in
        # fact the harness simply stopped asking. Either flag alone would be
        # defensible; what matters is that the choice is one rule, applied
        # identically on every step.
        #
        # The budget counts adapter steps, not page ticks: a page advanced by
        # `frames_per_step` ticks per action would otherwise burn its budget in
        # a fraction of the declared episode length.
        truncated = self._step_id >= self._max_steps
        terminated = not state.alive and not truncated
        threat_dx = state.features[1] if self._feature_count >= 2 else 0.0
        threat_dy = state.features[2] if self._feature_count >= 3 else 0.0
        return TimeStep(
            observation={
                "features": normalize_features(state.features, expected=self._feature_count)
            },
            reward=np.array(
                [
                    dodge_reward(
                        survived=not terminated,
                        alive_reward=self._alive_reward,
                        dodge_bonus=self._dodge_bonus,
                        crash_penalty=self._crash_penalty,
                        threat_dx=float(threat_dx),
                        threat_dy=float(threat_dy),
                    )
                ],
                dtype=np.float32,
            ),
            terminated=np.array([terminated], dtype=np.bool_),
            truncated=np.array([truncated], dtype=np.bool_),
            action_mask={"choice": np.ones(self._action_count, dtype=np.bool_)},
            episode_id=self._episode_id,
            step_id=self._step_id,
            info={
                "score": state.score,
                "alive": state.alive,
                "web_steps": state.steps,
                "termination_reason": "failed" if not state.alive else "",
            },
        )


class InstrumentedWebGameEnvironment(_DodgeEnvironmentBase):
    """Drive a page that exposes a ``window.__glr`` state hook.

    The hook is a small contract the page owner implements:

    ```js
    window.__glr = {
      ready: true,
      reset(seed) { /* start a fresh episode */ },
      // `features` may be an array or an object; both are decoded the same way.
      state() {
        return {
          features: [0.1, -0.2, 0.3, 0.4],   // or {player_x: 0.1, ...}
          alive: true, score: 12, steps: 12
        };
      },
      step(actionIndex) { /* apply the action and advance the world */ }
    };
    ```

    `alive`, `score`, and `steps` are optional and default to `true`, `0`, and
    `0`; `features` is required and must hold exactly `feature_count` numbers.

    Because the adapter owns the hook, the observation is structured state
    rather than pixels, which is what makes a short training run learn.
    """

    def __init__(
        self,
        bridge: BrowserBridge,
        *,
        environment_id: str = "web.dodge-instrumented-v1",
        hook: str = "window.__glr",
        max_steps: int = 256,
        feature_count: int = len(DODGE_FEATURES),
        frames_per_step: int = 1,
        alive_reward: float = 0.01,
        dodge_bonus: float = 0.05,
        crash_penalty: float = -1.0,
        close_bridge: bool = False,
    ) -> None:
        if not hook:
            raise ValueError("hook cannot be empty")
        if frames_per_step < 1:
            raise ValueError("frames_per_step must be a positive integer")
        super().__init__(
            bridge,
            environment_id=environment_id,
            max_steps=max_steps,
            feature_count=feature_count,
            alive_reward=alive_reward,
            dodge_bonus=dodge_bonus,
            crash_penalty=crash_penalty,
            close_bridge=close_bridge,
        )
        self._hook = hook
        self._frames_per_step = frames_per_step
        self._action_count = len(DODGE_ACTIONS)

    def _start_episode(self, *, seed: int | None, options: Mapping[str, Any] | None) -> None:
        del options
        # Every reset reseeds the page, so a caller that passes no seed still
        # gets a different episode each time instead of replaying episode zero.
        # A caller that passes one gets the same episode back, which is what
        # makes a validation run reproducible.
        chosen = int(seed) if seed is not None else int(uuid4().int & 0x7FFFFFFF)
        self._bridge.evaluate(f"{self._hook}.reset({chosen})")

    def _apply_action(self, choice: int) -> None:
        for _ in range(self._frames_per_step):
            self._bridge.evaluate(f"{self._hook}.step({choice})")

    def _read_state(self) -> InstrumentedStepResult:
        payload = require_mapping(
            self._bridge.evaluate(f"{self._hook}.state()"),
            path=f"{self._hook}.state()",
        )
        return InstrumentedStepResult(
            features=self._decode_features(payload),
            alive=require_bool(payload.get("alive", True), path="state.alive"),
            score=require_number(payload.get("score", 0.0), path="state.score"),
            steps=int(require_number(payload.get("steps", 0), path="state.steps")),
        )


class BlackBoxWebGameEnvironment(_DodgeEnvironmentBase):
    """Drive an unmodified web page through keyboard events and a state probe.

    No page cooperation is required beyond a JavaScript expression that returns
    the observable state. Actions are real key presses, so the page cannot tell
    this adapter from a human player. The trade-off is fidelity: whatever the
    expression cannot see is invisible to the learner, so a canvas-only game
    without a readable score yields an observation that cannot support learning.
    """

    def __init__(
        self,
        bridge: BrowserBridge,
        *,
        state_expression: str,
        action_keys: tuple[str, ...] = DODGE_KEYS,
        environment_id: str = "web.dodge-blackbox-v1",
        reset_key: str | None = None,
        max_steps: int = 256,
        feature_count: int = len(DODGE_FEATURES),
        alive_reward: float = 0.01,
        dodge_bonus: float = 0.05,
        crash_penalty: float = -1.0,
        close_bridge: bool = False,
    ) -> None:
        if not state_expression:
            raise ValueError("state_expression cannot be empty")
        if not action_keys:
            raise ValueError("action_keys cannot be empty")
        if any(not key for key in action_keys):
            raise ValueError("action_keys cannot contain an empty key")
        super().__init__(
            bridge,
            environment_id=environment_id,
            max_steps=max_steps,
            feature_count=feature_count,
            alive_reward=alive_reward,
            dodge_bonus=dodge_bonus,
            crash_penalty=crash_penalty,
            close_bridge=close_bridge,
            capabilities=frozenset({"action-mask", "deterministic-reset"}),
        )
        self._state_expression = state_expression
        self._action_keys = tuple(action_keys)
        self._reset_key = reset_key
        self._action_count = len(self._action_keys)

    def _start_episode(self, *, seed: int | None, options: Mapping[str, Any] | None) -> None:
        del seed, options
        if self._reset_key is not None:
            self._bridge.press(self._reset_key)

    def _apply_action(self, choice: int) -> None:
        self._bridge.press(self._action_keys[choice])

    def _read_state(self) -> InstrumentedStepResult:
        payload = require_mapping(
            self._bridge.evaluate(self._state_expression),
            path="state_expression",
        )
        return InstrumentedStepResult(
            features=self._decode_features(payload),
            alive=require_bool(payload.get("alive", True), path="state.alive"),
            score=require_number(payload.get("score", 0.0), path="state.score"),
            steps=int(require_number(payload.get("steps", 0), path="state.steps")),
        )


def dodge_page_spec(url: str, *, hook: str = "window.__glr") -> PageSpec:
    """Return the page contract for the bundled Three.js dodge page."""

    return PageSpec(
        url=url,
        ready_expression=f"{hook} && {hook}.ready === true",
        reset_expression=f"{hook}.reset(0)",
    )
