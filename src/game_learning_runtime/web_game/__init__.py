"""Browser and Three.js web-game adapters for GLR.

This subpackage answers one question: can the GLR contract drive a game that
lives in a browser tab instead of a desktop process. It ships the transport
port, two adapters, and the bundled demonstration page.

* :mod:`game_learning_runtime.web_game.bridge` -- the browser transport port and
  its Playwright and scripted implementations.
* :mod:`game_learning_runtime.web_game.environments` -- the
  :class:`~game_learning_runtime.environment.GameEnvironment` adapters.
* :mod:`game_learning_runtime.web_game.serving` -- the loopback static server a
  bundled page needs, because ES module imports are blocked from ``file://``.

Nothing here bypasses access control. Only pages the caller owns or is
authorized to automate may be driven; see ``docs/decisions/0048-*`` for the
recorded capability boundary.
"""

from __future__ import annotations

from game_learning_runtime.web_game.bridge import (
    BrowserBridge,
    PageSpec,
    PlaywrightBrowserBridge,
    ScriptedBrowserBridge,
    local_page_url,
)
from game_learning_runtime.web_game.environments import (
    DODGE_ACTIONS,
    DODGE_FEATURES,
    DODGE_KEYS,
    BlackBoxWebGameEnvironment,
    InstrumentedWebGameEnvironment,
    dodge_action_mask_spec,
    dodge_action_spec,
    dodge_environment_spec,
    dodge_observation_spec,
    dodge_page_spec,
    dodge_reward,
    normalize_features,
)
from game_learning_runtime.web_game.serving import (
    LocalPageServer,
    bundled_assets_dir,
)

__all__ = [
    "DODGE_ACTIONS",
    "DODGE_FEATURES",
    "DODGE_KEYS",
    "BlackBoxWebGameEnvironment",
    "BrowserBridge",
    "InstrumentedWebGameEnvironment",
    "LocalPageServer",
    "PageSpec",
    "PlaywrightBrowserBridge",
    "ScriptedBrowserBridge",
    "bundled_assets_dir",
    "dodge_action_mask_spec",
    "dodge_action_spec",
    "dodge_environment_spec",
    "dodge_observation_spec",
    "dodge_page_spec",
    "dodge_reward",
    "local_page_url",
    "normalize_features",
]
