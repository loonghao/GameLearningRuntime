"""Tests for the browser transport port and the bundled asset server.

No browser is launched here: the bridge contract, the scripted double, and the
loopback server are all testable in-process.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.web_game import (
    BrowserBridge,
    LocalPageServer,
    PageSpec,
    PlaywrightBrowserBridge,
    ScriptedBrowserBridge,
    bundled_assets_dir,
    local_page_url,
)
from game_learning_runtime.web_game.bridge import (
    dumps,
    require_bool,
    require_mapping,
    require_number,
)


def test_a_scripted_bridge_satisfies_the_protocol() -> None:
    bridge = ScriptedBrowserBridge(lambda expression: len(expression))
    assert isinstance(bridge, BrowserBridge)


def test_a_scripted_bridge_records_key_presses() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: None)
    bridge.press("ArrowLeft")
    bridge.press("ArrowRight")
    assert bridge.pressed == ("ArrowLeft", "ArrowRight")


def test_a_scripted_bridge_rejects_an_empty_expression() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: None)
    with pytest.raises(ValueError, match="expression cannot be empty"):
        bridge.evaluate("")


def test_a_scripted_bridge_rejects_an_empty_key() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: None)
    with pytest.raises(ValueError, match="key cannot be empty"):
        bridge.press("")


def test_a_closed_scripted_bridge_refuses_work() -> None:
    bridge = ScriptedBrowserBridge(lambda _expression: None)
    bridge.close()
    with pytest.raises(ContractViolation, match="bridge is closed"):
        bridge.evaluate("1")
    with pytest.raises(ContractViolation, match="bridge is closed"):
        bridge.press("a")


def test_a_scripted_bridge_requires_a_callable_handler() -> None:
    with pytest.raises(TypeError, match="handler must be callable"):
        ScriptedBrowserBridge(object())  # type: ignore[arg-type]


def test_a_scripted_bridge_requires_a_callable_press_hook() -> None:
    with pytest.raises(TypeError, match="on_press must be callable"):
        ScriptedBrowserBridge(lambda _e: None, on_press=object())  # type: ignore[arg-type]


# --- page spec --------------------------------------------------------------


def test_a_page_spec_defaults_to_an_immediately_ready_page() -> None:
    spec = PageSpec(url="http://127.0.0.1/game.html")
    assert spec.ready_expression == "true"
    assert spec.reset_expression is None


def test_a_page_spec_rejects_a_blank_ready_expression() -> None:
    with pytest.raises(ValueError, match="ready_expression cannot be empty"):
        PageSpec(url="http://127.0.0.1/game.html", ready_expression="")


def test_a_page_spec_rejects_a_blank_reset_expression() -> None:
    with pytest.raises(ValueError, match="reset_expression"):
        PageSpec(url="http://127.0.0.1/game.html", reset_expression="")


def test_a_page_spec_rejects_a_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="ready_timeout_ms"):
        PageSpec(url="http://127.0.0.1/game.html", ready_timeout_ms=0)


# --- helpers ----------------------------------------------------------------


def test_dumps_produces_a_javascript_literal() -> None:
    assert dumps({"a": 1, "b": [True, None]}) == '{"a":1,"b":[true,null]}'


def test_local_page_url_requires_an_existing_file(tmp_path: object) -> None:
    missing = f"{tmp_path}/nope.html"
    with pytest.raises(ContractViolation, match="does not exist"):
        local_page_url(missing)


def test_local_page_url_references_the_bundled_page() -> None:
    url = local_page_url(bundled_assets_dir() / "orbital_dodge.html")
    assert url.startswith("file:///")
    assert url.endswith("orbital_dodge.html")


def test_bundled_assets_dir_contains_the_demonstration_page() -> None:
    assert (bundled_assets_dir() / "orbital_dodge.html").is_file()
    assert (bundled_assets_dir() / "three.module.min.js").is_file()
    assert (bundled_assets_dir() / "three.core.min.js").is_file()


# --- loopback server --------------------------------------------------------


def test_the_server_serves_a_bundled_page() -> None:
    import urllib.request

    with (
        LocalPageServer(bundled_assets_dir()) as server,
        urllib.request.urlopen(server.url_for("orbital_dodge.html")) as response,
    ):
        assert server.base_url.startswith("http://127.0.0.1:")
        assert response.status == 200
        assert b"orbital" in response.read().lower()


def test_the_server_rejects_a_page_outside_its_directory() -> None:
    with (
        LocalPageServer(bundled_assets_dir()) as server,
        pytest.raises(ContractViolation, match="does not exist"),
    ):
        server.url_for("../pyproject.toml")


def test_the_server_rejects_a_missing_directory(tmp_path: object) -> None:
    with pytest.raises(ContractViolation, match="does not exist"):
        LocalPageServer(f"{tmp_path}/absent")


def test_the_server_refuses_a_non_loopback_bind() -> None:
    with pytest.raises(ValueError, match="loopback"):
        LocalPageServer(bundled_assets_dir(), host="0.0.0.0")


def test_the_server_reports_its_port_only_once_started() -> None:
    server = LocalPageServer(bundled_assets_dir())
    with pytest.raises(ContractViolation, match="not started"):
        _ = server.port
    server.start()
    try:
        assert server.port > 0
        assert server.base_url.endswith("/")
    finally:
        server.stop()


def test_the_server_is_idempotent_under_repeated_starts() -> None:
    with LocalPageServer(bundled_assets_dir()) as server:
        first = server.base_url
        assert server.start() == first


def test_url_for_rejects_an_empty_name() -> None:
    with (
        LocalPageServer(bundled_assets_dir()) as server,
        pytest.raises(ValueError, match="name cannot be empty"),
    ):
        server.url_for("")


def test_the_page_declares_the_expected_hook_contract() -> None:
    source = (bundled_assets_dir() / "orbital_dodge.html").read_text(encoding="utf-8")
    for fragment in ("__glr", "reset(", "step(", "state()"):
        assert fragment in source
    # The simulation must stay decoupled from the render loop, or a headless run
    # and a visible run would not produce the same trajectory.
    assert "stepWorld(world, action)" in source


def test_the_page_bundles_three_js_under_the_mit_license() -> None:
    header = (bundled_assets_dir() / "three.module.min.js").read_text(encoding="utf-8")[:200]
    assert "MIT" in header
    assert json.dumps(header)  # the header is decodable text, not binary
    assert isinstance(header, str)


def test_require_mapping_rejects_a_non_object() -> None:
    with pytest.raises(ContractViolation, match="must be a JSON object"):
        require_mapping([1, 2, 3], path="state")


def test_require_mapping_accepts_an_object() -> None:
    assert require_mapping({"a": 1}, path="state") == {"a": 1}


def test_require_number_rejects_a_string_and_a_bool() -> None:
    with pytest.raises(ContractViolation, match="must be a number"):
        require_number("1.0", path="score")
    # A bool is an int in Python, but a page returning `true` for a score is a
    # page contract bug, not a score of one.
    with pytest.raises(ContractViolation, match="must be a number"):
        require_number(True, path="score")


def test_require_number_accepts_ints_and_floats() -> None:
    assert require_number(3, path="score") == 3.0
    assert require_number(0.5, path="score") == 0.5


def test_require_bool_rejects_a_truthy_number() -> None:
    with pytest.raises(ContractViolation, match="must be a boolean"):
        require_bool(1, path="alive")


def test_require_bool_accepts_a_real_bool() -> None:
    assert require_bool(False, path="alive") is False


def test_the_playwright_bridge_rejects_a_non_page_spec() -> None:
    with pytest.raises(TypeError, match="page_spec must be a PageSpec"):
        PlaywrightBrowserBridge(object())  # type: ignore[arg-type]


def test_the_playwright_bridge_rejects_a_bad_viewport() -> None:
    spec = PageSpec(url="http://127.0.0.1/game.html")
    with pytest.raises(ValueError, match="viewport"):
        PlaywrightBrowserBridge(spec, viewport=(0, 100))


def test_the_playwright_bridge_reports_a_missing_dependency(monkeypatch: Any) -> None:
    import game_learning_runtime.web_game.bridge as bridge_module

    def missing() -> Any:
        raise ImportError("no playwright here")

    monkeypatch.setattr(bridge_module, "_import_playwright", missing)
    spec = PageSpec(url="http://127.0.0.1/game.html")
    with pytest.raises(ContractViolation, match="optional playwright dependency"):
        bridge_module.PlaywrightBrowserBridge(spec)


class _FakePlaywright:
    """Minimal Playwright stand-in that owns no browser and no chromium attr."""

    def __init__(self) -> None:
        self.closed = False

    def __enter__(self) -> _FakePlaywright:
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True


def test_the_playwright_bridge_rejects_an_unknown_browser(monkeypatch: Any) -> None:
    import game_learning_runtime.web_game.bridge as bridge_module

    monkeypatch.setattr(bridge_module, "_import_playwright", lambda: _FakePlaywright)
    spec = PageSpec(url="http://127.0.0.1/game.html")
    with pytest.raises(ContractViolation, match="unsupported browser"):
        bridge_module.PlaywrightBrowserBridge(spec, browser="netscape")


def test_the_playwright_bridge_releases_the_driver_when_startup_fails(
    monkeypatch: Any,
) -> None:
    import game_learning_runtime.web_game.bridge as bridge_module

    contexts: list[_FakePlaywright] = []

    def factory() -> _FakePlaywright:
        context = _FakePlaywright()
        contexts.append(context)
        return context

    monkeypatch.setattr(bridge_module, "_import_playwright", lambda: factory)
    spec = PageSpec(url="http://127.0.0.1/game.html")
    with pytest.raises(ContractViolation, match="unsupported browser"):
        bridge_module.PlaywrightBrowserBridge(spec, browser="netscape")
    # A failed start must not leak a driver process.
    assert contexts and contexts[0].closed


def test_a_scripted_bridge_can_replay_a_realistic_state() -> None:
    state: dict[str, Any] = {
        "features": {"player_x": 0.1, "threat_dx": -0.2, "threat_dy": 0.3, "neighbor_dx": 0.4},
        "alive": True,
        "score": 12.0,
        "steps": 12,
    }
    bridge = ScriptedBrowserBridge(lambda _expression: state)
    assert bridge.evaluate("window.__glr.state()") == state
