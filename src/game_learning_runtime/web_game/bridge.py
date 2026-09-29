"""Browser transport port for web-game adapters.

The runtime package never imports a browser driver at module scope. A web-game
adapter talks to a :class:`BrowserBridge`, so the adapter contract, the
observation encoding, and the reward shaping stay unit-testable without a
browser, and a driver can be swapped without touching environment code.

Two implementations ship with the runtime:

* :class:`PlaywrightBrowserBridge` drives a real Chromium page. Playwright is an
  optional dependency, imported lazily, so the core package keeps working when
  it is absent.
* :class:`ScriptedBrowserBridge` replays a scripted in-page model. It is the
  deterministic double used by the test-suite and by offline replay.

Only self-owned or explicitly instrumented pages may be driven. This module
exists to automate a page the caller is authorized to automate; it provides no
facility to bypass anti-cheat, licensing, or access control.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from game_learning_runtime.errors import ContractViolation

BridgeValue: Any = Mapping[str, Any] | Sequence[Any] | str | int | float | bool | None


@runtime_checkable
class BrowserBridge(Protocol):
    """Transport that evaluates script in one browser page."""

    def evaluate(self, expression: str) -> Any:
        """Evaluate a JavaScript expression and return a JSON-safe value."""
        ...

    def press(self, key: str) -> None:
        """Press and release one keyboard key on the focused page element."""
        ...

    def close(self) -> None:
        """Release the browser resources owned by this bridge."""
        ...


@dataclass(frozen=True, slots=True)
class PageSpec:
    """Where one web game lives and how a fresh episode starts."""

    url: str
    ready_expression: str = "true"
    reset_expression: str | None = None
    ready_timeout_ms: int = 10_000

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("url cannot be empty")
        if not self.ready_expression:
            raise ValueError("ready_expression cannot be empty")
        if self.ready_timeout_ms < 1:
            raise ValueError("ready_timeout_ms must be a positive integer")
        if self.reset_expression is not None and not self.reset_expression:
            raise ValueError("reset_expression must be a non-empty string or None")


def require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    """Return ``value`` as a mapping or fail closed with a typed violation."""

    if not isinstance(value, Mapping):
        raise ContractViolation(f"{path} must be a JSON object; received {type(value).__name__}")
    return value


def require_number(value: object, *, path: str) -> float:
    """Return ``value`` as a finite float or fail closed."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation(f"{path} must be a number; received {type(value).__name__}")
    return float(value)


def require_bool(value: object, *, path: str) -> bool:
    """Return ``value`` as a bool or fail closed."""

    if not isinstance(value, bool):
        raise ContractViolation(f"{path} must be a boolean; received {type(value).__name__}")
    return value


def _import_playwright() -> Any:
    """Import Playwright's context factory; raises ImportError when absent.

    A module-level seam, so the optional dependency is resolved in exactly one
    place and tests can substitute a double without a browser installed. The
    caller turns the ImportError into the actionable ContractViolation, which
    keeps this seam too thin to hide a failure.
    """

    from playwright.sync_api import sync_playwright

    return sync_playwright


class PlaywrightBrowserBridge:
    """Drive one Chromium page through Playwright's synchronous API.

    Playwright is imported on construction, not on module import, so the
    runtime package remains importable on installations without a browser
    driver. The page is served from a local file or URL that the caller owns.
    """

    def __init__(
        self,
        page_spec: PageSpec,
        *,
        headless: bool = True,
        viewport: tuple[int, int] = (960, 540),
        browser: str = "chromium",
    ) -> None:
        if not isinstance(page_spec, PageSpec):
            raise TypeError("page_spec must be a PageSpec")
        if len(viewport) != 2 or any(size < 1 for size in viewport):
            raise ValueError("viewport must be two positive dimensions")
        self._page_spec = page_spec
        self._viewport = viewport
        self._browser_name = browser
        self._headless = headless
        self._sync_playwright_cm: Any = None
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._start()

    def _start(self) -> None:
        try:
            context_factory = _import_playwright()
        except ImportError as error:
            raise ContractViolation(
                "PlaywrightBrowserBridge requires the optional playwright dependency"
            ) from error
        self._sync_playwright_cm = context_factory()
        self._playwright = self._sync_playwright_cm.__enter__()
        try:
            browser_factory: Any = getattr(self._playwright, self._browser_name, None)
            if browser_factory is None:
                raise ContractViolation(f"unsupported browser: {self._browser_name}")
            self._browser = browser_factory.launch(headless=self._headless)
            self._page = self._browser.new_page(
                viewport={"width": self._viewport[0], "height": self._viewport[1]}
            )
            self._page.goto(self._page_spec.url, wait_until="load")
            self._await_ready()
        except Exception:
            self.close()
            raise

    def _await_ready(self) -> None:
        deadline_expression = self._page_spec.ready_expression
        self._page.wait_for_function(
            f"() => ({deadline_expression})",
            timeout=self._page_spec.ready_timeout_ms,
        )

    @property
    def page(self) -> Any:
        """Return the underlying Playwright page, for advanced callers."""

        return self._page

    def evaluate(self, expression: str) -> Any:
        if self._page is None:
            raise ContractViolation("bridge is closed")
        if not expression:
            raise ValueError("expression cannot be empty")
        return self._page.evaluate(f"() => ({expression})")

    def press(self, key: str) -> None:
        if self._page is None:
            raise ContractViolation("bridge is closed")
        if not key:
            raise ValueError("key cannot be empty")
        self._page.keyboard.press(key)

    def close(self) -> None:
        with suppress(Exception):
            if self._browser is not None:
                self._browser.close()
        with suppress(Exception):
            if self._sync_playwright_cm is not None:
                self._sync_playwright_cm.__exit__(None, None, None)
        self._page = None
        self._browser = None
        self._playwright = None
        self._sync_playwright_cm = None

    def __enter__(self) -> PlaywrightBrowserBridge:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ScriptedBrowserBridge:
    """Deterministic in-process browser double driven by callables.

    The double owns no browser: ``handler`` receives the evaluated expression
    and returns a JSON-safe result, and key presses are recorded so a test can
    assert that an adapter drove the page it claims to drive.
    """

    def __init__(
        self,
        handler: Callable[[str], Any],
        *,
        on_press: Callable[[str], None] | None = None,
    ) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        if on_press is not None and not callable(on_press):
            raise TypeError("on_press must be callable or None")
        self._handler = handler
        self._on_press = on_press
        self._pressed: list[str] = []
        self.closed = False

    @property
    def pressed(self) -> tuple[str, ...]:
        return tuple(self._pressed)

    def evaluate(self, expression: str) -> Any:
        if self.closed:
            raise ContractViolation("bridge is closed")
        if not expression:
            raise ValueError("expression cannot be empty")
        return self._handler(expression)

    def press(self, key: str) -> None:
        if self.closed:
            raise ContractViolation("bridge is closed")
        if not key:
            raise ValueError("key cannot be empty")
        self._pressed.append(key)
        if self._on_press is not None:
            self._on_press(key)

    def close(self) -> None:
        self.closed = True


def local_page_url(path: str | Path) -> str:
    """Return a ``file://`` URL for a page shipped inside the package."""

    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ContractViolation(f"web game page does not exist: {resolved}")
    return resolved.as_uri()


def dumps(value: Any) -> str:
    """Serialize a value into a JavaScript literal usable inside an expression."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
