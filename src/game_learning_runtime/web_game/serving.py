"""Serve a bundled web-game directory to a browser over loopback HTTP.

ES module imports and import maps are blocked from a ``file://`` origin, so a
Three.js page has to be served over HTTP even when every asset is local. This
module starts a short-lived, loopback-bound static server for one directory and
shuts it down deterministically.

The server binds ``127.0.0.1`` and refuses to follow paths outside the served
directory, so it exposes the bundled assets and nothing else.
"""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from game_learning_runtime.errors import ContractViolation


class _DirectoryHandler(SimpleHTTPRequestHandler):
    """Static handler pinned to one directory, with request logging off."""

    def __init__(self, *args: Any, directory: Path, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(directory), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request logging; collection loops are noisy enough."""


class LocalPageServer:
    """A loopback static server for one directory of bundled game assets."""

    def __init__(self, directory: str | Path, *, host: str = "127.0.0.1") -> None:
        root = Path(directory)
        if not root.is_dir():
            raise ContractViolation(f"web game directory does not exist: {root}")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("host must be a loopback address")
        self._directory = root
        self._host = host
        self._server: ThreadingHTTPServer | None = None

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def port(self) -> int:
        if self._server is None:
            raise ContractViolation("server is not started")
        return int(self._server.server_address[1])

    def start(self) -> str:
        """Bind and serve in a daemon thread; return the base URL."""

        if self._server is not None:
            return self.base_url
        handler = partial(_DirectoryHandler, directory=self._directory)
        self._server = ThreadingHTTPServer((self._host, 0), handler)
        self._server.daemon_threads = True
        import threading

        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self.base_url

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise ContractViolation("server is not started")
        return f"http://{self._host}:{self.port}/"

    def url_for(self, name: str) -> str:
        """Return the absolute URL of one page inside the served directory."""

        if not name:
            raise ValueError("name cannot be empty")
        candidate = (self._directory / name).resolve()
        if not candidate.is_file():
            raise ContractViolation(f"web game page does not exist: {candidate}")
        return f"{self.base_url}{name}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> LocalPageServer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def bundled_assets_dir() -> Path:
    """Return the directory holding the runtime's bundled web-game pages."""

    return Path(__file__).resolve().parent / "assets"
