"""Passive bridge diagnostics over localhost HTTP or the standalone GLR CLI.

Prepare a batch once and retain it until acknowledged. Retrying the same batch
is idempotent; this client never retries actions or launches a game runtime.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

SCHEMA = "glr.bridge-telemetry.v1"


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class BridgeTelemetry:
    """Bounded synchronous publisher; call from a worker, not an engine frame loop."""

    def __init__(
        self,
        run_id: str,
        source: str,
        *,
        url: str | None = None,
        token: str | None = None,
        cli_path: str | None = None,
        project_root: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        if not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise ValueError("timeout must be in (0, 30] seconds")
        if url is not None:
            parsed = urlsplit(url)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in ("127.0.0.1", "localhost")
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path != "/api/v1/telemetry"
                or parsed.query
                or parsed.fragment
                or not token
                or cli_path is not None
            ):
                raise ValueError("HTTP telemetry requires an exact localhost endpoint and token")
        elif not cli_path or not Path(cli_path).is_absolute() or not project_root:
            raise ValueError("CLI telemetry requires an absolute executable and project root")
        self.run_id, self.source = run_id, source
        self.url, self._token = url, token
        self.cli_path, self.project_root = cli_path, project_root
        self.timeout = timeout

    @classmethod
    def from_env(cls, source: str) -> BridgeTelemetry | None:
        """Use credentials supplied to a GLR job, or its standalone CLI binding."""
        run = os.environ.get("GLR_RUN_ID")
        if not run:
            return None
        url = os.environ.get("GLR_TELEMETRY_URL")
        if url:
            return cls(run, source, url=url, token=os.environ.get("GLR_TELEMETRY_TOKEN"))
        executable, root = os.environ.get("GLR_CLI_PATH"), os.environ.get("GLR_PROJECT_ROOT")
        if not executable or not root:
            return None
        return cls(run, source, cli_path=executable, project_root=root)

    def prepare(
        self,
        *,
        events: Sequence[Mapping[str, Any]] = (),
        metrics: Sequence[Mapping[str, Any]] = (),
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        """Build JSON data that can also be written to a durable JSONL spool."""
        batch = {
            "schema_version": SCHEMA,
            "run_id": self.run_id,
            "source": self.source,
            "batch_id": batch_id or f"batch-{uuid4().hex}",
            "events": list(events),
            "metrics": list(metrics),
        }
        self._encode(batch)
        # Copy caller-owned nested objects so later mutations do not alter retries.
        return dict(json.loads(json.dumps(batch, allow_nan=False)))

    def _encode(self, batch: Mapping[str, Any]) -> bytes:
        if batch.get("run_id") != self.run_id or batch.get("source") != self.source:
            raise ValueError("batch does not match this publisher's run and source")
        count = len(batch.get("events", ())) + len(batch.get("metrics", ()))
        if not 1 <= count <= 64:
            raise ValueError("a batch requires 1..64 events and metrics")
        encoded = json.dumps(batch, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > 65536:
            raise ValueError("batch exceeds 64 KiB")
        return encoded

    def send(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Commit once and return a durable receipt; no automatic network retry.

        On timeout the outcome is unknown. Retain and retry this exact batch ID
        and content. JSONL import commits per line, never the entire file at once.
        """
        encoded = self._encode(batch)
        if self.url is not None:
            request = Request(
                self.url,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            # Never forward local credentials through a proxy or redirect.
            opener = build_opener(ProxyHandler({}), _NoRedirect())
            with opener.open(request, timeout=self.timeout) as response:
                body = response.read(65537)
            if len(body) > 65536:
                raise ValueError("telemetry receipt exceeds 64 KiB")
            receipt = json.loads(body)
        else:
            assert self.cli_path is not None and self.project_root is not None
            result = subprocess.run(
                [
                    self.cli_path,
                    "--project",
                    self.project_root,
                    "--json",
                    "telemetry",
                    "ingest",
                    "--file",
                    "-",
                ],
                input=encoded,
                capture_output=True,
                timeout=self.timeout,
                check=True,
                env={**os.environ, "GLR_NO_UPDATE_CHECK": "1"},
            )
            receipt = json.loads(result.stdout)["data"]["receipts"][0]
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != SCHEMA
            or any(receipt.get(key) != batch.get(key) for key in ("run_id", "source", "batch_id"))
        ):
            raise ValueError("telemetry receipt does not match the submitted batch")
        return receipt
