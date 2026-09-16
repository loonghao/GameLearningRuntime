"""Learner-neutral durable diagnostics, projected by ``glr observe``.

Telemetry does not execute actions or establish reward/terminal authority.
Learners explicitly report updates; GLR never infers them from process output.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from game_learning_runtime.run_store import TrainingStore


class Telemetry:
    """Persist first, then optionally print structured diagnostics to stderr."""

    def __init__(self, store: TrainingStore, run_id: str, *, console: bool = True) -> None:
        self.store = store
        self.run_id = run_id
        self.console = console

    @classmethod
    def from_env(cls) -> Telemetry | None:
        """Bind only to an explicitly supplied GLR run, never create a run."""
        path, run_id = os.environ.get("GLR_STORE_PATH"), os.environ.get("GLR_RUN_ID")
        if not path or not run_id:
            return None
        return _environment_telemetry(path, run_id, os.environ.get("GLR_TELEMETRY_STDERR") != "0")

    def event(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        step_id: int | None = None,
        episode_id: str | None = None,
    ) -> None:
        event = self.store.append_event(
            self.run_id, kind=kind, payload=payload, step_id=step_id, episode_id=episode_id
        )
        if self.console:
            # Printing failure must not invalidate an already committed record.
            with contextlib.suppress(OSError):
                print(
                    json.dumps(
                        {
                            "schema_version": "glr.telemetry.v1",
                            "run_id": self.run_id,
                            "sequence_id": event.sequence_id,
                            "timestamp_ns": event.timestamp_ns,
                            "kind": kind,
                            "step_id": step_id,
                            "episode_id": episode_id,
                            "payload": dict(payload),
                        },
                        ensure_ascii=True,
                        allow_nan=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    def metric(self, name: str, value: float, *, step_id: int | None = None) -> None:
        self.store.record_metric(
            self.run_id,
            name=name,
            value=value,
            step_id=step_id,
            metadata={"source": "learner", "authority": "diagnostic"},
        )

    def learning_update(
        self,
        *,
        step_id: int,
        metrics: Mapping[str, float],
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Record an explicit learner update; diagnostic metrics are not rewards."""
        self.event(
            "learning.update",
            {"metrics": dict(metrics), "details": dict(details or {})},
            step_id=step_id,
        )
        for name, value in metrics.items():
            self.metric(name, value, step_id=step_id)

    def route_sample(
        self,
        position: Sequence[float],
        *,
        step_id: int,
        episode_id: str,
        world_id: str = "default",
        route_id: str = "default",
    ) -> None:
        import math

        if len(position) not in (2, 3) or not all(math.isfinite(v) for v in position):
            raise ValueError("route position requires two or three finite coordinates")
        self.event(
            "navigation.route_sample",
            {"position": list(position), "world_id": world_id, "route_id": route_id},
            step_id=step_id,
            episode_id=episode_id,
        )


@lru_cache(maxsize=16)
def _environment_telemetry(path: str, run_id: str, console: bool) -> Telemetry:
    return Telemetry(TrainingStore(path), run_id, console=console)


def decision_event(kind: str, payload: Mapping[str, Any], step_id: int | None) -> None:
    """Passive default hook: a diagnostic failure never retries a game action."""
    try:
        telemetry = Telemetry.from_env()
        if telemetry is not None:
            telemetry.event(kind, payload, step_id=step_id)
    except Exception as error:
        with contextlib.suppress(OSError):
            print(f"GLR telemetry unavailable: {type(error).__name__}", file=sys.stderr, flush=True)
