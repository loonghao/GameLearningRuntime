"""Fail-closed accounting for the metrics an adapter promises to emit.

A metric that is declared but never emitted reads as a healthy zero: the run
store has nothing, ``glr observe`` projects nothing, and every gate that
consumes metrics sees a number instead of a gap. Nothing errors, nothing is
missing from the schema, and one number quietly loses its meaning.

This module closes the hole the same way capture output, checkpoint contracts,
and host readiness already do -- by naming the gap instead of reporting a quiet
zero. The check runs at episode close, because a gap discovered while a report
is rendered is discovered after the next round has already been spent on the
broken instrumentation.

Three counters are always first class, in the run store and in
``glr.cli-output.v1``:

``declared_metrics``
    How many metrics the adapter promised to emit for every episode.

``emitted_metrics``
    How many of those promises the episode actually kept.

``missing_metrics_count``
    How many promises it broke, as a count. The names live in the audit event
    and in the summary, never in a metric: a count and a name list must not
    share one field name inside the same envelope.

The counters are the half that gets the instrumentation fixed. The typed
:class:`MissingDeclaredMetric` error is the half that stops it from coming back,
and it stays opt-in behind the ``strict-metrics-v1`` capability so an adapter
that starts declaring metrics does not start failing its runs on the same
commit. An adapter that declares nothing behaves exactly as it did before.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from time import time_ns
from typing import TYPE_CHECKING, Any

from game_learning_runtime.errors import GLRError

if TYPE_CHECKING:  # pragma: no cover - typing-only imports; specs imports this module
    from game_learning_runtime.run_store import TrainingStore
    from game_learning_runtime.specs import EnvironmentSpec

DECLARED_METRICS_SCHEMA_VERSION = "glr.declared-metrics.v1"

#: Run-store event kind carrying one closed episode's accounting.
DECLARED_METRICS_EVENT = "declared_metrics.audit"

#: Capability that turns a missing declared metric into a run-fatal error.
STRICT_METRICS_CAPABILITY = "strict-metrics-v1"

#: Spelling used by GLR issue #157, accepted so the issue and the code agree.
STRICT_METRICS_CAPABILITY_ALIAS = "strict_metrics"

#: First-class run metric names for the three counters.
DECLARED_METRICS_METRIC = "declared_metrics"
EMITTED_METRICS_METRIC = "emitted_metrics"
MISSING_METRICS_COUNT_METRIC = "missing_metrics_count"

#: Every metric name a run store accepts: lowercase, dot-separated, no spaces.
_METRIC_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")

#: Bounds so a declaration stays a declaration and not a data structure.
_MAX_METRIC_NAME_LENGTH = 128
_MAX_DECLARED_METRICS = 64

_LEDGERS: dict[tuple[str, str], DeclaredMetricLedger] = {}
_LEDGERS_LOCK = threading.Lock()


class MissingDeclaredMetric(GLRError):
    """A declared metric was never emitted, so its value would read as zero.

    Deliberately not a :class:`ContractViolation` and not a transport error:
    the adapter kept its contract, it simply measured nothing. The message
    names every missing metric and every way out, because the reader is usually
    an operator who did not write the instrumentation.
    """

    def __init__(
        self,
        *,
        missing_metrics: Sequence[str],
        declared_metrics: int,
        emitted_metrics: int,
        episode_id: str | None = None,
    ) -> None:
        names = sorted(set(missing_metrics))
        if not names:
            raise ValueError("missing_metrics cannot be empty")
        scope = f" in episode {episode_id}" if episode_id is not None else ""
        super().__init__(
            f"missing declared metrics{scope}: {names} "
            f"(declared {declared_metrics}, emitted {emitted_metrics}); "
            "a declared metric that is never emitted reads as a real zero, so "
            "emit it, drop it from the declaration, or mark it optional"
        )
        self.missing_metrics: tuple[str, ...] = tuple(names)
        self.declared_metrics = declared_metrics
        self.emitted_metrics = emitted_metrics
        self.episode_id = episode_id


def _metric_names(values: Iterable[object], *, path: str) -> tuple[str, ...]:
    """Validate, normalize, and de-duplicate one declared metric name set."""

    if isinstance(values, str) or not isinstance(values, Iterable):
        raise TypeError(f"{path} must be a sequence of metric names")
    resolved: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{path} entries must be strings")
        if not value or len(value) > _MAX_METRIC_NAME_LENGTH:
            raise ValueError(f"{path} entries must contain 1-{_MAX_METRIC_NAME_LENGTH} characters")
        if _METRIC_NAME.fullmatch(value) is None:
            raise ValueError(f"{path} entry {value!r} must match {_METRIC_NAME.pattern!r}")
        resolved.add(value)
    if len(resolved) > _MAX_DECLARED_METRICS:
        raise ValueError(f"{path} cannot declare more than {_MAX_DECLARED_METRICS} metrics")
    return tuple(sorted(resolved))


def _optional_metric_names(values: Iterable[object], *, path: str) -> tuple[str, ...]:
    return () if values is None else _metric_names(values, path=path)


def _string_tuple(value: object, *, path: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise ValueError(f"{path} must be a list of metric names")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{path} must contain only metric names")
        names.append(item)
    return tuple(names)


def _strict_flag(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_schema_version(value: Mapping[str, Any]) -> None:
    version = value.get("schema_version")
    if version != DECLARED_METRICS_SCHEMA_VERSION:
        raise ValueError(
            f"declared metric audit must declare "
            f"{DECLARED_METRICS_SCHEMA_VERSION!r}, got {version!r}"
        )


@dataclass(frozen=True, slots=True)
class MetricDeclaration:
    """The metrics one adapter promises to emit during every episode.

    ``expected`` names the promises: a name listed here that an episode never
    emits is a gap, and it is reported by name. ``optional`` names the extras
    an adapter may or may not emit, so a reader can tell "declared and absent"
    from "never part of the contract".
    """

    expected: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    strict: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", _metric_names(self.expected, path="expected"))
        object.__setattr__(self, "optional", _optional_metric_names(self.optional, path="optional"))
        overlap = sorted(set(self.expected) & set(self.optional))
        if overlap:
            raise ValueError(f"metrics cannot be both expected and optional: {overlap}")
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be bool")

    @property
    def declared(self) -> tuple[str, ...]:
        """Every name this adapter promised, expected or optional."""

        return tuple(sorted(set(self.expected) | set(self.optional)))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": DECLARED_METRICS_SCHEMA_VERSION,
            "expected": list(self.expected),
            "optional": list(self.optional),
            "strict": self.strict,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MetricDeclaration:
        if not isinstance(value, Mapping):
            raise TypeError("metric declaration must be a mapping")
        _require_schema_version(value)
        return cls(
            expected=_string_tuple(value.get("expected", ()), path="expected"),
            optional=_string_tuple(value.get("optional", ()), path="optional"),
            strict=_strict_flag(value.get("strict", False), path="strict"),
        )


@dataclass(frozen=True, slots=True)
class DeclaredMetricAudit:
    """The declared-versus-emitted accounting for one closed episode.

    The three counters are properties of the name tuples, so a persisted audit
    can never claim a count its names disagree with.
    """

    episode_id: str | None = None
    expected: tuple[str, ...] = ()
    emitted: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    emitted_optional: tuple[str, ...] = ()
    strict: bool = False
    timestamp_ns: int | None = None

    @property
    def missing_metrics(self) -> tuple[str, ...]:
        """Declared metrics this episode never emitted, by name."""

        return tuple(sorted(set(self.expected) - set(self.emitted)))

    @property
    def declared_metrics(self) -> int:
        return len(self.expected)

    @property
    def emitted_metrics(self) -> int:
        return len(self.emitted)

    @property
    def missing_metrics_count(self) -> int:
        return len(self.missing_metrics)

    @property
    def optional_metrics(self) -> int:
        return len(self.optional)

    @property
    def emitted_optional_metrics(self) -> int:
        return len(self.emitted_optional)

    @property
    def complete(self) -> bool:
        """Whether every declared metric was emitted."""

        return not self.missing_metrics

    def require(self) -> DeclaredMetricAudit:
        """Raise when strict mode is on and a declared metric never arrived.

        Non-strict audits return unchanged: the counters are still on the
        record, which is the half that gets the instrumentation fixed.
        """

        if not self.strict or self.complete:
            return self
        raise MissingDeclaredMetric(
            missing_metrics=self.missing_metrics,
            declared_metrics=self.declared_metrics,
            emitted_metrics=self.emitted_metrics,
            episode_id=self.episode_id,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": DECLARED_METRICS_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "timestamp_ns": self.timestamp_ns,
            "strict": self.strict,
            "complete": self.complete,
            "declared_metrics": self.declared_metrics,
            "emitted_metrics": self.emitted_metrics,
            "missing_metrics": list(self.missing_metrics),
            "missing_metrics_count": self.missing_metrics_count,
            "expected": list(self.expected),
            "emitted": list(self.emitted),
            "optional": list(self.optional),
            "emitted_optional": list(self.emitted_optional),
            "optional_metrics": self.optional_metrics,
            "emitted_optional_metrics": self.emitted_optional_metrics,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DeclaredMetricAudit:
        if not isinstance(value, Mapping):
            raise TypeError("declared metric audit must be a mapping")
        _require_schema_version(value)
        episode_id = value.get("episode_id")
        timestamp_ns = value.get("timestamp_ns")
        return cls(
            episode_id=None if episode_id is None else str(episode_id),
            expected=_string_tuple(value.get("expected", ()), path="expected"),
            emitted=_string_tuple(value.get("emitted", ()), path="emitted"),
            optional=_string_tuple(value.get("optional", ()), path="optional"),
            emitted_optional=_string_tuple(
                value.get("emitted_optional", ()), path="emitted_optional"
            ),
            strict=_strict_flag(value.get("strict", False), path="strict"),
            timestamp_ns=None if timestamp_ns is None else int(timestamp_ns),
        )


def _ledger_key(store: TrainingStore | str | os.PathLike[str], run_id: str) -> tuple[str, str]:
    """Key a ledger by the absolute store path, so the two sides always agree.

    A store owns a ``path``; a caller that only has the path string can bind by
    the same value. Both are normalized with ``abspath`` so a relative path
    from one side still meets an absolute path from the other.
    """

    path: object = getattr(store, "path", store)
    return (os.path.abspath(str(path)), run_id)


def bind_declared_metrics(
    store: TrainingStore | str | os.PathLike[str],
    run_id: str,
    ledger: DeclaredMetricLedger,
) -> None:
    """Bind one ledger to a run so every recorded metric is counted.

    Metrics reach the run store through :class:`~game_learning_runtime.telemetry.Telemetry`
    or a direct ``record_metric`` call, and neither knows about the collector
    that owns the ledger. Binding by store path and run id is what lets a
    broken adapter -- one that never calls anything -- be counted as missing
    rather than as zero.
    """

    if not isinstance(ledger, DeclaredMetricLedger):
        raise TypeError("ledger must be a DeclaredMetricLedger")
    if not run_id:
        raise ValueError("run_id cannot be empty")
    key = _ledger_key(store, run_id)
    with _LEDGERS_LOCK:
        bound = _LEDGERS.get(key)
        # Re-binding the same ledger is a no-op: a ledger constructed with a
        # store is already bound, and a caller that then scopes the run with
        # ``bound_declared_metrics`` must not be told it raced itself.
        if bound is not None and bound is not ledger:
            raise ValueError(f"a different declared-metric ledger is already bound to {key[1]}")
        _LEDGERS[key] = ledger


def release_declared_metrics(store: TrainingStore | str | os.PathLike[str], run_id: str) -> None:
    """Unbind the ledger for one run, if any is bound."""

    with _LEDGERS_LOCK:
        _LEDGERS.pop(_ledger_key(store, run_id), None)


def declared_metrics_for(
    store: TrainingStore | str | os.PathLike[str], run_id: str
) -> DeclaredMetricLedger | None:
    """Return the ledger bound to one run, or ``None``."""

    with _LEDGERS_LOCK:
        return _LEDGERS.get(_ledger_key(store, run_id))


@contextmanager
def bound_declared_metrics(
    store: TrainingStore | str | os.PathLike[str],
    run_id: str,
    ledger: DeclaredMetricLedger,
) -> Iterator[None]:
    """Bind a ledger for the duration of one run and always release it."""

    bind_declared_metrics(store, run_id, ledger)
    try:
        yield
    finally:
        release_declared_metrics(store, run_id)


class DeclaredMetricLedger:
    """Per-episode bookkeeping for one run's declared metrics.

    ``record`` notes that a declared name was emitted; ``close_episode`` ends
    one episode, returns its audit, and persists it through the bound store so
    the counters survive even when the caller raises immediately afterwards.
    """

    def __init__(
        self,
        declaration: MetricDeclaration,
        *,
        strict: bool = False,
        store: TrainingStore | None = None,
        run_id: str | None = None,
    ) -> None:
        if not isinstance(declaration, MetricDeclaration):
            raise TypeError("declaration must be a MetricDeclaration")
        if (store is None) != (run_id is None):
            raise ValueError("store and run_id must be supplied together")
        if not isinstance(strict, bool):
            raise TypeError("strict must be bool")
        self._declaration = declaration
        self._strict = strict or declaration.strict
        self._store = store
        self._run_id = run_id
        self._watched = frozenset(declaration.declared)
        self._emitted: set[str] = set()
        self._audits: list[DeclaredMetricAudit] = []
        self._lock = threading.Lock()
        if store is not None and run_id is not None:
            bind_declared_metrics(store, run_id, self)

    @property
    def declaration(self) -> MetricDeclaration:
        return self._declaration

    @property
    def strict(self) -> bool:
        return self._strict

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def emitted(self) -> tuple[str, ...]:
        """Declared names emitted during the open episode."""

        with self._lock:
            return tuple(sorted(self._emitted))

    @property
    def audits(self) -> tuple[DeclaredMetricAudit, ...]:
        """Every closed episode's audit, oldest first."""

        with self._lock:
            return tuple(self._audits)

    def record(self, name: str) -> bool:
        """Note that ``name`` was emitted; ``True`` when it was declared."""

        if not isinstance(name, str):
            raise TypeError("metric name must be a string")
        if name not in self._watched:
            return False
        with self._lock:
            self._emitted.add(name)
        return True

    def close_episode(
        self, episode_id: str | None = None, *, timestamp_ns: int | None = None
    ) -> DeclaredMetricAudit:
        """Close one episode, persist its accounting, and return the audit.

        Persistence happens before any caller can raise, because the counters
        are what gets the instrumentation fixed; the error only stops it from
        coming back.
        """

        if episode_id is not None and (not episode_id or len(episode_id) > 128):
            raise ValueError("episode_id must contain 1-128 characters or None")
        if timestamp_ns is not None and (not isinstance(timestamp_ns, int) or timestamp_ns < 0):
            raise ValueError("timestamp_ns must be a non-negative integer or None")
        with self._lock:
            emitted = frozenset(self._emitted)
            self._emitted.clear()
        expected = frozenset(self._declaration.expected)
        optional = frozenset(self._declaration.optional)
        audit = DeclaredMetricAudit(
            episode_id=episode_id,
            expected=self._declaration.expected,
            emitted=tuple(sorted(emitted & expected)),
            optional=self._declaration.optional,
            emitted_optional=tuple(sorted(emitted & optional)),
            strict=self._strict,
            timestamp_ns=time_ns() if timestamp_ns is None else timestamp_ns,
        )
        with self._lock:
            self._audits.append(audit)
        if self._store is not None and self._run_id is not None:
            self._store.record_declared_metric_audit(
                self._run_id, audit, timestamp_ns=audit.timestamp_ns
            )
        return audit

    def release(self) -> None:
        """Unbind this ledger from its run, if it was bound."""

        if self._store is not None and self._run_id is not None:
            release_declared_metrics(self._store, self._run_id)

    def summary(self) -> dict[str, Any]:
        """Project the newest audit so a caller can gate without walking events."""

        return summarize_declared_metrics(self.audits)


def build_declared_metrics(
    declaration: MetricDeclaration | None = None,
    *,
    capabilities: Iterable[str] = (),
    strict: bool | None = None,
    store: TrainingStore | None = None,
    run_id: str | None = None,
) -> DeclaredMetricLedger | None:
    """Build a ledger when something is declared, else ``None``.

    A caller with no declaration gets no ledger: no counters, no event, and no
    behaviour that differs from before the declaration existed.
    """

    if declaration is None:
        return None
    if not isinstance(declaration, MetricDeclaration):
        raise TypeError("declaration must be a MetricDeclaration or None")
    resolved_strict = strict
    if resolved_strict is None:
        granted = frozenset(capabilities)
        resolved_strict = (
            STRICT_METRICS_CAPABILITY in granted
            or STRICT_METRICS_CAPABILITY_ALIAS in granted
            or declaration.strict
        )
    return DeclaredMetricLedger(declaration, strict=resolved_strict, store=store, run_id=run_id)


def declared_metrics_from_spec(
    spec: EnvironmentSpec,
    *,
    store: TrainingStore | None = None,
    run_id: str | None = None,
) -> DeclaredMetricLedger | None:
    """Build a ledger from an adapter spec: its declaration and capabilities."""

    return build_declared_metrics(
        spec.metrics, capabilities=spec.capabilities, store=store, run_id=run_id
    )


def summarize_declared_metrics(audits: Sequence[DeclaredMetricAudit]) -> dict[str, Any]:
    """Project the newest audit so a scheduler can gate without parsing logs.

    An absent audit reports ``reported=False`` and null counters, never a
    passing zero: "nothing was measured" and "nothing was missing" are
    different facts and must not look the same.
    """

    if not audits:
        return {
            "schema_version": DECLARED_METRICS_SCHEMA_VERSION,
            "reported": False,
            "audit_count": 0,
            "episode_id": None,
            "strict": False,
            "complete": None,
            "declared_metrics": None,
            "emitted_metrics": None,
            "missing_metrics": [],
            "missing_metrics_count": None,
        }
    latest = audits[-1]
    missing = sorted({name for audit in audits for name in audit.missing_metrics})
    return {
        "schema_version": DECLARED_METRICS_SCHEMA_VERSION,
        "reported": True,
        "audit_count": len(audits),
        "episode_id": latest.episode_id,
        "strict": latest.strict,
        "complete": latest.complete,
        "declared_metrics": latest.declared_metrics,
        "emitted_metrics": latest.emitted_metrics,
        "missing_metrics": missing,
        "missing_metrics_count": latest.missing_metrics_count,
    }


__all__ = [
    "DECLARED_METRICS_EVENT",
    "DECLARED_METRICS_METRIC",
    "DECLARED_METRICS_SCHEMA_VERSION",
    "EMITTED_METRICS_METRIC",
    "MISSING_METRICS_COUNT_METRIC",
    "STRICT_METRICS_CAPABILITY",
    "STRICT_METRICS_CAPABILITY_ALIAS",
    "DeclaredMetricAudit",
    "DeclaredMetricLedger",
    "MetricDeclaration",
    "MissingDeclaredMetric",
    "bind_declared_metrics",
    "bound_declared_metrics",
    "build_declared_metrics",
    "declared_metrics_for",
    "declared_metrics_from_spec",
    "release_declared_metrics",
    "summarize_declared_metrics",
]
