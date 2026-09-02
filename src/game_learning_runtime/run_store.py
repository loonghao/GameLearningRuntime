"""Durable local training-run metadata and query projections."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from time import time_ns
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from game_learning_runtime.agent_goal import (
    ResearchBundle,
    ResearchCategory,
    ResearchFinding,
    ResearchScope,
    ResearchSource,
    ResearchStatus,
)
from game_learning_runtime.contracts import (
    EnvironmentConfigSnapshot,
    environment_config_digest,
    normalize_environment_config,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.training import KnowledgeAuthority

RUN_STORE_SCHEMA_VERSION = 2
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_RUN_STATE_BYTES = 64 * 1024


class RunStatus(str, Enum):
    """Terminal and active states for one local GLR execution."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must match {_IDENTIFIER.pattern!r}")
    return value


def _optional_identifier(value: object, *, path: str) -> str | None:
    return None if value is None else _identifier(value, path=path)


def _resolved_environment_config_digest(
    snapshot: EnvironmentConfigSnapshot | None,
    digest: str | None,
) -> str | None:
    normalized = normalize_environment_config(snapshot)
    expected = environment_config_digest(normalized)
    if digest is None:
        return expected
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("environment_config_digest must be a lowercase SHA-256 digest")
    if expected is not None and digest != expected:
        raise ValueError("environment_config_digest does not match the snapshot")
    return digest


def _non_negative_integer(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _limit(value: object, *, maximum: int = 1000) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return value


def _json_mapping(value: Mapping[str, Any], *, path: str) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be an object with string keys")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain finite JSON data") from error
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{path} exceeds the 1 MiB limit")
    decoded = json.loads(encoded)
    return encoded, MappingProxyType(decoded)


def _run_state_namespace(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError("run state namespace must be 1-128 characters")
    parts = value.split("/")
    if any(_IDENTIFIER.fullmatch(part) is None for part in parts):
        raise ValueError("run state namespace must contain identifier segments separated by '/'")
    return value


def _run_state_schema_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError("run state schema_version must be between 1 and 65535")
    return value


def _run_state_mapping(value: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or _IDENTIFIER.fullmatch(key) is None for key in value
    ):
        raise ValueError("run state must be an object with identifier keys")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("run state must contain finite JSON data") from error
    if len(encoded.encode("utf-8")) > _MAX_RUN_STATE_BYTES:
        raise ValueError("run state exceeds the 64 KiB limit")
    decoded = json.loads(encoded)
    return encoded, MappingProxyType(decoded)


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    environment_id: str
    protocol_version: str
    kind: str
    status: RunStatus
    started_at_ns: int
    finished_at_ns: int | None
    exit_code: int | None
    metadata: Mapping[str, Any]
    environment_config_digest: str | None = None


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence_id: int
    timestamp_ns: int
    kind: str
    episode_id: str | None
    step_id: int | None
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class MetricRecord:
    run_id: str
    metric_id: int
    timestamp_ns: int
    name: str
    value: float
    step_id: int | None
    metadata: Mapping[str, Any]
    environment_config_digest: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    run_id: str
    path: str
    role: str
    media_type: str
    sha256: str
    size_bytes: int
    metadata: Mapping[str, Any]


class RunState(MutableMapping[str, Any]):
    """Adapter-owned, run-scoped JSON state with write-through persistence.

    Values are opaque to GLR. Assign a key to persist a new snapshot; nested
    mutable values should be replaced after editing rather than mutated in
    place, because only mapping operations are write-through.
    """

    def __init__(
        self,
        store: TrainingStore,
        run_id: str,
        namespace: str,
        schema_version: int,
        values: Mapping[str, Any],
    ) -> None:
        self._store = store
        self.run_id = run_id
        self.namespace = namespace
        self.schema_version = schema_version
        self._values = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        _identifier(key, path="run state key")
        updated = dict(self._values)
        updated[key] = value
        encoded, frozen = _run_state_mapping(updated)
        self._store._write_run_state(
            self.run_id,
            self.namespace,
            self.schema_version,
            encoded,
        )
        self._values = dict(frozen)

    def __delitem__(self, key: str) -> None:
        if key not in self._values:
            raise KeyError(key)
        updated = dict(self._values)
        del updated[key]
        encoded, frozen = _run_state_mapping(updated)
        self._store._write_run_state(
            self.run_id,
            self.namespace,
            self.schema_version,
            encoded,
        )
        self._values = dict(frozen)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def snapshot(self) -> Mapping[str, Any]:
        """Return an immutable copy of the currently loaded state."""

        _, frozen = _run_state_mapping(self._values)
        return frozen


def _portable_path(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"{path} must be a portable relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{path} must be a portable relative path")
    return candidate.as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _position(value: Sequence[float], *, path: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{path} must contain exactly three coordinates")
    result = tuple(float(coordinate) for coordinate in value)
    if any(not math.isfinite(coordinate) for coordinate in result):
        raise ValueError(f"{path} coordinates must be finite")
    return result  # type: ignore[return-value]


def _bounded_label(value: object, *, path: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{path} must be non-empty text up to 256 characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{path} cannot contain control characters")
    return value


@dataclass(frozen=True, slots=True)
class SpatialEntity:
    """One previously observed object position with explicit provenance."""

    environment_id: str
    world_id: str
    entity_id: str
    kind: str
    label: str
    position: tuple[float, float, float]
    coordinate_frame: str
    authority: KnowledgeAuthority
    confidence: float
    observed_at_ns: int
    source_run_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for path, value in (
            ("environment_id", self.environment_id),
            ("world_id", self.world_id),
            ("entity_id", self.entity_id),
            ("entity kind", self.kind),
            ("coordinate_frame", self.coordinate_frame),
            ("source_run_id", self.source_run_id),
        ):
            _identifier(value, path=path)
        object.__setattr__(self, "label", _bounded_label(self.label, path="entity label"))
        object.__setattr__(self, "position", _position(self.position, path="entity position"))
        object.__setattr__(self, "authority", KnowledgeAuthority(self.authority))
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("entity confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        if (
            not isinstance(self.observed_at_ns, int)
            or isinstance(self.observed_at_ns, bool)
            or self.observed_at_ns < 0
        ):
            raise ValueError("observed_at_ns must be a non-negative integer")
        _, frozen = _json_mapping(self.metadata, path="entity metadata")
        object.__setattr__(self, "metadata", frozen)


@dataclass(frozen=True, slots=True)
class RouteWaypoint:
    """One non-executable waypoint in an advisory route."""

    index: int
    position: tuple[float, float, float]
    tolerance: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError("waypoint index must be a non-negative integer")
        object.__setattr__(self, "position", _position(self.position, path="waypoint position"))
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("waypoint tolerance must be positive and finite")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(
            self, "label", _bounded_label(self.label, path="waypoint label", optional=True)
        )


@dataclass(frozen=True, slots=True)
class SpatialRoute:
    """A provenance-bound advisory route that must be revalidated while moving."""

    environment_id: str
    world_id: str
    route_id: str
    name: str
    from_entity_id: str | None
    to_entity_id: str | None
    coordinate_frame: str
    confidence: float
    verified_at_ns: int
    source_run_id: str
    waypoints: tuple[RouteWaypoint, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for path, value in (
            ("environment_id", self.environment_id),
            ("world_id", self.world_id),
            ("route_id", self.route_id),
            ("coordinate_frame", self.coordinate_frame),
            ("source_run_id", self.source_run_id),
        ):
            _identifier(value, path=path)
        _optional_identifier(self.from_entity_id, path="from_entity_id")
        _optional_identifier(self.to_entity_id, path="to_entity_id")
        object.__setattr__(self, "name", _bounded_label(self.name, path="route name"))
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("route confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        if (
            not isinstance(self.verified_at_ns, int)
            or isinstance(self.verified_at_ns, bool)
            or self.verified_at_ns < 0
        ):
            raise ValueError("verified_at_ns must be a non-negative integer")
        waypoints = tuple(self.waypoints)
        if len(waypoints) < 2 or any(
            not isinstance(waypoint, RouteWaypoint) for waypoint in waypoints
        ):
            raise ValueError("a route requires at least two RouteWaypoint values")
        if tuple(waypoint.index for waypoint in waypoints) != tuple(range(len(waypoints))):
            raise ValueError("route waypoint indexes must be contiguous from zero")
        object.__setattr__(self, "waypoints", waypoints)
        _, frozen = _json_mapping(self.metadata, path="route metadata")
        object.__setattr__(self, "metadata", frozen)


class TrainingStore:
    """SQLite-backed local authority for run summaries and agent queries."""

    def __init__(self, path: str | Path) -> None:
        requested = Path(path)
        if requested.is_symlink():
            raise FileExistsError("run store cannot be a symlink")
        requested.parent.mkdir(parents=True, exist_ok=True)
        self._path = requested.resolve()
        self._initialize()

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, 1, RUN_STORE_SCHEMA_VERSION}:
                raise ContractViolation(f"unsupported run store schema version: {version}")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    environment_id TEXT NOT NULL,
                    protocol_version TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at_ns INTEGER NOT NULL,
                    finished_at_ns INTEGER,
                    exit_code INTEGER,
                    metadata_json TEXT NOT NULL,
                    environment_config_digest TEXT
                );
                CREATE INDEX IF NOT EXISTS runs_environment_started
                    ON runs(environment_id, started_at_ns DESC);
                CREATE TABLE IF NOT EXISTS run_state (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    namespace TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at_ns INTEGER NOT NULL,
                    PRIMARY KEY(run_id, namespace)
                );
                CREATE INDEX IF NOT EXISTS run_state_run
                    ON run_state(run_id, namespace);
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    sequence_id INTEGER NOT NULL,
                    timestamp_ns INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    episode_id TEXT,
                    step_id INTEGER,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, sequence_id)
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    timestamp_ns INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    step_id INTEGER,
                    metadata_json TEXT NOT NULL,
                    environment_config_digest TEXT
                );
                CREATE INDEX IF NOT EXISTS metrics_run_name_step
                    ON metrics(run_id, name, step_id);
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    role TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, path)
                );
                CREATE TABLE IF NOT EXISTS spatial_entities (
                    environment_id TEXT NOT NULL,
                    world_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    coordinate_frame TEXT NOT NULL,
                    authority TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    observed_at_ns INTEGER NOT NULL,
                    source_run_id TEXT NOT NULL REFERENCES runs(run_id),
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(environment_id, world_id, entity_id)
                );
                CREATE INDEX IF NOT EXISTS spatial_entities_lookup
                    ON spatial_entities(environment_id, world_id, kind, label);
                CREATE TABLE IF NOT EXISTS spatial_routes (
                    environment_id TEXT NOT NULL,
                    world_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    from_entity_id TEXT,
                    to_entity_id TEXT,
                    coordinate_frame TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    verified_at_ns INTEGER NOT NULL,
                    source_run_id TEXT NOT NULL REFERENCES runs(run_id),
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY(environment_id, world_id, route_id)
                );
                CREATE INDEX IF NOT EXISTS spatial_routes_lookup
                    ON spatial_routes(environment_id, world_id, from_entity_id, to_entity_id);
                CREATE TABLE IF NOT EXISTS route_waypoints (
                    environment_id TEXT NOT NULL,
                    world_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    waypoint_index INTEGER NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    tolerance REAL NOT NULL,
                    label TEXT,
                    PRIMARY KEY(environment_id, world_id, route_id, waypoint_index),
                    FOREIGN KEY(environment_id, world_id, route_id)
                        REFERENCES spatial_routes(environment_id, world_id, route_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS research_sources (
                    source_id TEXT PRIMARY KEY,
                    media_type TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    source_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_findings (
                    finding_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_id TEXT,
                    finding_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS research_findings_lookup
                    ON research_findings(scope, scope_id, category, status);
                CREATE TABLE IF NOT EXISTS research_finding_sources (
                    finding_id TEXT NOT NULL REFERENCES research_findings(finding_id)
                        ON DELETE CASCADE,
                    source_id TEXT NOT NULL REFERENCES research_sources(source_id),
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY(finding_id, source_id)
                );
                CREATE TABLE IF NOT EXISTS research_finding_tags (
                    finding_id TEXT NOT NULL REFERENCES research_findings(finding_id)
                        ON DELETE CASCADE,
                    tag TEXT NOT NULL,
                    PRIMARY KEY(finding_id, tag)
                );
                CREATE INDEX IF NOT EXISTS research_finding_tags_lookup
                    ON research_finding_tags(tag, finding_id);
                """
            )
            run_columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
            metric_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(metrics)")
            }
            if "environment_config_digest" not in run_columns:
                connection.execute("ALTER TABLE runs ADD COLUMN environment_config_digest TEXT")
            if "environment_config_digest" not in metric_columns:
                connection.execute("ALTER TABLE metrics ADD COLUMN environment_config_digest TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_environment_config "
                "ON runs(environment_id, environment_config_digest, started_at_ns)"
            )
            connection.execute(f"PRAGMA user_version = {RUN_STORE_SCHEMA_VERSION}")

    def create_run(
        self,
        *,
        environment_id: str,
        protocol_version: str,
        kind: str,
        metadata: Mapping[str, Any] | None = None,
        environment_config_snapshot: EnvironmentConfigSnapshot | None = None,
        environment_config_digest: str | None = None,
        run_id: str | None = None,
        started_at_ns: int | None = None,
    ) -> RunRecord:
        resolved_run_id = run_id or f"run-{uuid4().hex}"
        _identifier(resolved_run_id, path="run_id")
        _identifier(environment_id, path="environment_id")
        _identifier(kind, path="run kind")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise ValueError("protocol_version cannot be empty")
        resolved_config_digest = _resolved_environment_config_digest(
            environment_config_snapshot, environment_config_digest
        )
        encoded_metadata, _ = _json_mapping(metadata or {}, path="run metadata")
        started = time_ns() if started_at_ns is None else started_at_ns
        _non_negative_integer(started, path="started_at_ns")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, environment_id, protocol_version, kind, status,
                    started_at_ns, finished_at_ns, exit_code, metadata_json,
                    environment_config_digest
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    resolved_run_id,
                    environment_id,
                    protocol_version,
                    kind,
                    RunStatus.RUNNING.value,
                    started,
                    encoded_metadata,
                    resolved_config_digest,
                ),
            )
        return self.get_run(resolved_run_id)

    def finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        exit_code: int | None,
        finished_at_ns: int | None = None,
    ) -> RunRecord:
        resolved_status = RunStatus(status)
        if resolved_status is RunStatus.RUNNING:
            raise ValueError("finish_run requires a terminal status")
        if exit_code is not None and (
            not isinstance(exit_code, int) or isinstance(exit_code, bool)
        ):
            raise TypeError("exit_code must be an integer or None")
        finished = time_ns() if finished_at_ns is None else finished_at_ns
        _non_negative_integer(finished, path="finished_at_ns")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = ?, finished_at_ns = ?, exit_code = ?
                WHERE run_id = ? AND status = ?
                """,
                (
                    resolved_status.value,
                    finished,
                    exit_code,
                    run_id,
                    RunStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ContractViolation("run is missing or already terminal")
            connection.execute("DELETE FROM run_state WHERE run_id = ?", (run_id,))
        return self.get_run(run_id)

    def run_state(
        self,
        run_id: str,
        namespace: str,
        *,
        schema_version: int,
    ) -> RunState:
        """Open adapter-owned state scoped to one active run.

        A namespace is created lazily. Existing state must be opened with the
        same schema version; a mismatch fails closed instead of returning data
        the adapter may misinterpret.
        """

        _identifier(run_id, path="run_id")
        resolved_namespace = _run_state_namespace(namespace)
        resolved_schema = _run_state_schema_version(schema_version)
        with self._connect() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown run_id: {run_id}")
            if run["status"] != RunStatus.RUNNING.value:
                raise ContractViolation("run state is available only while the run is running")
            row = connection.execute(
                "SELECT schema_version, state_json FROM run_state "
                "WHERE run_id = ? AND namespace = ?",
                (run_id, resolved_namespace),
            ).fetchone()
            if row is None:
                encoded, frozen = _run_state_mapping({})
                connection.execute(
                    "INSERT INTO run_state("
                    "run_id, namespace, schema_version, state_json, updated_at_ns"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (run_id, resolved_namespace, resolved_schema, encoded, time_ns()),
                )
            else:
                if row["schema_version"] != resolved_schema:
                    raise ContractViolation(
                        f"run state namespace {resolved_namespace!r} uses schema_version "
                        f"{row['schema_version']}, requested {resolved_schema}"
                    )
                try:
                    decoded = json.loads(row["state_json"])
                except (TypeError, json.JSONDecodeError) as error:
                    raise ContractViolation(
                        f"run state namespace {resolved_namespace!r} is corrupt"
                    ) from error
                if not isinstance(decoded, Mapping):
                    raise ContractViolation(
                        f"run state namespace {resolved_namespace!r} is not a JSON object"
                    )
                try:
                    _, frozen = _run_state_mapping(decoded)
                except (TypeError, ValueError) as error:
                    raise ContractViolation(
                        f"run state namespace {resolved_namespace!r} is corrupt"
                    ) from error
        return RunState(self, run_id, resolved_namespace, resolved_schema, frozen)

    def _write_run_state(
        self,
        run_id: str,
        namespace: str,
        schema_version: int,
        encoded: str,
    ) -> None:
        with self._connect() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown run_id: {run_id}")
            if run["status"] != RunStatus.RUNNING.value:
                raise ContractViolation("cannot write run state after the run is terminal")
            cursor = connection.execute(
                """
                UPDATE run_state
                SET state_json = ?, updated_at_ns = ?
                WHERE run_id = ? AND namespace = ? AND schema_version = ?
                """,
                (encoded, time_ns(), run_id, namespace, schema_version),
            )
            if cursor.rowcount != 1:
                raise ContractViolation(
                    f"run state namespace {namespace!r} disappeared or changed schema_version"
                )

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id: {run_id}")
        return self._run_from_row(row)

    def list_runs(
        self,
        *,
        environment_id: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> tuple[RunRecord, ...]:
        _limit(limit)
        clauses: list[str] = []
        parameters: list[object] = []
        if environment_id is not None:
            clauses.append("environment_id = ?")
            parameters.append(environment_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(RunStatus(status).value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM runs{where} ORDER BY started_at_ns DESC LIMIT ?",
                parameters,
            ).fetchall()
        return tuple(self._run_from_row(row) for row in rows)

    def list_environment_config_changes(
        self,
        *,
        environment_id: str,
        limit: int = 100,
    ) -> tuple[RunRecord, ...]:
        """Return runs whose environment configuration differs from the prior run.

        Results are ordered by campaign order (oldest first), which makes the
        first returned record the run at which a changed configuration became
        observable. A missing digest is treated as an unknown configuration and
        never returned as a change by itself.
        """

        _identifier(environment_id, path="environment_id")
        _limit(limit)
        runs = tuple(reversed(self.list_runs(environment_id=environment_id, limit=1000)))
        changes: list[RunRecord] = []
        previous: str | None = None
        have_previous = False
        for run in runs:
            current = run.environment_config_digest
            if have_previous and current is not None and current != previous:
                changes.append(run)
            previous = current
            have_previous = True
        return tuple(changes[-limit:])

    def query_environment_config_changes(
        self,
        *,
        environment_id: str,
        limit: int = 100,
    ) -> tuple[RunRecord, ...]:
        """Alias for :meth:`list_environment_config_changes`."""

        return self.list_environment_config_changes(environment_id=environment_id, limit=limit)

    def append_event(
        self,
        run_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
        episode_id: str | None = None,
        step_id: int | None = None,
        timestamp_ns: int | None = None,
    ) -> RunEvent:
        _identifier(kind, path="event kind")
        _optional_identifier(episode_id, path="episode_id")
        if step_id is not None and (
            not isinstance(step_id, int) or isinstance(step_id, bool) or step_id < 0
        ):
            raise ValueError("step_id must be a non-negative integer or None")
        encoded_payload, frozen_payload = _json_mapping(payload, path="event payload")
        timestamp = time_ns() if timestamp_ns is None else timestamp_ns
        _non_negative_integer(timestamp, path="timestamp_ns")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            status_row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if status_row is None:
                raise KeyError(f"unknown run_id: {run_id}")
            if status_row["status"] != RunStatus.RUNNING.value:
                raise ContractViolation("cannot append an event to a terminal run")
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence_id), 0) + 1 FROM events WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0]
            )
            connection.execute(
                """
                INSERT INTO events(
                    run_id, sequence_id, timestamp_ns, kind, episode_id, step_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, sequence, timestamp, kind, episode_id, step_id, encoded_payload),
            )
        return RunEvent(
            run_id=run_id,
            sequence_id=sequence,
            timestamp_ns=timestamp,
            kind=kind,
            episode_id=episode_id,
            step_id=step_id,
            payload=frozen_payload,
        )

    def list_events(self, run_id: str, *, limit: int = 1000) -> tuple[RunEvent, ...]:
        _limit(limit)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM events WHERE run_id = ?
                ORDER BY sequence_id ASC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return tuple(
            RunEvent(
                run_id=row["run_id"],
                sequence_id=row["sequence_id"],
                timestamp_ns=row["timestamp_ns"],
                kind=row["kind"],
                episode_id=row["episode_id"],
                step_id=row["step_id"],
                payload=MappingProxyType(json.loads(row["payload_json"])),
            )
            for row in rows
        )

    def record_metric(
        self,
        run_id: str,
        *,
        name: str,
        value: float,
        step_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        environment_config_snapshot: EnvironmentConfigSnapshot | None = None,
        environment_config_digest: str | None = None,
        timestamp_ns: int | None = None,
    ) -> MetricRecord:
        _identifier(name, path="metric name")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("metric value must be finite")
        if step_id is not None and (
            not isinstance(step_id, int) or isinstance(step_id, bool) or step_id < 0
        ):
            raise ValueError("step_id must be a non-negative integer or None")
        resolved_config_digest = _resolved_environment_config_digest(
            environment_config_snapshot, environment_config_digest
        )
        encoded_metadata, frozen_metadata = _json_mapping(metadata or {}, path="metric metadata")
        timestamp = time_ns() if timestamp_ns is None else timestamp_ns
        _non_negative_integer(timestamp, path="timestamp_ns")
        with self._connect() as connection:
            status_row = connection.execute(
                "SELECT status, environment_config_digest FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if status_row is None:
                raise KeyError(f"unknown run_id: {run_id}")
            if status_row["status"] != RunStatus.RUNNING.value:
                raise ContractViolation("cannot record a metric for a terminal run")
            if resolved_config_digest is None:
                resolved_config_digest = status_row["environment_config_digest"]
            cursor = connection.execute(
                """
                INSERT INTO metrics(
                    run_id, timestamp_ns, name, value, step_id, metadata_json,
                    environment_config_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    timestamp,
                    name,
                    numeric,
                    step_id,
                    encoded_metadata,
                    resolved_config_digest,
                ),
            )
            if cursor.lastrowid is None:
                raise ContractViolation("SQLite did not return a metric identifier")
            metric_id = int(cursor.lastrowid)
        return MetricRecord(
            run_id=run_id,
            metric_id=metric_id,
            timestamp_ns=timestamp,
            name=name,
            value=numeric,
            step_id=step_id,
            metadata=frozen_metadata,
            environment_config_digest=resolved_config_digest,
        )

    def list_metrics(self, run_id: str, *, limit: int = 1000) -> tuple[MetricRecord, ...]:
        _limit(limit)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM metrics WHERE run_id = ?
                ORDER BY metric_id ASC LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return tuple(
            MetricRecord(
                run_id=row["run_id"],
                metric_id=row["metric_id"],
                timestamp_ns=row["timestamp_ns"],
                name=row["name"],
                value=row["value"],
                step_id=row["step_id"],
                metadata=MappingProxyType(json.loads(row["metadata_json"])),
                environment_config_digest=row["environment_config_digest"],
            )
            for row in rows
        )

    def latest_metric_id(self, run_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(metric_id) FROM metrics WHERE run_id = ?", (run_id,)
            ).fetchone()
        return 0 if row is None or row[0] is None else int(row[0])

    def has_metric_evidence(
        self,
        run_id: str,
        *,
        after_metric_id: int,
        name: str,
        value: float,
        source: str,
        authority: KnowledgeAuthority,
    ) -> bool:
        _non_negative_integer(after_metric_id, path="after_metric_id")
        _identifier(name, path="metric name")
        expected = float(value)
        if not math.isfinite(expected):
            raise ValueError("metric evidence value must be finite")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT value FROM metrics
                WHERE run_id = ? AND metric_id > ? AND name = ?
                  AND json_extract(metadata_json, '$.source') = ?
                  AND json_extract(metadata_json, '$.authority') = ?
                """,
                (run_id, after_metric_id, name, source, KnowledgeAuthority(authority).value),
            ).fetchall()
        return any(
            math.isclose(row["value"], expected, rel_tol=1e-9, abs_tol=1e-12) for row in rows
        )

    def upsert_entity(self, entity: SpatialEntity) -> SpatialEntity:
        if not isinstance(entity, SpatialEntity):
            raise TypeError("entity must be a SpatialEntity")
        metadata_json, _ = _json_mapping(entity.metadata, path="entity metadata")
        with self._connect() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (entity.source_run_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"unknown source_run_id: {entity.source_run_id}")
            connection.execute(
                """
                INSERT INTO spatial_entities(
                    environment_id, world_id, entity_id, kind, label, x, y, z,
                    coordinate_frame, authority, confidence, observed_at_ns,
                    source_run_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(environment_id, world_id, entity_id) DO UPDATE SET
                    kind = excluded.kind,
                    label = excluded.label,
                    x = excluded.x,
                    y = excluded.y,
                    z = excluded.z,
                    coordinate_frame = excluded.coordinate_frame,
                    authority = excluded.authority,
                    confidence = excluded.confidence,
                    observed_at_ns = excluded.observed_at_ns,
                    source_run_id = excluded.source_run_id,
                    metadata_json = excluded.metadata_json
                WHERE excluded.observed_at_ns >= spatial_entities.observed_at_ns
                """,
                (
                    entity.environment_id,
                    entity.world_id,
                    entity.entity_id,
                    entity.kind,
                    entity.label,
                    *entity.position,
                    entity.coordinate_frame,
                    entity.authority.value,
                    entity.confidence,
                    entity.observed_at_ns,
                    entity.source_run_id,
                    metadata_json,
                ),
            )
        return entity

    def register_artifact(
        self,
        run_id: str,
        *,
        path: str,
        source: str | Path,
        role: str,
        media_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        relative = _portable_path(path, path="artifact path")
        _identifier(role, path="artifact role")
        if (
            not isinstance(media_type, str)
            or not media_type
            or len(media_type) > 128
            or any(ord(character) < 32 for character in media_type)
        ):
            raise ValueError("artifact media_type must be non-empty printable text")
        candidate = Path(source)
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError("artifact source must be an existing regular non-symlink file")
        encoded_metadata, frozen_metadata = _json_mapping(metadata or {}, path="artifact metadata")
        artifact = ArtifactRecord(
            run_id=run_id,
            path=relative,
            role=role,
            media_type=media_type,
            sha256=_file_sha256(candidate),
            size_bytes=candidate.stat().st_size,
            metadata=frozen_metadata,
        )
        with self._connect() as connection:
            if (
                connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
                is None
            ):
                raise KeyError(f"unknown run_id: {run_id}")
            connection.execute(
                """
                INSERT INTO artifacts(
                    run_id, path, role, media_type, sha256, size_bytes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, path) DO UPDATE SET
                    role = excluded.role,
                    media_type = excluded.media_type,
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    metadata_json = excluded.metadata_json
                """,
                (
                    artifact.run_id,
                    artifact.path,
                    artifact.role,
                    artifact.media_type,
                    artifact.sha256,
                    artifact.size_bytes,
                    encoded_metadata,
                ),
            )
        return artifact

    def list_artifacts(self, run_id: str) -> tuple[ArtifactRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY path ASC", (run_id,)
            ).fetchall()
        return tuple(
            ArtifactRecord(
                run_id=row["run_id"],
                path=row["path"],
                role=row["role"],
                media_type=row["media_type"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                metadata=MappingProxyType(json.loads(row["metadata_json"])),
            )
            for row in rows
        )

    def query_entities(
        self,
        *,
        environment_id: str,
        world_id: str,
        kind: str | None = None,
        name: str | None = None,
        near: Sequence[float] | None = None,
        radius: float | None = None,
        limit: int = 100,
    ) -> tuple[SpatialEntity, ...]:
        clauses = ["environment_id = ?", "world_id = ?"]
        parameters: list[object] = [environment_id, world_id]
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(kind)
        if name is not None:
            clauses.append("label LIKE ?")
            parameters.append(f"%{name}%")
        order = "observed_at_ns DESC, entity_id ASC"
        if near is not None:
            x, y, z = _position(near, path="near")
            if radius is None:
                raise ValueError("radius is required when near is provided")
            resolved_radius = float(radius)
            if not math.isfinite(resolved_radius) or resolved_radius <= 0:
                raise ValueError("radius must be positive and finite")
            distance = "((x - ?) * (x - ?) + (y - ?) * (y - ?) + (z - ?) * (z - ?))"
            clauses.append(f"{distance} <= ?")
            parameters.extend((x, x, y, y, z, z, resolved_radius * resolved_radius))
            order = f"{distance} ASC, entity_id ASC"
            parameters.extend((x, x, y, y, z, z))
        elif radius is not None:
            raise ValueError("near is required when radius is provided")
        _limit(limit)
        parameters.append(limit)
        query = (
            "SELECT * FROM spatial_entities WHERE "
            + " AND ".join(clauses)
            + f" ORDER BY {order} LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._entity_from_row(row) for row in rows)

    def list_entities(self, *, environment_id: str) -> tuple[SpatialEntity, ...]:
        _identifier(environment_id, path="environment_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM spatial_entities WHERE environment_id = ?
                ORDER BY world_id ASC, entity_id ASC
                """,
                (environment_id,),
            ).fetchall()
        return tuple(self._entity_from_row(row) for row in rows)

    def record_route(self, route: SpatialRoute) -> SpatialRoute:
        if not isinstance(route, SpatialRoute):
            raise TypeError("route must be a SpatialRoute")
        metadata_json, _ = _json_mapping(route.metadata, path="route metadata")
        with self._connect() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (route.source_run_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"unknown source_run_id: {route.source_run_id}")
            connection.execute(
                """
                INSERT INTO spatial_routes(
                    environment_id, world_id, route_id, name, from_entity_id,
                    to_entity_id, coordinate_frame, confidence, verified_at_ns,
                    source_run_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(environment_id, world_id, route_id) DO UPDATE SET
                    name = excluded.name,
                    from_entity_id = excluded.from_entity_id,
                    to_entity_id = excluded.to_entity_id,
                    coordinate_frame = excluded.coordinate_frame,
                    confidence = excluded.confidence,
                    verified_at_ns = excluded.verified_at_ns,
                    source_run_id = excluded.source_run_id,
                    metadata_json = excluded.metadata_json
                WHERE excluded.verified_at_ns >= spatial_routes.verified_at_ns
                """,
                (
                    route.environment_id,
                    route.world_id,
                    route.route_id,
                    route.name,
                    route.from_entity_id,
                    route.to_entity_id,
                    route.coordinate_frame,
                    route.confidence,
                    route.verified_at_ns,
                    route.source_run_id,
                    metadata_json,
                ),
            )
            connection.execute(
                """
                DELETE FROM route_waypoints
                WHERE environment_id = ? AND world_id = ? AND route_id = ?
                """,
                (route.environment_id, route.world_id, route.route_id),
            )
            connection.executemany(
                """
                INSERT INTO route_waypoints(
                    environment_id, world_id, route_id, waypoint_index,
                    x, y, z, tolerance, label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        route.environment_id,
                        route.world_id,
                        route.route_id,
                        waypoint.index,
                        *waypoint.position,
                        waypoint.tolerance,
                        waypoint.label,
                    )
                    for waypoint in route.waypoints
                ),
            )
        return route

    def query_routes(
        self,
        *,
        environment_id: str,
        world_id: str,
        from_entity_id: str | None = None,
        to_entity_id: str | None = None,
        limit: int = 100,
    ) -> tuple[SpatialRoute, ...]:
        _limit(limit)
        clauses = ["environment_id = ?", "world_id = ?"]
        parameters: list[object] = [environment_id, world_id]
        if from_entity_id is not None:
            clauses.append("from_entity_id = ?")
            parameters.append(from_entity_id)
        if to_entity_id is not None:
            clauses.append("to_entity_id = ?")
            parameters.append(to_entity_id)
        parameters.append(limit)
        query = (
            "SELECT * FROM spatial_routes WHERE "
            + " AND ".join(clauses)
            + " ORDER BY confidence DESC, verified_at_ns DESC, route_id ASC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
            routes = tuple(self._route_from_row(connection, row) for row in rows)
        return routes

    def list_routes(self, *, environment_id: str) -> tuple[SpatialRoute, ...]:
        _identifier(environment_id, path="environment_id")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM spatial_routes WHERE environment_id = ?
                ORDER BY world_id ASC, route_id ASC
                """,
                (environment_id,),
            ).fetchall()
            return tuple(self._route_from_row(connection, row) for row in rows)

    def upsert_research_bundle(self, bundle: ResearchBundle) -> ResearchBundle:
        """Persist validated findings without making them executable action authority."""

        if not isinstance(bundle, ResearchBundle):
            raise TypeError("bundle must be a ResearchBundle")
        with self._connect() as connection:
            for source in bundle.sources:
                connection.execute(
                    """
                    INSERT INTO research_sources(source_id, media_type, accessed_at, source_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        media_type = excluded.media_type,
                        accessed_at = excluded.accessed_at,
                        source_json = excluded.source_json
                    """,
                    (
                        source.source_id,
                        source.media_type.value,
                        source.accessed_at.isoformat(),
                        json.dumps(source.to_mapping(), sort_keys=True, separators=(",", ":")),
                    ),
                )
            for finding in bundle.findings:
                connection.execute(
                    """
                    INSERT INTO research_findings(
                        finding_id, category, status, scope, scope_id, finding_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(finding_id) DO UPDATE SET
                        category = excluded.category,
                        status = excluded.status,
                        scope = excluded.scope,
                        scope_id = excluded.scope_id,
                        finding_json = excluded.finding_json
                    """,
                    (
                        finding.finding_id,
                        finding.category.value,
                        finding.status.value,
                        finding.scope.value,
                        finding.scope_id,
                        json.dumps(finding.to_mapping(), sort_keys=True, separators=(",", ":")),
                    ),
                )
                connection.execute(
                    "DELETE FROM research_finding_sources WHERE finding_id = ?",
                    (finding.finding_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO research_finding_sources(finding_id, source_id, ordinal)
                    VALUES (?, ?, ?)
                    """,
                    (
                        (finding.finding_id, source_id, ordinal)
                        for ordinal, source_id in enumerate(finding.source_ids)
                    ),
                )
                connection.execute(
                    "DELETE FROM research_finding_tags WHERE finding_id = ?",
                    (finding.finding_id,),
                )
                connection.executemany(
                    "INSERT INTO research_finding_tags(finding_id, tag) VALUES (?, ?)",
                    ((finding.finding_id, tag) for tag in finding.tags),
                )
        return bundle

    def query_research(
        self,
        *,
        environment_id: str,
        environment_family: str,
        tags: Sequence[str] = (),
        category: ResearchCategory | None = None,
        include_unverified: bool = True,
        limit: int = 100,
    ) -> tuple[ResearchFinding, ...]:
        """Query exact-environment, same-family, and generic non-rejected findings."""

        _identifier(environment_id, path="environment_id")
        _identifier(environment_family, path="environment_family")
        resolved_tags = tuple(_identifier(tag, path="research tag") for tag in tags)
        if len(set(resolved_tags)) != len(resolved_tags):
            raise ValueError("research tags must be unique")
        _limit(limit)
        clauses = [
            "status != ?",
            "((scope = ? AND scope_id = ?) OR (scope = ? AND scope_id = ?) OR scope = ?)",
        ]
        parameters: list[object] = [
            ResearchStatus.REJECTED.value,
            ResearchScope.ENVIRONMENT.value,
            environment_id,
            ResearchScope.FAMILY.value,
            environment_family,
            ResearchScope.GENERIC.value,
        ]
        if not include_unverified:
            clauses.append("status = ?")
            parameters.append(ResearchStatus.RUNTIME_VERIFIED.value)
        if category is not None:
            clauses.append("category = ?")
            parameters.append(ResearchCategory(category).value)
        for tag in resolved_tags:
            clauses.append(
                "EXISTS (SELECT 1 FROM research_finding_tags AS tags "
                "WHERE tags.finding_id = research_findings.finding_id AND tags.tag = ?)"
            )
            parameters.append(tag)
        parameters.append(limit)
        query = (
            "SELECT finding_json FROM research_findings WHERE "
            + " AND ".join(clauses)
            + " ORDER BY CASE status WHEN 'runtime-verified' THEN 0 ELSE 1 END, "
            "finding_id ASC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(ResearchFinding.from_mapping(json.loads(row["finding_json"])) for row in rows)

    def get_research_sources(self, finding: ResearchFinding) -> tuple[ResearchSource, ...]:
        if not isinstance(finding, ResearchFinding):
            raise TypeError("finding must be a ResearchFinding")
        sources: list[ResearchSource] = []
        with self._connect() as connection:
            for source_id in finding.source_ids:
                row = connection.execute(
                    "SELECT source_json FROM research_sources WHERE source_id = ?",
                    (source_id,),
                ).fetchone()
                if row is None:
                    raise ContractViolation(
                        f"research finding references missing source: {source_id}"
                    )
                sources.append(ResearchSource.from_mapping(json.loads(row["source_json"])))
        return tuple(sources)

    @staticmethod
    def _entity_from_row(row: sqlite3.Row) -> SpatialEntity:
        return SpatialEntity(
            environment_id=row["environment_id"],
            world_id=row["world_id"],
            entity_id=row["entity_id"],
            kind=row["kind"],
            label=row["label"],
            position=(row["x"], row["y"], row["z"]),
            coordinate_frame=row["coordinate_frame"],
            authority=KnowledgeAuthority(row["authority"]),
            confidence=row["confidence"],
            observed_at_ns=row["observed_at_ns"],
            source_run_id=row["source_run_id"],
            metadata=MappingProxyType(json.loads(row["metadata_json"])),
        )

    @staticmethod
    def _route_from_row(connection: sqlite3.Connection, row: sqlite3.Row) -> SpatialRoute:
        waypoint_rows = connection.execute(
            """
            SELECT * FROM route_waypoints
            WHERE environment_id = ? AND world_id = ? AND route_id = ?
            ORDER BY waypoint_index ASC
            """,
            (row["environment_id"], row["world_id"], row["route_id"]),
        ).fetchall()
        return SpatialRoute(
            environment_id=row["environment_id"],
            world_id=row["world_id"],
            route_id=row["route_id"],
            name=row["name"],
            from_entity_id=row["from_entity_id"],
            to_entity_id=row["to_entity_id"],
            coordinate_frame=row["coordinate_frame"],
            confidence=row["confidence"],
            verified_at_ns=row["verified_at_ns"],
            source_run_id=row["source_run_id"],
            waypoints=tuple(
                RouteWaypoint(
                    index=waypoint["waypoint_index"],
                    position=(waypoint["x"], waypoint["y"], waypoint["z"]),
                    tolerance=waypoint["tolerance"],
                    label=waypoint["label"],
                )
                for waypoint in waypoint_rows
            ),
            metadata=MappingProxyType(json.loads(row["metadata_json"])),
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            environment_id=row["environment_id"],
            protocol_version=row["protocol_version"],
            kind=row["kind"],
            status=RunStatus(row["status"]),
            started_at_ns=row["started_at_ns"],
            finished_at_ns=row["finished_at_ns"],
            exit_code=row["exit_code"],
            metadata=MappingProxyType(json.loads(row["metadata_json"])),
            environment_config_digest=row["environment_config_digest"],
        )


__all__ = [
    "RUN_STORE_SCHEMA_VERSION",
    "ArtifactRecord",
    "MetricRecord",
    "RouteWaypoint",
    "RunEvent",
    "RunRecord",
    "RunState",
    "RunStatus",
    "SpatialEntity",
    "SpatialRoute",
    "TrainingStore",
]
