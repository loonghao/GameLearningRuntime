"""Finite local spool reception, with an owned schema-1 index and no services."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from game_learning_runtime.fleet_measured import (
    MeasuredAuthority,
    VerifiedMeasuredShard,
    verify_measured,
    verify_measured_manifest,
)
from game_learning_runtime.fleet_payload import (
    DEFAULT_LIMITS,
    DecodedShard,
    EncodedShard,
    FleetError,
    FleetLimits,
    SourceSpec,
    canonical,
    closed,
    counter,
    decode_shard,
    digest,
    identifier,
    parse_manifest,
    sha256,
)
from game_learning_runtime.offline_parsing import OfflineParseError, parse_json_object
from game_learning_runtime.serialization import transition_to_record

if TYPE_CHECKING:
    from game_learning_runtime.fleet_learner import RealTrainingEnablement


def utc(value: int | None) -> str | None:
    if value is None:
        return None
    try:
        return (
            datetime.fromtimestamp(value / 1000, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    except (ValueError, OverflowError, OSError):
        raise FleetError("invalid_utc") from None


def _regular(path: Path, limit: int) -> bytes:
    try:
        return _read_regular(path, limit)
    except OSError:
        raise FleetError("artifact_unavailable") from None


def _read_regular(path: Path, limit: int) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or getattr(before, "st_file_attributes", 0) & 0x400:
        raise FleetError("nonregular_artifact")
    if before.st_size > limit:
        raise FleetError("artifact_byte_limit")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        payload = stream.read(limit + 1)
    after = path.lstat()
    if (
        (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
        or before.st_size != len(payload)
        or len(payload) > limit
    ):
        raise FleetError("artifact_changed")
    return payload


def _directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError:
        raise FleetError("unsafe_directory") from None
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise FleetError("unsafe_directory")


def _atomic(path: Path, payload: bytes) -> None:
    _directory(path.parent)
    temporary = path.with_name(f".pending-{uuid4().hex}")
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json(payload: bytes, limit: int) -> dict[str, Any]:
    try:
        return dict(parse_json_object(payload, max_bytes=limit).data)
    except OfflineParseError:
        raise FleetError("invalid_spool_json") from None


@dataclass(frozen=True, slots=True)
class UploadReceipt:
    shard_id: str
    status: str
    next_chunk_index: int
    duplicate: bool = False


@dataclass(frozen=True, slots=True)
class IngestReceipt:
    shard_id: str
    status: str
    transition_count: int
    duplicate: bool
    trust_level: str


@dataclass(frozen=True, slots=True)
class SyncReceipt:
    scanned_shards: int
    completed_shards: int
    duplicate_shards: int
    quarantined_shards: int
    conflicts: int = 0


@dataclass(frozen=True, slots=True)
class MeasuredDestination:
    """An owner named local destination, not a network address or authentication."""

    destination_id: str
    destination_sha256: str

    def __post_init__(self) -> None:
        from game_learning_runtime.fleet_payload import identifier

        identifier(self.destination_id)
        digest(self.destination_sha256)


@dataclass(frozen=True, slots=True)
class MeasuredEvaluationSnapshot:
    evaluation_id: str
    suite_sha256: str
    snapshot_sha256: str
    source_spec_sha256s: tuple[str, ...]
    shard_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeasuredEvaluationCase:
    case_id: str
    source_id: str
    source_epoch: str
    shard_id: str
    payload_sha256: str
    proof_sha256: str

    def __post_init__(self) -> None:
        for name in ("case_id", "source_id", "source_epoch"):
            identifier(getattr(self, name))
        for name in ("shard_id", "payload_sha256", "proof_sha256"):
            digest(getattr(self, name))


@dataclass(frozen=True, slots=True)
class MeasuredEvaluationMetric:
    name: str
    aggregation: str
    direction: str
    require_count: int

    def __post_init__(self) -> None:
        identifier(self.name)
        if (
            not isinstance(self.aggregation, str)
            or not isinstance(self.direction, str)
            or self.aggregation not in {"sum", "mean", "min", "max"}
            or self.direction not in {"maximize", "minimize"}
        ):
            raise FleetError("invalid_measured_metric")
        counter(self.require_count, minimum=1, maximum=131072)


@dataclass(frozen=True, slots=True)
class MeasuredEvaluationSuite:
    """Immutable evaluation definition and artifacts; never executed by the hub."""

    suite_id: str
    evaluation_domain_id: str
    evidence_kind: str
    evaluator_source_commit: str
    evaluator_artifact: bytes = field(repr=False)
    cases: tuple[MeasuredEvaluationCase, ...]
    metrics: tuple[MeasuredEvaluationMetric, ...]

    def __post_init__(self) -> None:
        identifier(self.suite_id)
        identifier(self.evaluation_domain_id)
        digest(self.evaluator_source_commit, 40)
        if not isinstance(self.evidence_kind, str) or self.evidence_kind not in {
            "synthetic_contract_fixture",
            "owner_authorized_local_measured",
        }:
            raise FleetError("invalid_measured_evidence_kind")
        if (
            not isinstance(self.evaluator_artifact, bytes)
            or not 1 <= len(self.evaluator_artifact) <= 1048576
        ):
            raise FleetError("measured_evaluator_byte_limit")
        if (
            not isinstance(self.cases, tuple)
            or not 1 <= len(self.cases) <= 1024
            or any(type(item) is not MeasuredEvaluationCase for item in self.cases)
            or len({item.case_id for item in self.cases}) != len(self.cases)
            or len({item.shard_id for item in self.cases}) != len(self.cases)
        ):
            raise FleetError("invalid_measured_evaluation_cases")
        if (
            not isinstance(self.metrics, tuple)
            or not 1 <= len(self.metrics) <= 32
            or any(type(item) is not MeasuredEvaluationMetric for item in self.metrics)
            or len({item.name for item in self.metrics}) != len(self.metrics)
        ):
            raise FleetError("invalid_measured_evaluation_metrics")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "glr.fleet.measured-evaluation-suite.v1",
            "suite_id": self.suite_id,
            "evaluation_domain_id": self.evaluation_domain_id,
            "evidence_kind": self.evidence_kind,
            "evaluator_source_commit": self.evaluator_source_commit,
            "evaluator_artifact_sha256": sha256(self.evaluator_artifact),
            "evaluator_artifact_bytes": len(self.evaluator_artifact),
            "cases": [asdict(item) for item in self.cases],
            "metrics": [asdict(item) for item in self.metrics],
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical(self.to_record()))

    @classmethod
    def from_record(cls, value: object, artifact: bytes) -> MeasuredEvaluationSuite:
        record = closed(
            value,
            {
                "schema",
                "suite_id",
                "evaluation_domain_id",
                "evidence_kind",
                "evaluator_source_commit",
                "evaluator_artifact_sha256",
                "evaluator_artifact_bytes",
                "cases",
                "metrics",
            },
        )
        if (
            record["schema"] != "glr.fleet.measured-evaluation-suite.v1"
            or digest(record["evaluator_artifact_sha256"]) != sha256(artifact)
            or counter(record["evaluator_artifact_bytes"], minimum=1, maximum=1048576)
            != len(artifact)
        ):
            raise FleetError("measured_evaluator_integrity")
        if (
            not isinstance(record["cases"], (list, tuple))
            or not 1 <= len(record["cases"]) <= 1024
            or not isinstance(record["metrics"], (list, tuple))
            or not 1 <= len(record["metrics"]) <= 32
        ):
            raise FleetError("invalid_measured_evaluation_suite")
        return cls(
            record["suite_id"],
            record["evaluation_domain_id"],
            record["evidence_kind"],
            record["evaluator_source_commit"],
            artifact,
            tuple(
                MeasuredEvaluationCase(
                    **closed(item, set(MeasuredEvaluationCase.__dataclass_fields__))
                )
                for item in record["cases"]
            ),
            tuple(
                MeasuredEvaluationMetric(
                    **closed(item, set(MeasuredEvaluationMetric.__dataclass_fields__))
                )
                for item in record["metrics"]
            ),
        )


class FleetHub:
    """Owns only its directory, receipts and bounded payloads, never learner state."""

    def __init__(
        self, root: Path, limits: FleetLimits, clock_ms: Callable[[], int], owner_epoch: str
    ) -> None:
        self.root = root
        self.database = root / "index.sqlite3"
        self.limits = limits
        self._clock = clock_ms
        self._closed = False
        self._owner_epoch = owner_epoch
        self._measured_authority: MeasuredAuthority | None = None
        self._measured_destination: MeasuredDestination | None = None

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        limits: FleetLimits = DEFAULT_LIMITS,
        clock_ms: Callable[[], int] = lambda: time_ns() // 1_000_000,
        measured_authority: MeasuredAuthority | None = None,
        measured_destination: MeasuredDestination | None = None,
    ) -> FleetHub:
        root = Path(root).absolute()
        if type(limits) is not FleetLimits:
            raise FleetError("invalid_limits")
        if root.exists() and (root.is_symlink() or any(root.iterdir())):
            raise FleetError("unowned_directory")
        root.mkdir(parents=True, exist_ok=True)
        _directory(root)
        (root / "artifacts").mkdir()
        epoch = uuid4().hex
        _atomic(
            root / "hub-owner.json", canonical({"schema": "glr.fleet.owner.v1", "epoch": epoch})
        )
        hub = cls(root, limits, clock_ms, epoch)
        connection = sqlite3.connect(hub.database)
        try:
            connection.executescript("""
                PRAGMA user_version=1;
                CREATE TABLE hub_identity(epoch TEXT PRIMARY KEY, limits_json
                TEXT NOT NULL);
                CREATE TABLE sources(
                    source_id TEXT NOT NULL, source_epoch TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0, heartbeat_seq
                    INTEGER NOT NULL DEFAULT -1,
                    heartbeat_declared_ms INTEGER, heartbeat_received_ms
                    INTEGER,
                    declared_status TEXT NOT NULL DEFAULT 'unknown',
                    last_shard_seq INTEGER NOT NULL DEFAULT -1,
                    data_observed_ms INTEGER, data_received_ms INTEGER,
                    last_plan_eligible INTEGER,
                    last_plan_reasons TEXT NOT NULL DEFAULT '[]',
                    eval_frozen INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(source_id,source_epoch));
                CREATE TABLE run_assignments(run_id TEXT PRIMARY KEY,
                assignment_sha256 TEXT NOT NULL);
                CREATE TABLE shards(
                    shard_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                    source_epoch TEXT NOT NULL,
                    shard_seq INTEGER NOT NULL, manifest BLOB NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL, payload_bytes INTEGER NOT
                    NULL, transition_count INTEGER NOT NULL,
                    status TEXT NOT NULL, next_chunk_index INTEGER NOT NULL
                    DEFAULT 0,
                    received_ms INTEGER, duplicate_count INTEGER NOT NULL
                    DEFAULT 0,
                    retained_bytes INTEGER NOT NULL,
                    UNIQUE(source_id,source_epoch,shard_seq));
                CREATE TABLE plans(plan_id TEXT PRIMARY KEY, selection_sha256
                TEXT NOT NULL, shards_json TEXT NOT NULL);
                CREATE TABLE transition_reservations(
                    episode_id TEXT NOT NULL, step_id INTEGER NOT NULL,
                    record_sha256 TEXT NOT NULL,
                    input_sha256 TEXT NOT NULL, cohort_sha256 TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    split TEXT NOT NULL, shard_id TEXT NOT NULL, PRIMARY
                    KEY(episode_id,step_id));
                CREATE INDEX transition_inputs ON
                transition_reservations(cohort_sha256,input_sha256);
                CREATE TABLE holdout_inputs(cohort_sha256 TEXT NOT
                NULL,input_sha256 TEXT NOT NULL,
                    shard_id TEXT NOT NULL,PRIMARY
                    KEY(cohort_sha256,input_sha256));
                CREATE TABLE episode_tails(
                    episode_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_epoch TEXT NOT NULL, cohort_sha256 TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    timestamp_ns INTEGER NOT NULL, next_observation_sha256
                    TEXT NOT NULL,
                    done INTEGER NOT NULL);
                CREATE TABLE consumptions(
                    receipt_id TEXT PRIMARY KEY, ticket_id TEXT UNIQUE NOT
                    NULL, plan_id TEXT NOT NULL,
                    learner_id TEXT NOT NULL, shard_id TEXT UNIQUE NOT NULL,
                    source_id TEXT NOT NULL,
                    source_epoch TEXT NOT NULL, status TEXT NOT NULL,
                    transition_count INTEGER NOT NULL DEFAULT 0,
                    callback_completed INTEGER NOT NULL DEFAULT 0,
                    declared_updates INTEGER, finished_ms INTEGER);
""")
            connection.execute(
                "INSERT INTO hub_identity VALUES(?,?)", (epoch, canonical(asdict(limits)).decode())
            )
            connection.commit()
        finally:
            connection.close()
        hub.configure_measured(measured_authority, destination=measured_destination)
        return hub

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        clock_ms: Callable[[], int] = lambda: time_ns() // 1_000_000,
        measured_authority: MeasuredAuthority | None = None,
        measured_destination: MeasuredDestination | None = None,
    ) -> FleetHub:
        root = Path(root).absolute()
        _directory(root)
        _directory(root / "artifacts")
        owner = _json(_regular(root / "hub-owner.json", 1024), 1024)
        if set(owner) != {"schema", "epoch"} or owner["schema"] != "glr.fleet.owner.v1":
            raise FleetError("unowned_directory")
        database_info = (root / "index.sqlite3").lstat()
        if (
            not stat.S_ISREG(database_info.st_mode)
            or getattr(database_info, "st_file_attributes", 0) & 0x400
        ):
            raise FleetError("nonregular_database")
        connection = sqlite3.connect((root / "index.sqlite3").as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise FleetError("unsupported_fleet_schema")
            row = connection.execute(
                "SELECT epoch,limits_json FROM hub_identity LIMIT 2"
            ).fetchall()
            if len(row) != 1 or row[0][0] != owner["epoch"]:
                raise FleetError("hub_identity_mismatch")
            limits = FleetLimits(**_json(row[0][1].encode(), 4096))
            if "eval_frozen" not in {
                item[1] for item in connection.execute("PRAGMA table_info(sources)")
            }:
                raise FleetError("unsupported_fleet_schema")
            if (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='episode_tails'"
                ).fetchone()
                is None
            ):
                raise FleetError("unsupported_fleet_schema")
        finally:
            connection.close()
        hub = cls(root, limits, clock_ms, owner["epoch"])
        hub.configure_measured(measured_authority, destination=measured_destination)
        return hub

    @property
    def measured_authority(self) -> MeasuredAuthority | None:
        return self._measured_authority

    @property
    def measured_destination(self) -> MeasuredDestination | None:
        return self._measured_destination

    def configure_measured(
        self,
        authority: MeasuredAuthority | None,
        *,
        destination: MeasuredDestination | None = None,
    ) -> None:
        """Replace caller supplied RAM trust; restart deliberately forgets it.

        This grants proof verification, never permission to invoke a learner.
        Passing None immediately closes real admission. No keys are persisted.
        """
        if authority is not None and type(authority) is not MeasuredAuthority:
            raise FleetError("invalid_measured_authority")
        if destination is not None and type(destination) is not MeasuredDestination:
            raise FleetError("invalid_measured_destination")
        if authority is not None:
            with self._connection(write=True) as connection:
                # An optional extension leaves v1 database/open semantics intact.
                for statement in (
                    "CREATE TABLE IF NOT EXISTS measured_proofs(shard_id TEXT PRIMARY KEY,"
                    "envelope BLOB NOT NULL,envelope_sha256 TEXT NOT NULL,"
                    "authority_sha256 TEXT NOT NULL,expires_ms INTEGER NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS measured_reward_tails(episode_id TEXT PRIMARY "
                    "KEY,tail_json TEXT NOT NULL,shard_id TEXT NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS measured_actions(run_id TEXT NOT NULL,"
                    "target_id TEXT NOT NULL,action_id TEXT NOT NULL,shard_id TEXT NOT NULL,"
                    "PRIMARY KEY(run_id,target_id,action_id))",
                    "CREATE TABLE IF NOT EXISTS measured_holdout_inputs(domain_id TEXT NOT "
                    "NULL,input_sha256 TEXT NOT NULL,shard_id TEXT NOT NULL,"
                    "PRIMARY KEY(domain_id,input_sha256))",
                    "CREATE TABLE IF NOT EXISTS measured_inputs(domain_id TEXT NOT NULL,"
                    "input_sha256 TEXT NOT NULL,shard_id TEXT NOT NULL,"
                    "PRIMARY KEY(domain_id,input_sha256,shard_id))",
                    "CREATE TABLE IF NOT EXISTS measured_domains(source_spec_sha256 TEXT "
                    "PRIMARY KEY,domain_id TEXT NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS measured_domain_semantics(domain_id TEXT "
                    "PRIMARY KEY,semantic_sha256 TEXT NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS measured_semantic_domains(semantic_identity TEXT "
                    "PRIMARY KEY,domain_id TEXT NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS measured_evaluations(evaluation_id TEXT PRIMARY "
                    "KEY,suite_sha256 TEXT NOT NULL,snapshot_sha256 TEXT NOT NULL,"
                    "snapshot_json TEXT NOT NULL,evaluator_artifact BLOB NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS measured_approvals(approval_id TEXT PRIMARY KEY,"
                    "approval_sha256 TEXT UNIQUE NOT NULL,enablement_sha256 TEXT NOT NULL,"
                    "used_transitions INTEGER NOT NULL,used_callback_calls INTEGER NOT NULL)",
                    "CREATE TABLE IF NOT EXISTS measured_attempts(ticket_id TEXT PRIMARY KEY,"
                    "approval_id TEXT NOT NULL,enablement_sha256 TEXT NOT NULL,"
                    "shard_id TEXT NOT NULL,"
                    "transitions INTEGER NOT NULL)",
                ):
                    connection.execute(statement)
                for grant in authority.grants:
                    source_sha = sha256(canonical(grant.source.to_record()))
                    semantic_sha = sha256(canonical(grant.source.compatibility.to_record()))
                    semantic_identity = self._domain_identity(grant.source)
                    old = connection.execute(
                        "SELECT domain_id FROM measured_domains WHERE source_spec_sha256=?",
                        (source_sha,),
                    ).fetchone()
                    if old is not None and old["domain_id"] != grant.evaluation_domain_id:
                        raise FleetError("evaluation_domain_conflict")
                    reverse = connection.execute(
                        "SELECT domain_id FROM measured_semantic_domains WHERE semantic_identity=?",
                        (semantic_identity,),
                    ).fetchone()
                    if reverse is not None and reverse["domain_id"] != grant.evaluation_domain_id:
                        raise FleetError("evaluation_domain_alias")
                    semantic = connection.execute(
                        "SELECT semantic_sha256 FROM measured_domain_semantics WHERE domain_id=?",
                        (grant.evaluation_domain_id,),
                    ).fetchone()
                    if semantic is not None and semantic["semantic_sha256"] != semantic_sha:
                        raise FleetError("evaluation_domain_semantics_conflict")
                    if (
                        old is None
                        and connection.execute("SELECT COUNT(*) FROM measured_domains").fetchone()[
                            0
                        ]
                        >= self.limits.max_sources
                    ):
                        raise FleetError("measured_grant_quota")
                    connection.execute(
                        "INSERT OR IGNORE INTO measured_domains VALUES(?,?)",
                        (source_sha, grant.evaluation_domain_id),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO measured_domain_semantics VALUES(?,?)",
                        (grant.evaluation_domain_id, semantic_sha),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO measured_semantic_domains VALUES(?,?)",
                        (semantic_identity, grant.evaluation_domain_id),
                    )
                self._backfill_measured_history(connection)
        self._measured_authority = authority
        self._measured_destination = destination

    def _backfill_measured_history(self, connection: sqlite3.Connection) -> None:
        """Atomically include pre-extension reservations without reading payloads.

        Reservations, holdout tombstones and callback journals survive purge.
        Their semantic protection must survive later domain registration too;
        indexing them never authenticates or promotes an old numeric carrier.
        """
        input_limit = self.limits.max_shards * self.limits.max_transitions
        for table, maximum in (
            ("sources", self.limits.max_sources),
            ("shards", self.limits.max_shards),
            ("consumptions", self.limits.max_shards),
            ("transition_reservations", input_limit),
            ("holdout_inputs", input_limit),
            ("measured_inputs", input_limit),
            ("measured_holdout_inputs", input_limit),
        ):
            count = connection.execute(
                f"SELECT COUNT(*) FROM (SELECT 1 FROM {table} LIMIT ?)", (maximum + 1,)
            ).fetchone()[0]
            if count > maximum:
                raise FleetError("measured_history_quota")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS measured_inputs_shard ON measured_inputs(shard_id)"
        )
        source_domains = {}
        for row in connection.execute("SELECT source_id,source_epoch,spec_json FROM sources"):
            source = SourceSpec.from_record(
                _json(row["spec_json"].encode(), self.limits.max_manifest_bytes)
            )
            registered = connection.execute(
                "SELECT domain_id FROM measured_semantic_domains WHERE semantic_identity=?",
                (self._domain_identity(source),),
            ).fetchone()
            if registered is not None:
                source_domains[(row["source_id"], row["source_epoch"])] = registered["domain_id"]
        for table in ("transition_reservations", "holdout_inputs"):
            for row in connection.execute(
                f"SELECT t.input_sha256,t.shard_id,s.source_id,s.source_epoch FROM {table} t "
                "JOIN shards s ON s.shard_id=t.shard_id"
            ):
                domain = source_domains.get((row["source_id"], row["source_epoch"]))
                if domain is None:
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO measured_inputs VALUES(?,?,?)",
                    (domain, row["input_sha256"], row["shard_id"]),
                )
                if table == "holdout_inputs":
                    connection.execute(
                        "INSERT OR IGNORE INTO measured_holdout_inputs VALUES(?,?,?)",
                        (domain, row["input_sha256"], row["shard_id"]),
                    )
        for table in ("measured_inputs", "measured_holdout_inputs"):
            count = connection.execute(
                f"SELECT COUNT(*) FROM (SELECT 1 FROM {table} LIMIT ?)", (input_limit + 1,)
            ).fetchone()[0]
            if count > input_limit:
                raise FleetError("measured_history_quota")
        # A possibly used training copy disqualifies an existing ready holdout.
        # Frozen definitions remain immutable; their conflicted inputs stop
        # being eligible instead of being relabelled or silently replaced.
        connection.execute(
            "UPDATE shards SET status='quarantine' WHERE status='ready' AND EXISTS("
            "SELECT 1 FROM sources p WHERE p.source_id=shards.source_id "
            "AND p.source_epoch=shards.source_epoch "
            "AND json_extract(p.spec_json,'$.split')='evaluation_holdout') AND EXISTS("
            "SELECT 1 FROM measured_inputs held JOIN measured_inputs trained "
            "ON trained.domain_id=held.domain_id AND trained.input_sha256=held.input_sha256 "
            "JOIN shards prior ON prior.shard_id=trained.shard_id JOIN sources p "
            "ON p.source_id=prior.source_id AND p.source_epoch=prior.source_epoch "
            "LEFT JOIN consumptions c ON c.shard_id=prior.shard_id "
            "WHERE held.shard_id=shards.shard_id AND json_extract(p.spec_json,'$.split')='train' "
            "AND (prior.status IN ('claimed','calling','consumed','unknown_effect') "
            "OR c.status IN ('claimed','calling','consumed','unknown_effect') "
            "OR c.callback_completed=1))"
        )
        connection.execute(
            "UPDATE shards SET status='quarantine' WHERE status='ready' AND EXISTS("
            "SELECT 1 FROM sources p WHERE p.source_id=shards.source_id "
            "AND p.source_epoch=shards.source_epoch "
            "AND json_extract(p.spec_json,'$.split')='train') "
            "AND EXISTS(SELECT 1 FROM measured_inputs t JOIN measured_holdout_inputs h "
            "ON t.domain_id=h.domain_id AND t.input_sha256=h.input_sha256 "
            "WHERE t.shard_id=shards.shard_id)"
        )

    @staticmethod
    def _has_measured(connection: sqlite3.Connection) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='measured_proofs'"
            ).fetchone()
            is not None
        )

    @staticmethod
    def _domain_identity(source: SourceSpec) -> str:
        return sha256(
            canonical(
                {
                    "game_id": source.game_id,
                    "environment_id": source.compatibility.environment_id,
                    "observation": [asdict(item) for item in source.compatibility.observation],
                    "action": [asdict(item) for item in source.compatibility.action],
                    "masks": [asdict(item) for item in source.compatibility.masks],
                    "reward_dtype": source.compatibility.reward_dtype,
                    "reward_length": source.compatibility.reward_length,
                }
            )
        )

    def _retained_usage(self, connection: sqlite3.Connection) -> int:
        used = int(
            connection.execute("SELECT COALESCE(SUM(retained_bytes),0) FROM shards").fetchone()[0]
        )
        if self._has_measured(connection):
            for table, column in (
                ("measured_evaluations", "snapshot_json"),
                ("measured_evaluations", "evaluator_artifact"),
                ("measured_reward_tails", "tail_json"),
            ):
                used += int(
                    connection.execute(
                        f"SELECT COALESCE(SUM(LENGTH(CAST({column} AS BLOB))),0) FROM {table}"
                    ).fetchone()[0]
                )
        return used

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise FleetError("hub_closed")
        _directory(self.root)
        _directory(self.root / "artifacts")
        connection = sqlite3.connect(
            self.database.as_uri() + "?mode=" + ("rw" if write else "ro"), uri=True, timeout=1.0
        )
        connection.row_factory = sqlite3.Row
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            else:
                connection.execute("PRAGMA query_only=ON")
            identity = connection.execute(
                "SELECT epoch,limits_json FROM hub_identity LIMIT 2"
            ).fetchall()
            if (
                connection.execute("PRAGMA user_version").fetchone()[0] != 1
                or len(identity) != 1
                or identity[0]["epoch"] != self._owner_epoch
                or identity[0]["limits_json"] != canonical(asdict(self.limits)).decode()
            ):
                raise FleetError("hub_identity_mismatch")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _now(self) -> int:
        value = counter(self._clock())
        utc(value)
        return value

    def _artifact(self, shard_id: str) -> Path:
        digest(shard_id)
        directory = self.root / "artifacts" / shard_id
        if directory.exists():
            _directory(directory)
        return directory

    def register_source(self, source: SourceSpec) -> None:
        if type(source) is not SourceSpec:
            raise FleetError("invalid_source")
        encoded = canonical(source.to_record()).decode()
        if len(encoded.encode()) > self.limits.max_manifest_bytes:
            raise FleetError("source_byte_limit")
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT spec_json,revoked FROM sources WHERE source_id=? AND source_epoch=?",
                (source.source_id, source.source_epoch),
            ).fetchone()
            if row is not None:
                if row["spec_json"] != encoded or row["revoked"]:
                    raise FleetError("source_conflict")
                return
            if (
                connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
                >= self.limits.max_sources
            ):
                raise FleetError("source_quota")
            if connection.execute(
                "SELECT 1 FROM sources WHERE source_id=? AND revoked=0 LIMIT 1", (source.source_id,)
            ).fetchone():
                raise FleetError("source_epoch_busy")
            assignment = sha256(
                canonical([source.assignment_id, source.split, source.compatibility_group_sha256])
            )
            previous = connection.execute(
                "SELECT assignment_sha256 FROM run_assignments WHERE run_id=?", (source.run_id,)
            ).fetchone()
            if previous is not None and previous["assignment_sha256"] != assignment:
                raise FleetError("run_assignment_conflict")
            connection.execute(
                "INSERT OR IGNORE INTO run_assignments VALUES(?,?)", (source.run_id, assignment)
            )
            connection.execute(
                "INSERT INTO sources(source_id,source_epoch,spec_json) VALUES(?,?,?)",
                (source.source_id, source.source_epoch, encoded),
            )

    def receive_heartbeat(
        self,
        source_id: str,
        source_epoch: str,
        heartbeat_seq: int,
        *,
        declared_at_utc_ms: int,
        declared_status: str = "unknown",
    ) -> bool:
        counter(heartbeat_seq)
        counter(declared_at_utc_ms)
        utc(declared_at_utc_ms)
        now = self._now()
        if declared_at_utc_ms > now:
            raise FleetError("future_timestamp")
        if declared_status not in {"unknown", "running", "stopped"}:
            raise FleetError("invalid_declared_status")
        with self._connection(write=True) as connection:
            row = self._source(connection, source_id, source_epoch)
            if row["revoked"]:
                raise FleetError("revoked")
            if heartbeat_seq == row["heartbeat_seq"]:
                if (
                    row["heartbeat_declared_ms"] != declared_at_utc_ms
                    or row["declared_status"] != declared_status
                ):
                    raise FleetError("heartbeat_conflict")
                return False
            if heartbeat_seq != row["heartbeat_seq"] + 1:
                raise FleetError("heartbeat_sequence")
            if row["heartbeat_received_ms"] is not None and now < row["heartbeat_received_ms"]:
                raise FleetError("clock_mismatch")
            connection.execute(
                "UPDATE sources SET heartbeat_seq=?,heartbeat_declared_ms=?,heartbeat_received_"
                "ms=?,declared_status=? WHERE source_id=? AND source_epoch=?",
                (heartbeat_seq, declared_at_utc_ms, now, declared_status, source_id, source_epoch),
            )
        return True

    @staticmethod
    def _source(connection: sqlite3.Connection, source_id: str, epoch: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM sources WHERE source_id=? AND source_epoch=?", (source_id, epoch)
        ).fetchone()
        if not isinstance(row, sqlite3.Row):
            raise FleetError("unknown_source")
        return row

    def begin_upload(self, manifest: bytes) -> UploadReceipt:
        return self._begin_upload(manifest)

    def begin_measured_upload(self, manifest: bytes, envelope: bytes) -> UploadReceipt:
        """Authenticate a complete bounded header before reserving any bytes."""
        authority = self.measured_authority
        if authority is None:
            raise FleetError("measured_authority_missing")
        if not isinstance(envelope, bytes) or len(envelope) > self.limits.max_shard_bytes:
            raise FleetError("measured_proof_byte_limit")
        header = verify_measured_manifest(
            envelope, manifest, authority, self._now(), limits=self.limits
        )
        return self._begin_upload(
            manifest,
            measured=(
                envelope,
                header.envelope_sha256,
                header.authority_sha256,
                header.expires_at_utc_ms,
            ),
        )

    def _begin_upload(
        self, manifest: bytes, *, measured: tuple[bytes, str, str, int] | None = None
    ) -> UploadReceipt:
        description = parse_manifest(manifest, self.limits)
        utc(description.produced_at_utc_ms)
        if description.produced_at_utc_ms > self._now():
            raise FleetError("future_timestamp")
        source = description.source
        with self._connection(write=True) as connection:
            old = connection.execute(
                "SELECT * FROM shards WHERE shard_id=?", (description.shard_id,)
            ).fetchone()
            if old is not None:
                if old["manifest_sha256"] != description.manifest_sha256:
                    raise FleetError("shard_identity_conflict")
                if measured is not None:
                    proof = connection.execute(
                        "SELECT envelope FROM measured_proofs WHERE shard_id=?",
                        (description.shard_id,),
                    ).fetchone()
                    if proof is None or bytes(proof["envelope"]) != measured[0]:
                        raise FleetError("measured_proof_conflict")
                if old["status"] == "uploading":
                    directory = self._artifact(description.shard_id)
                    directory.mkdir(exist_ok=True)
                    path = directory / "manifest.json"
                    if not path.exists():
                        _atomic(path, manifest)
                    elif _regular(path, self.limits.max_manifest_bytes) != manifest:
                        raise FleetError("stored_manifest_integrity")
                return UploadReceipt(
                    description.shard_id, old["status"], old["next_chunk_index"], True
                )
            if (
                connection.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
                >= self.limits.max_shards
            ):
                raise FleetError("shard_quota")
            reserved = len(manifest) + description.payload_bytes
            if measured is not None:
                reserved += len(measured[0])
            if self._retained_usage(connection) + reserved > self.limits.max_retained_bytes:
                raise FleetError("retained_byte_quota")
            registered = connection.execute(
                "SELECT * FROM sources WHERE source_id=? AND source_epoch=?",
                (source.source_id, source.source_epoch),
            ).fetchone()
            valid = (
                registered is not None
                and not registered["revoked"]
                and not registered["eval_frozen"]
                and registered["spec_json"] == canonical(source.to_record()).decode()
            )
            status = "uploading" if valid else "rejected"
            if valid and description.shard_seq != registered["last_shard_seq"] + 1:
                raise FleetError("shard_sequence")
            connection.execute(
                "INSERT INTO shards(shard_id,source_id,source_epoch,shard_seq,manifest,manifest"
                "_sha256,payload_sha256,payload_bytes,transition_count,status,retained_bytes) V"
                "ALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    description.shard_id,
                    source.source_id,
                    source.source_epoch,
                    description.shard_seq,
                    manifest,
                    description.manifest_sha256,
                    description.payload_sha256,
                    description.payload_bytes,
                    description.transition_count,
                    status,
                    reserved if valid else 0,
                ),
            )
            if valid:
                if measured is not None:
                    connection.execute(
                        "INSERT INTO measured_proofs VALUES(?,?,?,?,?)",
                        (description.shard_id, *measured),
                    )
                connection.execute(
                    "UPDATE sources SET last_shard_seq=? WHERE source_id=? AND source_epoch=?",
                    (description.shard_seq, source.source_id, source.source_epoch),
                )
        # Ledger/budget reservation precedes file publication, so interruption retains identity.
        if status == "uploading":
            directory = self._artifact(description.shard_id)
            directory.mkdir(exist_ok=True)
            path = directory / "manifest.json"
            if path.exists():
                if _regular(path, self.limits.max_manifest_bytes) != manifest:
                    raise FleetError("stored_manifest_integrity")
            else:
                _atomic(path, manifest)
        return UploadReceipt(description.shard_id, status, 0)

    def put_chunk(self, shard_id: str, index: int, payload: bytes) -> UploadReceipt:
        counter(index)
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM shards WHERE shard_id=?", (digest(shard_id),)
            ).fetchone()
            if row is None or row["status"] != "uploading":
                raise FleetError("upload_not_open")
            source = self._source(connection, row["source_id"], row["source_epoch"])
            if source["revoked"]:
                raise FleetError("revoked")
            manifest = parse_manifest(bytes(row["manifest"]), self.limits)
            if index >= len(manifest.chunks) or index > row["next_chunk_index"]:
                raise FleetError("chunk_sequence")
            expected, length = manifest.chunks[index]
            if (
                not isinstance(payload, bytes)
                or len(payload) != length
                or sha256(payload) != expected
            ):
                raise FleetError("chunk_integrity")
            path = self._artifact(shard_id) / f"chunk-{index}.bin"
            duplicate = index < row["next_chunk_index"]
            if path.exists():
                if _regular(path, self.limits.max_chunk_bytes) != payload:
                    raise FleetError("stored_chunk_integrity")
            else:
                _atomic(path, payload)
            next_index = max(row["next_chunk_index"], index + 1)
            connection.execute(
                "UPDATE shards SET next_chunk_index=? WHERE shard_id=?", (next_index, shard_id)
            )
        return UploadReceipt(shard_id, "uploading", next_index, duplicate)

    def _decoded(self, row: sqlite3.Row) -> DecodedShard:
        description = parse_manifest(bytes(row["manifest"]), self.limits)
        directory = self._artifact(row["shard_id"])
        if _regular(directory / "manifest.json", self.limits.max_manifest_bytes) != bytes(
            row["manifest"]
        ):
            raise FleetError("stored_manifest_integrity")
        parts = []
        used = 0
        for index, (expected, length) in enumerate(description.chunks):
            part = _regular(directory / f"chunk-{index}.bin", length)
            if len(part) != length or sha256(part) != expected:
                raise FleetError("stored_chunk_integrity")
            used += len(part)
            if used > description.payload_bytes:
                raise FleetError("stored_payload_limit")
            parts.append(part)
        payload = b"".join(parts)
        return decode_shard(bytes(row["manifest"]), payload, limits=self.limits)

    def _measurement(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        decoded: DecodedShard | None = None,
    ) -> VerifiedMeasuredShard | None:
        if not self._has_measured(connection):
            return None
        proof = connection.execute(
            "SELECT * FROM measured_proofs WHERE shard_id=?", (row["shard_id"],)
        ).fetchone()
        if proof is None:
            return None
        authority = self.measured_authority
        if authority is None:
            raise FleetError("measured_authority_missing")
        envelope = bytes(proof["envelope"])
        if (
            len(envelope) > self.limits.max_shard_bytes
            or sha256(envelope) != proof["envelope_sha256"]
        ):
            raise FleetError("stored_measured_proof_integrity")
        verified = verify_measured(
            envelope, decoded or self._decoded(row), authority, self._now(), limits=self.limits
        )
        if (
            verified.envelope_sha256 != proof["envelope_sha256"]
            or verified.authority_sha256 != proof["authority_sha256"]
            or verified.expires_at_utc_ms != proof["expires_ms"]
        ):
            raise FleetError("measured_authority_drift")
        return verified

    def _learner_decoded(self, row: sqlite3.Row) -> DecodedShard:
        decoded = self._decoded(row)
        with self._connection() as connection:
            verified = self._measurement(connection, row, decoded)
        if verified is None:
            return decoded
        # Only the queue namespace changes. Signed original actor metadata
        # stays in the proof; all typed transitions and contexts stay intact.
        return DecodedShard(
            verified.decoded.manifest,
            replace(verified.decoded.unroll, actor_id=verified.carrier_decoded.unroll.actor_id),
        )

    def _measured_proof_sha(self, connection: sqlite3.Connection, shard_id: str) -> str | None:
        if not self._has_measured(connection):
            return None
        row = connection.execute(
            "SELECT envelope_sha256 FROM measured_proofs WHERE shard_id=?", (shard_id,)
        ).fetchone()
        return None if row is None else str(row["envelope_sha256"])

    def _real_training_reason(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        source: SourceSpec,
        enablement: RealTrainingEnablement,
        *,
        ticket_id: str | None = None,
    ) -> str:
        """Recheck current RAM trust, fixed holdout and finite owner permission."""
        if self._now() >= enablement.expires_at_utc_ms:
            return "real_enablement_expired"
        destination = self.measured_destination
        if (
            destination is None
            or destination.destination_id != enablement.destination_id
            or destination.destination_sha256 != enablement.destination_sha256
        ):
            return "real_destination_mismatch"
        if sha256(canonical(source.to_record())) not in enablement.allowed_source_spec_sha256s:
            return "real_source_not_allowed"
        if not self._has_measured(connection):
            return "real_proof_missing"
        budget_reason = self._real_budget_reason(connection, row, enablement, ticket_id=ticket_id)
        if budget_reason != "eligible":
            return budget_reason
        try:
            measurement = self._measurement(connection, row)
            if measurement is None:
                return "real_proof_missing"
            if measurement.authority_sha256 != enablement.authority_sha256:
                return "measured_authority_drift"
            evaluation = connection.execute(
                "SELECT * FROM measured_evaluations WHERE evaluation_id=?",
                (enablement.evaluation_id,),
            ).fetchone()
            if evaluation is None:
                return "measured_evaluation_missing"
            encoded = evaluation["snapshot_json"].encode()
            if (
                evaluation["suite_sha256"] != enablement.evaluation_suite_sha256
                or evaluation["snapshot_sha256"] != enablement.evaluation_snapshot_sha256
                or sha256(encoded) != enablement.evaluation_snapshot_sha256
            ):
                return "measured_evaluation_binding"
            record = _json(encoded, self.limits.max_retained_bytes)
            suite = MeasuredEvaluationSuite.from_record(
                record.get("suite"), bytes(evaluation["evaluator_artifact"])
            )
            if (
                suite.sha256 != enablement.evaluation_suite_sha256
                or suite.evidence_kind != enablement.evidence_kind
                or measurement.grant.evidence_kind != enablement.evidence_kind
            ):
                return "measured_evaluation_kind_or_suite"
            shards = record.get("shards")
            if (
                not isinstance(shards, (list, tuple))
                or not 1 <= len(shards) <= self.limits.max_shards
            ):
                return "measured_evaluation_empty"
            domains = set()
            for shard_id, proof_sha in shards:
                heldout = connection.execute(
                    "SELECT * FROM shards WHERE shard_id=?", (shard_id,)
                ).fetchone()
                if heldout is None or heldout["status"] != "ready":
                    return "measured_evaluation_unavailable"
                registered = self._source(connection, heldout["source_id"], heldout["source_epoch"])
                heldout_source = SourceSpec.from_record(json.loads(registered["spec_json"]))
                if (
                    registered["revoked"]
                    or not registered["eval_frozen"]
                    or heldout_source.split != "evaluation_holdout"
                    or heldout_source.simulated
                ):
                    return "measured_evaluation_revoked"
                fixed = self._measurement(connection, heldout)
                if (
                    fixed is None
                    or fixed.envelope_sha256 != proof_sha
                    or fixed.authority_sha256 != enablement.authority_sha256
                    or fixed.grant.evidence_kind != suite.evidence_kind
                ):
                    return "measured_evaluation_binding"
                domains.add(fixed.grant.evaluation_domain_id)
            if measurement.grant.evaluation_domain_id not in domains:
                return "measured_evaluation_domain"
            if self._protected_shard(connection, row["shard_id"]):
                return "holdout_copy"
        except FleetError as error:
            return str(error)
        return "eligible"

    @staticmethod
    def _real_budget_reason(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        enablement: RealTrainingEnablement,
        *,
        ticket_id: str | None = None,
    ) -> str:
        approval = connection.execute(
            "SELECT * FROM measured_approvals WHERE approval_id=? OR approval_sha256=?",
            (enablement.approval_id, enablement.approval_sha256),
        ).fetchone()
        if approval is not None and (
            approval["approval_id"] != enablement.approval_id
            or approval["approval_sha256"] != enablement.approval_sha256
            or approval["enablement_sha256"] != enablement.binding_sha256
        ):
            return "real_approval_binding"
        if connection.execute(
            "SELECT 1 FROM measured_attempts a JOIN consumptions c ON c.ticket_id=a.ticket_id "
            "WHERE a.approval_id=? AND c.status IN ('claimed','calling','unknown_effect') "
            "AND (? IS NULL OR a.ticket_id!=?) LIMIT 1",
            (enablement.approval_id, ticket_id, ticket_id),
        ).fetchone():
            return "real_approval_effect_unknown"
        if ticket_id is not None:
            attempt = connection.execute(
                "SELECT * FROM measured_attempts WHERE ticket_id=?", (ticket_id,)
            ).fetchone()
            if (
                attempt is None
                or attempt["approval_id"] != enablement.approval_id
                or attempt["enablement_sha256"] != enablement.binding_sha256
                or attempt["shard_id"] != row["shard_id"]
                or attempt["transitions"] != row["transition_count"]
            ):
                return "real_attempt_binding"
            return "eligible"
        used_transitions = 0 if approval is None else approval["used_transitions"]
        used_calls = 0 if approval is None else approval["used_callback_calls"]
        if (
            used_transitions + row["transition_count"] > enablement.max_transitions
            or used_calls + 1 > enablement.max_callback_calls
        ):
            return "real_budget_exhausted"
        return "eligible"

    def _reserve_real_attempt(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        enablement: RealTrainingEnablement,
        ticket_id: str,
    ) -> None:
        reason = self._real_budget_reason(connection, row, enablement)
        if reason != "eligible":
            raise FleetError(reason)
        connection.execute(
            "INSERT OR IGNORE INTO measured_approvals VALUES(?,?,?,0,0)",
            (enablement.approval_id, enablement.approval_sha256, enablement.binding_sha256),
        )
        connection.execute(
            "UPDATE measured_approvals SET used_transitions=used_transitions+?,"
            "used_callback_calls=used_callback_calls+1 WHERE approval_id=?",
            (row["transition_count"], enablement.approval_id),
        )
        connection.execute(
            "INSERT INTO measured_attempts VALUES(?,?,?,?,?)",
            (
                ticket_id,
                enablement.approval_id,
                enablement.binding_sha256,
                row["shard_id"],
                row["transition_count"],
            ),
        )

    def finish_upload(self, shard_id: str) -> IngestReceipt:
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM shards WHERE shard_id=?", (digest(shard_id),)
            ).fetchone()
            if row is None or row["status"] not in {
                "uploading",
                "ready",
                "quarantine",
                "consumed",
                "duplicate",
            }:
                raise FleetError("upload_not_open")
            description = parse_manifest(bytes(row["manifest"]), self.limits)
            if row["status"] == "uploading" and row["next_chunk_index"] != len(description.chunks):
                raise FleetError("upload_incomplete")
            duplicate = row["status"] != "uploading"
            if (
                not duplicate
                and connection.execute(
                    "SELECT 1 FROM shards WHERE source_id=? AND source_epoch=? "
                    "AND shard_seq<? AND status='uploading' LIMIT 1",
                    (row["source_id"], row["source_epoch"], row["shard_seq"]),
                ).fetchone()
            ):
                raise FleetError("upload_predecessor_incomplete")
            carrier = self._decoded(row)
            measurement = self._measurement(connection, row, carrier)
            decoded = carrier if measurement is None else measurement.decoded
            source = self._source(connection, row["source_id"], row["source_epoch"])
            if source["revoked"]:
                raise FleetError("revoked")
            if duplicate:
                connection.execute(
                    "UPDATE shards SET duplicate_count=MIN(duplicate_count+1,?) WHERE shard_id=?",
                    (2**53 - 1, shard_id),
                )
                return IngestReceipt(
                    shard_id,
                    row["status"],
                    row["transition_count"],
                    True,
                    "measured_authenticated"
                    if measurement is not None
                    else "simulated_declared"
                    if decoded.source.simulated
                    else "unknown",
                )
            if row["next_chunk_index"] != len(decoded.manifest.chunks):
                raise FleetError("upload_incomplete")
            status = (
                "ready"
                if (decoded.source.simulated or measurement is not None)
                and decoded.source.split != "quarantine"
                else "quarantine"
            )
            now = self._now()
            observed = max(item.timestamp_ns for item in decoded.unroll.transitions) // 1_000_000
            utc(observed)
            if (
                observed > decoded.manifest.produced_at_utc_ms
                or decoded.manifest.produced_at_utc_ms > now
            ):
                raise FleetError("future_or_inconsistent_timestamp")
            if source["data_received_ms"] is not None and now < source["data_received_ms"]:
                raise FleetError("clock_mismatch")
            status = self._reserve_steps(connection, decoded, status)
            if status == "ready" and self._has_measured(connection):
                domain = connection.execute(
                    "SELECT domain_id FROM measured_semantic_domains WHERE semantic_identity=?",
                    (self._domain_identity(decoded.source),),
                ).fetchone()
                if domain is not None:
                    status = self._reserve_domain_holdout(connection, decoded, domain["domain_id"])
            if status == "ready":
                status = self._advance_episode(connection, decoded)
            if status == "ready" and measurement is not None:
                self._advance_measured_reward(connection, measurement)
            connection.execute(
                "UPDATE shards SET status=?,received_ms=? WHERE shard_id=?", (status, now, shard_id)
            )
            if status != "duplicate":
                connection.execute(
                    "UPDATE sources SET data_observed_ms=?,data_received_ms=? WHERE source_id=?"
                    " AND source_epoch=?",
                    (observed, now, row["source_id"], row["source_epoch"]),
                )
            else:
                connection.execute(
                    "UPDATE shards SET duplicate_count=1 WHERE shard_id=?", (shard_id,)
                )
        return IngestReceipt(
            shard_id,
            status,
            len(decoded.unroll.transitions),
            duplicate or status == "duplicate",
            "measured_authenticated"
            if measurement is not None
            else "simulated_declared"
            if decoded.source.simulated
            else "unknown",
        )

    def ingest(self, manifest: bytes, payload: bytes) -> IngestReceipt:
        description = parse_manifest(manifest, self.limits)
        upload = self.begin_upload(manifest)
        if upload.status == "rejected":
            raise FleetError("source_not_admitted")
        if upload.status == "uploading":
            offset = 0
            for index, (_, length) in enumerate(description.chunks):
                part = payload[offset : offset + length]
                if index >= upload.next_chunk_index:
                    self.put_chunk(upload.shard_id, index, part)
                offset += length
            if offset != len(payload):
                raise FleetError("payload_integrity")
        return self.finish_upload(upload.shard_id)

    def ingest_measured(self, manifest: bytes, payload: bytes, envelope: bytes) -> IngestReceipt:
        """Finite local receive; proof readiness is separate from training permission."""
        authority = self.measured_authority
        if authority is None:
            raise FleetError("measured_authority_missing")
        if not isinstance(envelope, bytes) or len(envelope) > self.limits.max_shard_bytes:
            raise FleetError("measured_proof_byte_limit")
        # Full verification precedes ledger and file mutations in the finite convenience path.
        verify_measured(
            envelope,
            decode_shard(manifest, payload, limits=self.limits),
            authority,
            self._now(),
            limits=self.limits,
        )
        description = parse_manifest(manifest, self.limits)
        upload = self.begin_measured_upload(manifest, envelope)
        if upload.status == "rejected":
            raise FleetError("source_not_admitted")
        if upload.status == "uploading":
            offset = 0
            for index, (_, length) in enumerate(description.chunks):
                part = payload[offset : offset + length]
                if index >= upload.next_chunk_index:
                    self.put_chunk(upload.shard_id, index, part)
                offset += length
            if offset != len(payload):
                raise FleetError("payload_integrity")
        return self.finish_upload(upload.shard_id)

    def sync_local_spools(
        self, directories: Sequence[Path], *, max_shards: int = 32
    ) -> SyncReceipt:
        counter(max_shards, minimum=1, maximum=64)
        if not isinstance(directories, (list, tuple)) or len(directories) > self.limits.max_sources:
            raise FleetError("spool_limit")
        completed = duplicates = quarantined = scanned = 0
        for raw in directories:
            root = Path(raw).absolute()
            _directory(root)
            _directory(root / "shards")
            owner = _json(_regular(root / "spool-owner.json", 1024), 1024)
            if (
                set(owner) != {"schema", "source_id", "source_epoch"}
                or owner["schema"] != "glr.fleet.spool.v1"
            ):
                raise FleetError("invalid_spool_owner")
            catalog = _json(_regular(root / "catalog.json", 1_048_576), 1_048_576)
            entries = catalog.get("shards")
            if (
                set(catalog) != {"shards"}
                or not isinstance(entries, (list, tuple))
                or len(entries) > self.limits.max_shards
            ):
                raise FleetError("invalid_spool_catalog")
            seen = set()
            for entry in entries:
                if scanned >= max_shards:
                    break
                shard_id = digest(entry)
                if shard_id in seen:
                    raise FleetError("duplicate_catalog_entry")
                seen.add(shard_id)
                directory = root / "shards" / shard_id
                _directory(directory)
                manifest = _regular(directory / "manifest.json", self.limits.max_manifest_bytes)
                description = parse_manifest(manifest, self.limits)
                if (
                    description.shard_id != shard_id
                    or description.source.source_id != owner["source_id"]
                    or description.source.source_epoch != owner["source_epoch"]
                ):
                    raise FleetError("spool_identity_mismatch")
                upload = self.begin_upload(manifest)
                if upload.status == "rejected":
                    raise FleetError("source_not_admitted")
                for index, (expected_sha, expected_length) in enumerate(description.chunks):
                    part = _regular(directory / f"chunk-{index}.bin", expected_length)
                    if len(part) != expected_length or sha256(part) != expected_sha:
                        raise FleetError("spool_chunk_integrity")
                    if upload.status == "uploading" and index >= upload.next_chunk_index:
                        self.put_chunk(
                            shard_id,
                            index,
                            part,
                        )
                receipt = self.finish_upload(shard_id)
                scanned += 1
                completed += not receipt.duplicate
                duplicates += receipt.duplicate
                quarantined += receipt.status == "quarantine"
            if scanned >= max_shards:
                break
        return SyncReceipt(scanned, completed, duplicates, quarantined)

    def revoke_source(self, source_id: str, source_epoch: str) -> None:
        with self._connection(write=True) as connection:
            self._source(connection, source_id, source_epoch)
            connection.execute(
                "UPDATE sources SET revoked=1 WHERE source_id=? AND source_epoch=?",
                (source_id, source_epoch),
            )

    @staticmethod
    def _input_record(transition: Any) -> dict[str, Any]:
        record = transition_to_record(transition)
        return {
            key: record[key]
            for key in (
                "observation",
                "action",
                "next_observation",
                "reward",
                "terminated",
                "truncated",
                "action_mask",
                "next_action_mask",
            )
        }

    def _reserve_steps(
        self, connection: sqlite3.Connection, decoded: DecodedShard, status: str
    ) -> str:
        source = decoded.source
        cohort = source.compatibility_group_sha256
        rows = [
            (
                str(item.episode_id),
                item.step_id,
                sha256(canonical(transition_to_record(item))),
                sha256(canonical(self._input_record(item))),
            )
            for item in decoded.unroll.transitions
        ]
        if source.split == "evaluation_holdout" and status == "ready":
            matched = set()
            for _, _, _, input_sha in rows:
                for prior in connection.execute(
                    "SELECT DISTINCT s.shard_id,s.status FROM transition_reservations t JOIN sh"
                    "ards s ON t.shard_id=s.shard_id WHERE t.cohort_sha256=? AND t.input_sha256"
                    "=? AND t.split='train'",
                    (cohort, input_sha),
                ):
                    if prior["status"] in {"claimed", "calling", "consumed", "unknown_effect"}:
                        return "quarantine"
                    matched.add(prior["shard_id"])
            for shard_id in matched:
                connection.execute(
                    "UPDATE shards SET status='quarantine' WHERE shard_id=? AND status='ready'",
                    (shard_id,),
                )
            for _, _, _, input_sha in rows:
                connection.execute(
                    "INSERT OR IGNORE INTO holdout_inputs VALUES(?,?,?)",
                    (cohort, input_sha, decoded.manifest.shard_id),
                )
        duplicate_count = conflicts = 0
        for episode, step, record_sha, input_sha in rows:
            previous = connection.execute(
                "SELECT * FROM transition_reservations WHERE episode_id=? AND step_id=?",
                (episode, step),
            ).fetchone()
            if previous is not None:
                duplicate_count += 1
                conflicts += (
                    previous["record_sha256"] != record_sha or previous["run_id"] != source.run_id
                )
            if (
                source.split == "train"
                and connection.execute(
                    "SELECT 1 FROM holdout_inputs WHERE cohort_sha256=? AND input_sha256=?",
                    (cohort, input_sha),
                ).fetchone()
            ):
                conflicts += 1
        for episode, step, record_sha, input_sha in rows:
            connection.execute(
                "INSERT OR IGNORE INTO transition_reservations VALUES(?,?,?,?,?,?,?,?)",
                (
                    episode,
                    step,
                    record_sha,
                    input_sha,
                    cohort,
                    source.run_id,
                    source.split,
                    decoded.manifest.shard_id,
                ),
            )
        if source.split == "evaluation_holdout" and status == "ready":
            return "ready"
        if conflicts or 0 < duplicate_count < len(rows):
            return "quarantine"
        return "duplicate" if duplicate_count == len(rows) else status

    @staticmethod
    def _advance_episode(connection: sqlite3.Connection, decoded: DecodedShard) -> str:
        """Persist the lifecycle boundary independently of removable chunk files."""
        source = decoded.source
        first, last = decoded.unroll.transitions[0], decoded.unroll.transitions[-1]
        previous = connection.execute(
            "SELECT * FROM episode_tails WHERE episode_id=?", (str(first.episode_id),)
        ).fetchone()
        first_record = transition_to_record(first)
        if previous is not None and (
            previous["done"]
            or previous["run_id"] != source.run_id
            or previous["source_id"] != source.source_id
            or previous["source_epoch"] != source.source_epoch
            or previous["cohort_sha256"] != source.compatibility_group_sha256
            or first.step_id != previous["step_id"] + 1
            or first.timestamp_ns < previous["timestamp_ns"]
            or sha256(canonical(first_record["observation"])) != previous["next_observation_sha256"]
        ):
            return "quarantine"
        connection.execute(
            "INSERT INTO episode_tails VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(episode_id) DO UPDATE SET step_id=excluded.step_id, "
            "timestamp_ns=excluded.timestamp_ns, "
            "next_observation_sha256=excluded.next_observation_sha256,done=excluded.done",
            (
                str(last.episode_id),
                source.run_id,
                source.source_id,
                source.source_epoch,
                source.compatibility_group_sha256,
                last.step_id,
                last.timestamp_ns,
                sha256(canonical(transition_to_record(last)["next_observation"])),
                int(last.done),
            ),
        )
        return "ready"

    def _reserve_domain_holdout(
        self, connection: sqlite3.Connection, decoded: DecodedShard, domain: str
    ) -> str:
        """Protect exact numeric copies across source/build/policy aliases.

        The owner registered semantic domain survives build/source relabels;
        independent game domains may contain the same numbers. Near copies are
        not claimed as detected.
        """
        inputs = [
            sha256(canonical(self._input_record(item))) for item in decoded.unroll.transitions
        ]
        if decoded.source.split == "evaluation_holdout":
            matches = set()
            for input_sha in inputs:
                for row in connection.execute(
                    "SELECT DISTINCT s.shard_id,s.status,c.status AS consumption_status,"
                    "c.callback_completed FROM measured_inputs t "
                    "JOIN shards s ON s.shard_id=t.shard_id JOIN sources p ON "
                    "p.source_id=s.source_id AND p.source_epoch=s.source_epoch "
                    "LEFT JOIN consumptions c ON c.shard_id=s.shard_id "
                    "WHERE t.domain_id=? AND t.input_sha256=? "
                    "AND json_extract(p.spec_json,'$.split')='train'",
                    (domain, input_sha),
                ):
                    if (
                        row["status"] in {"claimed", "calling", "consumed", "unknown_effect"}
                        or row["consumption_status"]
                        in {"claimed", "calling", "consumed", "unknown_effect"}
                        or row["callback_completed"]
                    ):
                        return "quarantine"
                    matches.add(row["shard_id"])
            for shard_id in matches:
                connection.execute(
                    "UPDATE shards SET status='quarantine' WHERE shard_id=? AND status='ready'",
                    (shard_id,),
                )
            for input_sha in inputs:
                connection.execute(
                    "INSERT OR IGNORE INTO measured_holdout_inputs VALUES(?,?,?)",
                    (domain, input_sha, decoded.manifest.shard_id),
                )
        elif any(
            connection.execute(
                "SELECT 1 FROM measured_holdout_inputs WHERE domain_id=? AND input_sha256=?",
                (domain, item),
            ).fetchone()
            is not None
            for item in inputs
        ):
            return "quarantine"
        for input_sha in inputs:
            connection.execute(
                "INSERT OR IGNORE INTO measured_inputs VALUES(?,?,?)",
                (domain, input_sha, decoded.manifest.shard_id),
            )
        return "ready"

    def _advance_measured_reward(
        self, connection: sqlite3.Connection, measurement: VerifiedMeasuredShard
    ) -> None:
        """Check the signed budget boundary against the durable prior boundary."""
        first, last = measurement.steps[0], measurement.steps[-1]
        source = measurement.decoded.source
        episode_id = str(first.before.episode_id)
        previous = connection.execute(
            "SELECT tail_json FROM measured_reward_tails WHERE episode_id=?", (episode_id,)
        ).fetchone()
        identity = {
            "source_spec_sha256": sha256(canonical(source.to_record())),
            "life_id": first.life_id,
            "training_config_sha256": measurement.grant.training_config_sha256,
            "authority_sha256": measurement.authority_sha256,
        }
        if previous is None:
            if first.before.step_id != 0 or first.budget_before.action_count != 0:
                raise FleetError("measured_reward_tail_missing")
        else:
            tail = _json(previous["tail_json"].encode(), self.limits.max_shard_bytes)
            if (
                any(tail.get(name) != value for name, value in identity.items())
                or tail.get("budget_after") != asdict(first.budget_before)
                or tail.get("after") != first.before.to_mapping()
                or tail["budget_after"]["closed"]
            ):
                raise FleetError("measured_reward_tail_conflict")
        for step in measurement.steps:
            if connection.execute(
                "SELECT 1 FROM measured_actions WHERE run_id=? AND target_id=? AND action_id=?",
                (source.run_id, step.before.target_id, step.receipt.action_id),
            ).fetchone():
                raise FleetError("measured_action_reused")
            connection.execute(
                "INSERT INTO measured_actions VALUES(?,?,?,?)",
                (
                    source.run_id,
                    step.before.target_id,
                    step.receipt.action_id,
                    measurement.decoded.manifest.shard_id,
                ),
            )
        record = {
            **identity,
            "life_id": last.life_id,
            "after": last.after.to_mapping(),
            "budget_after": asdict(last.budget_after),
        }
        encoded = canonical(record)
        old_bytes = 0 if previous is None else len(previous["tail_json"].encode())
        if (
            self._retained_usage(connection) - old_bytes + len(encoded)
            > self.limits.max_retained_bytes
        ):
            raise FleetError("retained_byte_quota")
        connection.execute(
            "INSERT INTO measured_reward_tails VALUES(?,?,?) ON CONFLICT(episode_id) "
            "DO UPDATE SET tail_json=excluded.tail_json,shard_id=excluded.shard_id",
            (episode_id, encoded.decode(), measurement.decoded.manifest.shard_id),
        )

    def freeze_measured_evaluation(
        self,
        evaluation_id: str,
        sources: tuple[tuple[str, str], ...],
        *,
        suite: MeasuredEvaluationSuite,
    ) -> MeasuredEvaluationSnapshot:
        """Freeze actual admitted measured holdout bytes, never an empty approval."""
        identifier(evaluation_id)
        if type(suite) is not MeasuredEvaluationSuite:
            raise FleetError("invalid_measured_evaluation_suite")
        suite_sha256 = suite.sha256
        if (
            not isinstance(sources, tuple)
            or not 1 <= len(sources) <= self.limits.max_sources
            or len(set(sources)) != len(sources)
        ):
            raise FleetError("measured_evaluation_sources")
        shards: list[tuple[str, str]] = []
        actual_cases = {}
        spec_shas = []
        with self._connection(write=True) as connection:
            if not self._has_measured(connection):
                raise FleetError("measured_authority_missing")
            for source_id, epoch in sorted(sources):
                source_row = self._source(connection, source_id, epoch)
                source = SourceSpec.from_record(json.loads(source_row["spec_json"]))
                if (
                    source.simulated
                    or source.split != "evaluation_holdout"
                    or source_row["revoked"]
                ):
                    raise FleetError("evaluation_not_eligible")
                rows = connection.execute(
                    "SELECT * FROM shards WHERE source_id=? AND source_epoch=? ORDER BY shard_seq",
                    (source_id, epoch),
                ).fetchall()
                if not rows or any(row["status"] != "ready" for row in rows):
                    raise FleetError("evaluation_incomplete_or_conflicted")
                for row in rows:
                    measurement = self._measurement(connection, row)
                    if measurement is None:
                        raise FleetError("evaluation_not_measured")
                    if (
                        measurement.grant.evaluation_domain_id != suite.evaluation_domain_id
                        or measurement.grant.evidence_kind != suite.evidence_kind
                    ):
                        raise FleetError("measured_evaluation_scope")
                    shards.append((row["shard_id"], measurement.envelope_sha256))
                    actual_cases[row["shard_id"]] = (
                        source_id,
                        epoch,
                        row["payload_sha256"],
                        measurement.envelope_sha256,
                    )
                spec_shas.append(sha256(canonical(source.to_record())))
            if {case.shard_id for case in suite.cases} != set(actual_cases):
                raise FleetError("measured_evaluation_case_coverage")
            for case in suite.cases:
                if actual_cases[case.shard_id] != (
                    case.source_id,
                    case.source_epoch,
                    case.payload_sha256,
                    case.proof_sha256,
                ):
                    raise FleetError("measured_evaluation_case_binding")
            record = {
                "schema": "glr.fleet.measured-evaluation.v1",
                "evaluation_id": evaluation_id,
                "suite_sha256": suite_sha256,
                "source_spec_sha256s": sorted(spec_shas),
                "shards": sorted(shards),
                "suite": suite.to_record(),
            }
            encoded = canonical(record)
            snapshot_sha = sha256(encoded)
            old = connection.execute(
                "SELECT * FROM measured_evaluations WHERE evaluation_id=?", (evaluation_id,)
            ).fetchone()
            if old is not None:
                if (
                    old["snapshot_sha256"] != snapshot_sha
                    or old["snapshot_json"] != encoded.decode()
                    or bytes(old["evaluator_artifact"]) != suite.evaluator_artifact
                ):
                    raise FleetError("measured_evaluation_conflict")
            else:
                if (
                    connection.execute("SELECT COUNT(*) FROM measured_evaluations").fetchone()[0]
                    >= self.limits.max_sources
                ):
                    raise FleetError("measured_evaluation_quota")
                if (
                    self._retained_usage(connection) + len(encoded) + len(suite.evaluator_artifact)
                    > self.limits.max_retained_bytes
                ):
                    raise FleetError("retained_byte_quota")
                connection.execute(
                    "INSERT INTO measured_evaluations VALUES(?,?,?,?,?)",
                    (
                        evaluation_id,
                        suite_sha256,
                        snapshot_sha,
                        encoded.decode(),
                        suite.evaluator_artifact,
                    ),
                )
            for source_id, epoch in sources:
                connection.execute(
                    "UPDATE sources SET eval_frozen=1 WHERE source_id=? AND source_epoch=?",
                    (source_id, epoch),
                )
        return MeasuredEvaluationSnapshot(
            evaluation_id,
            suite_sha256,
            snapshot_sha,
            tuple(sorted(spec_shas)),
            tuple(item[0] for item in sorted(shards)),
        )

    def freeze_evaluation(self, source_id: str, source_epoch: str) -> None:
        with self._connection(write=True) as connection:
            source_row = self._source(connection, source_id, source_epoch)
            source = SourceSpec.from_record(json.loads(source_row["spec_json"]))
            if (
                source_row["revoked"]
                or source.split != "evaluation_holdout"
                or not source.simulated
            ):
                raise FleetError("evaluation_not_eligible")
            statuses = [
                row[0]
                for row in connection.execute(
                    "SELECT status FROM shards WHERE source_id=? AND source_epoch=? LIMIT ?",
                    (source_id, source_epoch, self.limits.max_shards),
                )
            ]
            if not statuses or any(value != "ready" for value in statuses):
                raise FleetError("evaluation_incomplete_or_conflicted")
            connection.execute(
                "UPDATE sources SET eval_frozen=1 WHERE source_id=? AND source_epoch=?",
                (source_id, source_epoch),
            )

    @staticmethod
    def _holdout_blocked(connection: sqlite3.Connection, source: SourceSpec) -> bool:
        for row in connection.execute("SELECT spec_json,eval_frozen,revoked FROM sources LIMIT 64"):
            other = SourceSpec.from_record(json.loads(row["spec_json"]))
            if (
                other.split == "evaluation_holdout"
                and other.compatibility_group_sha256 == source.compatibility_group_sha256
                and (not row["eval_frozen"] or row["revoked"])
            ):
                return True
        return False

    @staticmethod
    def _protected_shard(connection: sqlite3.Connection, shard_id: str) -> bool:
        protected = (
            connection.execute(
                "SELECT 1 FROM transition_reservations t JOIN holdout_inputs h ON t.cohort_sha2"
                "56=h.cohort_sha256 AND t.input_sha256=h.input_sha256 WHERE t.shard_id=? AND t."
                "split='train' LIMIT 1",
                (shard_id,),
            ).fetchone()
            is not None
        )
        if protected:
            return True
        return (
            FleetHub._has_measured(connection)
            and connection.execute(
                "SELECT 1 FROM measured_inputs t JOIN measured_holdout_inputs h ON "
                "t.domain_id=h.domain_id AND t.input_sha256=h.input_sha256 "
                "JOIN transition_reservations r ON r.shard_id=t.shard_id "
                "WHERE t.shard_id=? AND r.split='train' LIMIT 1",
                (shard_id,),
            ).fetchone()
            is not None
        )

    def resume(self) -> dict[str, int]:
        with self._connection(write=True) as connection:
            uncertain = connection.execute(
                "SELECT COUNT(*) FROM shards WHERE status IN ('claimed','calling')"
            ).fetchone()[0]
            connection.execute(
                "UPDATE shards SET status='unknown_effect' WHERE status IN ('claimed','calling')"
            )
            connection.execute(
                "UPDATE consumptions SET status='unknown_effect' WHERE status IN ('claimed','ca"
                "lling')"
            )
            uploading = connection.execute(
                "SELECT COUNT(*) FROM shards WHERE status='uploading'"
            ).fetchone()[0]
        return {"unknown_effect_shards": uncertain, "incomplete_uploads": uploading}

    def purge_shard(self, shard_id: str) -> None:
        with self._connection(write=True) as connection:
            row = connection.execute(
                "SELECT status,manifest FROM shards WHERE shard_id=?", (digest(shard_id),)
            ).fetchone()
            if row is None or row["status"] in {"claimed", "calling", "unknown_effect"}:
                raise FleetError("unsafe_purge")
            connection.execute("UPDATE shards SET status='purged' WHERE shard_id=?", (shard_id,))
        directory = self._artifact(shard_id)
        if directory.exists():
            description = parse_manifest(bytes(row["manifest"]), self.limits)
            for name in [
                "manifest.json",
                *(f"chunk-{index}.bin" for index in range(len(description.chunks))),
            ]:
                path = directory / name
                if path.exists():
                    _regular(path, max(self.limits.max_chunk_bytes, self.limits.max_manifest_bytes))
                    path.unlink()
            if any(directory.iterdir()):
                raise FleetError("unowned_artifact_in_purge")
            directory.rmdir()
        with self._connection(write=True) as connection:
            if self._has_measured(connection):
                connection.execute("DELETE FROM measured_proofs WHERE shard_id=?", (shard_id,))
            connection.execute(
                "UPDATE shards SET retained_bytes=0 WHERE shard_id=? AND status='purged'",
                (shard_id,),
            )

    @staticmethod
    def _snapshot_reasons(encoded: str) -> list[str]:
        # Snapshot v1 is consumed by a closed native enum. Durable SDK
        # admission details stay in the ledger; additive REAL refusals project
        # to the existing quarantine value without expanding that wire format.
        allowed = {
            "eligible",
            "revoked",
            "holdout",
            "quarantine",
            "compatibility_mismatch",
            "simulated_not_allowed",
            "policy_mismatch",
            "off_policy_not_allowed",
            "already_claimed",
            "no_ready_data",
        }
        values = json.loads(encoded)
        return [value if value in allowed else "quarantine" for value in values]

    def snapshot(self) -> dict[str, Any]:
        machines = []
        datasets = []
        receipts = []
        with self._connection() as connection:
            for row in connection.execute(
                "SELECT * FROM sources ORDER BY source_id,source_epoch LIMIT 64"
            ):
                source = SourceSpec.from_record(
                    _json(row["spec_json"].encode(), self.limits.max_manifest_bytes)
                )
                machines.append(
                    {
                        "source_id": source.source_id,
                        "source_epoch": source.source_epoch,
                        "machine_id": source.machine_id,
                        "simulated": source.simulated,
                        "revoked": bool(row["revoked"]),
                        "declared_status": row["declared_status"],
                        "run_id": source.run_id,
                        "game_id": source.game_id,
                        "environment_id": source.compatibility.environment_id,
                        "source_revision": source.source_revision,
                        "runtime_source_commit": source.runtime_source_commit,
                        "adapter_source_sha256": source.adapter_source_sha256,
                        "behavior_policy_sha256": source.behavior_policy_sha256,
                        "checkpoint_sha256": source.checkpoint_sha256,
                        "heartbeat_declared_at_utc": utc(row["heartbeat_declared_ms"]),
                        "heartbeat_received_at_utc": utc(row["heartbeat_received_ms"]),
                        "data_observed_at_utc": utc(row["data_observed_ms"]),
                        "data_received_at_utc": utc(row["data_received_ms"]),
                    }
                )
                totals = connection.execute(
                    "SELECT COALESCE(SUM(CASE WHEN status IN ('ready','claimed','calling','cons"
                    "umed','unknown_effect') THEN transition_count ELSE 0 END),0) accepted,COAL"
                    "ESCE(SUM(transition_count*duplicate_count),0) duplicates,SUM(status='rejec"
                    "ted') rejected,SUM(status='ready') ready,SUM(status='quarantine') quaranti"
                    "ne,COALESCE(SUM(retained_bytes),0) retained FROM shards WHERE source_id=? "
                    "AND source_epoch=?",
                    (source.source_id, source.source_epoch),
                ).fetchone()
                quality = (
                    "revoked"
                    if row["revoked"]
                    else (
                        "quarantine"
                        if totals["quarantine"]
                        else ("ready" if totals["accepted"] else "empty")
                    )
                )
                datasets.append(
                    {
                        "source_id": source.source_id,
                        "source_epoch": source.source_epoch,
                        "compatibility_group_sha256": source.compatibility_group_sha256,
                        "assignment_id": source.assignment_id,
                        "split": source.split,
                        "quality_status": quality,
                        "accepted_transition_count": totals["accepted"],
                        "duplicate_transition_count": totals["duplicates"],
                        "rejected_shard_count": totals["rejected"] or 0,
                        "ready_shard_count": totals["ready"] or 0,
                        "retained_bytes": totals["retained"],
                        "last_plan_eligible": None
                        if row["last_plan_eligible"] is None
                        else bool(row["last_plan_eligible"]),
                        "last_plan_reason_codes": self._snapshot_reasons(row["last_plan_reasons"]),
                    }
                )
            for row in connection.execute(
                "SELECT * FROM consumptions ORDER BY rowid DESC LIMIT 64"
            ):
                receipts.append(
                    {
                        "receipt_id": row["receipt_id"],
                        "plan_id": row["plan_id"],
                        "learner_id": row["learner_id"],
                        "source_ids": [row["source_id"]],
                        "status": row["status"]
                        if row["status"] in {"consumed", "rejected"}
                        else "unknown_effect",
                        "transition_count": row["transition_count"],
                        "callback_completed": bool(row["callback_completed"]),
                        "learner_declared_updates": row["declared_updates"],
                        "finished_at_utc": utc(row["finished_ms"]),
                    }
                )
        result = {
            "schema_version": "glr.fleet.snapshot.v1",
            "generated_at_utc": utc(self._now()),
            "scope": "local_trusted_sources",
            "heartbeat_ttl_seconds": 120,
            "machines": machines,
            "datasets": datasets,
            "consumer_receipts": receipts,
        }
        if len(canonical(result)) > 1_048_576:
            raise FleetError("snapshot_byte_limit")
        return result

    def write_snapshot(self) -> Path:
        path = self.root / "launcher-snapshot.json"
        _atomic(path, canonical(self.snapshot()))
        return path

    def close(self) -> None:
        self._closed = True


def write_local_shard(
    directory: Path, shard: EncodedShard, *, limits: FleetLimits = DEFAULT_LIMITS
) -> str:
    """Publish a finite producer shard; local roots are selected by the owner."""
    decoded = decode_shard(shard.manifest, shard.payload, limits=limits)
    directory = Path(directory).absolute()
    if not directory.exists():
        directory.mkdir(parents=True)
    _directory(directory)
    marker = directory / "spool-owner.json"
    owner = {
        "schema": "glr.fleet.spool.v1",
        "source_id": decoded.source.source_id,
        "source_epoch": decoded.source.source_epoch,
    }
    if marker.exists():
        if _json(_regular(marker, 1024), 1024) != owner:
            raise FleetError("spool_owner_conflict")
    else:
        if any(directory.iterdir()):
            raise FleetError("unowned_spool_directory")
        _atomic(marker, canonical(owner))
    if not (directory / "shards").exists():
        (directory / "shards").mkdir()
    _directory(directory / "shards")
    catalog_path = directory / "catalog.json"
    if not catalog_path.exists():
        if any((directory / "shards").iterdir()):
            raise FleetError("missing_spool_catalog")
        _atomic(catalog_path, canonical({"shards": []}))
    catalog = _json(_regular(catalog_path, 1_048_576), 1_048_576)
    if set(catalog) != {"shards"} or not isinstance(catalog["shards"], (list, tuple)):
        raise FleetError("invalid_spool_catalog")
    entries = [digest(item) for item in catalog["shards"]]
    if len(entries) > limits.max_shards or len(set(entries)) != len(entries):
        raise FleetError("invalid_spool_catalog")
    shard_id = decoded.manifest.shard_id
    if shard_id not in entries and len(entries) >= limits.max_shards:
        raise FleetError("spool_shard_quota")
    output = directory / "shards" / shard_id
    if output.exists():
        _directory(output)
        if (output / "manifest.json").exists():
            if _regular(output / "manifest.json", limits.max_manifest_bytes) != shard.manifest:
                raise FleetError("spool_shard_conflict")
        else:
            if any(output.iterdir()):
                raise FleetError("unowned_spool_fragment")
            _atomic(output / "manifest.json", shard.manifest)
    else:
        output.mkdir()
        _atomic(output / "manifest.json", shard.manifest)
    offset = 0
    for index, (_, length) in enumerate(decoded.manifest.chunks):
        part = shard.payload[offset : offset + length]
        path = output / f"chunk-{index}.bin"
        if path.exists():
            if _regular(path, limits.max_chunk_bytes) != part:
                raise FleetError("spool_chunk_conflict")
        else:
            _atomic(path, part)
        offset += length
    if shard_id not in entries:
        entries.append(shard_id)
        _atomic(catalog_path, canonical({"shards": entries}))
    return shard_id
