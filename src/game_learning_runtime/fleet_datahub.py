"""Finite local spool reception, with an owned schema-1 index and no services."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import time_ns
from typing import Any
from uuid import uuid4

from game_learning_runtime.fleet_payload import (
    DEFAULT_LIMITS,
    DecodedShard,
    EncodedShard,
    FleetError,
    FleetLimits,
    SourceSpec,
    canonical,
    counter,
    decode_shard,
    digest,
    parse_manifest,
    sha256,
)
from game_learning_runtime.offline_parsing import OfflineParseError, parse_json_object
from game_learning_runtime.serialization import transition_to_record


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

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        limits: FleetLimits = DEFAULT_LIMITS,
        clock_ms: Callable[[], int] = lambda: time_ns() // 1_000_000,
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
        return hub

    @classmethod
    def open(
        cls, root: Path, *, clock_ms: Callable[[], int] = lambda: time_ns() // 1_000_000
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
        return cls(root, limits, clock_ms, owner["epoch"])

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
            if (
                connection.execute("SELECT COALESCE(SUM(retained_bytes),0) FROM shards").fetchone()[
                    0
                ]
                + reserved
                > self.limits.max_retained_bytes
            ):
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
            decoded = self._decoded(row)
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
                    "simulated_declared" if decoded.source.simulated else "unknown",
                )
            if row["next_chunk_index"] != len(decoded.manifest.chunks):
                raise FleetError("upload_incomplete")
            status = (
                "ready"
                if decoded.source.simulated and decoded.source.split != "quarantine"
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
            if status == "ready":
                status = self._advance_episode(connection, decoded)
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
            "simulated_declared" if decoded.source.simulated else "unknown",
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
        return (
            connection.execute(
                "SELECT 1 FROM transition_reservations t JOIN holdout_inputs h ON t.cohort_sha2"
                "56=h.cohort_sha256 AND t.input_sha256=h.input_sha256 WHERE t.shard_id=? AND t."
                "split='train' LIMIT 1",
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
            connection.execute(
                "UPDATE shards SET retained_bytes=0 WHERE shard_id=? AND status='purged'",
                (shard_id,),
            )

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
                        "last_plan_reason_codes": json.loads(row["last_plan_reasons"]),
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
