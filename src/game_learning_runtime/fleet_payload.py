"""Closed numeric-vector shards for a finite, local trusted-source fleet.

This v1 profile carries no action/correlation proof. Real sources are therefore
quarantined by the hub. Hashes establish content identity, not authentication.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any, cast
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.contracts import Transition, Unroll
from game_learning_runtime.offline_parsing import OfflineParseError, parse_json_object, parse_jsonl
from game_learning_runtime.serialization import (
    RECORD_SCHEMA,
    transition_from_record,
    transition_to_record,
)

MAX_COUNTER = 2**53 - 1
_DTYPES = {"<f4", "<f8", "<i4", "<i8", "|u1", "|b1"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")


class FleetError(ValueError):
    """A fixed reason code; never includes raw payload or local paths."""


def identifier(value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise FleetError("invalid_identifier")
    return value


def digest(value: object, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
        raise FleetError("invalid_digest")
    return value


def counter(value: object, *, minimum: int = 0, maximum: int = MAX_COUNTER) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise FleetError("invalid_counter")
    return value


def canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError):
        raise FleetError("invalid_json") from None


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def closed(value: object, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise FleetError("invalid_fields")
    return value


@dataclass(frozen=True, slots=True)
class FleetLimits:
    max_chunk_bytes: int = 1_048_576
    max_shard_bytes: int = 1_048_576
    max_manifest_bytes: int = 16_384
    max_transitions: int = 128
    max_tensor_bytes: int = 65_536
    max_leaves: int = 32
    max_vector_length: int = 8192
    max_sources: int = 64
    max_shards: int = 1024
    max_retained_bytes: int = 67_108_864
    max_upload_chunks: int = 128
    max_plans: int = 1024

    def __post_init__(self) -> None:
        for name in FleetLimits.__dataclass_fields__:
            value = getattr(self, name)
            counter(value, minimum=1)
        if self.max_sources > 64:
            raise FleetError("source_limit")


DEFAULT_LIMITS = FleetLimits()


@dataclass(frozen=True, slots=True)
class VectorSpec:
    name: str
    dtype: str
    length: int

    def __post_init__(self) -> None:
        identifier(self.name)
        if not isinstance(self.dtype, str) or self.dtype not in _DTYPES:
            raise FleetError("unsafe_dtype")
        counter(self.length, minimum=1)


@dataclass(frozen=True, slots=True)
class CompatibilitySpec:
    environment_id: str
    protocol_version: str
    environment_config_sha256: str
    observation: tuple[VectorSpec, ...]
    action: tuple[VectorSpec, ...]
    reward_contract_sha256: str
    reward_dtype: str = "<f4"
    reward_length: int = 1
    masks: tuple[VectorSpec, ...] = ()

    def __post_init__(self) -> None:
        identifier(self.environment_id)
        identifier(self.protocol_version)
        digest(self.environment_config_sha256)
        digest(self.reward_contract_sha256)
        if not isinstance(self.reward_dtype, str) or self.reward_dtype not in {"<f4", "<f8"}:
            raise FleetError("unsafe_reward_dtype")
        counter(self.reward_length, minimum=1)
        for name in ("observation", "action", "masks"):
            fields = tuple(getattr(self, name))
            if any(type(item) is not VectorSpec for item in fields):
                raise FleetError("invalid_vector_spec")
            if len({item.name for item in fields}) != len(fields):
                raise FleetError("duplicate_vector_field")
            object.__setattr__(self, name, fields)
        if (
            not self.observation
            or not self.action
            or any(item.dtype != "|b1" for item in self.masks)
        ):
            raise FleetError("invalid_layout")

    def to_record(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in CompatibilitySpec.__dataclass_fields__}
        for name in ("observation", "action", "masks"):
            result[name] = [
                {key: getattr(item, key) for key in VectorSpec.__dataclass_fields__}
                for item in getattr(self, name)
            ]
        return result

    @classmethod
    def from_record(cls, value: object) -> CompatibilitySpec:
        record = dict(closed(value, set(cls.__dataclass_fields__)))
        for name in ("observation", "action", "masks"):
            entries = record[name]
            if not isinstance(entries, (list, tuple)) or len(entries) > 32:
                raise FleetError("invalid_layout")
            record[name] = tuple(
                VectorSpec(**closed(entry, {"name", "dtype", "length"})) for entry in entries
            )
        return CompatibilitySpec(**record)

    @property
    def observation_spec_sha256(self) -> str:
        return sha256(canonical([asdict(item) for item in self.observation]))

    @property
    def action_spec_sha256(self) -> str:
        return sha256(
            canonical(
                {
                    "action": [asdict(item) for item in self.action],
                    "masks": [asdict(item) for item in self.masks],
                }
            )
        )


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    source_epoch: str
    machine_id: str
    source_revision: str
    source_sha256: str
    runtime_source_commit: str
    adapter_source_sha256: str
    run_id: str
    game_id: str
    compatibility: CompatibilitySpec
    policy_epoch: str
    policy_version: int
    behavior_policy_sha256: str
    assignment_id: str
    split: str = "train"
    simulated: bool = False
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "source_epoch",
            "machine_id",
            "source_revision",
            "run_id",
            "game_id",
            "policy_epoch",
            "assignment_id",
        ):
            identifier(getattr(self, name))
        for name in ("source_sha256", "adapter_source_sha256", "behavior_policy_sha256"):
            digest(getattr(self, name))
        digest(self.runtime_source_commit, 40)
        if self.checkpoint_sha256 is not None:
            digest(self.checkpoint_sha256)
        counter(self.policy_version)
        if type(self.compatibility) is not CompatibilitySpec:
            raise FleetError("invalid_compatibility")
        if (
            not isinstance(self.simulated, bool)
            or not isinstance(self.split, str)
            or self.split not in {"train", "evaluation_holdout", "quarantine"}
        ):
            raise FleetError("invalid_assignment")

    def to_record(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in SourceSpec.__dataclass_fields__}
        result["compatibility"] = CompatibilitySpec.to_record(self.compatibility)
        return result

    @classmethod
    def from_record(cls, value: object) -> SourceSpec:
        record = dict(closed(value, set(cls.__dataclass_fields__)))
        record["compatibility"] = CompatibilitySpec.from_record(record["compatibility"])
        return SourceSpec(**record)

    @property
    def compatibility_group_sha256(self) -> str:
        return sha256(
            canonical(
                {
                    "game_id": self.game_id,
                    "runtime_source_commit": self.runtime_source_commit,
                    "adapter_source_sha256": self.adapter_source_sha256,
                    "compatibility": self.compatibility.to_record(),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class EncodedShard:
    manifest: bytes
    payload: bytes
    chunks: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class ShardManifest:
    source: SourceSpec
    shard_seq: int
    produced_at_utc_ms: int
    transition_count: int
    payload_sha256: str
    payload_bytes: int
    chunks: tuple[tuple[str, int], ...]
    manifest_sha256: str

    @property
    def shard_id(self) -> str:
        return sha256(canonical([self.source.source_id, self.source.source_epoch, self.shard_seq]))


def parse_manifest(payload: bytes, limits: FleetLimits = DEFAULT_LIMITS) -> ShardManifest:
    if type(limits) is not FleetLimits:
        raise FleetError("invalid_limits")
    try:
        record = closed(
            parse_json_object(payload, max_bytes=limits.max_manifest_bytes).data,
            {
                "schema",
                "source",
                "shard_seq",
                "produced_at_utc_ms",
                "transition_count",
                "payload_sha256",
                "payload_bytes",
                "chunks",
            },
        )
    except OfflineParseError:
        raise FleetError("invalid_manifest_json") from None
    if record["schema"] != "glr.fleet.shard.v1":
        raise FleetError("invalid_manifest_schema")
    source = SourceSpec.from_record(record["source"])
    chunks = record["chunks"]
    if not isinstance(chunks, (list, tuple)) or not 1 <= len(chunks) <= limits.max_upload_chunks:
        raise FleetError("chunk_count_limit")
    chunks_parsed = []
    for item in chunks:
        chunk = closed(item, {"sha256", "bytes"})
        chunks_parsed.append(
            (
                digest(chunk["sha256"]),
                counter(chunk["bytes"], minimum=1, maximum=limits.max_chunk_bytes),
            )
        )
    size = counter(record["payload_bytes"], minimum=1, maximum=limits.max_shard_bytes)
    if sum(item[1] for item in chunks_parsed) != size:
        raise FleetError("chunk_size_mismatch")
    return ShardManifest(
        source,
        counter(record["shard_seq"]),
        counter(record["produced_at_utc_ms"], maximum=253402300799999),
        counter(record["transition_count"], minimum=1, maximum=limits.max_transitions),
        digest(record["payload_sha256"]),
        size,
        tuple(chunks_parsed),
        sha256(payload),
    )


def _array(value: object, expected: VectorSpec, limits: FleetLimits) -> None:
    record = closed(value, {"dtype", "shape", "data"})
    shape = record["shape"]
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        raise FleetError("tensor_spec_mismatch")
    length = counter(shape[0], minimum=1, maximum=limits.max_vector_length)
    if record["dtype"] != expected.dtype or length != expected.length:
        raise FleetError("tensor_spec_mismatch")
    if expected.length > limits.max_vector_length:
        raise FleetError("vector_limit")
    size = expected.length * np.dtype(expected.dtype).itemsize
    if size > limits.max_tensor_bytes:
        raise FleetError("tensor_byte_limit")
    text = record["data"]
    if not isinstance(text, str) or len(text) != 4 * ((size + 2) // 3):
        raise FleetError("tensor_size_mismatch")
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, binascii.Error):
        raise FleetError("invalid_tensor_encoding") from None
    if len(raw) != size or not np.all(np.isfinite(np.frombuffer(raw, dtype=expected.dtype))):
        raise FleetError("nonfinite_or_invalid_tensor")


def _tree(value: object, expected: tuple[VectorSpec, ...], limits: FleetLimits) -> None:
    record = closed(value, {item.name for item in expected})
    for item in expected:
        _array(closed(record[item.name], {"tensor"})["tensor"], item, limits)


def _transition(
    record: Mapping[str, Any], spec: CompatibilitySpec, limits: FleetLimits
) -> Transition:
    closed(
        record,
        {
            "schema",
            "episode_id",
            "step_id",
            "timestamp_ns",
            "observation",
            "action",
            "action_mask",
            "reward",
            "next_observation",
            "next_action_mask",
            "action_receipt",
            "terminated",
            "truncated",
            "events",
            "info",
            "provenance",
        },
    )
    if (
        record["schema"] != RECORD_SCHEMA
        or record["events"]
        or record["info"]
        or record["provenance"] is not None
        or record["action_receipt"] is not None
    ):
        raise FleetError("unsupported_transition_profile")
    if record["events"] not in ([], ()) or not isinstance(record["info"], Mapping):
        raise FleetError("unsupported_transition_profile")
    try:
        if (
            not isinstance(record["episode_id"], str)
            or str(UUID(record["episode_id"])) != record["episode_id"]
        ):
            raise ValueError
    except (ValueError, AttributeError):
        raise FleetError("invalid_episode") from None
    counter(record["step_id"])
    counter(record["timestamp_ns"], maximum=2**63 - 1)
    if 2 * len(spec.observation) + len(spec.action) + 2 * len(spec.masks) + 3 > limits.max_leaves:
        raise FleetError("leaf_limit")
    _tree(record["observation"], spec.observation, limits)
    _tree(record["next_observation"], spec.observation, limits)
    _tree(record["action"], spec.action, limits)
    for name in ("action_mask", "next_action_mask"):
        if spec.masks:
            _tree(record[name], spec.masks, limits)
        elif record[name] is not None:
            raise FleetError("unexpected_mask")
    _array(record["reward"], VectorSpec("reward", spec.reward_dtype, spec.reward_length), limits)
    for name in ("terminated", "truncated"):
        _array(record[name], VectorSpec(name, "|b1", spec.reward_length), limits)
    transition = transition_from_record(record)
    if np.any(transition.terminated & transition.truncated):
        raise FleetError("ambiguous_lifecycle")
    return transition


@dataclass(frozen=True, slots=True)
class DecodedShard:
    manifest: ShardManifest
    unroll: Unroll

    @property
    def source(self) -> SourceSpec:
        return self.manifest.source


def decode_shard(
    manifest: bytes, payload: bytes, *, limits: FleetLimits = DEFAULT_LIMITS
) -> DecodedShard:
    description = parse_manifest(manifest, limits)
    if (
        not isinstance(payload, bytes)
        or len(payload) != description.payload_bytes
        or sha256(payload) != description.payload_sha256
    ):
        raise FleetError("payload_integrity")
    cursor = 0
    for expected, length in description.chunks:
        if sha256(payload[cursor : cursor + length]) != expected:
            raise FleetError("chunk_integrity")
        cursor += length
    try:
        records = parse_jsonl(
            payload,
            max_bytes=limits.max_shard_bytes,
            max_line_bytes=limits.max_shard_bytes,
            max_records=limits.max_transitions,
        ).records
    except OfflineParseError:
        raise FleetError("invalid_transition_json") from None
    if len(records) != description.transition_count:
        raise FleetError("transition_count_mismatch")
    transitions = tuple(
        _transition(item.data, description.source.compatibility, limits) for item in records
    )
    for previous, current in pairwise(transitions):
        if (
            previous.done
            or previous.episode_id != current.episode_id
            or current.step_id != previous.step_id + 1
            or current.timestamp_ns < previous.timestamp_ns
        ):
            raise FleetError("transition_sequence")
        if any(
            not np.array_equal(
                cast(NDArray[Any], previous.next_observation[key]),
                cast(NDArray[Any], current.observation[key]),
            )
            for key in previous.next_observation
        ):
            raise FleetError("transition_continuity")
    unroll = Unroll(
        transitions,
        actor_id=f"{description.source.source_id}:{description.source.source_epoch}",
        sequence_id=description.shard_seq,
        policy_version=description.source.policy_version,
        environment_config_digest=description.source.compatibility.environment_config_sha256,
    )
    return DecodedShard(description, unroll)


def encode_shard(
    source: SourceSpec,
    *,
    shard_seq: int,
    unroll: Unroll,
    produced_at_utc_ms: int,
    limits: FleetLimits = DEFAULT_LIMITS,
) -> EncodedShard:
    if type(source) is not SourceSpec or type(limits) is not FleetLimits:
        raise FleetError("unsupported_contract_type")
    counter(shard_seq)
    counter(produced_at_utc_ms)
    if type(unroll) is not Unroll or not 1 <= len(unroll.transitions) <= limits.max_transitions:
        raise FleetError("transition_count_limit")
    if (
        unroll.actor_id != source.source_id
        or unroll.sequence_id != shard_seq
        or unroll.policy_version != source.policy_version
        or unroll.environment_config_digest != source.compatibility.environment_config_sha256
    ):
        raise FleetError("unroll_provenance")
    records = []
    used = 0
    for transition in unroll.transitions:
        extent = _live_extent(transition, source.compatibility, limits)
        if used + extent + 1 > limits.max_shard_bytes:
            raise FleetError("shard_byte_limit")
        encoded = canonical(transition_to_record(transition))
        if len(encoded) != extent:
            raise FleetError("codec_profile_changed")
        records.append(encoded)
        used += extent + 1
    payload = b"\n".join(records) + b"\n"
    if len(payload) > limits.max_shard_bytes:
        raise FleetError("shard_byte_limit")
    chunks = tuple(
        payload[index : index + limits.max_chunk_bytes]
        for index in range(0, len(payload), limits.max_chunk_bytes)
    )
    manifest = canonical(
        {
            "schema": "glr.fleet.shard.v1",
            "source": source.to_record(),
            "shard_seq": shard_seq,
            "produced_at_utc_ms": produced_at_utc_ms,
            "transition_count": len(unroll.transitions),
            "payload_sha256": sha256(payload),
            "payload_bytes": len(payload),
            "chunks": [{"sha256": sha256(part), "bytes": len(part)} for part in chunks],
        }
    )
    decode_shard(manifest, payload, limits=limits)
    return EncodedShard(manifest, payload, chunks)


def _live_extent(transition: Transition, spec: CompatibilitySpec, limits: FleetLimits) -> int:
    """Exact wire-byte estimate without allocating base64 for any live tensor."""
    if (
        type(transition) is not Transition
        or transition.info
        or transition.events
        or transition.provenance is not None
        or transition.action_receipt is not None
    ):
        raise FleetError("unsupported_transition_profile")
    if not isinstance(transition.episode_id, UUID):
        raise FleetError("invalid_episode")
    counter(transition.step_id)
    counter(transition.timestamp_ns, maximum=2**63 - 1)
    if 2 * len(spec.observation) + len(spec.action) + 2 * len(spec.masks) + 3 > limits.max_leaves:
        raise FleetError("leaf_limit")
    data_bytes = 0

    def array(value: object, expected: VectorSpec) -> dict[str, Any]:
        nonlocal data_bytes
        if (
            not isinstance(value, np.ndarray)
            or value.dtype.str != expected.dtype
            or value.shape != (expected.length,)
            or value.size > limits.max_vector_length
            or value.nbytes > limits.max_tensor_bytes
            or not np.all(np.isfinite(value))
        ):
            raise FleetError("unsafe_tensor")
        data_bytes += 4 * ((value.nbytes + 2) // 3)
        return {"dtype": value.dtype.str, "shape": list(value.shape), "data": ""}

    def tree(value: object, layout: tuple[VectorSpec, ...]) -> dict[str, Any] | None:
        if not layout:
            if value is not None:
                raise FleetError("unexpected_mask")
            return None
        values = closed(value, {item.name for item in layout})
        return {item.name: {"tensor": array(values[item.name], item)} for item in layout}

    skeleton = {
        "schema": RECORD_SCHEMA,
        "episode_id": str(transition.episode_id),
        "step_id": transition.step_id,
        "timestamp_ns": transition.timestamp_ns,
        "observation": tree(transition.observation, spec.observation),
        "action": tree(transition.action, spec.action),
        "next_observation": tree(transition.next_observation, spec.observation),
        "action_mask": tree(transition.action_mask, spec.masks),
        "next_action_mask": tree(transition.next_action_mask, spec.masks),
        "reward": array(
            transition.reward, VectorSpec("reward", spec.reward_dtype, spec.reward_length)
        ),
        "terminated": array(
            transition.terminated, VectorSpec("terminated", "|b1", spec.reward_length)
        ),
        "truncated": array(
            transition.truncated, VectorSpec("truncated", "|b1", spec.reward_length)
        ),
        "events": [],
        "info": {},
        "provenance": None,
        "action_receipt": None,
    }
    if np.any(transition.terminated & transition.truncated):
        raise FleetError("ambiguous_lifecycle")
    return len(canonical(skeleton)) + data_bytes
