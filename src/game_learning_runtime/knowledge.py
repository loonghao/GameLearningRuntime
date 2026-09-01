"""Validated, bounded advisory knowledge contexts for learner-side injection.

Knowledge snapshots are passive data. They cannot expand an action mask,
acknowledge an action, or satisfy an authoritative reward term.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.training import KnowledgeIntent, TrainingConfig

KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION = "glr.knowledge-snapshot.v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_MAX_ITEMS_PER_SNAPSHOT = 256
_MAX_SUMMARY_CHARS = 512
_MAX_SUBJECT_CHARS = 128
_MAX_TAGS = 16


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} requires string keys")
    return value


def _sequence(value: object, *, path: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{path} must be an array")
    return value


def _reject_unknown(value: Mapping[str, Any], *, allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unexpected fields: {unknown}")


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{path} must match {_IDENTIFIER.pattern!r}")
    return value


def _bounded_text(value: object, *, path: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be non-empty text")
    if len(value) > maximum:
        raise ValueError(f"{path} cannot exceed {maximum} characters")
    return value


def _number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    return result


def _stage(value: object, *, path: str, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _timestamp(value: object, *, path: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be an RFC 3339 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{path} must be an RFC 3339 timestamp") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{path} must include a timezone")
    return result.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class KnowledgeItem:
    """One advisory recommendation with explicit intent and applicability."""

    item_id: str
    source_id: str
    snapshot_id: str
    intent: KnowledgeIntent
    subject: str
    summary: str
    tags: tuple[str, ...] = ()
    priority: int = 0
    confidence: float = 0.0
    min_stage: int = 0
    max_stage: int | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source_id: str,
        snapshot_id: str,
    ) -> KnowledgeItem:
        _reject_unknown(
            value,
            allowed=frozenset(
                {
                    "id",
                    "intent",
                    "subject",
                    "summary",
                    "tags",
                    "priority",
                    "confidence",
                    "min_stage",
                    "max_stage",
                }
            ),
            path="items[]",
        )
        try:
            intent = KnowledgeIntent(value.get("intent"))
        except (TypeError, ValueError) as error:
            raise ValueError("items[].intent is not a supported knowledge intent") from error
        raw_tags = _sequence(value.get("tags", ()), path="items[].tags")
        if len(raw_tags) > _MAX_TAGS:
            raise ValueError(f"items[].tags cannot contain more than {_MAX_TAGS} values")
        tags = tuple(_identifier(tag, path="items[].tags[]") for tag in raw_tags)
        if len(set(tags)) != len(tags):
            raise ValueError("items[].tags contains duplicates")
        priority = value.get("priority", 0)
        if not isinstance(priority, int) or isinstance(priority, bool) or not 0 <= priority <= 100:
            raise ValueError("items[].priority must be an integer between 0 and 100")
        confidence = _number(value.get("confidence", 0.0), path="items[].confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("items[].confidence must be between 0 and 1")
        min_stage = _stage(value.get("min_stage", 0), path="items[].min_stage")
        max_stage = _stage(value.get("max_stage"), path="items[].max_stage", optional=True)
        assert min_stage is not None
        if max_stage is not None and min_stage > max_stage:
            raise ValueError("items[].min_stage cannot exceed items[].max_stage")
        return cls(
            item_id=_identifier(value.get("id"), path="items[].id"),
            source_id=source_id,
            snapshot_id=snapshot_id,
            intent=intent,
            subject=_bounded_text(
                value.get("subject"), path="items[].subject", maximum=_MAX_SUBJECT_CHARS
            ),
            summary=_bounded_text(
                value.get("summary"), path="items[].summary", maximum=_MAX_SUMMARY_CHARS
            ),
            tags=tags,
            priority=priority,
            confidence=confidence,
            min_stage=min_stage,
            max_stage=max_stage,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeSnapshot:
    """A validated point-in-time set of advisory knowledge items."""

    snapshot_id: str
    source_id: str
    created_at: datetime
    items: tuple[KnowledgeItem, ...]
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class KnowledgeQuery:
    """Learner-provided context request; it carries no action authority."""

    intents: frozenset[KnowledgeIntent] = field(default_factory=lambda: frozenset(KnowledgeIntent))
    stage: int = 0
    tags: frozenset[str] = field(default_factory=frozenset)
    max_items: int | None = None
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        try:
            intents = frozenset(KnowledgeIntent(intent) for intent in self.intents)
        except (TypeError, ValueError) as error:
            raise ValueError("query.intents contains an unsupported knowledge intent") from error
        if not intents:
            raise ValueError("query.intents cannot be empty")
        object.__setattr__(self, "intents", intents)
        parsed_stage = _stage(self.stage, path="query.stage")
        assert parsed_stage is not None
        object.__setattr__(self, "stage", parsed_stage)
        tags = frozenset(_identifier(tag, path="query.tags[]") for tag in self.tags)
        object.__setattr__(self, "tags", tags)
        if self.max_items is not None and (
            not isinstance(self.max_items, int)
            or isinstance(self.max_items, bool)
            or self.max_items <= 0
        ):
            raise ValueError("query.max_items must be a positive integer")
        confidence = _number(self.min_confidence, path="query.min_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("query.min_confidence must be between 0 and 1")
        object.__setattr__(self, "min_confidence", confidence)


@dataclass(frozen=True, slots=True)
class KnowledgeContext:
    """Immutable, provenance-bound knowledge selected for one learner decision."""

    items: tuple[KnowledgeItem, ...]
    source_ids: tuple[str, ...]
    snapshot_ids: tuple[str, ...]
    payload_sha256: str
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class KnowledgeInjector:
    """Validate snapshots and deterministically build a bounded context."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def inject(
        self,
        payloads: Iterable[bytes],
        query: KnowledgeQuery,
        *,
        observed_at: datetime,
    ) -> KnowledgeContext:
        policy = self._config.knowledge_injection
        if not policy.enabled:
            raise ContractViolation("knowledge injection is disabled by training policy")
        if not isinstance(observed_at, datetime):
            raise TypeError("observed_at must be a datetime")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        now = observed_at.astimezone(timezone.utc)
        if not isinstance(query, KnowledgeQuery):
            raise TypeError("query must be a KnowledgeQuery")
        disallowed = query.intents - policy.allowed_intents
        if disallowed:
            disallowed_names = sorted(intent.value for intent in disallowed)
            raise ContractViolation(
                f"query intents are not allowed by training policy: {disallowed_names}"
            )

        snapshots = tuple(self._load_snapshot(payload, now=now) for payload in payloads)
        source_ids = [snapshot.source_id for snapshot in snapshots]
        if len(set(source_ids)) != len(source_ids):
            raise ContractViolation("knowledge payloads contain duplicate source snapshots")
        required_sources = {
            source.source_id
            for source in self._config.knowledge_sources
            if source.provides_context and source.required
        }
        missing_sources = sorted(required_sources - set(source_ids))
        if missing_sources:
            raise ContractViolation(
                f"missing required knowledge context sources: {missing_sources}"
            )

        minimum_confidence = max(policy.min_confidence, query.min_confidence)
        selected = [
            item
            for snapshot in snapshots
            for item in snapshot.items
            if item.intent in query.intents
            and item.confidence >= minimum_confidence
            and item.min_stage <= query.stage
            and (item.max_stage is None or query.stage <= item.max_stage)
            and (not query.tags or bool(query.tags.intersection(item.tags)))
        ]
        selected.sort(
            key=lambda item: (
                -item.priority,
                -item.confidence,
                item.intent.value,
                item.source_id,
                item.item_id,
            )
        )
        limit = policy.max_items
        if query.max_items is not None:
            limit = min(limit, query.max_items)

        ordered_snapshots = sorted(snapshots, key=lambda snapshot: snapshot.source_id)
        digest = hashlib.sha256()
        for snapshot in ordered_snapshots:
            digest.update(bytes.fromhex(snapshot.payload_sha256))
        query_digest = hashlib.sha256(
            json.dumps(
                {
                    "intents": sorted(intent.value for intent in query.intents),
                    "stage": query.stage,
                    "tags": sorted(query.tags),
                    "max_items": query.max_items,
                    "min_confidence": query.min_confidence,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return KnowledgeContext(
            items=tuple(selected[:limit]),
            source_ids=tuple(snapshot.source_id for snapshot in ordered_snapshots),
            snapshot_ids=tuple(snapshot.snapshot_id for snapshot in ordered_snapshots),
            payload_sha256=digest.hexdigest(),
            metadata={
                "schema_version": KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION,
                "stage": query.stage,
                "selected_count": min(len(selected), limit),
                "query_sha256": query_digest,
                "observed_at": now.isoformat().replace("+00:00", "Z"),
            },
        )

    def _load_snapshot(self, payload: bytes, *, now: datetime) -> KnowledgeSnapshot:
        if not isinstance(payload, bytes):
            raise TypeError("knowledge payloads must be bytes")
        try:
            text = payload.decode("utf-8")
            value = _mapping(json.loads(text), path="knowledge snapshot")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("knowledge payload must be UTF-8 JSON") from error
        _reject_unknown(
            value,
            allowed=frozenset(
                {"schema_version", "snapshot_id", "source_id", "created_at", "items"}
            ),
            path="knowledge snapshot",
        )
        if value.get("schema_version") != KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"knowledge snapshot.schema_version must be {KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION!r}"
            )
        source_id = _identifier(value.get("source_id"), path="knowledge snapshot.source_id")
        source = self._config.knowledge_by_id.get(source_id)
        if source is None:
            raise ContractViolation(
                f"knowledge snapshot references undeclared source {source_id!r}"
            )
        if not source.provides_context:
            raise ContractViolation(
                f"knowledge source {source_id!r} is not enabled to provide learner context"
            )
        if source.max_payload_bytes is not None and len(payload) > source.max_payload_bytes:
            raise ContractViolation(f"knowledge source {source_id!r} exceeded its payload limit")
        created_at = _timestamp(value.get("created_at"), path="knowledge snapshot.created_at")
        age = (now - created_at).total_seconds()
        if age < 0:
            raise ContractViolation(f"knowledge source {source_id!r} has a future timestamp")
        if source.max_age_seconds is not None and age > source.max_age_seconds:
            raise ContractViolation(f"knowledge source {source_id!r} is stale")
        snapshot_id = _identifier(value.get("snapshot_id"), path="knowledge snapshot.snapshot_id")
        raw_items = _sequence(value.get("items"), path="knowledge snapshot.items")
        if len(raw_items) > _MAX_ITEMS_PER_SNAPSHOT:
            message = (
                "knowledge snapshot.items cannot contain more than "
                f"{_MAX_ITEMS_PER_SNAPSHOT} values"
            )
            raise ValueError(message)
        items = tuple(
            KnowledgeItem.from_mapping(
                _mapping(item, path="knowledge snapshot.items[]"),
                source_id=source_id,
                snapshot_id=snapshot_id,
            )
            for item in raw_items
        )
        item_ids = [item.item_id for item in items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("knowledge snapshot.items contains duplicate ids")
        return KnowledgeSnapshot(
            snapshot_id=snapshot_id,
            source_id=source_id,
            created_at=created_at,
            items=items,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )


__all__ = [
    "KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION",
    "KnowledgeContext",
    "KnowledgeInjector",
    "KnowledgeItem",
    "KnowledgeQuery",
    "KnowledgeSnapshot",
]
