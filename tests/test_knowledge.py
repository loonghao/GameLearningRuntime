from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from game_learning_runtime import (
    KnowledgeInjector,
    KnowledgeIntent,
    KnowledgeQuery,
    TrainingConfig,
)
from game_learning_runtime.errors import ContractViolation


def _training_mapping(*, enabled: bool = True) -> dict[str, object]:
    return {
        "schema_version": "glr.training.v1",
        "knowledge_sources": [
            {
                "id": "strategy-kb",
                "authority": "advisory",
                "required": False,
                "provides_context": True,
                "max_age_seconds": 3600,
                "max_payload_bytes": 4096,
            }
        ],
        "knowledge_injection": {
            "enabled": enabled,
            "allowed_intents": ["acquire", "engage", "upgrade", "avoid"],
            "max_items": 3,
            "min_confidence": 0.5,
        },
        "reward": {
            "terms": [
                {
                    "name": "outcome",
                    "source": "strategy-kb",
                    "minimum_authority": "advisory",
                    "required": False,
                }
            ]
        },
    }


def _snapshot_mapping() -> dict[str, object]:
    return {
        "schema_version": "glr.knowledge-snapshot.v1",
        "snapshot_id": "build-42",
        "source_id": "strategy-kb",
        "created_at": "2026-09-01T00:00:00Z",
        "items": [
            {
                "id": "upgrade-core",
                "intent": "upgrade",
                "subject": "core-module",
                "summary": "Prioritize the core module after stage two.",
                "tags": ["economy", "ranged"],
                "priority": 90,
                "confidence": 0.9,
                "min_stage": 2,
            },
            {
                "id": "acquire-resource",
                "intent": "acquire",
                "subject": "resource-cache",
                "summary": "Collect a nearby resource cache when capacity is available.",
                "tags": ["economy"],
                "priority": 70,
                "confidence": 0.8,
                "max_stage": 4,
            },
            {
                "id": "engage-elite",
                "intent": "engage",
                "subject": "elite-target",
                "summary": "Engage only after the core upgrade is available.",
                "tags": ["combat", "ranged"],
                "priority": 80,
                "confidence": 0.7,
                "min_stage": 3,
            },
            {
                "id": "avoid-unknown",
                "intent": "avoid",
                "subject": "unknown-hazard",
                "summary": "Avoid the unverified hazard.",
                "tags": ["safety"],
                "priority": 100,
                "confidence": 0.4,
            },
        ],
    }


def _payload(mapping: dict[str, object] | None = None) -> bytes:
    return json.dumps(mapping or _snapshot_mapping(), separators=(",", ":")).encode()


def test_training_config_loads_bounded_knowledge_injection_policy() -> None:
    config = TrainingConfig.from_mapping(_training_mapping())

    assert config.knowledge_injection.enabled
    assert config.knowledge_injection.allowed_intents == frozenset(KnowledgeIntent)
    assert config.knowledge_injection.max_items == 3
    assert config.knowledge_injection.min_confidence == pytest.approx(0.5)


def test_injector_selects_ranked_stage_and_tag_relevant_context() -> None:
    injector = KnowledgeInjector(TrainingConfig.from_mapping(_training_mapping()))

    context = injector.inject(
        [_payload()],
        KnowledgeQuery(
            intents=frozenset(
                {KnowledgeIntent.ACQUIRE, KnowledgeIntent.ENGAGE, KnowledgeIntent.UPGRADE}
            ),
            stage=3,
            tags=frozenset({"ranged"}),
        ),
        observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
    )

    assert [item.item_id for item in context.items] == ["upgrade-core", "engage-elite"]
    assert context.source_ids == ("strategy-kb",)
    assert context.snapshot_ids == ("build-42",)
    assert len(context.payload_sha256) == 64
    assert context.items[0].summary == "Prioritize the core module after stage two."
    assert context.items[0].source_id == "strategy-kb"
    assert context.items[0].snapshot_id == "build-42"


def test_injector_applies_policy_limits_and_query_intent() -> None:
    training = _training_mapping()
    injection = training["knowledge_injection"]
    assert isinstance(injection, dict)
    injection["max_items"] = 1
    injector = KnowledgeInjector(TrainingConfig.from_mapping(training))

    context = injector.inject(
        [_payload()],
        KnowledgeQuery(intents=frozenset({KnowledgeIntent.ACQUIRE}), stage=3),
        observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
    )

    assert [item.item_id for item in context.items] == ["acquire-resource"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda snapshot: snapshot.update({"source_id": "missing"}), "undeclared source"),
        (lambda snapshot: snapshot.update({"created_at": "2026-08-31T22:00:00Z"}), "stale"),
        (lambda snapshot: snapshot.update({"created_at": "2026-09-01T01:00:00Z"}), "future"),
        (lambda snapshot: snapshot.update({"unexpected": True}), "unexpected fields"),
        (
            lambda snapshot: snapshot["items"][0].update({"summary": "x" * 513}),
            "summary",
        ),
    ],
)
def test_injector_rejects_untrusted_or_invalid_snapshots(mutate: object, message: str) -> None:
    snapshot = deepcopy(_snapshot_mapping())
    assert callable(mutate)
    mutate(snapshot)
    injector = KnowledgeInjector(TrainingConfig.from_mapping(_training_mapping()))

    with pytest.raises((ContractViolation, ValueError), match=message):
        injector.inject(
            [_payload(snapshot)],
            KnowledgeQuery(stage=3),
            observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
        )


def test_injector_rejects_disabled_and_oversized_inputs() -> None:
    disabled = KnowledgeInjector(TrainingConfig.from_mapping(_training_mapping(enabled=False)))
    with pytest.raises(ContractViolation, match="disabled"):
        disabled.inject(
            [_payload()],
            KnowledgeQuery(),
            observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
        )

    training = _training_mapping()
    sources = training["knowledge_sources"]
    assert isinstance(sources, list)
    sources[0]["max_payload_bytes"] = 32
    oversized = KnowledgeInjector(TrainingConfig.from_mapping(training))
    with pytest.raises(ContractViolation, match="payload limit"):
        oversized.inject(
            [_payload()],
            KnowledgeQuery(),
            observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
        )


def test_injector_requires_declared_required_context_sources() -> None:
    training = _training_mapping()
    sources = training["knowledge_sources"]
    assert isinstance(sources, list)
    sources[0]["required"] = True
    injector = KnowledgeInjector(TrainingConfig.from_mapping(training))

    with pytest.raises(ContractViolation, match="missing required knowledge context"):
        injector.inject(
            [],
            KnowledgeQuery(),
            observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_items", 0, "positive integer"),
        ("min_confidence", 1.1, "between 0 and 1"),
        ("allowed_intents", ["execute"], "allowed_intents"),
    ],
)
def test_training_config_rejects_invalid_injection_policy(
    field: str, value: object, message: str
) -> None:
    training = _training_mapping()
    injection = training["knowledge_injection"]
    assert isinstance(injection, dict)
    injection[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        TrainingConfig.from_mapping(training)


def test_injector_rejects_disallowed_intent_duplicate_source_and_non_context_source() -> None:
    training = _training_mapping()
    injection = training["knowledge_injection"]
    assert isinstance(injection, dict)
    injection["allowed_intents"] = ["acquire"]
    injector = KnowledgeInjector(TrainingConfig.from_mapping(training))
    now = datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc)

    with pytest.raises(ContractViolation, match="not allowed"):
        injector.inject(
            [_payload()],
            KnowledgeQuery(intents=frozenset({KnowledgeIntent.UPGRADE})),
            observed_at=now,
        )

    injection["allowed_intents"] = ["acquire", "engage", "upgrade", "avoid"]
    injector = KnowledgeInjector(TrainingConfig.from_mapping(training))
    with pytest.raises(ContractViolation, match="duplicate source"):
        injector.inject([_payload(), _payload()], KnowledgeQuery(), observed_at=now)

    sources = training["knowledge_sources"]
    assert isinstance(sources, list)
    sources[0]["provides_context"] = False
    injector = KnowledgeInjector(TrainingConfig.from_mapping(training))
    with pytest.raises(ContractViolation, match="not enabled"):
        injector.inject([_payload()], KnowledgeQuery(), observed_at=now)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"{", "UTF-8 JSON"),
        (b"[]", "must be an object"),
        (
            _payload({**_snapshot_mapping(), "schema_version": "glr.knowledge-snapshot.v2"}),
            "schema_version",
        ),
        (
            _payload({**_snapshot_mapping(), "created_at": "2026-09-01T00:00:00"}),
            "timezone",
        ),
    ],
    ids=["invalid-json", "non-object", "wrong-schema", "missing-timezone"],
)
def test_injector_rejects_malformed_snapshot_envelopes(payload: bytes, message: str) -> None:
    injector = KnowledgeInjector(TrainingConfig.from_mapping(_training_mapping()))

    with pytest.raises((TypeError, ValueError), match=message):
        injector.inject(
            [payload],
            KnowledgeQuery(),
            observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
        )


def test_injector_rejects_snapshot_item_count_above_hard_limit() -> None:
    training = _training_mapping()
    sources = training["knowledge_sources"]
    assert isinstance(sources, list)
    sources[0]["max_payload_bytes"] = 100_000
    snapshot = _snapshot_mapping()
    items = snapshot["items"]
    assert isinstance(items, list)
    snapshot["items"] = items * 65
    injector = KnowledgeInjector(TrainingConfig.from_mapping(training))

    with pytest.raises(ValueError, match="more than 256"):
        injector.inject(
            [_payload(snapshot)],
            KnowledgeQuery(),
            observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"priority": 101}, "priority"),
        ({"confidence": float("nan")}, "finite"),
        ({"confidence": -0.1}, "between 0 and 1"),
        ({"min_stage": -1}, "non-negative"),
        ({"min_stage": 3, "max_stage": 2}, "cannot exceed"),
        ({"tags": ["same", "same"]}, "duplicates"),
        ({"intent": "execute"}, "intent"),
        ({"subject": ""}, "non-empty"),
    ],
)
def test_injector_rejects_malformed_items(change: dict[str, object], message: str) -> None:
    snapshot = _snapshot_mapping()
    items = snapshot["items"]
    assert isinstance(items, list)
    items[0].update(change)
    injector = KnowledgeInjector(TrainingConfig.from_mapping(_training_mapping()))

    with pytest.raises((TypeError, ValueError), match=message):
        injector.inject(
            [_payload(snapshot)],
            KnowledgeQuery(),
            observed_at=datetime(2026, 9, 1, 0, 30, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"intents": frozenset()}, "cannot be empty"),
        ({"intents": frozenset({"execute"})}, "unsupported"),
        ({"stage": -1}, "non-negative"),
        ({"tags": frozenset({"local/path"})}, "must match"),
        ({"max_items": 0}, "positive integer"),
        ({"min_confidence": 2}, "between 0 and 1"),
    ],
)
def test_knowledge_query_rejects_invalid_boundaries(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        KnowledgeQuery(**kwargs)  # type: ignore[arg-type]


def test_injector_requires_bytes_and_timezone_aware_observation_time() -> None:
    injector = KnowledgeInjector(TrainingConfig.from_mapping(_training_mapping()))

    with pytest.raises(TypeError, match="must be bytes"):
        injector.inject(
            ["not-bytes"],  # type: ignore[list-item]
            KnowledgeQuery(),
            observed_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
    with pytest.raises(ValueError, match="timezone"):
        injector.inject([_payload()], KnowledgeQuery(), observed_at=datetime(2026, 9, 1))
