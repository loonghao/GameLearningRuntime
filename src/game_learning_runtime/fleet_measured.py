"""Owner-authorized, bounded measured proof beside the numeric fleet carrier.

HMAC authenticates an authorized exporter's declaration, not a physical game's
effects. Keys are caller-owned RAM values. Verification retains original typed
receipts and checks their arithmetic; it never composes a replacement receipt.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from itertools import pairwise
from types import MappingProxyType
from typing import Any, NoReturn, cast
from uuid import UUID

import numpy as np

from game_learning_runtime.contracts import (
    ActionOutcome,
    ActionReceipt,
    Event,
    RefusalReasonClass,
    TimeStep,
    Transition,
    Unroll,
    environment_config_digest,
)
from game_learning_runtime.correlated_rewards import (
    OBSERVATION_CONTEXT_KEY,
    REWARD_EVIDENCE_KEY,
    CorrelatedRewardReceipt,
    CorrelationPolicy,
    EffectState,
    ObservationContext,
    RewardAttribution,
    tensor_tree_sha256,
)
from game_learning_runtime.fleet_payload import (
    DEFAULT_LIMITS,
    MAX_COUNTER,
    CompatibilitySpec,
    DecodedShard,
    EncodedShard,
    FleetError,
    FleetLimits,
    ShardManifest,
    SourceSpec,
    VectorSpec,
    canonical,
    decode_shard,
    digest,
    encode_shard,
    identifier,
    parse_manifest,
    sha256,
)
from game_learning_runtime.offline_parsing import OfflineParseError, parse_json_object
from game_learning_runtime.phases import EnvironmentPhase
from game_learning_runtime.realtime import RealtimeActionReceipt, RealtimeActionStatus
from game_learning_runtime.serialization import transition_to_record
from game_learning_runtime.training import (
    BridgeConfig,
    KnowledgeAuthority,
    KnowledgeInjectionConfig,
    KnowledgeIntent,
    KnowledgeSourceSpec,
    LifecycleConfig,
    RewardConfig,
    RewardTermSpec,
    TrainingConfig,
)
from game_learning_runtime.training_safety import GuardedRewardResult, RewardSafetyConfig

MEASURED_SCHEMA = "glr.fleet.measured.v1"
MEASURED_CAPTURE_SCHEMA = "glr.fleet.measured-capture.v1"
MAX_MEASURED_ENVELOPE_BYTES = 1_048_576
_NS_MAX = 2**63 - 1
_CLOCK_DOMAIN = "unix-utc-ns"
_KINDS = {"synthetic_contract_fixture", "owner_authorized_local_measured"}
_INFO = {OBSERVATION_CONTEXT_KEY, "observation_sequence", REWARD_EVIDENCE_KEY}
_PROVENANCE = {"correlated_reward", "correlated_reward_sha256"}
_TRANSITION_FIELDS = {
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
}
_ACTION_FIELDS = {item.name for item in fields(ActionReceipt)}
_CONTEXT_FIELDS = {item.name for item in fields(ObservationContext)}
_CLAIM_FIELDS = {item.name for item in fields(RewardAttribution)}
_RESULT_FIELDS = {item.name for item in fields(GuardedRewardResult)}
_RECEIPT_FIELDS = {item.name for item in fields(CorrelatedRewardReceipt)}
_CAPTURE_FIELDS = {
    "schema_version",
    "source",
    "shard_seq",
    "produced_at_utc_ms",
    "expires_at_utc_ms",
    "unroll",
    "steps",
}
_BODY_FIELDS = {
    "grant_id",
    "grant_sha256",
    "key_id",
    "source_spec_sha256",
    "exporter_source_sha256",
    "training_config_sha256",
    "safety_config_sha256",
    "evaluation_domain_id",
    "evidence_kind",
    "clock_domain",
    "carrier_manifest_sha256",
    "carrier_payload_sha256",
    "capture",
}


def _fail(reason: str) -> NoReturn:
    raise FleetError(reason)


def _closed(value: object, keys: set[str], reason: str = "measured_fields") -> Mapping[str, Any]:
    if type(value) not in (dict, MappingProxyType) or set(cast(Mapping[str, Any], value)) != keys:
        _fail(reason)
    return cast(Mapping[str, Any], value)


def _integer(value: object, *, minimum: int = 0, maximum: int = MAX_COUNTER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail("measured_counter")
    return value


def _number(value: object, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        _fail("measured_number")
    try:
        result = float(cast(float, value))
    except (OverflowError, ValueError):
        _fail("measured_number")
    if not math.isfinite(result) or (nonnegative and result < 0):
        _fail("measured_number")
    return result


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        _fail("measured_boolean")
    return value


def _text(value: object, *, maximum: int = 128) -> str:
    if type(value) is not str or not 1 <= len(value) <= maximum:
        _fail("measured_text")
    return value


def _id(value: object) -> str:
    return identifier(_text(value, maximum=96))


def _sha(value: object, length: int = 64) -> str:
    return digest(_text(value, maximum=length), length)


def _uuid(value: object) -> UUID:
    text = _text(value, maximum=36)
    try:
        parsed = UUID(text)
    except ValueError:
        _fail("measured_episode")
    if str(parsed) != text:
        _fail("measured_episode")
    return parsed


def _items(value: object, *, maximum: int = 256, minimum: int = 0) -> tuple[Any, ...]:
    if (
        type(value) not in (list, tuple)
        or not minimum <= len(cast(tuple[Any, ...], value)) <= maximum
    ):
        _fail("measured_list")
    return tuple(cast(tuple[Any, ...], value))


def _names(value: object, *, maximum: int = 64) -> tuple[str, ...]:
    result = tuple(_id(item) for item in _items(value, maximum=maximum))
    if len(set(result)) != len(result):
        _fail("measured_duplicates")
    return result


def _json(value: object, *, depth: int = 0, budget: list[int] | None = None) -> Any:
    """Copy native metadata without arbitrary exporters, objects, or coercion."""
    if budget is None:
        budget = [4096, MAX_MEASURED_ENVELOPE_BYTES]
    budget[0] -= 1
    budget[1] -= 8
    if depth > 16 or min(budget) < 0:
        _fail("measured_metadata_bound")
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        _number(value)
        return value
    if type(value) is str:
        if len(value) > 4096:
            _fail("measured_metadata_bound")
        budget[1] -= len(value.encode("utf-8"))
        if budget[1] < 0:
            _fail("measured_metadata_bound")
        return value
    if type(value) in (dict, MappingProxyType):
        mapping = cast(Mapping[str, Any], value)
        if len(mapping) > 256 or any(type(key) is not str or len(key) > 128 for key in mapping):
            _fail("measured_metadata_bound")
        return {key: _json(item, depth=depth + 1, budget=budget) for key, item in mapping.items()}
    if type(value) in (list, tuple):
        return [_json(item, depth=depth + 1, budget=budget) for item in _items(value)]
    _fail("measured_metadata_type")


def _wire_json(value: object, *, depth: int = 0, budget: list[int] | None = None) -> Any:
    """Thaw bounded parser containers without calling arbitrary exporters."""
    if budget is None:
        budget = [32768, MAX_MEASURED_ENVELOPE_BYTES]
    budget[0] -= 1
    if budget[0] < 0 or depth > 24:
        _fail("measured_envelope_bound")
    if type(value) is str:
        if len(value) > MAX_MEASURED_ENVELOPE_BYTES:
            _fail("measured_envelope_bound")
        budget[1] -= len(value.encode("utf-8"))
        if budget[1] < 0:
            _fail("measured_envelope_bound")
        return value
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        _number(value)
        return value
    if type(value) in (dict, MappingProxyType):
        mapping = cast(Mapping[str, Any], value)
        if len(mapping) > 512 or any(type(key) is not str for key in mapping):
            _fail("measured_envelope_bound")
        for key in mapping:
            if len(key) > 128:
                _fail("measured_envelope_bound")
            budget[1] -= len(key.encode("utf-8"))
        if budget[1] < 0:
            _fail("measured_envelope_bound")
        return {
            key: _wire_json(item, depth=depth + 1, budget=budget) for key, item in mapping.items()
        }
    if type(value) in (tuple, list):
        return [
            _wire_json(item, depth=depth + 1, budget=budget) for item in _items(value, maximum=8192)
        ]
    _fail("measured_metadata_type")


def _source_record(source: SourceSpec) -> dict[str, Any]:
    if type(source) is not SourceSpec or type(source.compatibility) is not CompatibilitySpec:
        _fail("measured_contract_type")
    for name in SourceSpec.__dataclass_fields__:
        if name == "compatibility":
            continue
        value = getattr(source, name)
        if name == "policy_version":
            _integer(value)
        elif name == "simulated":
            _boolean(value)
        elif name == "checkpoint_sha256":
            if value is not None:
                _sha(value)
        else:
            _text(value)
    spec = source.compatibility
    if type(spec.reward_length) is not int or spec.reward_length != 1:
        _fail("measured_scalar_reward")
    for layout in (spec.observation, spec.action, spec.masks):
        if type(layout) is not tuple or len(layout) > 32:
            _fail("measured_layout")
        for vector in layout:
            if type(vector) is not VectorSpec:
                _fail("measured_contract_type")
            _id(vector.name)
            _text(vector.dtype)
            _integer(vector.length, minimum=1, maximum=8192)
    result = SourceSpec.to_record(source)
    _json(result)
    return result


def _source_from_record(value: object) -> SourceSpec:
    record = _closed(value, set(SourceSpec.__dataclass_fields__))
    _json(record)
    try:
        source = SourceSpec.from_record(record)
    except (ValueError, TypeError, KeyError):
        _fail("measured_source")
    _source_record(source)
    if canonical(record) != canonical(source.to_record()):
        _fail("measured_source")
    return source


def _training_record(training: TrainingConfig) -> dict[str, Any]:
    if (
        type(training) is not TrainingConfig
        or type(training.lifecycle) is not LifecycleConfig
        or type(training.bridge) is not BridgeConfig
        or type(training.knowledge_injection) is not KnowledgeInjectionConfig
        or type(training.reward) is not RewardConfig
        or type(training.knowledge_sources) is not tuple
        or type(training.reward.terms) is not tuple
    ):
        _fail("measured_config_type")
    if len(training.reward.terms) > 256 or len(training.knowledge_sources) > 64:
        _fail("measured_config_bound")
    sources = []
    for source in training.knowledge_sources:
        if (
            type(source) is not KnowledgeSourceSpec
            or type(source.authority) is not KnowledgeAuthority
        ):
            _fail("measured_config_type")
        sources.append(
            {
                "id": source.source_id,
                "authority": source.authority.value,
                "required": _boolean(source.required),
                "provides_context": _boolean(source.provides_context),
                "max_age_seconds": source.max_age_seconds,
                "max_payload_bytes": source.max_payload_bytes,
            }
        )
    terms = []
    for term in training.reward.terms:
        if (
            type(term) is not RewardTermSpec
            or type(term.minimum_authority) is not KnowledgeAuthority
        ):
            _fail("measured_config_type")
        terms.append(
            {
                "name": term.name,
                "source": term.source,
                "weight": _number(term.weight),
                "minimum": None if term.minimum is None else _number(term.minimum),
                "maximum": None if term.maximum is None else _number(term.maximum),
                "required": _boolean(term.required),
                "minimum_authority": term.minimum_authority.value,
            }
        )
    injection = training.knowledge_injection
    if type(injection.allowed_intents) is not frozenset or any(
        type(item) is not KnowledgeIntent for item in injection.allowed_intents
    ):
        _fail("measured_config_type")
    result = {
        "schema_version": training.schema_version,
        "lifecycle": {
            "start_mode": training.lifecycle.start_mode,
            "stop_on_done": _boolean(training.lifecycle.stop_on_done),
        },
        "bridge": {"required_capabilities": sorted(training.bridge.required_capabilities)},
        "knowledge_sources": sources,
        "knowledge_injection": {
            "enabled": _boolean(injection.enabled),
            "allowed_intents": sorted(item.value for item in injection.allowed_intents),
            "max_items": _integer(injection.max_items, minimum=1),
            "min_confidence": _number(injection.min_confidence),
        },
        "reward": {
            "terms": terms,
            "minimum": None
            if training.reward.minimum is None
            else _number(training.reward.minimum),
            "maximum": None
            if training.reward.maximum is None
            else _number(training.reward.maximum),
        },
    }
    try:
        if TrainingConfig.from_mapping(result) != training:
            _fail("measured_config")
    except (ValueError, TypeError):
        _fail("measured_config")
    _json(result)
    return result


def _safety_record(safety: RewardSafetyConfig) -> dict[str, Any]:
    if type(safety) is not RewardSafetyConfig:
        _fail("measured_config_type")
    result = {item.name: getattr(safety, item.name) for item in fields(RewardSafetyConfig)}
    for name in ("shaping_signals", "unbudgeted_signals"):
        result[name] = list(_names(result[name], maximum=256))
    _json(result)
    _boolean(safety.require_terminal_outcome)
    for name in (
        "max_positive_shaping_per_step",
        "max_positive_shaping_per_episode",
        "failure_episode_maximum",
        "max_negative_shaping_per_step",
        "max_negative_shaping_per_episode",
    ):
        if result[name] is not None:
            _number(result[name])
    try:
        if RewardSafetyConfig.from_mapping(result) != safety:
            _fail("measured_config")
    except (ValueError, TypeError):
        _fail("measured_config")
    return result


@dataclass(frozen=True, slots=True)
class MeasuredActionBinding:
    action_key: str
    mask_key: str
    command_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _id(self.action_key)
        _id(self.mask_key)
        commands = _names(self.command_ids, maximum=256)
        if not commands or type(self.command_ids) is not tuple:
            _fail("measured_action_mapping")


@dataclass(frozen=True, slots=True)
class MeasuredQuality:
    before_fresh: bool
    after_fresh: bool
    mask_fresh: bool
    observation_quality: str
    reward_quality: str
    legal_mask_quality: str

    def __post_init__(self) -> None:
        for name in ("before_fresh", "after_fresh", "mask_fresh"):
            _boolean(getattr(self, name))
        for name in ("observation_quality", "reward_quality", "legal_mask_quality"):
            _text(getattr(self, name))


@dataclass(frozen=True, slots=True)
class LegalActionEvidence:
    command_ids: tuple[str, ...]
    observation_sequence: int
    action_mask_sha256: str

    def __post_init__(self) -> None:
        if type(self.command_ids) is not tuple:
            _fail("measured_action_mapping")
        _names(self.command_ids, maximum=32)
        _integer(self.observation_sequence)
        _sha(self.action_mask_sha256)


@dataclass(frozen=True, slots=True)
class RewardBudgetState:
    episode_total: float
    positive_shaping_total: float
    negative_shaping_total: float
    suppressed_positive_shaping_total: float
    suppressed_negative_shaping_total: float
    action_count: int
    closed: bool

    def __post_init__(self) -> None:
        for name in (
            "episode_total",
            "positive_shaping_total",
            "negative_shaping_total",
            "suppressed_positive_shaping_total",
            "suppressed_negative_shaping_total",
        ):
            _number(getattr(self, name), nonnegative=name != "episode_total")
        _integer(self.action_count, maximum=4096)
        _boolean(self.closed)


@dataclass(frozen=True, slots=True)
class MeasuredStep:
    receipt: CorrelatedRewardReceipt
    life_id: str
    budget_before: RewardBudgetState
    budget_after: RewardBudgetState
    quality: MeasuredQuality
    legal_action: LegalActionEvidence

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not CorrelatedRewardReceipt
            or type(self.budget_before) is not RewardBudgetState
            or type(self.budget_after) is not RewardBudgetState
            or type(self.quality) is not MeasuredQuality
            or type(self.legal_action) is not LegalActionEvidence
        ):
            _fail("measured_contract_type")
        _id(self.life_id)
        _receipt_record(self.receipt)

    @property
    def before(self) -> ObservationContext:
        return self.receipt.before

    @property
    def after(self) -> ObservationContext:
        return self.receipt.after

    @property
    def result(self) -> GuardedRewardResult:
        return self.receipt.result


@dataclass(frozen=True, slots=True)
class MeasuredGrant:
    grant_id: str
    key_id: str
    source: SourceSpec
    exporter_source_sha256: str
    target_id: str
    clock_domain: str
    training: TrainingConfig
    safety: RewardSafetyConfig
    action_bindings: tuple[MeasuredActionBinding, ...]
    evaluation_domain_id: str
    evidence_kind: str
    max_age_ms: int
    max_clock_skew_ms: int
    expires_at_utc_ms: int
    max_actions_per_episode: int = 4096
    extra_info_keys: tuple[str, ...] = ()
    extra_provenance_keys: tuple[str, ...] = ()
    event_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("grant_id", "key_id", "target_id", "evaluation_domain_id"):
            _id(getattr(self, name))
        _sha(self.exporter_source_sha256)
        _source_record(self.source)
        if (
            self.source.simulated
            or self.clock_domain != _CLOCK_DOMAIN
            or self.evidence_kind not in _KINDS
        ):
            _fail("measured_grant_profile")
        _text(self.clock_domain)
        _text(self.evidence_kind)
        _integer(self.max_age_ms, minimum=1)
        _integer(self.max_clock_skew_ms)
        _integer(self.expires_at_utc_ms, minimum=1, maximum=_NS_MAX // 1_000_000)
        _integer(self.max_actions_per_episode, minimum=1, maximum=4096)
        _training_record(self.training)
        _safety_record(self.safety)
        source_authorities = {
            item.source_id: item.authority for item in self.training.knowledge_sources
        }
        if any(
            term.minimum_authority is not KnowledgeAuthority.AUTHORITATIVE
            or source_authorities.get(term.source) is not KnowledgeAuthority.AUTHORITATIVE
            for term in self.training.reward.terms
        ):
            _fail("measured_reward_authority")
        if self.training.reward.minimum is not None and self.training.reward.minimum > 0:
            _fail("measured_positive_global_minimum")
        terms = {term.name: term for term in self.training.reward.terms}
        safety = self.safety
        classified = (
            set(safety.shaping_signals) | set(safety.unbudgeted_signals) | {safety.outcome_signal}
        )
        if (
            not terms
            or set(terms) != classified
            or safety.outcome_signal not in terms
            or terms[safety.outcome_signal].weight <= 0
        ):
            _fail("measured_reward_classification")
        if type(self.action_bindings) is not tuple or not 1 <= len(self.action_bindings) <= 32:
            _fail("measured_action_mapping")
        spec = self.source.compatibility
        actions = {item.name: item for item in spec.action}
        masks = {item.name: item for item in spec.masks}
        if (
            any(type(item) is not MeasuredActionBinding for item in self.action_bindings)
            or {item.action_key for item in self.action_bindings} != set(actions)
            or {item.mask_key for item in self.action_bindings} != set(masks)
            or len(self.action_bindings) != len(actions)
            or len(self.action_bindings) != len(masks)
        ):
            _fail("measured_action_mapping")
        for item in self.action_bindings:
            action, mask = actions[item.action_key], masks[item.mask_key]
            if (
                action.dtype not in {"<i4", "<i8"}
                or action.length != 1
                or mask.length != len(item.command_ids)
            ):
                _fail("measured_action_mapping")
        for name in ("extra_info_keys", "extra_provenance_keys", "event_names"):
            if type(getattr(self, name)) is not tuple:
                _fail("measured_metadata_grant")
            _names(getattr(self, name))
        if set(self.extra_info_keys) & _INFO or set(self.extra_provenance_keys) & _PROVENANCE:
            _fail("measured_metadata_grant")

    @property
    def source_spec_sha256(self) -> str:
        return sha256(canonical(_source_record(self.source)))

    @property
    def training_config_sha256(self) -> str:
        return sha256(canonical(_training_record(self.training)))

    @property
    def safety_config_sha256(self) -> str:
        return sha256(canonical(_safety_record(self.safety)))

    @property
    def sha256(self) -> str:
        return sha256(canonical(self.to_record()))

    def to_record(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(MeasuredGrant)}
        result["source"] = _source_record(self.source)
        result["training"] = _training_record(self.training)
        result["safety"] = _safety_record(self.safety)
        result["action_bindings"] = [_plain(item) for item in self.action_bindings]
        for name in ("extra_info_keys", "extra_provenance_keys", "event_names"):
            result[name] = list(result[name])
        return result


def _key(key: object) -> bytes:
    if type(key) is not bytes or not 32 <= len(key) <= 4096:
        _fail("measured_key")
    return key


class MeasuredAuthority:
    """Immutable opaque RAM authority; generic dataclass exporters cannot dump keys."""

    __slots__ = ("_authority_id", "_grants", "_keys", "_sha256")
    _authority_id: str
    _grants: tuple[MeasuredGrant, ...]
    _keys: Mapping[str, bytes]
    _sha256: str

    def __init__(
        self, authority_id: str, grants: tuple[MeasuredGrant, ...], keys: Mapping[str, bytes]
    ) -> None:
        _id(authority_id)
        if type(grants) is not tuple or not 1 <= len(grants) <= 64:
            _fail("measured_grants")
        if type(keys) not in (dict, MappingProxyType) or not 1 <= len(keys) <= 64:
            _fail("measured_keys")
        private = {_id(name): _key(key) for name, key in keys.items()}
        if (
            any(type(grant) is not MeasuredGrant for grant in grants)
            or len({grant.grant_id for grant in grants}) != len(grants)
            or len({(grant.source.source_id, grant.source.source_epoch) for grant in grants})
            != len(grants)
            or {grant.key_id for grant in grants} != set(private)
        ):
            _fail("measured_grants")
        domains: dict[str, bytes] = {}
        for grant in grants:
            semantics = canonical(grant.source.compatibility.to_record())
            if domains.setdefault(grant.evaluation_domain_id, semantics) != semantics:
                _fail("measured_evaluation_domain")
        public_grants: list[dict[str, Any]] = [grant.to_record() for grant in grants]
        ordered_grants = sorted(public_grants, key=lambda item: _id(item["grant_id"]))
        binding = sha256(
            canonical(
                {
                    "authority_id": authority_id,
                    "grants": ordered_grants,
                    "key_ids": sorted(private),
                }
            )
        )
        object.__setattr__(self, "_authority_id", authority_id)
        object.__setattr__(self, "_grants", grants)
        object.__setattr__(self, "_keys", MappingProxyType(private))
        object.__setattr__(self, "_sha256", binding)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("measured authority is immutable")

    def __reduce_ex__(self, protocol: Any) -> Any:
        raise TypeError("measured authority is not serializable")

    def __repr__(self) -> str:
        return "MeasuredAuthority(keys=<private>)"

    @property
    def authority_id(self) -> str:
        return self._authority_id

    @property
    def grants(self) -> tuple[MeasuredGrant, ...]:
        return self._grants

    @property
    def sha256(self) -> str:
        return self._sha256


@dataclass(frozen=True, slots=True)
class MeasuredCapture:
    source: SourceSpec
    shard_seq: int
    produced_at_utc_ms: int
    expires_at_utc_ms: int
    unroll: Unroll
    steps: tuple[MeasuredStep, ...]


@dataclass(frozen=True, slots=True)
class EncodedMeasuredShard:
    carrier: EncodedShard
    envelope: bytes


@dataclass(frozen=True, slots=True)
class VerifiedMeasuredHeader:
    grant: MeasuredGrant
    envelope_sha256: str
    expires_at_utc_ms: int
    authority_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedMeasuredShard:
    decoded: DecodedShard
    carrier_decoded: DecodedShard
    steps: tuple[MeasuredStep, ...]
    grant: MeasuredGrant
    envelope_sha256: str
    expires_at_utc_ms: int
    authority_sha256: str

    @property
    def reward_steps(self) -> tuple[MeasuredStep, ...]:
        return self.steps


def _plain(value: object) -> dict[str, Any]:
    return {item.name: getattr(value, item.name) for item in fields(cast(Any, value))}


def _context_record(context: ObservationContext) -> dict[str, Any]:
    if type(context) is not ObservationContext or type(context.phase) is not EnvironmentPhase:
        _fail("measured_contract_type")
    result = ObservationContext.to_mapping(context)
    _context_from_record(result)
    return result


def _context_from_record(value: object) -> ObservationContext:
    data = dict(_closed(value, _CONTEXT_FIELDS))
    for name in ("run_id", "environment_id", "protocol_version", "target_id"):
        _id(data[name])
    _sha(data["environment_config_sha256"])
    data["episode_id"] = _uuid(data["episode_id"])
    for name in ("step_id", "producer_sequence"):
        _integer(data[name])
    _integer(data["timestamp_ns"], maximum=_NS_MAX)
    _text(data["phase"])
    if data["alive"] is not None:
        _boolean(data["alive"])
    try:
        data["phase"] = EnvironmentPhase(data["phase"])
        return ObservationContext(**data)
    except (ValueError, TypeError):
        _fail("measured_context")


def _claim_record(claim: RewardAttribution) -> dict[str, Any]:
    if type(claim) is not RewardAttribution or type(claim.effect) is not EffectState:
        _fail("measured_contract_type")
    result = RewardAttribution.to_mapping(claim)
    _claim_from_record(result)
    return result


def _claim_from_record(value: object) -> RewardAttribution:
    data = dict(_closed(value, _CLAIM_FIELDS))
    for name in ("signal_name", "source"):
        _id(data[name])
    _text(data["action_id"])
    for name in ("before_sequence", "after_sequence"):
        _integer(data[name])
    _text(data["effect"])
    try:
        data["effect"] = EffectState(data["effect"])
        return RewardAttribution(**data)
    except (ValueError, TypeError):
        _fail("measured_attribution")


def _result_record(result: GuardedRewardResult) -> dict[str, Any]:
    if type(result) is not GuardedRewardResult:
        _fail("measured_contract_type")
    data = _plain(result)
    for name in _RESULT_FIELDS - {"contributions", "terminal"}:
        _number(data[name], nonnegative=name not in {"total", "episode_total"})
    _boolean(result.terminal)
    if (
        type(result.contributions) not in (dict, MappingProxyType)
        or len(result.contributions) > 257
    ):
        _fail("measured_contributions")
    data["contributions"] = {
        _text(name): _number(value) for name, value in result.contributions.items()
    }
    return data


def _receipt_record(receipt: CorrelatedRewardReceipt) -> dict[str, Any]:
    if type(receipt) is not CorrelatedRewardReceipt or type(receipt.outcome) is not ActionOutcome:
        _fail("measured_contract_type")
    if receipt.observed_reward is None:
        _fail("measured_unverified_reward")
    _number(receipt.observed_reward)
    for name in ("state_sha256", "action_sha256", "next_state_sha256"):
        _sha(getattr(receipt, name))
    _text(receipt.action_id)
    return {
        "before": _context_record(receipt.before),
        "after": _context_record(receipt.after),
        "action_id": receipt.action_id,
        "outcome": receipt.outcome.value,
        "result": _result_record(receipt.result),
        "attributions": [_claim_record(item) for item in _items(receipt.attributions)],
        "state_sha256": receipt.state_sha256,
        "action_sha256": receipt.action_sha256,
        "next_state_sha256": receipt.next_state_sha256,
        "observed_reward": receipt.observed_reward,
    }


def _receipt_from_record(value: object) -> CorrelatedRewardReceipt:
    data = dict(_closed(value, _RECEIPT_FIELDS))
    data["before"] = _context_from_record(data["before"])
    data["after"] = _context_from_record(data["after"])
    _text(data["action_id"])
    _text(data["outcome"])
    result = dict(_closed(data["result"], _RESULT_FIELDS))
    for name in _RESULT_FIELDS - {"contributions", "terminal"}:
        _number(result[name], nonnegative=name not in {"total", "episode_total"})
    _boolean(result["terminal"])
    contributions = result["contributions"]
    if type(contributions) is not dict or len(contributions) > 257:
        _fail("measured_contributions")
    for name, number in contributions.items():
        _text(name)
        _number(number)
    data["result"] = GuardedRewardResult(**result)
    data["attributions"] = tuple(_claim_from_record(item) for item in _items(data["attributions"]))
    for name in ("state_sha256", "action_sha256", "next_state_sha256"):
        _sha(data[name])
    if data["observed_reward"] is None:
        _fail("measured_unverified_reward")
    _number(data["observed_reward"])
    try:
        data["outcome"] = ActionOutcome(data["outcome"])
        return CorrelatedRewardReceipt(**data)
    except (ValueError, TypeError):
        _fail("measured_reward_receipt")


def _step_record(step: MeasuredStep) -> dict[str, Any]:
    if type(step) is not MeasuredStep:
        _fail("measured_contract_type")
    return {
        "receipt": _receipt_record(step.receipt),
        "life_id": step.life_id,
        "budget_before": _plain(step.budget_before),
        "budget_after": _plain(step.budget_after),
        "quality": _plain(step.quality),
        "legal_action": {
            **_plain(step.legal_action),
            "command_ids": list(step.legal_action.command_ids),
        },
    }


def _step_from_record(value: object) -> MeasuredStep:
    data = _closed(value, {item.name for item in fields(MeasuredStep)})
    _id(data["life_id"])
    before = _closed(data["budget_before"], {item.name for item in fields(RewardBudgetState)})
    after = _closed(data["budget_after"], {item.name for item in fields(RewardBudgetState)})
    quality = _closed(data["quality"], {item.name for item in fields(MeasuredQuality)})
    legal = dict(_closed(data["legal_action"], {item.name for item in fields(LegalActionEvidence)}))
    legal["command_ids"] = tuple(_items(legal["command_ids"], maximum=32))
    return MeasuredStep(
        _receipt_from_record(data["receipt"]),
        data["life_id"],
        RewardBudgetState(**before),
        RewardBudgetState(**after),
        MeasuredQuality(**quality),
        LegalActionEvidence(**legal),
    )


def _action_from_record(value: object) -> ActionReceipt:
    data = dict(_closed(value, _ACTION_FIELDS))
    _text(data["action_id"])
    data["episode_id"] = _uuid(data["episode_id"])
    _integer(data["step_id"], minimum=1)
    for name in ("issued_timestamp_ns", "observed_timestamp_ns"):
        _integer(data[name], maximum=_NS_MAX)
    for name in ("authoritative_observation_sequence", "issued_against_observation_sequence"):
        if data[name] is not None:
            _integer(data[name])
    _text(data["postcondition"])
    if data["progress_delta"] is not None:
        _number(data["progress_delta"])
    _boolean(data["retryable"])
    if data["target_id"] is not None:
        _id(data["target_id"])
    if data["reason_class"] is not None:
        _text(data["reason_class"])
    _text(data["outcome"])
    realtime = data["realtime"]
    if realtime is not None:
        rt = dict(_closed(realtime, {item.name for item in fields(RealtimeActionReceipt)}))
        _text(rt["action_id"])
        _text(rt["status"])
        for name in ("deadline_ns", "quantum_ns"):
            _integer(rt[name], minimum=1, maximum=_NS_MAX)
        _integer(rt["issued_at_ns"], maximum=_NS_MAX)
        for name in ("consumed_at_ns", "settled_at_ns"):
            if rt[name] is not None:
                _integer(rt[name], maximum=_NS_MAX)
        if rt["cancellation_token"] is not None:
            _fail("measured_privileged_metadata")
        try:
            rt["status"] = RealtimeActionStatus(rt["status"])
            data["realtime"] = RealtimeActionReceipt(**rt)
        except (ValueError, TypeError):
            _fail("measured_realtime")
    try:
        data["outcome"] = ActionOutcome(data["outcome"])
        if data["reason_class"] is not None:
            data["reason_class"] = RefusalReasonClass(data["reason_class"])
        return ActionReceipt(**data)
    except (ValueError, TypeError):
        _fail("measured_action_receipt")


def _numeric_record(record: Mapping[str, Any]) -> dict[str, Any]:
    _closed(record, _TRANSITION_FIELDS)
    return {**record, "action_receipt": None, "events": [], "info": {}, "provenance": None}


def _restore_transition(record: object, numeric: Transition) -> Transition:
    data = _closed(record, _TRANSITION_FIELDS)
    if canonical(_numeric_record(data)) != canonical(transition_to_record(numeric)):
        _fail("measured_carrier_projection")
    events = []
    for item in _items(data["events"], maximum=64):
        event = _closed(item, {"name", "timestamp_ns", "payload"})
        _id(event["name"])
        _integer(event["timestamp_ns"], maximum=_NS_MAX)
        payload = _json(event["payload"])
        if type(payload) is not dict:
            _fail("measured_event")
        events.append(
            Event(name=event["name"], timestamp_ns=event["timestamp_ns"], payload=payload)
        )
    info, provenance = _json(data["info"]), _json(data["provenance"])
    if type(info) is not dict or type(provenance) is not dict:
        _fail("measured_metadata")
    return replace(
        numeric,
        action_receipt=_action_from_record(data["action_receipt"]),
        events=tuple(events),
        info=info,
        provenance=provenance,
    )


def _snapshot(value: object) -> dict[str, str]:
    if type(value) not in (dict, MappingProxyType) or len(cast(Mapping[str, Any], value)) > 64:
        _fail("measured_environment_snapshot")
    result = {}
    for name, text in cast(Mapping[str, Any], value).items():
        _text(name)
        if type(text) is not str or len(text) > 4096:
            _fail("measured_environment_snapshot")
        result[name] = text
    return result


def _capture_record(capture: MeasuredCapture) -> dict[str, Any]:
    unroll = capture.unroll
    if type(unroll) is not Unroll or type(capture.steps) is not tuple:
        _fail("measured_contract_type")
    snapshot = _snapshot(unroll.environment_config_snapshot)
    if (
        unroll.actor_id != capture.source.source_id
        or unroll.sequence_id != capture.shard_seq
        or unroll.policy_version != capture.source.policy_version
        or environment_config_digest(snapshot)
        != capture.source.compatibility.environment_config_sha256
        or unroll.environment_config_digest
        != capture.source.compatibility.environment_config_sha256
    ):
        _fail("measured_unroll_provenance")
    records = []
    metadata_budget = [32768, MAX_MEASURED_ENVELOPE_BYTES]
    for transition in unroll.transitions:
        if (
            type(transition) is not Transition
            or type(transition.action_receipt) is not ActionReceipt
        ):
            _fail("measured_contract_type")
        if (
            transition.action_receipt.realtime is not None
            and type(transition.action_receipt.realtime) is not RealtimeActionReceipt
        ):
            _fail("measured_contract_type")
        if any(type(event) is not Event for event in transition.events):
            _fail("measured_contract_type")
        # Validate metadata before serialization can call arbitrary object exporters.
        _json(transition.info, budget=metadata_budget)
        _json(transition.provenance, budget=metadata_budget)
        for event in transition.events:
            _json(event.payload, budget=metadata_budget)
        try:
            record = transition_to_record(transition)
        except (TypeError, ValueError, OverflowError):
            _fail("measured_transition")
        _action_from_record(record["action_receipt"])
        records.append(record)
    return {
        "schema_version": MEASURED_CAPTURE_SCHEMA,
        "source": _source_record(capture.source),
        "shard_seq": _integer(capture.shard_seq),
        "produced_at_utc_ms": _integer(capture.produced_at_utc_ms, maximum=_NS_MAX // 1_000_000),
        "expires_at_utc_ms": _integer(capture.expires_at_utc_ms, maximum=_NS_MAX // 1_000_000),
        "unroll": {
            "actor_id": unroll.actor_id,
            "sequence_id": unroll.sequence_id,
            "policy_version": unroll.policy_version,
            "environment_config_snapshot": snapshot,
            "environment_config_digest": unroll.environment_config_digest,
            "transitions": records,
        },
        "steps": [_step_record(step) for step in capture.steps],
    }


def require_measured_capture(
    value: Mapping[str, object], *, limits: FleetLimits = DEFAULT_LIMITS
) -> MeasuredCapture:
    """Strict preflight independent of registration; this does not admit a source."""
    if type(limits) is not FleetLimits:
        _fail("measured_contract_type")
    data = _closed(value, _CAPTURE_FIELDS, "missing_measured_capture")
    data = _wire_json(data)
    if len(canonical(data)) > min(MAX_MEASURED_ENVELOPE_BYTES, limits.max_shard_bytes):
        _fail("measured_envelope_bound")
    if data["schema_version"] != MEASURED_CAPTURE_SCHEMA or type(data["schema_version"]) is not str:
        _fail("measured_capture_schema")
    source = _source_from_record(data["source"])
    if source.simulated:
        _fail("measured_grant_profile")
    seq = _integer(data["shard_seq"])
    produced = _integer(data["produced_at_utc_ms"], maximum=_NS_MAX // 1_000_000)
    expires = _integer(data["expires_at_utc_ms"], minimum=1, maximum=_NS_MAX // 1_000_000)
    if expires <= produced:
        _fail("measured_expiry")
    unroll = _closed(data["unroll"], {item.name for item in fields(Unroll)})
    _id(unroll["actor_id"])
    _integer(unroll["sequence_id"])
    _integer(unroll["policy_version"])
    _sha(unroll["environment_config_digest"])
    snapshot = _snapshot(unroll["environment_config_snapshot"])
    if (
        unroll["actor_id"] != source.source_id
        or unroll["sequence_id"] != seq
        or unroll["policy_version"] != source.policy_version
        or environment_config_digest(snapshot) != source.compatibility.environment_config_sha256
        or unroll["environment_config_digest"] != source.compatibility.environment_config_sha256
    ):
        _fail("measured_unroll_provenance")
    records = _items(unroll["transitions"], maximum=min(128, limits.max_transitions), minimum=1)
    numeric_payload = (
        b"\n".join(
            canonical(_numeric_record(_closed(item, _TRANSITION_FIELDS))) for item in records
        )
        + b"\n"
    )
    if len(numeric_payload) > limits.max_shard_bytes:
        _fail("measured_envelope_bound")
    chunks = tuple(
        numeric_payload[index : index + limits.max_chunk_bytes]
        for index in range(0, len(numeric_payload), limits.max_chunk_bytes)
    )
    manifest = canonical(
        {
            "schema": "glr.fleet.shard.v1",
            "source": source.to_record(),
            "shard_seq": seq,
            "produced_at_utc_ms": produced,
            "transition_count": len(records),
            "payload_sha256": sha256(numeric_payload),
            "payload_bytes": len(numeric_payload),
            "chunks": [{"sha256": sha256(chunk), "bytes": len(chunk)} for chunk in chunks],
        }
    )
    numeric = decode_shard(manifest, numeric_payload, limits=limits)
    original = tuple(
        _restore_transition(record, item)
        for record, item in zip(records, numeric.unroll.transitions, strict=True)
    )
    steps = tuple(
        _step_from_record(item)
        for item in _items(data["steps"], maximum=min(128, limits.max_transitions), minimum=1)
    )
    if len(steps) != len(original):
        _fail("measured_proof_count")
    return MeasuredCapture(
        source,
        seq,
        produced,
        expires,
        Unroll(
            original,
            source.source_id,
            seq,
            source.policy_version,
            snapshot,
            source.compatibility.environment_config_sha256,
        ),
        steps,
    )


def sign_measured_body(body: Mapping[str, object], key: bytes) -> str:
    """Sign a native bounded body; semantic validation belongs to verification."""
    if type(body) not in (dict, MappingProxyType):
        _fail("measured_fields")
    payload = canonical(_wire_json(body))
    if len(payload) > MAX_MEASURED_ENVELOPE_BYTES:
        _fail("measured_envelope_bound")
    return hmac.new(
        _key(key), MEASURED_SCHEMA.encode() + b"\0" + payload, hashlib.sha256
    ).hexdigest()


def _envelope(envelope: bytes, limits: FleetLimits) -> Mapping[str, Any]:
    if type(limits) is not FleetLimits or type(envelope) is not bytes:
        _fail("measured_contract_type")
    try:
        parsed = parse_json_object(
            envelope,
            max_bytes=min(MAX_MEASURED_ENVELOPE_BYTES, limits.max_shard_bytes),
            max_depth=24,
        ).data
    except OfflineParseError:
        _fail("measured_envelope_json")
    record = _wire_json(parsed)
    wrapper = _closed(record, {"schema_version", "body", "signature"})
    if wrapper["schema_version"] != MEASURED_SCHEMA or type(wrapper["schema_version"]) is not str:
        _fail("measured_envelope_schema")
    _sha(wrapper["signature"])
    _closed(wrapper["body"], _BODY_FIELDS)
    if canonical(record) != envelope:
        _fail("measured_envelope_canonical")
    return wrapper


def _header(
    envelope: bytes,
    manifest: bytes | ShardManifest,
    authority: MeasuredAuthority,
    now_ms: int,
    limits: FleetLimits,
) -> tuple[VerifiedMeasuredHeader, Mapping[str, Any], ShardManifest]:
    if type(authority) is not MeasuredAuthority:
        _fail("measured_authority_required")
    now = _integer(now_ms, maximum=_NS_MAX // 1_000_000)
    if type(manifest) is bytes:
        description = parse_manifest(manifest, limits)
    elif type(manifest) is ShardManifest:
        description = manifest
    else:
        _fail("measured_contract_type")
    _integer(description.shard_seq)
    _integer(description.produced_at_utc_ms, maximum=_NS_MAX // 1_000_000)
    _integer(description.transition_count, minimum=1, maximum=min(128, limits.max_transitions))
    _integer(description.payload_bytes, minimum=1, maximum=limits.max_shard_bytes)
    _sha(description.payload_sha256)
    _sha(description.manifest_sha256)
    if type(description.chunks) is not tuple:
        _fail("measured_contract_type")
    checked_chunks = []
    for item in _items(description.chunks, minimum=1, maximum=limits.max_upload_chunks):
        if type(item) is not tuple or len(item) != 2:
            _fail("measured_fields")
        checked_chunks.append(
            {
                "sha256": _sha(item[0]),
                "bytes": _integer(item[1], minimum=1, maximum=limits.max_chunk_bytes),
            }
        )
    manifest_record = {
        "schema": "glr.fleet.shard.v1",
        "source": _source_record(description.source),
        "shard_seq": description.shard_seq,
        "produced_at_utc_ms": description.produced_at_utc_ms,
        "transition_count": description.transition_count,
        "payload_sha256": description.payload_sha256,
        "payload_bytes": description.payload_bytes,
        "chunks": checked_chunks,
    }
    canonical_manifest = canonical(manifest_record)
    if sha256(canonical_manifest) != description.manifest_sha256:
        _fail("measured_manifest_canonical")
    parse_manifest(canonical_manifest, limits)
    wrapper = _envelope(envelope, limits)
    body = wrapper["body"]
    _id(body["grant_id"])
    grant = next((item for item in authority.grants if item.grant_id == body["grant_id"]), None)
    if grant is None:
        _fail("measured_unregistered_source")
    # Authenticate the complete body before parsing any transition proof.
    if not hmac.compare_digest(
        wrapper["signature"], sign_measured_body(body, authority._keys[grant.key_id])
    ):
        _fail("measured_signature")
    expected = {
        "grant_id": grant.grant_id,
        "grant_sha256": grant.sha256,
        "key_id": grant.key_id,
        "source_spec_sha256": grant.source_spec_sha256,
        "exporter_source_sha256": grant.exporter_source_sha256,
        "training_config_sha256": grant.training_config_sha256,
        "safety_config_sha256": grant.safety_config_sha256,
        "evaluation_domain_id": grant.evaluation_domain_id,
        "evidence_kind": grant.evidence_kind,
        "clock_domain": grant.clock_domain,
        "carrier_manifest_sha256": description.manifest_sha256,
        "carrier_payload_sha256": description.payload_sha256,
    }
    if any(type(body[name]) is not str or body[name] != value for name, value in expected.items()):
        _fail("measured_grant_binding")
    if _source_record(description.source) != _source_record(grant.source):
        _fail("measured_grant_binding")
    capture = _closed(body["capture"], _CAPTURE_FIELDS, "missing_measured_capture")
    if (
        canonical(capture["source"]) != canonical(grant.source.to_record())
        or _integer(capture["shard_seq"]) != description.shard_seq
        or _integer(capture["produced_at_utc_ms"]) != description.produced_at_utc_ms
    ):
        _fail("measured_carrier_binding")
    expires = _integer(capture["expires_at_utc_ms"], minimum=1, maximum=_NS_MAX // 1_000_000)
    if (
        expires > grant.expires_at_utc_ms
        or expires <= description.produced_at_utc_ms
        or now >= expires
        or now >= grant.expires_at_utc_ms
    ):
        _fail("measured_expired")
    if description.produced_at_utc_ms > now + grant.max_clock_skew_ms:
        _fail("measured_clock_future")
    if now - description.produced_at_utc_ms > grant.max_age_ms:
        _fail("measured_stale")
    return (
        VerifiedMeasuredHeader(grant, sha256(envelope), expires, authority.sha256),
        body,
        description,
    )


def verify_measured_manifest(
    envelope: bytes,
    manifest: bytes | ShardManifest,
    authority: MeasuredAuthority,
    now_ms: int,
    *,
    limits: FleetLimits = DEFAULT_LIMITS,
) -> VerifiedMeasuredHeader:
    """Authenticate/bind the header before reservation; full proof is still required."""
    return _header(envelope, manifest, authority, now_ms, limits)[0]


def _clip(value: float, lower: float | None, upper: float | None) -> float:
    if lower is not None:
        value = max(value, lower)
    if upper is not None:
        value = min(value, upper)
    return value


def _signals_and_claims(transition: Transition, step: MeasuredStep) -> dict[str, Mapping[str, Any]]:
    evidence = _closed(transition.info[REWARD_EVIDENCE_KEY], {"signals", "attributions"})
    signals = {}
    for item in _items(evidence["signals"]):
        signal = _closed(item, {"name", "source", "value"})
        name = _id(signal["name"])
        _id(signal["source"])
        _number(signal["value"])
        if name in signals:
            _fail("measured_duplicate_signal")
        signals[name] = signal
    claims = tuple(_claim_from_record(item) for item in _items(evidence["attributions"]))
    if claims != step.receipt.attributions or len({claim.signal_name for claim in claims}) != len(
        claims
    ):
        _fail("measured_attribution_binding")
    for claim in claims:
        if claim.effect is EffectState.UNKNOWN:
            _fail("measured_unknown_effect")
        claimed_signal = signals.get(claim.signal_name)
        if (
            claimed_signal is None
            or claim.source != claimed_signal["source"]
            or claim.action_id != step.receipt.action_id
            or claim.before_sequence != step.before.producer_sequence
            or claim.after_sequence != step.after.producer_sequence
        ):
            _fail("measured_attribution_binding")
    return signals


def _check_reward(transition: Transition, step: MeasuredStep, grant: MeasuredGrant) -> None:
    """Check frozen-config arithmetic against original counters; no guard/compose call."""
    before, after, result = step.budget_before, step.budget_after, step.result
    if before.closed or before.action_count >= grant.max_actions_per_episode:
        _fail("measured_budget_closed")
    if before.action_count != transition.step_id:
        _fail("measured_budget_action_count")
    zero = RewardBudgetState(0.0, 0.0, 0.0, 0.0, 0.0, 0, False)
    if transition.step_id == 0 and before != zero:
        _fail("measured_budget_reset")
    safety = grant.safety
    if before.positive_shaping_total > safety.max_positive_shaping_per_episode or (
        safety.max_negative_shaping_per_episode is not None
        and before.negative_shaping_total > safety.max_negative_shaping_per_episode
    ):
        _fail("measured_budget_limit")
    signals = _signals_and_claims(transition, step)
    terms = {term.name: term for term in grant.training.reward.terms}
    if set(signals) - set(terms) or any(
        term.required and term.name not in signals for term in terms.values()
    ):
        _fail("measured_reward_signals")
    contributions = {}
    for term in grant.training.reward.terms:
        signal = signals.get(term.name)
        if signal is not None:
            if signal["source"] != term.source:
                _fail("measured_reward_source")
            contributions[term.name] = (
                _clip(_number(signal["value"]), term.minimum, term.maximum) * term.weight
            )
    outcome = signals.get(safety.outcome_signal)
    if (outcome is not None and not transition.done) or (
        transition.done and safety.require_terminal_outcome and outcome is None
    ):
        _fail("measured_terminal_outcome")
    claims = {claim.signal_name: claim for claim in step.receipt.attributions}
    if (
        transition.done
        and outcome is not None
        and outcome["value"] < 0
        and safety.outcome_signal not in claims
        and step.after.alive is not False
    ):
        _fail("measured_terminal_life_evidence")
    for name, value in contributions.items():
        if value > 0 and (
            name not in claims
            or claims[name].effect is not EffectState.CONFIRMED
            or step.after.alive is not True
        ):
            _fail("measured_positive_effect")
    if step.after.alive is False and (
        any(value > 0 for value in contributions.values())
        or (outcome is not None and outcome["value"] > 0)
    ):
        _fail("measured_dead_positive")
    positive = math.fsum(
        value
        for name, value in contributions.items()
        if name in safety.shaping_signals and value > 0
    )
    negative = -math.fsum(
        value
        for name, value in contributions.items()
        if name in safety.shaping_signals and value < 0
    )
    accepted = min(
        positive,
        safety.max_positive_shaping_per_step,
        max(0.0, safety.max_positive_shaping_per_episode - before.positive_shaping_total),
    )
    negative_budget = math.inf
    if safety.max_negative_shaping_per_step is not None:
        negative_budget = min(negative_budget, safety.max_negative_shaping_per_step)
    if safety.max_negative_shaping_per_episode is not None:
        negative_budget = min(
            negative_budget,
            max(0.0, safety.max_negative_shaping_per_episode - before.negative_shaping_total),
        )
    negative_accepted = min(negative, negative_budget)
    for name in safety.shaping_signals:
        shaping_value = contributions.get(name)
        if shaping_value is not None and shaping_value > 0:
            contributions[name] = shaping_value * (accepted / positive)
        elif shaping_value is not None and shaping_value < 0:
            contributions[name] = shaping_value * (negative_accepted / negative)
    total = _clip(
        math.fsum(contributions.values()),
        grant.training.reward.minimum,
        grant.training.reward.maximum,
    )
    episode_total = before.episode_total + total
    if (
        transition.done
        and outcome is not None
        and outcome["value"] < 0
        and episode_total > safety.failure_episode_maximum
    ):
        correction = safety.failure_episode_maximum - episode_total
        contributions["guardrail.failure-correction"] = correction
        total += correction
        episode_total = safety.failure_episode_maximum
    expected_result = {
        "total": total,
        "contributions": contributions,
        "episode_total": episode_total,
        "positive_shaping_total": before.positive_shaping_total + accepted,
        "negative_shaping_total": before.negative_shaping_total + negative_accepted,
        "suppressed_positive_shaping": positive - accepted,
        "suppressed_negative_shaping": negative - negative_accepted,
        "terminal": transition.done,
    }
    if _result_record(result) != expected_result:
        _fail("measured_reward_arithmetic")
    expected_budget = RewardBudgetState(
        episode_total,
        before.positive_shaping_total + accepted,
        before.negative_shaping_total + negative_accepted,
        before.suppressed_positive_shaping_total + result.suppressed_positive_shaping,
        before.suppressed_negative_shaping_total + result.suppressed_negative_shaping,
        before.action_count + 1,
        transition.done,
    )
    if after != expected_budget:
        _fail("measured_budget_arithmetic")
    actual = transition.reward
    if (
        type(actual) is not np.ndarray
        or actual.dtype.str not in {"<f4", "<f8"}
        or actual.shape != (1,)
    ):
        _fail("measured_scalar_reward")
    observed = float(actual.item())
    with np.errstate(over="ignore", invalid="ignore"):
        quantized = float(np.asarray(result.total, dtype=actual.dtype).item())
    if (
        not math.isfinite(quantized)
        or observed != quantized
        or step.receipt.observed_reward != observed
    ):
        _fail("measured_observed_reward")


def _check_step(
    transition: Transition,
    step: MeasuredStep,
    grant: MeasuredGrant,
    now_ms: int,
    produced_at_utc_ms: int,
) -> None:
    _receipt_record(step.receipt)
    quality = step.quality
    if (
        not quality.before_fresh
        or not quality.after_fresh
        or not quality.mask_fresh
        or any(
            getattr(quality, name) != "authoritative"
            for name in ("observation_quality", "reward_quality", "legal_mask_quality")
        )
    ):
        _fail("measured_quality")
    if (
        set(transition.info) != _INFO | set(grant.extra_info_keys)
        or transition.provenance is None
        or set(transition.provenance) != _PROVENANCE | set(grant.extra_provenance_keys)
    ):
        _fail("measured_metadata_grant")
    if any(event.name not in grant.event_names for event in transition.events):
        _fail("measured_event_grant")
    if (
        _context_from_record(transition.info[OBSERVATION_CONTEXT_KEY]) != step.after
        or _integer(transition.info["observation_sequence"]) != step.after.producer_sequence
        or canonical(_wire_json(transition.provenance["correlated_reward"]))
        != canonical(step.receipt.to_mapping())
        or type(transition.provenance["correlated_reward_sha256"]) is not str
        or transition.provenance["correlated_reward_sha256"] != step.receipt.sha256
    ):
        _fail("measured_original_proof_binding")
    before = TimeStep(
        observation=transition.observation,
        reward=np.zeros_like(transition.reward),
        terminated=np.zeros_like(transition.terminated),
        truncated=np.zeros_like(transition.truncated),
        episode_id=transition.episode_id,
        step_id=transition.step_id,
        action_mask=transition.action_mask,
        timestamp_ns=step.before.timestamp_ns,
        info={
            OBSERVATION_CONTEXT_KEY: step.before.to_mapping(),
            "observation_sequence": step.before.producer_sequence,
        },
    )
    after = TimeStep(
        observation=transition.next_observation,
        reward=transition.reward,
        terminated=transition.terminated,
        truncated=transition.truncated,
        episode_id=transition.episode_id,
        step_id=transition.step_id + 1,
        action_mask=transition.next_action_mask,
        timestamp_ns=transition.timestamp_ns,
        action_receipt=transition.action_receipt,
        events=transition.events,
        info=transition.info,
    )
    spec = grant.source.compatibility
    policy = CorrelationPolicy(
        grant.source.run_id,
        spec.environment_id,
        spec.protocol_version,
        grant.target_id,
        spec.environment_config_sha256,
    )
    try:
        policy.validate_interval(before, after)
    except (ValueError, TypeError):
        _fail("measured_action_interval")
    action = transition.action_receipt
    if (
        action is None
        or type(action) is not ActionReceipt
        or action.outcome is not ActionOutcome.ACCEPTED
    ):
        _fail("measured_action_not_accepted")
    if (
        step.receipt.action_id != action.action_id
        or step.receipt.outcome is not action.outcome
        or action.postcondition.strip().lower() in {"", "unknown"}
        or action.retryable
        or action.reason_class is not None
    ):
        _fail("measured_action_binding")
    realtime = action.realtime
    if (
        type(realtime) is not RealtimeActionReceipt
        or realtime.status is not RealtimeActionStatus.CONSUMED
        or realtime.action_id != action.action_id
        or realtime.issued_at_ns != action.issued_timestamp_ns
        or realtime.consumed_at_ns is None
        or realtime.settled_at_ns is None
        or realtime.cancellation_token is not None
        or not realtime.issued_at_ns
        <= realtime.consumed_at_ns
        <= realtime.settled_at_ns
        <= action.observed_timestamp_ns
        or realtime.settled_at_ns > realtime.issued_at_ns + realtime.deadline_ns
    ):
        _fail("measured_realtime_unsettled")
    for context in (step.before, step.after):
        now_ns = now_ms * 1_000_000
        if context.timestamp_ns > now_ns + grant.max_clock_skew_ms * 1_000_000:
            _fail("measured_clock_future")
        if now_ns - context.timestamp_ns > grant.max_age_ms * 1_000_000:
            _fail("measured_stale")
        if context.timestamp_ns > (produced_at_utc_ms + grant.max_clock_skew_ms) * 1_000_000:
            _fail("measured_produced_before_observation")
    if (
        tensor_tree_sha256(transition.observation) != step.receipt.state_sha256
        or tensor_tree_sha256(transition.action) != step.receipt.action_sha256
        or tensor_tree_sha256(transition.next_observation) != step.receipt.next_state_sha256
    ):
        _fail("measured_tensor_binding")
    commands = []
    for binding in grant.action_bindings:
        index = int(cast(np.ndarray[Any, Any], transition.action[binding.action_key]).item())
        mask = cast(
            np.ndarray[Any, Any], cast(Mapping[str, Any], transition.action_mask)[binding.mask_key]
        )
        if not 0 <= index < len(binding.command_ids) or not bool(mask[index]):
            _fail("measured_illegal_action")
        commands.append(binding.command_ids[index])
    if (
        tuple(commands) != step.legal_action.command_ids
        or step.legal_action.observation_sequence != step.before.producer_sequence
        or tensor_tree_sha256(cast(Any, transition.action_mask))
        != step.legal_action.action_mask_sha256
    ):
        _fail("measured_legal_mapping")
    _check_reward(transition, step, grant)


def verify_measured(
    envelope: bytes,
    decoded: DecodedShard,
    authority: MeasuredAuthority,
    now_ms: int,
    *,
    limits: FleetLimits = DEFAULT_LIMITS,
) -> VerifiedMeasuredShard:
    """Verify original proof and return proof-bearing learner transitions."""
    if (
        type(decoded) is not DecodedShard
        or type(decoded.manifest) is not ShardManifest
        or type(decoded.unroll) is not Unroll
        or type(limits) is not FleetLimits
    ):
        _fail("measured_contract_type")
    if len(decoded.unroll.transitions) != _integer(
        decoded.manifest.transition_count, minimum=1, maximum=min(128, limits.max_transitions)
    ):
        _fail("measured_proof_count")
    _source_record(decoded.manifest.source)
    _check_live_tensors(decoded.unroll, decoded.manifest.source, limits)
    source = decoded.manifest.source
    if (
        type(decoded.unroll.actor_id) is not str
        or decoded.unroll.actor_id != f"{source.source_id}:{source.source_epoch}"
        or _integer(decoded.unroll.sequence_id) != decoded.manifest.shard_seq
        or _integer(decoded.unroll.policy_version) != source.policy_version
        or decoded.unroll.environment_config_snapshot is not None
        or type(decoded.unroll.environment_config_digest) is not str
        or decoded.unroll.environment_config_digest
        != source.compatibility.environment_config_sha256
        or any(
            item.info
            or item.events
            or item.provenance is not None
            or item.action_receipt is not None
            for item in decoded.unroll.transitions
        )
    ):
        _fail("measured_carrier_metadata")
    header, body, _ = _header(envelope, decoded.manifest, authority, now_ms, limits)
    capture = require_measured_capture(body["capture"], limits=limits)
    # Re-encode only the independent numeric carrier to compare all original
    # arrays/dtypes/masks/metadata identities. No reward guard is constructed.
    numeric_unroll = replace(
        capture.unroll,
        transitions=tuple(
            replace(item, action_receipt=None, events=(), info={}, provenance=None)
            for item in capture.unroll.transitions
        ),
    )
    carrier = encode_shard(
        capture.source,
        shard_seq=capture.shard_seq,
        unroll=numeric_unroll,
        produced_at_utc_ms=capture.produced_at_utc_ms,
        limits=limits,
    )
    actual = decode_shard(carrier.manifest, carrier.payload, limits=limits)
    if (
        actual.manifest != decoded.manifest
        or len(actual.unroll.transitions) != len(decoded.unroll.transitions)
        or any(
            canonical(transition_to_record(left)) != canonical(transition_to_record(right))
            for left, right in zip(
                actual.unroll.transitions, decoded.unroll.transitions, strict=True
            )
        )
    ):
        _fail("measured_carrier_binding")
    action_ids = set()
    for transition, step in zip(capture.unroll.transitions, capture.steps, strict=True):
        if step.receipt.action_id in action_ids:
            _fail("measured_duplicate_action")
        action_ids.add(step.receipt.action_id)
        _check_step(transition, step, header.grant, now_ms, capture.produced_at_utc_ms)
    for previous, current in pairwise(capture.steps):
        if (
            previous.budget_after.closed
            or current.budget_before != previous.budget_after
            or current.before != previous.after
            or current.life_id != previous.life_id
            or current.receipt.state_sha256 != previous.receipt.next_state_sha256
        ):
            _fail("measured_proof_continuity")
    return VerifiedMeasuredShard(
        DecodedShard(decoded.manifest, capture.unroll),
        actual,
        capture.steps,
        header.grant,
        header.envelope_sha256,
        header.expires_at_utc_ms,
        header.authority_sha256,
    )


def _check_live_tensors(unroll: Unroll, source: SourceSpec, limits: FleetLimits) -> None:
    """Reject oversized live arrays before projection constructors copy them."""
    spec = source.compatibility
    if not 1 <= len(unroll.transitions) <= min(128, limits.max_transitions):
        _fail("measured_proof_count")
    leaves = 2 * len(spec.observation) + len(spec.action) + 2 * len(spec.masks) + 3
    if leaves > min(DEFAULT_LIMITS.max_leaves, limits.max_leaves):
        _fail("measured_tensor_bound")
    raw_bytes = 0

    def array(value: object, dtype: str, length: int) -> None:
        nonlocal raw_bytes
        if (
            type(value) is not np.ndarray
            or value.dtype.str != dtype
            or value.shape != (length,)
            or value.size > min(DEFAULT_LIMITS.max_vector_length, limits.max_vector_length)
            or value.nbytes > min(DEFAULT_LIMITS.max_tensor_bytes, limits.max_tensor_bytes)
        ):
            _fail("measured_tensor_bound")
        raw_bytes += value.nbytes
        if raw_bytes > min(MAX_MEASURED_ENVELOPE_BYTES, limits.max_shard_bytes):
            _fail("measured_tensor_bound")
        if not np.all(np.isfinite(value)):
            _fail("measured_tensor_nonfinite")

    def tree(value: object, layout: tuple[VectorSpec, ...]) -> None:
        mapping = _closed(value, {item.name for item in layout})
        for vector in layout:
            array(mapping[vector.name], vector.dtype, vector.length)

    for transition in unroll.transitions:
        if type(transition) is not Transition:
            _fail("measured_contract_type")
        if type(transition.episode_id) is not UUID:
            _fail("measured_episode")
        _integer(transition.step_id)
        _integer(transition.timestamp_ns, maximum=_NS_MAX)
        tree(transition.observation, spec.observation)
        tree(transition.next_observation, spec.observation)
        tree(transition.action, spec.action)
        tree(transition.action_mask, spec.masks)
        tree(transition.next_action_mask, spec.masks)
        array(transition.reward, spec.reward_dtype, 1)
        array(transition.terminated, "|b1", 1)
        array(transition.truncated, "|b1", 1)


def encode_measured_shard(
    source: SourceSpec,
    unroll: Unroll,
    *,
    grant: MeasuredGrant,
    key: bytes,
    shard_seq: int,
    produced_at_utc_ms: int,
    expires_at_utc_ms: int,
    steps: tuple[MeasuredStep, ...],
    limits: FleetLimits = DEFAULT_LIMITS,
) -> EncodedMeasuredShard:
    """Retain original typed proof and sign a separate numeric transport carrier."""
    if type(grant) is not MeasuredGrant or _source_record(source) != _source_record(grant.source):
        _fail("measured_grant_binding")
    if type(limits) is not FleetLimits or type(unroll) is not Unroll:
        _fail("measured_contract_type")
    if not 1 <= len(unroll.transitions) <= min(128, limits.max_transitions) or len(steps) != len(
        unroll.transitions
    ):
        _fail("measured_proof_count")
    _check_live_tensors(unroll, source, limits)
    # Numeric validation runs before base64 encoding the full proof-bearing rows.
    numeric_unroll = replace(
        unroll,
        transitions=tuple(
            replace(item, action_receipt=None, events=(), info={}, provenance=None)
            for item in unroll.transitions
        ),
    )
    carrier = encode_shard(
        source,
        shard_seq=shard_seq,
        unroll=numeric_unroll,
        produced_at_utc_ms=produced_at_utc_ms,
        limits=limits,
    )
    capture = _capture_record(
        MeasuredCapture(source, shard_seq, produced_at_utc_ms, expires_at_utc_ms, unroll, steps)
    )
    body = {
        "grant_id": grant.grant_id,
        "grant_sha256": grant.sha256,
        "key_id": grant.key_id,
        "source_spec_sha256": grant.source_spec_sha256,
        "exporter_source_sha256": grant.exporter_source_sha256,
        "training_config_sha256": grant.training_config_sha256,
        "safety_config_sha256": grant.safety_config_sha256,
        "evaluation_domain_id": grant.evaluation_domain_id,
        "evidence_kind": grant.evidence_kind,
        "clock_domain": grant.clock_domain,
        "carrier_manifest_sha256": sha256(carrier.manifest),
        "carrier_payload_sha256": sha256(carrier.payload),
        "capture": capture,
    }
    envelope = canonical(
        {
            "schema_version": MEASURED_SCHEMA,
            "body": body,
            "signature": sign_measured_body(body, key),
        }
    )
    if len(envelope) > min(MAX_MEASURED_ENVELOPE_BYTES, limits.max_shard_bytes):
        _fail("measured_envelope_bound")
    # Export validation uses the same original proof path as ingest, at the
    # explicit producer timestamp. It creates neither keys nor training work.
    authority = MeasuredAuthority("export-validation", (grant,), {grant.key_id: key})
    verify_measured(
        envelope,
        decode_shard(carrier.manifest, carrier.payload, limits=limits),
        authority,
        produced_at_utc_ms,
        limits=limits,
    )
    return EncodedMeasuredShard(carrier, envelope)


__all__ = [
    "MAX_MEASURED_ENVELOPE_BYTES",
    "MEASURED_CAPTURE_SCHEMA",
    "MEASURED_SCHEMA",
    "EncodedMeasuredShard",
    "LegalActionEvidence",
    "MeasuredActionBinding",
    "MeasuredAuthority",
    "MeasuredCapture",
    "MeasuredGrant",
    "MeasuredQuality",
    "MeasuredStep",
    "RewardBudgetState",
    "VerifiedMeasuredHeader",
    "VerifiedMeasuredShard",
    "encode_measured_shard",
    "require_measured_capture",
    "sign_measured_body",
    "verify_measured",
    "verify_measured_manifest",
]
