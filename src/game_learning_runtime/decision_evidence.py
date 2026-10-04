"""Bounded decision diagnostics, without raw observations, commands or private reasoning.

All evidence is diagnostic. Binding checks establish consistency of supplied
records, not external game effects, actual table reads or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, cast

import numpy as np

from game_learning_runtime.contracts import ActionReceipt, Transition
from game_learning_runtime.correlated_rewards import CorrelatedRewardReceipt, tensor_tree_sha256
from game_learning_runtime.knowledge_evidence import DecisionConsumptionReceipt, RuleIndexBinding
from game_learning_runtime.run_store import RunRecord

DECISION_EVIDENCE_SCHEMA = "glr.decision-evidence.v1"
DECISION_EVENT_KIND = "agent.decision.evidence"
MAX_EVIDENCE_BYTES = 64 * 1024
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_RULE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}")
_SHA = re.compile(r"[0-9a-f]{64}")
_BASIS = {None, "score_order", "exploration", "rule_requirement", "legal_mask", "policy_reported"}
_REASON = {None, "mask_rejected", "capability_missing", "stale_observation", "not_legal", "unknown"}


def _fields(value: Any, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("evidence requires exactly the declared safe fields")
    return value


def _label(value: Any, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or _LABEL.fullmatch(value) is None:
        raise ValueError("evidence identifiers must be bounded portable labels")


def _sha(value: Any) -> None:
    if value is not None and (not isinstance(value, str) or _SHA.fullmatch(value) is None):
        raise ValueError("evidence digest must be a lowercase SHA-256 or unknown")


def _number(value: Any, *, integer: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ValueError("evidence quantities must be numbers or unknown")
    try:
        finite = math.isfinite(value)
    except OverflowError as error:
        raise ValueError("evidence quantity exceeds its numeric range") from error
    if not finite or (integer and value < 0):
        raise ValueError("evidence quantities must be finite and counts nonnegative")


def _bool(value: Any) -> None:
    if value is not None and type(value) is not bool:
        raise ValueError("evidence flag must be bool or unknown")


def _plain(value: Any, *, depth: int = 0) -> Any:
    """Detach once into plain bounded data before any validation or JSON export."""
    if depth > 16:
        raise ValueError("evidence nesting exceeds its budget")
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Mapping):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 257 or type(key) is not str or key in result:
                raise ValueError("evidence maps need bounded unique plain string keys")
            result[key] = _plain(item, depth=depth + 1)
        return result
    if type(value) in (list, tuple) and len(value) <= 257:
        return [_plain(item, depth=depth + 1) for item in value]
    raise ValueError("evidence contains a non-data object")


def _validated(value: Mapping[str, Any]) -> str:
    value = _plain(value)
    data = _fields(
        value,
        {
            "schema_version",
            "identity",
            "observation",
            "selection",
            "rules",
            "execution",
            "reward",
            "outcome",
            "policy",
            "comparison",
        },
    )
    if data["schema_version"] != DECISION_EVIDENCE_SCHEMA:
        raise ValueError("unsupported decision evidence schema")
    identity = _fields(
        data["identity"],
        {
            "run_id",
            "environment_id",
            "protocol_version",
            "environment_config_sha256",
            "target_id",
            "episode_id",
            "step_id",
            "decision_id",
        },
    )
    _label(identity["run_id"])
    for name in ("environment_id", "protocol_version", "episode_id", "decision_id"):
        _label(identity[name], optional=True)
    _label(identity["target_id"], optional=True)
    _sha(identity["environment_config_sha256"])
    _number(identity["step_id"], integer=True)
    observation = _fields(
        data["observation"],
        {
            "producer_sequence",
            "timestamp_ns",
            "freshness",
            "confidence",
            "state_sha256",
            "lifecycle",
        },
    )
    for name in ("producer_sequence", "timestamp_ns"):
        _number(observation[name], integer=True)
    _number(observation["confidence"])
    if observation["confidence"] is not None and not 0 <= observation["confidence"] <= 1:
        raise ValueError("confidence must be in [0,1] or unknown")
    _sha(observation["state_sha256"])
    if observation["freshness"] not in {"fresh", "stale", "unknown"} or observation[
        "lifecycle"
    ] not in {"gameplay", "dead", "loading", "menu", "cutscene", "modal", "unavailable", "unknown"}:
        raise ValueError("unsupported observation evidence state")
    selection = _fields(
        data["selection"], {"candidates", "chosen_candidate_id", "action_sha256", "basis", "source"}
    )
    if selection["source"] not in {"unknown", "policy_reported"}:
        raise ValueError("selection source must remain explicit")
    _label(selection["chosen_candidate_id"], optional=True)
    _sha(selection["action_sha256"])
    if selection["basis"] not in _BASIS:
        raise ValueError("basis is an explicit bounded fact code, never private reasoning")
    candidates = selection["candidates"]
    if candidates is not None:
        if not isinstance(candidates, (list, tuple)) or len(candidates) > 64:
            raise ValueError("at most 64 explicitly reported candidates are accepted")
        ids = set()
        for candidate in candidates:
            item = _fields(candidate, {"id", "legal", "rejection_reason", "score", "basis"})
            _label(item["id"])
            _bool(item["legal"])
            _number(item["score"])
            if item["rejection_reason"] not in _REASON or item["basis"] not in _BASIS:
                raise ValueError("candidate reason and basis must be declared codes")
            if item["id"] in ids:
                raise ValueError("candidate identities must be unique")
            ids.add(item["id"])
        if (
            selection["chosen_candidate_id"] is not None
            and selection["chosen_candidate_id"] not in ids
        ):
            raise ValueError("chosen candidate must belong to the reported candidates")
    rules = _fields(data["rules"], {"source_id", "binding", "consumptions", "verification"})
    _label(rules["source_id"], optional=True)
    if rules["verification"] not in {"unknown", "binding_checked"}:
        raise ValueError("rule use is reported evidence, not authenticated table access")
    binding = None if rules["binding"] is None else RuleIndexBinding.from_mapping(rules["binding"])
    if binding is not None and _RULE_VERSION.fullmatch(binding.rules_version) is None:
        raise ValueError("safe rule revisions must use a bounded portable version label")
    if binding is not None and (binding.environment_id, binding.protocol_version) != (
        identity["environment_id"],
        identity["protocol_version"],
    ):
        raise ValueError("rules belong to a different environment or protocol")
    receipts = rules["consumptions"]
    if receipts is not None:
        if binding is None or not isinstance(receipts, (list, tuple)) or len(receipts) > 64:
            raise ValueError("consumption evidence requires a bound, bounded rule index")
        for item in receipts:
            receipt = DecisionConsumptionReceipt.from_mapping(item)
            receipt.assert_for_decision(
                decision_id=identity["decision_id"],
                finding_id=receipt.finding_id,
                finding_sha256=receipt.finding_sha256,
                consumer_id=receipt.consumer_id,
                binding=binding,
            )
    if rules["verification"] == "binding_checked" and (binding is None or receipts is None):
        raise ValueError("missing rule evidence cannot be binding checked")
    execution = data["execution"]
    if execution is not None:
        execution = _fields(
            execution,
            {
                "action_id",
                "outcome",
                "target_id",
                "before_sequence",
                "after_sequence",
                "issued_timestamp_ns",
                "observed_timestamp_ns",
            },
        )
        _label(execution["action_id"])
        _label(execution["target_id"], optional=True)
        from game_learning_runtime.contracts import ActionOutcome

        ActionOutcome(execution["outcome"])
        for name in (
            "before_sequence",
            "after_sequence",
            "issued_timestamp_ns",
            "observed_timestamp_ns",
        ):
            _number(execution[name], integer=True)
        if execution["target_id"] != identity["target_id"]:
            raise ValueError("execution target differs from decision identity")
    if observation["freshness"] != "unknown":
        if execution is None or any(
            execution[name] is None for name in ("before_sequence", "after_sequence")
        ):
            raise ValueError("freshness cannot be established without both receipt sequences")
        expected_freshness = (
            "fresh" if execution["after_sequence"] > execution["before_sequence"] else "stale"
        )
        if (
            observation["producer_sequence"] != execution["before_sequence"]
            or observation["freshness"] != expected_freshness
        ):
            raise ValueError("reported freshness disagrees with the captured receipt interval")
    reward = data["reward"]
    if reward is not None:
        reward = _fields(
            reward,
            {
                "receipt_sha256",
                "action_id",
                "before_sequence",
                "after_sequence",
                "terms",
                "total",
                "correlation",
                "lifecycle_before",
                "lifecycle_after",
                "next_state_sha256",
            },
        )
        _sha(reward["receipt_sha256"])
        _sha(reward["next_state_sha256"])
        _label(reward["action_id"])
        for name in ("before_sequence", "after_sequence"):
            _number(reward[name], integer=True)
        _number(reward["total"])
        terms = reward["terms"]
        if not isinstance(terms, Mapping) or len(terms) > 257:
            raise ValueError("reward terms must be a bounded named scalar projection")
        for name, amount in terms.items():
            _label(name)
            _number(amount)
        if reward["correlation"] == "binding_checked" and any(
            reward[name] is None
            for name in (
                "receipt_sha256",
                "next_state_sha256",
                "before_sequence",
                "after_sequence",
                "total",
            )
        ):
            raise ValueError("missing reward operands cannot be binding checked")
        if reward["correlation"] == "binding_checked" and any(
            value is None
            for value in (
                identity["environment_id"],
                identity["protocol_version"],
                identity["environment_config_sha256"],
                identity["target_id"],
                identity["episode_id"],
                identity["step_id"],
                observation["state_sha256"],
                selection["action_sha256"],
            )
        ):
            raise ValueError("missing source or tensor identities cannot be binding checked")
        if reward["correlation"] not in {"declared", "binding_checked"}:
            raise ValueError("unsupported reward binding level")
        if execution is None or any(
            reward[name] != execution[name]
            for name in ("action_id", "before_sequence", "after_sequence")
        ):
            raise ValueError("reward and action receipt have different identities")
        for name in ("lifecycle_before", "lifecycle_after"):
            if reward[name] not in {
                "gameplay",
                "dead",
                "loading",
                "menu",
                "cutscene",
                "modal",
                "unavailable",
                "unknown",
            }:
                raise ValueError("unsupported reward lifecycle")
    outcome = _fields(
        data["outcome"], {"terminated", "truncated", "success", "source_id", "evidence_sha256"}
    )
    for name in ("terminated", "truncated", "success"):
        _bool(outcome[name])
    _label(outcome["source_id"], optional=True)
    _sha(outcome["evidence_sha256"])
    if outcome["success"] is not None and (
        outcome["source_id"] is None or outcome["evidence_sha256"] is None
    ):
        raise ValueError("success needs explicit source evidence; otherwise it remains unknown")
    policy = _fields(data["policy"], {"version", "sha256", "checkpoint_sha256", "mode"})
    _number(policy["version"], integer=True)
    _sha(policy["sha256"])
    _sha(policy["checkpoint_sha256"])
    if policy["mode"] not in {"train", "evaluate", "unknown"}:
        raise ValueError("mode must be explicit or unknown")
    comparison = _fields(
        data["comparison"],
        {
            "status",
            "baseline_score",
            "candidate_score",
            "suite_sha256",
            "evaluator_sha256",
            "budget_steps",
            "direction",
            "minimum_improvement",
        },
    )
    for name in ("baseline_score", "candidate_score", "minimum_improvement"):
        _number(comparison[name])
    _number(comparison["budget_steps"], integer=True)
    for name in ("suite_sha256", "evaluator_sha256"):
        _sha(comparison[name])
    if comparison["direction"] not in {"min", "max", "unknown"}:
        raise ValueError("unsupported comparison direction")
    if comparison["minimum_improvement"] is not None and comparison["minimum_improvement"] < 0:
        raise ValueError("minimum improvement must be nonnegative even before comparison")
    if comparison["budget_steps"] is not None and comparison["budget_steps"] <= 0:
        raise ValueError("a declared comparison budget must be positive")
    expected_status = "unknown"
    if (
        all(
            comparison[name] is not None
            for name in (
                "baseline_score",
                "candidate_score",
                "suite_sha256",
                "evaluator_sha256",
                "budget_steps",
                "minimum_improvement",
            )
        )
        and comparison["direction"] != "unknown"
    ):
        if comparison["minimum_improvement"] < 0 or comparison["budget_steps"] <= 0:
            raise ValueError("comparison needs a fixed positive budget and nonnegative improvement")
        delta = comparison["candidate_score"] - comparison["baseline_score"]
        if comparison["direction"] == "min":
            delta = -delta
        expected_status = (
            "candidate_better"
            if delta > comparison["minimum_improvement"]
            else "tie"
            if delta == 0
            else "baseline_better"
            if delta < 0
            else "improvement_below_threshold"
        )
    if comparison["status"] != expected_status:
        raise ValueError("comparison status must agree with complete fixed inputs")
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_EVIDENCE_BYTES:
        raise ValueError("decision evidence exceeds its byte budget")
    return encoded


@dataclass(frozen=True, slots=True)
class DecisionEvidence:
    value: InitVar[Mapping[str, Any]]
    _json: str = field(init=False, repr=False)

    def __post_init__(self, value: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_json", _validated(value))

    def to_mapping(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._json))  # detached plain data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._json.encode("utf-8")).hexdigest()


def _captured_reward(transition: Transition, receipt: CorrelatedRewardReceipt) -> float:
    reward = transition.reward
    if reward.size != 1 or reward.dtype.kind != "f" or reward.dtype.itemsize not in (4, 8):
        raise ValueError("captured reward must be a finite float32 or float64 scalar")
    actual = float(reward.item())
    expected = receipt.observed_reward
    if expected is None:
        with np.errstate(over="ignore", invalid="ignore"):
            expected = float(np.asarray(receipt.result.total, dtype=reward.dtype).item())
    if not math.isfinite(actual) or not math.isfinite(expected) or actual != expected:
        raise ValueError("captured reward differs from the strict composition")
    return actual


def capture_transition(
    transition: Transition,
    *,
    run: RunRecord,
    decision_id: str | None = None,
    policy_version: int | None = None,
    mode: str = "unknown",
    policy_sha256: str | None = None,
    checkpoint_sha256: str | None = None,
    selection: Mapping[str, Any] | None = None,
    rule_source_id: str | None = None,
    rule_binding: RuleIndexBinding | None = None,
    consumptions: Sequence[DecisionConsumptionReceipt] | None = None,
    correlated_reward: CorrelatedRewardReceipt | None = None,
) -> DecisionEvidence:
    """Project only captured facts; missing policy internals remain unknown.

    No info/events, raw frames, commands, parameters, run metadata or arbitrary
    receipt details are copied. Explicit rule use remains a consumer declaration.
    """
    if type(transition) is not Transition or type(run) is not RunRecord:
        raise TypeError("base transition and run contracts are required")
    if rule_binding is not None and type(rule_binding) is not RuleIndexBinding:
        raise TypeError("rule binding must be a base RuleIndexBinding")
    if consumptions is not None and any(
        type(item) is not DecisionConsumptionReceipt for item in consumptions
    ):
        raise TypeError("consumption receipts must use the base data contract")
    receipt = transition.action_receipt
    if receipt is not None and (
        type(receipt) is not ActionReceipt
        or receipt.episode_id != transition.episode_id
        or receipt.step_id != transition.step_id + 1
    ):
        raise ValueError("execution receipt belongs to another transition")
    before_seq = None if receipt is None else receipt.issued_against_observation_sequence
    after_seq = None if receipt is None else receipt.authoritative_observation_sequence
    observation = {
        "producer_sequence": before_seq,
        "timestamp_ns": None,
        "confidence": None,
        "freshness": "unknown"
        if before_seq is None or after_seq is None
        else "fresh"
        if after_seq > before_seq
        else "stale",
        "state_sha256": tensor_tree_sha256(transition.observation),
        "lifecycle": "unknown",
    }
    action_sha = tensor_tree_sha256(transition.action)
    execution = (
        None
        if receipt is None
        else {
            "action_id": receipt.action_id,
            "outcome": receipt.outcome.value,
            "target_id": receipt.target_id,
            "before_sequence": before_seq,
            "after_sequence": after_seq,
            "issued_timestamp_ns": receipt.issued_timestamp_ns,
            "observed_timestamp_ns": receipt.observed_timestamp_ns,
        }
    )
    reward = None
    if correlated_reward is not None:
        if type(correlated_reward) is not CorrelatedRewardReceipt or receipt is None:
            raise TypeError("typed correlated reward needs a captured execution receipt")
        if (
            correlated_reward.before.run_id != run.run_id
            or correlated_reward.after.run_id != run.run_id
            or correlated_reward.before.episode_id != transition.episode_id
            or correlated_reward.before.step_id != transition.step_id
            or correlated_reward.after.episode_id != transition.episode_id
            or correlated_reward.after.step_id != transition.step_id + 1
            or correlated_reward.action_id != receipt.action_id
            or correlated_reward.action_sha256 != action_sha
            or correlated_reward.state_sha256 != observation["state_sha256"]
            or correlated_reward.next_state_sha256
            != tensor_tree_sha256(transition.next_observation)
            or correlated_reward.outcome != receipt.outcome
            or correlated_reward.result.terminal != transition.done
        ):
            raise ValueError("reward does not belong to the captured transition")
        for context in (correlated_reward.before, correlated_reward.after):
            if (
                context.environment_id != run.environment_id
                or context.protocol_version != run.protocol_version
                or context.environment_config_sha256 != run.environment_config_digest
                or context.target_id != receipt.target_id
            ):
                raise ValueError("reward source differs from the durable run binding")
        if (
            correlated_reward.before.producer_sequence != before_seq
            or correlated_reward.after.producer_sequence != after_seq
        ):
            raise ValueError("reward and execution intervals differ")
        captured_reward = _captured_reward(transition, correlated_reward)
        if not (
            correlated_reward.before.timestamp_ns
            <= receipt.issued_timestamp_ns
            <= receipt.observed_timestamp_ns
            <= correlated_reward.after.timestamp_ns
        ):
            raise ValueError("reward context timestamps disagree with execution")

        def lifecycle(context: Any) -> str:
            if context.alive is None or context.phase.value == "unknown":
                return "unknown"
            if context.phase.value != "gameplay":
                return str(context.phase.value)
            return "gameplay" if context.alive is True else "dead"

        observation.update(
            timestamp_ns=correlated_reward.before.timestamp_ns,
            lifecycle=lifecycle(correlated_reward.before),
        )
        reward = {
            "receipt_sha256": correlated_reward.sha256,
            "action_id": receipt.action_id,
            "before_sequence": before_seq,
            "after_sequence": after_seq,
            "terms": dict(correlated_reward.result.contributions),
            "total": captured_reward,
            "correlation": "declared"
            if correlated_reward.observed_reward is None
            else "binding_checked",
            "next_state_sha256": correlated_reward.next_state_sha256,
            "lifecycle_before": observation["lifecycle"],
            "lifecycle_after": lifecycle(correlated_reward.after),
        }
    declared_selection = {"candidates": None, "chosen_candidate_id": None, "basis": None}
    if selection is not None:
        declared_selection = dict(
            _fields(selection, {"candidates", "chosen_candidate_id", "basis"})
        )
    return DecisionEvidence(
        {
            "schema_version": DECISION_EVIDENCE_SCHEMA,
            "identity": {
                "run_id": run.run_id,
                "environment_id": run.environment_id,
                "protocol_version": run.protocol_version,
                "environment_config_sha256": run.environment_config_digest,
                "target_id": None if receipt is None else receipt.target_id,
                "episode_id": str(transition.episode_id),
                "step_id": transition.step_id,
                "decision_id": decision_id,
            },
            "observation": observation,
            "selection": {
                **declared_selection,
                "action_sha256": action_sha,
                "source": "unknown" if selection is None else "policy_reported",
            },
            "rules": {
                "source_id": rule_source_id,
                "binding": None
                if rule_binding is None
                else RuleIndexBinding.to_mapping(rule_binding),
                "consumptions": None
                if consumptions is None
                else [DecisionConsumptionReceipt.to_mapping(item) for item in consumptions],
                "verification": "unknown"
                if rule_binding is None or consumptions is None
                else "binding_checked",
            },
            "execution": execution,
            "reward": reward,
            "outcome": {
                "terminated": bool(transition.terminated.any()),
                "truncated": bool(transition.truncated.any()),
                "success": None,
                "source_id": None,
                "evidence_sha256": None,
            },
            "policy": {
                "version": policy_version,
                "sha256": policy_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "mode": mode,
            },
            "comparison": {
                "status": "unknown",
                "baseline_score": None,
                "candidate_score": None,
                "suite_sha256": None,
                "evaluator_sha256": None,
                "budget_steps": None,
                "direction": "unknown",
                "minimum_improvement": None,
            },
        }
    )
