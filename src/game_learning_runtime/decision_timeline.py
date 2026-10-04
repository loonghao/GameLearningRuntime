"""Read-only, bounded replay of safe decision evidence from existing run events.

Event ordering is presentation order. Only exact receipt bindings associate a
reward with a reported update; adjacent timestamps and step IDs never suffice.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from game_learning_runtime.decision_evidence import (
    DECISION_EVENT_KIND,
    MAX_EVIDENCE_BYTES,
    DecisionEvidence,
    _label,
    _number,
    _sha,
)
from game_learning_runtime.run_store import RunEvent

TIMELINE_SCHEMA = "glr.decision-timeline.v1"
MAX_TIMELINE_EVENTS = 250


def _binding(evidence: Mapping[str, Any]) -> tuple[Any, ...] | None:
    identity, reward = evidence["identity"], evidence["reward"]
    if reward is None or any(
        value is None
        for value in (
            reward["receipt_sha256"],
            identity["episode_id"],
            identity["step_id"],
            reward["before_sequence"],
            reward["after_sequence"],
            evidence["observation"]["state_sha256"],
            evidence["selection"]["action_sha256"],
            reward["next_state_sha256"],
        )
    ):
        return None
    return (
        identity["run_id"],
        identity["episode_id"],
        identity["step_id"],
        reward["receipt_sha256"],
        reward["action_id"],
        reward["before_sequence"],
        reward["after_sequence"],
    )


def _empty(run_id: str, episode: str | None, step: int | None) -> dict[str, Any]:
    return {
        "schema_version": "glr.decision-evidence.v1",
        "identity": {
            "run_id": run_id,
            "environment_id": None,
            "protocol_version": None,
            "environment_config_sha256": None,
            "target_id": None,
            "episode_id": episode,
            "step_id": step,
            "decision_id": None,
        },
        "observation": {
            "producer_sequence": None,
            "timestamp_ns": None,
            "freshness": "unknown",
            "confidence": None,
            "state_sha256": None,
            "lifecycle": "unknown",
        },
        "selection": {
            "candidates": None,
            "chosen_candidate_id": None,
            "action_sha256": None,
            "basis": None,
            "source": "unknown",
        },
        "rules": {
            "source_id": None,
            "binding": None,
            "consumptions": None,
            "verification": "unknown",
        },
        "execution": None,
        "reward": None,
        "outcome": {
            "terminated": None,
            "truncated": None,
            "success": None,
            "source_id": None,
            "evidence_sha256": None,
        },
        "policy": {"version": None, "sha256": None, "checkpoint_sha256": None, "mode": "unknown"},
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


def _episode(event: RunEvent) -> str | None:
    return event.episode_id.removeprefix("episode-") if event.episode_id is not None else None


def _legacy(event: RunEvent) -> dict[str, Any]:
    """Whitelist reported IDs; exclude raw state, commands and receipt extras."""
    value = _empty(event.run_id, _episode(event), event.step_id)
    selected = event.payload.get("selected_key")
    if selected is not None:
        _label(selected)
        value["selection"]["chosen_candidate_id"] = selected
        value["selection"]["source"] = "policy_reported"
    policy = event.payload.get("policy_digest")
    if isinstance(policy, str) and len(policy) == 64:
        _sha(policy)
        value["policy"]["sha256"] = policy
    mode = event.payload.get("mode")
    if mode in {"train", "evaluate"}:
        value["policy"]["mode"] = mode
    return DecisionEvidence(value).to_mapping()


def _reward(event: RunEvent) -> dict[str, Any]:
    payload = event.payload
    receipt = payload["receipt"]
    if not isinstance(receipt, Mapping):
        raise ValueError("reward receipt must be a mapping")
    digest = payload["receipt_sha256"]
    _sha(digest)
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if digest is None or hashlib.sha256(encoded.encode("utf-8")).hexdigest() != digest:
        raise ValueError("reward receipt checksum does not match its captured record")
    before, after = receipt["before"], receipt["after"]
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ValueError("reward contexts must be mappings")
    if (
        any(
            before[name] != after[name]
            for name in (
                "run_id",
                "environment_id",
                "protocol_version",
                "environment_config_sha256",
                "target_id",
                "episode_id",
            )
        )
        or before["run_id"] != event.run_id
    ):
        raise ValueError("reward contexts crossed a source identity")
    if (
        before["episode_id"] != _episode(event)
        or before["step_id"] != event.step_id
        or after["step_id"] != before["step_id"] + 1
    ):
        raise ValueError("reward event metadata differs from its receipt")
    value = _empty(event.run_id, before["episode_id"], before["step_id"])
    value["identity"].update(
        environment_id=before["environment_id"],
        protocol_version=before["protocol_version"],
        environment_config_sha256=before["environment_config_sha256"],
        target_id=before["target_id"],
    )

    def lifecycle(context: Mapping[str, Any]) -> str:
        if context.get("alive") is None or context.get("phase") == "unknown":
            return "unknown"
        if context["phase"] != "gameplay":
            return str(context["phase"])
        return "gameplay" if context["alive"] is True else "dead"

    value["observation"].update(
        producer_sequence=before["producer_sequence"],
        timestamp_ns=before["timestamp_ns"],
        state_sha256=receipt["state_sha256"],
        lifecycle=lifecycle(before),
        freshness="fresh" if after["producer_sequence"] > before["producer_sequence"] else "stale",
    )
    value["selection"]["action_sha256"] = receipt["action_sha256"]
    value["execution"] = {
        "action_id": receipt["action_id"],
        "outcome": receipt["outcome"],
        "target_id": before["target_id"],
        "before_sequence": before["producer_sequence"],
        "after_sequence": after["producer_sequence"],
        "issued_timestamp_ns": None,
        "observed_timestamp_ns": None,
    }
    value["reward"] = {
        "receipt_sha256": digest,
        "action_id": receipt["action_id"],
        "before_sequence": before["producer_sequence"],
        "after_sequence": after["producer_sequence"],
        "terms": receipt["contributions"],
        "total": receipt["reward"],
        "correlation": "declared",
        "next_state_sha256": receipt["next_state_sha256"],
        "lifecycle_before": lifecycle(before),
        "lifecycle_after": lifecycle(after),
    }
    return DecisionEvidence(value).to_mapping()


def project_events(events: Sequence[RunEvent], *, run_id: str) -> dict[str, Any]:
    """Project one scanned event window. Missing links stay visible and unverified."""
    _label(run_id)
    if len(events) > MAX_TIMELINE_EVENTS:
        raise ValueError("timeline window exceeds 250 scanned events")
    entries: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    updates: list[RunEvent] = []
    previous = -1
    for event in events:
        if type(event) is not RunEvent:
            raise ValueError("events must use the base run event contract")
        for value in (event.sequence_id, event.timestamp_ns):
            if type(value) is not int or value < 0:
                raise ValueError("event sequence and timestamp must be nonnegative native integers")
        _number(event.step_id, integer=True)
        _label(event.episode_id, optional=True)
        if type(event) is not RunEvent or event.run_id != run_id or event.sequence_id <= previous:
            raise ValueError("events must be one run in strictly increasing source sequence order")
        previous = event.sequence_id
        if event.kind == "learning.correlated-update":
            updates.append(event)
            continue
        if event.kind not in {
            DECISION_EVENT_KIND,
            "reward.correlated",
            "agent.decision",
            "agent.execution",
            "agent.execution_failed",
        }:
            continue
        try:
            evidence = (
                DecisionEvidence(event.payload).to_mapping()
                if event.kind == DECISION_EVENT_KIND
                else _reward(event)
                if event.kind == "reward.correlated"
                else _legacy(event)
            )
            identity = evidence["identity"]
            if (
                identity["run_id"] != run_id
                or identity["episode_id"] != _episode(event)
                or identity["step_id"] != event.step_id
            ):
                raise ValueError("decision event metadata differs from its captured identity")
            entries.append(
                {
                    "sequence_id": event.sequence_id,
                    "timestamp_ns": event.timestamp_ns,
                    "kind": event.kind,
                    "authority": "diagnostic",
                    "evidence": evidence,
                    "relation": "legacy_unverified"
                    if event.kind.startswith("agent.") and event.kind != DECISION_EVENT_KIND
                    else "captured",
                    "learning_updates": [],
                }
            )
        except (ValueError, TypeError, KeyError):
            warnings.append({"sequence_id": event.sequence_id, "code": "invalid_evidence"})
    unlinked: list[int] = []
    for event in updates:
        try:
            update = event.payload["update"]
            if not isinstance(update, Mapping):
                raise ValueError("update must be a mapping")
            key = (
                run_id,
                _episode(event),
                event.step_id,
                update["reward_receipt_sha256"],
                event.payload["action_id"],
                event.payload["before_sequence"],
                event.payload["after_sequence"],
            )
            matches = [entry for entry in entries if _binding(entry["evidence"]) == key]
            if not matches:
                unlinked.append(event.sequence_id)
                continue
            safe = {
                name: update[name]
                for name in (
                    "learner_id",
                    "table_id",
                    "policy_version",
                    "reward_receipt_sha256",
                    "state_sha256",
                    "action_sha256",
                    "next_state_sha256",
                )
            }
            for name in ("learner_id", "table_id"):
                _label(safe[name])
            _number(safe["policy_version"], integer=True)
            for name in (
                "reward_receipt_sha256",
                "state_sha256",
                "action_sha256",
                "next_state_sha256",
            ):
                _sha(safe[name])
            for entry in matches:
                evidence = entry["evidence"]
                if (
                    safe["state_sha256"] != evidence["observation"]["state_sha256"]
                    or safe["action_sha256"] != evidence["selection"]["action_sha256"]
                    or safe["next_state_sha256"] != evidence["reward"]["next_state_sha256"]
                ):
                    raise ValueError("learning update tensor digests differ from its receipt")
            for entry in matches:
                entry["learning_updates"].append(
                    {"sequence_id": event.sequence_id, "status": "reported_binding_checked", **safe}
                )
        except (ValueError, TypeError, KeyError):
            warnings.append({"sequence_id": event.sequence_id, "code": "invalid_update"})
    return {
        "schema_version": TIMELINE_SCHEMA,
        "run_id": run_id,
        "entries": entries,
        "unlinked_update_sequences": unlinked,
        "warnings": warnings,
        "source_window": {
            "first_sequence": events[0].sequence_id if events else None,
            "last_sequence": events[-1].sequence_id if events else None,
        },
    }


def read_timeline(
    path: str | Path, run_id: str, *, after_sequence: int = -1, limit: int = 250
) -> dict[str, Any]:
    """Read existing SQLite events without TrainingStore construction, migrations or writes.

    Cursor advances across every scanned row, including ignored private logs.
    Links outside the bounded page stay unlinked; the caller may project a
    larger explicitly bounded retained window without guessing neighboring data.
    """
    _label(run_id)
    if (
        isinstance(after_sequence, bool)
        or not isinstance(after_sequence, int)
        or after_sequence < -1
    ):
        raise ValueError("cursor must be an integer at least -1")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_TIMELINE_EVENTS
    ):
        raise ValueError("timeline page limit must be in [1,250]")
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("an existing regular run database is required")
    connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT run_id,sequence_id,timestamp_ns,kind,episode_id,step_id,"
            "substr(payload_json,1,?) AS payload_json FROM events WHERE run_id=? AND sequence_id>? "
            "ORDER BY sequence_id ASC LIMIT ?",
            (MAX_EVIDENCE_BYTES + 1, run_id, after_sequence, limit + 1),
        ).fetchall()
    finally:
        connection.close()
    events = []
    for row in rows[:limit]:
        try:
            encoded_payload = row["payload_json"]
            payload = (
                json.loads(encoded_payload)
                if type(encoded_payload) is str
                and len(encoded_payload.encode("utf-8")) <= MAX_EVIDENCE_BYTES
                else {}
            )
            if not isinstance(payload, Mapping):
                payload = {}
        except (ValueError, TypeError, RecursionError):
            payload = {}
        events.append(
            RunEvent(
                row["run_id"],
                row["sequence_id"],
                row["timestamp_ns"],
                row["kind"],
                row["episode_id"],
                row["step_id"],
                payload,
            )
        )
    result = project_events(events, run_id=run_id)
    result.update(
        cursor={"events_after": events[-1].sequence_id if events else after_sequence},
        more=len(rows) > limit,
    )
    return result
