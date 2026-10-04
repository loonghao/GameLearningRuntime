"""One-use durable admission to the existing SDK actor queue and owner callback."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from game_learning_runtime.collector import BoundedActorQueue, QueuedUnroll
from game_learning_runtime.contracts import Unroll
from game_learning_runtime.fleet_datahub import FleetHub
from game_learning_runtime.fleet_payload import (
    CompatibilitySpec,
    FleetError,
    SourceSpec,
    canonical,
    counter,
    digest,
    identifier,
    sha256,
)


@dataclass(frozen=True, slots=True)
class LearnerSelection:
    compatibility: CompatibilitySpec
    policy_sha256: str
    game_id: str
    runtime_source_commit: str
    adapter_source_sha256: str
    mode: str = "on_policy"
    allowed_source_ids: tuple[str, ...] = ()
    allow_simulated: bool = False
    algorithm: str | None = None
    algorithm_sha256: str | None = None
    allowed_behavior_policy_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.compatibility) is not CompatibilitySpec:
            raise FleetError("invalid_compatibility")
        digest(self.policy_sha256)
        digest(self.runtime_source_commit, 40)
        digest(self.adapter_source_sha256)
        identifier(self.game_id)
        if self.mode not in {"on_policy", "off_policy"} or not isinstance(
            self.allow_simulated, bool
        ):
            raise FleetError("invalid_selection")
        for field, validator in (
            ("allowed_source_ids", identifier),
            ("allowed_behavior_policy_sha256s", digest),
        ):
            entries = getattr(self, field)
            if (
                not isinstance(entries, tuple)
                or len(entries) > 64
                or len(set(entries)) != len(entries)
            ):
                raise FleetError("invalid_allowlist")
            for entry in entries:
                validator(entry)
        if self.mode == "off_policy":
            if (
                not self.allowed_source_ids
                or not self.allowed_behavior_policy_sha256s
                or self.algorithm is None
                or self.algorithm_sha256 is None
            ):
                raise FleetError("off_policy_requires_explicit_contract")
            identifier(self.algorithm)
            digest(self.algorithm_sha256)

    @property
    def binding_sha256(self) -> str:
        return sha256(
            canonical(
                {
                    "compatibility": CompatibilitySpec.to_record(self.compatibility),
                    "policy_sha256": self.policy_sha256,
                    "game_id": self.game_id,
                    "runtime_source_commit": self.runtime_source_commit,
                    "adapter_source_sha256": self.adapter_source_sha256,
                    "mode": self.mode,
                    "allowed_source_ids": self.allowed_source_ids,
                    "allow_simulated": self.allow_simulated,
                    "algorithm": self.algorithm,
                    "algorithm_sha256": self.algorithm_sha256,
                    "allowed_behavior_policy_sha256s": self.allowed_behavior_policy_sha256s,
                }
            )
        )


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    plan_id: str
    shard_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConsumptionTicket:
    ticket_id: str
    plan_id: str
    shard_id: str
    learner_id: str
    source_id: str
    source_epoch: str


@dataclass(frozen=True, slots=True)
class LearnerResult:
    declared_updates: int | None

    def __post_init__(self) -> None:
        if self.declared_updates is not None:
            counter(self.declared_updates)


@dataclass(frozen=True, slots=True)
class ConsumptionReceipt:
    receipt_id: str
    plan_id: str
    learner_id: str
    source_ids: tuple[str, ...]
    status: str
    transition_count: int
    callback_completed: bool
    learner_declared_updates: int | None
    finished_at_utc_ms: int | None


class FleetConsumer:
    """Finite synchronous owner boundary; no optimizer, service or automatic retry.

    The caller owns an exclusive queue and callable. A completed call is observed;
    update counts are declared. The hub cannot undo callback-side weight changes.
    """

    def __init__(
        self,
        hub: FleetHub,
        queue: BoundedActorQueue,
        *,
        learner_id: str,
        selection: LearnerSelection,
    ) -> None:
        identifier(learner_id)
        if type(selection) is not LearnerSelection:
            raise FleetError("invalid_selection")
        if not isinstance(queue, BoundedActorQueue):
            raise FleetError("invalid_queue")
        metrics = queue.metrics()
        if metrics.depth or metrics.in_flight_unrolls or metrics.overflow_policy != "fail":
            raise FleetError("queue_requires_exclusive_fail_policy")
        self._hub = hub
        self._queue = queue
        self._learner_id = learner_id
        self._selection = selection

    @property
    def hub(self) -> FleetHub:
        return self._hub

    @property
    def queue(self) -> BoundedActorQueue:
        return self._queue

    @property
    def learner_id(self) -> str:
        return self._learner_id

    @property
    def selection(self) -> LearnerSelection:
        return self._selection

    def _reason(self, source: SourceSpec, revoked: bool) -> str:
        selection = self.selection
        if revoked:
            return "revoked"
        if source.split == "evaluation_holdout":
            return "holdout"
        if source.split == "quarantine" or not source.simulated:
            return "quarantine"
        if (
            source.compatibility != selection.compatibility
            or source.game_id != selection.game_id
            or source.runtime_source_commit != selection.runtime_source_commit
            or source.adapter_source_sha256 != selection.adapter_source_sha256
        ):
            return "compatibility_mismatch"
        if source.simulated and not selection.allow_simulated:
            return "simulated_not_allowed"
        if (
            selection.mode == "on_policy"
            and source.behavior_policy_sha256 != selection.policy_sha256
        ):
            return "policy_mismatch"
        if selection.mode == "off_policy" and (
            source.source_id not in selection.allowed_source_ids
            or source.behavior_policy_sha256 not in selection.allowed_behavior_policy_sha256s
        ):
            return "off_policy_not_allowed"
        return "eligible"

    def plan(self, *, max_shards: int = 1) -> TrainingPlan:
        if counter(max_shards, minimum=1) != 1:
            raise FleetError("one_unroll_per_plan")
        selected: list[str] = []
        plan_id = f"plan-{uuid4().hex}"
        with self.hub._connection(write=True) as connection:
            blocked = (
                connection.execute(
                    "SELECT 1 FROM consumptions WHERE learner_id=? AND status IN ('claimed','ca"
                    "lling','unknown_effect') LIMIT 1",
                    (self.learner_id,),
                ).fetchone()
                is not None
            )
            for row in connection.execute(
                "SELECT * FROM sources ORDER BY source_id,source_epoch LIMIT 64"
            ):
                source = SourceSpec.from_record(json.loads(row["spec_json"]))
                reason = self._reason(source, bool(row["revoked"]))
                if reason == "eligible" and self.hub._holdout_blocked(connection, source):
                    reason = "quarantine"
                shards = []
                if reason == "eligible":
                    if blocked:
                        reason = "already_claimed"
                    else:
                        shards = connection.execute(
                            """
                            SELECT s.shard_id FROM shards s WHERE
                            s.source_id=? AND s.source_epoch=? AND
                            s.status='ready'
                            AND NOT EXISTS(SELECT 1 FROM shards p
                            WHERE p.source_id=s.source_id AND
                            p.source_epoch=s.source_epoch
                               AND p.shard_seq<s.shard_seq AND
                               p.status IN
                               ('uploading','ready','claimed','calling','unknown_effect'))
                            ORDER BY s.shard_seq LIMIT 64
""",
                            (source.source_id, source.source_epoch),
                        ).fetchall()
                        if not shards:
                            reason = "no_ready_data"
                connection.execute(
                    "UPDATE sources SET last_plan_eligible=?,last_plan_reasons=? WHERE source_i"
                    "d=? AND source_epoch=?",
                    (
                        int(reason == "eligible"),
                        json.dumps([reason]),
                        source.source_id,
                        source.source_epoch,
                    ),
                )
                for shard in shards:
                    if len(selected) < max_shards:
                        selected.append(shard["shard_id"])
            if selected:
                encoded_selection = self.selection.binding_sha256
                encoded_shards = json.dumps(selected)
                existing = connection.execute(
                    "SELECT plan_id FROM plans WHERE selection_sha256=? AND shards_json=? LIMIT 1",
                    (encoded_selection, encoded_shards),
                ).fetchone()
                if existing is not None:
                    plan_id = existing["plan_id"]
                else:
                    if (
                        connection.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
                        >= self.hub.limits.max_plans
                    ):
                        raise FleetError("plan_quota")
                    connection.execute(
                        "INSERT INTO plans VALUES(?,?,?)",
                        (plan_id, encoded_selection, encoded_shards),
                    )
        return TrainingPlan(plan_id, tuple(selected))

    def consume_one(
        self, plan: TrainingPlan, learner: Callable[[Unroll, ConsumptionTicket], LearnerResult]
    ) -> ConsumptionReceipt:
        if type(plan) is not TrainingPlan or not callable(learner):
            raise FleetError("invalid_consumer_request")
        if not plan.shard_ids:
            raise FleetError("no_ready_data")
        ticket_id = uuid4().hex
        receipt_id = f"receipt-{uuid4().hex}"
        selection_sha = self.selection.binding_sha256
        learner_id = self.learner_id
        with self.hub._connection(write=True) as connection:
            saved = connection.execute(
                "SELECT * FROM plans WHERE plan_id=?", (plan.plan_id,)
            ).fetchone()
            if (
                saved is None
                or saved["selection_sha256"] != self.selection.binding_sha256
                or tuple(json.loads(saved["shards_json"])) != plan.shard_ids
            ):
                raise FleetError("plan_binding")
            if connection.execute(
                "SELECT 1 FROM consumptions WHERE learner_id=? AND status IN ('claimed','callin"
                "g','unknown_effect') LIMIT 1",
                (self.learner_id,),
            ).fetchone():
                raise FleetError("learner_effect_unknown")
            row = connection.execute(
                "SELECT * FROM shards WHERE shard_id=?", (plan.shard_ids[0],)
            ).fetchone()
            if row is None or row["status"] != "ready":
                raise FleetError("already_claimed")
            source_row = self.hub._source(connection, row["source_id"], row["source_epoch"])
            source = SourceSpec.from_record(json.loads(source_row["spec_json"]))
            reason = self._reason(source, bool(source_row["revoked"]))
            if self.hub._holdout_blocked(connection, source) or self.hub._protected_shard(
                connection, row["shard_id"]
            ):
                reason = "quarantine"
            if reason != "eligible":
                raise FleetError(reason)
            ticket = ConsumptionTicket(
                ticket_id,
                plan.plan_id,
                row["shard_id"],
                self.learner_id,
                row["source_id"],
                row["source_epoch"],
            )
            connection.execute(
                "UPDATE shards SET status='claimed' WHERE shard_id=?", (ticket.shard_id,)
            )
            connection.execute(
                "INSERT INTO consumptions(receipt_id,ticket_id,plan_id,learner_id,shard_id,sour"
                "ce_id,source_epoch,status) VALUES(?,?,?,?,?,?,?,'claimed')",
                (
                    receipt_id,
                    ticket_id,
                    plan.plan_id,
                    self.learner_id,
                    ticket.shard_id,
                    ticket.source_id,
                    ticket.source_epoch,
                ),
            )
        leased: QueuedUnroll | None = None
        completed = False
        declared: int | None = None
        status = "rejected"
        try:
            decoded = self.hub._decoded(row)
            if self.queue.metrics().depth or self.queue.metrics().in_flight_unrolls:
                raise FleetError("queue_not_exclusive")
            self.queue.put_nowait(decoded.unroll)
            leased = self.queue.get_nowait()
            if leased.unroll is not decoded.unroll:
                raise FleetError("queue_binding")
            with self.hub._connection(write=True) as connection:
                latest = self.hub._source(connection, ticket.source_id, ticket.source_epoch)
                if (
                    self.selection.binding_sha256 != selection_sha
                    or self.learner_id != learner_id
                    or latest["spec_json"] != source_row["spec_json"]
                    or self._reason(source, bool(latest["revoked"])) != "eligible"
                    or self.hub._holdout_blocked(connection, source)
                    or self.hub._protected_shard(connection, ticket.shard_id)
                ):
                    raise FleetError("revoked")
                changed = connection.execute(
                    "UPDATE shards SET status='calling' WHERE shard_id=? AND status='claimed'",
                    (ticket.shard_id,),
                ).rowcount
                if changed != 1:
                    raise FleetError("ticket_not_current")
                connection.execute(
                    "UPDATE consumptions SET status='calling' WHERE ticket_id=? AND status='cla"
                    "imed'",
                    (ticket_id,),
                )
            # Persist the ambiguous boundary BEFORE invoking owner code.
            status = "unknown_effect"
            result = learner(decoded.unroll, ticket)
            completed = True
            if type(result) is LearnerResult:
                declared = result.declared_updates
            with self.hub._connection(write=True) as connection:
                latest = self.hub._source(connection, ticket.source_id, ticket.source_epoch)
                current = connection.execute(
                    "SELECT status FROM shards WHERE shard_id=?", (ticket.shard_id,)
                ).fetchone()
                candidate_status = "unknown_effect"
                if (
                    declared is not None
                    and self.selection.binding_sha256 == selection_sha
                    and self.learner_id == learner_id
                    and latest["spec_json"] == source_row["spec_json"]
                    and self._reason(source, bool(latest["revoked"])) == "eligible"
                    and current["status"] == "calling"
                    and not self.hub._holdout_blocked(connection, source)
                    and not self.hub._protected_shard(connection, ticket.shard_id)
                ):
                    self.queue.commit(leased)
                    leased = None
                    candidate_status = "consumed"
                now = self.hub._now()
                changed = connection.execute(
                    "UPDATE shards SET status=? WHERE shard_id=? AND status='calling'",
                    (candidate_status, ticket.shard_id),
                ).rowcount
                if changed != 1:
                    raise FleetError("ticket_not_current")
                connection.execute(
                    "UPDATE consumptions SET status=?,transition_count=?,callback_completed=?,d"
                    "eclared_updates=?,finished_ms=? WHERE ticket_id=? AND status='calling'",
                    (
                        candidate_status,
                        len(decoded.unroll.transitions) if candidate_status == "consumed" else 0,
                        int(completed),
                        declared,
                        now,
                        ticket_id,
                    ),
                )
            status = candidate_status
        except Exception:
            # Do not expose raw callback exceptions or retry a possibly mutated learner.
            with self.hub._connection(write=True) as connection:
                if status != "rejected":
                    status = "unknown_effect"
                connection.execute(
                    "UPDATE shards SET status=? WHERE shard_id=? AND status IN ('claimed','call"
                    "ing','consumed') AND EXISTS(SELECT 1 FROM consumptions WHERE shard_id=? AN"
                    "D ticket_id=?)",
                    (status, ticket.shard_id, ticket.shard_id, ticket_id),
                )
                connection.execute(
                    "UPDATE consumptions SET status=?,callback_completed=?,declared_updates=?,f"
                    "inished_ms=?,transition_count=0 WHERE ticket_id=? AND status IN ('claimed'"
                    ",'calling','consumed')",
                    (status, int(completed), declared, self.hub._now(), ticket_id),
                )
        finally:
            if leased is not None:
                self.queue.abort(leased)
        with self.hub._connection() as connection:
            saved_receipt = connection.execute(
                "SELECT * FROM consumptions WHERE ticket_id=?", (ticket_id,)
            ).fetchone()
        return ConsumptionReceipt(
            saved_receipt["receipt_id"],
            saved_receipt["plan_id"],
            saved_receipt["learner_id"],
            (saved_receipt["source_id"],),
            saved_receipt["status"],
            saved_receipt["transition_count"],
            bool(saved_receipt["callback_completed"]),
            saved_receipt["declared_updates"],
            saved_receipt["finished_ms"],
        )
