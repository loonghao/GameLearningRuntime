"""Durable, bounded campaign admission and reviewed candidate promotion.

This SDK is a policy kernel for trusted project roles, not a process launcher,
agent sandbox, scheduler, or authentication service. Proposals are passive
data. Existing GLR tasks, adapters, supervisors and evaluators own execution.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from time import time_ns
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from game_learning_runtime.agent_goal import AgentGoal, GoalEvidence, ResearchMediaType
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.run_store import RunStatus, TrainingStore
from game_learning_runtime.training import KnowledgeAuthority

if TYPE_CHECKING:
    from game_learning_runtime.replay_evaluation import FixedReplaySuite, ReplayPolicy

CAMPAIGN_SCHEMA_VERSION = "glr.learning-campaign.v1"
PROPOSAL_SCHEMA_VERSION = "glr.learning-proposal.v1"
EVALUATION_ZERO_METRICS = (
    "evaluation.parameter_mutations",
    "evaluation.reset_identity_mismatches",
    "evaluation.stale_observation_updates",
    "evaluation.dead_or_loading_updates",
    "evaluation.illegal_action_bootstraps",
    "evaluation.reward_attribution_errors",
    "evaluation.action_interval_errors",
)
EVALUATION_CHECK_SOURCE = "evaluation.contract"
_ID = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_HASH = re.compile(r"[0-9a-f]{64}")


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("expected a bounded portable identifier")


def _digest(value: str) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("expected a lowercase SHA-256 digest")


def _integer(value: int, *, positive: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(
            "expected a positive integer" if positive else "expected a nonnegative integer"
        )


def _json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > 65536:
        raise ValueError("campaign data exceeds 64 KiB")
    return encoded


def _sha(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _declared_fields(contract: type[Any], value: object) -> dict[str, Any]:
    """Project the fixed public schema, excluding subclass extension fields."""
    return {item.name: getattr(value, item.name) for item in fields(contract)}


def _unique(values: tuple[str, ...], *, required: bool = True) -> None:
    if (
        not isinstance(values, tuple)
        or (required and not values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("identifiers must be a unique tuple")
    for value in values:
        _identifier(value)


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """Owner-reviewed immutable admission, evaluation and budget contract."""

    campaign_id: str
    goal: AgentGoal
    environment_id: str
    protocol_version: str
    target_id: str
    environment_config_sha256: str
    evaluator_sha256: str
    evaluation_suite_sha256: str
    resources: tuple[str, ...]
    admitted_actions: tuple[str, ...]
    reviewers: tuple[str, ...]
    baseline_checkpoint_sha256: str | None = None
    baseline_score: float | None = None
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        _identifier(self.campaign_id)
        _identifier(self.environment_id)
        _identifier(self.target_id)
        if not self.protocol_version or len(self.protocol_version) > 64:
            raise ValueError("protocol version must be bounded nonempty text")
        object.__setattr__(self, "goal", AgentGoal.from_mapping(AgentGoal.to_mapping(self.goal)))
        for digest in (
            self.environment_config_sha256,
            self.evaluator_sha256,
            self.evaluation_suite_sha256,
        ):
            _digest(digest)
        for values in (self.resources, self.admitted_actions, self.reviewers):
            _unique(values)
        if self.baseline_checkpoint_sha256 is not None:
            _digest(self.baseline_checkpoint_sha256)
        for number in (self.baseline_score, self.minimum_improvement):
            if number is not None and (isinstance(number, bool) or not math.isfinite(number)):
                raise ValueError("scores must be finite numbers")
        if self.minimum_improvement < 0:
            raise ValueError("minimum improvement cannot be negative")
        if (self.goal.promotion is None) != (self.baseline_score is None):
            raise ValueError("promotion metric and baseline score must be declared together")
        if (
            self.goal.promotion is not None
            and sum(
                item.metric == self.goal.promotion.metric for item in self.goal.success_criteria
            )
            != 1
        ):
            raise ValueError("promotion metric requires a fixed source in goal criteria")

    def to_mapping(self) -> dict[str, Any]:
        result = _declared_fields(CampaignSpec, self)
        result["goal"] = AgentGoal.to_mapping(self.goal)
        result["schema_version"] = CAMPAIGN_SCHEMA_VERSION
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CampaignSpec:
        fields = dict(value)
        if fields.pop("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError("unsupported campaign schema")
        fields["goal"] = AgentGoal.from_mapping(fields["goal"])
        for name in ("resources", "admitted_actions", "reviewers"):
            fields[name] = tuple(fields[name])
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A source revision, never an assertion of runtime authority."""

    source_id: str
    revision_sha256: str
    media_type: ResearchMediaType
    locator: str

    def __post_init__(self) -> None:
        _identifier(self.source_id)
        _digest(self.revision_sha256)
        object.__setattr__(self, "media_type", ResearchMediaType(self.media_type))
        if not isinstance(self.locator, str) or not self.locator.strip() or len(self.locator) > 512:
            raise ValueError("source locator must be bounded nonempty text")

    def to_mapping(self) -> dict[str, Any]:
        return _declared_fields(EvidenceRef, self)


@dataclass(frozen=True, slots=True)
class Proposal:
    """An inert, content-addressed candidate with its evidence lineage."""

    proposal_id: str
    proposer_id: str
    kind: str
    summary: str
    artifact_sha256: str
    base_checkpoint_sha256: str | None
    sources: tuple[EvidenceRef, ...]
    requested_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.proposal_id)
        _identifier(self.proposer_id)
        if self.kind not in {"knowledge", "policy", "interface"}:
            raise ValueError("unknown candidate kind")
        if (
            not isinstance(self.summary, str)
            or not self.summary.strip()
            or len(self.summary) > 2048
        ):
            raise ValueError("proposal summary must be bounded nonempty text")
        _digest(self.artifact_sha256)
        if self.base_checkpoint_sha256 is not None:
            _digest(self.base_checkpoint_sha256)
        if not isinstance(self.sources, tuple) or not self.sources or len(self.sources) > 256:
            raise ValueError("a proposal requires 1..256 source revisions")
        if any(not isinstance(source, EvidenceRef) for source in self.sources):
            raise TypeError("sources require EvidenceRef values")
        if len({(s.source_id, s.revision_sha256) for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate source revisions")
        _unique(self.requested_actions, required=False)

    def to_mapping(self) -> dict[str, Any]:
        result = _declared_fields(Proposal, self)
        result["sources"] = tuple(EvidenceRef.to_mapping(source) for source in self.sources)
        result["schema_version"] = PROPOSAL_SCHEMA_VERSION
        return result

    @property
    def sha256(self) -> str:
        return _sha(Proposal.to_mapping(self))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Proposal:
        fields = dict(value)
        if fields.pop("schema_version") != PROPOSAL_SCHEMA_VERSION:
            raise ValueError("unsupported proposal schema")
        fields["sources"] = tuple(EvidenceRef(**source) for source in fields["sources"])
        fields["requested_actions"] = tuple(fields["requested_actions"])
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class TrialTicket:
    trial_id: str
    campaign_id: str
    proposal_id: str
    worker_id: str
    token: str
    started_at_ns: int
    deadline_ns: int


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    reviewer_id: str
    evaluation_sha256: str
    approved: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.reviewer_id)
        _digest(self.evaluation_sha256)
        _digest(self.receipt_sha256)
        if not isinstance(self.approved, bool):
            raise ValueError("review approval must be a boolean")


@dataclass(frozen=True, slots=True)
class HostAuthority:
    """Trusted host configuration, never candidate input or OS authentication.

    The embedding host keeps the secret, capabilities, ledger and reviewer
    outside candidate access. Reopening a ledger requires the same configured
    authority; admitted role names alone cannot take it over.
    """

    authority_id: str
    evaluator_ids: tuple[str, ...]
    supervisor_ids: tuple[str, ...]
    reviewer_ids: tuple[str, ...]
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.authority_id)
        for roles in (self.evaluator_ids, self.supervisor_ids, self.reviewer_ids):
            _unique(roles)
        if not isinstance(self.secret, bytes) or not 32 <= len(self.secret) <= 128:
            raise ValueError("host authority requires 32..128 secret bytes")

    @property
    def fingerprint(self) -> str:
        return _sha(
            {
                "authority_id": self.authority_id,
                "evaluator_ids": self.evaluator_ids,
                "supervisor_ids": self.supervisor_ids,
                "reviewer_ids": self.reviewer_ids,
                "key_sha256": hashlib.sha256(self.secret).hexdigest(),
            }
        )

    def _principals(self, role: str) -> tuple[str, ...]:
        roles = {
            "evaluator": self.evaluator_ids,
            "supervisor": self.supervisor_ids,
            "reviewer": self.reviewer_ids,
        }
        if role not in roles:
            raise ValueError("unknown host role")
        return roles[role]


@dataclass(frozen=True, slots=True)
class HostRoleCapability:
    """Opaque authority scoped to one store epoch, trial and role; do not serialize."""

    store_epoch: str
    store_binding_sha256: str
    campaign_id: str
    trial_id: str
    trial_token: str = field(repr=False)
    role: str
    principal_id: str
    authority_fingerprint: str
    proof: bytes = field(repr=False)

    def _message(self) -> bytes:
        return _json(
            {
                "store_epoch": self.store_epoch,
                "store_binding_sha256": self.store_binding_sha256,
                "campaign_id": self.campaign_id,
                "trial_id": self.trial_id,
                "trial_token": self.trial_token,
                "role": self.role,
                "principal_id": self.principal_id,
                "authority_fingerprint": self.authority_fingerprint,
            }
        ).encode()


@dataclass(frozen=True, slots=True)
class SupervisorStopReceipt:
    """Host supervisor observation backed by a terminal worker run and event.

    This kernel validates persisted provenance, not OS liveness. The trusted
    supervisor must first observe its exact owned worker/process handles stop;
    a candidate cannot write that observation or obtain a supervisor capability.
    """

    receipt_id: str
    supervisor_id: str
    trial_id: str
    trial_token: str
    worker_run_id: str
    worker_ids: tuple[str, ...]
    environment_id: str
    protocol_version: str
    target_id: str
    environment_config_sha256: str
    process_identity_sha256: str
    source_sha256: str
    stopped_at_ns: int

    def __post_init__(self) -> None:
        for name in ("receipt_id", "supervisor_id", "worker_run_id", "environment_id", "target_id"):
            _identifier(getattr(self, name))
        _unique(self.worker_ids)
        for name in ("environment_config_sha256", "process_identity_sha256", "source_sha256"):
            _digest(getattr(self, name))
        _integer(self.stopped_at_ns, positive=False)
        if not self.protocol_version or len(self.protocol_version) > 64:
            raise ValueError("stop protocol version must be bounded nonempty text")
        if not self.trial_id or not self.trial_token:
            raise ValueError("stop receipt requires exact trial identity")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "glr.supervisor-stop.v1",
            **_declared_fields(SupervisorStopReceipt, self),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SupervisorStopReceipt:
        fields = dict(value)
        if fields.pop("schema_version") != "glr.supervisor-stop.v1":
            raise ValueError("unsupported supervisor stop schema")
        fields["worker_ids"] = tuple(fields["worker_ids"])
        return cls(**fields)


class CampaignStore:
    """Single-host durable policy state shared by all participating workers.

    The caller must keep this store and the external evaluator/reviewer out of
    an untrusted candidate's write scope. Resource locks only protect workers
    admitted through this same store; existing controllers need integration.
    """

    def __init__(self, path: str | Path, *, host_authority: HostAuthority | None = None) -> None:
        if host_authority is not None and not isinstance(host_authority, HostAuthority):
            raise TypeError("host_authority must be trusted HostAuthority configuration")
        self._host_authority = host_authority
        self._path_identity: tuple[int, int] | None = None
        self.path = Path(path).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._regular_parents(self.path)
        self._store_binding = hashlib.sha256(
            os.path.normcase(str(self.path.resolve())).encode()
        ).hexdigest()
        with self._connect() as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            allowed_tables = {
                "campaigns",
                "proposals",
                "trials",
                "leases",
                "evaluations",
                "audit",
                "campaign_host",
                "supervisor_stops",
            }
            if (
                version not in {0, 1}
                or (version == 0 and tables)
                or (version == 1 and (not tables <= allowed_tables or "campaigns" not in tables))
            ):
                raise ContractViolation(
                    "campaign store must be a new or compatible dedicated database"
                )
            statements = (
                "CREATE TABLE IF NOT EXISTS campaigns("
                "id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS proposals("
                "campaign TEXT NOT NULL REFERENCES campaigns(id), id TEXT NOT NULL, "
                "body TEXT NOT NULL, PRIMARY KEY(campaign,id))",
                "CREATE TABLE IF NOT EXISTS trials("
                "id TEXT PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), "
                "proposal TEXT NOT NULL, ticket TEXT NOT NULL, status TEXT NOT NULL, "
                "evaluation TEXT, evaluation_hash TEXT, stopped_receipt TEXT)",
                "CREATE TABLE IF NOT EXISTS leases("
                "resource TEXT PRIMARY KEY, trial TEXT NOT NULL REFERENCES trials(id))",
                "CREATE TABLE IF NOT EXISTS evaluations("
                "run_id TEXT PRIMARY KEY, trial TEXT NOT NULL REFERENCES trials(id))",
                "CREATE TABLE IF NOT EXISTS audit("
                "sequence INTEGER PRIMARY KEY, campaign TEXT NOT NULL REFERENCES campaigns(id), "
                "kind TEXT NOT NULL, body TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS campaign_host("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1), fingerprint TEXT, "
                "epoch TEXT NOT NULL, "
                "store_binding_sha256 TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS supervisor_stops("
                "trial TEXT PRIMARY KEY REFERENCES trials(id), body TEXT NOT NULL, "
                "store_path TEXT NOT NULL, ledger_sha256 TEXT NOT NULL)",
                "PRAGMA user_version=1",
            )
            for statement in statements:
                connection.execute(statement)
            identity = connection.execute("SELECT * FROM campaign_host").fetchone()
            if identity is None:
                connection.execute(
                    "INSERT INTO campaign_host VALUES(1,NULL,?,?)",
                    (uuid4().hex, self._store_binding),
                )
                identity = connection.execute("SELECT * FROM campaign_host").fetchone()
            assert identity is not None
            if identity["store_binding_sha256"] != self._store_binding:
                raise ContractViolation("campaign ledger was copied to a foreign store path")
            self._host_epoch = identity["epoch"]
            if host_authority is not None:
                if identity["fingerprint"] is None:
                    if any(
                        connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
                        for table in (
                            "campaigns",
                            "proposals",
                            "trials",
                            "leases",
                            "evaluations",
                            "audit",
                            "supervisor_stops",
                        )
                    ):
                        raise ContractViolation(
                            "legacy host ownership is unknown; cannot adopt live claims"
                        )
                    connection.execute(
                        "UPDATE campaign_host SET fingerprint=?", (host_authority.fingerprint,)
                    )
                elif identity["fingerprint"] != host_authority.fingerprint:
                    raise ContractViolation("host authority differs from the ledger owner")
        details = self.path.stat()
        self._path_identity = (details.st_dev, details.st_ino)

    def role_capability(
        self, authority: HostAuthority, ticket: TrialTicket | str, *, role: str, principal_id: str
    ) -> HostRoleCapability:
        """Host-only issuance; never expose the authority or this seam to workers."""
        if (
            self._host_authority is None
            or authority.fingerprint != self._host_authority.fingerprint
        ):
            raise ContractViolation("host authority is required to issue a role capability")
        if principal_id not in authority._principals(role):
            raise ContractViolation("principal is not admitted to this host role")
        campaign_id = ticket.campaign_id if isinstance(ticket, TrialTicket) else ticket
        with self._connect() as connection:
            if isinstance(ticket, TrialTicket):
                self._trial(connection, ticket)
            else:
                self._campaign(connection, campaign_id)
                if role != "supervisor":
                    raise ContractViolation("campaign scope only authorizes a stop request")
        fields = {
            "store_epoch": self._host_epoch,
            "store_binding_sha256": self._store_binding,
            "campaign_id": campaign_id,
            "trial_id": ticket.trial_id if isinstance(ticket, TrialTicket) else "",
            "trial_token": ticket.token if isinstance(ticket, TrialTicket) else "",
            "role": role,
            "principal_id": principal_id,
            "authority_fingerprint": authority.fingerprint,
        }
        unsigned = HostRoleCapability(**fields, proof=b"")
        return HostRoleCapability(
            **fields, proof=hmac.digest(authority.secret, unsigned._message(), "sha256")
        )

    def _require_host(
        self, capability: HostRoleCapability | None, ticket: TrialTicket | str, role: str
    ) -> str:
        authority = self._host_authority
        if authority is None or not isinstance(capability, HostRoleCapability):
            raise ContractViolation("privileged operation requires a trusted host role capability")
        with self._connect() as connection:
            identity = connection.execute("SELECT * FROM campaign_host").fetchone()
            if (
                identity is None
                or identity["epoch"] != self._host_epoch
                or identity["fingerprint"] != authority.fingerprint
                or identity["store_binding_sha256"] != self._store_binding
            ):
                raise ContractViolation("persisted host authority changed")
        if (
            capability.store_epoch != self._host_epoch
            or capability.store_binding_sha256 != self._store_binding
            or capability.campaign_id
            != (ticket.campaign_id if isinstance(ticket, TrialTicket) else ticket)
            or capability.trial_id != (ticket.trial_id if isinstance(ticket, TrialTicket) else "")
            or capability.trial_token != (ticket.token if isinstance(ticket, TrialTicket) else "")
            or capability.role != role
            or capability.principal_id not in authority._principals(role)
            or capability.authority_fingerprint != authority.fingerprint
            or not hmac.compare_digest(
                capability.proof, hmac.digest(authority.secret, capability._message(), "sha256")
            )
        ):
            raise ContractViolation("host role capability has foreign scope or invalid authority")
        return capability.principal_id

    @staticmethod
    def _regular_parents(path: Path) -> None:
        for part in (path, *path.parents):
            if part.exists() or part.is_symlink():
                details = part.lstat()
                if (
                    stat.S_ISLNK(details.st_mode)
                    or getattr(details, "st_file_attributes", 0) & 0x400
                ):
                    raise ContractViolation("campaign paths cannot contain links or reparse points")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._regular_parents(self.path)
        if self._path_identity is not None:
            details = self.path.stat()
            if (details.st_dev, details.st_ino) != self._path_identity:
                raise ContractViolation("campaign ledger identity changed")
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            connection.execute("BEGIN IMMEDIATE")
            if hasattr(self, "_host_epoch"):
                identity = connection.execute("SELECT * FROM campaign_host").fetchone()
                expected = self._host_authority.fingerprint if self._host_authority else None
                if (
                    identity is None
                    or identity["epoch"] != self._host_epoch
                    or identity["store_binding_sha256"] != self._store_binding
                    or (expected is not None and identity["fingerprint"] != expected)
                ):
                    raise ContractViolation("persisted host authority changed during operation")
            yield connection
            if self._path_identity is not None:
                details = self.path.stat()
                if (details.st_dev, details.st_ino) != self._path_identity:
                    raise ContractViolation("campaign ledger identity changed during operation")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _campaign(
        connection: sqlite3.Connection, campaign_id: str
    ) -> tuple[CampaignSpec, dict[str, Any]]:
        row = connection.execute(
            "SELECT spec,state FROM campaigns WHERE id=?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise KeyError(campaign_id)
        return CampaignSpec.from_mapping(json.loads(row["spec"])), json.loads(row["state"])

    @staticmethod
    def _proposal(connection: sqlite3.Connection, campaign_id: str, proposal_id: str) -> Proposal:
        row = connection.execute(
            "SELECT body FROM proposals WHERE campaign=? AND id=?", (campaign_id, proposal_id)
        ).fetchone()
        if row is None:
            raise KeyError(proposal_id)
        return Proposal.from_mapping(json.loads(row["body"]))

    @staticmethod
    def _audit(connection: sqlite3.Connection, campaign_id: str, kind: str, body: object) -> None:
        connection.execute(
            "INSERT INTO audit(campaign,kind,body) VALUES(?,?,?)", (campaign_id, kind, _json(body))
        )

    @staticmethod
    def _state(connection: sqlite3.Connection, campaign_id: str, state: dict[str, Any]) -> None:
        connection.execute("UPDATE campaigns SET state=? WHERE id=?", (_json(state), campaign_id))

    def create(self, spec: CampaignSpec, *, now_ns: int | None = None) -> None:
        now = time_ns() if now_ns is None else now_ns
        _integer(now, positive=False)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO campaigns VALUES(?,?,?)",
                (
                    spec.campaign_id,
                    _json(CampaignSpec.to_mapping(spec)),
                    _json(
                        {
                            "status": "running",
                            "created_at_ns": now,
                            "trials": 0,
                            "reserved_steps": 0,
                            "reserved_sources": 0,
                            "checkpoint_sha256": spec.baseline_checkpoint_sha256,
                            "score": spec.baseline_score,
                            "accepted_proposals": {},
                            "goal_satisfied": False,
                        }
                    ),
                ),
            )
            self._audit(
                connection,
                spec.campaign_id,
                "campaign.created",
                {"spec_sha256": _sha(CampaignSpec.to_mapping(spec))},
            )

    def snapshot(self, campaign_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            spec, state = self._campaign(connection, campaign_id)
            trials = [
                dict(row)
                for row in connection.execute(
                    "SELECT id,status,evaluation_hash,stopped_receipt FROM trials "
                    "WHERE campaign=? ORDER BY rowid",
                    (campaign_id,),
                )
            ]
            return {"spec": CampaignSpec.to_mapping(spec), "state": state, "trials": trials}

    def propose(self, campaign_id: str, proposal: Proposal) -> None:
        with self._connect() as connection:
            spec, state = self._campaign(connection, campaign_id)
            if state["status"] != "running":
                raise ContractViolation("campaign is not accepting proposals")
            if proposal.base_checkpoint_sha256 != state["checkpoint_sha256"]:
                raise ContractViolation("proposal must bind the current incumbent checkpoint")
            if not set(proposal.requested_actions) <= set(spec.admitted_actions):
                raise ContractViolation("proposal cannot expand adapter action authority")
            if any(s.media_type not in spec.goal.allowed_research_media for s in proposal.sources):
                raise ContractViolation("proposal contains a disallowed research source")
            connection.execute(
                "INSERT INTO proposals VALUES(?,?,?)",
                (campaign_id, proposal.proposal_id, _json(Proposal.to_mapping(proposal))),
            )
            self._audit(connection, campaign_id, "proposal.recorded", Proposal.to_mapping(proposal))

    def claim(
        self,
        campaign_id: str,
        proposal_id: str,
        *,
        worker_id: str,
        training_steps: int,
        wall_seconds: int,
        now_ns: int | None = None,
    ) -> TrialTicket:
        _identifier(worker_id)
        _integer(training_steps, positive=False)
        _integer(wall_seconds)
        now = time_ns() if now_ns is None else now_ns
        _integer(now, positive=False)
        with self._connect() as connection:
            spec, state = self._campaign(connection, campaign_id)
            proposal = self._proposal(connection, campaign_id, proposal_id)
            if (
                state["status"] != "running"
                or proposal.base_checkpoint_sha256 != state["checkpoint_sha256"]
            ):
                raise ContractViolation("campaign stopped or proposal has a stale checkpoint")
            if connection.execute(
                "SELECT 1 FROM trials WHERE campaign=? "
                "AND status IN ('claimed','quarantined','awaiting-review')",
                (campaign_id,),
            ).fetchone():
                raise ContractViolation("campaign has an unresolved trial")
            if connection.execute(
                "SELECT 1 FROM trials WHERE campaign=? AND proposal=?", (campaign_id, proposal_id)
            ).fetchone():
                raise ContractViolation("each candidate revision gets only one trial")
            budget = spec.goal.budget
            if now < state["created_at_ns"]:
                raise ContractViolation("campaign clock regressed")
            deadline = state["created_at_ns"] + budget.max_wall_seconds * 10**9
            if (
                now >= deadline
                or now + wall_seconds * 10**9 > deadline
                or state["trials"] >= budget.max_trials
                or state["reserved_steps"] + training_steps > budget.max_training_steps
                or state["reserved_sources"] + len(proposal.sources) > budget.max_research_sources
            ):
                raise ContractViolation("campaign budget exhausted")
            if any(
                connection.execute("SELECT 1 FROM leases WHERE resource=?", (resource,)).fetchone()
                for resource in spec.resources
            ):
                raise ContractViolation(
                    "a campaign resource is held; expired leases are never stolen"
                )
            ticket = TrialTicket(
                f"trial-{uuid4().hex}",
                campaign_id,
                proposal_id,
                worker_id,
                uuid4().hex,
                now,
                now + wall_seconds * 10**9,
            )
            connection.execute(
                "INSERT INTO trials(id,campaign,proposal,ticket,status) VALUES(?,?,?,?,'claimed')",
                (
                    ticket.trial_id,
                    campaign_id,
                    proposal_id,
                    _json(_declared_fields(TrialTicket, ticket)),
                ),
            )
            connection.executemany(
                "INSERT INTO leases VALUES(?,?)",
                ((resource, ticket.trial_id) for resource in spec.resources),
            )
            state["trials"] += 1
            state["reserved_steps"] += training_steps
            state["reserved_sources"] += len(proposal.sources)
            self._state(connection, campaign_id, state)
            self._audit(
                connection,
                campaign_id,
                "trial.claimed",
                {
                    **_declared_fields(TrialTicket, ticket),
                    "reserved_steps": training_steps,
                    "reserved_sources": len(proposal.sources),
                },
            )
            return ticket

    @staticmethod
    def _trial(connection: sqlite3.Connection, ticket: TrialTicket) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM trials WHERE id=?", (ticket.trial_id,)).fetchone()
        if row is None or json.loads(row["ticket"]) != _declared_fields(TrialTicket, ticket):
            raise ContractViolation("unknown or stale trial ticket")
        return cast(sqlite3.Row, row)

    def candidate_directory(self, ticket: TrialTicket) -> Path:
        with self._connect() as connection:
            self._trial(connection, ticket)
        return self.path.parent / "candidates" / ticket.trial_id

    def _artifact(self, ticket: TrialTicket, relative: str) -> tuple[str, int]:
        if not isinstance(relative, str) or "\\" in relative or ":" in relative:
            raise ContractViolation("candidate path must be portable and relative")
        parts = relative.split("/")
        if (
            not parts
            or any(part in {"", ".", ".."} for part in parts)
            or PurePosixPath(relative).is_absolute()
        ):
            raise ContractViolation("candidate path escapes its isolated trial directory")
        root = self.path.parent / "candidates" / ticket.trial_id
        path = root.joinpath(*parts)
        self._regular_parents(path)
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise ContractViolation("candidate must be a regular file of at most 64 MiB")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), path.stat().st_size

    @staticmethod
    def _ledger_snapshot(path: Path, run_id: str, *, worker_stop: bool = False) -> dict[str, Any]:
        CampaignStore._regular_parents(path)
        details = path.stat()
        identity = {
            "store_identity": [details.st_dev, details.st_ino],
            "store_path": str(path.resolve()),
        }
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as ledger:
            ledger.row_factory = sqlite3.Row
            ledger.execute("BEGIN")
            run = ledger.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                raise ContractViolation("persisted evidence run is missing")
            if worker_stop:
                events = ledger.execute(
                    "SELECT * FROM events WHERE run_id=? AND kind='supervisor.worker-stopped' "
                    "ORDER BY sequence_id LIMIT 65",
                    (run_id,),
                ).fetchall()
                if len(events) > 64:
                    raise ContractViolation("supervisor evidence exceeds its bounded scope")
                return {**identity, "run": dict(run), "events": [dict(row) for row in events]}
            rows = ledger.execute(
                "SELECT * FROM metrics WHERE run_id=? ORDER BY metric_id LIMIT 2049", (run_id,)
            ).fetchall()
            if len(rows) > 2048:
                raise ContractViolation("evaluation evidence exceeds its bounded scope")
            events = ledger.execute(
                "SELECT * FROM events WHERE run_id=? AND kind='evaluation.replay' "
                "ORDER BY sequence_id LIMIT 2",
                (run_id,),
            ).fetchall()
            return {
                **identity,
                "run": dict(run),
                "metrics": [dict(row) for row in rows],
                "replay_events": [dict(row) for row in events],
            }

    @staticmethod
    def _metric_matches(
        snapshot: Mapping[str, Any],
        item: GoalEvidence,
        config_sha256: str,
        started_at_ns: int,
        finished_at_ns: int,
        *,
        policy_candidate: bool = False,
    ) -> bool:
        # Read the authoritative ledger without changing its schema or latest-value
        # semantics. Contradictory same-source results cannot be cherry-picked.
        rows = [
            row
            for row in snapshot["metrics"]
            if row["run_id"] == item.run_id
            and row["name"] == item.metric
            and json.loads(row["metadata_json"]).get("source") == item.source
            and json.loads(row["metadata_json"]).get("authority") == item.authority.value
        ]
        fixed_counter = (
            item.metric in EVALUATION_ZERO_METRICS and item.source == EVALUATION_CHECK_SOURCE
        )
        return bool(rows) and all(
            (
                row["value"] == item.value == 0
                if fixed_counter
                else math.isclose(row["value"], item.value, rel_tol=1e-9, abs_tol=1e-12)
            )
            and row.get("environment_config_digest") == config_sha256
            and started_at_ns <= row["timestamp_ns"] <= finished_at_ns
            and (
                not policy_candidate
                or (
                    json.loads(row["metadata_json"]).get("coverage") in ("audited", "measured")
                    and json.loads(row["metadata_json"]).get("not_applicable_reason") is None
                    and json.loads(row["metadata_json"]).get("evaluation_scope")
                    != "offline-replay-inert-reference"
                )
            )
            for row in rows
        )

    def evaluate(
        self,
        ticket: TrialTicket,
        *,
        store: TrainingStore,
        run_id: str,
        evidence: Sequence[GoalEvidence],
        candidate_path: str,
        now_ns: int | None = None,
        host: HostRoleCapability | None = None,
    ) -> str:
        """Bind fixed evaluation to authoritative ledger evidence and exact bytes.

        A validation error leaves the claim locked. The caller must quarantine
        it and obtain a trusted supervisor stop receipt before freeing resources.
        """
        evaluator_id = self._require_host(host, ticket, "evaluator")
        now = time_ns() if now_ns is None else now_ns
        _integer(now, positive=False)
        if not evidence or len(evidence) > 256:
            raise ContractViolation("evaluation requires 1..256 bounded evidence items")
        with self._connect() as connection:
            row = self._trial(connection, ticket)
            spec, state = self._campaign(connection, ticket.campaign_id)
            proposal = self._proposal(connection, ticket.campaign_id, ticket.proposal_id)
            if evaluator_id in {ticket.worker_id, proposal.proposer_id}:
                raise ContractViolation(
                    "fixed external evaluator must be independent of candidate workers"
                )
            if (
                row["status"] != "claimed"
                or state["status"] != "running"
                or not ticket.started_at_ns <= now <= ticket.deadline_ns
            ):
                raise ContractViolation("evaluation requires an active, unexpired campaign claim")
            artifact_hash, size = self._artifact(ticket, candidate_path)
            if artifact_hash != proposal.artifact_sha256:
                raise ContractViolation("candidate bytes differ from proposed revision")
            run = store.get_run(run_id)
            snapshot = self._ledger_snapshot(store.path, run_id)
            persisted = snapshot["run"]
            if (
                any(
                    persisted.get(key) != getattr(run, key)
                    for key in (
                        "run_id",
                        "environment_id",
                        "protocol_version",
                        "kind",
                        "started_at_ns",
                        "finished_at_ns",
                        "exit_code",
                        "environment_config_digest",
                    )
                )
                or persisted["status"] != run.status.value
                or json.loads(persisted["metadata_json"]) != dict(run.metadata)
            ):
                raise ContractViolation(
                    "evaluation run changed while taking its fixed ledger snapshot"
                )
            expected = {
                "campaign_trial_id": ticket.trial_id,
                "campaign_token": ticket.token,
                "candidate_sha256": artifact_hash,
                "evaluator_sha256": spec.evaluator_sha256,
                "evaluation_suite_sha256": spec.evaluation_suite_sha256,
                "target_id": spec.target_id,
                "evaluator_id": evaluator_id,
                "proposal_sha256": proposal.sha256,
            }
            if (
                run.kind != "evaluation"
                or run.status is not RunStatus.SUCCEEDED
                or run.exit_code != 0
                or run.environment_id != spec.environment_id
                or run.protocol_version != spec.protocol_version
                or run.environment_config_digest != spec.environment_config_sha256
                or run.started_at_ns < ticket.started_at_ns
                or run.finished_at_ns is None
                or not run.started_at_ns <= run.finished_at_ns <= min(now, ticket.deadline_ns)
                or any(run.metadata.get(key) != value for key, value in expected.items())
            ):
                raise ContractViolation(
                    "evaluation run is not bound to this candidate and fixed evaluation contract"
                )
            if (
                proposal.kind == "policy"
                and run.metadata.get("evaluation_scope") == "offline-replay-inert-reference"
            ):
                raise ContractViolation("inert offline replay cannot promote a policy checkpoint")
            if run.metadata.get("evaluation_scope") not in {
                "fixed-external",
                "offline-replay-inert-reference",
            }:
                raise ContractViolation("evaluation producer scope is missing or unknown")
            for item in evidence:
                if (
                    item.run_id != run_id
                    or item.authority is not KnowledgeAuthority.AUTHORITATIVE
                    or not self._metric_matches(
                        snapshot,
                        item,
                        spec.environment_config_sha256,
                        run.started_at_ns,
                        run.finished_at_ns,
                        policy_candidate=proposal.kind == "policy",
                    )
                ):
                    if (
                        item.metric in EVALUATION_ZERO_METRICS
                        and item.source == EVALUATION_CHECK_SOURCE
                    ):
                        raise ContractViolation(
                            "frozen correctness gate requires exact zero "
                            "authoritative persisted metrics"
                        )
                    raise ContractViolation(
                        "evaluation requires matching authoritative persisted metrics"
                    )
            by_key = {(item.metric, item.source): item for item in evidence}
            if len(by_key) != len(evidence):
                raise ContractViolation("duplicate evaluation evidence")
            required = {(item.metric, item.source) for item in spec.goal.success_criteria}
            required.update((metric, EVALUATION_CHECK_SOURCE) for metric in EVALUATION_ZERO_METRICS)
            if not required <= set(by_key):
                raise ContractViolation("evaluation is missing objective or correctness evidence")
            if any(
                by_key[(metric, EVALUATION_CHECK_SOURCE)].value != 0
                for metric in EVALUATION_ZERO_METRICS
            ):
                raise ContractViolation(
                    "frozen evaluation failed a lifecycle or learning correctness gate"
                )
            result = spec.goal.evaluate(evidence)
            score = None
            if spec.goal.promotion is not None:
                criterion = next(
                    c for c in spec.goal.success_criteria if c.metric == spec.goal.promotion.metric
                )
                matches = [
                    e
                    for e in evidence
                    if (e.metric, e.source) == (criterion.metric, criterion.source)
                ]
                if len(matches) != 1:
                    raise ContractViolation("fixed promotion metric evidence is missing")
                score = matches[0].value
                delta = score - state["score"]
                if spec.goal.promotion.mode.value == "min":
                    delta = -delta
                eligible = delta > spec.minimum_improvement
            else:
                eligible = result.satisfied
            evaluation = {
                "run_id": run_id,
                "ticket": _declared_fields(TrialTicket, ticket),
                "candidate_path": candidate_path,
                "candidate_sha256": artifact_hash,
                "candidate_size": size,
                "score": score,
                "goal_satisfied": result.satisfied,
                "eligible": eligible,
                "evaluator_sha256": spec.evaluator_sha256,
                "evaluation_suite_sha256": spec.evaluation_suite_sha256,
                "environment_id": spec.environment_id,
                "protocol_version": spec.protocol_version,
                "target_id": spec.target_id,
                "environment_config_sha256": spec.environment_config_sha256,
                "evaluator_id": evaluator_id,
                "proposal_sha256": proposal.sha256,
                "store_path": str(store.path),
                "candidate_kind": proposal.kind,
                "evaluation_scope": run.metadata["evaluation_scope"],
                "evidence": [GoalEvidence.to_mapping(e) for e in evidence],
            }
            evaluation["ledger_sha256"] = _sha(snapshot)
            measurements = []
            for item in evidence:
                matching = [
                    metric
                    for metric in snapshot["metrics"]
                    if metric["name"] == item.metric
                    and json.loads(metric["metadata_json"]).get("source") == item.source
                    and json.loads(metric["metadata_json"]).get("authority") == item.authority.value
                ]
                if not matching:
                    raise ContractViolation("final persisted measurement is missing")
                measurements.append(
                    {
                        "name": item.metric,
                        "source": item.source,
                        "metric_id": matching[-1]["metric_id"],
                        "value": item.value,
                    }
                )
            primary = spec.goal.success_criteria[0]
            if spec.goal.promotion is not None:
                primary = next(
                    c for c in spec.goal.success_criteria if c.metric == spec.goal.promotion.metric
                )
            evaluation["measurements"] = measurements
            evaluation["final_measurement"] = next(
                m
                for m in measurements
                if (m["name"], m["source"]) == (primary.metric, primary.source)
            )
            digest = _sha(evaluation)
            connection.execute("INSERT INTO evaluations VALUES(?,?)", (run_id, ticket.trial_id))
            connection.execute(
                "UPDATE trials SET status=?,evaluation=?,evaluation_hash=? WHERE id=?",
                (
                    "awaiting-review" if eligible else "rejected",
                    _json(evaluation),
                    digest,
                    ticket.trial_id,
                ),
            )
            self._audit(
                connection,
                ticket.campaign_id,
                "trial.evaluated",
                {"trial_id": ticket.trial_id, "evaluation_sha256": digest, **evaluation},
            )
            return digest

    def evaluate_replay(
        self,
        ticket: TrialTicket,
        *,
        store: TrainingStore,
        suite: FixedReplaySuite,
        policy: ReplayPolicy,
        candidate_path: str,
        now_ns: int | None = None,
        host: HostRoleCapability | None = None,
    ) -> str:
        """Execute the fixed offline evaluator and persist its derived verdict.

        This seam admits only inert knowledge/interface references, never a
        policy checkpoint or live-game improvement. The policy is a trusted,
        caller-supplied replay consumer; candidate bytes are never imported or
        executed. The embedding application still owns execution isolation.
        Unknown checks remain unknown and fail admission. Every failure leaves
        the resource claim intact until a supervisor reconciles the worker.
        """
        from game_learning_runtime.replay_evaluation import (
            FixedReplaySuite,
            ReplayEvaluationResult,
            _fixed_suite_sha256,
            evaluate_replay,
            evaluator_sha256,
        )

        evaluator_id = self._require_host(host, ticket, "evaluator")
        started = time_ns() if now_ns is None else now_ns
        _integer(started, positive=False)
        with self._connect() as connection:
            row = self._trial(connection, ticket)
            spec, state = self._campaign(connection, ticket.campaign_id)
            proposal = self._proposal(connection, ticket.campaign_id, ticket.proposal_id)
            if evaluator_id in {ticket.worker_id, proposal.proposer_id}:
                raise ContractViolation(
                    "fixed external evaluator must be independent of candidate workers"
                )
            if (
                row["status"] != "claimed"
                or state["status"] != "running"
                or not ticket.started_at_ns <= started <= ticket.deadline_ns
            ):
                raise ContractViolation("replay requires an active campaign claim")
            if proposal.kind not in {"knowledge", "interface"} or spec.goal.promotion is not None:
                raise ContractViolation(
                    "offline replay admits inert references, not policy promotion"
                )
            criteria = spec.goal.success_criteria
            if (
                len(criteria) != 1
                or criteria[0].metric != "replay.contract_passed"
                or criteria[0].source != "evaluation.replay"
                or criteria[0].target != 1
                or criteria[0].operator.value not in {"eq", "gte"}
            ):
                raise ContractViolation("replay requires its fixed contract-pass objective")
            if (
                suite.spec.environment_id != spec.environment_id
                or suite.spec.protocol_version != spec.protocol_version
                or suite.spec.metadata.get("target_id") != spec.target_id
                or suite.spec.metadata.get("environment_config_sha256")
                != spec.environment_config_sha256
            ):
                raise ContractViolation("replay suite has a foreign environment or configuration")
            revisions = {(episode.source_id, episode.source_sha256) for episode in suite.episodes}
            if any((s.source_id, s.revision_sha256) not in revisions for s in proposal.sources):
                raise ContractViolation("replay does not cover the proposal's evidence revisions")
            semantics = {
                action.semantic
                for episode in suite.episodes
                for frame in episode.frames
                for action in frame.legal_actions
            }
            semantics.update(
                action.semantic for episode in suite.episodes for action in episode.recorded_actions
            )
            if not semantics <= set(spec.admitted_actions):
                raise ContractViolation("replay surface exceeds the campaign's admitted actions")
            FixedReplaySuite.verify_integrity(suite)
            if _fixed_suite_sha256(suite) != spec.evaluation_suite_sha256:
                raise ContractViolation("fixed replay suite changed")
            if evaluator_sha256(policy) != spec.evaluator_sha256:
                raise ContractViolation("fixed replay evaluator implementation changed")
            artifact_hash, _ = self._artifact(ticket, candidate_path)
            if artifact_hash != proposal.artifact_sha256:
                raise ContractViolation("candidate bytes differ from proposed revision")
        candidate = (self.path.parent / "candidates" / ticket.trial_id).joinpath(
            *candidate_path.split("/")
        )
        run = store.create_run(
            environment_id=spec.environment_id,
            protocol_version=spec.protocol_version,
            kind="evaluation",
            environment_config_digest=spec.environment_config_sha256,
            started_at_ns=started,
            metadata={
                "campaign_trial_id": ticket.trial_id,
                "campaign_token": ticket.token,
                "candidate_sha256": artifact_hash,
                "evaluator_sha256": spec.evaluator_sha256,
                "evaluation_suite_sha256": spec.evaluation_suite_sha256,
                "evaluation_scope": "offline-replay-inert-reference",
                "target_id": spec.target_id,
                "evaluator_id": evaluator_id,
                "proposal_sha256": proposal.sha256,
            },
        )

        def guard_step() -> None:
            current_time = time_ns() if now_ns is None else now_ns
            with self._connect() as connection:
                current_trial = self._trial(connection, ticket)
                _, current_state = self._campaign(connection, ticket.campaign_id)
                if (
                    current_trial["status"] != "claimed"
                    or current_state["status"] != "running"
                    or not ticket.started_at_ns <= current_time <= ticket.deadline_ns
                ):
                    raise ContractViolation("replay ownership or wall budget ended")

        try:
            result = evaluate_replay(
                suite,
                candidate,
                policy,
                expected_suite_sha256=spec.evaluation_suite_sha256,
                expected_candidate_sha256=proposal.artifact_sha256,
                step_guard=guard_step,
            )
            if result.evaluator_sha256 != spec.evaluator_sha256:
                raise ContractViolation("fixed replay evaluator implementation changed")
            finished = time_ns() if now_ns is None else now_ns
            store.append_event(
                run.run_id,
                kind="evaluation.replay",
                payload=ReplayEvaluationResult.to_mapping(result),
                timestamp_ns=finished,
            )
            values = {"replay.contract_passed": float(result.passed)}
            values.update({k: float(v) for k, v in result.check_counts.items() if v is not None})
            # Explicit suite exclusions are scoped to inert replay admission.
            # The persisted replay event retains null counts and every reason;
            # this compatibility value is never a measured live-game zero.
            values.update({metric: 0.0 for metric in result.not_applicable})
            evidence = []
            for metric, value in values.items():
                source = (
                    "evaluation.replay"
                    if metric == "replay.contract_passed"
                    else EVALUATION_CHECK_SOURCE
                )
                store.record_metric(
                    run.run_id,
                    name=metric,
                    value=value,
                    metadata={
                        "source": source,
                        "authority": "authoritative",
                        "coverage": (
                            result.coverage[metric].value
                            if metric in result.coverage
                            else "derived-objective"
                        ),
                        "not_applicable_reason": result.not_applicable.get(metric),
                        "evaluation_scope": "offline-replay-inert-reference",
                    },
                    timestamp_ns=finished,
                    environment_config_digest=spec.environment_config_sha256,
                )
                evidence.append(
                    GoalEvidence(
                        metric, value, source, KnowledgeAuthority.AUTHORITATIVE, run.run_id
                    )
                )
        except BaseException:
            store.finish_run(
                run.run_id,
                status=RunStatus.FAILED,
                exit_code=1,
                finished_at_ns=time_ns() if now_ns is None else now_ns,
            )
            raise
        store.finish_run(
            run.run_id,
            status=RunStatus.SUCCEEDED if result.passed else RunStatus.FAILED,
            exit_code=0 if result.passed else 1,
            finished_at_ns=finished,
        )
        if not result.passed:
            raise ContractViolation(
                f"fixed replay failed or lacks evidence; retained run {run.run_id}"
            )
        return self.evaluate(
            ticket,
            store=store,
            run_id=run.run_id,
            evidence=evidence,
            candidate_path=candidate_path,
            now_ns=finished,
            host=host,
        )

    def quarantine(self, ticket: TrialTicket, *, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
            raise ValueError("quarantine requires a bounded reason")
        with self._connect() as connection:
            row = self._trial(connection, ticket)
            if row["status"] not in {"claimed", "awaiting-review", "quarantined"}:
                raise ContractViolation("terminal trial cannot be quarantined")
            # Reconciled workers cannot become an unresolvable owner again.
            status = "rejected" if row["stopped_receipt"] is not None else "quarantined"
            connection.execute("UPDATE trials SET status=? WHERE id=?", (status, ticket.trial_id))
            self._audit(
                connection,
                ticket.campaign_id,
                "trial.quarantined",
                {"trial_id": ticket.trial_id, "reason": reason, "status": status},
            )

    @staticmethod
    def _validate_worker_stop(
        ticket: TrialTicket, spec: CampaignSpec, receipt: SupervisorStopReceipt, store_path: Path
    ) -> dict[str, Any]:
        snapshot = CampaignStore._ledger_snapshot(
            store_path, receipt.worker_run_id, worker_stop=True
        )
        run = snapshot["run"]
        metadata = json.loads(run["metadata_json"])
        expected = {
            "campaign_trial_id": ticket.trial_id,
            "campaign_token": ticket.token,
            "target_id": spec.target_id,
            "worker_ids": list(receipt.worker_ids),
            "process_identity_sha256": receipt.process_identity_sha256,
        }
        if (
            receipt.trial_id != ticket.trial_id
            or receipt.trial_token != ticket.token
            or receipt.worker_ids != (ticket.worker_id,)
            or receipt.environment_id != spec.environment_id
            or receipt.protocol_version != spec.protocol_version
            or receipt.target_id != spec.target_id
            or receipt.environment_config_sha256 != spec.environment_config_sha256
            or run["kind"] != "campaign-worker"
            or run["status"] not in {"succeeded", "failed", "interrupted"}
            or run["exit_code"] is None
            or run["finished_at_ns"] is None
            or run["environment_id"] != spec.environment_id
            or run["protocol_version"] != spec.protocol_version
            or run.get("environment_config_digest") != spec.environment_config_sha256
            or not ticket.started_at_ns
            <= run["started_at_ns"]
            <= run["finished_at_ns"]
            <= receipt.stopped_at_ns
            or any(metadata.get(key) != value for key, value in expected.items())
        ):
            raise ContractViolation("supervisor stop requires this exact terminal bound worker run")
        events = snapshot["events"]
        if (
            len(events) != 1
            or json.loads(events[0]["payload_json"])
            != json.loads(_json(SupervisorStopReceipt.to_mapping(receipt)))
            or events[0]["timestamp_ns"] != receipt.stopped_at_ns
        ):
            raise ContractViolation(
                "supervisor stop requires its persisted exact observation receipt"
            )
        return snapshot

    def confirm_stopped(
        self,
        ticket: TrialTicket,
        *,
        supervisor_receipt_sha256: str | None = None,
        stop_receipt: SupervisorStopReceipt | None = None,
        store: TrainingStore | None = None,
        host: HostRoleCapability | None = None,
    ) -> None:
        """Accept host-owned terminal worker provenance; unknown stop retains locks.

        A failed/interrupted terminal worker may be safely returned. This is
        independent of the successful fixed evaluation required for approval.
        The host supervisor owns and verifies actual process handles first.
        """
        supervisor_id = self._require_host(host, ticket, "supervisor")
        if not isinstance(stop_receipt, SupervisorStopReceipt) or store is None:
            raise ContractViolation(
                "host stop requires typed persisted supervisor evidence, not a digest"
            )
        if stop_receipt.supervisor_id != supervisor_id:
            raise ContractViolation("supervisor receipt and host principal differ")
        with self._connect() as connection:
            row = self._trial(connection, ticket)
            spec, _ = self._campaign(connection, ticket.campaign_id)
            proposal = self._proposal(connection, ticket.campaign_id, ticket.proposal_id)
            evaluation = json.loads(row["evaluation"]) if row["evaluation"] else {}
            if supervisor_id in {
                ticket.worker_id,
                proposal.proposer_id,
                evaluation.get("evaluator_id"),
            }:
                raise ContractViolation(
                    "supervisor must be independent of worker, proposer and evaluator"
                )
            if row["stopped_receipt"] is not None:
                raise ContractViolation("worker stop already confirmed")
            if row["status"] not in {"claimed", "quarantined", "rejected", "awaiting-review"}:
                raise ContractViolation("trial is terminal")
            snapshot = self._validate_worker_stop(ticket, spec, stop_receipt, store.path)
            details = store.path.stat()
            body = {
                "receipt": SupervisorStopReceipt.to_mapping(stop_receipt),
                "store_identity": [details.st_dev, details.st_ino],
            }
            receipt_sha256 = _sha(body)
            if (
                supervisor_receipt_sha256 is not None
                and supervisor_receipt_sha256 != receipt_sha256
            ):
                raise ContractViolation("supervisor digest differs from persisted typed evidence")
            connection.execute(
                "INSERT INTO supervisor_stops VALUES(?,?,?,?)",
                (ticket.trial_id, _json(body), str(store.path), _sha(snapshot)),
            )
            status = "rejected" if row["status"] in {"claimed", "quarantined"} else row["status"]
            connection.execute(
                "UPDATE trials SET status=?,stopped_receipt=? WHERE id=?",
                (status, receipt_sha256, ticket.trial_id),
            )
            connection.execute("DELETE FROM leases WHERE trial=?", (ticket.trial_id,))
            self._audit(
                connection,
                ticket.campaign_id,
                "trial.stopped",
                {
                    "trial_id": ticket.trial_id,
                    "supervisor_receipt_sha256": receipt_sha256,
                    "supervisor_id": supervisor_id,
                },
            )

    def review(
        self,
        ticket: TrialTicket,
        decision: ReviewDecision,
        *,
        store: TrainingStore | None = None,
        host: HostRoleCapability | None = None,
        now_ns: int | None = None,
    ) -> None:
        """Record owner review and atomically move a reference, never install code."""
        reviewer_id = self._require_host(host, ticket, "reviewer")
        now = time_ns() if now_ns is None else now_ns
        _integer(now, positive=False)
        with self._connect() as connection:
            row = self._trial(connection, ticket)
            spec, state = self._campaign(connection, ticket.campaign_id)
            proposal = self._proposal(connection, ticket.campaign_id, ticket.proposal_id)
            if (
                row["status"] != "awaiting-review"
                or row["stopped_receipt"] is None
                or (decision.approved and state["status"] != "running")
            ):
                raise ContractViolation("review requires completed evaluation and a stopped worker")
            if (
                decision.reviewer_id not in spec.reviewers
                or decision.reviewer_id != reviewer_id
                or decision.reviewer_id in {proposal.proposer_id, ticket.worker_id}
                or decision.evaluation_sha256 != row["evaluation_hash"]
            ):
                raise ContractViolation(
                    "review requires an admitted independent reviewer bound to this evaluation"
                )
            evaluation = json.loads(row["evaluation"])
            stop = connection.execute(
                "SELECT * FROM supervisor_stops WHERE trial=?", (ticket.trial_id,)
            ).fetchone()
            if stop is None or _sha(json.loads(stop["body"])) != row["stopped_receipt"]:
                raise ContractViolation("persisted host stop evidence is missing or changed")
            stop_body = json.loads(stop["body"])
            receipt = SupervisorStopReceipt.from_mapping(stop_body["receipt"])
            if (
                reviewer_id in {evaluation["evaluator_id"], receipt.supervisor_id}
                or _sha(evaluation) != row["evaluation_hash"]
                or evaluation["proposal_sha256"] != proposal.sha256
            ):
                raise ContractViolation(
                    "review requires an independent principal and unchanged evaluation"
                )
            if proposal.base_checkpoint_sha256 != state["checkpoint_sha256"]:
                raise ContractViolation("incumbent changed since candidate was proposed")
            if decision.approved:
                if (
                    store is None
                    or str(store.path) != evaluation["store_path"]
                    or _sha(self._ledger_snapshot(store.path, evaluation["run_id"]))
                    != evaluation["ledger_sha256"]
                    or not ticket.started_at_ns
                    <= receipt.stopped_at_ns
                    <= now
                    <= ticket.deadline_ns
                    or not evaluation["eligible"]
                ):
                    raise ContractViolation(
                        "approval requires unchanged successful fixed evaluation and current budget"
                    )
                worker_path = Path(stop["store_path"])
                details = worker_path.stat()
                if [details.st_dev, details.st_ino] != stop_body["store_identity"] or _sha(
                    self._validate_worker_stop(ticket, spec, receipt, worker_path)
                ) != stop["ledger_sha256"]:
                    raise ContractViolation("supervisor worker evidence changed before approval")
                digest, size = self._artifact(ticket, evaluation["candidate_path"])
                if (digest, size) != (proposal.artifact_sha256, evaluation["candidate_size"]):
                    raise ContractViolation("candidate changed after evaluation")
                state["accepted_proposals"][proposal.kind] = proposal.proposal_id
                if proposal.kind == "policy":
                    state["checkpoint_sha256"] = digest
                    if evaluation["score"] is not None:
                        state["score"] = evaluation["score"]
                state["goal_satisfied"] = evaluation["goal_satisfied"]
                if evaluation["goal_satisfied"]:
                    state["status"] = "succeeded"
                self._state(connection, ticket.campaign_id, state)
            connection.execute(
                "UPDATE trials SET status=? WHERE id=?",
                ("promoted" if decision.approved else "rejected", ticket.trial_id),
            )
            authorization = None
            if decision.approved:
                authorization = {
                    "schema_version": "glr.reference-promotion-authorization.v1",
                    "authorization_id": "authorization-" + uuid4().hex,
                    "candidate_kind": proposal.kind,
                    "evaluation_scope": evaluation["evaluation_scope"],
                    "artifact_sha256": proposal.artifact_sha256,
                    "environment_id": spec.environment_id,
                    "protocol_version": spec.protocol_version,
                    "target_id": spec.target_id,
                    "environment_config_sha256": spec.environment_config_sha256,
                    "run_id": evaluation["run_id"],
                    "trial_id": ticket.trial_id,
                    "trial_token": ticket.token,
                    "evaluation_sha256": row["evaluation_hash"],
                    "evaluator_sha256": spec.evaluator_sha256,
                    "evaluation_suite_sha256": spec.evaluation_suite_sha256,
                    "final_measurement": evaluation["final_measurement"],
                    "reviewer_id": reviewer_id,
                    "proposer_id": proposal.proposer_id,
                    "worker_ids": list(receipt.worker_ids),
                    "evaluator_id": evaluation["evaluator_id"],
                    "supervisor_id": receipt.supervisor_id,
                    "stop_receipt_sha256": row["stopped_receipt"],
                    "store_binding_sha256": self._store_binding,
                    "goal_id": spec.goal.goal_id,
                }
            self._audit(
                connection,
                ticket.campaign_id,
                "trial.reviewed",
                {
                    "trial_id": ticket.trial_id,
                    **_declared_fields(ReviewDecision, decision),
                    "authorization": authorization,
                },
            )

    def stop(
        self, campaign_id: str, *, reason: str, host: HostRoleCapability | None = None
    ) -> None:
        supervisor_id = self._require_host(host, campaign_id, "supervisor")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
            raise ValueError("stop requires a bounded reason")
        with self._connect() as connection:
            _, state = self._campaign(connection, campaign_id)
            state["status"] = "stopped"
            self._state(connection, campaign_id, state)
            self._audit(
                connection,
                campaign_id,
                "campaign.stopped",
                {"reason": reason, "supervisor_id": supervisor_id},
            )

    def events(self, campaign_id: str) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            self._campaign(connection, campaign_id)
            return tuple(
                {"sequence": row["sequence"], "kind": row["kind"], "body": json.loads(row["body"])}
                for row in connection.execute(
                    "SELECT * FROM audit WHERE campaign=? ORDER BY sequence", (campaign_id,)
                )
            )
