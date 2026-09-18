"""Episode terminal-state contract: why an episode ended, and what may follow.

Every episode must record *why* it ended. A missing reason is a contract
violation, not a warning, because every downstream gate reads termination
state: an episode that does not say how it ended reads as green.

The runtime owns the seam. An adapter declares *facts* (death counts, stall
lengths) and *caps* (:class:`EpisodeCaps`); the runtime turns those into a
:class:`TerminationReason` at close time. An adapter may also declare the
reason itself through the ``termination_reason`` / ``termination_detail``
keys of ``TimeStep.info``; that declaration is validated against the closed
enum before it is accepted.

:class:`TerminationReason.ENV_INDETERMINATE` is absorbing. Once an action is
reported with :attr:`~game_learning_runtime.contracts.ActionOutcome.INDETERMINATE`
the episode ends immediately, the step that produced the receipt is never
recorded as learner data, and no further step is admitted unless the caller
explicitly re-attaches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from time import time_ns
from typing import Any
from uuid import UUID

from game_learning_runtime.contracts import ActionOutcome, ActionReceipt
from game_learning_runtime.errors import GLRError

TERMINATION_SCHEMA_VERSION = "glr.episode-termination.v1"

#: ``TimeStep.info`` key an adapter may use to declare why the episode ended.
TERMINATION_REASON_KEY = "termination_reason"
#: ``TimeStep.info`` key carrying the free-text companion to the reason.
TERMINATION_DETAIL_KEY = "termination_detail"
#: ``TimeStep.info`` keys carrying the facts the runtime attributes caps from.
DEATHS_KEY = "episode_deaths"
STALL_STEPS_KEY = "episode_stall_steps"

#: Who supplied the reason. Kept in the manifest so a reader can tell an
#: adapter declaration from a runtime attribution.
ATTRIBUTION_SOURCES = ("runtime", "adapter", "caller")

_MAX_DETAIL = 512


def _timestamp(value: int | None, *, path: str) -> int:
    if value is None:
        return time_ns()
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer or None")
    return value


def _sequence(value: object, *, path: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer or None")
    return value


def _detail(value: object, *, path: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string or None")
    if len(value) > _MAX_DETAIL:
        raise ValueError(f"{path} cannot exceed {_MAX_DETAIL} characters")
    return value


def _reason(value: object, *, path: str = TERMINATION_REASON_KEY) -> TerminationReason:
    if isinstance(value, TerminationReason):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a TerminationReason or its string value")
    try:
        return TerminationReason(value)
    except ValueError as error:
        supported = ", ".join(item.value for item in TerminationReason)
        raise ValueError(f"unsupported {path}: {value!r} (supported: {supported})") from error


class TerminationReason(str, Enum):
    """Closed set of reasons an episode may end.

    ``GOAL_REACHED`` is the only success value. ``ENV_INDETERMINATE`` means the
    environment consequence of an action is unknown and the correct response is
    a supervised restart rather than another action attempt.
    """

    GOAL_REACHED = "goal_reached"
    FAILED = "failed"
    STEP_BUDGET = "step_budget"
    TIME_BUDGET = "time_budget"
    DEATH_CAP = "death_cap"
    STALLED = "stalled"
    ENV_FROZEN = "env_frozen"
    HOST_UNAVAILABLE = "host_unavailable"
    ENV_INDETERMINATE = "env_indeterminate"
    CALLER_ABORTED = "caller_aborted"


class TerminationError(GLRError, ValueError):
    """Base contract violation in the episode termination lifecycle.

    Catchable and fielded: ``field`` names the contract field that was
    violated so a caller can report it without parsing a message.
    """

    def __init__(
        self,
        message: str,
        *,
        episode_id: UUID | None = None,
        field: str | None = None,
        step_id: int | None = None,
    ) -> None:
        if not message:
            raise ValueError("termination error message cannot be empty")
        super().__init__(message)
        self.message = message
        self.episode_id = episode_id
        self.field = field
        self.step_id = step_id

    def to_mapping(self) -> dict[str, Any]:
        return {
            "error": type(self).__name__,
            "message": self.message,
            "field": self.field,
            "episode_id": None if self.episode_id is None else str(self.episode_id),
            "step_id": self.step_id,
        }


class MissingTerminationReason(TerminationError):
    """Raised when an episode is closed without a termination reason."""

    def __init__(
        self,
        *,
        episode_id: UUID | None = None,
        step_id: int | None = None,
        detail: str | None = None,
    ) -> None:
        message = (
            f"episode closed without `{TERMINATION_REASON_KEY}`: declare the reason in "
            f"`TimeStep.info[{TERMINATION_REASON_KEY!r}]`, declare adapter caps on "
            "`EnvironmentSpec.episode_caps` so the runtime can attribute the close, or pass "
            "`reason` to `EpisodeTerminationGuard.close()`"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(
            message, episode_id=episode_id, field=TERMINATION_REASON_KEY, step_id=step_id
        )


class EpisodeClosedError(TerminationError):
    """Raised when a step is attempted on an episode that already ended."""

    def __init__(
        self,
        *,
        episode_id: UUID | None = None,
        step_id: int | None = None,
        reason: TerminationReason | None = None,
    ) -> None:
        ended = "unknown" if reason is None else reason.value
        message = (
            f"episode already ended with `{ended}`; no further step is admitted "
            "until the caller re-attaches"
        )
        super().__init__(
            message, episode_id=episode_id, field=TERMINATION_REASON_KEY, step_id=step_id
        )
        self.reason = reason


class IndeterminateOutcomeError(TerminationError):
    """Raised when an indeterminate outcome leaves an episode with no usable data."""

    def __init__(
        self,
        *,
        episode_id: UUID | None = None,
        step_id: int | None = None,
        detail: str | None = None,
    ) -> None:
        message = (
            "episode latched `env_indeterminate` before any step could be recorded; "
            "the correct response is a supervised restart, not another action"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(
            message, episode_id=episode_id, field=TERMINATION_REASON_KEY, step_id=step_id
        )


@dataclass(frozen=True, slots=True)
class EpisodeCaps:
    """Adapter-declared budgets the runtime may attribute a close to.

    Declaring a cap is how an adapter asks the runtime to own attribution: the
    adapter reports the underlying fact (death count, stall length) and the
    runtime decides the reason. An adapter that declares no caps can still set
    the reason explicitly, but a value is then mandatory.
    """

    max_steps: int | None = None
    max_time_ns: int | None = None
    death_cap: int | None = None
    stall_steps: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_time_ns", "death_cap", "stall_steps"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer or None")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_time_ns": self.max_time_ns,
            "death_cap": self.death_cap,
            "stall_steps": self.stall_steps,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EpisodeCaps:
        if not isinstance(value, Mapping):
            raise TypeError("episode caps must be a mapping")
        unknown = set(value) - set(cls().to_mapping())
        if unknown:
            raise ValueError(f"unknown episode caps field(s): {sorted(unknown)}")
        return cls(**{name: value[name] for name in cls().to_mapping() if name in value})


@dataclass(frozen=True, slots=True)
class EpisodeProgress:
    """Facts observed for one episode, used only to attribute a close."""

    steps: int = 0
    elapsed_ns: int = 0
    deaths: int = 0
    stall_steps: int = 0

    def __post_init__(self) -> None:
        for name in ("steps", "elapsed_ns", "deaths", "stall_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @classmethod
    def from_info(
        cls, info: Mapping[str, Any], *, steps: int = 0, elapsed_ns: int = 0
    ) -> EpisodeProgress:
        """Read the attribution facts an adapter reports through ``TimeStep.info``."""

        if not isinstance(info, Mapping):
            raise TypeError("info must be a mapping")
        deaths = _sequence(info.get(DEATHS_KEY), path=DEATHS_KEY)
        stall_steps = _sequence(info.get(STALL_STEPS_KEY), path=STALL_STEPS_KEY)
        return cls(
            steps=steps,
            elapsed_ns=elapsed_ns,
            deaths=0 if deaths is None else deaths,
            stall_steps=0 if stall_steps is None else stall_steps,
        )


def attribute_reason(
    caps: EpisodeCaps | None, progress: EpisodeProgress
) -> TerminationReason | None:
    """Return the reason a declared cap accounts for, or ``None``.

    Order is fixed so attribution is reproducible: a death cap outranks a step
    budget, which outranks a time budget, which outranks a stall.
    """

    if caps is None:
        return None
    if not isinstance(caps, EpisodeCaps):
        raise TypeError("caps must be an EpisodeCaps or None")
    if caps.death_cap is not None and progress.deaths >= caps.death_cap:
        return TerminationReason.DEATH_CAP
    if caps.max_steps is not None and progress.steps >= caps.max_steps:
        return TerminationReason.STEP_BUDGET
    if caps.max_time_ns is not None and progress.elapsed_ns >= caps.max_time_ns:
        return TerminationReason.TIME_BUDGET
    if caps.stall_steps is not None and progress.stall_steps >= caps.stall_steps:
        return TerminationReason.STALLED
    return None


def termination_from_info(info: Mapping[str, Any]) -> tuple[TerminationReason | None, str]:
    """Read and validate an adapter-declared termination reason.

    An unknown value is a contract violation rather than a silent fallback:
    a reason that is not in the closed enum cannot be gated on.
    """

    if not isinstance(info, Mapping):
        raise TypeError("info must be a mapping")
    raw = info.get(TERMINATION_REASON_KEY)
    if raw is None:
        return None, ""
    reason = _reason(raw)
    return reason, _detail(info.get(TERMINATION_DETAIL_KEY), path=TERMINATION_DETAIL_KEY)


@dataclass(frozen=True, slots=True)
class EpisodeTermination:
    """Why one episode ended, with the evidence needed to audit that decision."""

    episode_id: UUID
    reason: TerminationReason
    step_id: int = 0
    timestamp_ns: int = 0
    detail: str = ""
    last_known_sequence: int | None = None
    attributed_by: str = "runtime"
    latched_at_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        object.__setattr__(self, "reason", _reason(self.reason))
        if not isinstance(self.step_id, int) or isinstance(self.step_id, bool) or self.step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        if (
            not isinstance(self.timestamp_ns, int)
            or isinstance(self.timestamp_ns, bool)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer")
        object.__setattr__(self, "detail", _detail(self.detail, path="termination_detail"))
        object.__setattr__(
            self,
            "last_known_sequence",
            _sequence(self.last_known_sequence, path="last_known_sequence"),
        )
        if self.attributed_by not in ATTRIBUTION_SOURCES:
            raise ValueError(f"attributed_by must be one of {list(ATTRIBUTION_SOURCES)}")
        object.__setattr__(
            self, "latched_at_ns", _sequence(self.latched_at_ns, path="latched_at_ns")
        )

    @property
    def reached_goal(self) -> bool:
        """Whether this episode ended by reaching its goal."""

        return self.reason is TerminationReason.GOAL_REACHED

    @property
    def indeterminate(self) -> bool:
        """Whether this episode ended because an action outcome was indeterminate."""

        return self.reason is TerminationReason.ENV_INDETERMINATE

    @property
    def is_failure(self) -> bool:
        """Whether this episode ended for an environmental or host reason."""

        return self.reason in {
            TerminationReason.FAILED,
            TerminationReason.STALLED,
            TerminationReason.ENV_FROZEN,
            TerminationReason.HOST_UNAVAILABLE,
            TerminationReason.ENV_INDETERMINATE,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": TERMINATION_SCHEMA_VERSION,
            "episode_id": str(self.episode_id),
            "termination_reason": self.reason.value,
            "termination_detail": self.detail,
            "step_id": self.step_id,
            "timestamp_ns": self.timestamp_ns,
            "last_known_sequence": self.last_known_sequence,
            "attributed_by": self.attributed_by,
            "latched_at_ns": self.latched_at_ns,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EpisodeTermination:
        if not isinstance(value, Mapping):
            raise TypeError("episode termination must be a mapping")
        schema = value.get("schema_version")
        if schema != TERMINATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported episode termination schema: {schema!r}")
        expected = set(cls(UUID(int=0), TerminationReason.FAILED).to_mapping())
        unknown = set(value) - expected
        if unknown:
            raise ValueError(f"unknown episode termination field(s): {sorted(unknown)}")
        missing = sorted({"episode_id", "termination_reason"} - set(value))
        if missing:
            raise ValueError(f"episode termination is missing field(s): {missing}")
        return cls(
            episode_id=UUID(str(value["episode_id"])),
            reason=_reason(value["termination_reason"]),
            step_id=int(value.get("step_id", 0)),
            timestamp_ns=int(value.get("timestamp_ns", 0)),
            detail=_detail(value.get("termination_detail"), path="termination_detail"),
            last_known_sequence=_sequence(
                value.get("last_known_sequence"), path="last_known_sequence"
            ),
            attributed_by=str(value.get("attributed_by", "runtime")),
            latched_at_ns=_sequence(value.get("latched_at_ns"), path="latched_at_ns"),
        )


class EpisodeTerminationGuard:
    """Runtime-owned lifecycle seam for exactly one episode.

    The guard is the only place a termination is produced. It counts steps,
    remembers the last observation sequence whose consequence was known, latches
    an indeterminate outcome as absorbing, and refuses to close without a
    reason.
    """

    def __init__(
        self,
        episode_id: UUID,
        *,
        caps: EpisodeCaps | None = None,
        now_ns: int | None = None,
    ) -> None:
        if not isinstance(episode_id, UUID):
            raise TypeError("episode_id must be a UUID")
        if caps is not None and not isinstance(caps, EpisodeCaps):
            raise TypeError("caps must be an EpisodeCaps or None")
        self._episode_id = episode_id
        self._caps = caps
        self._started_ns = _timestamp(now_ns, path="now_ns")
        self._steps = 0
        self._last_step_id = 0
        self._last_known_sequence: int | None = None
        self._latched_ns: int | None = None
        self._termination: EpisodeTermination | None = None

    @property
    def episode_id(self) -> UUID:
        return self._episode_id

    @property
    def caps(self) -> EpisodeCaps | None:
        return self._caps

    @property
    def closed(self) -> bool:
        return self._termination is not None

    @property
    def termination(self) -> EpisodeTermination | None:
        return self._termination

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def last_known_sequence(self) -> int | None:
        return self._last_known_sequence

    @property
    def latched_at_ns(self) -> int | None:
        return self._latched_ns

    @property
    def indeterminate(self) -> bool:
        return self._latched_ns is not None

    def progress(self, *, now_ns: int | None = None) -> EpisodeProgress:
        """Return the facts observed so far, excluding adapter-reported ones."""

        now = _timestamp(now_ns, path="now_ns")
        return EpisodeProgress(steps=self._steps, elapsed_ns=max(0, now - self._started_ns))

    def require_open(self, *, step_id: int | None = None) -> None:
        """Raise if the episode already ended."""

        if self._termination is not None:
            raise EpisodeClosedError(
                episode_id=self._episode_id,
                step_id=step_id,
                reason=self._termination.reason,
            )

    def records_step(self, *, step_id: int | None = None) -> bool:
        """Whether a step taken now may enter the training dataset."""

        del step_id
        return self._termination is None

    def note_step(
        self,
        step_id: int,
        *,
        observation_sequence: int | None = None,
        now_ns: int | None = None,
    ) -> None:
        """Register one step attempt and advance the last known sequence."""

        del now_ns
        self.require_open(step_id=step_id)
        if not isinstance(step_id, int) or isinstance(step_id, bool) or step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        self._steps += 1
        self._last_step_id = step_id
        if observation_sequence is not None:
            self._last_known_sequence = _sequence(
                observation_sequence, path="observation_sequence"
            )

    def observe_outcome(
        self,
        receipt: ActionReceipt | ActionOutcome | None,
        *,
        step_id: int | None = None,
        observation_sequence: int | None = None,
        now_ns: int | None = None,
    ) -> bool:
        """Latch an indeterminate outcome and end the episode.

        Returns ``True`` only when this call originated the latch. A second
        report is idempotent: the latch is absorbing, so the first one wins and
        a retry is caught earlier by :meth:`require_open`.
        """

        if receipt is None:
            return False
        outcome = receipt.outcome if isinstance(receipt, ActionReceipt) else receipt
        if not isinstance(outcome, ActionOutcome):
            raise TypeError("outcome must be an ActionReceipt, an ActionOutcome, or None")
        if outcome is not ActionOutcome.INDETERMINATE:
            return False
        if self._latched_ns is not None:
            return False
        now = _timestamp(now_ns, path="now_ns")
        self._latched_ns = now
        if step_id is not None:
            observed_step = _sequence(step_id, path="step_id")
            if observed_step is not None:
                self._last_step_id = observed_step
        if observation_sequence is not None:
            self._last_known_sequence = _sequence(observation_sequence, path="observation_sequence")
        self._termination = EpisodeTermination(
            episode_id=self._episode_id,
            reason=TerminationReason.ENV_INDETERMINATE,
            step_id=self._last_step_id,
            timestamp_ns=now,
            detail=(
                "an action outcome was reported indeterminate; "
                "the environment consequence is unknown"
            ),
            last_known_sequence=self._last_known_sequence,
            attributed_by="runtime",
            latched_at_ns=now,
        )
        return True

    def close(
        self,
        *,
        reason: TerminationReason | str | None = None,
        detail: str | None = None,
        info: Mapping[str, Any] | None = None,
        progress: EpisodeProgress | None = None,
        step_id: int | None = None,
        now_ns: int | None = None,
    ) -> EpisodeTermination:
        """End the episode and return its terminal state.

        Resolution order: an absorbing indeterminate latch, then an explicit
        ``reason``, then an adapter declaration in ``info``, then runtime
        attribution from declared caps. If none supply a reason the close is a
        contract violation and :class:`MissingTerminationReason` is raised.
        Closing an episode that already ended returns its existing termination.
        """

        if self._termination is not None:
            return self._termination
        now = _timestamp(now_ns, path="now_ns")
        resolved_step = self._last_step_id
        if step_id is not None:
            observed_step = _sequence(step_id, path="step_id")
            if observed_step is not None:
                resolved_step = observed_step
        mapping_info: Mapping[str, Any] = {} if info is None else info
        declared, declared_detail = termination_from_info(mapping_info)
        if reason is not None:
            resolved = _reason(reason)
            source = "caller"
            resolved_detail = _detail(detail, path="termination_detail")
        elif declared is not None:
            resolved = declared
            source = "adapter"
            resolved_detail = _detail(detail, path="termination_detail") or declared_detail
        else:
            observed = self.progress(now_ns=now)
            if progress is not None:
                if not isinstance(progress, EpisodeProgress):
                    raise TypeError("progress must be an EpisodeProgress or None")
                observed = replace(
                    progress,
                    steps=max(progress.steps, observed.steps),
                    elapsed_ns=max(progress.elapsed_ns, observed.elapsed_ns),
                )
            else:
                observed = EpisodeProgress.from_info(
                    mapping_info, steps=observed.steps, elapsed_ns=observed.elapsed_ns
                )
            attributed = attribute_reason(self._caps, observed)
            if attributed is None:
                raise MissingTerminationReason(
                    episode_id=self._episode_id,
                    step_id=resolved_step,
                    detail="no declared caps matched the observed progress",
                )
            resolved = attributed
            source = "runtime"
            resolved_detail = _detail(detail, path="termination_detail")
        self._termination = EpisodeTermination(
            episode_id=self._episode_id,
            reason=resolved,
            step_id=resolved_step,
            timestamp_ns=now,
            detail=resolved_detail,
            last_known_sequence=self._last_known_sequence,
            attributed_by=source,
            latched_at_ns=self._latched_ns,
        )
        return self._termination

    def reached_goal(self) -> bool:
        """Whether this episode is closed and ended by reaching its goal."""

        return self._termination is not None and self._termination.reached_goal

    def to_mapping(self, *, now_ns: int | None = None) -> dict[str, Any]:
        """Lifecycle view for manifests, CLI JSON, and run stores."""

        return {
            "schema_version": TERMINATION_SCHEMA_VERSION,
            "episode_id": str(self._episode_id),
            "closed": self.closed,
            "steps": self._steps,
            "last_known_sequence": self._last_known_sequence,
            "latched_at_ns": self._latched_ns,
            "elapsed_ns": self.progress(now_ns=now_ns).elapsed_ns,
            "caps": None if self._caps is None else self._caps.to_mapping(),
            "termination": None if self._termination is None else self._termination.to_mapping(),
        }


__all__ = [
    "ATTRIBUTION_SOURCES",
    "DEATHS_KEY",
    "STALL_STEPS_KEY",
    "TERMINATION_DETAIL_KEY",
    "TERMINATION_REASON_KEY",
    "TERMINATION_SCHEMA_VERSION",
    "EpisodeCaps",
    "EpisodeClosedError",
    "EpisodeProgress",
    "EpisodeTermination",
    "EpisodeTerminationGuard",
    "IndeterminateOutcomeError",
    "MissingTerminationReason",
    "TerminationError",
    "TerminationReason",
    "attribute_reason",
    "termination_from_info",
]
