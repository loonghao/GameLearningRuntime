"""Exception hierarchy for Game Learning Runtime."""

from __future__ import annotations

from time import time_ns
from uuid import UUID

from game_learning_runtime.contracts import ActionOutcome, ActionReceipt, RefusalReasonClass


class GLRError(Exception):
    """Base class for all GLR errors."""


class ContractViolation(GLRError, ValueError):
    """Raised when an environment violates its declared contract."""


class CommandRefusal(ContractViolation):
    """Structured domain refusal that must pass through the common funnel.

    Adapter code may raise this for a domain outcome, but should not raise a
    generic ``ContractViolation`` for a command that was safely refused.
    Programming and wire-contract errors remain ordinary exceptions.
    """

    def __init__(
        self,
        *,
        target_id: str,
        reason_class: RefusalReasonClass | str,
        message: str,
        action_id: str | None = None,
        retryable: bool | None = None,
        episode_id: UUID | None = None,
        step_id: int | None = None,
        issued_timestamp_ns: int | None = None,
        observed_timestamp_ns: int | None = None,
    ) -> None:
        if (
            not target_id
            or len(target_id) > 128
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in target_id
            )
        ):
            raise ValueError("target_id must contain 1-128 non-control characters")
        if not message or len(message) > 512:
            raise ValueError("refusal message must contain 1-512 characters")
        try:
            resolved_reason = RefusalReasonClass(reason_class)
        except ValueError as error:
            raise ValueError(f"unsupported refusal reason class: {reason_class!r}") from error
        if action_id is not None and (not action_id or len(action_id) > 128):
            raise ValueError("action_id must contain 1-128 characters or None")
        if step_id is not None and (
            not isinstance(step_id, int) or isinstance(step_id, bool) or step_id <= 0
        ):
            raise ValueError("step_id must be a positive integer or None")
        if retryable is None:
            retryable = resolved_reason is RefusalReasonClass.TRANSIENT
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be bool or None")
        super().__init__(message)
        self.action_id = action_id
        self.target_id = target_id
        self.reason_class = resolved_reason
        self.message = message
        self.retryable = retryable
        self.episode_id = episode_id
        self.step_id = step_id
        self.issued_timestamp_ns = issued_timestamp_ns
        self.observed_timestamp_ns = observed_timestamp_ns

    def to_receipt(
        self,
        *,
        episode_id: UUID,
        step_id: int,
        action_id: str | None = None,
        issued_timestamp_ns: int | None = None,
        observed_timestamp_ns: int | None = None,
    ) -> ActionReceipt:
        """Normalize an exception refusal into the same typed receipt as a return refusal."""

        resolved_action_id = action_id or self.action_id
        if resolved_action_id is None:
            resolved_action_id = f"step-{step_id}"
        issued = (
            self.issued_timestamp_ns
            if issued_timestamp_ns is None and self.issued_timestamp_ns is not None
            else (time_ns() if issued_timestamp_ns is None else issued_timestamp_ns)
        )
        observed = time_ns() if observed_timestamp_ns is None else observed_timestamp_ns
        return ActionReceipt(
            action_id=resolved_action_id,
            episode_id=episode_id,
            step_id=step_id,
            outcome=ActionOutcome.REJECTED,
            issued_timestamp_ns=issued,
            observed_timestamp_ns=max(issued, observed),
            postcondition="refused",
            retryable=self.retryable,
            target_id=self.target_id,
            reason_class=self.reason_class,
        )


class OptionalDependencyError(GLRError, ImportError):
    """Raised when an optional integration is used without its dependencies."""


class HostProtocolError(GLRError, ValueError):
    """Raised when a Runtime Host violates framing or wire-schema rules."""


class HostRemoteError(GLRError):
    """A structured, non-retried error returned by a Runtime Host."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        refusal: CommandRefusal | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.refusal = refusal


# Readable alias for adapters that prefer a past-tense exception name.
CommandRefused = CommandRefusal
