"""One typed refusal funnel for adapter command safeguards."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Protocol
from uuid import UUID

from game_learning_runtime.contracts import (
    ActionOutcome,
    ActionReceipt,
    RefusalReasonClass,
    TimeStep,
)
from game_learning_runtime.errors import CommandRefusal, ContractViolation

RefusalHandler = Callable[[ActionReceipt], None]


class RefusalPolicy(Protocol):
    """Adapter policy called once for each distinct command refusal."""

    def __call__(self, receipt: ActionReceipt) -> None:
        """Apply backoff, quarantine, or circuit-breaking policy."""
        ...


class RefusalFunnel:
    """Normalize return-value and exception refusals before adapter policy runs.

    A funnel is intentionally small and transport-neutral. It does not retry a
    command; it only validates the refusal identity and invokes the policy.
    Duplicate delivery of the same action identity is suppressed so a host
    and an outer contract wrapper can safely share one funnel.
    """

    def __init__(self, handler: RefusalHandler | None = None, *, max_recent: int = 1024) -> None:
        if max_recent < 1:
            raise ValueError("max_recent must be positive")
        self._handler = handler
        self._max_recent = max_recent
        self._seen: dict[tuple[UUID, int, str, str], None] = {}
        self._lock = RLock()

    def observe(self, receipt: ActionReceipt) -> ActionReceipt:
        """Route a typed refusal receipt through the policy exactly once."""

        if receipt.outcome not in {ActionOutcome.REJECTED, ActionOutcome.BLOCKED}:
            return receipt
        if receipt.target_id is None or receipt.reason_class is None:
            raise ContractViolation("refusal receipt requires target_id and reason_class")
        key = (receipt.episode_id, receipt.step_id, receipt.action_id, receipt.target_id)
        with self._lock:
            if key in self._seen:
                return receipt
            self._seen[key] = None
            if len(self._seen) > self._max_recent:
                self._seen.pop(next(iter(self._seen)))
            if self._handler is not None:
                self._handler(receipt)
        return receipt

    def observe_exception(
        self,
        refusal: CommandRefusal,
        *,
        episode_id: UUID,
        step_id: int,
        action_id: str | None = None,
        issued_timestamp_ns: int | None = None,
    ) -> ActionReceipt:
        """Convert an exception refusal and route it through the same policy."""

        return self.observe(
            refusal.to_receipt(
                episode_id=episode_id,
                step_id=step_id,
                action_id=action_id,
                issued_timestamp_ns=issued_timestamp_ns,
            )
        )

    def observe_timestep(self, timestep: TimeStep) -> ActionReceipt | None:
        """Route a refusal receipt carried by a returned time step."""

        receipt = timestep.action_receipt
        if receipt is None or not receipt.is_refusal:
            return None
        return self.observe(receipt)


__all__ = [
    "RefusalFunnel",
    "RefusalHandler",
    "RefusalPolicy",
    "RefusalReasonClass",
]
