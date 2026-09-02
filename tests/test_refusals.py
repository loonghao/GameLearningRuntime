from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from game_learning_runtime import (
    ActionOutcome,
    ActionReceipt,
    CommandRefusal,
    ContractEnvironment,
    RefusalFunnel,
    RefusalReasonClass,
    TimeStep,
)
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.examples import CounterEnvironment


def test_exception_and_return_refusals_share_one_policy_funnel() -> None:
    receipts: list[ActionReceipt] = []
    funnel = RefusalFunnel(receipts.append)

    class RaisingEnvironment(CounterEnvironment):
        def step(self, action: Mapping[str, Any]):
            del action
            raise CommandRefusal(
                action_id="move-1",
                target_id="card-1",
                reason_class=RefusalReasonClass.STRUCTURAL,
                message="postcondition failed",
            )

    raising = ContractEnvironment(RaisingEnvironment(target=2), refusal_funnel=funnel)
    initial = raising.reset()
    with pytest.raises(CommandRefusal):
        raising.step({"choice": np.array([1], dtype=np.int64)})

    class ReturningEnvironment(CounterEnvironment):
        def step(self, action: Mapping[str, Any]):
            result = super().step(action)
            return _with_receipt(result, action_id="move-2")

    returning = ContractEnvironment(ReturningEnvironment(target=2), refusal_funnel=funnel)
    returned_initial = returning.reset()
    returned = returning.step({"choice": np.array([1], dtype=np.int64)})

    assert returned_initial.episode_id != initial.episode_id
    assert returned.action_receipt is not None
    assert returned.action_receipt.outcome is ActionOutcome.REJECTED
    assert returned.action_receipt.reason_class is RefusalReasonClass.STRUCTURAL
    assert [receipt.action_id for receipt in receipts] == ["move-1", "move-2"]
    assert receipts[0].target_id == receipts[1].target_id == "card-1"
    assert receipts[0].reason_class is receipts[1].reason_class

    funnel.observe(receipts[0])
    assert len(receipts) == 2


def test_refusal_funnel_validates_identity_and_suppresses_non_refusals() -> None:
    with pytest.raises(ValueError, match="max_recent"):
        RefusalFunnel(max_recent=0)

    receipts: list[ActionReceipt] = []
    funnel = RefusalFunnel(receipts.append, max_recent=1)
    accepted = ActionReceipt(
        action_id="accepted",
        episode_id=CounterEnvironment(target=1).reset().episode_id,
        step_id=1,
        outcome=ActionOutcome.ACCEPTED,
        issued_timestamp_ns=1,
        observed_timestamp_ns=2,
    )
    assert funnel.observe(accepted) is accepted
    assert (
        funnel.observe_timestep(
            TimeStep(
                observation={"state": np.array([0], dtype=np.int64)},
                reward=np.array([0.0], dtype=np.float32),
                terminated=np.array([False], dtype=np.bool_),
                truncated=np.array([False], dtype=np.bool_),
                episode_id=accepted.episode_id,
                step_id=1,
            )
        )
        is None
    )

    incomplete = ActionReceipt(
        action_id="refused",
        episode_id=accepted.episode_id,
        step_id=1,
        outcome=ActionOutcome.REJECTED,
        issued_timestamp_ns=1,
        observed_timestamp_ns=2,
    )
    with pytest.raises(ContractViolation, match="target_id"):
        funnel.observe(incomplete)

    first_refusal = incomplete.__class__(
        action_id="first",
        episode_id=accepted.episode_id,
        step_id=1,
        outcome=ActionOutcome.REJECTED,
        issued_timestamp_ns=1,
        observed_timestamp_ns=2,
        target_id="card-1",
        reason_class=RefusalReasonClass.STRUCTURAL,
    )
    second_refusal = first_refusal.__class__(
        action_id="second",
        episode_id=accepted.episode_id,
        step_id=2,
        outcome=ActionOutcome.BLOCKED,
        issued_timestamp_ns=2,
        observed_timestamp_ns=3,
        target_id="card-1",
        reason_class=RefusalReasonClass.TRANSIENT,
    )
    funnel.observe(first_refusal)
    funnel.observe(second_refusal)
    assert [receipt.action_id for receipt in receipts] == ["first", "second"]


def test_command_refusal_validation_and_receipt_conversion() -> None:
    with pytest.raises(ValueError, match="target_id"):
        CommandRefusal(target_id="bad target", reason_class="structural", message="no")
    with pytest.raises(ValueError, match="message"):
        CommandRefusal(target_id="card", reason_class="structural", message="")
    with pytest.raises(ValueError, match="reason class"):
        CommandRefusal(target_id="card", reason_class="unknown", message="no")
    with pytest.raises(ValueError, match="action_id"):
        CommandRefusal(target_id="card", reason_class="structural", message="no", action_id="")
    with pytest.raises(ValueError, match="step_id"):
        CommandRefusal(target_id="card", reason_class="structural", message="no", step_id=0)
    with pytest.raises(TypeError, match="retryable"):
        CommandRefusal(target_id="card", reason_class="structural", message="no", retryable=1)  # type: ignore[arg-type]

    refusal = CommandRefusal(
        target_id="card",
        reason_class=RefusalReasonClass.TRANSIENT,
        message="try later",
    )
    accepted_episode = CounterEnvironment(target=1).reset().episode_id
    fallback_receipt = refusal.to_receipt(episode_id=accepted_episode, step_id=1)
    assert fallback_receipt.action_id == "step-1"
    receipt = refusal.to_receipt(
        episode_id=accepted_episode,
        step_id=1,
        action_id="move",
        issued_timestamp_ns=10,
        observed_timestamp_ns=9,
    )
    assert receipt.episode_id == accepted_episode
    assert receipt.retryable is True


def _with_receipt(result: Any, *, action_id: str) -> TimeStep:
    return result.__class__(
        observation=result.observation,
        reward=result.reward,
        terminated=result.terminated,
        truncated=result.truncated,
        episode_id=result.episode_id,
        step_id=result.step_id,
        action_mask=result.action_mask,
        action_receipt=ActionReceipt(
            action_id=action_id,
            episode_id=result.episode_id,
            step_id=result.step_id,
            outcome=ActionOutcome.REJECTED,
            issued_timestamp_ns=1,
            observed_timestamp_ns=2,
            target_id="card-1",
            reason_class=RefusalReasonClass.STRUCTURAL,
        ),
    )
