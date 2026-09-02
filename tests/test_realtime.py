from __future__ import annotations

import pytest

from game_learning_runtime import (
    InputLeaseBook,
    InputLeaseOperation,
    InputLeaseReceipt,
    InputLeaseRequest,
    InputLeaseStatus,
    InputLeaseToken,
    RealtimeActionReceipt,
    RealtimeActionStatus,
    RealtimeStepTiming,
    RealtimeTimingContract,
)


def test_realtime_timing_contract_bounds_short_windows_and_quantum() -> None:
    contract = RealtimeTimingContract(
        minimum_hold_ns=10,
        maximum_hold_ns=100,
        settle_deadline_ns=200,
        simulation_quantum_ns=20,
    )
    RealtimeStepTiming(50, 10, 20).validate_against(contract)
    assert RealtimeStepTiming(50, 10, 20).to_mapping()["hold_ns"] == 20

    with pytest.raises(ValueError, match="settle_deadline"):
        RealtimeStepTiming(201, 10).validate_against(contract)
    with pytest.raises(ValueError, match="simulation_quantum"):
        RealtimeStepTiming(50, 21).validate_against(contract)
    with pytest.raises(ValueError, match="hold bounds"):
        RealtimeStepTiming(50, 10, 5).validate_against(contract)
    assert RealtimeTimingContract.from_mapping(contract.to_mapping()) == contract

    with pytest.raises(ValueError, match="unsupported realtime control schema"):
        RealtimeTimingContract(1, 2, 3, 1, schema_version="glr.realtime-control.v0")
    with pytest.raises(ValueError, match="minimum_hold_ns"):
        RealtimeTimingContract(3, 2, 3, 1)
    with pytest.raises(ValueError, match="maximum_hold_ns"):
        RealtimeTimingContract(1, 4, 3, 1)
    with pytest.raises(ValueError, match="clock_source"):
        RealtimeTimingContract(1, 2, 3, 1, clock_source="")
    with pytest.raises(ValueError, match="missing"):
        RealtimeTimingContract.from_mapping({"schema_version": contract.schema_version})
    with pytest.raises(ValueError, match="unexpected"):
        RealtimeTimingContract.from_mapping({**contract.to_mapping(), "extra": True})
    with pytest.raises(ValueError, match="positive"):
        RealtimeStepTiming(0, 1)
    with pytest.raises(ValueError, match="positive"):
        RealtimeStepTiming(1, 0)
    with pytest.raises(ValueError, match="positive"):
        RealtimeStepTiming(1, 1, 0)


def test_input_lease_book_fences_identity_expiry_and_preemption() -> None:
    book = InputLeaseBook(clock=lambda: 100)
    acquired = book.apply(
        InputLeaseRequest(
            InputLeaseOperation.ACQUIRE,
            session_id="session.one",
            target_id="target.game",
            expires_at_ns=200,
        )
    )
    assert acquired.status is InputLeaseStatus.ACQUIRED
    assert acquired.token is not None
    token = acquired.token
    assert book.authorize(token)

    held = book.apply(
        InputLeaseRequest(
            InputLeaseOperation.ACQUIRE,
            session_id="session.two",
            target_id="target.game",
            expires_at_ns=300,
        )
    )
    assert held.status is InputLeaseStatus.REJECTED

    mismatch = book.apply(
        InputLeaseRequest(
            InputLeaseOperation.RENEW,
            session_id="session.two",
            target_id="target.game",
            lease_id=token.lease_id,
            expires_at_ns=300,
        )
    )
    assert mismatch.status is InputLeaseStatus.REJECTED
    renewed = book.apply(
        InputLeaseRequest(
            InputLeaseOperation.RENEW,
            session_id=token.session_id,
            target_id=token.target_id,
            lease_id=token.lease_id,
            expires_at_ns=300,
        )
    )
    assert renewed.status is InputLeaseStatus.RENEWED
    expired_renewal = book.apply(
        InputLeaseRequest(
            InputLeaseOperation.RENEW,
            session_id=token.session_id,
            target_id=token.target_id,
            lease_id=token.lease_id,
            expires_at_ns=100,
        )
    )
    assert expired_renewal.status is InputLeaseStatus.REJECTED
    preempted = book.apply(
        InputLeaseRequest(
            InputLeaseOperation.PREEMPT,
            session_id=token.session_id,
            target_id=token.target_id,
            lease_id=token.lease_id,
        )
    )
    assert preempted.status is InputLeaseStatus.PREEMPTED
    assert not book.authorize(token)

    released = InputLeaseBook(clock=lambda: 100)
    token = released.apply(
        InputLeaseRequest(
            InputLeaseOperation.ACQUIRE,
            "session.one",
            "target.game",
            expires_at_ns=200,
        )
    ).token
    assert token is not None
    receipt = released.apply(
        InputLeaseRequest(
            InputLeaseOperation.RELEASE,
            "session.one",
            "target.game",
            lease_id=token.lease_id,
        )
    )
    assert receipt.status is InputLeaseStatus.RELEASED
    assert released.active is None
    assert (
        released.apply(
            InputLeaseRequest(
                InputLeaseOperation.RENEW,
                "session.one",
                "target.game",
                lease_id=token.lease_id,
                expires_at_ns=300,
            )
        ).reason
        == "lease is absent or expired"
    )

    assert not released.authorize(token, now_ns=300)

    with pytest.raises(ValueError, match="expires_at_ns"):
        InputLeaseRequest(
            InputLeaseOperation.ACQUIRE,
            "session.one",
            "target.game",
            expires_at_ns=0,
        )
    with pytest.raises(ValueError, match="lease_id"):
        InputLeaseRequest(
            InputLeaseOperation.ACQUIRE,
            "session.one",
            "target.game",
            lease_id="session.one.lease",
        )
    with pytest.raises(ValueError, match="lease_id is required"):
        InputLeaseRequest(InputLeaseOperation.RELEASE, "session.one", "target.game")
    with pytest.raises(ValueError, match="non-negative"):
        InputLeaseReceipt(InputLeaseStatus.REJECTED, None, -1)
    with pytest.raises(TypeError, match="InputLeaseToken"):
        InputLeaseReceipt(InputLeaseStatus.REJECTED, "bad", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="after"):
        InputLeaseReceipt(InputLeaseStatus.ACQUIRED, None, 10, expires_at_ns=10)
    with pytest.raises(ValueError, match="reason"):
        InputLeaseReceipt(InputLeaseStatus.REJECTED, None, 1, reason="")


def test_realtime_receipts_distinguish_consumed_expired_and_cancelled() -> None:
    consumed = RealtimeActionReceipt(
        action_id="action.one",
        status=RealtimeActionStatus.CONSUMED,
        deadline_ns=100,
        quantum_ns=10,
        issued_at_ns=1_000,
        consumed_at_ns=1_050,
        settled_at_ns=1_060,
    )
    expired = RealtimeActionReceipt(
        action_id="action.two",
        status=RealtimeActionStatus.EXPIRED,
        deadline_ns=5,
        quantum_ns=1,
        issued_at_ns=2_000,
    )
    cancelled = RealtimeActionReceipt(
        action_id="action.three",
        status=RealtimeActionStatus.CANCELLED,
        deadline_ns=5,
        quantum_ns=1,
        issued_at_ns=3_000,
        cancellation_token="cancel.three",
    )

    assert consumed.to_mapping()["status"] == "consumed"
    assert expired.status is RealtimeActionStatus.EXPIRED
    assert cancelled.cancellation_token == "cancel.three"
    with pytest.raises(ValueError, match="exceeds deadline"):
        RealtimeActionReceipt(
            action_id="action.bad",
            status=RealtimeActionStatus.CONSUMED,
            deadline_ns=1,
            quantum_ns=1,
            issued_at_ns=10,
            consumed_at_ns=12,
        )
    with pytest.raises(ValueError, match="cannot precede"):
        RealtimeActionReceipt(
            action_id="action.bad",
            status=RealtimeActionStatus.CONSUMED,
            deadline_ns=10,
            quantum_ns=1,
            issued_at_ns=10,
            settled_at_ns=9,
        )
    with pytest.raises(ValueError, match="identifier"):
        RealtimeActionReceipt(
            action_id="Action/bad",
            status=RealtimeActionStatus.REJECTED,
            deadline_ns=1,
            quantum_ns=1,
            issued_at_ns=0,
        )
    with pytest.raises(ValueError, match="cannot precede"):
        RealtimeActionReceipt(
            action_id="action.bad",
            status=RealtimeActionStatus.CONSUMED,
            deadline_ns=10,
            quantum_ns=1,
            issued_at_ns=10,
            consumed_at_ns=11,
            settled_at_ns=10,
        )


def test_input_lease_token_rejects_unbound_identifiers() -> None:
    with pytest.raises(ValueError, match="identifier"):
        InputLeaseToken("Lease/one", "session.one", "target.game")
