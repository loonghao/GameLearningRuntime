from __future__ import annotations

import pytest

from game_learning_runtime import (
    ArtifactOwnershipError,
    ExclusiveInstanceLease,
    LeaseConflictError,
    ProcessIdentity,
    ProcessSupervisor,
    StopAction,
    SupervisionError,
)


class FakeProcess:
    def __init__(self) -> None:
        self.alive: set[ProcessIdentity] = set()
        self.actions: list[str] = []
        self.counter = 0

    def is_alive(self, identity: ProcessIdentity) -> bool:
        return identity in self.alive

    def invoke(self, action: str, identity: ProcessIdentity) -> None:
        self.actions.append(action)
        if action in {"request-close", "terminate", "kill"}:
            self.alive.discard(identity)

    def launch(self) -> ProcessIdentity:
        self.counter += 1
        identity = ProcessIdentity(self.counter, self.counter)
        self.alive.add(identity)
        return identity


def test_supervisor_enforces_lease_and_graceful_stop() -> None:
    probe = FakeProcess()
    identity = probe.launch()
    lease = ExclusiveInstanceLease()
    first = ProcessSupervisor(probe, lease=lease)
    first.attach(identity)
    second = ProcessSupervisor(probe, lease=lease)
    with pytest.raises(LeaseConflictError):
        second.attach(identity)
    with pytest.raises(ArtifactOwnershipError):
        first.require_artifact_stopped("prune")
    result = first.stop()
    assert result.stopped and result.ended_by == "request-close"
    first.require_artifact_stopped("prune")
    assert probe.actions == ["request-close"]


def test_restart_keeps_exclusivity_and_counts_launches() -> None:
    probe = FakeProcess()
    supervisor = ProcessSupervisor(probe)
    supervisor.attach(probe.launch())
    restarted = supervisor.restart()
    assert supervisor.identity == restarted
    assert supervisor.restart_count == 1
    with pytest.raises(SupervisionError, match="already attached"):
        supervisor.attach(restarted)


def test_process_contract_validation_and_escalation() -> None:
    with pytest.raises(ValueError, match="positive"):
        ProcessIdentity(0, 1)
    with pytest.raises(ValueError, match="timeout"):
        StopAction("kill", 0)
    with pytest.raises(ValueError, match="empty"):
        ProcessSupervisor(FakeProcess(), stop_sequence=())
    with pytest.raises(ValueError, match="start_time"):
        ProcessIdentity(1, -1)
    assert ProcessIdentity(1, 2).to_mapping() == {"pid": 1, "start_time_ns": 2}
    with pytest.raises(ValueError, match="printable"):
        StopAction("bad\nname", 1)


def test_lease_and_supervisor_fail_closed_on_invalid_ownership() -> None:
    probe = FakeProcess()
    lease = ExclusiveInstanceLease()
    identity = ProcessIdentity(1, 1)
    with pytest.raises(SupervisionError, match="identity"):
        lease.release(identity)
    supervisor = ProcessSupervisor(probe, lease=lease)
    with pytest.raises(SupervisionError, match="not alive"):
        supervisor.attach(identity)
    with pytest.raises(ValueError, match="printable"):
        supervisor.require_artifact_stopped("bad\noperation")
    with pytest.raises(SupervisionError, match="attached"):
        supervisor.stop()


def test_stop_escalates_and_records_result() -> None:
    class Escalating(FakeProcess):
        def invoke(self, action: str, identity: ProcessIdentity) -> None:
            self.actions.append(action)
            if action == "terminate":
                self.alive.discard(identity)

    probe = Escalating()
    identity = probe.launch()
    supervisor = ProcessSupervisor(
        probe,
        stop_sequence=(StopAction("request-close", 0.001), StopAction("terminate", 0.1)),
        sleep_fn=lambda _: None,
    )
    supervisor.attach(identity)
    result = supervisor.stop()
    assert result.ended_by == "terminate"
    assert probe.actions == ["request-close", "terminate"]
    assert result.to_mapping()["stopped"] is True


def test_stopped_verification_and_restart_failure_are_explicit() -> None:
    probe = FakeProcess()
    identity = probe.launch()
    supervisor = ProcessSupervisor(probe)
    supervisor.attach(identity)
    probe.alive.clear()
    result = supervisor.stop()
    assert result.ended_by == "already-stopped"

    class Stubborn(FakeProcess):
        def invoke(self, action: str, identity: ProcessIdentity) -> None:
            self.actions.append(action)

    stubborn = Stubborn()
    stubborn_identity = stubborn.launch()
    blocked = ProcessSupervisor(
        stubborn,
        stop_sequence=(StopAction("request-close", 0.0001),),
        sleep_fn=lambda _: None,
    )
    blocked.attach(stubborn_identity)
    with pytest.raises(SupervisionError, match="did not stop"):
        blocked.restart()
