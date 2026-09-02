from __future__ import annotations

from uuid import uuid4

import pytest

from game_learning_runtime import (
    DeepSeekHarnessProvider,
    HarnessDisabledError,
    HarnessEvent,
    HarnessPermission,
    HarnessPermissionError,
    HarnessRecoveryError,
    HarnessResultStatus,
    HarnessTask,
    LocalHarnessOrchestrator,
)


def _task(**kwargs: object) -> HarnessTask:
    values: dict[str, object] = {
        "kind": "analysis",
        "payload": {"question": "summarize"},
        "idempotency_key": "task-1",
    }
    values.update(kwargs)
    return HarnessTask(**values)  # type: ignore[arg-type]


def test_deepseek_provider_is_disabled_without_explicit_enablement() -> None:
    provider = DeepSeekHarnessProvider(handler=lambda task: {"ok": True})
    with pytest.raises(HarnessDisabledError):
        provider.submit(_task())


def test_provider_declares_advisory_capabilities_without_runtime_mutation() -> None:
    provider = DeepSeekHarnessProvider()
    capabilities = provider.capabilities
    assert capabilities.provider == "deepseek-harness"
    assert HarnessPermission.RUNTIME_ACT not in capabilities.permissions
    assert capabilities.state_recovery


def test_provider_returns_structured_result_and_deduplicates() -> None:
    calls = 0

    def handler(task: HarnessTask) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"kind": task.kind, "answer": 42}

    provider = DeepSeekHarnessProvider(enabled=True, handler=handler)
    first = provider.submit(_task())
    duplicate = provider.submit(_task(task_id=uuid4()))
    assert first.status is HarnessResultStatus.COMPLETED
    assert duplicate == first
    assert calls == 1
    with pytest.raises(TypeError):
        first.output["answer"] = 1  # type: ignore[index]


def test_permission_boundary_rejects_runtime_action() -> None:
    provider = DeepSeekHarnessProvider(enabled=True, handler=lambda task: {})
    task = _task(permissions=frozenset({HarnessPermission.RUNTIME_ACT}))
    with pytest.raises(HarnessPermissionError, match=r"runtime\.act"):
        provider.submit(task)
    with pytest.raises(HarnessPermissionError, match="cannot be granted"):
        DeepSeekHarnessProvider(allowed_permissions=(HarnessPermission.RUNTIME_ACT,))


def test_failure_and_timeout_are_idempotent_results() -> None:
    failed = DeepSeekHarnessProvider(enabled=True, handler=lambda task: 1 / 0)
    result = failed.submit(_task(idempotency_key="failed"))
    assert result.status is HarnessResultStatus.FAILED
    assert failed.submit(_task(idempotency_key="failed")) == result

    def timeout_handler(task: HarnessTask) -> dict[str, object]:
        raise TimeoutError

    timed_out = DeepSeekHarnessProvider(enabled=True, handler=timeout_handler)
    result = timed_out.submit(_task(idempotency_key="timeout"))
    assert result.status is HarnessResultStatus.TIMED_OUT
    assert timed_out.submit(_task(idempotency_key="timeout")) == result


def test_orchestrator_events_and_state_recovery() -> None:
    provider = DeepSeekHarnessProvider(enabled=True, handler=lambda task: {"ok": True})
    orchestrator = LocalHarnessOrchestrator(provider)
    result = orchestrator.submit(_task())
    assert result.status is HarnessResultStatus.COMPLETED
    assert [event.event_type for event in orchestrator.events()] == [
        "task.submitted",
        "task.completed",
    ]
    snapshot = orchestrator.snapshot()

    restored = LocalHarnessOrchestrator(
        DeepSeekHarnessProvider(enabled=True, handler=lambda task: {"unexpected": True})
    )
    restored.restore(snapshot)
    assert restored.submit(_task(task_id=uuid4())) == result
    assert len(restored.events()) == 4

    with pytest.raises(HarnessRecoveryError):
        restored.restore(type(snapshot)("other", snapshot.schema_version))


def test_event_requires_non_negative_sequence() -> None:
    with pytest.raises(ValueError, match="sequence"):
        HarnessEvent("task.completed", uuid4(), -1)
