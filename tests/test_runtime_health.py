from __future__ import annotations

import pytest

from game_learning_runtime import (
    RUNTIME_HEALTH_SCHEMA_VERSION,
    RuntimeHealth,
    RuntimeHealthStatus,
    RuntimeIdentity,
    RuntimeLease,
)


def test_runtime_health_round_trips_identity_and_lease() -> None:
    identity = RuntimeIdentity("runtime.family", "1.2.3")
    lease = RuntimeLease("upgrade.lease", "launcher.one", 200)
    health = RuntimeHealth(
        identity,
        RuntimeHealthStatus.DRAINING,
        observed_at_ns=100,
        accepting_new_sessions=False,
        active_sessions=2,
        lease=lease,
    )

    restored = RuntimeHealth.from_mapping(health.to_mapping())

    assert restored == health
    assert restored.schema_version == RUNTIME_HEALTH_SCHEMA_VERSION
    assert RuntimeIdentity.from_mapping(identity.to_mapping()) == identity
    assert RuntimeLease.from_mapping(lease.to_mapping()) == lease


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: RuntimeIdentity("bad/path", "1.0.0"), "runtime_id"),
        (lambda: RuntimeIdentity("runtime", ""), "runtime_version"),
        (lambda: RuntimeLease("lease", "owner", 0), "expires_at_ns"),
        (
            lambda: RuntimeHealth(
                RuntimeIdentity("runtime", "1.0.0"),
                RuntimeHealthStatus.READY,
                10,
                True,
                0,
                RuntimeLease("lease", "owner", 10),
            ),
            "expire",
        ),
    ],
)
def test_runtime_health_rejects_invalid_values(factory: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()  # type: ignore[operator]


def test_runtime_health_parser_rejects_missing_unknown_and_bad_nested_values() -> None:
    base = RuntimeHealth(
        RuntimeIdentity("runtime", "1.0.0"),
        RuntimeHealthStatus.READY,
        observed_at_ns=10,
        accepting_new_sessions=True,
        active_sessions=0,
    ).to_mapping()

    for invalid in (
        {key: value for key, value in base.items() if key != "identity"},
        {**base, "unexpected": True},
        {**base, "identity": "runtime"},
        {**base, "lease": "lease"},
        {**base, "status": "unknown"},
    ):
        with pytest.raises((TypeError, ValueError)):
            RuntimeHealth.from_mapping(invalid)


def test_runtime_health_lease_is_optional_on_the_wire() -> None:
    health = RuntimeHealth(
        RuntimeIdentity("runtime", "1.0.0"),
        RuntimeHealthStatus.READY,
        observed_at_ns=10,
        accepting_new_sessions=True,
        active_sessions=0,
    )

    wire = health.to_mapping()
    assert "lease" not in wire
    assert RuntimeHealth.from_mapping(wire).lease is None
