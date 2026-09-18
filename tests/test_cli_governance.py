from __future__ import annotations

import json
from pathlib import Path

import pytest

from game_learning_runtime.cli import (
    CANONICAL_ORIGIN_URL,
    FORK_GATE_SCHEMA_VERSIONS,
    LOCAL_SCHEMA_VERSIONS,
    main,
)
from game_learning_runtime.errors import ContractViolation


def test_pinned_fork_gate_schema_versions_match_this_checkout() -> None:
    """The gate's expectation and the modules' own constants must stay equal.

    The expectation side is pinned as a literal so a local edit to a schema
    constant is *detected* rather than silently matching itself. Keeping the two
    in sync is then a mechanical step: bump the literal with the module.
    """

    assert dict(FORK_GATE_SCHEMA_VERSIONS) == dict(LOCAL_SCHEMA_VERSIONS)
    assert FORK_GATE_SCHEMA_VERSIONS.keys() == LOCAL_SCHEMA_VERSIONS.keys()


def _fork_gate_payload(tmp_path: Path, *arguments: str) -> dict[str, object]:
    """Run `glr fork-gate` against a scratch root and return the emitted data."""

    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["--project", str(tmp_path), "--json", "fork-gate", *arguments])
    payload = json.loads(buffer.getvalue())
    assert payload["command"] == "fork-gate"
    assert payload["schema_version"] == "glr.cli-output.v1"
    assert code == payload["data"]["exit_code"]
    return dict(payload["data"])


def test_fork_gate_reports_blocking_findings_for_a_foreign_origin(tmp_path: Path) -> None:
    data = _fork_gate_payload(tmp_path, "--origin", "https://example.com/other.git")
    assert data["passed"] is False
    assert data["exit_code"] == 5
    assert "origin-url" in data["blockers"]


def test_fork_gate_accepts_the_canonical_origin_when_it_is_unreachable(tmp_path: Path) -> None:
    data = _fork_gate_payload(
        tmp_path,
        "--origin",
        CANONICAL_ORIGIN_URL,
        "--allow-foreign-origin",
        "--allow-missing-upstream",
        "--allow-version-drift",
    )
    assert data["passed"] is True
    assert data["exit_code"] == 0


def test_fork_gate_can_skip_schema_checks(tmp_path: Path) -> None:
    baseline = _fork_gate_payload(
        tmp_path,
        "--allow-foreign-origin",
        "--allow-missing-upstream",
        "--allow-version-drift",
    )
    skipped = _fork_gate_payload(
        tmp_path,
        "--allow-foreign-origin",
        "--allow-missing-upstream",
        "--allow-version-drift",
        "--ignore-schema-versions",
    )
    assert any(str(check).startswith("schema-version:") for check in _checks(baseline))
    assert not any(str(check).startswith("schema-version:") for check in _checks(skipped))


def _checks(data: dict[str, object]) -> list[str]:
    findings = data["findings"]
    assert isinstance(findings, list)
    return [str(item["check"]) for item in findings]  # type: ignore[index]


def test_fork_gate_uses_a_file_project_path(tmp_path: Path) -> None:
    marker = tmp_path / "glr-project.json"
    marker.write_text("{}", encoding="utf-8")
    data = _fork_gate_payload(
        marker,
        "--allow-foreign-origin",
        "--allow-missing-upstream",
        "--allow-version-drift",
    )
    assert data["exit_code"] == 0


def _watchdog_payload(tmp_path: Path, *arguments: str) -> tuple[int, dict[str, object]]:
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["--json", "watchdog", "tick", *arguments])
    payload = json.loads(buffer.getvalue())
    assert payload["command"] == "watchdog.tick"
    return code, dict(payload["data"])


def test_watchdog_tick_reports_a_healthy_source(tmp_path: Path) -> None:
    log = tmp_path / "heartbeats.jsonl"
    log.write_text(
        json.dumps({"source": "trainer", "sequence": 1, "observed_at_ns": 10**18}) + "\n",
        encoding="utf-8",
    )
    code, data = _watchdog_payload(tmp_path, "--source", "trainer", "--heartbeats", str(log))
    decision = data["decisions"][0]  # type: ignore[index]
    assert code == 0
    assert decision["status"] == "healthy"
    assert decision["action"] == "none"


def test_watchdog_tick_discovers_sources_from_the_heartbeat_log(tmp_path: Path) -> None:
    log = tmp_path / "heartbeats.jsonl"
    log.write_text(
        "\n".join(
            json.dumps({"source": name, "sequence": 1, "observed_at_ns": 10**18})
            for name in ("collector", "trainer")
        )
        + "\n",
        encoding="utf-8",
    )
    code, data = _watchdog_payload(tmp_path, "--heartbeats", str(log))
    sources = sorted(str(d["source"]) for d in data["decisions"])  # type: ignore[index]
    assert code == 0
    assert sources == ["collector", "trainer"]


def test_watchdog_tick_escalates_when_recovery_keeps_failing(tmp_path: Path) -> None:
    log = tmp_path / "heartbeats.jsonl"
    log.write_text(
        json.dumps({"source": "trainer", "sequence": 1, "observed_at_ns": 1}) + "\n",
        encoding="utf-8",
    )
    code, data = _watchdog_payload(
        tmp_path,
        "--source",
        "trainer",
        "--heartbeats",
        str(log),
        "--timeout",
        "0.001",
        "--restart-attempt-limit",
        "1",
        "--recovery-command",
        "cmd",
        "--recovery-command",
        "/c",
        "--recovery-command",
        "exit 1",
    )
    assert code == 4
    assert data["exit_code"] == 4
    assert data["escalated"] == ["trainer"]


def test_watchdog_tick_requires_a_source_or_a_log(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="requires at least one --source"):
        main(["--json", "watchdog", "tick"])
