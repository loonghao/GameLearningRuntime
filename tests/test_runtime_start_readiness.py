"""`glr runtime start` and the project-declared startup readiness window."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from game_learning_runtime.cli import RUNTIME_NOT_READY_EXIT_CODE, main
from game_learning_runtime.run_store import RunStatus, TrainingStore

ROLE_SOURCE = """
import json
import os
import sys
from pathlib import Path

ready_after = int(sys.argv[1])
mode = sys.argv[2]
run_dir = Path(os.environ["GLR_RUN_DIR"])
counter = run_dir / "attempts.txt"
attempt = int(counter.read_text(encoding="utf-8")) + 1 if counter.is_file() else 1
counter.write_text(str(attempt), encoding="utf-8")
configured = os.environ.get("GLR_READINESS_PATH")
print(f"attempt={attempt} mode={mode} window={os.environ.get('GLR_READINESS_ATTEMPT')}")
if configured is None:
    print("no declared readiness window")
    sys.exit(63)
receipt = Path(configured)
if mode == "unreported":
    sys.exit(17)
if mode == "garbage":
    receipt.write_text("not json", encoding="utf-8")
    sys.exit(18)
if mode == "offschema":
    receipt.write_text(
        json.dumps({"schema_version": "glr.environment-readiness.v2", "state": "not_ready"}),
        encoding="utf-8",
    )
    sys.exit(18)
if mode == "oversized":
    receipt.write_text("x" * (65 * 1024), encoding="utf-8")
    sys.exit(18)
if mode == "directory":
    receipt.mkdir()
    sys.exit(18)
if mode == "unavailable":
    state, exit_code = "unavailable", 19
elif mode == "inconsistent":
    state, exit_code = "ready", 23
elif attempt >= ready_after:
    state, exit_code = "ready", 0
else:
    state, exit_code = "not_ready", 63
receipt.write_text(
    json.dumps(
        {
            "schema_version": "glr.environment-readiness.v1",
            "state": state,
            "reason": f"attempt {attempt}",
            "checked_at_ns": attempt,
        }
    ),
    encoding="utf-8",
)
sys.exit(exit_code)
"""


def _project(
    root: Path,
    *,
    ready_after: int = 99,
    mode: str = "park",
    timeout_seconds: float | None = 1.0,
    poll_interval_seconds: float = 0.05,
) -> Path:
    """Write a project whose runtime role parks until the host is usable."""

    (root / "bridge").mkdir()
    role = root / "runtime_role.py"
    role.write_text(ROLE_SOURCE.strip() + "\n", encoding="utf-8")
    interpreter = Path(sys.executable).as_posix()
    lines = [
        'schema_version = "glr.project.v1"',
        'environment_id = "example.adventure-v1"',
        'environment_family = "action-rpg"',
        'protocol_version = "1.0"',
        'data_dir = ".glr"',
        'bridge_path = "bridge"',
        "",
        "[runtime]",
        f'argv = ["{interpreter}", "{role.as_posix()}", "{ready_after}", "{mode}"]',
        "[trainer]",
        'argv = ["python", "-c", "print(\'train\')"]',
        "[player]",
        'argv = ["python", "-c", "print(\'play\')", "{bundle}"]',
    ]
    if timeout_seconds is not None:
        lines += [
            "",
            "[runtime.readiness]",
            f"timeout_seconds = {timeout_seconds}",
            f"poll_interval_seconds = {poll_interval_seconds}",
        ]
    (root / "glr-project.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _start(root: Path, capsys: object) -> tuple[int, dict]:
    exit_code = main(["--project", str(root), "--json", "runtime", "start"])
    return exit_code, json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]


def _attempts(root: Path, run_id: str) -> int:
    counter = root / ".glr" / "runs" / run_id / "attempts.txt"
    return int(counter.read_text(encoding="utf-8"))


def test_runtime_start_parks_until_the_host_reports_ready(tmp_path: Path, capsys: object) -> None:
    # A window wide enough that the run ends when the host becomes ready and
    # never on the clock: process spawn cost must not decide this test.
    _project(tmp_path, ready_after=3, timeout_seconds=60)

    exit_code, output = _start(tmp_path, capsys)

    assert exit_code == 0
    data = output["data"]
    assert data["status"] == "succeeded"
    assert data["readiness"]["verdict"] == "succeeded"
    assert data["readiness"]["exhausted"] is False
    assert [item["index"] for item in data["readiness"]["attempts"]] == [1, 2, 3]
    assert [item["readiness"]["state"] for item in data["readiness"]["attempts"]] == [
        "not_ready",
        "not_ready",
        "ready",
    ]
    assert _attempts(tmp_path, data["run_id"]) == 3


def test_runtime_start_records_a_booting_host_apart_from_a_crash(
    tmp_path: Path, capsys: object
) -> None:
    parked = tmp_path / "parked"
    parked.mkdir()
    _project(parked, ready_after=99, timeout_seconds=2.0)
    refused = tmp_path / "refused"
    refused.mkdir()
    _project(refused, mode="unreported")

    parked_exit, parked_output = _start(parked, capsys)
    refused_exit, refused_output = _start(refused, capsys)

    assert parked_exit == RUNTIME_NOT_READY_EXIT_CODE
    assert refused_exit == 17
    parked_run = parked_output["data"]
    refused_run = refused_output["data"]
    assert parked_run["status"] == refused_run["status"] == "failed"
    assert parked_run["exit_code"] == RUNTIME_NOT_READY_EXIT_CODE
    assert parked_run["readiness"]["verdict"] == "not_ready"
    assert parked_run["readiness"]["exhausted"] is True
    assert refused_run["readiness"]["verdict"] == "unreported"
    assert refused_run["readiness"]["exhausted"] is False
    assert _attempts(parked, parked_run["run_id"]) > 1
    assert _attempts(refused, refused_run["run_id"]) == 1

    store = TrainingStore(parked / ".glr/runs.sqlite3")
    events = store.list_events(parked_run["run_id"])
    assert [event.kind for event in events][-1] == "readiness.outcome"
    assert {event.kind for event in events} == {"readiness.attempt", "readiness.outcome"}
    assert events[-1].payload["verdict"] == "not_ready"
    assert events[-1].payload["exhausted"] is True
    assert events[-1].payload["attempts"][0]["readiness"]["schema_version"] == (
        "glr.environment-readiness.v1"
    )
    assert store.get_run(parked_run["run_id"]).status is RunStatus.FAILED
    attempts = _attempts(parked, parked_run["run_id"])
    assert sorted(
        artifact.path for artifact in store.list_artifacts(parked_run["run_id"])
    ) == sorted(
        ["runtime.log", *(f"runtime-attempt{index}.log" for index in range(2, attempts + 1))]
    )


def test_runtime_start_without_a_declared_window_is_unchanged(
    tmp_path: Path, capsys: object
) -> None:
    _project(tmp_path, ready_after=99, timeout_seconds=None)

    exit_code, output = _start(tmp_path, capsys)

    data = output["data"]
    assert exit_code == 63
    assert data["status"] == "failed"
    assert data["exit_code"] == 63
    assert "readiness" not in data
    assert _attempts(tmp_path, data["run_id"]) == 1
    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    assert store.list_events(data["run_id"]) == ()
    assert [artifact.path for artifact in store.list_artifacts(data["run_id"])] == ["runtime.log"]


@pytest.mark.parametrize(
    ("mode", "exit_code", "verdict"),
    [
        ("unavailable", 19, "unavailable"),
        ("inconsistent", 23, "inconsistent"),
        ("unreported", 17, "unreported"),
        ("garbage", 18, "unreported"),
        ("offschema", 18, "unreported"),
        ("oversized", 18, "unreported"),
        ("directory", 18, "unreported"),
    ],
)
def test_runtime_start_never_retries_a_terminal_role_report(
    tmp_path: Path, capsys: object, mode: str, exit_code: int, verdict: str
) -> None:
    _project(tmp_path, mode=mode)

    started, output = _start(tmp_path, capsys)

    data = output["data"]
    assert started == exit_code
    assert data["readiness"]["verdict"] == verdict
    assert data["readiness"]["exhausted"] is False
    assert len(data["readiness"]["attempts"]) == 1
    assert _attempts(tmp_path, data["run_id"]) == 1
