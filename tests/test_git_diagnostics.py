"""Synthetic Git failures: no real executable or process is started."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired

import pytest

import game_learning_runtime.fork_gate as fork_module
from game_learning_runtime.cli import main
from game_learning_runtime.fork_gate import (
    ForkGatePolicy,
    ForkGateReport,
    GitRepositoryProbe,
    StaticRepositoryProbe,
    evaluate_fork_gate,
)
from game_learning_runtime.git_diagnostics import (
    GIT_PROBE_DIAGNOSTIC_SCHEMA,
    GitProbeDiagnostic,
    GitProbeOperation,
    GitProbeStatus,
    redact_git_stderr,
)


@pytest.fixture
def resolved_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    executable = str(tmp_path / "synthetic-git")
    monkeypatch.setattr(fork_module, "which", lambda _: executable)
    return executable


def test_success_records_resolved_executable_caller_and_utc_without_args_or_stdout(
    tmp_path: Path, resolved_git: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def execute(command, **options):
        calls.append((command, options))
        return CompletedProcess(command, 0, "synthetic-branch\n", "")

    monkeypatch.setattr(fork_module, "run", execute)
    probe = GitRepositoryProbe(tmp_path)
    assert probe.current_branch() == "synthetic-branch"
    assert len(calls) == 1
    assert calls[0][0][0] == str(Path(resolved_git).resolve())
    assert calls[0][1]["timeout"] == 30
    diagnostic = probe.diagnostics[0]
    payload = diagnostic.to_mapping()
    assert diagnostic.status is GitProbeStatus.SUCCESS
    assert diagnostic.operation is GitProbeOperation.BRANCH
    assert diagnostic.exit_code == 0
    assert diagnostic.caller_pid == os.getpid()
    assert diagnostic.executable_path == str(Path(resolved_git).resolve())
    assert diagnostic.process_terminal is True
    assert datetime.fromisoformat(diagnostic.started_utc).utcoffset() == timedelta(0)
    assert datetime.fromisoformat(diagnostic.completed_utc).utcoffset() == timedelta(0)
    assert payload["schema_version"] == GIT_PROBE_DIAGNOSTIC_SCHEMA
    assert payload["source"] == "glr.fork-gate"
    assert payload["timed_out"] is False
    for key in ["args", "command", "environment", "stdout", "root"]:
        assert key not in payload
    assert "synthetic-branch" not in json.dumps(payload)


def test_selected_symlink_wrapper_name_is_preserved_while_receipt_uses_canonical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "synthetic-wrapper"
    canonical = tmp_path / "synthetic-multicall"
    original_resolve = Path.resolve
    calls = []

    def resolve(path, *args, **kwargs):
        return canonical if path == selected else original_resolve(path, *args, **kwargs)

    def execute(command, **options):
        calls.append(command)
        return CompletedProcess(command, 0, "branch", "")

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(fork_module, "which", lambda _: str(selected))
    monkeypatch.setattr(fork_module, "run", execute)
    probe = GitRepositoryProbe(tmp_path)
    assert probe.current_branch() == "branch"
    assert len(calls) == 1
    assert calls[0][0] == str(selected.absolute())
    assert probe.diagnostics[0].executable_path == str(canonical)


@pytest.mark.parametrize("exit_code", [-1073741502, 3221225794, 128])
def test_nonzero_initialization_result_is_visible_without_changing_optional_return(
    tmp_path: Path, resolved_git: str, exit_code: int, caplog: pytest.LogCaptureFixture
) -> None:
    calls = []

    def execute(command):
        calls.append(command)
        return CompletedProcess(command, exit_code, "untrusted stdout", "0xc0000142 private-detail")

    probe = GitRepositoryProbe(tmp_path, runner=execute)
    with caplog.at_level(logging.WARNING):
        assert probe.origin_url() is None
    assert len(calls) == 1
    diagnostic = probe.diagnostics[0]
    assert diagnostic.status is GitProbeStatus.NONZERO_EXIT
    assert diagnostic.exit_code == exit_code
    assert "0xc0000142" in diagnostic.stderr
    assert "private-detail" not in caplog.text
    assert "untrusted stdout" not in caplog.text
    assert "nonzero_exit" in caplog.text
    assert str(exit_code) in caplog.text


def test_optional_missing_git_does_not_start_any_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fork_module, "which", lambda _: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("missing Git must not launch a process")

    monkeypatch.setattr(fork_module, "run", forbidden)
    probe = GitRepositoryProbe(tmp_path)
    assert probe.origin_url() is None
    assert probe.diagnostics[0].status is GitProbeStatus.MISSING_EXECUTABLE
    assert probe.diagnostics[0].executable_path is None
    assert probe.diagnostics[0].exit_code is None
    assert probe.diagnostics[0].process_terminal is True


def test_spawn_exception_is_recorded_then_preserves_existing_exception_semantics(
    tmp_path: Path,
    resolved_git: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = OSError("synthetic confidential exception detail")

    def execute(*args, **kwargs):
        raise error

    monkeypatch.setattr(fork_module, "run", execute)
    probe = GitRepositoryProbe(tmp_path)
    with pytest.raises(OSError) as captured, caplog.at_level(logging.WARNING):
        probe.origin_url()
    assert captured.value is error
    assert probe.diagnostics[0].status is GitProbeStatus.SPAWN_ERROR
    assert probe.diagnostics[0].exit_code is None
    assert probe.diagnostics[0].process_terminal is None
    assert "confidential exception detail" not in caplog.text


@pytest.mark.parametrize("injected", [False, True])
def test_timeout_records_terminal_evidence_without_logging_arguments(
    tmp_path: Path,
    resolved_git: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    injected: bool,
) -> None:
    error = TimeoutExpired(["never-log-this-argument"], 30, stderr=b"token=synthetic-secret")
    calls = []

    def execute(*args, **kwargs):
        calls.append(args)
        raise error

    monkeypatch.setattr(fork_module, "run", execute)
    probe = GitRepositoryProbe(tmp_path, runner=execute if injected else None)
    with pytest.raises(TimeoutExpired) as captured, caplog.at_level(logging.WARNING):
        probe.divergence("main")
    assert captured.value is error
    assert len(calls) == 1
    diagnostic = probe.diagnostics[0]
    assert diagnostic.status is GitProbeStatus.TIMEOUT
    assert diagnostic.to_mapping()["timed_out"] is True
    assert diagnostic.process_terminal is (None if injected else True)
    assert "never-log-this-argument" not in caplog.text
    assert "synthetic-secret" not in caplog.text


@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: authentication failed for https://synthetic-user:synthetic-secret@example.invalid/private",
        "Authorization: Bearer synthetic-secret",
        "password=synthetic-secret",
        "credential.helper=synthetic-secret",
        "fatal: not a git repository: /private/synthetic-secret",
        "fatal: ambiguous argument synthetic-secret",
        "fatal: unable to access https://example.invalid/private?token=synthetic-secret",
        "unknown exception detail contains synthetic-secret",
        b"token=synthetic-secret\xff",
    ],
)
def test_stderr_projection_denies_all_variable_credentials_paths_and_urls(stderr) -> None:
    redacted, truncated = redact_git_stderr(stderr)
    assert redacted
    assert truncated is False
    for forbidden in [
        "synthetic-secret",
        "synthetic-user",
        "example.invalid",
        "/private",
        "https://",
    ]:
        assert forbidden not in redacted


def test_stderr_projection_has_input_line_and_output_bounds() -> None:
    projected, truncated = redact_git_stderr(
        "fatal: authentication failed synthetic-secret\n" * 5000
    )
    assert truncated is True
    assert len(projected.encode("utf-8")) <= 2048
    assert projected.endswith("[truncated]")
    assert "synthetic-secret" not in projected
    assert redact_git_stderr(None) == ("", False)
    assert redact_git_stderr("\n  \n") == ("", False)
    with pytest.raises(TypeError):
        redact_git_stderr(123)


@pytest.mark.parametrize(
    "stderr",
    [
        "fatal: unknown revision synthetic-secret",
        "fatal: not a git repository /private/synthetic-secret",
        "error: no such remote synthetic-secret",
        "fatal: authentication failed synthetic-secret",
        "fatal: unable to access synthetic-secret",
        "0xc0000142 synthetic-secret",
    ],
)
def test_fixed_error_categories_remain_stable_when_a_receipt_is_sanitized_again(stderr) -> None:
    projected, truncated = redact_git_stderr(stderr)
    assert redact_git_stderr(projected) == (projected, truncated)


def test_constructor_also_redacts_untrusted_stderr_and_exports_only_declared_fields() -> None:
    diagnostic = GitProbeDiagnostic(
        operation=GitProbeOperation.ORIGIN,
        sequence_id=1,
        executable_path=None,
        caller_pid=1,
        started_utc="2025-01-01T00:00:00+00:00",
        completed_utc="2025-01-01T00:00:01+00:00",
        status=GitProbeStatus.NONZERO_EXIT,
        exit_code=128,
        stderr="opaque synthetic-secret",
    )
    assert "synthetic-secret" not in json.dumps(diagnostic.to_mapping())
    assert "redacted" in diagnostic.stderr
    with pytest.raises(ValueError):
        replace(diagnostic, caller_pid=True)
    with pytest.raises(ValueError):
        replace(diagnostic, started_utc="2025-01-01T00:00:00+01:00")
    with pytest.raises(ValueError):
        replace(diagnostic, exit_code=True)


def test_diagnostic_sink_receives_all_attempts_with_bounded_local_history(
    tmp_path: Path,
    resolved_git: str,
) -> None:
    receipts = []
    calls = []

    def execute(command):
        calls.append(command)
        return CompletedProcess(command, 0, "branch", "")

    probe = GitRepositoryProbe(tmp_path, runner=execute, diagnostic_sink=receipts.append)
    for _ in range(18):
        assert probe.current_branch() == "branch"
    assert len(calls) == len(receipts) == 18
    assert len(probe.diagnostics) == 16
    assert [item.sequence_id for item in probe.diagnostics] == list(range(3, 19))


def test_diagnostic_sink_failure_does_not_retry_or_hide_the_original_result(
    tmp_path: Path,
    resolved_git: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = []

    def execute(command):
        calls.append(command)
        return CompletedProcess(command, 128, "", "error: no such remote synthetic-secret")

    def fail_sink(receipt):
        raise ValueError("synthetic-secret")

    probe = GitRepositoryProbe(tmp_path, runner=execute, diagnostic_sink=fail_sink)
    with caplog.at_level(logging.WARNING):
        assert probe.origin_url() is None
    assert len(calls) == 1
    assert probe.diagnostics[0].exit_code == 128
    assert "sink failed" in caplog.text
    assert "synthetic-secret" not in caplog.text


def test_existing_gate_report_exposes_diagnostics_even_when_missing_data_is_allowed(
    tmp_path: Path,
    resolved_git: str,
) -> None:
    probe = GitRepositoryProbe(
        tmp_path,
        runner=lambda command: CompletedProcess(command, -1073741502, "", ""),
    )
    policy = ForkGatePolicy(
        "https://example.invalid/public.git",
        require_origin_match=False,
        require_upstream_ref=False,
        require_version_alignment=False,
    )
    report = evaluate_fork_gate(probe, policy)
    assert report.passed is True
    diagnostics = report.to_mapping()["git_diagnostics"]
    assert len(diagnostics) == 2
    assert all(item["exit_code"] == -1073741502 for item in diagnostics)
    assert all(item["status"] == "nonzero_exit" for item in diagnostics)
    assert evaluate_fork_gate(StaticRepositoryProbe(), policy).git_diagnostics == ()
    with pytest.raises(ValueError):
        ForkGateReport((), git_diagnostics=(object(),))


@pytest.mark.parametrize("status", ["success", "nonzero_exit", "missing_executable"])
def test_cli_keeps_its_json_envelope_and_exposes_the_same_probe_receipts(
    tmp_path: Path,
    resolved_git: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    calls = []

    def execute(command, **options):
        calls.append(command)
        assert status != "missing_executable"
        if status == "nonzero_exit":
            return CompletedProcess(command, 128, "", "token=synthetic-secret")
        stdout = "https://example.invalid/public.git" if "remote" in command else "0 0"
        return CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(fork_module, "run", execute)
    if status == "missing_executable":
        monkeypatch.setattr(fork_module, "which", lambda _: None)
    code = main(
        [
            "--project",
            str(tmp_path),
            "--format",
            "json",
            "fork-gate",
            "--origin",
            "https://example.invalid/public.git",
            "--allow-version-drift",
            "--ignore-schema-versions",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "glr.cli-output.v1"
    assert payload["command"] == "fork-gate"
    data = payload["data"]
    assert data["schema_version"] == "glr.fork-gate-report.v1"
    assert code == data["exit_code"] == (0 if status == "success" else 5)
    assert len(calls) == (0 if status == "missing_executable" else 2)
    assert [item["operation"] for item in data["git_diagnostics"]] == ["origin", "divergence"]
    assert all(item["status"] == status for item in data["git_diagnostics"])
    assert "synthetic-secret" not in json.dumps(payload)
