"""Bounded local receipts for existing Git probes, without command or secret output.

These diagnostics do not establish a crash's cause or observe other processes.
Stderr is projected to fixed categories; arbitrary text is denied by default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

GIT_PROBE_DIAGNOSTIC_SCHEMA = "glr.git-probe-diagnostic.v1"
_INPUT_LIMIT = 16384
_STDERR_LIMIT = 2048
_LINE_LIMIT = 64
_REDACTED = "[Git stderr redacted]"
_TRUNCATED = "[truncated]"
_CATEGORIES = (
    (r"not a git repository", "fatal: not a Git repository [location redacted]"),
    (r"no such remote", "error: no such remote [name redacted]"),
    (
        r"unknown revision|bad revision|ambiguous argument",
        "fatal: invalid revision [details redacted]",
    ),
    (
        r"authentication|credential|authorization|could not read username|password|token",
        "error: authentication or credential failure [details redacted]",
    ),
    (
        r"unable to access|could not resolve host|failed to connect|schannel",
        "error: transport access or initialization failure [details redacted]",
    ),
    (r"0xc0000142", "process initialization status 0xc0000142 [details redacted]"),
)


class GitProbeStatus(str, Enum):
    SUCCESS = "success"
    NONZERO_EXIT = "nonzero_exit"
    MISSING_EXECUTABLE = "missing_executable"
    SPAWN_ERROR = "spawn_error"
    TIMEOUT = "timeout"


class GitProbeOperation(str, Enum):
    ORIGIN = "origin"
    BRANCH = "branch"
    DIVERGENCE = "divergence"


def redact_git_stderr(value: str | bytes | None) -> tuple[str, bool]:
    """Project bounded stderr to safe categories, discarding all variable text."""
    if value is None:
        return "", False
    if not isinstance(value, (str, bytes)):
        raise TypeError("Git stderr must be text, bytes, or None")
    truncated = len(value) > _INPUT_LIMIT
    prefix = value[:_INPUT_LIMIT]
    text = prefix.decode("utf-8", errors="replace") if isinstance(prefix, bytes) else prefix
    lines = text.splitlines()
    truncated = truncated or len(lines) > _LINE_LIMIT
    projected: list[str] = []
    for line in lines[:_LINE_LIMIT]:
        if not line.strip():
            continue
        if line in {_REDACTED, _TRUNCATED} or any(line == safe for _, safe in _CATEGORIES):
            projected.append(line)
            continue
        projected.append(
            next(
                (safe for pattern, safe in _CATEGORIES if re.search(pattern, line, re.IGNORECASE)),
                _REDACTED,
            )
        )
    result = "\n".join(projected)
    truncated = truncated or len(result) > _STDERR_LIMIT
    if truncated:
        result = result[: _STDERR_LIMIT - len(_TRUNCATED) - 1] + "\n" + _TRUNCATED
    return result, truncated


@dataclass(frozen=True, slots=True)
class GitProbeDiagnostic:
    """One local probe attempt, with caller identity rather than inferred parentage.

    ``process_terminal`` is unknown for an injected runner's timeout. The default
    ``subprocess.run`` path kills and waits for its owned child before raising a
    timeout. A missing executable has no child to stop.
    """

    operation: GitProbeOperation
    sequence_id: int
    executable_path: str | None
    caller_pid: int
    started_utc: str
    completed_utc: str
    status: GitProbeStatus
    exit_code: int | None
    stderr: str | bytes | None = ""
    process_terminal: bool | None = True
    stderr_truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.operation) is not GitProbeOperation or type(self.status) is not GitProbeStatus:
            raise TypeError("Git probe operation and status must be typed enums")
        for value in (self.sequence_id, self.caller_pid):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Git probe identifiers must be positive integers")
        if self.executable_path is not None and (
            not isinstance(self.executable_path, str)
            or not self.executable_path
            or len(self.executable_path) > 4096
            or any(ord(character) < 32 for character in self.executable_path)
        ):
            raise ValueError("Git executable path must be bounded text or None")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise ValueError("Git exit code must be an integer or None")
        if self.process_terminal is not None and not isinstance(self.process_terminal, bool):
            raise TypeError("Git terminal evidence must be bool or None")
        if not isinstance(self.stderr_truncated, bool):
            raise TypeError("stderr_truncated must be bool")
        for timestamp_text in (self.started_utc, self.completed_utc):
            if not isinstance(timestamp_text, str) or len(timestamp_text) > 40:
                raise ValueError("Git probe timestamps must be bounded UTC text")
            timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
            if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
                raise ValueError("Git probe timestamps must be UTC")
        stderr, truncated = redact_git_stderr(self.stderr)
        object.__setattr__(self, "stderr", stderr)
        object.__setattr__(self, "stderr_truncated", self.stderr_truncated or truncated)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": GIT_PROBE_DIAGNOSTIC_SCHEMA,
            "source": "glr.fork-gate",
            "operation": self.operation.value,
            "sequence_id": self.sequence_id,
            "executable_path": self.executable_path,
            "caller_pid": self.caller_pid,
            "started_utc": self.started_utc,
            "completed_utc": self.completed_utc,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stderr": self.stderr,
            "stderr_truncated": self.stderr_truncated,
            "timed_out": self.status is GitProbeStatus.TIMEOUT,
            "process_terminal": self.process_terminal,
        }
