"""Anti-fork drift gate for derived checkouts of this repository.

A fork or copy of GLR drifts when it silently stops tracking the canonical
upstream: the ``origin`` remote is repointed, the branch falls hundreds of
commits behind, or version and schema metadata are edited locally until the
copy is no longer able to merge back.  This module turns that drift into an
explicit, machine-readable gate so a scheduled job or a local recipe can fail
before the copy becomes unrecoverable.

Every check is expressed as a :class:`ForkGateFinding`.  Blocking findings fail
the gate; advisory findings are reported so a fork owner can see drift while it
is still cheap to fix.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from subprocess import run
from typing import Protocol

FORK_GATE_SCHEMA_VERSION = "glr.fork-gate-report.v1"

#: Scheduler-facing exit codes: 0 = gate passed, 5 = gate blocked.
FORK_GATE_EXIT_OK = 0
FORK_GATE_EXIT_BLOCKED = 5

_RELEASE_MANIFEST = ".release-please-manifest.json"
_GIT_TIMEOUT_SECONDS = 30.0
_SCP_LIKE_REMOTE = re.compile(r"^(?P<user>[^@/]+)@(?P<host>[^:/]+):(?P<path>.+)$")


class _CompletedCommand(Protocol):
    """Minimal completed-process view consumed by :class:`GitRepositoryProbe`."""

    returncode: int
    stdout: str


class _CommandRunner(Protocol):
    """Injectable command runner, replacing ``subprocess.run`` in tests."""

    def __call__(self, command: list[str]) -> _CompletedCommand: ...


class RepositoryProbe(Protocol):
    """Read-only view of one checkout, isolated from git for testability."""

    def origin_url(self) -> str | None: ...

    def current_branch(self) -> str | None: ...

    def divergence(self, default_branch: str) -> tuple[int | None, int | None]: ...

    def package_version(self) -> str | None: ...

    def manifest_version(self) -> str | None: ...

    def schema_versions(self) -> Mapping[str, str]: ...


def normalize_remote_url(value: str) -> str:
    """Normalize a git remote so cosmetic differences do not fail the gate.

    ``git@github.com:org/repo.git``, ``https://github.com/org/repo.git`` and
    ``https://github.com/org/repo/`` all describe the same upstream, so scheme,
    user, trailing slash and ``.git`` suffix are removed before comparison.
    """

    candidate = value.strip()
    if not candidate:
        return ""
    scp_match = _SCP_LIKE_REMOTE.match(candidate)
    if scp_match is not None:
        candidate = f"{scp_match.group('host')}/{scp_match.group('path')}"
    else:
        for prefix in ("https://", "http://", "ssh://", "git://", "git+ssh://"):
            if candidate.lower().startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
    if "@" in candidate.split("/", 1)[0]:
        candidate = candidate.split("@", 1)[1]
    candidate = candidate.rstrip("/")
    if candidate.endswith(".git"):
        candidate = candidate[: -len(".git")]
    return candidate.lower()


def _parse_version(text: str) -> str | None:
    match = re.search(r"^\s*version\s*=\s*\"(?P<version>[^\"]+)\"", text, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group("version")


class GitRepositoryProbe:
    """Repository probe backed by bounded, read-only git and file reads."""

    def __init__(
        self,
        root: Path,
        *,
        schema_versions: Mapping[str, str] | None = None,
        runner: _CommandRunner | None = None,
    ) -> None:
        if not isinstance(root, Path):
            raise ValueError("fork gate root must be a Path")
        self._root = root
        self._schema_versions: Mapping[str, str] = dict(schema_versions or {})
        self._runner = runner

    @property
    def root(self) -> Path:
        return self._root

    def _git(self, *arguments: str) -> str | None:
        command = ["git", "-C", str(self._root), *arguments]
        if self._runner is not None:
            completed = self._runner(command)
        else:
            completed = run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        if completed.returncode != 0:
            return None
        output = completed.stdout.strip()
        return output or None

    def origin_url(self) -> str | None:
        return self._git("remote", "get-url", "origin")

    def current_branch(self) -> str | None:
        return self._git("rev-parse", "--abbrev-ref", "HEAD")

    def divergence(self, default_branch: str) -> tuple[int | None, int | None]:
        output = self._git("rev-list", "--count", "--left-right", f"origin/{default_branch}...HEAD")
        if output is None:
            return (None, None)
        parts = output.split()
        if len(parts) != 2:
            return (None, None)
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return (None, None)

    def package_version(self) -> str | None:
        pyproject = self._root / "pyproject.toml"
        if not pyproject.is_file():
            return None
        return _parse_version(pyproject.read_text(encoding="utf-8"))

    def manifest_version(self) -> str | None:
        import json

        manifest = self._root / _RELEASE_MANIFEST
        if not manifest.is_file():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get(".")
        return value if isinstance(value, str) else None

    def schema_versions(self) -> Mapping[str, str]:
        return dict(self._schema_versions)


@dataclass(frozen=True, slots=True)
class ForkGatePolicy:
    """Expected upstream identity and the drift budget a fork may spend."""

    expected_origin_url: str
    default_branch: str = "main"
    max_commits_behind: int = 50
    max_commits_ahead: int = 200
    require_origin_match: bool = True
    require_version_alignment: bool = True
    require_upstream_ref: bool = True
    required_schema_versions: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not normalize_remote_url(self.expected_origin_url):
            raise ValueError("fork gate expected_origin_url must be a non-empty remote")
        if not self.default_branch or self.default_branch.strip() != self.default_branch:
            raise ValueError("fork gate default_branch must be a non-empty trimmed string")
        for label, value in (
            ("max_commits_behind", self.max_commits_behind),
            ("max_commits_ahead", self.max_commits_ahead),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"fork gate {label} must be a non-negative integer")

    def to_mapping(self) -> dict[str, object]:
        return {
            "expected_origin_url": self.expected_origin_url,
            "default_branch": self.default_branch,
            "max_commits_behind": self.max_commits_behind,
            "max_commits_ahead": self.max_commits_ahead,
            "require_origin_match": self.require_origin_match,
            "require_version_alignment": self.require_version_alignment,
            "require_upstream_ref": self.require_upstream_ref,
            "required_schema_versions": dict(self.required_schema_versions or {}),
        }


@dataclass(frozen=True, slots=True)
class ForkGateFinding:
    """One gate check with its observed and expected values."""

    check: str
    passed: bool
    detail: str
    blocking: bool = True
    observed: str | None = None
    expected: str | None = None

    def __post_init__(self) -> None:
        if not self.check or self.check.strip() != self.check:
            raise ValueError("fork gate finding check must be a non-empty trimmed string")
        if not isinstance(self.detail, str):
            raise ValueError("fork gate finding detail must be a string")

    def to_mapping(self) -> dict[str, object]:
        return {
            "check": self.check,
            "passed": self.passed,
            "blocking": self.blocking,
            "detail": self.detail,
            "observed": self.observed,
            "expected": self.expected,
        }


@dataclass(frozen=True, slots=True)
class ForkGateReport:
    """Complete gate result for one checkout."""

    findings: tuple[ForkGateFinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.findings, tuple):
            raise ValueError("fork gate findings must be a tuple")
        for finding in self.findings:
            if not isinstance(finding, ForkGateFinding):
                raise ValueError("fork gate findings must contain ForkGateFinding values")

    @property
    def blockers(self) -> tuple[ForkGateFinding, ...]:
        return tuple(f for f in self.findings if f.blocking and not f.passed)

    @property
    def advisories(self) -> tuple[ForkGateFinding, ...]:
        return tuple(f for f in self.findings if not f.blocking and not f.passed)

    @property
    def passed(self) -> bool:
        return not self.blockers

    @property
    def exit_code(self) -> int:
        return FORK_GATE_EXIT_OK if self.passed else FORK_GATE_EXIT_BLOCKED

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": FORK_GATE_SCHEMA_VERSION,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "blockers": [f.check for f in self.blockers],
            "advisories": [f.check for f in self.advisories],
            "findings": [f.to_mapping() for f in self.findings],
        }


@dataclass(frozen=True, slots=True)
class StaticRepositoryProbe:
    """Probe over fixed values, used by tests and by callers outside git."""

    origin: str | None = None
    branch: str | None = None
    behind: int | None = None
    ahead: int | None = None
    version: str | None = None
    manifest: str | None = None
    schemas: Mapping[str, str] | None = None

    def origin_url(self) -> str | None:
        return self.origin

    def current_branch(self) -> str | None:
        return self.branch

    def divergence(self, default_branch: str) -> tuple[int | None, int | None]:
        return (self.behind, self.ahead)

    def package_version(self) -> str | None:
        return self.version

    def manifest_version(self) -> str | None:
        return self.manifest

    def schema_versions(self) -> Mapping[str, str]:
        return dict(self.schemas or {})


def _origin_finding(probe: RepositoryProbe, policy: ForkGatePolicy) -> ForkGateFinding:
    observed = probe.origin_url()
    expected = normalize_remote_url(policy.expected_origin_url)
    if observed is None:
        return ForkGateFinding(
            check="origin-url",
            passed=False,
            blocking=policy.require_origin_match,
            detail="no origin remote is configured",
            observed=None,
            expected=expected,
        )
    normalized = normalize_remote_url(observed)
    return ForkGateFinding(
        check="origin-url",
        passed=normalized == expected,
        blocking=policy.require_origin_match,
        detail=(
            "origin matches the canonical upstream"
            if normalized == expected
            else "origin does not match the canonical upstream"
        ),
        observed=normalized,
        expected=expected,
    )


def _divergence_findings(
    probe: RepositoryProbe, policy: ForkGatePolicy
) -> tuple[ForkGateFinding, ...]:
    behind, ahead = probe.divergence(policy.default_branch)
    if behind is None or ahead is None:
        return (
            ForkGateFinding(
                check="upstream-divergence",
                passed=False,
                blocking=policy.require_upstream_ref,
                detail=(
                    f"upstream ref origin/{policy.default_branch} is unavailable; "
                    "run git fetch to measure divergence"
                ),
                observed=None,
                expected=f"origin/{policy.default_branch}",
            ),
        )
    return (
        ForkGateFinding(
            check="commits-behind",
            passed=behind <= policy.max_commits_behind,
            blocking=True,
            detail=f"{behind} commits behind origin/{policy.default_branch}",
            observed=str(behind),
            expected=f"<= {policy.max_commits_behind}",
        ),
        ForkGateFinding(
            check="commits-ahead",
            passed=ahead <= policy.max_commits_ahead,
            blocking=False,
            detail=f"{ahead} commits ahead of origin/{policy.default_branch}",
            observed=str(ahead),
            expected=f"<= {policy.max_commits_ahead}",
        ),
    )


def _version_finding(probe: RepositoryProbe, policy: ForkGatePolicy) -> ForkGateFinding:
    observed = probe.package_version()
    expected = probe.manifest_version()
    if observed is None or expected is None:
        return ForkGateFinding(
            check="version-alignment",
            passed=False,
            blocking=policy.require_version_alignment,
            detail="package version or release manifest version is unavailable",
            observed=observed,
            expected=expected,
        )
    return ForkGateFinding(
        check="version-alignment",
        passed=observed == expected,
        blocking=policy.require_version_alignment,
        detail=(
            "package version matches the release manifest"
            if observed == expected
            else "package version does not match the release manifest"
        ),
        observed=observed,
        expected=expected,
    )


def _schema_findings(probe: RepositoryProbe, policy: ForkGatePolicy) -> tuple[ForkGateFinding, ...]:
    required = dict(policy.required_schema_versions or {})
    if not required:
        return ()
    observed = probe.schema_versions()
    findings: list[ForkGateFinding] = []
    for label in sorted(required):
        expected_value = required[label]
        actual_value = observed.get(label)
        findings.append(
            ForkGateFinding(
                check=f"schema-version:{label}",
                passed=actual_value == expected_value,
                blocking=True,
                detail=(
                    f"schema {label} is aligned"
                    if actual_value == expected_value
                    else f"schema {label} drifted from the canonical version"
                ),
                observed=actual_value,
                expected=expected_value,
            )
        )
    return tuple(findings)


def evaluate_fork_gate(probe: RepositoryProbe, policy: ForkGatePolicy) -> ForkGateReport:
    """Evaluate every anti-fork check for one checkout."""

    if not isinstance(policy, ForkGatePolicy):
        raise ValueError("fork gate policy must be a ForkGatePolicy")
    findings: list[ForkGateFinding] = [_origin_finding(probe, policy)]
    findings.extend(_divergence_findings(probe, policy))
    findings.append(_version_finding(probe, policy))
    findings.extend(_schema_findings(probe, policy))
    return ForkGateReport(tuple(findings))


__all__ = [
    "FORK_GATE_EXIT_BLOCKED",
    "FORK_GATE_EXIT_OK",
    "FORK_GATE_SCHEMA_VERSION",
    "ForkGateFinding",
    "ForkGatePolicy",
    "ForkGateReport",
    "GitRepositoryProbe",
    "RepositoryProbe",
    "StaticRepositoryProbe",
    "evaluate_fork_gate",
    "normalize_remote_url",
]
