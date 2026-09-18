from __future__ import annotations

from pathlib import Path

import pytest

from game_learning_runtime import (
    FORK_GATE_EXIT_BLOCKED,
    FORK_GATE_EXIT_OK,
    ForkGateFinding,
    ForkGatePolicy,
    ForkGateReport,
    GitRepositoryProbe,
    StaticRepositoryProbe,
    evaluate_fork_gate,
    normalize_remote_url,
)

CANONICAL = "https://github.com/loonghao/GameLearningRuntime.git"


def _policy(**overrides: object) -> ForkGatePolicy:
    defaults: dict[str, object] = {
        "expected_origin_url": CANONICAL,
        "default_branch": "main",
        "max_commits_behind": 50,
        "max_commits_ahead": 200,
    }
    defaults.update(overrides)
    return ForkGatePolicy(**defaults)  # type: ignore[arg-type]


def _gate(**probe_overrides: object) -> ForkGateReport:
    defaults: dict[str, object] = {
        "origin": CANONICAL,
        "branch": "agent/example",
        "behind": 0,
        "ahead": 2,
        "version": "0.13.2",
        "manifest": "0.13.2",
    }
    defaults.update(probe_overrides)
    return evaluate_fork_gate(StaticRepositoryProbe(**defaults), _policy())  # type: ignore[arg-type]


def _finding(report: ForkGateReport, check: str) -> ForkGateFinding:
    for finding in report.findings:
        if finding.check == check:
            return finding
    raise AssertionError(f"missing finding {check!r} in {[f.check for f in report.findings]}")


def test_normalize_remote_url_ignores_cosmetic_differences() -> None:
    expected = "github.com/loonghao/gamelearningruntime"
    assert normalize_remote_url(CANONICAL) == expected
    assert normalize_remote_url("git@github.com:loonghao/GameLearningRuntime.git") == expected
    assert normalize_remote_url("https://github.com/loonghao/GameLearningRuntime/") == expected
    assert normalize_remote_url("ssh://git@github.com/loonghao/GameLearningRuntime") == expected
    assert normalize_remote_url("   ") == ""


def test_policy_rejects_blank_origin_and_negative_budgets() -> None:
    with pytest.raises(ValueError, match="expected_origin_url"):
        ForkGatePolicy(expected_origin_url="  ")
    with pytest.raises(ValueError, match="default_branch"):
        ForkGatePolicy(expected_origin_url=CANONICAL, default_branch=" ")
    with pytest.raises(ValueError, match="max_commits_behind"):
        ForkGatePolicy(expected_origin_url=CANONICAL, max_commits_behind=-1)
    with pytest.raises(ValueError, match="max_commits_ahead"):
        ForkGatePolicy(expected_origin_url=CANONICAL, max_commits_ahead=-1)


def test_aligned_checkout_passes_the_gate() -> None:
    report = _gate()
    assert report.passed is True
    assert report.blockers == ()
    assert report.exit_code == FORK_GATE_EXIT_OK


def test_foreign_origin_blocks_the_gate() -> None:
    report = _gate(origin="https://github.com/someone/else.git")
    assert report.passed is False
    assert report.exit_code == FORK_GATE_EXIT_BLOCKED
    assert "origin-url" in [f.check for f in report.blockers]
    assert _finding(report, "origin-url").expected == "github.com/loonghao/gamelearningruntime"


def test_foreign_origin_can_be_downgraded_to_an_advisory() -> None:
    report = evaluate_fork_gate(
        StaticRepositoryProbe(
            origin="https://github.com/someone/else.git",
            branch="x",
            behind=0,
            ahead=0,
            version="1.0.0",
            manifest="1.0.0",
        ),
        _policy(require_origin_match=False),
    )
    assert report.passed is True
    assert "origin-url" in [f.check for f in report.advisories]


def test_missing_origin_blocks_when_required() -> None:
    report = _gate(origin=None)
    assert _finding(report, "origin-url").detail == "no origin remote is configured"
    assert report.passed is False


def test_commits_behind_over_budget_blocks() -> None:
    assert _gate(behind=51).passed is False
    assert _gate(behind=50).passed is True
    assert _finding(_gate(behind=999), "commits-behind").observed == "999"


def test_commits_ahead_over_budget_is_only_an_advisory() -> None:
    report = _gate(ahead=201)
    assert report.passed is True
    assert [f.check for f in report.advisories] == ["commits-ahead"]


def test_unfetched_upstream_ref_is_blocking_by_default() -> None:
    report = _gate(behind=None, ahead=None)
    assert report.passed is False
    assert "upstream-divergence" in [f.check for f in report.blockers]


def test_unfetched_upstream_ref_can_be_allowed() -> None:
    report = evaluate_fork_gate(
        StaticRepositoryProbe(origin=CANONICAL, branch="x", version="1", manifest="1"),
        _policy(require_upstream_ref=False),
    )
    assert report.passed is True


def test_version_drift_blocks_the_gate() -> None:
    report = _gate(version="0.13.2", manifest="0.14.0")
    assert report.passed is False
    assert _finding(report, "version-alignment").observed == "0.13.2"
    assert _finding(report, "version-alignment").expected == "0.14.0"


def test_missing_version_data_blocks_when_required() -> None:
    assert _gate(version=None).passed is False
    assert _gate(manifest=None).passed is False


def test_version_drift_can_be_downgraded() -> None:
    report = evaluate_fork_gate(
        StaticRepositoryProbe(
            origin=CANONICAL, branch="x", behind=0, ahead=0, version="1", manifest="2"
        ),
        _policy(require_version_alignment=False),
    )
    assert report.passed is True
    assert "version-alignment" in [f.check for f in report.advisories]


def test_schema_version_drift_blocks_and_skips_when_unrequired() -> None:
    report = _gate(schemas={"protocol": "v1"})
    assert report.passed is True
    assert report.findings == tuple(f for f in report.findings if not f.check.startswith("schema-"))

    required = _policy(required_schema_versions={"protocol": "v2"})
    drifted = evaluate_fork_gate(
        StaticRepositoryProbe(
            origin=CANONICAL,
            branch="x",
            behind=0,
            ahead=0,
            version="1",
            manifest="1",
            schemas={"protocol": "v1"},
        ),
        required,
    )
    assert drifted.passed is False
    assert _finding(drifted, "schema-version:protocol").expected == "v2"

    aligned = evaluate_fork_gate(
        StaticRepositoryProbe(
            origin=CANONICAL,
            branch="x",
            behind=0,
            ahead=0,
            version="1",
            manifest="1",
            schemas={"protocol": "v2"},
        ),
        required,
    )
    assert aligned.passed is True


def test_report_rejects_non_findings() -> None:
    with pytest.raises(ValueError, match="ForkGateFinding"):
        ForkGateReport(("nope",))  # type: ignore[arg-type]


def test_finding_rejects_blank_check_name() -> None:
    with pytest.raises(ValueError, match="must be a non-empty trimmed string"):
        ForkGateFinding(check="  ", passed=True, detail="x")


def test_evaluate_rejects_non_policy() -> None:
    with pytest.raises(ValueError, match="ForkGatePolicy"):
        evaluate_fork_gate(StaticRepositoryProbe(), "not-a-policy")  # type: ignore[arg-type]


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def _git_probe(tmp_path: Path, responses: dict[tuple[str, ...], str | None]) -> GitRepositoryProbe:
    def runner(command: list[str]) -> _FakeCompleted:
        key = tuple(command[3:])
        value = responses.get(key)
        if value is None:
            return _FakeCompleted(1, "")
        return _FakeCompleted(0, value)

    return GitRepositoryProbe(tmp_path, runner=runner)


def test_git_probe_reads_origin_branch_and_divergence(tmp_path: Path) -> None:
    probe = _git_probe(
        tmp_path,
        {
            ("remote", "get-url", "origin"): "git@github.com:loonghao/GameLearningRuntime.git\n",
            ("rev-parse", "--abbrev-ref", "HEAD"): "agent/example\n",
            ("rev-list", "--count", "--left-right", "origin/main...HEAD"): "3\t7\n",
        },
    )
    assert probe.origin_url() == "git@github.com:loonghao/GameLearningRuntime.git"
    assert probe.current_branch() == "agent/example"
    assert probe.divergence("main") == (3, 7)
    assert probe.root == tmp_path


def test_git_probe_degrades_when_git_fails(tmp_path: Path) -> None:
    probe = _git_probe(tmp_path, {})
    assert probe.origin_url() is None
    assert probe.current_branch() is None
    assert probe.divergence("main") == (None, None)


def test_git_probe_parses_unparsable_divergence(tmp_path: Path) -> None:
    probe = _git_probe(
        tmp_path, {("rev-list", "--count", "--left-right", "origin/main...HEAD"): "garbage"}
    )
    assert probe.divergence("main") == (None, None)


def test_git_probe_reads_versions_and_schemas(tmp_path: Path) -> None:
    pyproject = "[project]" + chr(10) + 'name = "x"' + chr(10) + 'version = "1.2.3"' + chr(10)
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / ".release-please-manifest.json").write_text('{".": "1.2.3"}', encoding="utf-8")
    probe = GitRepositoryProbe(tmp_path, schema_versions={"protocol": "v1"})
    assert probe.package_version() == "1.2.3"
    assert probe.manifest_version() == "1.2.3"
    assert probe.schema_versions() == {"protocol": "v1"}


def test_git_probe_degrades_without_version_files(tmp_path: Path) -> None:
    probe = GitRepositoryProbe(tmp_path)
    assert probe.package_version() is None
    assert probe.manifest_version() is None
    assert probe.schema_versions() == {}


def test_git_probe_rejects_non_path_root() -> None:
    with pytest.raises(ValueError, match="must be a Path"):
        GitRepositoryProbe(".")  # type: ignore[arg-type]


def test_git_probe_tolerates_corrupt_manifest(tmp_path: Path) -> None:
    (tmp_path / ".release-please-manifest.json").write_text("not json", encoding="utf-8")
    assert GitRepositoryProbe(tmp_path).manifest_version() is None
    (tmp_path / ".release-please-manifest.json").write_text("[1,2]", encoding="utf-8")
    assert GitRepositoryProbe(tmp_path).manifest_version() is None
    (tmp_path / ".release-please-manifest.json").write_text('{".": 5}', encoding="utf-8")
    assert GitRepositoryProbe(tmp_path).manifest_version() is None
