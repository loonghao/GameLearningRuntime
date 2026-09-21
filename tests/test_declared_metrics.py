"""Declared metrics that are never emitted must fail closed, not read as zero."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from game_learning_runtime.cli import main
from game_learning_runtime.collector import SyncCollector
from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.declared_metrics import (
    DECLARED_METRICS_EVENT,
    DECLARED_METRICS_METRIC,
    DECLARED_METRICS_SCHEMA_VERSION,
    EMITTED_METRICS_METRIC,
    MISSING_METRICS_COUNT_METRIC,
    STRICT_METRICS_CAPABILITY,
    STRICT_METRICS_CAPABILITY_ALIAS,
    DeclaredMetricAudit,
    DeclaredMetricLedger,
    MetricDeclaration,
    MissingDeclaredMetric,
    bind_declared_metrics,
    bound_declared_metrics,
    build_declared_metrics,
    declared_metrics_for,
    declared_metrics_from_spec,
    release_declared_metrics,
    summarize_declared_metrics,
)
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.run_store import RunStatus, TrainingStore
from game_learning_runtime.specs import CompositeSpec, EnvironmentSpec, SpaceKind, TensorSpec
from game_learning_runtime.telemetry import Telemetry
from game_learning_runtime.termination import EpisodeCaps, TerminationReason

EXPECTED = ("inherited_rows", "episode_reward", "steps_per_second")


class _DeclaringCounter(GameEnvironment):
    """Counter that ends an episode after ``target`` increments.

    The adapter declares its metrics on the spec; whether they are actually
    emitted is decided by the caller, which is exactly the failure mode this
    module exists to catch.
    """

    def __init__(
        self,
        *,
        emitted: Sequence[str] = (),
        declaration: MetricDeclaration | None = None,
        capabilities: frozenset[str] = frozenset(),
        target: int = 2,
    ) -> None:
        self._emitted = tuple(emitted)
        self._target = target
        self._position = 0
        self._step_id = 0
        self._episode_id = uuid4()
        self._spec = EnvironmentSpec(
            environment_id="example.declared-counter-v1",
            observation=CompositeSpec(
                {"position": TensorSpec((1,), np.int64, minimum=0, maximum=target)}
            ),
            action=CompositeSpec(
                {
                    "choice": TensorSpec(
                        (1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=1
                    )
                }
            ),
            capabilities=capabilities,
            metrics=declaration,
            # The step budget is the safety net; the goal is the real reason.
            # Report it in ``info`` rather than letting the cap stand in for it.
            episode_caps=EpisodeCaps(max_steps=target),
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        del seed, options
        self._position = 0
        self._step_id = 0
        self._episode_id = uuid4()
        return self._timestep()

    def step(self, action: TensorTree) -> TimeStep:
        self._position = min(self._position + int(action["choice"][0]), self._target)
        self._step_id += 1
        return self._timestep()

    def _timestep(self) -> TimeStep:
        reached = self._position == self._target
        return TimeStep(
            observation={"position": np.array([self._position], dtype=np.int64)},
            reward=np.array([1.0 if reached else 0.0], dtype=np.float32),
            terminated=np.array([reached], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
            episode_id=self._episode_id,
            step_id=self._step_id,
            # Reaching the target is the goal, so say so rather than letting a
            # cap that happens to equal the distance stand in for it.
            info=({"termination_reason": TerminationReason.GOAL_REACHED.value} if reached else {}),
        )


def _increment(timestep: TimeStep) -> TensorTree:
    del timestep
    return {"choice": np.array([1], dtype=np.int64)}


def _run(tmp_path: Path) -> tuple[TrainingStore, Any]:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    return store, store.create_run(
        environment_id="example.declared-counter-v1",
        protocol_version="1.0",
        kind="training",
    )


def _emit(telemetry: Telemetry, names: Sequence[str], *, step_id: int = 0) -> None:
    """Emit metrics the way an adapter does: through the Telemetry surface."""

    for index, name in enumerate(names):
        telemetry.metric(name, float(index + 1), step_id=step_id + index)


def test_declared_metric_never_emitted_is_named_in_the_manifest(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=EXPECTED)
    telemetry = Telemetry(store, run.run_id, console=False)
    ledger = build_declared_metrics(declaration, store=store, run_id=run.run_id)

    with bound_declared_metrics(store, run.run_id, ledger):
        _emit(telemetry, ("inherited_rows", "episode_reward"))
        SyncCollector(_DeclaringCounter(declaration=declaration), declared_metrics=ledger).collect(
            _increment, steps=2, stop_on_done=True
        )

    audits = store.list_declared_metric_audits(run.run_id)
    assert len(audits) == 1
    audit = audits[0]
    assert audit.declared_metrics == 3
    assert audit.emitted_metrics == 2
    assert audit.missing_metrics == ("steps_per_second",)
    assert audit.complete is False

    counters = {metric.name: metric.value for metric in store.list_metrics(run.run_id)}
    assert counters[DECLARED_METRICS_METRIC] == 3.0
    assert counters[EMITTED_METRICS_METRIC] == 2.0
    assert counters[MISSING_METRICS_COUNT_METRIC] == 1.0


def test_collector_binds_a_spec_declaration_to_the_run_store(tmp_path: Path) -> None:
    """The documented path: hand the collector ``spec.metrics`` and the run.

    A declaration alone is inert -- emissions are counted by the run store, so
    without a binding the counters stay at zero and the run reads as if it had
    never been measured. This is the wiring the guide promises.
    """

    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=EXPECTED)
    telemetry = Telemetry(store, run.run_id, console=False)
    environment = _DeclaringCounter(declaration=declaration)

    collector = SyncCollector(
        environment,
        declared_metrics=environment.spec.metrics,
        store=store,
        run_id=run.run_id,
    )
    ledger = collector.declared_metrics
    assert ledger is not None
    # Bound by the collector, so the store counts emissions straight into it.
    assert declared_metrics_for(store, run.run_id) is ledger

    _emit(telemetry, ("inherited_rows", "episode_reward"))
    collector.collect(_increment, steps=2, stop_on_done=True)

    audits = store.list_declared_metric_audits(run.run_id)
    assert len(audits) == 1
    assert audits[0].missing_metrics == ("steps_per_second",)
    counters = {metric.name: metric.value for metric in store.list_metrics(run.run_id)}
    assert counters[DECLARED_METRICS_METRIC] == 3.0
    assert counters[EMITTED_METRICS_METRIC] == 2.0
    assert counters[MISSING_METRICS_COUNT_METRIC] == 1.0
    assert collector.declared_metric_audits() == audits

    collector.release_declared_metrics()
    assert declared_metrics_for(store, run.run_id) is None


def test_collector_store_binding_is_validated(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    ledger = build_declared_metrics(MetricDeclaration(expected=EXPECTED))
    assert ledger is not None
    for kwargs in ({"store": store}, {"run_id": run.run_id}):
        with pytest.raises(ValueError, match="store and run_id"):
            SyncCollector(_DeclaringCounter(), **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be combined with a DeclaredMetricLedger"):
        SyncCollector(_DeclaringCounter(), declared_metrics=ledger, store=store, run_id=run.run_id)


def test_aborted_episode_is_still_audited(tmp_path: Path) -> None:
    """An episode ended by an environment error must not look undeclared.

    The audit lands, but strict mode stays quiet: the environment failure is
    the error worth propagating, and a missing-metric error raised on the way
    out would hide it.
    """

    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=EXPECTED, strict=True)

    class _FailingCounter(_DeclaringCounter):
        """Answers ``fail_after`` steps, then stops answering entirely."""

        def __init__(self, *, fail_after: int = 0, **kwargs: object) -> None:
            super().__init__(**kwargs)  # type: ignore[arg-type]
            self._fail_after = fail_after
            self._taken = 0

        def step(self, action: TensorTree) -> TimeStep:
            if self._taken >= self._fail_after:
                raise RuntimeError("the game stopped answering")
            self._taken += 1
            return super().step(action)

    first = _FailingCounter(declaration=declaration, fail_after=0)
    collector = SyncCollector(
        first,
        declared_metrics=first.spec.metrics,
        store=store,
        run_id=run.run_id,
    )
    with pytest.raises(RuntimeError, match="stopped answering"):
        collector.collect(_increment, steps=2)

    audits = store.list_declared_metric_audits(run.run_id)
    assert len(audits) == 1
    assert audits[0].emitted_metrics == 0
    assert audits[0].missing_metrics == tuple(sorted(EXPECTED))
    collector.release_declared_metrics()

    # ``on_error='partial'`` returns the truncated unroll instead of raising,
    # and the truncated episode keeps its own audit.
    second = _FailingCounter(declaration=declaration, fail_after=1)
    partial = SyncCollector(
        second,
        declared_metrics=second.spec.metrics,
        store=store,
        run_id=run.run_id,
    )
    unroll = partial.collect(_increment, steps=2, on_error="partial")
    assert len(unroll.transitions) == 1
    assert bool(unroll.transitions[0].truncated[0]) is True
    audits = store.list_declared_metric_audits(run.run_id)
    assert len(audits) == 2
    assert audits[1].emitted_metrics == 0


def test_audits_survive_a_run_with_more_events_than_the_window(tmp_path: Path) -> None:
    """The newest audits survive a run whose event stream outgrows ``limit``.

    Taking the first N events and filtering by kind would keep the oldest
    episodes and drop the newest, which is the under-report this accounting
    exists to prevent.
    """

    store, run = _run(tmp_path)
    ledger = build_declared_metrics(
        MetricDeclaration(expected=("inherited_rows",)), store=store, run_id=run.run_id
    )
    assert ledger is not None
    for _ in range(1200):
        store.append_event(run.run_id, kind="heartbeat", payload={})

    newest = ledger.close_episode("newest-episode")

    audits = store.list_declared_metric_audits(run.run_id)
    assert len(audits) == 1
    assert audits[0].episode_id == newest.episode_id
    assert summarize_declared_metrics(audits)["episode_id"] == "newest-episode"


def test_every_declared_metric_emitted_leaves_no_gap(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=EXPECTED)
    telemetry = Telemetry(store, run.run_id, console=False)
    ledger = build_declared_metrics(declaration, store=store, run_id=run.run_id)

    with bound_declared_metrics(store, run.run_id, ledger):
        _emit(telemetry, EXPECTED)
        collector = SyncCollector(
            _DeclaringCounter(declaration=declaration), declared_metrics=ledger
        )
        collector.collect(_increment, steps=2, stop_on_done=True)

    audit = store.list_declared_metric_audits(run.run_id)[0]
    assert audit.missing_metrics == ()
    assert audit.complete is True
    assert audit.emitted_metrics == 3
    counters = {metric.name: metric.value for metric in store.list_metrics(run.run_id)}
    assert counters[MISSING_METRICS_COUNT_METRIC] == 0.0


def test_strict_capability_fails_the_run_and_names_the_metric(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=EXPECTED)
    telemetry = Telemetry(store, run.run_id, console=False)
    ledger = build_declared_metrics(
        declaration,
        capabilities=frozenset({STRICT_METRICS_CAPABILITY}),
        store=store,
        run_id=run.run_id,
    )
    assert ledger is not None and ledger.strict is True

    with (
        pytest.raises(MissingDeclaredMetric) as raised,
        bound_declared_metrics(store, run.run_id, ledger),
    ):
        _emit(telemetry, ("inherited_rows", "episode_reward"))
        SyncCollector(_DeclaringCounter(declaration=declaration), declared_metrics=ledger).collect(
            _increment, steps=2, stop_on_done=True
        )

    error = raised.value
    assert error.missing_metrics == ("steps_per_second",)
    assert "steps_per_second" in str(error)
    assert error.declared_metrics == 3
    assert error.emitted_metrics == 2

    # The audit is recorded before the error escapes, so a failed run still
    # leaves the counters on the record.
    audits = store.list_declared_metric_audits(run.run_id)
    assert len(audits) == 1
    assert audits[0].missing_metrics == ("steps_per_second",)


def test_strict_capability_alias_is_accepted(tmp_path: Path) -> None:
    declaration = MetricDeclaration(expected=EXPECTED)
    ledger = build_declared_metrics(
        declaration, capabilities=frozenset({STRICT_METRICS_CAPABILITY_ALIAS})
    )
    assert ledger is not None and ledger.strict is True


def test_declaration_strict_flag_enables_the_error_without_a_capability() -> None:
    ledger = build_declared_metrics(MetricDeclaration(expected=EXPECTED, strict=True))
    assert ledger is not None and ledger.strict is True
    audit = ledger.close_episode("episode-1")
    with pytest.raises(MissingDeclaredMetric):
        audit.require()


def test_non_strict_run_completes_and_still_reports_the_gap(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=EXPECTED)
    telemetry = Telemetry(store, run.run_id, console=False)
    ledger = build_declared_metrics(declaration, store=store, run_id=run.run_id)

    with bound_declared_metrics(store, run.run_id, ledger):
        _emit(telemetry, ("inherited_rows",))
        SyncCollector(_DeclaringCounter(declaration=declaration), declared_metrics=ledger).collect(
            _increment, steps=2, stop_on_done=True
        )

    # Nothing raised, the run can still be finished, and the gap is on record.
    assert store.get_run(run.run_id).status is RunStatus.RUNNING
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
    summary = summarize_declared_metrics(store.list_declared_metric_audits(run.run_id))
    assert summary["missing_metrics"] == ["episode_reward", "steps_per_second"]


def test_audit_is_written_before_the_run_is_reported_complete(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=EXPECTED)
    telemetry = Telemetry(store, run.run_id, console=False)
    ledger = build_declared_metrics(declaration, store=store, run_id=run.run_id)

    with bound_declared_metrics(store, run.run_id, ledger):
        _emit(telemetry, EXPECTED)
        SyncCollector(_DeclaringCounter(declaration=declaration), declared_metrics=ledger).collect(
            _increment, steps=2, stop_on_done=True
        )

    # Still running: the check happened at episode close, not at report time.
    assert store.get_run(run.run_id).status is RunStatus.RUNNING
    assert [event.kind for event in store.list_events(run.run_id)].count(
        DECLARED_METRICS_EVENT
    ) == 1
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
    assert store.get_run(run.run_id).status is RunStatus.SUCCEEDED


def test_adapter_that_declares_nothing_behaves_exactly_as_before(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    telemetry = Telemetry(store, run.run_id, console=False)
    assert build_declared_metrics(None) is None
    assert declared_metrics_from_spec(_DeclaringCounter().spec) is None

    collector = SyncCollector(_DeclaringCounter())
    collector.collect(_increment, steps=2, stop_on_done=True)
    telemetry.metric("undeclared.metric", 1.0)

    assert collector.declared_metrics is None
    assert collector.declared_metric_audits() == ()
    assert store.list_declared_metric_audits(run.run_id) == ()
    assert [
        event.kind
        for event in store.list_events(run.run_id)
        if event.kind == DECLARED_METRICS_EVENT
    ] == []
    assert [metric.name for metric in store.list_metrics(run.run_id)] == ["undeclared.metric"]


def test_optional_metrics_are_reported_but_never_missing(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=("inherited_rows",), optional=("preview_only",))
    telemetry = Telemetry(store, run.run_id, console=False)
    ledger = build_declared_metrics(declaration, store=store, run_id=run.run_id)

    with bound_declared_metrics(store, run.run_id, ledger):
        _emit(telemetry, ("inherited_rows", "preview_only"))
        SyncCollector(_DeclaringCounter(declaration=declaration), declared_metrics=ledger).collect(
            _increment, steps=2, stop_on_done=True
        )

    audit = store.list_declared_metric_audits(run.run_id)[0]
    assert audit.missing_metrics == ()
    assert audit.optional_metrics == 1
    assert audit.emitted_optional_metrics == 1
    assert audit.complete is True


def test_each_episode_is_audited_separately(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    declaration = MetricDeclaration(expected=("inherited_rows",))
    telemetry = Telemetry(store, run.run_id, console=False)
    ledger = build_declared_metrics(declaration, store=store, run_id=run.run_id)

    with bound_declared_metrics(store, run.run_id, ledger):
        _emit(telemetry, ("inherited_rows",))
        collector = SyncCollector(
            _DeclaringCounter(declaration=declaration), declared_metrics=ledger
        )
        # Two episodes: the second one emits nothing, so it must report a gap.
        collector.collect(_increment, steps=2, stop_on_done=True)
        collector.collect(_increment, steps=2, stop_on_done=True)

    audits = store.list_declared_metric_audits(run.run_id)
    assert [audit.emitted_metrics for audit in audits] == [1, 0]
    assert [audit.missing_metrics for audit in audits] == [(), ("inherited_rows",)]
    assert audits[0].episode_id != audits[1].episode_id


def test_cli_json_exposes_all_three_counters(tmp_path: Path, capsys: object) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "bridge").mkdir()
    (root / "glr-project.json").write_text(
        json.dumps(
            {
                "schema_version": "glr.project.v1",
                "environment_id": "example.declared-counter-v1",
                "environment_family": "action-rpg",
                "protocol_version": "1.0",
                "data_dir": ".glr",
                "bridge_path": "bridge",
                "runtime": {"argv": ["python", "-c", "print('runtime')"]},
                "trainer": {"argv": ["python", "-c", "print('train')"]},
                "player": {"argv": ["python", "-c", "print('play')", "{bundle}"]},
            }
        ),
        encoding="utf-8",
    )
    store = TrainingStore(root / ".glr" / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.declared-counter-v1",
        protocol_version="1.0",
        kind="training",
    )
    declaration = MetricDeclaration(expected=EXPECTED)
    telemetry = Telemetry(store, run.run_id, console=False)
    ledger = build_declared_metrics(declaration, store=store, run_id=run.run_id)
    with bound_declared_metrics(store, run.run_id, ledger):
        _emit(telemetry, ("inherited_rows", "episode_reward"))
        SyncCollector(_DeclaringCounter(declaration=declaration), declared_metrics=ledger).collect(
            _increment, steps=2, stop_on_done=True
        )

    assert main(["--project", str(root), "--json", "runs", "show", run.run_id]) == 0
    data = json.loads(capsys.readouterr().out)["data"]  # type: ignore[attr-defined]
    summary = data["declared_metrics_summary"]
    assert summary["schema_version"] == DECLARED_METRICS_SCHEMA_VERSION
    assert summary["declared_metrics"] == 3
    assert summary["emitted_metrics"] == 2
    assert summary["missing_metrics"] == ["steps_per_second"]
    assert summary["missing_metrics_count"] == 1
    assert data["declared_metrics"][0]["missing_metrics"] == ["steps_per_second"]
    counters = {item["name"]: item["value"] for item in data["metrics"]}
    assert counters[DECLARED_METRICS_METRIC] == 3.0
    assert counters[EMITTED_METRICS_METRIC] == 2.0
    assert counters[MISSING_METRICS_COUNT_METRIC] == 1.0


def test_summary_reports_absent_accounting_never_as_a_pass() -> None:
    summary = summarize_declared_metrics(())
    assert summary["reported"] is False
    assert summary["declared_metrics"] is None
    assert summary["missing_metrics"] == []
    assert summary["complete"] is None


def test_declaration_rejects_unusable_names_and_overlap() -> None:
    with pytest.raises(ValueError, match="must match"):
        MetricDeclaration(expected=("Inherited-Rows",))
    with pytest.raises(TypeError):
        MetricDeclaration(expected=(1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be both expected and optional"):
        MetricDeclaration(expected=("inherited_rows",), optional=("inherited_rows",))
    with pytest.raises(TypeError):
        MetricDeclaration(expected=(), strict="yes")  # type: ignore[arg-type]


def test_declaration_normalizes_and_round_trips() -> None:
    declaration = MetricDeclaration(
        expected=("steps_per_second", "inherited_rows"), optional=("preview_only",)
    )
    assert declaration.expected == ("inherited_rows", "steps_per_second")
    assert declaration.declared == ("inherited_rows", "preview_only", "steps_per_second")
    restored = MetricDeclaration.from_mapping(declaration.to_mapping())
    assert restored == declaration
    with pytest.raises(ValueError, match=re.escape("glr.declared-metrics.v1")):
        MetricDeclaration.from_mapping({"schema_version": "glr.other.v1"})


def test_spec_rejects_a_non_declaration() -> None:
    with pytest.raises(TypeError, match="metrics must be a MetricDeclaration or None"):
        EnvironmentSpec(
            environment_id="example.bad-v1",
            observation=CompositeSpec({"p": TensorSpec((1,), np.int64)}),
            action=CompositeSpec({"c": TensorSpec((1,), np.int64)}),
            metrics={"expected": ["inherited_rows"]},  # type: ignore[arg-type]
        )


def test_audit_round_trips_and_requires_its_schema_version() -> None:
    audit = DeclaredMetricAudit(
        episode_id="episode-1",
        expected=("inherited_rows", "episode_reward"),
        emitted=("inherited_rows",),
        optional=("preview_only",),
        emitted_optional=("preview_only",),
        strict=True,
        timestamp_ns=7,
    )
    restored = DeclaredMetricAudit.from_mapping(audit.to_mapping())
    assert restored == audit
    assert restored.missing_metrics == ("episode_reward",)
    with pytest.raises(ValueError, match=re.escape("glr.declared-metrics.v1")):
        DeclaredMetricAudit.from_mapping({"schema_version": "glr.other.v1"})
    with pytest.raises(TypeError):
        DeclaredMetricAudit.from_mapping(["not-a-mapping"])  # type: ignore[arg-type]


def test_ledger_ignores_undeclared_names_and_rejects_bad_input() -> None:
    ledger = DeclaredMetricLedger(MetricDeclaration(expected=("inherited_rows",)))
    assert ledger.record("inherited_rows") is True
    assert ledger.record("undeclared.metric") is False
    assert ledger.emitted == ("inherited_rows",)
    with pytest.raises(TypeError, match="declaration must be a MetricDeclaration"):
        DeclaredMetricLedger({"expected": []})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="store and run_id"):
        DeclaredMetricLedger(MetricDeclaration(expected=("a",)), run_id="run-1")
    with pytest.raises(ValueError, match="episode_id"):
        ledger.close_episode("")
    with pytest.raises(TypeError):
        ledger.record(1)  # type: ignore[arg-type]


def test_registry_binds_releases_and_rejects_double_binding(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(environment_id="example.x", protocol_version="1.0", kind="training")
    ledger = DeclaredMetricLedger(MetricDeclaration(expected=("inherited_rows",)))
    bind_declared_metrics(store, run.run_id, ledger)
    assert declared_metrics_for(store, run.run_id) is ledger
    # A different store object on the same path resolves to the same binding.
    assert declared_metrics_for(str(store.path), run.run_id) is ledger
    with pytest.raises(ValueError, match="a different declared-metric ledger"):
        bind_declared_metrics(store, run.run_id, DeclaredMetricLedger(MetricDeclaration()))
    release_declared_metrics(store, run.run_id)
    assert declared_metrics_for(store, run.run_id) is None
    release_declared_metrics(store, run.run_id)
    with pytest.raises(TypeError):
        bind_declared_metrics(store, run.run_id, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="run_id cannot be empty"):
        bind_declared_metrics(store, "", ledger)


def test_missing_declared_metric_refuses_an_empty_gap() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        MissingDeclaredMetric(missing_metrics=(), declared_metrics=0, emitted_metrics=0)


def test_declaration_rejects_a_string_and_unusable_entries() -> None:
    with pytest.raises(TypeError, match="must be a sequence"):
        MetricDeclaration(expected="inherited_rows")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="entries must be strings"):
        MetricDeclaration(expected=(None,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1-128 characters"):
        MetricDeclaration(expected=("inherited_rows" * 10,))
    with pytest.raises(ValueError, match="cannot declare more than 64"):
        MetricDeclaration(expected=(f"metric{index}" for index in range(65)))


def test_declaration_mapping_rejects_unusable_values() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        MetricDeclaration.from_mapping(["not-a-mapping"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a list of metric names"):
        MetricDeclaration.from_mapping(
            {"schema_version": DECLARED_METRICS_SCHEMA_VERSION, "expected": "inherited_rows"}
        )
    with pytest.raises(ValueError, match="must contain only metric names"):
        MetricDeclaration.from_mapping(
            {"schema_version": DECLARED_METRICS_SCHEMA_VERSION, "optional": [1]}
        )
    with pytest.raises(ValueError, match="must be a boolean"):
        MetricDeclaration.from_mapping(
            {"schema_version": DECLARED_METRICS_SCHEMA_VERSION, "strict": "yes"}
        )


def test_ledger_exposes_its_binding_and_rejects_bad_arguments() -> None:
    declaration = MetricDeclaration(expected=("inherited_rows",))
    ledger = DeclaredMetricLedger(declaration)
    assert ledger.declaration is declaration
    assert ledger.strict is False
    assert ledger.run_id is None
    # Nothing was bound, so release and close_episode have no store to write to.
    ledger.release()
    assert ledger.close_episode("episode-1").missing_metrics == ("inherited_rows",)
    assert ledger.summary()["reported"] is True
    with pytest.raises(TypeError, match="strict must be bool"):
        DeclaredMetricLedger(declaration, strict="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timestamp_ns"):
        ledger.close_episode("episode-1", timestamp_ns=-1)
    with pytest.raises(TypeError, match="declaration must be a MetricDeclaration or None"):
        build_declared_metrics({"expected": []})  # type: ignore[arg-type]


def test_ledger_release_unbinds_a_bound_run(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    ledger = build_declared_metrics(
        MetricDeclaration(expected=EXPECTED), store=store, run_id=run.run_id
    )
    assert ledger is not None
    assert declared_metrics_for(store, run.run_id) is ledger
    ledger.release()
    assert declared_metrics_for(store, run.run_id) is None


def test_explicit_strict_overrides_capability_resolution(tmp_path: Path) -> None:
    store, run = _run(tmp_path)
    # No capability is granted and the declaration is not strict, so only the
    # explicit argument can turn the error on.
    ledger = build_declared_metrics(
        MetricDeclaration(expected=EXPECTED), strict=True, store=store, run_id=run.run_id
    )
    assert ledger is not None
    assert ledger.strict is True
    with pytest.raises(MissingDeclaredMetric, match="episode_reward"):
        ledger.close_episode("episode-1").require()
