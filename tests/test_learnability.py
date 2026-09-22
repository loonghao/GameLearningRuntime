"""Learnability budget: declared cardinality against the observed step rate."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

import game_learning_runtime
from game_learning_runtime import cli
from game_learning_runtime.cli import main
from game_learning_runtime.collector import SyncCollector
from game_learning_runtime.contracts import TimeStep
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.errors import ContractViolation, GLRError
from game_learning_runtime.learnability import (
    COVERAGE_RATIO_METRIC,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_VISIT_TARGET,
    LEARNABILITY_BUDGET_EVENT,
    LEARNABILITY_BUDGET_SCHEMA_VERSION,
    LEARNABILITY_CAPABILITY,
    LEARNABILITY_CELL_KEY,
    PROJECTED_STEPS_METRIC,
    STATE_ACTION_CELLS_METRIC,
    CardinalityKind,
    LearnabilityBudget,
    LearnabilityBudgetError,
    LearnabilityDeclaration,
    LearnabilityPlan,
    LearnabilityReport,
    LearnabilityStatus,
    LearnabilityTracker,
    SpaceCardinality,
    StateCellResolver,
    _string_cell,
    build_tracker,
    coerce_cell_identity,
    derive_action_cardinality,
    derive_state_cardinality,
)
from game_learning_runtime.run_store import RunStatus, TrainingStore
from game_learning_runtime.specs import CompositeSpec, EnvironmentSpec, SpaceKind, TensorSpec
from game_learning_runtime.termination import TerminationReason

BUDGET_STEPS = 100


class _CellEnvironment(GameEnvironment):
    """Tabular fixture: a seeded walk over ``cells`` discrete state cells."""

    def __init__(self, spec: EnvironmentSpec, *, seed: int = 7) -> None:
        self._spec = spec
        self._random = random.Random(seed)
        self._position = 0
        self._step_id = 0
        self._episode_id = uuid4()

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

    def step(self, action: Mapping[str, Any]) -> TimeStep:
        del action
        cells = int(self._spec.observation.flatten()["cell"].maximum) + 1
        self._position = self._random.randrange(cells)
        self._step_id += 1
        return self._timestep()

    def _timestep(self) -> TimeStep:
        return TimeStep(
            observation={"cell": np.array([self._position], dtype=np.int64)},
            reward=np.array([0.0], dtype=np.float32),
            terminated=np.array([False], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
            episode_id=self._episode_id,
            step_id=self._step_id,
            info={LEARNABILITY_CELL_KEY: self._position},
        )


class _NumpyCellEnvironment(_CellEnvironment):
    """The same walk, reporting the cell as the numpy scalar an adapter sees.

    ``observation[0]`` on a numpy observation is a ``numpy`` integer, which is
    not a Python ``int``.
    """

    def _timestep(self) -> TimeStep:
        return TimeStep(
            observation={"cell": np.array([self._position], dtype=np.int64)},
            reward=np.array([0.0], dtype=np.float32),
            terminated=np.array([False], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
            episode_id=self._episode_id,
            step_id=self._step_id,
            info={LEARNABILITY_CELL_KEY: np.int64(self._position)},
        )


def _cell_spec(cells: int) -> CompositeSpec:
    return CompositeSpec(
        {
            "cell": TensorSpec(
                (1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=max(cells - 1, 0)
            )
        }
    )


def _action_spec(actions: int) -> CompositeSpec:
    return CompositeSpec(
        {
            "choice": TensorSpec(
                (1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=max(actions - 1, 0)
            )
        }
    )


def _declaration(
    cells: int,
    *,
    actions: int = 1,
    kind: CardinalityKind = CardinalityKind.TABULAR,
    effective_capacity: int | None = None,
) -> LearnabilityDeclaration:
    return LearnabilityDeclaration(
        schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
        kind=kind,
        state=SpaceCardinality(cells=cells, bins={"cell": cells}),
        action=SpaceCardinality(cells=actions),
        effective_capacity=effective_capacity,
    )


def _environment(
    cells: int,
    *,
    actions: int = 1,
    declared: bool = True,
    seed: int = 7,
    **declaration_kwargs: Any,
) -> _CellEnvironment:
    return _CellEnvironment(
        EnvironmentSpec(
            environment_id="fixture.cells-v1",
            observation=_cell_spec(cells),
            action=_action_spec(actions),
            capabilities=frozenset({LEARNABILITY_CAPABILITY}) if declared else frozenset(),
            learnability=_declaration(cells, actions=actions, **declaration_kwargs)
            if declared
            else None,
        ),
        seed=seed,
    )


def _policy(timestep: TimeStep) -> dict[str, Any]:
    del timestep
    return {"choice": np.array([0], dtype=np.int64)}


class _StubGameLauncher:
    """Stands in for :class:`~game_learning_runtime.game_launcher.GameLauncher`.

    A CLI regression only needs the shape the train command consumes: one
    instance id, one manifest file, and a close fence. Spawning a real game
    process would test the launcher, not the mapping under test.
    """

    def __init__(self, config: object, *, project_root: object) -> None:
        self.config = config

    def start(self, run_dir: Path) -> _StubGameSet:
        manifest = Path(run_dir) / "games" / "game-instances.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps({"schema_version": "glr.game-instances.v1", "instances": []}),
            encoding="utf-8",
        )
        return _StubGameSet(manifest)


@dataclass(slots=True)
class _StubGameInstance:
    instance_id: str = "fixture.game-0000"


@dataclass(slots=True)
class _StubGameSet:
    manifest_path: Path
    instances: tuple[_StubGameInstance, ...] = (_StubGameInstance(),)

    def close(self) -> None:
        return None


def _budget(min_coverage: float | None = 0.5) -> LearnabilityBudget:
    return LearnabilityBudget(
        budget_steps=BUDGET_STEPS,
        min_coverage=min_coverage,
        budget_seconds=80.0,
    )


def _plan(min_coverage: float | None = 0.5) -> LearnabilityPlan:
    return LearnabilityPlan(budget=_budget(min_coverage))


def _upper_bound_declaration(cells: int = 1000) -> LearnabilityDeclaration:
    """A declaration that names an upper bound and no bins.

    This is the other legal path: the resolver is never built, so the cell can
    only arrive through ``info[LEARNABILITY_CELL_KEY]``.
    """

    return LearnabilityDeclaration(
        schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
        kind=CardinalityKind.TABULAR,
        state=SpaceCardinality(cells=cells, upper_bound=True),
        action=SpaceCardinality(cells=1),
    )


def _upper_bound_plan(min_coverage: float | None = 0.5) -> LearnabilityPlan:
    return LearnabilityPlan(budget=_budget(min_coverage), declaration=_upper_bound_declaration())


def test_cardinality_bins_must_multiply_to_cells() -> None:
    assert SpaceCardinality(cells=6, bins={"a": 2, "b": 3}).cells == 6
    with pytest.raises(ValueError, match="multiply to 12"):
        SpaceCardinality(cells=6, bins={"a": 3, "b": 4})
    with pytest.raises(ValueError, match="must be between"):
        SpaceCardinality(cells=0)


def test_declaration_uses_the_discrete_product() -> None:
    declaration = _declaration(1000, actions=4)
    assert declaration.state_action_cells == 4000


def test_declaration_accepts_declared_effective_capacity() -> None:
    declaration = _declaration(
        1000,
        actions=4,
        kind=CardinalityKind.FUNCTION_APPROXIMATION,
        effective_capacity=512,
    )
    assert declaration.state_action_cells == 512


def test_declaration_requires_a_schema_version_and_a_resolvable_size() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        LearnabilityDeclaration(schema_version="glr.other.v1", kind=CardinalityKind.TABULAR)
    with pytest.raises(ValueError, match="requires effective_capacity"):
        LearnabilityDeclaration(
            schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
            kind=CardinalityKind.TABULAR,
            state=SpaceCardinality(cells=10),
        )


def test_declaration_round_trips_through_mapping() -> None:
    declaration = _declaration(1000, actions=4)
    restored = LearnabilityDeclaration.from_mapping(declaration.to_mapping())
    assert restored == declaration
    assert restored.state_action_cells == 4000


def test_environment_spec_rejects_a_foreign_declaration() -> None:
    with pytest.raises(TypeError, match="learnability must be"):
        EnvironmentSpec(
            environment_id="fixture.cells-v1",
            observation=_cell_spec(4),
            action=_action_spec(1),
            learnability=SpaceCardinality(cells=4),  # type: ignore[arg-type]
        )


def test_resolver_quantizes_continuous_bounds_in_leaf_order() -> None:
    spec = CompositeSpec(
        {
            "pos": TensorSpec((1,), np.float32, minimum=0.0, maximum=1.0),
            "phase": TensorSpec((1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=1),
        }
    )
    resolver = StateCellResolver({"pos": 4, "phase": 2}, observation_spec=spec)
    assert resolver.paths == ("phase", "pos")
    seen = {
        resolver.resolve(
            {"pos": np.array([pos], dtype=np.float32), "phase": np.array([0], np.int64)}
        )
        for pos in (0.0, 0.3, 0.6, 0.9)
    }
    assert len(seen) == 4
    # The top bin is closed, so saturation cannot spill into a phantom cell.
    assert resolver.resolve(
        {"pos": np.array([0.9], dtype=np.float32), "phase": np.array([0], np.int64)}
    ) == resolver.resolve(
        {"pos": np.array([1.0], dtype=np.float32), "phase": np.array([0], np.int64)}
    )


def test_resolver_rejects_unknown_and_unbounded_leaves() -> None:
    spec = _cell_spec(4)
    with pytest.raises(ValueError, match="unknown leaves"):
        StateCellResolver({"missing": 4}, observation_spec=spec)
    unbounded = CompositeSpec({"pos": TensorSpec((1,), np.float32)})
    with pytest.raises(ValueError, match="needs bounds"):
        StateCellResolver({"pos": 4}, observation_spec=unbounded)


def test_derive_cardinality_reports_bounds_and_refuses_continuous_spaces() -> None:
    assert derive_state_cardinality(_cell_spec(1000)) == SpaceCardinality(
        cells=1000, upper_bound=True
    )
    assert derive_action_cardinality(_action_spec(4)) == SpaceCardinality(cells=4, upper_bound=True)
    assert derive_state_cardinality(CompositeSpec({"pos": TensorSpec((1,), np.float32)})) is None


def test_fail_fast_stops_before_the_budget_is_gone() -> None:
    collector = SyncCollector(_environment(1000), learnability=_plan())
    with pytest.raises(LearnabilityBudgetError) as raised:
        collector.collect(_policy, steps=BUDGET_STEPS)
    report = raised.value.report
    assert report.state_action_cells == 1000
    # The whole point: the run stops while most of the budget is unspent.
    assert report.steps < BUDGET_STEPS
    assert report.status is LearnabilityStatus.FAILED
    assert report.coverage_ratio < 0.5


def test_failing_fixture_reports_the_projection_and_three_numbered_remedies() -> None:
    collector = SyncCollector(_environment(1000), learnability=_plan())
    with pytest.raises(LearnabilityBudgetError) as raised:
        collector.collect(_policy, steps=BUDGET_STEPS)
    error = raised.value
    report = error.report
    assert report.projected_steps_to_k_visits is not None
    assert report.projected_steps_to_k_visits > BUDGET_STEPS
    assert report.visit_target == DEFAULT_VISIT_TARGET
    assert len(error.remedies) == 3
    message = str(error)
    # Every remedy has to name a number, or it is just a guess moved elsewhere.
    assert "1. shrink the declared state-action space from 1,000" in message
    assert "2. replace the tabular encoding" in message
    assert "3. raise throughput from" in message
    assert "steps/s" in message


def test_passing_fixture_covers_half_the_space_within_budget() -> None:
    collector = SyncCollector(_environment(50), learnability=_plan())
    collector.collect(_policy, steps=BUDGET_STEPS)
    report = collector.learnability_report()
    assert report is not None
    assert report.state_action_cells == 50
    assert report.status is LearnabilityStatus.OK
    assert report.coverage_ratio >= 0.5
    assert report.steps == BUDGET_STEPS


def test_unconfigured_run_warns_without_stopping() -> None:
    collector = SyncCollector(_environment(1000), learnability=_plan(min_coverage=None))
    collector.collect(_policy, steps=BUDGET_STEPS)
    report = collector.learnability_report()
    assert report is not None
    assert report.coverage_configured is False
    assert report.min_coverage == DEFAULT_MIN_COVERAGE
    assert report.status is LearnabilityStatus.WARNING
    assert report.coverage_ratio < DEFAULT_MIN_COVERAGE


def test_budget_error_is_not_a_contract_violation_or_a_transport_error() -> None:
    tracker = LearnabilityTracker(_declaration(1000), _budget())
    for _ in range(BUDGET_STEPS):
        tracker.observe(cell=0, elapsed_seconds=0.01)
    with pytest.raises(LearnabilityBudgetError) as raised:
        tracker.require()
    assert isinstance(raised.value, GLRError)
    assert not isinstance(raised.value, ContractViolation)


def test_undeclared_adapter_behaves_exactly_as_before() -> None:
    collector = SyncCollector(_environment(1000, declared=False))
    assert collector.learnability is None
    assert collector.learnability_report() is None
    unroll = collector.collect(_policy, steps=BUDGET_STEPS)
    assert len(unroll.transitions) == BUDGET_STEPS


def test_declared_adapter_stays_inert_without_a_plan() -> None:
    collector = SyncCollector(_environment(50))
    assert collector.learnability is None
    unroll = collector.collect(_policy, steps=BUDGET_STEPS)
    assert len(unroll.transitions) == BUDGET_STEPS


def test_plan_without_any_declaration_fails_closed_at_construction() -> None:
    with pytest.raises(ValueError, match="requires a declaration"):
        SyncCollector(_environment(50, declared=False), learnability=_plan())


def test_plan_requires_a_learnability_plan() -> None:
    with pytest.raises(TypeError, match="LearnabilityPlan"):
        SyncCollector(_environment(50), learnability=_budget())  # type: ignore[arg-type]


def test_function_approximation_tracks_declared_capacity_not_the_product() -> None:
    tracker = build_tracker(
        LearnabilityPlan(
            budget=_budget(min_coverage=None),
            declaration=_declaration(
                1000,
                actions=4,
                kind=CardinalityKind.FUNCTION_APPROXIMATION,
                effective_capacity=512,
            ),
        ),
        observation_spec=_cell_spec(1000),
    )
    assert tracker.state_action_cells == 512
    for cell in range(256):
        tracker.observe(cell=cell, elapsed_seconds=0.01)
    report = tracker.report()
    assert report.state_action_cells == 512
    assert abs(report.coverage_ratio - 0.5) < 1e-9
    assert any("effective capacity" in note for note in report.notes)


def test_unresolvable_cells_are_charged_and_reported() -> None:
    declaration = LearnabilityDeclaration(
        schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
        kind=CardinalityKind.TABULAR,
        state=SpaceCardinality(cells=1000, upper_bound=True),
        action=SpaceCardinality(cells=1),
    )
    tracker = LearnabilityTracker(declaration, _budget(min_coverage=None))
    tracker.observe(steps=10)
    report = tracker.report()
    assert report.distinct_cells_visited == 0
    assert report.unresolved_steps == 10
    assert report.steps == 10
    assert any("no state cell was resolved" in note for note in report.notes)


def test_cell_identity_may_arrive_through_info() -> None:
    environment = _environment(1000)
    declaration = environment.spec.learnability
    assert declaration is not None
    declared_only = replace(declaration, state=SpaceCardinality(cells=1000, upper_bound=True))
    tracker = LearnabilityTracker(declared_only, _budget(min_coverage=None))
    tracker.observe(cell=environment.reset().info[LEARNABILITY_CELL_KEY])
    assert tracker.distinct_cells_visited == 1


def test_budget_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="budget_steps"):
        LearnabilityBudget(budget_steps=0)
    with pytest.raises(ValueError, match="min_coverage"):
        LearnabilityBudget(budget_steps=10, min_coverage=0.0)
    with pytest.raises(ValueError, match="visit_target"):
        LearnabilityBudget(budget_steps=10, visit_target=0)


def test_report_round_trips_through_the_run_store_projection() -> None:
    collector = SyncCollector(_environment(1000), learnability=_plan())
    with pytest.raises(LearnabilityBudgetError):
        collector.collect(_policy, steps=BUDGET_STEPS)
    original = collector.learnability_report()
    assert original is not None
    restored = LearnabilityReport.from_mapping(original.to_mapping())
    assert restored.to_mapping() == original.to_mapping()


def test_run_store_exposes_all_three_numbers_as_first_class_metrics(tmp_path: Path) -> None:
    """Scale is recorded next to the two ratios it explains.

    A coverage ratio alone cannot say whether the run covered 10 cells or
    10,000, and cardinality is the subject of this capability, so
    ``state_action_cells`` is a metric of its own rather than a field a
    consumer has to reconstruct from the event payload.
    """

    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="fixture.cells-v1", protocol_version="1.0", kind="training"
    )
    tracker = LearnabilityTracker(_declaration(1000), _budget())
    for cell in range(100):
        tracker.observe(cell=cell, elapsed_seconds=0.01)
    report = tracker.report()
    store.record_learnability(run.run_id, report)

    metrics = {metric.name: metric.value for metric in store.list_metrics(run.run_id)}
    assert set(metrics) == {
        COVERAGE_RATIO_METRIC,
        PROJECTED_STEPS_METRIC,
        STATE_ACTION_CELLS_METRIC,
    }
    assert metrics[STATE_ACTION_CELLS_METRIC] == 1000.0
    assert metrics[COVERAGE_RATIO_METRIC] == report.coverage_ratio
    restored = store.list_learnability(run.run_id)
    assert len(restored) == 1
    assert restored[0].coverage_ratio == report.coverage_ratio
    assert restored[0].projected_steps_to_k_visits == report.projected_steps_to_k_visits
    assert restored[0].state_action_cells == 1000
    kinds = {event.kind for event in store.list_events(run.run_id)}
    assert LEARNABILITY_BUDGET_EVENT in kinds


def test_run_store_records_the_scale_of_a_function_approximation_run(tmp_path: Path) -> None:
    """The recorded scale is the effective capacity, not the bin product."""

    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="fixture.cells-v1", protocol_version="1.0", kind="training"
    )
    tracker = build_tracker(
        LearnabilityPlan(
            budget=_budget(min_coverage=None),
            declaration=_declaration(
                1000,
                actions=4,
                kind=CardinalityKind.FUNCTION_APPROXIMATION,
                effective_capacity=512,
            ),
        ),
        observation_spec=_cell_spec(1000),
    )
    for cell in range(256):
        tracker.observe(cell=cell, elapsed_seconds=0.01)
    store.record_learnability(run.run_id, tracker.report())
    metrics = {metric.name: metric.value for metric in store.list_metrics(run.run_id)}
    assert metrics[STATE_ACTION_CELLS_METRIC] == 512.0
    assert metrics[COVERAGE_RATIO_METRIC] == pytest.approx(0.5)


def test_run_store_rejects_a_foreign_report(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="fixture.cells-v1", protocol_version="1.0", kind="training"
    )
    with pytest.raises(TypeError, match="LearnabilityReport"):
        store.record_learnability(run.run_id, {"coverage_ratio": 0.5})  # type: ignore[arg-type]


def _project(root: Path) -> None:
    (root / "bridge").mkdir()
    (root / "glr-project.json").write_text(
        json.dumps(
            {
                "schema_version": "glr.project.v1",
                "environment_id": "fixture.cells-v1",
                "environment_family": "fixture",
                "protocol_version": "1.0",
                "data_dir": ".glr",
                "bridge_path": "bridge",
                "runtime": {"argv": [sys.executable, "-c", "print('runtime')"]},
                "trainer": {"argv": [sys.executable, "-c", "print('train')"]},
                "player": {"argv": [sys.executable, "-c", "print('play')", "{bundle}"]},
            }
        ),
        encoding="utf-8",
    )


def test_cli_json_exposes_both_numbers_without_parsing_a_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)
    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = store.create_run(
        environment_id="fixture.cells-v1", protocol_version="1.0", kind="training"
    )
    tracker = LearnabilityTracker(_declaration(1000), _budget())
    for cell in range(100):
        tracker.observe(cell=cell, elapsed_seconds=0.01)
    expected = tracker.report()
    store.record_learnability(run.run_id, expected)
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)

    assert main(["--project", str(tmp_path), "--json", "runs", "show", run.run_id]) == 0
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["learnability_summary"]["reported"] is True
    assert data["learnability_summary"]["coverage_ratio"] == expected.coverage_ratio
    assert data["learnability"][0]["projected_steps_to_k_visits"] == (
        expected.projected_steps_to_k_visits
    )


def test_cli_reports_an_absent_verdict_explicitly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)
    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = store.create_run(
        environment_id="fixture.cells-v1", protocol_version="1.0", kind="training"
    )
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
    assert main(["--project", str(tmp_path), "--json", "runs", "show", run.run_id]) == 0
    summary = json.loads(capsys.readouterr().out)["data"]["learnability_summary"]
    assert summary["reported"] is False
    assert summary["status"] is None


def test_cli_rejects_a_coverage_floor_outside_the_unit_interval(tmp_path: Path) -> None:
    _project(tmp_path)
    with pytest.raises(ContractViolation, match="--min-coverage"):
        main(["--project", str(tmp_path), "--json", "train", "--min-coverage", "0"])


def test_cli_fails_a_green_trainer_that_recorded_a_failed_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _project(tmp_path)
    (tmp_path / "trainer.py").write_text(
        """
import os
from game_learning_runtime.learnability import (
    LEARNABILITY_BUDGET_SCHEMA_VERSION,
    CardinalityKind,
    LearnabilityBudget,
    LearnabilityDeclaration,
    LearnabilityTracker,
    SpaceCardinality,
)
from game_learning_runtime.run_store import TrainingStore

store = TrainingStore(os.environ["GLR_STORE_PATH"])
tracker = LearnabilityTracker(
    LearnabilityDeclaration(
        schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
        kind=CardinalityKind.TABULAR,
        state=SpaceCardinality(cells=1000, upper_bound=True),
        action=SpaceCardinality(cells=1),
    ),
    LearnabilityBudget(budget_steps=100, min_coverage=0.5, budget_seconds=80.0),
)
tracker.observe(cell=1, elapsed_seconds=0.01, steps=20)
store.record_learnability(os.environ["GLR_RUN_ID"], tracker.report())
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "glr-project.json").write_text(
        json.dumps(
            {
                "schema_version": "glr.project.v1",
                "environment_id": "fixture.cells-v1",
                "environment_family": "fixture",
                "protocol_version": "1.0",
                "data_dir": ".glr",
                "bridge_path": "bridge",
                "runtime": {"argv": [sys.executable, "-c", "print('runtime')"]},
                "player": {"argv": [sys.executable, "-c", "print('play')", "{bundle}"]},
                "trainer": {"argv": [sys.executable, str(tmp_path / "trainer.py")]},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["--project", str(tmp_path), "--json", "train", "--min-coverage", "0.5"])
    assert exit_code == 1
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["status"] == "failed"
    assert data["learnability"]["status"] == "failed"


def test_cardinality_rejects_malformed_bins_and_bounds() -> None:
    with pytest.raises(TypeError, match=r"cardinality\.cells must be an integer"):
        SpaceCardinality(cells=4.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"cardinality\.upper_bound must be a boolean"):
        SpaceCardinality(cells=4, upper_bound="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"cardinality\.bins must be an object"):
        SpaceCardinality(cells=4, bins="cell")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"cardinality\.bins requires string keys"):
        SpaceCardinality(cells=4, bins={1: 4})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="invalid leaf name"):
        SpaceCardinality(cells=4, bins={"bad name": 4})
    with pytest.raises(TypeError, match=r"cardinality\.bins\.leaf must be an integer"):
        SpaceCardinality(cells=4, bins={"leaf": "4"})  # type: ignore[dict-item]


def test_declaration_from_mapping_is_strict_about_every_field() -> None:
    payload = _declaration(100, actions=2).to_mapping()
    with pytest.raises(TypeError, match="learnability must be an object"):
        LearnabilityDeclaration.from_mapping("declaration")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unexpected fields"):
        LearnabilityDeclaration.from_mapping({**payload, "extra": 1})
    with pytest.raises(ValueError, match=r"learnability\.kind must be one of"):
        LearnabilityDeclaration.from_mapping({**payload, "kind": "lookup-table"})
    with pytest.raises(TypeError, match="cardinality must be an object"):
        LearnabilityDeclaration.from_mapping({**payload, "state": 4})

    capacity_only = LearnabilityDeclaration(
        schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
        kind=CardinalityKind.FUNCTION_APPROXIMATION,
        effective_capacity=64,
    )
    assert LearnabilityDeclaration.from_mapping(capacity_only.to_mapping()) == capacity_only
    assert capacity_only.to_mapping()["state"] is None


def test_budget_reports_the_threshold_it_will_enforce() -> None:
    unconfigured = LearnabilityBudget(budget_steps=100)
    assert unconfigured.configured is False
    assert unconfigured.threshold == DEFAULT_MIN_COVERAGE
    assert unconfigured.to_mapping()["min_coverage"] is None
    configured = LearnabilityBudget(budget_steps=100, min_coverage=0.25)
    assert configured.configured is True
    assert configured.to_mapping()["threshold"] == 0.25
    with pytest.raises(ValueError, match="budget_seconds"):
        LearnabilityBudget(budget_steps=100, budget_seconds=0)
    with pytest.raises(TypeError, match="min_coverage must be a number"):
        LearnabilityBudget(budget_steps=100, min_coverage="half")  # type: ignore[arg-type]


def test_plan_rejects_foreign_members() -> None:
    with pytest.raises(TypeError, match=r"plan\.budget must be a LearnabilityBudget"):
        LearnabilityPlan(budget=100)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"plan\.declaration must be a LearnabilityDeclaration"):
        LearnabilityPlan(budget=_budget(), declaration=SpaceCardinality(cells=4))  # type: ignore[arg-type]


def test_report_flags_failure_and_the_k_visit_target() -> None:
    tracker = LearnabilityTracker(_declaration(20), _budget())
    for cell in range(20):
        tracker.observe(cell=cell, elapsed_seconds=0.01)
    report = tracker.report()
    assert report.status is LearnabilityStatus.OK
    assert report.failed is False
    assert report.projected_steps_to_k_visits == 20 * DEFAULT_VISIT_TARGET
    assert report.within_budget is True


def test_report_from_mapping_rejects_a_foreign_payload() -> None:
    tracker = LearnabilityTracker(_declaration(1000), _budget())
    tracker.observe(cell=1, elapsed_seconds=0.1)
    payload = tracker.report().to_mapping()
    with pytest.raises(TypeError, match="learnability report must be an object"):
        LearnabilityReport.from_mapping("report")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema_version"):
        LearnabilityReport.from_mapping({**payload, "schema_version": "glr.other.v1"})
    with pytest.raises(ValueError, match="status must be one of"):
        LearnabilityReport.from_mapping({**payload, "status": "maybe"})
    with pytest.raises(TypeError, match="coverage_configured must be a boolean"):
        LearnabilityReport.from_mapping({**payload, "coverage_configured": "yes"})
    with pytest.raises(ValueError, match="coverage_ratio"):
        LearnabilityReport.from_mapping({**payload, "coverage_ratio": 1.4})
    with pytest.raises(TypeError, match="remedies must be an array of strings"):
        LearnabilityReport.from_mapping({**payload, "remedies": [1]})
    with pytest.raises(ValueError, match="projected_steps_to_k_visits"):
        LearnabilityReport.from_mapping({**payload, "projected_steps_to_k_visits": 0})
    assert LearnabilityReport.from_mapping({**payload, "notes": None, "remedies": None}).notes == ()


def test_resolver_rejects_empty_bins_and_malformed_observations() -> None:
    with pytest.raises(ValueError, match="bins cannot be empty"):
        StateCellResolver({})
    resolver = StateCellResolver({"cell": 4, "phase": 2})
    with pytest.raises(KeyError, match="missing learnability leaf"):
        resolver.resolve({"cell": np.array([1], dtype=np.int64)})
    with pytest.raises(ValueError, match="must hold exactly one value"):
        resolver.resolve(
            {"cell": np.array([1, 2], dtype=np.int64), "phase": np.array([0], dtype=np.int64)}
        )
    with pytest.raises(ValueError, match="outside 4 bins"):
        resolver.resolve(
            {"cell": np.array([9], dtype=np.int64), "phase": np.array([0], dtype=np.int64)}
        )


def test_resolver_checks_discrete_bounds_and_degenerate_spans() -> None:
    bounded = StateCellResolver(
        {"cell": 4},
        observation_spec=CompositeSpec(
            {"cell": TensorSpec((1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=3)}
        ),
    )
    assert bounded.resolve({"cell": np.array([3], dtype=np.int64)}) == 3
    with pytest.raises(ValueError, match="outside 4 bins from 0"):
        bounded.resolve({"cell": np.array([4], dtype=np.int64)})

    degenerate = StateCellResolver(
        {"pos": 4},
        observation_spec=CompositeSpec(
            {"pos": TensorSpec((1,), np.float32, minimum=1.0, maximum=1.0)}
        ),
    )
    assert degenerate.resolve({"pos": np.array([1.0], dtype=np.float32)}) == 0


def test_tracker_rejects_foreign_arguments() -> None:
    with pytest.raises(TypeError, match="declaration must be a LearnabilityDeclaration"):
        LearnabilityTracker(SpaceCardinality(cells=4), _budget())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="budget must be a LearnabilityBudget"):
        LearnabilityTracker(_declaration(4), 100)  # type: ignore[arg-type]


def test_tracker_accepts_string_cells_and_external_updates() -> None:
    tracker = LearnabilityTracker(_declaration(1000), _budget(min_coverage=None))
    with pytest.raises(TypeError, match="cell must be an integer, a string, or None"):
        tracker.observe(cell=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="steps must be a non-negative integer"):
        tracker.observe(cell=1, steps=-1)
    with pytest.raises(ValueError, match="now_ns must be a non-negative integer or None"):
        tracker.observe(cell=1, now_ns=-1)
    with pytest.raises(ValueError, match="cannot exceed 512 characters"):
        tracker.observe(cell="x" * 513)

    tracker.observe(cell="zone-a")
    report = tracker.note_updates(7)
    assert report.steps == 1
    assert report.updates == 7
    assert report.distinct_cells_visited == 1
    # A second, different string cell must not collide with the first.
    tracker.observe(cell="zone-b")
    assert tracker.distinct_cells_visited == 2


def test_tracker_measures_elapsed_time_from_timestamps() -> None:
    tracker = LearnabilityTracker(_declaration(1000), _budget(min_coverage=None))
    base = 1_000_000_000
    tracker.observe(cell=1, now_ns=base)
    tracker.observe(cell=2, now_ns=base + 500_000_000)
    report = tracker.report()
    assert report.steps == 2
    assert report.elapsed_seconds is not None
    assert report.elapsed_seconds == pytest.approx(0.5)
    assert report.steps_per_second == pytest.approx(4.0)


def test_build_tracker_refuses_bins_without_an_observation_spec() -> None:
    plan = LearnabilityPlan(budget=_budget(), declaration=_declaration(1000))
    assert build_tracker(plan, observation_spec=_cell_spec(1000)).state_action_cells == 1000
    with pytest.raises(TypeError, match="plan must be a LearnabilityPlan"):
        build_tracker(_budget())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="learnability requires a declaration"):
        build_tracker(LearnabilityPlan(budget=_budget()))
    with pytest.raises(ValueError, match="learnability bins require an observation spec"):
        build_tracker(plan)
    explicit = build_tracker(plan, declaration=_declaration(64), observation_spec=_cell_spec(64))
    assert explicit.state_action_cells == 64


def test_derive_cardinality_covers_binary_multi_and_dynamic_leaves() -> None:
    binary = CompositeSpec({"flag": TensorSpec((1,), np.bool_, kind=SpaceKind.BINARY)})
    assert derive_state_cardinality(binary) == SpaceCardinality(cells=2, upper_bound=True)
    multi = CompositeSpec(
        {"pair": TensorSpec((2,), np.int64, kind=SpaceKind.MULTI_DISCRETE, minimum=0, maximum=1)}
    )
    assert derive_state_cardinality(multi) == SpaceCardinality(cells=4, upper_bound=True)
    unbounded = CompositeSpec(
        {"cell": TensorSpec((1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0)}
    )
    assert derive_state_cardinality(unbounded) is None
    dynamic = CompositeSpec(
        {"cell": TensorSpec((None,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=3)}
    )
    assert derive_state_cardinality(dynamic) is None


def test_throughput_remedy_falls_back_to_a_step_multiple() -> None:
    budget = LearnabilityBudget(budget_steps=100, min_coverage=0.5)
    tracker = LearnabilityTracker(_declaration(1000), budget)
    with pytest.raises(LearnabilityBudgetError) as error:
        for cell in range(40):
            tracker.observe(cell=cell, elapsed_seconds=0.01)
            tracker.require()
    message = str(error.value)
    assert "x the 100-step budget" in message
    assert "declare budget_seconds" in message
    assert len(error.value.remedies) == 3


def test_cardinality_from_mapping_is_strict() -> None:
    with pytest.raises(ValueError, match="cardinality contains unexpected fields"):
        SpaceCardinality.from_mapping({"cells": 4, "bins": {}, "extra": 1})


def test_declaration_rejects_foreign_member_types() -> None:
    with pytest.raises(TypeError, match=r"learnability\.kind must be a CardinalityKind"):
        LearnabilityDeclaration(schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION, kind="tabular")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"learnability\.state must be a SpaceCardinality"):
        LearnabilityDeclaration(
            schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
            kind=CardinalityKind.TABULAR,
            state=4,  # type: ignore[arg-type]
            action=SpaceCardinality(cells=1),
        )
    with pytest.raises(TypeError, match=r"learnability\.action must be a SpaceCardinality"):
        LearnabilityDeclaration(
            schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
            kind=CardinalityKind.TABULAR,
            state=SpaceCardinality(cells=4),
            action=4,  # type: ignore[arg-type]
        )


def test_report_rejects_a_foreign_status() -> None:
    payload = {
        "schema_version": LEARNABILITY_BUDGET_SCHEMA_VERSION,
        "state_action_cells": 1000,
        "distinct_cells_visited": 1,
        "steps": 1,
        "updates": 0,
        "visit_target": DEFAULT_VISIT_TARGET,
        "coverage_ratio": 0.0,
        "projected_steps_to_k_visits": None,
        "projected_coverage_at_budget": None,
        "steps_per_second": None,
        "projected_seconds_to_k_visits": None,
        "required_steps_per_second": None,
        "elapsed_seconds": None,
        "budget_steps": BUDGET_STEPS,
        "budget_seconds": None,
        "min_coverage": DEFAULT_MIN_COVERAGE,
        "coverage_configured": True,
        "unresolved_steps": 0,
        "remedies": (),
    }
    with pytest.raises(TypeError, match="status must be a LearnabilityStatus"):
        LearnabilityReport(**payload, status="failed")  # type: ignore[arg-type]
    healthy = LearnabilityReport(**payload, status=LearnabilityStatus.FAILED)
    assert healthy.failed is True
    assert healthy.notes == ()


def test_tracker_projects_nothing_before_the_first_step() -> None:
    budget = _budget()
    tracker = LearnabilityTracker(_declaration(1000), budget)
    assert tracker.steps == 0
    assert tracker.updates == 0
    assert tracker.budget is budget
    report = tracker.report()
    assert report.projected_steps_to_k_visits is None
    assert report.steps_per_second is None
    assert report.status is LearnabilityStatus.OK
    with pytest.raises(TypeError, match="elapsed_seconds must be a number"):
        tracker.observe(cell=1, elapsed_seconds="fast")  # type: ignore[arg-type]


def test_tracker_resolves_cells_from_an_observation() -> None:
    declaration = _declaration(1000)
    assert declaration.state is not None
    resolver = StateCellResolver(declaration.state.bins, observation_spec=_cell_spec(1000))
    tracker = LearnabilityTracker(declaration, _budget(min_coverage=None), resolver=resolver)
    tracker.observe(observation={"cell": np.array([7], dtype=np.int64)})
    tracker.observe(observation={"cell": np.array([7], dtype=np.int64)})
    tracker.observe(observation={"cell": np.array([8], dtype=np.int64)})
    assert tracker.distinct_cells_visited == 2
    assert tracker.steps == 3


def test_resolver_bins_a_binary_leaf_directly() -> None:
    spec = CompositeSpec({"flag": TensorSpec((1,), np.bool_, kind=SpaceKind.BINARY)})
    resolver = StateCellResolver({"flag": 2}, observation_spec=spec)
    assert resolver.resolve({"flag": np.array([False], dtype=np.bool_)}) == 0
    assert resolver.resolve({"flag": np.array([True], dtype=np.bool_)}) == 1


def test_report_from_mapping_rejects_a_non_numeric_ratio() -> None:
    tracker = LearnabilityTracker(_declaration(1000), _budget())
    tracker.observe(cell=1, elapsed_seconds=0.1)
    payload = tracker.report().to_mapping()
    with pytest.raises(TypeError, match="coverage_ratio must be a number"):
        LearnabilityReport.from_mapping({**payload, "coverage_ratio": None})


def test_numpy_cell_reported_through_info_is_resolved() -> None:
    """An observation is a numpy array, so its scalar is a numpy integer.

    Upper-bound-only adapters take this path: with no declared bins no
    resolver is built, so dropping the scalar would charge every step to no
    cell, hold coverage at zero, and never fire the floor -- while the note
    tells the operator to emit the key they already emit.
    """

    tracker = LearnabilityTracker(_upper_bound_declaration(), _budget(min_coverage=None))
    tracker.observe(cell=np.int64(5))
    tracker.observe(cell=np.int32(5))
    tracker.observe(cell=np.array(6))
    assert tracker.distinct_cells_visited == 2
    report = tracker.report()
    assert report.unresolved_steps == 0
    assert not any("no state cell was resolved" in note for note in report.notes)


def test_upper_bound_adapter_reached_through_info_meets_the_floor() -> None:
    """The collector agrees with the tracker: a numpy cell still counts."""

    collector = SyncCollector(
        _NumpyCellEnvironment(_environment(1000).spec), learnability=_upper_bound_plan()
    )
    with pytest.raises(LearnabilityBudgetError):
        collector.collect(_policy, steps=BUDGET_STEPS)
    report = collector.learnability_report()
    assert report is not None
    assert report.distinct_cells_visited > 0
    assert report.unresolved_steps == 0


def test_foreign_cell_values_stay_inert_or_loud() -> None:
    """The two reporting paths agree: what one resolves the other resolves."""

    tracker = LearnabilityTracker(_upper_bound_declaration(), _budget(min_coverage=None))
    with pytest.raises(TypeError, match="cell must be an integer, a string, or None"):
        tracker.observe(cell=1.5)  # type: ignore[arg-type]
    # A value that is not a cell identity at all is unresolvable, not wrong:
    # it stays charged as an unresolved step, exactly as it was before.
    assert coerce_cell_identity(object()) is None
    assert coerce_cell_identity(np.array([1, 2])) is None
    assert coerce_cell_identity(np.int64(7)) == 7
    assert coerce_cell_identity("zone-a") == "zone-a"


def test_learnability_abort_settles_the_episode_it_stopped(tmp_path: Path) -> None:
    """A gate abort owes a terminal state, like an environment error does.

    An episode that neither recorded a reason nor raised a violation must not
    be observable: the store would otherwise show a run that ended an episode
    as having ended none, and its declared-metric audit would never settle.
    """

    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="fixture.cells-v1", protocol_version="1.0", kind="training"
    )
    collector = SyncCollector(
        _environment(1000),
        learnability=_plan(),
        on_termination=store.termination_sink(run.run_id),
    )
    with pytest.raises(LearnabilityBudgetError):
        collector.collect(_policy, steps=BUDGET_STEPS)
    terminations = store.list_episode_terminations(run.run_id)
    assert len(terminations) == 1
    assert terminations[0].reason is TerminationReason.FAILED
    assert len(collector.terminations) == 1


def test_list_learnability_survives_a_long_event_stream(tmp_path: Path) -> None:
    """The verdict window is bounded by verdicts, not by the events around them.

    A run that records a failed verdict after a long telemetry stream still has
    to be readable, or a trainer that exits zero reads green while `runs show`
    reports the verdict as unreported.
    """

    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="fixture.cells-v1", protocol_version="1.0", kind="training"
    )
    for index in range(1200):
        store.append_event(run.run_id, kind="telemetry", payload={"index": index})
    tracker = LearnabilityTracker(_upper_bound_declaration(), _budget())
    tracker.observe(cell=1, elapsed_seconds=0.01, steps=20)
    report = tracker.report()
    assert report.failed
    store.record_learnability(run.run_id, report)
    verdicts = store.list_learnability(run.run_id)
    assert len(verdicts) == 1
    assert verdicts[-1].failed


def test_train_passes_the_coverage_floor_when_a_game_is_launched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The floor has to survive the game launch.

    Rebinding the extra mapping at game launch dropped it, and a game-backed
    project is the ordinary shape of a real run, so the gate never reached the
    trainer while the run metadata still claimed it was set.
    """

    seen = tmp_path / "min-coverage.txt"
    (tmp_path / "trainer.py").write_text(
        f"""
import os
from pathlib import Path

Path(r"{seen}").write_text(os.environ.get("GLR_MIN_COVERAGE", ""), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "bridge").mkdir()
    project = {
        "schema_version": "glr.project.v1",
        "environment_id": "fixture.cells-v1",
        "environment_family": "fixture",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": [sys.executable, "-c", "print('runtime')"]},
        "player": {"argv": [sys.executable, "-c", "print('play')", "{bundle}"]},
        "trainer": {"argv": [sys.executable, str(tmp_path / "trainer.py")]},
        "game": {
            "schema_version": "glr.game-launch.v1",
            "game_id": "fixture.game",
            "command": {"argv": [sys.executable, "-c", "print('game')"]},
            "instances": 1,
            "readiness": {"kind": "file", "path": "ready.json"},
        },
    }
    (tmp_path / "glr-project.json").write_text(json.dumps(project), encoding="utf-8")
    monkeypatch.setattr(cli, "GameLauncher", _StubGameLauncher)
    assert main(["--project", str(tmp_path), "--json", "train", "--min-coverage", "0.5"]) == 0
    assert seen.read_text(encoding="utf-8") == "0.500000"


def test_coverage_efficiency_decays_on_a_long_run() -> None:
    """Efficiency is new cells per step, so it must fall as steps accumulate.

    Bounding the denominator by ``cells`` froze the ratio at the coverage
    ratio once ``steps`` passed ``cells``. The projection then shrank below the
    steps already spent and the fail-fast gate never fired -- on exactly the
    long runs this capability exists to catch.
    """

    declaration = LearnabilityDeclaration(
        schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
        kind=CardinalityKind.TABULAR,
        state=SpaceCardinality(cells=1_000),
        action=SpaceCardinality(cells=1),
    )
    budget = LearnabilityBudget(budget_steps=100_000, min_coverage=0.8, visit_target=4)
    tracker = LearnabilityTracker(declaration, budget)
    # 20,000 steps that cycle through only 300 of the 1,000 cells.
    for step in range(20_000):
        tracker.observe(cell=step % 300, steps=1)

    report = tracker.report()
    assert report.steps == 20_000
    assert report.distinct_cells_visited == 300
    # 300 new cells over 20,000 steps, not over min(20_000, 1_000).
    assert report.coverage_ratio == pytest.approx(0.3)
    projected = report.projected_steps_to_k_visits
    assert projected is not None
    # The old formula produced 13,334 -- fewer steps than were already spent.
    assert projected == 266_667
    assert projected > report.steps


def test_an_unresolvable_observation_is_charged_not_raised() -> None:
    """A declared binning that does not fit the observation must not escape.

    ``StateCellResolver.resolve`` raises ``KeyError`` for a missing leaf and
    ``ValueError`` for a wrong arity or an out-of-range value. Those are
    declared-configuration mismatches, not caller bugs, and the tracker's own
    contract is that an untracked step is not free.
    """

    plan = LearnabilityPlan(
        budget=LearnabilityBudget(budget_steps=1_000, min_coverage=None, visit_target=4),
        declaration=LearnabilityDeclaration(
            schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
            kind=CardinalityKind.TABULAR,
            state=SpaceCardinality(cells=100, bins={"a": 10, "b": 10}),
            action=SpaceCardinality(cells=1),
        ),
    )
    observation_spec = CompositeSpec(
        {
            "a": TensorSpec((1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=9),
            "b": TensorSpec((1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=9),
        }
    )
    tracker = build_tracker(plan, observation_spec=observation_spec)
    assert tracker is not None

    for observation in (
        {"a": 1},  # missing leaf -> KeyError
        {"a": [1, 2], "b": 3},  # wrong arity -> ValueError
        {"a": 99, "b": 3},  # outside the declared bins -> ValueError
    ):
        report = tracker.observe(observation=observation, steps=1)

    assert report.steps == 3
    assert report.unresolved_steps == 3
    assert report.distinct_cells_visited == 0


def test_string_cells_do_not_collide_with_each_other_or_with_integers() -> None:
    """Two different string cells must count as two visited cells.

    The previous position-weighted sum was not injective: ``"ab"`` and ``"ca"``
    both summed to 293, and ``""`` summed to 0, colliding with the very common
    integer cell ``0``. Either collision under-reports coverage and can turn a
    healthy run into a FAILED verdict.
    """

    assert _string_cell("ab") != _string_cell("ca")

    declaration = LearnabilityDeclaration(
        schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
        kind=CardinalityKind.TABULAR,
        state=SpaceCardinality(cells=1_000),
        action=SpaceCardinality(cells=1),
    )
    budget = LearnabilityBudget(budget_steps=1_000, min_coverage=None)

    strings = LearnabilityTracker(declaration, budget)
    strings.observe(cell="ab")
    strings.observe(cell="ca")
    assert strings.distinct_cells_visited == 2

    mixed = LearnabilityTracker(declaration, budget)
    mixed.observe(cell=0)
    mixed.observe(cell="")
    assert mixed.distinct_cells_visited == 2


def test_string_cell_identity_is_stable_across_processes() -> None:
    """A run must report the same coverage after a restart.

    :func:`hash` is salted per process, so using it here would let two runs of
    the same configuration disagree about how much space they covered.
    """

    script = (
        "from game_learning_runtime.learnability import _string_cell; print(_string_cell('zone-a'))"
    )
    first = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    second = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert first.stdout.strip() == second.stdout.strip()
    assert first.stdout.strip() == str(_string_cell("zone-a"))


def test_package_reexports_every_name_it_publishes() -> None:
    """`__all__` must not advertise a name the package does not bind.

    A re-export that lands in `__all__` but not in the import block breaks
    `from game_learning_runtime import *` for every consumer, and ruff cannot
    catch it: F822 is suppressed in `__init__.py`. The learnability metric
    constants are exactly the surface that shipped with that defect once.
    """

    missing = sorted(set(game_learning_runtime.__all__) - set(dir(game_learning_runtime)))
    assert missing == []
    for identifier in (
        "COVERAGE_RATIO_METRIC",
        "PROJECTED_STEPS_METRIC",
        "STATE_ACTION_CELLS_METRIC",
    ):
        assert identifier in game_learning_runtime.__all__
        assert getattr(game_learning_runtime, identifier) == getattr(
            game_learning_runtime.learnability, identifier
        )
    assert (
        game_learning_runtime.STATE_ACTION_CELLS_METRIC
        == STATE_ACTION_CELLS_METRIC
        == "learnability.state_action_cells"
    )
