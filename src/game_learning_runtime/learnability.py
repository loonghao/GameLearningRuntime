"""Learnability budget: state-action cardinality against the observed step rate.

An unattended run will spend its whole budget on a configuration that cannot
converge, and every report still reads healthy. The expensive part is not the
failure; it is that the failure is invisible, so an operator cannot tell "the
configuration is wrong" from "I have not run enough steps yet" and keeps tuning
the wrong variable.

This module closes that gap with two numbers.

``state_action_cells``
    Declared before the run. An adapter declares how it discretizes its state,
    or declares an upper bound, and the runtime multiplies the state bound by
    the action-set size. A function-approximation adapter declares an
    effective capacity instead, which is accepted verbatim rather than
    multiplied.

``coverage_ratio``
    Measured during the run: how much of that space the run actually visited,
    projected forward against the observed step rate and the configured
    budget.

The companion projection, ``projected_steps_to_k_visits``, answers the question
a healthy-looking report never asks: how many steps this configuration needs
before every cell has been visited ``K`` times. Below roughly four visits a
tabular value is a sample, not an estimate, so ``K`` defaults to ``4``.

Opting in is explicit. An adapter that declares nothing, or a caller that
supplies no :class:`LearnabilityPlan`, gets exactly the behaviour it had
before: no tracker, no metrics, no verdict.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from time import time_ns
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from game_learning_runtime.errors import GLRError

if TYPE_CHECKING:  # pragma: no cover - typing-only import; specs imports this module
    from game_learning_runtime.specs import CompositeSpec, TensorSpec

LEARNABILITY_BUDGET_SCHEMA_VERSION = "glr.learnability-budget.v1"

#: Capability string an adapter adds to ``EnvironmentSpec.capabilities``.
LEARNABILITY_CAPABILITY = "learnability-budget-v1"

#: Default number of visits a cell needs before its tabular value is treated
#: as an estimate rather than a sample.
DEFAULT_VISIT_TARGET = 4

#: Coverage floor used when the caller did not configure ``min_coverage``.
#: Half the declared space is the point below which the untouched half is
#: initialization rather than learning, so a run under it warns instead of
#: reading green.
DEFAULT_MIN_COVERAGE = 0.5

#: Steps collected before a negative verdict is trusted. The first few steps
#: cannot distinguish an unlearnable configuration from a cold start.
MIN_EVIDENCE_STEPS = 8

#: ``TimeStep.info`` key an adapter may use to report the state cell it is in.
LEARNABILITY_CELL_KEY = "learnability_cell"

#: Run-store event kind carrying one learnability verdict.
LEARNABILITY_BUDGET_EVENT = "learnability.budget"

#: First-class metric names, so both numbers are readable without a log.
COVERAGE_RATIO_METRIC = "learnability.coverage_ratio"
PROJECTED_STEPS_METRIC = "learnability.projected_steps_to_k_visits"

_MAX_CELLS = 2**62
_MAX_DETAIL = 512

#: String cell identities are offset beyond every integer cell so that a
#: reported integer cell can never be counted as the same cell as a string.
#: ``""`` hashing to ``0`` would otherwise merge with the very common
#: ``cell=0``, and an adapter mixing both kinds would under-report coverage.
_STRING_CELL_OFFSET = 2**63
_LEAF_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class LearnabilityStatus(str, Enum):
    """Verdict carried by a :class:`LearnabilityReport`."""

    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"


class CardinalityKind(str, Enum):
    """How a configuration represents value, which fixes what "cells" means."""

    TABULAR = "tabular"
    FUNCTION_APPROXIMATION = "function-approximation"


class LearnabilityBudgetError(GLRError):
    """Raised when a run cannot reach its coverage floor inside its budget.

    Deliberately not a :class:`~game_learning_runtime.errors.ContractViolation`
    and not a transport error: nothing was violated on the wire and nothing is
    retryable. The configuration is unlearnable at this throughput, and the
    message names the three ways out with the numbers that apply.
    """

    def __init__(self, report: LearnabilityReport) -> None:
        super().__init__(report.error_message())
        self.report = report

    @property
    def remedies(self) -> tuple[str, ...]:
        """The three numbered remediation paths, each carrying its numbers."""

        return self.report.remedies


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} requires string keys")
    return value


def _positive_integer(value: object, *, path: str, maximum: int = _MAX_CELLS) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{path} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{path} must be between 1 and {maximum}")
    return value


def _positive_number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{path} must be finite and positive")
    return result


def _unit_interval(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{path} must be finite and in (0, 1]")
    return result


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _bins(value: object, *, path: str) -> Mapping[str, int]:
    raw = _mapping(value, path=path)
    parsed = {
        _leaf_name(key, path=path): _positive_integer(count, path=f"{path}.{key}")
        for key, count in raw.items()
    }
    return MappingProxyType(parsed)


def _leaf_name(value: str, *, path: str) -> str:
    if _LEAF_NAME.fullmatch(value) is None:
        raise ValueError(f"{path} contains an invalid leaf name: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class SpaceCardinality:
    """Declared size of one discretized space, and how it was obtained.

    ``bins`` is the discretization itself: a mapping of flattened observation
    leaf path to the number of bins that leaf is encoded into. When it is
    supplied the runtime can resolve a cell identity from an observation on its
    own, and ``cells`` must equal the product of the bin counts. When it is
    omitted the adapter has declared a bound only, and cell identity has to
    arrive through the ``learnability_cell`` info key.
    """

    cells: int
    upper_bound: bool = False
    bins: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", _positive_integer(self.cells, path="cardinality.cells"))
        object.__setattr__(self, "bins", _bins(self.bins, path="cardinality.bins"))
        if not isinstance(self.upper_bound, bool):
            raise TypeError("cardinality.upper_bound must be a boolean")
        if self.bins:
            product = 1
            for count in self.bins.values():
                product *= count
            if product != self.cells:
                raise ValueError(
                    f"cardinality.cells is {self.cells} but cardinality.bins multiply to {product}"
                )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SpaceCardinality:
        """Parse one strict cardinality declaration."""

        raw = _mapping(value, path="cardinality")
        unknown = sorted(set(raw) - {"cells", "upper_bound", "bins"})
        if unknown:
            raise ValueError(f"cardinality contains unexpected fields: {unknown}")
        return cls(
            cells=_positive_integer(raw.get("cells"), path="cardinality.cells"),
            upper_bound=_boolean(raw.get("upper_bound", False), path="cardinality.upper_bound"),
            bins=_bins(raw.get("bins", {}), path="cardinality.bins"),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON-safe projection of this declaration."""

        return {
            "cells": self.cells,
            "upper_bound": self.upper_bound,
            "bins": dict(self.bins),
        }


@dataclass(frozen=True, slots=True)
class LearnabilityDeclaration:
    """What an adapter claims about the size of the space it asks to be learned.

    ``effective_capacity`` is the function-approximation escape hatch. A learner
    that does not tabulate the state does not have a cell product, so it
    declares the capacity that plays the same role and that number is used
    verbatim instead of ``state.cells * action.cells``.
    """

    schema_version: str
    kind: CardinalityKind
    state: SpaceCardinality | None = None
    action: SpaceCardinality | None = None
    effective_capacity: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != LEARNABILITY_BUDGET_SCHEMA_VERSION:
            raise ValueError(
                "learnability.schema_version must be "
                f"{LEARNABILITY_BUDGET_SCHEMA_VERSION!r}; "
                f"received {self.schema_version!r}"
            )
        if not isinstance(self.kind, CardinalityKind):
            raise TypeError("learnability.kind must be a CardinalityKind")
        if self.state is not None and not isinstance(self.state, SpaceCardinality):
            raise TypeError("learnability.state must be a SpaceCardinality or None")
        if self.action is not None and not isinstance(self.action, SpaceCardinality):
            raise TypeError("learnability.action must be a SpaceCardinality or None")
        if self.effective_capacity is not None:
            object.__setattr__(
                self,
                "effective_capacity",
                _positive_integer(self.effective_capacity, path="learnability.effective_capacity"),
            )
        if self.state_action_cells is None:
            raise ValueError(
                "learnability requires effective_capacity, or a state cardinality "
                "and an action cardinality"
            )

    @property
    def state_action_cells(self) -> int | None:
        """Cells the learner has to cover, or ``None`` when under-declared."""

        if self.effective_capacity is not None:
            return self.effective_capacity
        if self.state is None or self.action is None:
            return None
        return self.state.cells * self.action.cells

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LearnabilityDeclaration:
        """Parse one strict ``learnability-budget-v1`` declaration."""

        raw = _mapping(value, path="learnability")
        unknown = sorted(
            set(raw) - {"schema_version", "kind", "state", "action", "effective_capacity"}
        )
        if unknown:
            raise ValueError(f"learnability contains unexpected fields: {unknown}")
        try:
            kind = CardinalityKind(raw.get("kind"))
        except (TypeError, ValueError) as error:
            choices = ", ".join(item.value for item in CardinalityKind)
            raise ValueError(f"learnability.kind must be one of: {choices}") from error
        state = raw.get("state")
        action = raw.get("action")
        capacity = raw.get("effective_capacity")
        return cls(
            schema_version=str(raw.get("schema_version")),
            kind=kind,
            state=None if state is None else SpaceCardinality.from_mapping(state),
            action=None if action is None else SpaceCardinality.from_mapping(action),
            effective_capacity=None
            if capacity is None
            else _positive_integer(capacity, path="learnability.effective_capacity"),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON-safe projection of this declaration."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "state": None if self.state is None else self.state.to_mapping(),
            "action": None if self.action is None else self.action.to_mapping(),
            "effective_capacity": self.effective_capacity,
        }


@dataclass(frozen=True, slots=True)
class LearnabilityBudget:
    """The budget a run may spend, and the coverage it has to reach."""

    budget_steps: int
    min_coverage: float | None = None
    budget_seconds: float | None = None
    visit_target: int = DEFAULT_VISIT_TARGET

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "budget_steps", _positive_integer(self.budget_steps, path="budget_steps")
        )
        if self.min_coverage is not None:
            object.__setattr__(
                self, "min_coverage", _unit_interval(self.min_coverage, path="min_coverage")
            )
        if self.budget_seconds is not None:
            object.__setattr__(
                self,
                "budget_seconds",
                _positive_number(self.budget_seconds, path="budget_seconds"),
            )
        object.__setattr__(
            self,
            "visit_target",
            _positive_integer(self.visit_target, path="visit_target", maximum=1_000_000),
        )

    @property
    def threshold(self) -> float:
        """Coverage floor in force: the configured one, or the built-in one."""

        return DEFAULT_MIN_COVERAGE if self.min_coverage is None else self.min_coverage

    @property
    def configured(self) -> bool:
        """Whether the caller opted in to a hard coverage floor."""

        return self.min_coverage is not None

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON-safe projection of this budget."""

        return {
            "budget_steps": self.budget_steps,
            "budget_seconds": self.budget_seconds,
            "min_coverage": self.min_coverage,
            "threshold": self.threshold,
            "visit_target": self.visit_target,
        }


@dataclass(frozen=True, slots=True)
class LearnabilityPlan:
    """Everything the runtime needs to watch a run's learnability budget."""

    budget: LearnabilityBudget
    declaration: LearnabilityDeclaration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.budget, LearnabilityBudget):
            raise TypeError("plan.budget must be a LearnabilityBudget")
        if self.declaration is not None and not isinstance(
            self.declaration, LearnabilityDeclaration
        ):
            raise TypeError("plan.declaration must be a LearnabilityDeclaration or None")


@dataclass(frozen=True, slots=True)
class LearnabilityReport:
    """Cardinality, coverage, and the projection that makes them comparable."""

    schema_version: str
    state_action_cells: int
    distinct_cells_visited: int
    steps: int
    updates: int
    visit_target: int
    coverage_ratio: float
    projected_steps_to_k_visits: int | None
    projected_coverage_at_budget: float | None
    steps_per_second: float | None
    projected_seconds_to_k_visits: float | None
    required_steps_per_second: float | None
    elapsed_seconds: float | None
    budget_steps: int
    budget_seconds: float | None
    min_coverage: float
    coverage_configured: bool
    status: LearnabilityStatus
    unresolved_steps: int
    remedies: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, LearnabilityStatus):
            raise TypeError("status must be a LearnabilityStatus")
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "remedies", tuple(self.remedies))

    @property
    def failed(self) -> bool:
        """Whether this verdict is a hard failure."""

        return self.status is LearnabilityStatus.FAILED

    @property
    def within_budget(self) -> bool:
        """Whether the K-visit target fits in the configured step budget."""

        return (
            self.projected_steps_to_k_visits is not None
            and self.projected_steps_to_k_visits <= self.budget_steps
        )

    def error_message(self) -> str:
        """Return the operator-facing failure text with all three remedies."""

        lines = [
            (
                f"learnability budget exhausted: visited {self.distinct_cells_visited} of "
                f"{self.state_action_cells} state-action cells "
                f"(coverage {self.coverage_ratio:.3f} < {self.min_coverage:.3f}) after "
                f"{self.steps} of {self.budget_steps} budgeted steps"
            ),
            (
                f"projected {_format_steps(self.projected_steps_to_k_visits)} steps to reach "
                f"{self.visit_target} visits per cell; "
                f"observed throughput: {_format_rate(self.steps_per_second)}"
            ),
            "remedies:",
        ]
        lines.extend(f"  {index}. {remedy}" for index, remedy in enumerate(self.remedies, start=1))
        return "\n".join(lines)

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON-safe projection persisted in the run store."""

        return {
            "schema_version": self.schema_version,
            "state_action_cells": self.state_action_cells,
            "distinct_cells_visited": self.distinct_cells_visited,
            "steps": self.steps,
            "updates": self.updates,
            "visit_target": self.visit_target,
            "coverage_ratio": self.coverage_ratio,
            "projected_steps_to_k_visits": self.projected_steps_to_k_visits,
            "projected_coverage_at_budget": self.projected_coverage_at_budget,
            "steps_per_second": self.steps_per_second,
            "projected_seconds_to_k_visits": self.projected_seconds_to_k_visits,
            "required_steps_per_second": self.required_steps_per_second,
            "elapsed_seconds": self.elapsed_seconds,
            "budget_steps": self.budget_steps,
            "budget_seconds": self.budget_seconds,
            "min_coverage": self.min_coverage,
            "coverage_configured": self.coverage_configured,
            "status": self.status.value,
            "unresolved_steps": self.unresolved_steps,
            "remedies": list(self.remedies),
            "notes": list(self.notes),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LearnabilityReport:
        """Rebuild a report persisted by :meth:`to_mapping`."""

        raw = _mapping(value, path="learnability report")
        schema_version = raw.get("schema_version")
        if schema_version != LEARNABILITY_BUDGET_SCHEMA_VERSION:
            raise ValueError(
                "learnability report.schema_version must be "
                f"{LEARNABILITY_BUDGET_SCHEMA_VERSION!r}; received {schema_version!r}"
            )
        return cls(
            schema_version=str(schema_version),
            state_action_cells=_positive_integer(
                raw.get("state_action_cells"), path="state_action_cells"
            ),
            distinct_cells_visited=_non_negative_integer(
                raw.get("distinct_cells_visited"), path="distinct_cells_visited"
            ),
            steps=_non_negative_integer(raw.get("steps"), path="steps"),
            updates=_non_negative_integer(raw.get("updates"), path="updates"),
            visit_target=_positive_integer(
                raw.get("visit_target"), path="visit_target", maximum=1_000_000
            ),
            coverage_ratio=_ratio(raw.get("coverage_ratio"), path="coverage_ratio"),
            projected_steps_to_k_visits=_optional_positive_integer(
                raw.get("projected_steps_to_k_visits"), path="projected_steps_to_k_visits"
            ),
            projected_coverage_at_budget=_optional_ratio(
                raw.get("projected_coverage_at_budget"), path="projected_coverage_at_budget"
            ),
            steps_per_second=_optional_positive_number(
                raw.get("steps_per_second"), path="steps_per_second"
            ),
            projected_seconds_to_k_visits=_optional_positive_number(
                raw.get("projected_seconds_to_k_visits"), path="projected_seconds_to_k_visits"
            ),
            required_steps_per_second=_optional_positive_number(
                raw.get("required_steps_per_second"), path="required_steps_per_second"
            ),
            elapsed_seconds=_optional_positive_number(
                raw.get("elapsed_seconds"), path="elapsed_seconds"
            ),
            budget_steps=_positive_integer(raw.get("budget_steps"), path="budget_steps"),
            budget_seconds=_optional_positive_number(
                raw.get("budget_seconds"), path="budget_seconds"
            ),
            min_coverage=_unit_interval(raw.get("min_coverage"), path="min_coverage"),
            coverage_configured=_boolean(
                raw.get("coverage_configured"), path="coverage_configured"
            ),
            status=_status(raw.get("status")),
            unresolved_steps=_non_negative_integer(
                raw.get("unresolved_steps"), path="unresolved_steps"
            ),
            remedies=_strings(raw.get("remedies"), path="remedies"),
            notes=_strings(raw.get("notes"), path="notes"),
        )


class StateCellResolver:
    """Turn one observation into the declared state-cell identity.

    Bins are applied in sorted leaf order and combined in mixed radix, so the
    same observation always maps to the same cell and two different cells never
    collide. Discrete and binary leaves index directly; continuous leaves are
    quantized between their declared bounds.
    """

    def __init__(
        self,
        bins: Mapping[str, int],
        *,
        observation_spec: CompositeSpec | None = None,
    ) -> None:
        if not bins:
            raise ValueError("bins cannot be empty")
        self._paths = tuple(sorted(bins))
        self._bins = tuple(int(bins[path]) for path in self._paths)
        self._bounds: tuple[tuple[float, float, str], ...] = ()
        if observation_spec is not None:
            flattened = observation_spec.flatten()
            declared = set(self._paths)
            unknown = sorted(declared - set(flattened))
            if unknown:
                raise ValueError(f"learnability bins reference unknown leaves: {unknown}")
            bounds: list[tuple[float, float, str]] = []
            for path in self._paths:
                bounds.append(_leaf_bounds(flattened[path], path=path))
            self._bounds = tuple(bounds)

    @property
    def paths(self) -> tuple[str, ...]:
        """Flattened observation leaves this resolver reads, in index order."""

        return self._paths

    def resolve(self, observation: Mapping[str, Any]) -> int:
        """Return the cell identity of one observation tree."""

        cell = 0
        for index, path in enumerate(self._paths):
            value = observation
            for name in path.split("."):
                if not isinstance(value, Mapping) or name not in value:
                    raise KeyError(f"observation is missing learnability leaf {path!r}")
                value = value[name]
            array = np.asarray(value).reshape(-1)
            if array.size != 1:
                raise ValueError(
                    f"learnability leaf {path!r} must hold exactly one value; received {array.size}"
                )
            cell = cell * self._bins[index] + self._bin_index(array[0], index=index, path=path)
        return cell

    def _bin_index(self, value: Any, *, index: int, path: str) -> int:
        bins = self._bins[index]
        if not self._bounds:
            raw = int(value)
            if not 0 <= raw < bins:
                raise ValueError(f"learnability leaf {path!r} value {raw} is outside {bins} bins")
            return raw
        low, high, kind = self._bounds[index]
        number = float(value)
        if kind == "continuous":
            span = high - low
            if span <= 0:
                return 0
            normalized = min(max((number - low) / span, 0.0), 1.0)
            return min(int(normalized * bins), bins - 1)
        discrete = int(number) - int(low)
        if not 0 <= discrete < bins:
            raise ValueError(
                f"learnability leaf {path!r} value {int(number)} is outside {bins} bins "
                f"from {int(low)}"
            )
        return discrete


class LearnabilityTracker:
    """Accumulate coverage evidence and judge it against a declared budget.

    The tracker is cumulative across collection calls and never raises on its
    own: :meth:`require` is the opt-in fail-fast, and it raises as soon as the
    projection says the remaining budget cannot reach the coverage floor, so the
    run stops before the budget is gone rather than after.
    """

    def __init__(
        self,
        declaration: LearnabilityDeclaration,
        budget: LearnabilityBudget,
        *,
        resolver: StateCellResolver | None = None,
    ) -> None:
        if not isinstance(declaration, LearnabilityDeclaration):
            raise TypeError("declaration must be a LearnabilityDeclaration")
        if not isinstance(budget, LearnabilityBudget):
            raise TypeError("budget must be a LearnabilityBudget")
        cells = declaration.state_action_cells
        if cells is None:
            raise ValueError("declaration does not resolve to a state-action cell count")
        self._declaration = declaration
        self._budget = budget
        self._cells = cells
        self._resolver = resolver
        self._visited: set[int] = set()
        self._steps = 0
        self._updates = 0
        self._unresolved = 0
        self._elapsed_seconds = 0.0
        self._started_ns: int | None = None
        self._last_ns: int | None = None

    @property
    def declaration(self) -> LearnabilityDeclaration:
        return self._declaration

    @property
    def budget(self) -> LearnabilityBudget:
        return self._budget

    @property
    def state_action_cells(self) -> int:
        """Cells the learner has to cover."""

        return self._cells

    @property
    def distinct_cells_visited(self) -> int:
        """Cells seen at least once so far."""

        return len(self._visited)

    @property
    def steps(self) -> int:
        """Steps observed so far."""

        return self._steps

    @property
    def updates(self) -> int:
        """Learner updates recorded so far."""

        return self._updates

    def observe(
        self,
        *,
        cell: int | str | None = None,
        observation: Mapping[str, Any] | None = None,
        steps: int = 1,
        updates: int = 0,
        elapsed_seconds: float | None = None,
        now_ns: int | None = None,
    ) -> LearnabilityReport:
        """Record one or more steps and return the current verdict.

        Pass ``cell`` directly, or pass ``observation`` and let a resolver built
        from the declared bins derive it. A step whose cell cannot be resolved
        still counts toward the budget: an untracked step is not free.
        """

        # Validated before anything is charged, so a rejected observation cannot
        # leave the tracker half-updated.
        count = _non_negative_integer(steps, path="steps")
        update_count = _non_negative_integer(updates, path="updates")
        timestamp = _timestamp(now_ns)
        resolved = cell
        if resolved is None and observation is not None and self._resolver is not None:
            try:
                resolved = self._resolver.resolve(observation)
            except (KeyError, ValueError):
                # The declared bins do not describe this observation, so the
                # step has no cell. It is still charged: an untracked step is
                # not free. This is a declared-configuration mismatch rather
                # than a caller bug, so it must not escape collect() as a bare
                # KeyError or ValueError that no caller can handle as GLRError.
                resolved = None
        if resolved is not None:
            resolved = coerce_cell_identity(resolved)
            if resolved is None:
                # Not an unresolvable step but a caller mistake: charging it as
                # unresolved would hide it behind a coverage number.
                raise TypeError("cell must be an integer, a string, or None")
            key = resolved if isinstance(resolved, int) else _string_cell(resolved)
            self._visited.add(key)
        elif count:
            self._unresolved += count
        self._steps += count
        self._updates += update_count
        if self._started_ns is None:
            self._started_ns = timestamp
        if elapsed_seconds is not None:
            self._elapsed_seconds += _positive_number(elapsed_seconds, path="elapsed_seconds")
        elif self._last_ns is not None and timestamp > self._last_ns:
            self._elapsed_seconds += (timestamp - self._last_ns) / 1e9
        self._last_ns = timestamp
        return self.report()

    def note_updates(self, updates: int) -> LearnabilityReport:
        """Record learner updates that happened outside step collection."""

        return self.observe(steps=0, updates=updates)

    def report(self) -> LearnabilityReport:
        """Return the current verdict without raising."""

        steps = self._steps
        distinct = len(self._visited)
        cells = self._cells
        coverage = min(distinct / cells, 1.0) if cells else 0.0
        elapsed = self._elapsed_seconds if self._elapsed_seconds > 0 else None
        rate = steps / elapsed if elapsed else None
        threshold = self._budget.threshold
        efficiency = _coverage_efficiency(distinct=distinct, steps=steps, cells=cells)
        projected_steps = _projected_steps_to_k_visits(
            cells=cells,
            visit_target=self._budget.visit_target,
            efficiency=efficiency,
        )
        projected_coverage = _projected_coverage(
            distinct=distinct,
            steps=steps,
            cells=cells,
            budget_steps=self._budget.budget_steps,
            efficiency=efficiency,
        )
        projected_seconds = (
            projected_steps / rate if projected_steps is not None and rate is not None else None
        )
        required_rate = (
            projected_steps / self._budget.budget_seconds
            if projected_steps is not None and self._budget.budget_seconds is not None
            else None
        )
        notes: list[str] = []
        if steps and distinct == 0:
            notes.append(
                "no state cell was resolved; declare discretization bins or emit "
                f"info[{LEARNABILITY_CELL_KEY!r}] so coverage can be measured"
            )
        if self._declaration.effective_capacity is not None:
            notes.append("coverage is measured against a declared effective capacity")
        elif self._declaration.state is not None and self._declaration.state.upper_bound:
            notes.append("coverage is measured against a declared upper bound")
        status = _status_for(
            coverage=coverage,
            threshold=threshold,
            configured=self._budget.configured,
            distinct=distinct,
            steps=steps,
        )
        report = LearnabilityReport(
            schema_version=LEARNABILITY_BUDGET_SCHEMA_VERSION,
            state_action_cells=cells,
            distinct_cells_visited=distinct,
            steps=steps,
            updates=self._updates,
            visit_target=self._budget.visit_target,
            coverage_ratio=coverage,
            projected_steps_to_k_visits=projected_steps,
            projected_coverage_at_budget=projected_coverage,
            steps_per_second=rate,
            projected_seconds_to_k_visits=projected_seconds,
            required_steps_per_second=required_rate,
            elapsed_seconds=elapsed,
            budget_steps=self._budget.budget_steps,
            budget_seconds=self._budget.budget_seconds,
            min_coverage=threshold,
            coverage_configured=self._budget.configured,
            status=status,
            unresolved_steps=self._unresolved,
            remedies=(),
            notes=tuple(notes),
        )
        remedies = _remedies(report, efficiency=efficiency, observed_rate=rate)
        object.__setattr__(report, "remedies", remedies)
        return report

    def require(self) -> LearnabilityReport:
        """Fail fast when the budget cannot reach the coverage floor.

        Only raises when the caller configured ``min_coverage``, when enough
        steps have been collected to trust the projection, and when the
        remaining budget genuinely cannot close the gap. Everything else
        returns the same report :meth:`report` would, so a caller can use this
        as its only call site.
        """

        report = self.report()
        if not self._budget.configured or report.status is not LearnabilityStatus.FAILED:
            return report
        if self._steps < min(MIN_EVIDENCE_STEPS, self._budget.budget_steps):
            return report
        projected = report.projected_coverage_at_budget
        if projected is not None and projected >= self._budget.threshold:
            return report
        raise LearnabilityBudgetError(report)


def build_tracker(
    plan: LearnabilityPlan,
    *,
    declaration: LearnabilityDeclaration | None = None,
    observation_spec: CompositeSpec | None = None,
) -> LearnabilityTracker:
    """Build a tracker from a plan, resolving the declaration and cell source.

    An explicit ``declaration`` wins over the plan's. When the state
    declaration carries bins, a :class:`StateCellResolver` is built from them so
    the runtime can derive cell identities from observations alone.
    """

    if not isinstance(plan, LearnabilityPlan):
        raise TypeError("plan must be a LearnabilityPlan")
    resolved = declaration or plan.declaration
    if resolved is None:
        raise ValueError("learnability requires a declaration")
    resolver: StateCellResolver | None = None
    state = resolved.state
    if state is not None and state.bins:
        if observation_spec is None:
            raise ValueError(
                "learnability bins require an observation spec to resolve cell identities"
            )
        resolver = StateCellResolver(state.bins, observation_spec=observation_spec)
    return LearnabilityTracker(resolved, plan.budget, resolver=resolver)


def derive_state_cardinality(observation: CompositeSpec) -> SpaceCardinality | None:
    """Return the cell count an observation spec implies, or ``None``.

    Convenience for an adapter that wants to declare the bound its own contract
    already implies. A continuous or unbounded leaf makes the product unknown,
    which is the honest answer rather than a guess.
    """

    return _derive_cardinality(observation)


def derive_action_cardinality(action: CompositeSpec) -> SpaceCardinality | None:
    """Return the action-set size an action spec implies, or ``None``."""

    return _derive_cardinality(action)


def _derive_cardinality(spec: CompositeSpec) -> SpaceCardinality | None:
    flattened = spec.flatten()
    product = 1
    for _path, leaf in flattened.items():
        count = _leaf_cardinality(leaf)
        if count is None:
            return None
        product *= count
        if product > _MAX_CELLS:
            return None
    if product < 1:
        return None
    return SpaceCardinality(cells=product, upper_bound=True)


def _leaf_cardinality(leaf: TensorSpec) -> int | None:
    kind = _leaf_kind(leaf)
    if kind == "binary":
        return 2
    if kind not in {"discrete", "multi_discrete"}:
        return None
    if leaf.minimum is None or leaf.maximum is None or leaf.is_dynamic:
        return None
    # ``is_dynamic`` already rejected every ``None`` dimension, so the extent is
    # finite here and the product is over the declared extents only.
    shape = cast("tuple[int, ...]", leaf.shape)
    try:
        low = np.broadcast_to(np.asarray(leaf.minimum, dtype=np.int64), shape)
        high = np.broadcast_to(np.asarray(leaf.maximum, dtype=np.int64), shape)
    except ValueError:
        return None
    if np.any(low > high):
        return None
    total = int(np.prod(high - low + 1))
    return None if total < 1 or total > _MAX_CELLS else total


def _leaf_bounds(leaf: TensorSpec, *, path: str) -> tuple[float, float, str]:
    kind = _leaf_kind(leaf)
    if kind == "binary":
        return 0.0, 1.0, kind
    if kind in {"discrete", "multi_discrete"}:
        low = 0.0 if leaf.minimum is None else float(np.min(np.asarray(leaf.minimum)))
        high = low if leaf.maximum is None else float(np.max(np.asarray(leaf.maximum)))
        return low, high, kind
    minimum = leaf.minimum
    maximum = leaf.maximum
    if minimum is None or maximum is None:
        raise ValueError(f"learnability leaf {path!r} needs bounds to be quantized")
    return (
        float(np.min(np.asarray(minimum))),
        float(np.max(np.asarray(maximum))),
        "continuous",
    )


def _leaf_kind(leaf: TensorSpec) -> str:
    """Return the space-kind value without importing ``specs`` at run time.

    ``specs`` imports this module, so ``TensorSpec`` is only available for
    typing; the kind enum is read through its value instead.
    """

    kind = leaf.kind
    value = getattr(kind, "value", kind)
    return str(value)


def _coverage_efficiency(*, distinct: int, steps: int, cells: int) -> float | None:
    """New cells discovered per step, as a fraction of what was discoverable.

    The denominator is every step taken, not ``min(steps, cells)``. A bound of
    ``cells`` would stop the ratio decaying once ``steps`` passes ``cells``: on a
    long run the efficiency would freeze at the coverage ratio instead of
    falling, the projected step count would shrink below the steps already
    spent, and the fail-fast gate would stay on its permissive branch forever.
    ``min(..., 1.0)`` covers ``distinct > steps``, which cannot happen but is
    cheap to bound.
    """

    if steps <= 0:
        return None
    return min(distinct / steps, 1.0)


def _projected_steps_to_k_visits(
    *, cells: int, visit_target: int, efficiency: float | None
) -> int | None:
    """Steps needed for every cell to be visited ``visit_target`` times.

    ``visit_target * cells`` steps would suffice if every step landed on a cell
    that still needed a visit. Only a fraction of steps actually discover a new
    cell, so that floor is divided by the observed discovery efficiency.
    """

    if efficiency is None or efficiency <= 0:
        return None
    return min(math.ceil(visit_target * cells / efficiency), _MAX_CELLS)


def _projected_coverage(
    *, distinct: int, steps: int, cells: int, budget_steps: int, efficiency: float | None
) -> float | None:
    """Coverage the run would reach if it spent the whole budget at this rate."""

    if efficiency is None:
        return None
    remaining = max(0, budget_steps - steps)
    return min((distinct + remaining * efficiency) / cells, 1.0)


def _status_for(
    *, coverage: float, threshold: float, configured: bool, distinct: int, steps: int
) -> LearnabilityStatus:
    if steps <= 0 or (distinct == 0 and steps > 0):
        return LearnabilityStatus.WARNING if steps > 0 else LearnabilityStatus.OK
    if coverage >= threshold:
        return LearnabilityStatus.OK
    return LearnabilityStatus.FAILED if configured else LearnabilityStatus.WARNING


def _remedies(
    report: LearnabilityReport,
    *,
    efficiency: float | None,
    observed_rate: float | None,
) -> tuple[str, ...]:
    """Return the three remediation paths, each carrying its own numbers."""

    cells = report.state_action_cells
    budget_steps = report.budget_steps
    threshold = report.min_coverage
    target = report.visit_target
    projected = report.projected_steps_to_k_visits
    projected_text = _format_steps(projected)
    observed = _format_rate(observed_rate)
    needed = "unknown" if projected is None else f"{projected:,}"
    usable_efficiency = efficiency if efficiency and efficiency > 0 else 1.0
    max_cells = max(1, math.floor(budget_steps * usable_efficiency / threshold))
    cardinality = (
        f"shrink the declared state-action space from {cells:,} to at most {max_cells:,} "
        f"cells (coarser state bins, fewer state features, or a shorter horizon); "
        f"{cells:,} cells needs about {needed} steps for {target} visits per cell"
    )
    approximation = (
        f"replace the tabular encoding with a function approximator and declare its "
        f"effective capacity on the {LEARNABILITY_CAPABILITY} declaration instead of the "
        f"{cells:,}-cell product; a table over {cells:,} cells stays a sample collection "
        f"below {target} visits per cell"
    )
    if report.budget_seconds is not None:
        required = report.required_steps_per_second
        throughput = (
            f"raise throughput from {observed} to at least {required:,.1f} steps/s "
            f"({projected_text} steps inside the {report.budget_seconds:,.0f}s budget)"
            if required is not None
            else (
                f"raise throughput from {observed} so {projected_text} steps fit inside "
                f"the {report.budget_seconds:,.0f}s budget"
            )
        )
    else:
        ratio = projected / budget_steps if projected is not None and budget_steps else None
        throughput = (
            f"raise throughput from {observed} and give the run at least "
            f"{projected_text} steps ({ratio:,.1f}x the {budget_steps:,}-step budget) "
            f"for {target} visits per cell; declare budget_seconds to have this "
            f"expressed as a steps/s floor"
            if ratio is not None
            else (
                f"raise throughput from {observed} and give the run at least "
                f"{projected_text} steps for {target} visits per cell; declare "
                f"budget_seconds to have this expressed as a steps/s floor"
            )
        )
    return (cardinality, approximation, throughput)


def _format_steps(value: int | None) -> str:
    return "unknown" if value is None else f"{value:,}"


def _format_rate(value: float | None) -> str:
    return "unknown" if value is None else f"{value:,.1f} steps/s"


def _timestamp(value: int | None) -> int:
    if value is None:
        return time_ns()
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("now_ns must be a non-negative integer or None")
    return value


def _string_cell(value: str) -> int:
    """Project one string cell identity into the tracker's integer key space.

    The digest is a keyed-free BLAKE2b, so it is stable across processes and
    interpreter runs -- unlike :func:`hash`, which is salted per process and
    would make the same run report different coverage on a restart. The
    previous position-weighted sum was not injective: ``"ab"`` and ``"ca"``
    both summed to 293 and were counted as one cell.
    """

    if len(value) > _MAX_DETAIL:
        raise ValueError("cell identity cannot exceed 512 characters")
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return _STRING_CELL_OFFSET + int.from_bytes(digest, "big") % _MAX_CELLS


def coerce_cell_identity(value: object) -> int | str | None:
    """Normalize one reported state-cell identity to a Python native value.

    A ``numpy`` integer is not a Python ``int`` and an observation is a numpy
    array, so ``info[LEARNABILITY_CELL_KEY] = observation[0]`` is the natural
    way for an adapter to report the cell a step landed in. Treating that
    scalar as unresolvable charges every step to no cell: coverage stays at
    zero, the floor never fires, and the note tells the operator to emit the
    key they already emit. Integers of any flavour and strings resolve; a
    value that is neither is not a cell identity, so it resolves to ``None``
    and the step is charged as unresolved.
    """

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        # A zero-dimensional array or a scalar wrapper. A multi-element array
        # raises here, and it is not one cell either.
        try:
            return coerce_cell_identity(item())
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def _non_negative_integer(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _optional_positive_integer(value: object, *, path: str) -> int | None:
    return None if value is None else _positive_integer(value, path=path)


def _optional_positive_number(value: object, *, path: str) -> float | None:
    return None if value is None else _positive_number(value, path=path)


def _ratio(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} must be finite and between 0 and 1")
    return result


def _optional_ratio(value: object, *, path: str) -> float | None:
    return None if value is None else _ratio(value, path=path)


def _status(value: object) -> LearnabilityStatus:
    try:
        return LearnabilityStatus(value)
    except (TypeError, ValueError) as error:
        choices = ", ".join(item.value for item in LearnabilityStatus)
        raise ValueError(f"status must be one of: {choices}") from error


def _strings(value: object, *, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{path} must be an array of strings")
    return tuple(value)


__all__ = [
    "COVERAGE_RATIO_METRIC",
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_VISIT_TARGET",
    "LEARNABILITY_BUDGET_EVENT",
    "LEARNABILITY_BUDGET_SCHEMA_VERSION",
    "LEARNABILITY_CAPABILITY",
    "LEARNABILITY_CELL_KEY",
    "MIN_EVIDENCE_STEPS",
    "PROJECTED_STEPS_METRIC",
    "CardinalityKind",
    "LearnabilityBudget",
    "LearnabilityBudgetError",
    "LearnabilityDeclaration",
    "LearnabilityPlan",
    "LearnabilityReport",
    "LearnabilityStatus",
    "LearnabilityTracker",
    "SpaceCardinality",
    "StateCellResolver",
    "build_tracker",
    "coerce_cell_identity",
    "derive_action_cardinality",
    "derive_state_cardinality",
]
