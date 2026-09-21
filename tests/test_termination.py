"""Episode termination contract (GLR #156) and its absorbing outcome (GLR #155).

#156: every episode must record why it ended. #155: an ``indeterminate``
action outcome is distinct from ``rejected`` and ends the episode at once.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from game_learning_runtime import (
    ActionOutcome,
    ActionReceipt,
    EpisodeCaps,
    EpisodeClosedError,
    EpisodeProgress,
    EpisodeTermination,
    EpisodeTerminationGuard,
    IndeterminateOutcomeError,
    MissingTerminationReason,
    SyncCollector,
    TerminationReason,
    TrainingStore,
    attribute_reason,
)
from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.run_store import EPISODE_TERMINATION_EVENT, RunStatus
from game_learning_runtime.specs import CompositeSpec, EnvironmentSpec, SpaceKind, TensorSpec
from game_learning_runtime.termination import TERMINATION_REASON_KEY

OBSERVATION = CompositeSpec({"position": TensorSpec((1,), np.int64, minimum=0, maximum=8)})
ACTION = CompositeSpec(
    {"choice": TensorSpec((1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=1)}
)


def _spec(caps: EpisodeCaps | None = None) -> EnvironmentSpec:
    return EnvironmentSpec(
        environment_id="termination.contract-v1",
        observation=OBSERVATION,
        action=ACTION,
        episode_caps=caps,
    )


class ScriptedTerminationEnvironment(GameEnvironment):
    """End every episode at ``episode_length`` steps with a scripted info payload."""

    def __init__(
        self,
        *,
        episode_length: int = 2,
        info: Mapping[str, Any] | None = None,
        caps: EpisodeCaps | None = None,
        outcome: ActionOutcome | None = None,
        outcome_at_step: int = 1,
    ) -> None:
        self._episode_length = episode_length
        self._info = dict(info or {})
        self._caps = caps
        self._outcome = outcome
        self._outcome_at_step = outcome_at_step
        self._episode_id = uuid4()
        self._step_id = 0
        self.closed = False

    @property
    def spec(self) -> EnvironmentSpec:
        return _spec(self._caps)

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        del seed, options
        self._episode_id = uuid4()
        self._step_id = 0
        return self._timestep()

    def step(self, action: TensorTree) -> TimeStep:
        del action
        self._step_id += 1
        return self._timestep()

    def close(self) -> None:
        self.closed = True

    def _timestep(self) -> TimeStep:
        boundary = self._step_id >= self._episode_length
        receipt = None
        if self._outcome is not None and self._step_id == self._outcome_at_step:
            receipt = ActionReceipt(
                action_id=f"step-{self._step_id}",
                episode_id=self._episode_id,
                step_id=self._step_id,
                outcome=self._outcome,
                issued_timestamp_ns=self._step_id,
                observed_timestamp_ns=self._step_id + 1,
                postcondition="unknown",
                authoritative_observation_sequence=self._step_id,
            )
        return TimeStep(
            observation={"position": np.array([self._step_id], dtype=np.int64)},
            reward=np.array([0.25], dtype=np.float32),
            terminated=np.array([boundary], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
            episode_id=self._episode_id,
            step_id=self._step_id,
            action_receipt=receipt,
            info=dict(self._info) if boundary else {},
            timestamp_ns=self._step_id * 1_000_000,
        )


def _always_increment(timestep: TimeStep) -> TensorTree:
    del timestep
    return {"choice": np.array([1], dtype=np.int64)}


# --- the closed enum --------------------------------------------------------


def test_every_reason_round_trips_through_the_manifest() -> None:
    for reason in TerminationReason:
        termination = EpisodeTermination(
            episode_id=UUID(int=7),
            reason=reason,
            step_id=3,
            timestamp_ns=11,
            detail=f"why {reason.value}",
            last_known_sequence=9,
        )
        restored = EpisodeTermination.from_mapping(termination.to_mapping())
        assert restored == termination
        assert restored.reason is reason


def test_the_enum_is_closed() -> None:
    assert {item.value for item in TerminationReason} == {
        "goal_reached",
        "failed",
        "step_budget",
        "time_budget",
        "death_cap",
        "stalled",
        "env_frozen",
        "host_unavailable",
        "env_indeterminate",
        "caller_aborted",
    }


def test_an_unknown_reason_is_rejected_instead_of_stored() -> None:
    with pytest.raises(ValueError, match="unsupported termination_reason"):
        EpisodeTermination(episode_id=UUID(int=1), reason="because", step_id=0)  # type: ignore[arg-type]


def test_manifest_rejects_a_foreign_schema() -> None:
    payload = EpisodeTermination(UUID(int=1), TerminationReason.FAILED).to_mapping()
    payload["schema_version"] = "glr.episode-termination.v99"
    with pytest.raises(ValueError, match="unsupported episode termination schema"):
        EpisodeTermination.from_mapping(payload)


# --- a missing reason is a contract violation --------------------------------


def test_closing_without_a_reason_raises_a_typed_error_naming_the_field() -> None:
    guard = EpisodeTerminationGuard(uuid4())
    with pytest.raises(MissingTerminationReason) as raised:
        guard.close()
    assert raised.value.field == TERMINATION_REASON_KEY
    assert TERMINATION_REASON_KEY in str(raised.value)
    assert raised.value.to_mapping()["field"] == TERMINATION_REASON_KEY


def test_the_typed_error_is_catchable_as_a_glr_error() -> None:
    from game_learning_runtime.errors import GLRError

    guard = EpisodeTerminationGuard(uuid4())
    with pytest.raises(GLRError):
        guard.close()


def test_a_collector_raises_rather_than_recording_an_unexplained_episode() -> None:
    environment = ScriptedTerminationEnvironment(episode_length=2)
    with pytest.raises(MissingTerminationReason, match="termination_reason"):
        SyncCollector(environment).collect(_always_increment, steps=4)
    assert environment.closed is False


# --- runtime attribution from declared caps ----------------------------------


def test_a_step_budget_cap_is_attributed_by_the_runtime() -> None:
    caps = EpisodeCaps(max_steps=2)
    assert attribute_reason(caps, EpisodeProgress(steps=2, elapsed_ns=1)) is (
        TerminationReason.STEP_BUDGET
    )
    assert attribute_reason(caps, EpisodeProgress(steps=1, elapsed_ns=1)) is None


def test_a_time_budget_cap_is_attributed_by_the_runtime() -> None:
    caps = EpisodeCaps(max_time_ns=100)
    assert attribute_reason(caps, EpisodeProgress(steps=1, elapsed_ns=100)) is (
        TerminationReason.TIME_BUDGET
    )
    assert attribute_reason(caps, EpisodeProgress(steps=1, elapsed_ns=99)) is None


def test_a_death_cap_is_attributed_by_the_runtime() -> None:
    caps = EpisodeCaps(death_cap=3)
    assert attribute_reason(caps, EpisodeProgress(steps=1, elapsed_ns=1, deaths=3)) is (
        TerminationReason.DEATH_CAP
    )
    assert attribute_reason(caps, EpisodeProgress(steps=1, elapsed_ns=1, deaths=2)) is None


def test_a_stall_cap_is_attributed_by_the_runtime() -> None:
    caps = EpisodeCaps(stall_steps=2)
    assert attribute_reason(caps, EpisodeProgress(steps=9, elapsed_ns=1, stall_steps=2)) is (
        TerminationReason.STALLED
    )
    assert attribute_reason(caps, EpisodeProgress(steps=9, elapsed_ns=1, stall_steps=1)) is None


def test_a_death_cap_outranks_a_step_budget_so_attribution_is_reproducible() -> None:
    caps = EpisodeCaps(max_steps=2, death_cap=1)
    progress = EpisodeProgress(steps=2, elapsed_ns=1, deaths=1)
    assert attribute_reason(caps, progress) is TerminationReason.DEATH_CAP


def test_no_declared_caps_means_no_attribution() -> None:
    assert attribute_reason(None, EpisodeProgress(steps=99, elapsed_ns=99, deaths=99)) is None


def test_a_collector_attributes_each_cap_from_the_environment_spec() -> None:
    expectations = [
        (EpisodeCaps(max_steps=2), TerminationReason.STEP_BUDGET, {}),
        (EpisodeCaps(max_time_ns=2_000_000), TerminationReason.TIME_BUDGET, {}),
        (EpisodeCaps(death_cap=1), TerminationReason.DEATH_CAP, {"episode_deaths": 1}),
        (EpisodeCaps(stall_steps=2), TerminationReason.STALLED, {"episode_stall_steps": 2}),
    ]
    for caps, expected, info in expectations:
        environment = ScriptedTerminationEnvironment(episode_length=2, caps=caps, info=info)
        collector = SyncCollector(environment)
        collector.collect(_always_increment, steps=4)
        termination = collector.last_termination()
        assert termination is not None
        assert termination.reason is expected
        assert termination.attributed_by == "runtime"
        assert termination.step_id == 2


# --- adapter-declared and caller-supplied reasons ----------------------------


def test_an_adapter_declared_reason_is_validated_and_credited_to_the_adapter() -> None:
    environment = ScriptedTerminationEnvironment(
        episode_length=2, info={TERMINATION_REASON_KEY: "goal_reached"}
    )
    collector = SyncCollector(environment)
    collector.collect(_always_increment, steps=4)
    termination = collector.last_termination()
    assert termination is not None
    assert termination.reason is TerminationReason.GOAL_REACHED
    assert termination.attributed_by == "adapter"
    assert termination.reached_goal is True


def test_an_unknown_adapter_reason_is_a_contract_violation() -> None:
    environment = ScriptedTerminationEnvironment(
        episode_length=2, info={TERMINATION_REASON_KEY: "ran-out-of-ideas"}
    )
    with pytest.raises(ValueError, match="unsupported termination_reason"):
        SyncCollector(environment).collect(_always_increment, steps=4)


def test_a_caller_supplied_reason_wins_over_attribution() -> None:
    guard = EpisodeTerminationGuard(uuid4(), caps=EpisodeCaps(max_steps=1), now_ns=0)
    guard.note_step(1, now_ns=1)
    termination = guard.close(reason=TerminationReason.HOST_UNAVAILABLE, now_ns=2)
    assert termination.reason is TerminationReason.HOST_UNAVAILABLE
    assert termination.attributed_by == "caller"


def test_host_unavailable_and_env_frozen_are_reachable_from_the_runtime() -> None:
    for reason in (TerminationReason.HOST_UNAVAILABLE, TerminationReason.ENV_FROZEN):
        guard = EpisodeTerminationGuard(uuid4(), now_ns=0)
        assert guard.close(reason=reason, now_ns=1).reason is reason


def test_a_caller_abort_records_why_the_episode_stopped() -> None:
    guard = EpisodeTerminationGuard(uuid4(), now_ns=0)
    guard.note_step(4, now_ns=1)
    termination = guard.close(reason=TerminationReason.CALLER_ABORTED, now_ns=2)
    assert termination.reason is TerminationReason.CALLER_ABORTED
    assert termination.step_id == 4
    assert termination.to_mapping()["termination_reason"] == "caller_aborted"


# --- the caller predicate ----------------------------------------------------


def test_reached_goal_is_the_predicate_a_scheduler_asks_for() -> None:
    reached = EpisodeTerminationGuard(uuid4(), now_ns=0)
    reached.close(reason=TerminationReason.GOAL_REACHED, now_ns=1)
    assert reached.reached_goal() is True

    truncated = EpisodeTerminationGuard(uuid4(), caps=EpisodeCaps(max_steps=1), now_ns=0)
    truncated.note_step(1, now_ns=1)
    truncated.close(now_ns=2)
    assert truncated.reached_goal() is False

    open_guard = EpisodeTerminationGuard(uuid4(), now_ns=0)
    assert open_guard.reached_goal() is False


# --- no step after termination enters the dataset ----------------------------


def test_a_step_after_the_episode_ended_is_refused() -> None:
    guard = EpisodeTerminationGuard(uuid4(), now_ns=0)
    guard.note_step(1, now_ns=1)
    guard.close(reason=TerminationReason.GOAL_REACHED, now_ns=2)
    with pytest.raises(EpisodeClosedError) as raised:
        guard.note_step(2, now_ns=3)
    assert raised.value.field == TERMINATION_REASON_KEY
    assert raised.value.reason is TerminationReason.GOAL_REACHED
    assert guard.records_step() is False


def test_the_collector_stops_at_the_boundary_and_never_records_a_later_step() -> None:
    environment = ScriptedTerminationEnvironment(episode_length=2, caps=EpisodeCaps(max_steps=2))
    collector = SyncCollector(environment)
    unroll = collector.collect(_always_increment, steps=6, stop_on_done=True)
    assert [transition.step_id for transition in unroll.transitions] == [0, 1]
    assert len(collector.terminations) == 1


def test_closing_twice_returns_the_one_terminal_state() -> None:
    guard = EpisodeTerminationGuard(uuid4(), now_ns=0)
    first = guard.close(reason=TerminationReason.GOAL_REACHED, now_ns=1)
    assert guard.close(reason=TerminationReason.FAILED, now_ns=2) == first


# --- the three readable surfaces ---------------------------------------------


def test_the_run_store_persists_and_returns_terminations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="termination.contract-v1", protocol_version="1.0", kind="training"
    )
    termination = EpisodeTermination(
        episode_id=UUID(int=5),
        reason=TerminationReason.DEATH_CAP,
        step_id=12,
        timestamp_ns=99,
        detail="three deaths",
        last_known_sequence=11,
    )
    event = store.record_episode_termination(run.run_id, termination)
    assert event.kind == EPISODE_TERMINATION_EVENT
    # Episode UUIDs may start with a digit, so the column holds a prefixed id
    # while the payload keeps the canonical one.
    assert event.episode_id == f"episode-{termination.episode_id}"
    assert store.list_episode_terminations(run.run_id) == (termination,)

    kinds = [item.kind for item in store.list_events(run.run_id)]
    assert kinds == [EPISODE_TERMINATION_EVENT]
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)
    # A terminal run is immutable, and its terminations stay readable.
    assert store.list_episode_terminations(run.run_id) == (termination,)


def test_record_episode_termination_rejects_a_foreign_object(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="termination.contract-v1", protocol_version="1.0", kind="training"
    )
    with pytest.raises(TypeError, match="EpisodeTermination"):
        store.record_episode_termination(run.run_id, {"termination_reason": "failed"})  # type: ignore[arg-type]


def test_the_guard_projects_a_lifecycle_view_for_manifests() -> None:
    guard = EpisodeTerminationGuard(UUID(int=3), caps=EpisodeCaps(max_steps=5), now_ns=0)
    guard.note_step(1, observation_sequence=10, now_ns=1)
    assert guard.to_mapping(now_ns=2) == {
        "schema_version": "glr.episode-termination.v1",
        "episode_id": str(UUID(int=3)),
        "closed": False,
        "steps": 1,
        "last_known_sequence": 10,
        "latched_at_ns": None,
        "elapsed_ns": 2,
        "caps": {"max_steps": 5, "max_time_ns": None, "death_cap": None, "stall_steps": None},
        "termination": None,
    }


# --- validation --------------------------------------------------------------


def test_caps_reject_non_positive_values() -> None:
    for kwargs in ({"max_steps": 0}, {"max_time_ns": -1}, {"death_cap": True}):  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            EpisodeCaps(**kwargs)


def test_caps_round_trip_through_a_mapping() -> None:
    caps = EpisodeCaps(max_steps=4, max_time_ns=5, death_cap=6, stall_steps=7)
    assert EpisodeCaps.from_mapping(caps.to_mapping()) == caps
    with pytest.raises(ValueError, match="unknown episode caps field"):
        EpisodeCaps.from_mapping({"max_steps": 1, "nope": 2})


def test_the_guard_rejects_a_bad_episode_id_or_caps() -> None:
    with pytest.raises(TypeError, match="episode_id must be a UUID"):
        EpisodeTerminationGuard("not-a-uuid")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="caps must be an EpisodeCaps"):
        EpisodeTerminationGuard(uuid4(), caps={"max_steps": 1})  # type: ignore[arg-type]


def test_a_step_id_must_be_a_non_negative_integer() -> None:
    guard = EpisodeTerminationGuard(uuid4(), now_ns=0)
    with pytest.raises(ValueError, match="step_id"):
        guard.note_step(-1)


def test_the_detail_field_is_bounded() -> None:
    with pytest.raises(ValueError, match="cannot exceed 512"):
        EpisodeTermination(uuid4(), TerminationReason.FAILED, detail="x" * 513)


# --- an indeterminate outcome is absorbing -----------------------------------


def test_indeterminate_is_distinct_from_rejected() -> None:
    rejected = ActionReceipt(
        action_id="a1",
        episode_id=uuid4(),
        step_id=1,
        outcome=ActionOutcome.REJECTED,
        issued_timestamp_ns=0,
        observed_timestamp_ns=1,
    )
    indeterminate = ActionReceipt(
        action_id="a2",
        episode_id=uuid4(),
        step_id=1,
        outcome=ActionOutcome.INDETERMINATE,
        issued_timestamp_ns=0,
        observed_timestamp_ns=1,
    )
    assert rejected.is_refusal is True
    assert rejected.is_indeterminate is False
    assert indeterminate.is_refusal is False
    assert indeterminate.is_indeterminate is True


def test_an_indeterminate_receipt_cannot_be_marked_retryable() -> None:
    with pytest.raises(ValueError, match="cannot be retryable"):
        ActionReceipt(
            action_id="a1",
            episode_id=uuid4(),
            step_id=1,
            outcome=ActionOutcome.INDETERMINATE,
            issued_timestamp_ns=0,
            observed_timestamp_ns=1,
            retryable=True,
        )


def test_reporting_indeterminate_ends_the_episode_immediately() -> None:
    environment = ScriptedTerminationEnvironment(
        episode_length=4,
        outcome=ActionOutcome.INDETERMINATE,
        outcome_at_step=2,
        caps=EpisodeCaps(max_steps=4),
    )
    collector = SyncCollector(environment)
    unroll = collector.collect(_always_increment, steps=8)
    # Step 2 is the latched one. Transition 1 is dropped too: its successor
    # observation is the untrustworthy one the latched step produced.
    assert [transition.step_id for transition in unroll.transitions] == [0]
    assert bool(np.all(unroll.transitions[-1].truncated)) is True
    assert environment._step_id == 2  # the environment was never stepped again
    termination = collector.last_termination()
    assert termination is not None
    assert termination.reason is TerminationReason.ENV_INDETERMINATE
    assert termination.step_id == 2
    assert termination.attributed_by == "runtime"
    assert termination.indeterminate is True


def test_the_manifest_records_the_latch_time_and_last_known_sequence() -> None:
    environment = ScriptedTerminationEnvironment(
        episode_length=4,
        outcome=ActionOutcome.INDETERMINATE,
        outcome_at_step=3,
        caps=EpisodeCaps(max_steps=4),
    )
    collector = SyncCollector(environment)
    collector.collect(_always_increment, steps=8)
    termination = collector.last_termination()
    assert termination is not None
    # The latched step is 3, whose timestep timestamp is 3_000_000 ns.
    assert termination.latched_at_ns == 3_000_000
    assert termination.step_id == 3
    # The last known-good sequence is the one the latched receipt reported.
    assert termination.last_known_sequence == 3
    mapping = termination.to_mapping()
    assert mapping["latched_at_ns"] == 3_000_000
    assert mapping["last_known_sequence"] == 3


def test_latching_on_the_first_step_records_nothing_and_says_why() -> None:
    environment = ScriptedTerminationEnvironment(
        episode_length=4,
        outcome=ActionOutcome.INDETERMINATE,
        outcome_at_step=1,
        caps=EpisodeCaps(max_steps=4),
    )
    collector = SyncCollector(environment)
    with pytest.raises(IndeterminateOutcomeError, match="indeterminate"):
        collector.collect(_always_increment, steps=8)
    termination = collector.last_termination()
    assert termination is not None
    assert termination.reason is TerminationReason.ENV_INDETERMINATE
    assert termination.step_id == 1
    assert termination.latched_at_ns == 1_000_000


def test_no_step_is_admitted_after_the_latch() -> None:
    environment = ScriptedTerminationEnvironment(
        episode_length=8,
        outcome=ActionOutcome.INDETERMINATE,
        outcome_at_step=2,
        caps=EpisodeCaps(max_steps=8),
    )
    collector = SyncCollector(environment)
    unroll = collector.collect(_always_increment, steps=8)
    latched = collector.last_termination()
    assert latched is not None
    assert latched.step_id == 2
    assert max(transition.step_id for transition in unroll.transitions) < 2
    assert environment._step_id == 2
    # The episode is absorbing: collecting again is refused until the caller
    # re-attaches explicitly.
    with pytest.raises(IndeterminateOutcomeError, match="reattach"):
        collector.collect(_always_increment, steps=8)
    assert len(collector.terminations) == 1
    # An explicit re-attach opens a new episode, which records its own reason.
    collector.reattach()
    collector.collect(_always_increment, steps=3)
    assert len(collector.terminations) == 2
    resumed = collector.last_termination()
    assert resumed is not None
    assert resumed.episode_id != latched.episode_id
    assert resumed.indeterminate is True


def test_a_plain_rejection_lets_the_episode_continue() -> None:
    environment = ScriptedTerminationEnvironment(
        episode_length=4,
        outcome=ActionOutcome.REJECTED,
        outcome_at_step=2,
        info={TERMINATION_REASON_KEY: "goal_reached"},
    )
    collector = SyncCollector(environment)
    unroll = collector.collect(_always_increment, steps=8, stop_on_done=True)
    # The rejected step 2 is recorded and the episode runs to its own boundary.
    assert [transition.step_id for transition in unroll.transitions] == [0, 1, 2, 3]
    termination = collector.last_termination()
    assert termination is not None
    assert termination.reason is TerminationReason.GOAL_REACHED
    assert termination.indeterminate is False
    assert termination.latched_at_ns is None


def test_an_indeterminate_receipt_is_not_routed_through_the_refusal_funnel() -> None:
    from game_learning_runtime.refusals import RefusalFunnel

    seen: list[ActionReceipt] = []
    funnel = RefusalFunnel(seen.append)
    indeterminate = ActionReceipt(
        action_id="a1",
        episode_id=uuid4(),
        step_id=1,
        outcome=ActionOutcome.INDETERMINATE,
        issued_timestamp_ns=0,
        observed_timestamp_ns=1,
    )
    assert funnel.observe(indeterminate) is indeterminate
    assert seen == []


def test_an_adapter_that_never_reports_indeterminate_is_unchanged() -> None:
    environment = ScriptedTerminationEnvironment(
        episode_length=3, info={TERMINATION_REASON_KEY: "goal_reached"}
    )
    collector = SyncCollector(environment)
    unroll = collector.collect(_always_increment, steps=8, stop_on_done=True)
    assert [transition.step_id for transition in unroll.transitions] == [0, 1, 2]
    termination = collector.last_termination()
    assert termination is not None
    assert termination.reason is TerminationReason.GOAL_REACHED
    assert termination.latched_at_ns is None
    assert termination.indeterminate is False
    assert collector.terminations == (termination,)


class _RaisingStepEnvironment(GameEnvironment):
    """Fail a step so the collector has to abandon the episode."""

    def __init__(self, *, fail_at: int = 1) -> None:
        self._fail_at = fail_at
        self._episode_id = uuid4()
        self._step_id = 0

    @property
    def spec(self) -> EnvironmentSpec:
        return _spec(EpisodeCaps(max_steps=8))

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        del seed, options
        self._episode_id = uuid4()
        self._step_id = 0
        return self._timestep()

    def step(self, action: TensorTree) -> TimeStep:
        del action
        self._step_id += 1
        if self._step_id >= self._fail_at:
            raise RuntimeError("transient bridge failure")
        return self._timestep()

    def close(self) -> None:
        return None

    def _timestep(self) -> TimeStep:
        return TimeStep(
            observation={"position": np.array([self._step_id], dtype=np.int64)},
            reward=np.array([0.0], dtype=np.float32),
            terminated=np.array([False], dtype=np.bool_),
            truncated=np.array([False], dtype=np.bool_),
            episode_id=self._episode_id,
            step_id=self._step_id,
            timestamp_ns=self._step_id * 1_000_000,
        )


# --- the collector publishes every terminal state it produces ----------------


def test_a_collector_lands_an_indeterminate_termination_in_the_run_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: one real ``collect()`` is readable back from the store.

    Coverage that calls ``record_episode_termination`` by hand proves the
    store API but not the wiring between collector and store, which is what
    let a permanently empty ``glr runs show`` pass. This test only ever calls
    ``collect()``; the store is read cold afterwards.
    """

    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="termination.contract-v1", protocol_version="1.0", kind="training"
    )
    environment = ScriptedTerminationEnvironment(
        episode_length=8,
        outcome=ActionOutcome.INDETERMINATE,
        outcome_at_step=2,
        caps=EpisodeCaps(max_steps=8),
    )
    collector = SyncCollector(environment, on_termination=store.termination_sink(run.run_id))
    collector.collect(_always_increment, steps=8)

    persisted = store.list_episode_terminations(run.run_id)
    assert [item.reason for item in persisted] == [TerminationReason.ENV_INDETERMINATE]
    live = collector.last_termination()
    assert live is not None
    assert persisted[0] == live
    assert persisted[0].episode_id == live.episode_id
    assert persisted[0].step_id == 2
    assert persisted[0].latched_at_ns == 2_000_000
    assert persisted[0].indeterminate is True


def test_a_collector_bound_to_a_run_publishes_terminations_by_default(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Bind a run and the sink comes with it, with no ``on_termination``.

    This is the shape the guides document and the one a trainer actually
    writes: ``store`` and ``run_id`` are already what binds the metric
    ledger. Making terminations opt-in on top of that leaves
    ``glr runs show`` reporting zero episodes for a real run, which is
    indistinguishable from a run that never collected one.
    """

    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="termination.contract-v1", protocol_version="1.0", kind="training"
    )
    environment = ScriptedTerminationEnvironment(
        episode_length=2, info={TERMINATION_REASON_KEY: "goal_reached"}
    )
    collector = SyncCollector(environment, store=store, run_id=run.run_id)
    collector.collect(_always_increment, steps=6)

    persisted = store.list_episode_terminations(run.run_id)
    assert persisted == collector.terminations
    assert [item.reason for item in persisted] == [TerminationReason.GOAL_REACHED] * 3

    # An explicit sink still wins over the bound run.
    seen: list[EpisodeTermination] = []
    explicit = SyncCollector(
        environment, store=store, run_id=run.run_id, on_termination=seen.append
    )
    explicit.collect(_always_increment, steps=2)
    assert seen == list(explicit.terminations)
    assert len(store.list_episode_terminations(run.run_id)) == 3


def test_a_plain_run_lands_one_terminal_state_per_episode_in_the_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="termination.contract-v1", protocol_version="1.0", kind="training"
    )
    environment = ScriptedTerminationEnvironment(
        episode_length=2, info={TERMINATION_REASON_KEY: "goal_reached"}
    )
    collector = SyncCollector(environment, on_termination=store.termination_sink(run.run_id))
    collector.collect(_always_increment, steps=6)

    persisted = store.list_episode_terminations(run.run_id)
    assert len(persisted) == 3
    assert {item.reason for item in persisted} == {TerminationReason.GOAL_REACHED}
    assert len({item.episode_id for item in persisted}) == 3
    assert persisted == collector.terminations


def test_a_failed_step_closes_the_episode_in_the_default_mode() -> None:
    """``on_error="raise"`` is the default, so it is the mode that matters.

    Raising out of the collector used to skip the close, leaving an episode
    that neither recorded a reason nor raised a violation.
    """

    collector = SyncCollector(_RaisingStepEnvironment(fail_at=1))
    with pytest.raises(RuntimeError, match="transient bridge failure"):
        collector.collect(_always_increment, steps=2)

    termination = collector.last_termination()
    assert termination is not None
    assert termination.reason is TerminationReason.FAILED
    assert termination.attributed_by == "caller"
    assert "RuntimeError" in (termination.detail or "")


def test_a_termination_limit_bounds_terminations_not_the_events_around_them(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = TrainingStore(tmp_path / "runs.sqlite3")
    run = store.create_run(
        environment_id="termination.contract-v1", protocol_version="1.0", kind="training"
    )
    for index in range(10):
        store.append_event(run.run_id, kind="metric.sample", payload={"index": index})
        store.record_episode_termination(
            run.run_id,
            EpisodeTermination(
                episode_id=UUID(int=index + 1),
                reason=TerminationReason.STEP_BUDGET,
                step_id=index,
                timestamp_ns=index,
            ),
        )
    assert len(store.list_events(run.run_id)) == 20
    assert len(store.list_episode_terminations(run.run_id)) == 10
    # A limit of 5 used to be spent on the unrelated metric events, so a long
    # run was reported as having ended zero episodes.
    limited = store.list_episode_terminations(run.run_id, limit=5)
    assert [item.step_id for item in limited] == [0, 1, 2, 3, 4]
