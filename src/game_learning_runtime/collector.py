"""Framework-neutral synchronous and bounded actor collection primitives."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import UUID

import numpy as np

from game_learning_runtime.contracts import (
    EnvironmentConfigSnapshot,
    TensorTree,
    TimeStep,
    Transition,
    Unroll,
)
from game_learning_runtime.declared_metrics import (
    DeclaredMetricAudit,
    DeclaredMetricLedger,
    MetricDeclaration,
    build_declared_metrics,
)
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.learnability import (
    LEARNABILITY_CELL_KEY,
    LearnabilityPlan,
    LearnabilityReport,
    LearnabilityTracker,
    build_tracker,
)
from game_learning_runtime.termination import (
    EpisodeCaps,
    EpisodeTermination,
    EpisodeTerminationGuard,
    IndeterminateOutcomeError,
    TerminationReason,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import; run_store is runtime-bound
    from game_learning_runtime.run_store import TrainingStore


class Policy(Protocol):
    """Minimal policy port shared by custom PPO, IMPALA, BC, and evaluation."""

    def __call__(self, timestep: TimeStep) -> TensorTree:
        """Choose a structured action from a time step."""
        ...


QueueOverflowPolicy = Literal["block", "drop-oldest", "fail"]


class ActorQueueClosed(RuntimeError):
    """Raised when a queue is closed and no more unrolls can be read."""


class ActorQueueCancelled(RuntimeError):
    """Raised when a waiting queue operation is cancelled."""


class ActorQueueFull(RuntimeError):
    """Raised by the fail policy or a timed-out blocking enqueue."""


class ActorQueueStaleUnroll(RuntimeError):
    """Raised when an unroll exceeds the configured learner policy lag."""


class ActorQueueCommitError(RuntimeError):
    """Raised when an unroll is acknowledged more than once or is unknown."""


@dataclass(frozen=True, slots=True)
class QueuedUnroll:
    """A fenced unroll lease returned by :class:`BoundedActorQueue.get`."""

    unroll: Unroll
    token: int
    enqueued_at_ns: int


@dataclass(frozen=True, slots=True)
class ActorQueueMetrics:
    """Privacy-safe queue and learner-lag counters suitable for run summaries."""

    capacity: int
    overflow_policy: QueueOverflowPolicy
    depth: int
    max_depth: int
    enqueued_unrolls: int
    dequeued_unrolls: int
    committed_unrolls: int
    dropped_unrolls: int
    aborted_unrolls: int
    uncommitted_unrolls: int
    blocked_puts: int
    cancelled_operations: int
    max_policy_version_lag: int
    actor_lag: dict[str, int]
    enqueue_latency_ns_total: int
    dequeue_latency_ns_total: int
    carry_over_unrolls: int = 0
    in_flight_unrolls: int = 0
    paused: bool = False
    drain_count: int = 0
    drain_latency_ns_total: int = 0
    stale_dropped_unrolls: int = 0
    rejected_stale_unrolls: int = 0
    oldest_pending_age_ns: int = 0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe aggregate without observations or game metadata."""

        return {
            "capacity": self.capacity,
            "overflow_policy": self.overflow_policy,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "enqueued_unrolls": self.enqueued_unrolls,
            "dequeued_unrolls": self.dequeued_unrolls,
            "committed_unrolls": self.committed_unrolls,
            "dropped_unrolls": self.dropped_unrolls,
            "aborted_unrolls": self.aborted_unrolls,
            "uncommitted_unrolls": self.uncommitted_unrolls,
            "blocked_puts": self.blocked_puts,
            "cancelled_operations": self.cancelled_operations,
            "max_policy_version_lag": self.max_policy_version_lag,
            "actor_lag": dict(self.actor_lag),
            "enqueue_latency_ns_total": self.enqueue_latency_ns_total,
            "dequeue_latency_ns_total": self.dequeue_latency_ns_total,
            "carry_over_unrolls": self.carry_over_unrolls,
            "in_flight_unrolls": self.in_flight_unrolls,
            "paused": self.paused,
            "drain_count": self.drain_count,
            "drain_latency_ns_total": self.drain_latency_ns_total,
            "stale_dropped_unrolls": self.stale_dropped_unrolls,
            "rejected_stale_unrolls": self.rejected_stale_unrolls,
            "oldest_pending_age_ns": self.oldest_pending_age_ns,
        }


class BoundedActorQueue:
    """Thread-safe, learner-neutral queue for fixed actor unrolls.

    The implementation intentionally uses only the standard library. ``get`` returns a
    lease, and an unroll is not counted as a successful learner update until ``commit``
    is called. ``abort`` and queue drops remain visible in metrics. Closing wakes all
    waiters while allowing already queued leases to drain.
    """

    def __init__(
        self,
        capacity: int,
        *,
        overflow_policy: QueueOverflowPolicy = "block",
        learner_policy_version: int = 0,
        max_policy_version_lag: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if overflow_policy not in {"block", "drop-oldest", "fail"}:
            raise ValueError("overflow_policy must be 'block', 'drop-oldest', or 'fail'")
        _validate_policy_version(learner_policy_version)
        if max_policy_version_lag is not None and (
            isinstance(max_policy_version_lag, bool)
            or not isinstance(max_policy_version_lag, int)
            or max_policy_version_lag < 0
        ):
            raise ValueError("max_policy_version_lag must be a non-negative integer")
        self._capacity = capacity
        self._overflow_policy = overflow_policy
        self._learner_policy_version = learner_policy_version
        self._max_policy_version_lag = max_policy_version_lag
        self._items: deque[QueuedUnroll] = deque()
        self._in_flight: dict[int, QueuedUnroll] = {}
        self._last_sequence: dict[str, int] = {}
        self._next_token = 0
        self._max_depth = 0
        self._enqueued = 0
        self._dequeued = 0
        self._committed = 0
        self._dropped = 0
        self._aborted = 0
        self._blocked_puts = 0
        self._cancelled = 0
        self._max_policy_lag = 0
        self._enqueue_latency_ns_total = 0
        self._dequeue_latency_ns_total = 0
        self._paused = False
        self._draining = False
        self._drain_count = 0
        self._drain_latency_ns_total = 0
        self._stale_dropped = 0
        self._rejected_stale = 0
        self._closed = False
        self._condition = threading.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def paused(self) -> bool:
        """Whether new learner leases are paused; producers remain bounded."""

        with self._condition:
            return self._paused

    def pause(self) -> None:
        """Pause new learner leases while existing leases can commit or abort.

        This is a queue barrier, not a pause of game time or actor inference.
        Producers may still enqueue up to the normal capacity limit.
        """

        with self._condition:
            if self._closed:
                raise ActorQueueClosed("actor queue is closed")
            self._paused = True
            self._condition.notify_all()

    def resume(self) -> None:
        """Release the lease barrier after publishing the learner policy."""

        with self._condition:
            if self._draining:
                raise ActorQueueCommitError("cannot resume while drain is waiting")
            self._paused = False
            self._condition.notify_all()

    def drain(
        self,
        *,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ActorQueueMetrics:
        """Pause new leases and wait for in-flight leases to commit or abort.

        Queued work remains as carry-over. A timeout or cancellation leaves the
        queue paused, so the caller cannot mistake failure for a safe update.
        Call :meth:`resume` after publishing the learner policy. This does not
        synchronize policy objects used by actors; their owner must do that.
        """

        _validate_timeout(timeout)
        started = time.monotonic_ns()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._closed:
                raise ActorQueueClosed("actor queue is closed")
            if self._draining:
                raise ActorQueueCommitError("another drain is already waiting")
            self._paused = True
            self._draining = True
            self._condition.notify_all()
            try:
                while True:
                    self._raise_if_cancelled_locked(cancel_event)
                    if self._closed:
                        raise ActorQueueClosed("actor queue closed during drain")
                    if not self._in_flight:
                        self._drain_count += 1
                        break
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise ActorQueueFull("timed out waiting for actor queue drain")
                    self._condition.wait(timeout=_wait_interval(remaining, cancel_event))
            finally:
                self._drain_latency_ns_total += max(0, time.monotonic_ns() - started)
                self._draining = False
                self._condition.notify_all()
            return self.metrics()

    def set_learner_policy_version(self, policy_version: int) -> None:
        """Update lag telemetry/cutoffs and wake producers that may be stale."""

        _validate_policy_version(policy_version)
        with self._condition:
            self._learner_policy_version = policy_version
            self._max_policy_lag = max(
                self._max_policy_lag,
                self._current_policy_lag_locked(),
            )
            self._condition.notify_all()

    def put(
        self,
        unroll: Unroll,
        *,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Enqueue an unroll, returning ``False`` only when a policy drops it.

        ``block`` waits for capacity, ``drop-oldest`` evicts one queued unroll, and
        ``fail`` raises :class:`ActorQueueFull`. Sequence IDs are fenced per actor so
        a retried or duplicated unroll cannot silently become a second update.
        """

        if not isinstance(unroll, Unroll):
            raise TypeError("unroll must be an Unroll")
        _validate_policy_version(unroll.policy_version)
        _validate_timeout(timeout)
        started = time.monotonic_ns()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._raise_if_cancelled_locked(cancel_event)
            while True:
                self._raise_if_cancelled_locked(cancel_event)
                if self._closed:
                    raise ActorQueueClosed("actor queue is closed")
                previous = self._last_sequence.get(unroll.actor_id)
                if previous is not None and unroll.sequence_id <= previous:
                    raise ValueError(
                        f"unroll sequence_id must increase for actor {unroll.actor_id!r}"
                    )
                if self._is_stale_locked(unroll):
                    self._rejected_stale += 1
                    raise ActorQueueStaleUnroll("unroll exceeds maximum learner policy lag")
                if len(self._items) < self._capacity:
                    break
                if self._overflow_policy == "drop-oldest":
                    self._items.popleft()
                    self._dropped += 1
                    break
                if self._overflow_policy == "fail":
                    raise ActorQueueFull("actor queue is full")
                self._blocked_puts += 1
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise ActorQueueFull("timed out waiting for actor queue capacity")
                self._condition.wait(timeout=_wait_interval(remaining, cancel_event))
            token = self._next_token
            self._next_token += 1
            item = QueuedUnroll(unroll, token, time.monotonic_ns())
            self._items.append(item)
            self._last_sequence[unroll.actor_id] = unroll.sequence_id
            self._enqueued += 1
            self._enqueue_latency_ns_total += max(0, time.monotonic_ns() - started)
            self._max_depth = max(self._max_depth, len(self._items))
            self._max_policy_lag = max(self._max_policy_lag, self._policy_lag(unroll))
            self._condition.notify_all()
            return True

    def put_nowait(self, unroll: Unroll) -> bool:
        """Enqueue without waiting for capacity."""

        return self.put(unroll, timeout=0)

    def get(
        self,
        *,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> QueuedUnroll:
        """Lease the oldest unroll; call ``commit`` or ``abort`` exactly once."""

        _validate_timeout(timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                self._raise_if_cancelled_locked(cancel_event)
                if not self._paused and self._items:
                    item = self._items.popleft()
                    if self._is_stale_locked(item.unroll):
                        self._stale_dropped += 1
                        self._dropped += 1
                        self._condition.notify_all()
                        continue
                    break
                if self._closed:
                    raise ActorQueueClosed("actor queue is closed")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise ActorQueueFull("timed out waiting for an actor unroll")
                self._condition.wait(timeout=_wait_interval(remaining, cancel_event))
            self._in_flight[item.token] = item
            self._dequeued += 1
            self._dequeue_latency_ns_total += max(0, time.monotonic_ns() - item.enqueued_at_ns)
            self._max_policy_lag = max(self._max_policy_lag, self._policy_lag(item.unroll))
            self._condition.notify_all()
            return item

    def get_nowait(self) -> QueuedUnroll:
        """Lease without waiting for an available unroll."""

        return self.get(timeout=0)

    def commit(self, item: QueuedUnroll) -> None:
        """Mark a leased unroll as a successful learner update.

        A stale rejection retains the lease: the caller must explicitly abort it.
        This acknowledgement does not roll back any learner-side weight changes.
        """

        with self._condition:
            self._validate_in_flight_locked(item)
            if self._is_stale_locked(item.unroll):
                raise ActorQueueStaleUnroll("leased unroll exceeds maximum learner policy lag")
            self._take_in_flight_locked(item)
            self._committed += 1
            self._condition.notify_all()

    def abort(self, item: QueuedUnroll) -> None:
        """Discard a leased unroll while retaining an uncommitted metric."""

        with self._condition:
            self._take_in_flight_locked(item)
            self._aborted += 1
            self._condition.notify_all()

    def close(self) -> None:
        """Stop new puts and wake waiters; queued items can still be drained."""

        with self._condition:
            self._closed = True
            self._paused = False
            self._condition.notify_all()

    shutdown = close

    def metrics(self) -> ActorQueueMetrics:
        """Snapshot bounded queue state and aggregate learner/backpressure telemetry."""

        with self._condition:
            pending = (*self._items, *self._in_flight.values())
            actor_lag: dict[str, int] = {}
            for item in pending:
                actor_lag[item.unroll.actor_id] = actor_lag.get(item.unroll.actor_id, 0) + 1
            return ActorQueueMetrics(
                capacity=self._capacity,
                overflow_policy=self._overflow_policy,
                depth=len(self._items),
                max_depth=self._max_depth,
                enqueued_unrolls=self._enqueued,
                dequeued_unrolls=self._dequeued,
                committed_unrolls=self._committed,
                dropped_unrolls=self._dropped,
                aborted_unrolls=self._aborted,
                uncommitted_unrolls=len(pending),
                blocked_puts=self._blocked_puts,
                cancelled_operations=self._cancelled,
                max_policy_version_lag=self._max_policy_lag,
                actor_lag=actor_lag,
                enqueue_latency_ns_total=self._enqueue_latency_ns_total,
                dequeue_latency_ns_total=self._dequeue_latency_ns_total,
                carry_over_unrolls=sum(self._policy_lag(item.unroll) > 0 for item in pending),
                in_flight_unrolls=len(self._in_flight),
                paused=self._paused,
                drain_count=self._drain_count,
                drain_latency_ns_total=self._drain_latency_ns_total,
                stale_dropped_unrolls=self._stale_dropped,
                rejected_stale_unrolls=self._rejected_stale,
                oldest_pending_age_ns=max(
                    (max(0, time.monotonic_ns() - item.enqueued_at_ns) for item in pending),
                    default=0,
                ),
            )

    def run_summary(self) -> dict[str, object]:
        """Return the queue metrics in the shape accepted by run summaries."""

        return {"actor_queue": self.metrics().as_dict()}

    def _validate_in_flight_locked(self, item: QueuedUnroll) -> None:
        current = self._in_flight.get(item.token)
        if current is None or current is not item:
            raise ActorQueueCommitError("unknown or already finalized actor unroll")

    def _take_in_flight_locked(self, item: QueuedUnroll) -> QueuedUnroll:
        self._validate_in_flight_locked(item)
        return self._in_flight.pop(item.token)

    def _policy_lag(self, unroll: Unroll) -> int:
        return max(0, self._learner_policy_version - unroll.policy_version)

    def _current_policy_lag_locked(self) -> int:
        pending = (*self._items, *self._in_flight.values())
        return max((self._policy_lag(item.unroll) for item in pending), default=0)

    def _is_stale_locked(self, unroll: Unroll) -> bool:
        return (
            self._max_policy_version_lag is not None
            and self._policy_lag(unroll) > self._max_policy_version_lag
        )

    def _raise_if_cancelled_locked(self, cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            self._cancelled += 1
            raise ActorQueueCancelled("actor queue operation was cancelled")


def _validate_timeout(timeout: float | None) -> None:
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be finite and non-negative")


def _validate_policy_version(policy_version: int) -> None:
    if (
        isinstance(policy_version, bool)
        or not isinstance(policy_version, int)
        or policy_version < 0
    ):
        raise ValueError("policy_version must be a non-negative integer")


def _wait_interval(
    remaining: float | None,
    cancel_event: threading.Event | None,
) -> float | None:
    if cancel_event is None:
        return remaining
    interval = 0.05
    return interval if remaining is None else min(interval, max(remaining, 0.0))


class SyncCollector:
    """Collect fixed-length unrolls without coupling to a learner framework.

    When a :class:`~game_learning_runtime.declared_metrics.MetricDeclaration` or
    a ready :class:`~game_learning_runtime.declared_metrics.DeclaredMetricLedger`
    is supplied the collector also audits the run's metric promises: every
    episode close counts what was declared against what was emitted, and strict
    mode refuses to keep collecting on an episode that measured nothing.
    Without a declaration nothing is audited and behaviour is unchanged.

    A declaration alone is not enough to be counted: emissions are counted in
    the run store, so the ledger has to be bound to the run that owns them.
    Pass ``store`` and ``run_id`` together and the collector builds and binds
    the ledger itself; pass a ledger that was already bound by
    :func:`~game_learning_runtime.declared_metrics.bind_declared_metrics` to
    keep the caller's binding. A collector that built the ledger owns the
    binding and releases it through :meth:`release_declared_metrics`.

    The collector owns the episode termination seam. It opens one
    :class:`~game_learning_runtime.termination.EpisodeTerminationGuard` per
    episode, admits only steps the guard accepts, and closes every episode it
    ends with a recorded :class:`~game_learning_runtime.termination.EpisodeTermination`.
    An episode that ends without a reason is a contract violation, so a
    collection that cannot say why an episode ended raises instead of returning
    silent data.

    Terminal states stay in memory unless the caller supplies
    ``on_termination`` or binds a run. Pass
    :meth:`~game_learning_runtime.run_store.TrainingStore.termination_sink`
    to publish them where the caller chooses:

    ```python
    collector = SyncCollector(env, on_termination=store.termination_sink(run.run_id))
    ```

    Passing ``store`` and ``run_id`` together binds that sink by default,
    because declared metrics are already bound through the same pair: a
    collector that owns a run but leaves its terminations in memory makes the
    run look as though it ended no episodes at all.

    When a :class:`~game_learning_runtime.learnability.LearnabilityPlan` is
    supplied the collector also watches the run's learnability budget: every
    recorded transition contributes a state cell, and a configured coverage
    floor stops collection as soon as the remaining budget provably cannot
    reach it. Without a plan nothing is measured and behaviour is unchanged.

    """

    def __init__(
        self,
        environment: GameEnvironment,
        *,
        actor_id: str = "actor-0",
        start_mode: Literal["reset", "attach"] = "reset",
        reset_options: Mapping[str, Any] | None = None,
        declared_metrics: MetricDeclaration | DeclaredMetricLedger | None = None,
        store: TrainingStore | None = None,
        run_id: str | None = None,
        episode_caps: EpisodeCaps | None = None,
        on_termination: Callable[[EpisodeTermination], None] | None = None,
        learnability: LearnabilityPlan | None = None,
    ) -> None:
        if not actor_id:
            raise ValueError("actor_id cannot be empty")
        if start_mode not in {"reset", "attach"}:
            raise ValueError("start_mode must be 'reset' or 'attach'")
        if (store is None) != (run_id is None):
            raise ValueError("store and run_id must be supplied together")
        if episode_caps is not None and not isinstance(episode_caps, EpisodeCaps):
            raise TypeError("episode_caps must be an EpisodeCaps or None")
        if on_termination is not None and not callable(on_termination):
            raise TypeError("on_termination must be callable or None")
        if learnability is not None and not isinstance(learnability, LearnabilityPlan):
            raise TypeError("learnability must be a LearnabilityPlan or None")
        self._environment = (
            environment
            if isinstance(environment, ContractEnvironment)
            else ContractEnvironment(environment)
        )
        self._actor_id = actor_id
        self._start_mode = start_mode
        self._reset_options = reset_options
        self._current: TimeStep | None = None
        self._sequence_id = 0
        self._environment_config_snapshot: EnvironmentConfigSnapshot | None = None
        self._store = store
        self._run_id = run_id
        self._declared_metrics = self._resolve_declared_metrics(declared_metrics)
        self._guard: EpisodeTerminationGuard | None = None
        self._terminations: list[EpisodeTermination] = []
        if on_termination is None and store is not None and run_id is not None:
            on_termination = store.termination_sink(run_id)
        self._on_termination = on_termination
        if episode_caps is None:
            declared = self._environment.spec.episode_caps
            episode_caps = declared if isinstance(declared, EpisodeCaps) else None
        self._episode_caps = episode_caps
        self._learnability: LearnabilityTracker | None = None
        if learnability is not None:
            self._learnability = build_tracker(
                learnability,
                declaration=learnability.declaration or self._environment.spec.learnability,
                observation_spec=self._environment.spec.observation,
            )

    def _resolve_declared_metrics(
        self, value: MetricDeclaration | DeclaredMetricLedger | None
    ) -> DeclaredMetricLedger | None:
        """Build a ledger from a declaration, or adopt a caller-owned one.

        A bare declaration is resolved against the adapter's capabilities, so
        ``strict-metrics-v1`` is what turns a gap into a failure, and it is
        bound to ``store`` / ``run_id`` so the emissions counted by the run
        store reach it. A caller that already bound a ledger to a run keeps its
        own binding, which is why a ledger and ``store`` cannot be combined.
        """

        if value is None:
            return None
        if isinstance(value, DeclaredMetricLedger):
            if self._store is not None:
                raise ValueError(
                    "store and run_id cannot be combined with a DeclaredMetricLedger: "
                    "the ledger already owns its run binding"
                )
            return value
        if not isinstance(value, MetricDeclaration):
            raise TypeError(
                "declared_metrics must be a MetricDeclaration, a DeclaredMetricLedger, or None"
            )
        return build_declared_metrics(
            value,
            capabilities=self._environment.spec.capabilities,
            store=self._store,
            run_id=self._run_id,
        )

    @property
    def declared_metrics(self) -> DeclaredMetricLedger | None:
        """The metric ledger, or ``None`` when nothing was declared."""

        return self._declared_metrics

    def release_declared_metrics(self) -> None:
        """Unbind the run from a ledger this collector built.

        Ending a run is the caller's job, so unbinding is explicit rather than
        tied to garbage collection. Releasing a ledger the caller already owned
        is left to the caller.
        """

        if self._store is not None and self._declared_metrics is not None:
            self._declared_metrics.release()

    def declared_metric_audits(self) -> tuple[DeclaredMetricAudit, ...]:
        """Every closed episode's audit, or an empty tuple when nothing is watched."""

        return () if self._declared_metrics is None else self._declared_metrics.audits

    def _close_declared_metrics(
        self, episode_id: UUID, *, timestamp_ns: int, require: bool = True
    ) -> None:
        """Audit one closed episode before the run can be reported complete.

        Order matters: the audit is recorded first, so a strict failure still
        leaves the counters on the record. Raising is the half that stops the
        gap from recurring, not the half that reports it.

        ``require=False`` records the audit without raising, which is what an
        aborted episode needs: the environment failure that ended it is the
        error worth propagating, and a missing-metric error raised on the way
        out would hide it.
        """

        ledger = self._declared_metrics
        if ledger is None:
            return
        audit = ledger.close_episode(str(episode_id), timestamp_ns=timestamp_ns)
        if require:
            audit.require()

    def _start(self, *, seed: int | None = None) -> TimeStep:
        if self._start_mode == "attach":
            if seed is not None:
                raise ValueError("seed is not supported when start_mode='attach'")
            timestep = self._environment.attach(options=self._reset_options)
        else:
            timestep = self._environment.reset(seed=seed, options=self._reset_options)
        self._environment_config_snapshot = self._environment.config_snapshot()
        return timestep

    def _open_episode(
        self, timestep: TimeStep, *, now_ns: int | None = None
    ) -> EpisodeTerminationGuard:
        guard = EpisodeTerminationGuard(
            timestep.episode_id, caps=self._episode_caps, now_ns=now_ns or timestep.timestamp_ns
        )
        self._guard = guard
        return guard

    def _close_episode(
        self,
        *,
        reason: TerminationReason | None = None,
        detail: str | None = None,
        info: Mapping[str, Any] | None = None,
        step_id: int | None = None,
        now_ns: int | None = None,
    ) -> EpisodeTermination | None:
        """Close the current episode, if needed, and remember why it ended.

        An episode that an absorbing outcome already latched is closed; this
        only has to record the terminal state it produced.

        The terminal state is published to ``on_termination`` exactly once,
        so a sink that persists to the run store mirrors this collector
        instead of silently reporting zero episodes.
        """

        guard = self._guard
        if guard is None:
            return None
        if guard.closed:
            termination = guard.termination
        else:
            termination = guard.close(
                reason=reason, detail=detail, info=info, step_id=step_id, now_ns=now_ns
            )
        if termination is not None and not any(item is termination for item in self._terminations):
            self._terminations.append(termination)
            if self._on_termination is not None:
                self._on_termination(termination)
        return termination

    def reattach(self, *, seed: int | None = None) -> TimeStep:
        """Open a fresh episode after an absorbing outcome closed the last one.

        An episode that latched leaves the host state unknown, so collecting
        again is refused until the caller re-attaches here. The restart itself
        is supervised elsewhere; this is the seam that makes "no further step
        without an explicit re-attach" enforceable.
        """

        timestep = self._start(seed=seed)
        self._open_episode(timestep)
        self._current = timestep
        return timestep

    @property
    def terminations(self) -> tuple[EpisodeTermination, ...]:
        """Terminal state of every episode this collector has ended."""

        return tuple(self._terminations)

    def last_termination(self) -> EpisodeTermination | None:
        """Terminal state of the most recently ended episode, if any."""

        return self._terminations[-1] if self._terminations else None

    @property
    def learnability(self) -> LearnabilityTracker | None:
        """The learnability tracker, or ``None`` when no plan was supplied."""

        return self._learnability

    def learnability_report(self) -> LearnabilityReport | None:
        """Current learnability verdict, or ``None`` when nothing is watched."""

        return None if self._learnability is None else self._learnability.report()

    def _observe_learnability(self, timestep: TimeStep) -> None:
        """Charge one recorded step to the learnability budget.

        A declared binning resolves the cell from the observation; otherwise an
        adapter may report it through ``info[LEARNABILITY_CELL_KEY]``. A step that resolves
        to no cell is still charged, because an untracked step is not free.
        """

        tracker = self._learnability
        if tracker is None:
            return
        declared = tracker.declaration.state
        cell = timestep.info.get(LEARNABILITY_CELL_KEY)
        observation = None if declared is None or not declared.bins else timestep.observation
        tracker.observe(
            cell=cell if isinstance(cell, (int, str)) else None,
            observation=observation,
            now_ns=timestep.timestamp_ns,
        )
        tracker.require()

    def collect(
        self,
        policy: Policy,
        *,
        steps: int,
        policy_version: int = 0,
        seed: int | None = None,
        stop_on_done: bool = False,
        on_error: Literal["raise", "partial"] = "raise",
    ) -> Unroll:
        """Collect up to ``steps`` transitions.

        By default a terminal transition starts a fresh episode so the result
        remains fixed length. Set ``stop_on_done`` for long-running live games
        where an unroll must never mix progression from multiple episodes.

        A declared-metric ledger is audited at every episode close, so an
        episode that measured nothing is named before the next one starts.
        Steps taken after an episode ended are never recorded. An
        :attr:`~game_learning_runtime.contracts.ActionOutcome.INDETERMINATE`
        receipt ends the episode immediately and its own step is dropped, so a
        latched episode contributes zero steps to the dataset.

        When ``environment.step`` raises, the episode is abandoned: its
        consequence is unknown, so it is closed with
        :attr:`~game_learning_runtime.termination.TerminationReason.FAILED`
        before the exception propagates (``on_error="raise"``) or before the
        partial unroll is returned. Either way the episode owes a reason and
        the next collection starts a fresh one.

        A learnability plan is enforced here, so a configuration that cannot
        reach its coverage floor stops mid-collection instead of after the last
        budgeted step.

        """

        if steps <= 0:
            raise ValueError("steps must be positive")
        if policy_version < 0:
            raise ValueError("policy_version cannot be negative")
        if on_error not in {"raise", "partial"}:
            raise ValueError("on_error must be 'raise' or 'partial'")
        if self._guard is not None and self._guard.indeterminate:
            # Absorbing: the host state is unknown, so stepping a fresh episode
            # would attribute its consequences to the wrong episode. The caller
            # has to re-attach explicitly first.
            termination = self._guard.termination
            raise IndeterminateOutcomeError(
                episode_id=self._guard.episode_id,
                step_id=None if termination is None else termination.step_id,
                detail="call reattach() to open a new episode",
            )
        if self._current is None or self._current.done:
            self._current = self._start(seed=seed)
            self._open_episode(self._current)
        guard = self._guard if self._guard is not None else self._open_episode(self._current)

        config_snapshot = self._environment_config_snapshot
        transitions: list[Transition] = []
        for _ in range(steps):
            current = self._current
            action = policy(current)
            guard.note_step(
                current.step_id,
                observation_sequence=_observation_sequence(current),
                now_ns=current.timestamp_ns,
            )
            try:
                following = self._environment.step(action)
            except Exception as error:
                # The episode is abandoned, so it still owes a terminal state:
                # the environment consequence of the failed step is unknown.
                # It is settled before the error leaves the collector, in both
                # collection modes, so no caller can observe an episode that
                # neither recorded a reason nor raised a violation.
                self._close_episode(
                    reason=TerminationReason.FAILED,
                    detail=f"step raised {type(error).__name__}",
                    step_id=current.step_id,
                    now_ns=current.timestamp_ns,
                )
                # The episode ends here too. Recording its audit keeps an
                # aborted episode from looking like an adapter that never
                # declared anything -- the one difference the counters exist
                # to expose. Strict mode stays quiet: the environment error is
                # the failure worth raising.
                self._close_declared_metrics(
                    current.episode_id, timestamp_ns=time.time_ns(), require=False
                )
                self._current = None
                if on_error == "raise":
                    raise
                if not transitions:
                    raise
                # The environment state after a failed step is unknown. Mark the
                # last valid transition as a learner truncation and force reset.
                last = transitions[-1]
                transitions[-1] = replace(
                    last,
                    truncated=np.ones_like(last.truncated, dtype=np.bool_),
                )
                unroll = Unroll(
                    transitions=tuple(transitions),
                    actor_id=self._actor_id,
                    sequence_id=self._sequence_id,
                    policy_version=policy_version,
                    environment_config_snapshot=config_snapshot,
                )
                self._sequence_id += 1
                return unroll
            if guard.observe_outcome(
                following.action_receipt,
                step_id=following.step_id,
                observation_sequence=_observation_sequence(following),
                now_ns=following.timestamp_ns,
            ):
                # Absorbing: the step that produced the receipt is itself
                # untrustworthy, so it is dropped rather than recorded.
                self._close_episode(step_id=following.step_id, now_ns=following.timestamp_ns)
                self._current = None
                if not transitions:
                    raise IndeterminateOutcomeError(
                        episode_id=guard.episode_id, step_id=following.step_id
                    )
                # The state after the latched step is unknown, so the last
                # trustworthy transition is cut off here instead of inviting
                # the learner to bootstrap past it.
                last = transitions[-1]
                transitions[-1] = replace(
                    last,
                    truncated=np.ones_like(last.truncated, dtype=np.bool_),
                )
                break
            transitions.append(
                Transition(
                    episode_id=current.episode_id,
                    step_id=current.step_id,
                    observation=current.observation,
                    action=action,
                    action_mask=current.action_mask,
                    reward=following.reward,
                    next_observation=following.observation,
                    next_action_mask=following.action_mask,
                    action_receipt=following.action_receipt,
                    terminated=following.terminated,
                    truncated=following.truncated,
                    events=following.events,
                    info=following.info,
                    timestamp_ns=following.timestamp_ns,
                )
            )
            self._observe_learnability(current)
            self._current = following
            if following.done:
                # Episode close. A metric that was declared and never arrived is
                # a property of the episode that just ended, so it is checked
                # here rather than when a report is finally rendered.
                self._close_declared_metrics(
                    current.episode_id, timestamp_ns=following.timestamp_ns
                )
                self._close_episode(
                    info=following.info,
                    step_id=following.step_id,
                    now_ns=following.timestamp_ns,
                )
                if stop_on_done:
                    break
                if len(transitions) < steps:
                    self._current = self._start(seed=seed)
                    guard = self._open_episode(self._current)

        unroll = Unroll(
            transitions=tuple(transitions),
            actor_id=self._actor_id,
            sequence_id=self._sequence_id,
            policy_version=policy_version,
            environment_config_snapshot=config_snapshot,
        )
        self._sequence_id += 1
        return unroll


def _observation_sequence(timestep: TimeStep) -> int | None:
    """Read the runtime-owned liveness counter a timestep carries, if any."""

    receipt = timestep.action_receipt
    if receipt is not None and receipt.authoritative_observation_sequence is not None:
        return receipt.authoritative_observation_sequence
    value = timestep.info.get("observation_sequence")
    return value if isinstance(value, int) and not isinstance(value, bool) else None
