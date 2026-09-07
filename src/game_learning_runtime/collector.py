"""Framework-neutral synchronous and bounded actor collection primitives."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import numpy as np

from game_learning_runtime.contracts import (
    EnvironmentConfigSnapshot,
    TensorTree,
    TimeStep,
    Transition,
    Unroll,
)
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment


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
    """Collect fixed-length unrolls without coupling to a learner framework."""

    def __init__(
        self,
        environment: GameEnvironment,
        *,
        actor_id: str = "actor-0",
        start_mode: Literal["reset", "attach"] = "reset",
    ) -> None:
        if not actor_id:
            raise ValueError("actor_id cannot be empty")
        if start_mode not in {"reset", "attach"}:
            raise ValueError("start_mode must be 'reset' or 'attach'")
        self._environment = (
            environment
            if isinstance(environment, ContractEnvironment)
            else ContractEnvironment(environment)
        )
        self._actor_id = actor_id
        self._start_mode = start_mode
        self._current: TimeStep | None = None
        self._sequence_id = 0
        self._environment_config_snapshot: EnvironmentConfigSnapshot | None = None

    def _start(self, *, seed: int | None = None) -> TimeStep:
        if self._start_mode == "attach":
            if seed is not None:
                raise ValueError("seed is not supported when start_mode='attach'")
            timestep = self._environment.attach()
        else:
            timestep = self._environment.reset(seed=seed)
        self._environment_config_snapshot = self._environment.config_snapshot()
        return timestep

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
        """

        if steps <= 0:
            raise ValueError("steps must be positive")
        if policy_version < 0:
            raise ValueError("policy_version cannot be negative")
        if on_error not in {"raise", "partial"}:
            raise ValueError("on_error must be 'raise' or 'partial'")
        if self._current is None or self._current.done:
            self._current = self._start(seed=seed)

        config_snapshot = self._environment_config_snapshot
        transitions: list[Transition] = []
        for _ in range(steps):
            current = self._current
            action = policy(current)
            try:
                following = self._environment.step(action)
            except Exception:
                if on_error == "raise" or not transitions:
                    self._current = None if on_error == "partial" else self._current
                    raise
                # The environment state after a failed step is unknown. Mark the
                # last valid transition as a learner truncation and force reset.
                last = transitions[-1]
                transitions[-1] = replace(
                    last,
                    truncated=np.ones_like(last.truncated, dtype=np.bool_),
                )
                self._current = None
                unroll = Unroll(
                    transitions=tuple(transitions),
                    actor_id=self._actor_id,
                    sequence_id=self._sequence_id,
                    policy_version=policy_version,
                    environment_config_snapshot=config_snapshot,
                )
                self._sequence_id += 1
                return unroll
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
            self._current = following
            if following.done:
                if stop_on_done:
                    break
                if len(transitions) < steps:
                    self._current = self._start()

        unroll = Unroll(
            transitions=tuple(transitions),
            actor_id=self._actor_id,
            sequence_id=self._sequence_id,
            policy_version=policy_version,
            environment_config_snapshot=config_snapshot,
        )
        self._sequence_id += 1
        return unroll
