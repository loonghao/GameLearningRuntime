"""Declarative lifecycle hooks for the GLR control plane.

A hook attaches a named action to a named lifecycle event.  The event space is
open by design: :data:`PREDEFINED_HOOK_EVENTS` documents what the control plane
publishes today, and any other identifier may be published by a role, adapter,
or extension without changing this module.  A subscription therefore selects
events by name or by namespace wildcard, and narrows them along the dimensions
the control plane knows about: environment, run kind, stage, status, and exit
code.

Dispatch is best effort and isolated.  An action that raises, hangs, or is
misconfigured is reported through :class:`HookResult` and the module logger; it
never propagates to the caller and never changes the exit code of the run that
published the event. Callers that need the outcome read
:class:`HookDispatchReport` instead of catching exceptions.

This module is data and dispatch only.  Built-in notification actions live in
:mod:`game_learning_runtime.hook_actions`, and any object satisfying
:class:`HookAction` can be registered at runtime.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from threading import RLock, Thread
from time import monotonic_ns, time_ns
from types import MappingProxyType
from typing import Any, Protocol

from game_learning_runtime.errors import GLRError

_LOGGER = logging.getLogger("game_learning_runtime.hooks")

HOOK_SCHEMA_VERSION = "glr.hooks.v1"

#: Event selectors are lowercase dotted identifiers; a trailing ``.*`` matches
#: every event in that namespace.
_EVENT_SELECTOR = re.compile(r"^[a-z][a-z0-9_.-]*(\.\*)?$")
_EVENT_NAME = re.compile(r"^[a-z][a-z0-9_.-]*$")
_ACTION_NAME = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DIMENSION = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

MAX_HOOK_SUBSCRIPTIONS = 64
MIN_HOOK_TIMEOUT_SECONDS = 0.1
MAX_HOOK_TIMEOUT_SECONDS = 300.0
DEFAULT_HOOK_TIMEOUT_SECONDS = 5.0
MAX_MESSAGE_TEMPLATE_LENGTH = 512

#: Fields a message template may reference.  Templates are rendered by literal
#: replacement, never by ``str.format``, so a value can never inject a field.
MESSAGE_TEMPLATE_FIELDS: tuple[str, ...] = (
    "environment_family",
    "environment_id",
    "event",
    "exit_code",
    "kind",
    "reason",
    "run_id",
    "stage",
    "status",
)

DEFAULT_MESSAGE_TEMPLATE = "{event} {status} for {environment_id} (run {run_id})"

#: Events the control plane publishes. The list is documentation, not an
#: enum: subscriptions accept any identifier, so extensions can publish their
#: own events without changing the runtime.
PREDEFINED_HOOK_EVENTS: tuple[str, ...] = (
    "train.start",
    "train.complete",
    "train.failed",
    "record.start",
    "record.stop",
    "goal.start",
    "goal.complete",
    "goal.failed",
    "runtime.start",
    "runtime.complete",
    "runtime.failed",
    "play.start",
    "play.complete",
    "play.failed",
)


class HookConfigurationError(GLRError, ValueError):
    """Raised when a hook configuration, subscription, or action is invalid."""


class HookEventStatus(str, Enum):
    """Lifecycle verdict carried by one published event."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class HookActionStatus(str, Enum):
    """Outcome of one hook action invocation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HookConfigurationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise HookConfigurationError(f"{path} requires string keys")
    return value


def _reject_unknown(value: Mapping[str, Any], *, allowed: frozenset[str], path: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise HookConfigurationError(f"{path} contains unexpected fields: {unexpected}")


#: Shared unknown-field guard for hook configuration blocks. Built-in and
#: third-party actions reuse it so every action rejects incidental keys the
#: same way instead of duplicating the check.
hook_config_guard = _reject_unknown


def _event_selector(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _EVENT_SELECTOR.fullmatch(value) is None:
        raise HookConfigurationError(
            f"{path} must be a lowercase dotted identifier, optionally ending in '.*'"
        )
    return value


def _action_name(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _ACTION_NAME.fullmatch(value) is None:
        raise HookConfigurationError(f"{path} must match {_ACTION_NAME.pattern!r}")
    return value


def _dimension(value: object, *, path: str, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise HookConfigurationError(f"{path} must be a string")
    if not value:
        if allow_empty:
            return ""
        raise HookConfigurationError(f"{path} cannot be empty")
    if _DIMENSION.fullmatch(value) is None:
        raise HookConfigurationError(f"{path} must match {_DIMENSION.pattern!r}")
    return value


def _optional_exit_code(value: object, *, path: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise HookConfigurationError(f"{path} must be an integer or null")
    return value


def _timeout(value: object, *, path: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HookConfigurationError(f"{path} must be a number")
    resolved = float(value)
    if not math.isfinite(resolved) or not MIN_HOOK_TIMEOUT_SECONDS <= resolved <= (
        MAX_HOOK_TIMEOUT_SECONDS
    ):
        raise HookConfigurationError(
            f"{path} must be finite and between {MIN_HOOK_TIMEOUT_SECONDS} and "
            f"{MAX_HOOK_TIMEOUT_SECONDS} seconds"
        )
    return resolved


def _frozen_payload(value: object, *, path: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    mapping = dict(_mapping(value, path=path))
    try:
        json.dumps(mapping, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise HookConfigurationError(f"{path} must be JSON serializable") from error
    return MappingProxyType(mapping)


def _recordable_detail(value: object) -> Mapping[str, Any]:
    """Coerce one action result detail into a recordable mapping.

    An action is third-party code, so its return value is not trusted to be
    JSON serializable. Losing the detail must never lose the result: the
    delivery already happened, and this module promises that dispatch returns
    a report instead of raising.
    """

    if value is None:
        return MappingProxyType({})
    try:
        return _frozen_payload(value, path="hook result detail")
    except (HookConfigurationError, TypeError, ValueError):
        _LOGGER.warning(
            "hook action detail was dropped because it is not recordable: %r",
            type(value).__name__,
        )
        return MappingProxyType({"detail_dropped": "detail was not JSON serializable"})


def validate_message_template(value: object, *, path: str) -> str:
    """Validate a message template and return it unchanged.

    Templates use ``{field}`` placeholders from :data:`MESSAGE_TEMPLATE_FIELDS`
    and are rendered by literal replacement, so a rendered value can never
    introduce a new placeholder or a format expression.
    """

    if not isinstance(value, str) or not value:
        raise HookConfigurationError(f"{path} must be non-empty text")
    if len(value) > MAX_MESSAGE_TEMPLATE_LENGTH:
        raise HookConfigurationError(
            f"{path} cannot exceed {MAX_MESSAGE_TEMPLATE_LENGTH} characters"
        )
    if any(ord(character) < 32 for character in value):
        raise HookConfigurationError(f"{path} cannot contain control characters")
    unknown = sorted(set(_PLACEHOLDER.findall(value)) - set(MESSAGE_TEMPLATE_FIELDS))
    if unknown:
        raise HookConfigurationError(
            f"{path} uses unsupported placeholders: {unknown}; "
            f"supported: {sorted(MESSAGE_TEMPLATE_FIELDS)}"
        )
    return value


def render_message(template: str, event: HookEvent, *, path: str = "message") -> str:
    """Render one validated template against one event."""

    validate_message_template(template, path=path)
    values: dict[str, str] = {
        "environment_family": event.environment_family,
        "environment_id": event.environment_id,
        "event": event.name,
        "exit_code": "" if event.exit_code is None else str(event.exit_code),
        "kind": event.kind,
        "reason": event.reason or "",
        "run_id": event.run_id or "",
        "stage": event.stage,
        "status": str(HookEventStatus(event.status).value),
    }
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"{{{name}}}", value)
    return rendered


@dataclass(frozen=True, slots=True)
class HookEvent:
    """One published lifecycle event.

    ``kind`` carries the run dimension (``training``, ``record``, ``goal``,
    ``playback``, ...) and ``stage`` the phase inside that dimension
    (``trainer``, ``capture``, ``game-launch``, ...). Together with
    ``environment_id`` they form the "environment x dimension" space a
    subscription filters on.
    """

    name: str
    status: HookEventStatus | str = HookEventStatus.STARTED
    environment_id: str = ""
    environment_family: str = ""
    kind: str = ""
    stage: str = ""
    run_id: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    occurred_at_ns: int = field(default_factory=time_ns)
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _EVENT_NAME.fullmatch(self.name) is None:
            raise HookConfigurationError(
                f"hook event name must be a lowercase dotted identifier: {self.name!r}"
            )
        object.__setattr__(self, "status", HookEventStatus(self.status))
        object.__setattr__(
            self, "environment_id", _dimension(self.environment_id, path="hook environment_id")
        )
        object.__setattr__(
            self,
            "environment_family",
            _dimension(self.environment_family, path="hook environment_family"),
        )
        object.__setattr__(self, "kind", _dimension(self.kind, path="hook kind"))
        object.__setattr__(self, "stage", _dimension(self.stage, path="hook stage"))
        object.__setattr__(
            self, "exit_code", _optional_exit_code(self.exit_code, path="hook exit_code")
        )
        object.__setattr__(self, "payload", _frozen_payload(self.payload, path="hook payload"))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": HOOK_SCHEMA_VERSION,
            "event": self.name,
            "status": HookEventStatus(self.status).value,
            "environment_id": self.environment_id,
            "environment_family": self.environment_family,
            "kind": self.kind,
            "stage": self.stage,
            "run_id": self.run_id,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "occurred_at_ns": self.occurred_at_ns,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class HookEventFilter:
    """Optional dimension filters applied to a matched event.

    An unset field matches everything. This is the "when" clause of a
    subscription: environment, run dimension, stage, status, and exit-code
    range.
    """

    environment_id: str | None = None
    environment_family: str | None = None
    kind: str | None = None
    stage: str | None = None
    status: HookEventStatus | str | None = None
    exit_code_min: int | None = None
    exit_code_max: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "environment_id",
            None
            if self.environment_id is None
            else _dimension(
                self.environment_id, path="hook filter environment_id", allow_empty=False
            ),
        )
        object.__setattr__(
            self,
            "environment_family",
            None
            if self.environment_family is None
            else _dimension(
                self.environment_family, path="hook filter environment_family", allow_empty=False
            ),
        )
        object.__setattr__(
            self,
            "kind",
            None
            if self.kind is None
            else _dimension(self.kind, path="hook filter kind", allow_empty=False),
        )
        object.__setattr__(
            self,
            "stage",
            None
            if self.stage is None
            else _dimension(self.stage, path="hook filter stage", allow_empty=False),
        )
        if self.status is not None:
            object.__setattr__(self, "status", HookEventStatus(self.status))
        minimum = _optional_exit_code(self.exit_code_min, path="hook filter exit_code_min")
        maximum = _optional_exit_code(self.exit_code_max, path="hook filter exit_code_max")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise HookConfigurationError("hook filter exit_code_min cannot exceed exit_code_max")
        object.__setattr__(self, "exit_code_min", minimum)
        object.__setattr__(self, "exit_code_max", maximum)

    def matches(self, event: HookEvent) -> bool:
        if self.environment_id is not None and event.environment_id != self.environment_id:
            return False
        if (
            self.environment_family is not None
            and event.environment_family != self.environment_family
        ):
            return False
        if self.kind is not None and event.kind != self.kind:
            return False
        if self.stage is not None and event.stage != self.stage:
            return False
        if self.status is not None and event.status is not HookEventStatus(self.status):
            return False
        if self.exit_code_min is None and self.exit_code_max is None:
            return True
        if event.exit_code is None:
            return False
        below_min = self.exit_code_min is not None and event.exit_code < self.exit_code_min
        above_max = self.exit_code_max is not None and event.exit_code > self.exit_code_max
        return not (below_min or above_max)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "environment_family": self.environment_family,
            "kind": self.kind,
            "stage": self.stage,
            "status": (None if self.status is None else HookEventStatus(self.status).value),
            "exit_code_min": self.exit_code_min,
            "exit_code_max": self.exit_code_max,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, path: str) -> HookEventFilter:
        _reject_unknown(
            value,
            allowed=frozenset(
                {
                    "environment_id",
                    "environment_family",
                    "kind",
                    "stage",
                    "status",
                    "exit_code_min",
                    "exit_code_max",
                }
            ),
            path=path,
        )
        return cls(
            environment_id=value.get("environment_id"),
            environment_family=value.get("environment_family"),
            kind=value.get("kind"),
            stage=value.get("stage"),
            status=value.get("status"),
            exit_code_min=value.get("exit_code_min"),
            exit_code_max=value.get("exit_code_max"),
        )


@dataclass(frozen=True, slots=True)
class HookSubscription:
    """One event-to-action binding with its dimension filter and budget."""

    event: str
    action: str
    when: HookEventFilter = field(default_factory=HookEventFilter)
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "event", _event_selector(self.event, path="hook event"))
        object.__setattr__(self, "action", _action_name(self.action, path="hook action"))
        object.__setattr__(
            self,
            "timeout_seconds",
            _timeout(
                self.timeout_seconds,
                path="hook timeout_seconds",
                default=DEFAULT_HOOK_TIMEOUT_SECONDS,
            ),
        )
        object.__setattr__(self, "config", _frozen_payload(self.config, path="hook config"))
        if not isinstance(self.enabled, bool):
            raise HookConfigurationError("hook enabled must be a boolean")

    @property
    def namespace(self) -> str:
        """Namespace prefix matched by a wildcard selector, dot included."""

        return self.event[:-1] if self.event.endswith(".*") else self.event

    def matches(self, event: HookEvent) -> bool:
        if self.event.endswith(".*"):
            if not event.name.startswith(self.namespace):
                return False
        elif event.name != self.event:
            return False
        return self.when.matches(event)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "action": self.action,
            "when": self.when.to_mapping(),
            "config": dict(self.config),
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        path: str,
        default_timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS,
    ) -> HookSubscription:
        _reject_unknown(
            value,
            allowed=frozenset({"event", "action", "when", "config", "timeout_seconds", "enabled"}),
            path=path,
        )
        if "event" not in value or "action" not in value:
            raise HookConfigurationError(f"{path} requires event and action")
        when = value.get("when")
        return cls(
            event=value["event"],
            action=value["action"],
            when=(
                HookEventFilter()
                if when is None
                else HookEventFilter.from_mapping(
                    _mapping(when, path=f"{path}.when"), path=f"{path}.when"
                )
            ),
            config=value.get("config", {}),
            timeout_seconds=value.get("timeout_seconds", default_timeout_seconds),
            enabled=value.get("enabled", True),
        )


@dataclass(frozen=True, slots=True)
class HookConfig:
    """Strict, data-only hook configuration for one project."""

    enabled: bool = True
    default_timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS
    subscriptions: tuple[HookSubscription, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise HookConfigurationError("hooks.enabled must be a boolean")
        object.__setattr__(
            self,
            "default_timeout_seconds",
            _timeout(
                self.default_timeout_seconds,
                path="hooks.default_timeout_seconds",
                default=DEFAULT_HOOK_TIMEOUT_SECONDS,
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": HOOK_SCHEMA_VERSION,
            "enabled": self.enabled,
            "default_timeout_seconds": self.default_timeout_seconds,
            "subscriptions": [item.to_mapping() for item in self.subscriptions],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, path: str = "hooks") -> HookConfig:
        _reject_unknown(
            value,
            allowed=frozenset({"enabled", "default_timeout_seconds", "subscriptions"}),
            path=path,
        )
        default_timeout = _timeout(
            value.get("default_timeout_seconds", DEFAULT_HOOK_TIMEOUT_SECONDS),
            path=f"{path}.default_timeout_seconds",
            default=DEFAULT_HOOK_TIMEOUT_SECONDS,
        )
        raw = value.get("subscriptions", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise HookConfigurationError(f"{path}.subscriptions must be an array")
        if len(raw) > MAX_HOOK_SUBSCRIPTIONS:
            raise HookConfigurationError(
                f"{path}.subscriptions cannot exceed {MAX_HOOK_SUBSCRIPTIONS} entries"
            )
        subscriptions = tuple(
            HookSubscription.from_mapping(
                _mapping(item, path=f"{path}.subscriptions[{index}]"),
                path=f"{path}.subscriptions[{index}]",
                default_timeout_seconds=default_timeout,
            )
            for index, item in enumerate(raw)
        )
        return cls(
            enabled=value.get("enabled", True),
            default_timeout_seconds=default_timeout,
            subscriptions=subscriptions,
        )


@dataclass(frozen=True, slots=True)
class HookResult:
    """Observability record for one invoked hook action."""

    event: str
    action: str
    status: HookActionStatus
    duration_ms: float
    error: str | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", HookActionStatus(self.status))
        object.__setattr__(self, "detail", _frozen_payload(self.detail, path="hook result detail"))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "action": self.action,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class HookDispatchReport:
    """Aggregate outcome of dispatching one event to every matching hook."""

    schema_version: str = HOOK_SCHEMA_VERSION
    event: str = ""
    results: tuple[HookResult, ...] = ()
    enabled: bool = True
    dry_run: bool = False

    @property
    def dispatched(self) -> int:
        return sum(1 for item in self.results if item.status is HookActionStatus.SUCCEEDED)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.results if item.status is HookActionStatus.FAILED)

    @property
    def timed_out(self) -> int:
        return sum(1 for item in self.results if item.status is HookActionStatus.TIMEOUT)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.timed_out == 0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event": self.event,
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "dispatched": self.dispatched,
            "failed": self.failed,
            "timed_out": self.timed_out,
            "results": [item.to_mapping() for item in self.results],
        }


class HookAction(Protocol):
    """One pluggable hook action.

    Implementations receive the published event and their own validated
    configuration, and return an optional JSON-safe detail mapping that is
    recorded with the result. Raising is allowed: the dispatcher records the
    failure instead of propagating it.
    """

    name: str

    def validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """Validate and normalize this action's configuration block."""

    def __call__(self, event: HookEvent, config: Mapping[str, Any]) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class CallableHookAction:
    """Adapt a plain function into a :class:`HookAction`."""

    name: str
    handler: Callable[[HookEvent, Mapping[str, Any]], Mapping[str, Any] | None]
    config_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _action_name(self.name, path="hook action name"))

    def validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.config_validator is None:
            return MappingProxyType(dict(config))
        return MappingProxyType(dict(self.config_validator(config)))

    def __call__(self, event: HookEvent, config: Mapping[str, Any]) -> Mapping[str, Any] | None:
        return self.handler(event, config)


class HookRegistry:
    """Register actions, hold subscriptions, and dispatch events safely."""

    def __init__(self, config: HookConfig | None = None) -> None:
        self._actions: dict[str, HookAction] = {}
        self._subscriptions: tuple[HookSubscription, ...] = ()
        self._enabled = True
        self._lock = RLock()
        if config is not None:
            self.load(config)

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def action_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._actions))

    @property
    def subscriptions(self) -> tuple[HookSubscription, ...]:
        with self._lock:
            return self._subscriptions

    def register_action(self, action: HookAction) -> None:
        name = _action_name(action.name, path="hook action name")
        with self._lock:
            if name in self._actions:
                raise HookConfigurationError(f"hook action {name!r} is already registered")
            self._actions[name] = action

    def subscribe(self, subscription: HookSubscription) -> HookSubscription:
        """Validate one subscription against the registered actions and store it."""

        resolved = self._resolve(subscription)
        with self._lock:
            self._subscriptions = (*self._subscriptions, resolved)
        return resolved

    def load(self, config: HookConfig) -> None:
        """Replace every subscription with the ones declared in ``config``."""

        resolved = tuple(self._resolve(item) for item in config.subscriptions)
        with self._lock:
            self._subscriptions = resolved
            self._enabled = config.enabled

    def _resolve(self, subscription: HookSubscription) -> HookSubscription:
        with self._lock:
            action = self._actions.get(subscription.action)
        if action is None:
            raise HookConfigurationError(f"unknown hook action {subscription.action!r}")
        validated = action.validate_config(subscription.config)
        return replace(subscription, config=MappingProxyType(dict(validated)))

    def dispatch(self, event: HookEvent, *, dry_run: bool = False) -> HookDispatchReport:
        """Dispatch one event to every matching subscription.

        This method never raises.  A misconfigured, failing, or hanging action
        is reported as a :class:`HookResult` so the publishing run keeps its own
        exit code.
        """

        if not self.enabled:
            return HookDispatchReport(event=event.name, enabled=False, dry_run=dry_run)
        results: list[HookResult] = []
        for subscription in self.subscriptions:
            try:
                matched = subscription.enabled and subscription.matches(event)
            except BaseException as error:
                _LOGGER.warning(
                    "hook subscription for %s could not be evaluated: %s: %s",
                    subscription.event,
                    type(error).__name__,
                    error,
                )
                continue
            if not matched:
                continue
            if dry_run:
                results.append(
                    HookResult(
                        event=event.name,
                        action=subscription.action,
                        status=HookActionStatus.SKIPPED,
                        duration_ms=0.0,
                        detail={"reason": "dry-run"},
                    )
                )
                continue
            results.append(self._invoke(subscription, event))
        return HookDispatchReport(event=event.name, results=tuple(results), dry_run=dry_run)

    def _invoke(self, subscription: HookSubscription, event: HookEvent) -> HookResult:
        started_ns = monotonic_ns()
        state: dict[str, Any] = {"detail": None, "error": None, "status": HookActionStatus.FAILED}

        def target() -> None:
            try:
                action = self._actions[subscription.action]
                state["detail"] = action(event, subscription.config)
                state["status"] = HookActionStatus.SUCCEEDED
            except BaseException as error:
                state["error"] = f"{type(error).__name__}: {error}"

        worker = Thread(
            target=target,
            name=f"glr-hook-{subscription.action}",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException as error:
            return HookResult(
                event=event.name,
                action=subscription.action,
                status=HookActionStatus.FAILED,
                duration_ms=0.0,
                error=f"{type(error).__name__}: {error}",
            )
        worker.join(subscription.timeout_seconds)
        duration_ms = (monotonic_ns() - started_ns) / 1_000_000
        if worker.is_alive():
            _LOGGER.warning(
                "hook action %s exceeded its %.1fs budget for event %s",
                subscription.action,
                subscription.timeout_seconds,
                event.name,
            )
            return HookResult(
                event=event.name,
                action=subscription.action,
                status=HookActionStatus.TIMEOUT,
                duration_ms=duration_ms,
                error=(
                    f"hook action exceeded {subscription.timeout_seconds:.1f}s "
                    "and was left running in the background"
                ),
            )
        status = HookActionStatus(state["status"])
        failure = state["error"]
        detail = state["detail"]
        if status is HookActionStatus.FAILED:
            _LOGGER.warning(
                "hook action %s failed for event %s after %.1f ms: %s",
                subscription.action,
                event.name,
                duration_ms,
                failure,
            )
        else:
            _LOGGER.info(
                "hook action %s handled event %s in %.1f ms",
                subscription.action,
                event.name,
                duration_ms,
            )
        return HookResult(
            event=event.name,
            action=subscription.action,
            status=status,
            duration_ms=duration_ms,
            error=failure,
            detail=_recordable_detail(detail),
        )


__all__ = [
    "DEFAULT_HOOK_TIMEOUT_SECONDS",
    "DEFAULT_MESSAGE_TEMPLATE",
    "HOOK_SCHEMA_VERSION",
    "MAX_HOOK_SUBSCRIPTIONS",
    "MAX_HOOK_TIMEOUT_SECONDS",
    "MAX_MESSAGE_TEMPLATE_LENGTH",
    "MESSAGE_TEMPLATE_FIELDS",
    "MIN_HOOK_TIMEOUT_SECONDS",
    "PREDEFINED_HOOK_EVENTS",
    "CallableHookAction",
    "HookAction",
    "HookActionStatus",
    "HookConfig",
    "HookConfigurationError",
    "HookDispatchReport",
    "HookEvent",
    "HookEventFilter",
    "HookEventStatus",
    "HookRegistry",
    "HookResult",
    "HookSubscription",
    "render_message",
    "validate_message_template",
]
