"""Built-in lifecycle hook actions shipped with the control plane.

Three actions are registered by default so a fresh project can notify on a
lifecycle event without writing code:

``notify.log``
    Emit one structured log record.
``notify.message``
    Append one JSON Lines message to a project-relative outbox file.
``notify.webhook``
    POST the structured event to one configured HTTP endpoint.

Anything else is registered by the caller through
:meth:`game_learning_runtime.hooks.HookRegistry.register_action`; this module
deliberately carries no business logic beyond delivering one message.
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlparse

from game_learning_runtime.hooks import (
    DEFAULT_MESSAGE_TEMPLATE,
    HOOK_SCHEMA_VERSION,
    HookConfigurationError,
    HookEvent,
    HookEventStatus,
    HookRegistry,
    render_message,
    validate_message_template,
)

_LOGGER = logging.getLogger("game_learning_runtime.hooks")

LOG_ACTION = "notify.log"
MESSAGE_ACTION = "notify.message"
WEBHOOK_ACTION = "notify.webhook"

DEFAULT_OUTBOX = "hooks/messages.jsonl"
MAX_WEBHOOK_URL_LENGTH = 2048
MAX_WEBHOOK_TIMEOUT_SECONDS = 30.0
_MIN_WEBHOOK_TIMEOUT_SECONDS = 0.1

_LOG_LEVELS: Mapping[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


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


def _resolve_inside(base_dir: Path, relative: object, *, path: str) -> Path:
    """Resolve a portable project-relative path that stays inside ``base_dir``."""

    if not isinstance(relative, str) or not relative:
        raise HookConfigurationError(f"{path} must be a non-empty relative path")
    if "\\" in relative or ":" in relative:
        raise HookConfigurationError(f"{path} must be a portable project-relative path")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise HookConfigurationError(f"{path} must be a portable project-relative path")
    root = base_dir.resolve()
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise HookConfigurationError(f"{path} must not traverse a symlink")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise HookConfigurationError(f"{path} must stay inside the project root")
    return resolved


def _validate_timeout(value: object, *, path: str, default: float, maximum: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HookConfigurationError(f"{path} must be a number")
    resolved = float(value)
    if not math.isfinite(resolved) or not _MIN_WEBHOOK_TIMEOUT_SECONDS <= resolved <= maximum:
        raise HookConfigurationError(
            f"{path} must be between {_MIN_WEBHOOK_TIMEOUT_SECONDS} and {maximum} seconds"
        )
    return resolved


class LogHookAction:
    """Emit one structured log record for a lifecycle event."""

    name = LOG_ACTION

    def validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        _reject_unknown(config, allowed=frozenset({"message", "level"}), path="notify.log config")
        level = str(config.get("level", "info"))
        if level not in _LOG_LEVELS:
            raise HookConfigurationError(f"notify.log level must be one of: {sorted(_LOG_LEVELS)}")
        message = validate_message_template(
            config.get("message", DEFAULT_MESSAGE_TEMPLATE), path="notify.log message"
        )
        return {"message": message, "level": level}

    def __call__(self, event: HookEvent, config: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = self.validate_config(config)
        message = render_message(str(validated["message"]), event, path="notify.log message")
        level_name = str(validated["level"])
        _LOGGER.log(_LOG_LEVELS[level_name], "hook %s: %s", event.name, message)
        return {"message": message, "level": level_name}


class MessageOutboxAction:
    """Append one JSON Lines message to a project-relative outbox file.

    The outbox is the offline notification lane: a scheduler, agent, or mailer
    reads it without GLR knowing anything about the transport.
    """

    name = MESSAGE_ACTION

    def __init__(self, base_dir: Path | str) -> None:
        self._base_dir = Path(base_dir)

    def validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        _reject_unknown(
            config, allowed=frozenset({"outbox", "message"}), path="notify.message config"
        )
        outbox = config.get("outbox", DEFAULT_OUTBOX)
        if not isinstance(outbox, str) or not outbox:
            raise HookConfigurationError("notify.message outbox must be a non-empty path")
        _resolve_inside(self._base_dir, outbox, path="notify.message outbox")
        message = validate_message_template(
            config.get("message", DEFAULT_MESSAGE_TEMPLATE), path="notify.message message"
        )
        return {"outbox": outbox, "message": message}

    def __call__(self, event: HookEvent, config: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = self.validate_config(config)
        path = _resolve_inside(self._base_dir, validated["outbox"], path="notify.message outbox")
        message = render_message(str(validated["message"]), event, path="notify.message message")
        record = {
            "schema_version": HOOK_SCHEMA_VERSION,
            "occurred_at_ns": event.occurred_at_ns,
            "event": event.name,
            "status": HookEventStatus(event.status).value,
            "environment_id": event.environment_id,
            "kind": event.kind,
            "stage": event.stage,
            "run_id": event.run_id,
            "exit_code": event.exit_code,
            "reason": event.reason,
            "message": message,
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
        return {
            "outbox": path.relative_to(self._base_dir.resolve()).as_posix(),
            "bytes": len(line.encode("utf-8")),
        }


class WebhookTransport(Protocol):
    """Deliver one JSON body to one endpoint and return the HTTP status."""

    def __call__(self, *, url: str, body: Mapping[str, Any], timeout_seconds: float) -> int: ...


def urllib_webhook_transport(*, url: str, body: Mapping[str, Any], timeout_seconds: float) -> int:
    """POST one JSON body with the standard library and return the HTTP status."""

    payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return int(getattr(response, "status", 0) or 0)
    except urllib.error.HTTPError as error:
        return int(error.code)


@dataclass(frozen=True, slots=True)
class WebhookRecordedCall:
    """One recorded delivery, used by tests and offline integrations."""

    url: str
    body: Mapping[str, Any]
    timeout_seconds: float


class RecordingWebhookTransport:
    """Capture deliveries instead of performing them."""

    def __init__(self, *, status: int = 200) -> None:
        self._status = status
        self.calls: list[WebhookRecordedCall] = []

    def __call__(self, *, url: str, body: Mapping[str, Any], timeout_seconds: float) -> int:
        self.calls.append(WebhookRecordedCall(url=url, body=body, timeout_seconds=timeout_seconds))
        return self._status


class WebhookHookAction:
    """POST the structured event to one configured HTTP endpoint.

    Only ``http`` and ``https`` URLs without embedded credentials are accepted;
    authenticated transports belong in a registered custom action so secrets
    never enter the project manifest.
    """

    name = WEBHOOK_ACTION

    def __init__(self, transport: WebhookTransport | None = None) -> None:
        self._transport: WebhookTransport = transport or urllib_webhook_transport

    def validate_config(self, config: Mapping[str, Any]) -> Mapping[str, Any]:
        _reject_unknown(
            config,
            allowed=frozenset({"url", "timeout_seconds", "message"}),
            path="notify.webhook config",
        )
        url = config.get("url")
        if not isinstance(url, str) or not url:
            raise HookConfigurationError("notify.webhook url is required")
        if len(url) > MAX_WEBHOOK_URL_LENGTH or any(ord(character) < 32 for character in url):
            raise HookConfigurationError("notify.webhook url is not a printable URL")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HookConfigurationError("notify.webhook url must be an http or https URL")
        if parsed.username or parsed.password:
            raise HookConfigurationError(
                "notify.webhook url must not embed credentials; register a custom action instead"
            )
        timeout = _validate_timeout(
            config.get("timeout_seconds"),
            path="notify.webhook timeout_seconds",
            default=5.0,
            maximum=MAX_WEBHOOK_TIMEOUT_SECONDS,
        )
        message = (
            None
            if config.get("message") is None
            else validate_message_template(config["message"], path="notify.webhook message")
        )
        return {"url": url, "timeout_seconds": timeout, "message": message}

    def __call__(self, event: HookEvent, config: Mapping[str, Any]) -> Mapping[str, Any]:
        validated = self.validate_config(config)
        body: dict[str, Any] = dict(event.to_mapping())
        if validated["message"] is not None:
            body["message"] = render_message(
                str(validated["message"]), event, path="notify.webhook message"
            )
        status = self._transport(
            url=str(validated["url"]),
            body=body,
            timeout_seconds=float(validated["timeout_seconds"]),
        )
        if not 200 <= status < 300:
            raise HookConfigurationError(f"webhook endpoint responded with HTTP {status}")
        return {"status": status, "delivered": True}


def register_builtin_actions(
    registry: HookRegistry,
    *,
    base_dir: Path | str,
    webhook_transport: WebhookTransport | None = None,
) -> HookRegistry:
    """Register the shipped notification actions on ``registry``."""

    registry.register_action(LogHookAction())
    registry.register_action(MessageOutboxAction(base_dir))
    registry.register_action(WebhookHookAction(transport=webhook_transport))
    return registry


__all__ = [
    "DEFAULT_OUTBOX",
    "LOG_ACTION",
    "MAX_WEBHOOK_TIMEOUT_SECONDS",
    "MAX_WEBHOOK_URL_LENGTH",
    "MESSAGE_ACTION",
    "WEBHOOK_ACTION",
    "LogHookAction",
    "MessageOutboxAction",
    "RecordingWebhookTransport",
    "WebhookHookAction",
    "WebhookRecordedCall",
    "WebhookTransport",
    "register_builtin_actions",
    "urllib_webhook_transport",
]
