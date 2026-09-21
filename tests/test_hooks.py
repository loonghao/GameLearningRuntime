from __future__ import annotations

import email.message
import json
import logging
import sys
import time
import urllib.error
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from game_learning_runtime.cli import main
from game_learning_runtime.hook_actions import (
    LogHookAction,
    MessageOutboxAction,
    NoRedirectHandler,
    RecordingWebhookTransport,
    WebhookHookAction,
    register_builtin_actions,
    urllib_webhook_transport,
)
from game_learning_runtime.hooks import (
    DEFAULT_MESSAGE_TEMPLATE,
    HOOK_SCHEMA_VERSION,
    MAX_HOOK_SUBSCRIPTIONS,
    CallableHookAction,
    HookActionStatus,
    HookConfig,
    HookConfigurationError,
    HookEvent,
    HookEventFilter,
    HookEventStatus,
    HookRegistry,
    HookSubscription,
    render_message,
    validate_message_template,
)
from game_learning_runtime.project import load_project
from game_learning_runtime.run_store import RunStatus, TrainingStore


def _project(
    root: Path,
    *,
    trainer_argv: list[str] | None = None,
    capture_argv: list[str] | None = None,
    hooks: dict[str, Any] | None = None,
) -> None:
    (root / "bridge").mkdir()
    value: dict[str, object] = {
        "schema_version": "glr.project.v1",
        "environment_id": "example.adventure-v1",
        "environment_family": "action-rpg",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": [sys.executable, "-c", "print('runtime')"]},
        "trainer": {"argv": trainer_argv or [sys.executable, "-c", "print('train')"]},
        "player": {"argv": [sys.executable, "-c", "print('play')", "{bundle}"]},
        "capture": (
            None
            if capture_argv is None
            else {
                "argv": capture_argv,
                "required": True,
                "stop": "stdin-q",
                "video_file": "capture.mp4",
                "index_file": "capture-index.jsonl",
                "codec": "h264",
                "frame_rate": 12,
                "width": 640,
                "height": 360,
            }
        ),
    }
    if hooks is not None:
        value["hooks"] = hooks
    (root / "glr-project.json").write_text(json.dumps(value), encoding="utf-8")


def _event(name: str = "train.failed", **overrides: Any) -> HookEvent:
    defaults: dict[str, Any] = {
        "name": name,
        "status": HookEventStatus.FAILED,
        "environment_id": "example.adventure-v1",
        "environment_family": "action-rpg",
        "kind": "training",
        "stage": "trainer",
        "run_id": "run-1",
        "exit_code": 7,
        "reason": "trainer crashed",
    }
    defaults.update(overrides)
    return HookEvent(**defaults)


def _registry(*subscriptions: HookSubscription, base_dir: Path | None = None) -> HookRegistry:
    registry = HookRegistry()
    register_builtin_actions(registry, base_dir=base_dir or Path.cwd())
    for subscription in subscriptions:
        registry.subscribe(subscription)
    return registry


# --------------------------------------------------------------------------- #
# Event and filter contracts
# --------------------------------------------------------------------------- #


def test_hook_event_rejects_malformed_names_and_normalizes_status() -> None:
    with pytest.raises(HookConfigurationError, match="lowercase dotted identifier"):
        HookEvent(name="Train.failed")
    with pytest.raises(ValueError, match="not a valid HookEventStatus"):
        HookEvent(name="train.failed", status="exploded")
    event = HookEvent(name="train.failed", status="failed", payload={"nested": {"a": 1}})
    assert event.status is HookEventStatus.FAILED
    assert event.payload == {"nested": {"a": 1}}
    assert event.to_mapping()["schema_version"] == HOOK_SCHEMA_VERSION
    assert event.to_mapping()["status"] == "failed"


def test_hook_event_rejects_non_serializable_payload() -> None:
    with pytest.raises(HookConfigurationError, match="JSON serializable"):
        HookEvent(name="train.failed", payload={"bad": object()})


def test_filter_matches_environment_kind_stage_status_and_exit_code_range() -> None:
    event = _event()
    assert HookEventFilter().matches(event)
    assert not HookEventFilter(environment_id="other.env").matches(event)
    assert not HookEventFilter(environment_family="racing").matches(event)
    assert not HookEventFilter(kind="record").matches(event)
    assert not HookEventFilter(stage="capture").matches(event)
    assert not HookEventFilter(status=HookEventStatus.SUCCEEDED).matches(event)
    assert HookEventFilter(exit_code_min=0, exit_code_max=10).matches(event)
    assert not HookEventFilter(exit_code_min=8).matches(event)
    assert not HookEventFilter(exit_code_max=6).matches(event)


def test_exit_code_filter_rejects_events_without_an_exit_code() -> None:
    assert not HookEventFilter(exit_code_min=1).matches(_event(exit_code=None))


def test_filter_rejects_an_inverted_exit_code_range() -> None:
    with pytest.raises(HookConfigurationError, match="exit_code_min cannot exceed"):
        HookEventFilter(exit_code_min=5, exit_code_max=1)


def test_subscription_matches_custom_events_and_namespace_wildcards() -> None:
    exact = HookSubscription(event="adapter.ready", action="notify.log")
    assert exact.matches(HookEvent(name="adapter.ready"))
    assert not exact.matches(HookEvent(name="adapter.notready"))

    wildcard = HookSubscription(event="train.*", action="notify.log")
    assert wildcard.matches(HookEvent(name="train.complete"))
    assert not wildcard.matches(HookEvent(name="record.start"))
    assert wildcard.namespace == "train."


def test_subscription_rejects_malformed_selector_action_and_timeout() -> None:
    with pytest.raises(HookConfigurationError, match="hook event"):
        HookSubscription(event="train.!bad", action="notify.log")
    with pytest.raises(HookConfigurationError, match="hook action"):
        HookSubscription(event="train.failed", action="Notify.Log")
    with pytest.raises(HookConfigurationError, match="timeout_seconds"):
        HookSubscription(event="train.failed", action="notify.log", timeout_seconds=0)


# --------------------------------------------------------------------------- #
# Message templates
# --------------------------------------------------------------------------- #


def test_message_template_rejects_unknown_placeholders_and_control_characters() -> None:
    with pytest.raises(HookConfigurationError, match="unsupported placeholders"):
        validate_message_template("{event} {secret}", path="message")
    with pytest.raises(HookConfigurationError, match="control characters"):
        validate_message_template("line\nbreak", path="message")


def test_render_message_substitutes_known_fields_literally() -> None:
    rendered = render_message("{event} {status} {exit_code} {reason} {run_id}", _event())
    assert rendered == "train.failed failed 7 trainer crashed run-1"
    assert render_message(DEFAULT_MESSAGE_TEMPLATE, _event(reason=None, run_id=None)) == (
        "train.failed failed for example.adventure-v1 (run )"
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_hook_config_parses_subscriptions_and_applies_the_default_timeout() -> None:
    config = HookConfig.from_mapping(
        {
            "default_timeout_seconds": 2.5,
            "subscriptions": [
                {
                    "event": "train.failed",
                    "action": "notify.message",
                    "when": {"kind": "training", "exit_code_min": 1},
                    "config": {"outbox": "hooks/messages.jsonl"},
                },
                {"event": "record.*", "action": "notify.log", "timeout_seconds": 1.0},
            ],
        }
    )
    assert config.enabled is True
    assert config.default_timeout_seconds == 2.5
    assert config.subscriptions[0].timeout_seconds == 2.5
    assert config.subscriptions[1].timeout_seconds == 1.0
    assert config.subscriptions[0].when.kind == "training"
    assert config.subscriptions[0].when.exit_code_min == 1
    assert config.subscriptions[0].config == {"outbox": "hooks/messages.jsonl"}
    assert config.to_mapping()["schema_version"] == HOOK_SCHEMA_VERSION


def test_hook_config_rejects_unknown_fields_and_oversized_subscription_lists() -> None:
    with pytest.raises(HookConfigurationError, match="unexpected fields"):
        HookConfig.from_mapping({"surprise": True})
    with pytest.raises(HookConfigurationError, match="cannot exceed"):
        HookConfig.from_mapping(
            {
                "subscriptions": [
                    {"event": "train.failed", "action": "notify.log"}
                    for _ in range(MAX_HOOK_SUBSCRIPTIONS + 1)
                ]
            }
        )


def test_hook_subscription_requires_event_and_action() -> None:
    with pytest.raises(HookConfigurationError, match="requires event and action"):
        HookSubscription.from_mapping({"action": "notify.log"}, path="hooks.subscriptions[0]")


# --------------------------------------------------------------------------- #
# Registry and dispatch
# --------------------------------------------------------------------------- #


def test_registry_rejects_duplicate_and_unknown_actions() -> None:
    registry = _registry()
    with pytest.raises(HookConfigurationError, match="already registered"):
        registry.register_action(LogHookAction())
    with pytest.raises(HookConfigurationError, match="unknown hook action"):
        registry.subscribe(HookSubscription(event="train.failed", action="notify.smoke"))


def test_registry_lists_registered_actions_in_stable_order() -> None:
    assert _registry().action_names == ("notify.log", "notify.message", "notify.webhook")


def test_callable_hook_action_adapts_a_plain_function() -> None:
    seen: list[str] = []

    def handler(event: HookEvent, config: Mapping[str, Any]) -> Mapping[str, Any]:
        seen.append(f"{event.name}:{config['label']}")
        return {"seen": len(seen)}

    registry = HookRegistry()
    registry.register_action(
        CallableHookAction(
            name="custom.trace",
            handler=handler,
            config_validator=lambda config: {"label": str(config.get("label", "default"))},
        )
    )
    registry.subscribe(
        HookSubscription(event="adapter.ready", action="custom.trace", config={"label": "alpha"})
    )
    report = registry.dispatch(HookEvent(name="adapter.ready", status="succeeded"))
    assert seen == ["adapter.ready:alpha"]
    assert report.dispatched == 1
    assert report.results[0].detail == {"seen": 1}


def test_dispatch_records_a_raising_action_without_propagating() -> None:
    def explode(event: HookEvent, config: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError("notification backend is down")

    registry = HookRegistry()
    registry.register_action(CallableHookAction(name="custom.boom", handler=explode))
    registry.subscribe(HookSubscription(event="train.*", action="custom.boom"))

    report = registry.dispatch(_event())

    assert report.failed == 1
    assert report.ok is False
    result = report.results[0]
    assert result.status is HookActionStatus.FAILED
    assert result.error is not None and "notification backend is down" in result.error
    assert result.duration_ms >= 0


def test_dispatch_reports_a_timed_out_action_and_keeps_the_budget() -> None:
    registry = HookRegistry()
    registry.register_action(
        CallableHookAction(name="custom.slow", handler=lambda event, config: time.sleep(5))
    )
    registry.subscribe(
        HookSubscription(event="train.failed", action="custom.slow", timeout_seconds=0.2)
    )

    started = time.monotonic()
    report = registry.dispatch(_event())
    elapsed = time.monotonic() - started

    assert elapsed < 3
    assert report.timed_out == 1
    assert report.results[0].status is HookActionStatus.TIMEOUT
    assert "exceeded" in (report.results[0].error or "")


def test_dispatch_skips_disabled_subscriptions_and_disabled_registries() -> None:
    registry = _registry(HookSubscription(event="train.failed", action="notify.log", enabled=False))
    assert registry.dispatch(_event()).results == ()

    disabled = _registry(HookSubscription(event="train.failed", action="notify.log"))
    disabled.load(HookConfig(enabled=False, subscriptions=disabled.subscriptions))
    report = disabled.dispatch(_event())
    assert report.enabled is False
    assert report.results == ()


def test_dispatch_dry_run_reports_matches_without_running_actions() -> None:
    registry = _registry(HookSubscription(event="train.failed", action="notify.log"))
    report = registry.dispatch(_event(), dry_run=True)
    assert report.dry_run is True
    assert report.results[0].status is HookActionStatus.SKIPPED
    assert report.results[0].detail == {"reason": "dry-run"}
    assert report.dispatched == 0


def test_dispatch_never_raises_when_a_subscription_matching_fails() -> None:
    class BrokenFilter(HookEventFilter):
        def matches(self, event: HookEvent) -> bool:
            raise RuntimeError("filter is broken")

    registry = _registry()
    subscription = HookSubscription(event="train.failed", action="notify.log")
    object.__setattr__(subscription, "when", BrokenFilter())
    registry._subscriptions = (subscription,)  # type: ignore[attr-defined]
    assert registry.dispatch(_event()).results == ()


# --------------------------------------------------------------------------- #
# Built-in actions
# --------------------------------------------------------------------------- #


def test_log_action_emits_one_structured_record(tmp_path: Path, caplog: object) -> None:
    action = LogHookAction()
    validated = action.validate_config({"message": "{event} -> {reason}", "level": "warning"})
    assert validated["message"] == "{event} -> {reason}"
    with caplog.at_level(logging.DEBUG, logger="game_learning_runtime.hooks"):  # type: ignore[attr-defined]
        detail = action(_event(), validated)
    assert detail == {"message": "train.failed -> trainer crashed", "level": "warning"}
    assert "train.failed -> trainer crashed" in caplog.text  # type: ignore[attr-defined]


def test_log_action_rejects_unsupported_levels() -> None:
    with pytest.raises(HookConfigurationError, match="level must be one of"):
        LogHookAction().validate_config({"level": "critical"})


def test_message_action_appends_one_json_line_inside_the_project(tmp_path: Path) -> None:
    action = MessageOutboxAction(tmp_path)
    config = action.validate_config({"outbox": "hooks/messages.jsonl"})
    detail = action(_event(), config)
    assert detail["outbox"] == "hooks/messages.jsonl"
    lines = (tmp_path / "hooks/messages.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "train.failed"
    assert record["exit_code"] == 7
    assert record["reason"] == "trainer crashed"
    assert record["message"]


def test_message_action_rejects_paths_outside_the_project(tmp_path: Path) -> None:
    action = MessageOutboxAction(tmp_path)
    for candidate in ("../outside.jsonl", "/absolute.jsonl", "C:/windows.jsonl"):
        with pytest.raises(HookConfigurationError, match="project-relative"):
            action.validate_config({"outbox": candidate})
    with pytest.raises(HookConfigurationError, match="unexpected fields"):
        action.validate_config({"outbox": "hooks/messages.jsonl", "secret": True})


def test_webhook_action_posts_the_structured_event() -> None:
    transport = RecordingWebhookTransport()
    action = WebhookHookAction(transport=transport)
    config = action.validate_config({"url": "https://hooks.example.test/glr", "message": "{event}"})
    detail = action(_event(), config)
    assert detail == {"status": 200, "delivered": True}
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call.url == "https://hooks.example.test/glr"
    assert call.body["event"] == "train.failed"
    assert call.body["exit_code"] == 7
    assert call.body["message"] == "train.failed"


def test_webhook_action_rejects_credentials_and_non_http_urls() -> None:
    action = WebhookHookAction(transport=RecordingWebhookTransport())
    for candidate in (
        "ftp://hooks.example.test/glr",
        "https://user:pass@hooks.example.test/glr",
        "not-a-url",
    ):
        with pytest.raises(HookConfigurationError):
            action.validate_config({"url": candidate})
    with pytest.raises(HookConfigurationError, match="url is required"):
        action.validate_config({})


def test_message_action_rejects_an_empty_or_oversized_outbox(tmp_path: Path) -> None:
    action = MessageOutboxAction(tmp_path)
    with pytest.raises(HookConfigurationError, match="non-empty path"):
        action.validate_config({"outbox": ""})
    with pytest.raises(HookConfigurationError, match="portable project-relative"):
        action.validate_config({"outbox": "hooks/../../escape.jsonl"})


def test_message_action_rejects_a_symlinked_outbox(tmp_path: Path) -> None:
    (tmp_path / "hooks").mkdir()
    target = tmp_path / "outside.jsonl"
    target.write_text("", encoding="utf-8")
    (tmp_path / "hooks" / "messages.jsonl").symlink_to(target)
    with pytest.raises(HookConfigurationError, match="must not traverse a symlink"):
        MessageOutboxAction(tmp_path)(_event(), {"outbox": "hooks/messages.jsonl"})


def test_webhook_action_rejects_malformed_urls_and_timeouts() -> None:
    action = WebhookHookAction(transport=RecordingWebhookTransport())
    with pytest.raises(HookConfigurationError, match="printable URL"):
        action.validate_config({"url": "https://hooks.example.test/" + "a" * 3000})
    with pytest.raises(HookConfigurationError, match="must be a number"):
        action.validate_config({"url": "https://hooks.example.test/glr", "timeout_seconds": "soon"})
    with pytest.raises(HookConfigurationError, match="between"):
        action.validate_config({"url": "https://hooks.example.test/glr", "timeout_seconds": 0})
    with pytest.raises(HookConfigurationError, match="unexpected fields"):
        action.validate_config({"url": "https://hooks.example.test/glr", "secret": "token"})


def test_urllib_webhook_transport_posts_json_and_reads_the_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        status = 202

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class _FakeOpener:
        def __init__(self, *handlers: object) -> None:
            captured["handlers"] = handlers

        def open(self, request: object, timeout: float) -> _FakeResponse:
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse()

    monkeypatch.setattr("urllib.request.build_opener", _FakeOpener)
    status = urllib_webhook_transport(
        url="https://hooks.example.test/glr", body={"event": "train.failed"}, timeout_seconds=2.5
    )
    assert status == 202
    assert captured["timeout"] == 2.5
    request = captured["request"]
    assert request.data == b'{"event": "train.failed"}'  # type: ignore[attr-defined]
    assert request.get_method() == "POST"  # type: ignore[attr-defined]
    assert request.headers == {"Content-type": "application/json"}  # type: ignore[attr-defined]
    assert captured["handlers"] == (NoRedirectHandler,)


def test_urllib_webhook_transport_refuses_redirects() -> None:
    handler = NoRedirectHandler()
    with pytest.raises(urllib.error.HTTPError, match="redirects are not followed"):
        handler.redirect_request(
            urllib.request.Request("https://hooks.example.test/glr"),
            None,
            302,
            "Found",
            email.message.Message(),
            "https://internal.example.test/admin",
        )


def test_urllib_webhook_transport_reports_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeOpener:
        def __init__(self, *handlers: object) -> None:
            self.handlers = handlers

        def open(self, request: object, timeout: float) -> object:
            raise urllib.error.HTTPError(
                "https://hooks.example.test/glr",
                500,
                "boom",
                {},
                None,  # type: ignore[arg-type]
            )

    monkeypatch.setattr("urllib.request.build_opener", _FakeOpener)
    assert (
        urllib_webhook_transport(url="https://hooks.example.test/glr", body={}, timeout_seconds=1.0)
        == 500
    )


def test_webhook_action_reports_a_non_success_status_as_a_failure() -> None:
    action = WebhookHookAction(transport=RecordingWebhookTransport(status=503))
    with pytest.raises(HookConfigurationError, match="HTTP 503"):
        action(_event(), action.validate_config({"url": "https://hooks.example.test/glr"}))


# --------------------------------------------------------------------------- #
# Project configuration
# --------------------------------------------------------------------------- #


def test_project_loads_hook_configuration(tmp_path: Path) -> None:
    _project(
        tmp_path,
        hooks={
            "default_timeout_seconds": 3.0,
            "subscriptions": [
                {
                    "event": "train.complete",
                    "action": "notify.message",
                    "config": {"outbox": "hooks/messages.jsonl"},
                }
            ],
        },
    )
    project = load_project(tmp_path)
    assert project.hooks.enabled is True
    assert project.hooks.default_timeout_seconds == 3.0
    assert [item.event for item in project.hooks.subscriptions] == ["train.complete"]


def test_project_without_hooks_gets_an_empty_default(tmp_path: Path) -> None:
    _project(tmp_path)
    assert load_project(tmp_path).hooks == HookConfig()


def test_project_rejects_a_malformed_hook_configuration(tmp_path: Path) -> None:
    _project(tmp_path, hooks={"subscriptions": [{"event": "train.failed"}]})
    with pytest.raises(HookConfigurationError):
        load_project(tmp_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_hooks_list_reports_actions_and_subscriptions(tmp_path: Path, capsys: object) -> None:
    _project(
        tmp_path,
        hooks={
            "subscriptions": [
                {"event": "train.*", "action": "notify.message", "when": {"status": "failed"}}
            ]
        },
    )

    assert main(["--project", str(tmp_path), "--format", "json", "hooks", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    data = payload["data"]
    assert payload["command"] == "hooks.list"
    assert data["actions"] == ["notify.log", "notify.message", "notify.webhook"]
    assert data["subscriptions"][0]["event"] == "train.*"
    assert data["subscriptions"][0]["when"]["status"] == "failed"
    assert "train.failed" in data["predefined_events"]


def test_cli_hooks_emit_dispatches_and_reports_the_outcome(tmp_path: Path, capsys: object) -> None:
    _project(
        tmp_path,
        hooks={
            "subscriptions": [
                {
                    "event": "train.failed",
                    "action": "notify.message",
                    "config": {"outbox": "hooks/messages.jsonl"},
                }
            ]
        },
    )
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "--format",
                "json",
                "hooks",
                "emit",
                "--event",
                "train.failed",
                "--status",
                "failed",
                "--kind",
                "training",
                "--stage",
                "trainer",
                "--exit-code",
                "4",
                "--reason",
                "out of memory",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["data"]["dispatched"] == 1
    assert payload["data"]["results"][0]["action"] == "notify.message"
    record = json.loads(
        (tmp_path / "hooks/messages.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["reason"] == "out of memory"
    assert record["exit_code"] == 4


def test_cli_hooks_emit_dry_run_does_not_side_effect(tmp_path: Path, capsys: object) -> None:
    _project(
        tmp_path,
        hooks={"subscriptions": [{"event": "train.failed", "action": "notify.message"}]},
    )
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "--format",
                "json",
                "hooks",
                "emit",
                "--event",
                "train.failed",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["data"]["dry_run"] is True
    assert payload["data"]["results"][0]["status"] == "skipped"
    assert not (tmp_path / "hooks").exists()


def test_cli_hooks_emit_returns_one_when_an_action_fails(tmp_path: Path, capsys: object) -> None:
    _project(
        tmp_path,
        hooks={
            "subscriptions": [
                {
                    "event": "train.failed",
                    "action": "notify.message",
                    "config": {"outbox": "bridge"},
                }
            ]
        },
    )
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "--format",
                "json",
                "hooks",
                "emit",
                "--event",
                "train.failed",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["data"]["failed"] == 1


def test_cli_hooks_list_rejects_an_unknown_action(tmp_path: Path) -> None:
    _project(
        tmp_path, hooks={"subscriptions": [{"event": "train.failed", "action": "notify.nope"}]}
    )
    with pytest.raises(HookConfigurationError, match="unknown hook action"):
        main(["--project", str(tmp_path), "--format", "json", "hooks", "list"])


def test_cli_train_failure_emits_structured_machine_readable_output(
    tmp_path: Path, capsys: object
) -> None:
    _project(
        tmp_path,
        trainer_argv=[sys.executable, "-c", "import sys; sys.exit(7)"],
        hooks={
            "subscriptions": [
                {
                    "event": "train.failed",
                    "action": "notify.message",
                    "config": {"outbox": "hooks/messages.jsonl"},
                }
            ]
        },
    )

    exit_code = main(["--project", str(tmp_path), "--format", "json", "train"])
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert exit_code == 7
    failure = payload["data"]["failure"]
    assert failure["stage"] == "trainer"
    assert failure["reason"] == "trainer command exited with code 7"
    assert failure["exit_code"] == 7
    record = json.loads(
        (tmp_path / "hooks/messages.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["event"] == "train.failed"
    assert record["exit_code"] == 7


def test_cli_train_records_hook_results_as_run_events(tmp_path: Path, capsys: object) -> None:
    _project(
        tmp_path,
        hooks={
            "subscriptions": [
                {"event": "train.start", "action": "notify.log"},
                {"event": "train.complete", "action": "notify.message"},
            ]
        },
    )

    assert main(["--project", str(tmp_path), "--format", "json", "train"]) == 0
    run_id = json.loads(capsys.readouterr().out)["data"]["run_id"]  # type: ignore[attr-defined]
    events = [
        event.payload
        for event in TrainingStore(tmp_path / ".glr/runs.sqlite3").list_events(run_id)
        if event.kind == "hook.dispatched"
    ]
    assert [item["event"] for item in events] == ["train.start", "train.complete"]
    assert events[1]["results"][0]["action"] == "notify.message"
    assert events[1]["results"][0]["status"] == "succeeded"


def test_cli_train_keeps_its_exit_code_when_a_hook_fails(tmp_path: Path, capsys: object) -> None:
    _project(
        tmp_path,
        trainer_argv=[sys.executable, "-c", "import sys; sys.exit(9)"],
        hooks={
            "subscriptions": [
                {
                    "event": "train.failed",
                    "action": "notify.message",
                    "config": {"outbox": "bridge"},
                }
            ]
        },
    )

    assert main(["--project", str(tmp_path), "--format", "json", "train"]) == 9
    run_id = json.loads(capsys.readouterr().out)["data"]["run_id"]  # type: ignore[attr-defined]
    events = [
        event.payload
        for event in TrainingStore(tmp_path / ".glr/runs.sqlite3").list_events(run_id)
        if event.kind == "hook.dispatched"
    ]
    assert events[-1]["failed"] == 1
    assert "bridge" in events[-1]["results"][0]["error"]


def test_cli_train_ignores_a_broken_hook_configuration(tmp_path: Path, capsys: object) -> None:
    _project(
        tmp_path,
        hooks={"subscriptions": [{"event": "train.start", "action": "notify.missing"}]},
    )

    assert main(["--project", str(tmp_path), "--format", "json", "train"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]["status"] == "succeeded"  # type: ignore[attr-defined]


def test_cli_train_publishes_record_lifecycle_events(tmp_path: Path, capsys: object) -> None:
    recorder = tmp_path / "recorder.py"
    recorder.write_text(
        """
import json
import os
import sys
from pathlib import Path

Path(os.environ["GLR_CAPTURE_VIDEO"]).write_bytes(b"synthetic-h264")
Path(os.environ["GLR_CAPTURE_INDEX"]).write_text(json.dumps({
    "schema_version": "glr.capture-frame.v1",
    "run_id": os.environ["GLR_RUN_ID"],
    "episode_id": "12345678-1234-5678-1234-567812345678",
    "step_id": 0,
    "frame_index": 0,
    "pts_ns": 0,
    "observation_timestamp_ns": 1
}) + "\\n", encoding="utf-8")
for line in sys.stdin:
    if line.strip() == "q":
        break
""".strip()
        + "\n",
        encoding="utf-8",
    )
    trainer = tmp_path / "capture_trainer.py"
    trainer.write_text(
        """
import os
import time
from pathlib import Path

video = Path(os.environ["GLR_CAPTURE_VIDEO"])
deadline = time.monotonic() + 5
while not video.is_file() and time.monotonic() < deadline:
    time.sleep(0.01)
assert video.is_file(), "recorder did not start concurrently"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _project(
        tmp_path,
        trainer_argv=[sys.executable, str(trainer)],
        capture_argv=[sys.executable, str(recorder)],
        hooks={
            "subscriptions": [
                {
                    "event": "record.*",
                    "action": "notify.message",
                    "config": {"outbox": "hooks/m.jsonl"},
                }
            ]
        },
    )

    assert main(["--project", str(tmp_path), "--format", "json", "train"]) == 0
    run_id = json.loads(capsys.readouterr().out)["data"]["run_id"]  # type: ignore[attr-defined]
    events = [
        event.payload
        for event in TrainingStore(tmp_path / ".glr/runs.sqlite3").list_events(run_id)
        if event.kind == "hook.dispatched"
    ]
    published = [item["event"] for item in events]
    assert published == ["record.start", "record.stop"]
    messages = [
        json.loads(line)
        for line in (tmp_path / "hooks/m.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {item["event"] for item in messages} == {"record.start", "record.stop"}
    assert {item["kind"] for item in messages} == {"record"}


def test_cli_runtime_role_publishes_lifecycle_events(tmp_path: Path, capsys: object) -> None:
    _project(
        tmp_path,
        hooks={"subscriptions": [{"event": "runtime.*", "action": "notify.message"}]},
    )

    assert main(["--project", str(tmp_path), "--format", "json", "runtime", "start"]) == 0
    run_id = json.loads(capsys.readouterr().out)["data"]["run_id"]  # type: ignore[attr-defined]
    events = [
        event.payload
        for event in TrainingStore(tmp_path / ".glr/runs.sqlite3").list_events(run_id)
        if event.kind == "hook.dispatched"
    ]
    assert [item["event"] for item in events] == ["runtime.start", "runtime.complete"]
    assert events[0]["results"][0]["action"] == "notify.message"


def test_cli_train_without_hooks_still_succeeds(tmp_path: Path, capsys: object) -> None:
    _project(tmp_path)
    assert main(["--project", str(tmp_path), "--format", "json", "train"]) == 0
    run_id = json.loads(capsys.readouterr().out)["data"]["run_id"]  # type: ignore[attr-defined]
    events = TrainingStore(tmp_path / ".glr/runs.sqlite3").list_events(run_id)
    assert [event.kind for event in events] == []


# --------------------------------------------------------------------------- #
# Review regressions: the failure path must still record its hook results
# --------------------------------------------------------------------------- #


def _exception_path_project(root: Path) -> None:
    """A project whose trainer cannot even be started.

    `_run_command` raises before any exit code exists, which is the path where
    `finish_run` used to run before the failure hook was published.
    """

    _project(
        root,
        trainer_argv=[str(root / "does-not-exist")],
        hooks={
            "subscriptions": [
                {
                    "event": "train.*",
                    "action": "notify.message",
                    "config": {"outbox": "hooks/messages.jsonl"},
                }
            ]
        },
    )


def test_exception_path_still_records_the_failure_hook_result(tmp_path: Path) -> None:
    _exception_path_project(tmp_path)

    with pytest.raises(FileNotFoundError):
        main(["--project", str(tmp_path), "--format", "json", "train"])

    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    runs = store.list_runs(environment_id="example.adventure-v1")
    assert len(runs) == 1
    assert runs[0].status is RunStatus.FAILED

    events = [
        event.payload
        for event in store.list_events(runs[0].run_id)
        if event.kind == "hook.dispatched"
    ]
    assert [item["event"] for item in events] == ["train.start", "train.failed"]
    failure_report = events[-1]
    assert failure_report["dispatched"] == 1
    assert failure_report["results"][0]["action"] == "notify.message"

    messages = [
        json.loads(line)
        for line in (tmp_path / "hooks/messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["event"] for item in messages] == ["train.start", "train.failed"]


def test_exception_path_records_the_record_stop_hook_result(tmp_path: Path) -> None:
    """`record.stop` is published from a `finally`, so it needs a live run too."""

    recorder = tmp_path / "recorder.py"
    recorder.write_text(
        """
import json
import os
import sys
from pathlib import Path

Path(os.environ["GLR_CAPTURE_VIDEO"]).write_bytes(b"synthetic-h264")
Path(os.environ["GLR_CAPTURE_INDEX"]).write_text(json.dumps({
    "schema_version": "glr.capture-frame.v1",
    "run_id": os.environ["GLR_RUN_ID"],
    "episode_id": "12345678-1234-5678-1234-567812345678",
    "step_id": 0,
    "frame_index": 0,
    "pts_ns": 0,
    "observation_timestamp_ns": 1
}) + "\\n", encoding="utf-8")
for line in sys.stdin:
    if line.strip() == "q":
        break
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _project(
        tmp_path,
        trainer_argv=[str(tmp_path / "does-not-exist")],
        capture_argv=[sys.executable, str(recorder)],
        hooks={"subscriptions": [{"event": "record.*", "action": "notify.message"}]},
    )

    with pytest.raises(FileNotFoundError):
        main(["--project", str(tmp_path), "--format", "json", "train"])

    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = store.list_runs(environment_id="example.adventure-v1")[0]
    events = [
        event.payload for event in store.list_events(run.run_id) if event.kind == "hook.dispatched"
    ]
    assert [item["event"] for item in events] == ["record.start", "record.stop"]
    assert run.status is RunStatus.FAILED


def test_runtime_failure_still_records_its_hook_result(tmp_path: Path) -> None:
    (tmp_path / "bridge").mkdir()
    (tmp_path / "glr-project.json").write_text(
        json.dumps(
            {
                "schema_version": "glr.project.v1",
                "environment_id": "example.adventure-v1",
                "environment_family": "action-rpg",
                "protocol_version": "1.0",
                "data_dir": ".glr",
                "bridge_path": "bridge",
                "runtime": {"argv": [str(tmp_path / "does-not-exist")]},
                "trainer": {"argv": [sys.executable, "-c", "print('train')"]},
                "player": {"argv": [sys.executable, "-c", "print('play')", "{bundle}"]},
                "hooks": {"subscriptions": [{"event": "runtime.*", "action": "notify.message"}]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        main(["--project", str(tmp_path), "--format", "json", "runtime", "start"])

    store = TrainingStore(tmp_path / ".glr/runs.sqlite3")
    run = store.list_runs(environment_id="example.adventure-v1")[0]
    assert run.status is RunStatus.FAILED
    events = [
        event.payload for event in store.list_events(run.run_id) if event.kind == "hook.dispatched"
    ]
    assert [item["event"] for item in events] == ["runtime.start", "runtime.failed"]


def test_dispatch_survives_an_unrecordable_action_detail() -> None:
    def handler(event: HookEvent, config: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"delivered": True, "unrecordable": object()}

    registry = HookRegistry()
    registry.register_action(CallableHookAction(name="custom.detail", handler=handler))
    registry.subscribe(HookSubscription(event="train.failed", action="custom.detail"))

    report = registry.dispatch(_event())

    assert report.dispatched == 1
    assert report.ok is True
    assert report.results[0].detail == {"detail_dropped": "detail was not JSON serializable"}
    assert report.to_mapping()["results"][0]["detail"] == {
        "detail_dropped": "detail was not JSON serializable"
    }


def test_dispatch_survives_a_nan_action_detail() -> None:
    registry = HookRegistry()
    registry.register_action(
        CallableHookAction(name="custom.nan", handler=lambda event, config: {"value": float("nan")})
    )
    registry.subscribe(HookSubscription(event="train.failed", action="custom.nan"))

    report = registry.dispatch(_event())

    assert report.dispatched == 1
    assert report.results[0].detail == {"detail_dropped": "detail was not JSON serializable"}


def test_dispatch_keeps_other_results_when_one_detail_is_unrecordable() -> None:
    registry = _registry()
    registry.register_action(
        CallableHookAction(name="custom.detail", handler=lambda event, config: {"bad": set()})
    )
    registry.subscribe(HookSubscription(event="train.failed", action="custom.detail"))
    registry.subscribe(HookSubscription(event="train.failed", action="notify.log"))

    report = registry.dispatch(_event())

    assert len(report.results) == 2
    assert report.failed == 0
    assert [item.action for item in report.results] == ["custom.detail", "notify.log"]


def test_cli_hooks_emit_rejects_an_illegal_event_name(tmp_path: Path) -> None:
    _project(tmp_path, hooks={"subscriptions": [{"event": "train.*", "action": "notify.log"}]})
    with pytest.raises(HookConfigurationError, match="lowercase dotted identifier"):
        main(
            [
                "--project",
                str(tmp_path),
                "--format",
                "json",
                "hooks",
                "emit",
                "--event",
                "Train Failed",
            ]
        )


def test_cli_reports_a_null_failure_key_on_success(tmp_path: Path, capsys: object) -> None:
    _project(tmp_path)
    assert main(["--project", str(tmp_path), "--format", "json", "train"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["data"]["failure"] is None
