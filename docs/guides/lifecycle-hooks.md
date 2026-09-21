# Lifecycle hooks

Hooks attach a named action to a named lifecycle event so a training run can
notify, record, or hand off without the runtime knowing anything about the
delivery mechanism.

Everything here is declarative: the event space and the action set are open, the
configuration is strict data, and a broken hook can never change the exit code
of the run that triggered it.

## The three moving parts

| Part | Where it lives | What it decides |
| --- | --- | --- |
| Event | Published by the control plane | "What just happened, in which environment, dimension, and stage" |
| Subscription | `hooks.subscriptions` in `glr-project.toml` / `.json` | "Which events this action cares about, and under which filters" |
| Action | `game_learning_runtime.hook_actions`, or your own registration | "What to do with a matched event" |

An event is selected by name or namespace wildcard, then narrowed by optional
filters. An action is looked up by name in the registry and receives the event
plus its own validated configuration block.

## Events

The control plane publishes these events today:

| Event | Published by | Status |
| --- | --- | --- |
| `train.start` | `glr train` | `started` |
| `train.complete` | `glr train` | `succeeded` |
| `train.failed` | `glr train` | `failed` or `interrupted` |
| `record.start` | `glr train` when a recorder is configured | `started` |
| `record.stop` | `glr train` when the recorder session ends | `succeeded` or `failed` |
| `goal.start` | `glr goal run` | `started` |
| `goal.complete` | `glr goal run` | `succeeded` |
| `goal.failed` | `glr goal run` | `failed` or `interrupted` |
| `runtime.start` / `runtime.complete` / `runtime.failed` | `glr runtime start` | `started` / `succeeded` / `failed` |
| `play.start` / `play.complete` / `play.failed` | `glr play` | `started` / `succeeded` / `failed` |

The list is documentation, **not** an enum. `PREDEFINED_HOOK_EVENTS` is exported
for discovery, but a subscription may name any lowercase dotted identifier, so
an adapter or extension can publish `adapter.ready` or `episode.terminated`
without a change to the runtime.

Every event carries the dimensions a subscription can filter on:

```json
{
  "schema_version": "glr.hooks.v1",
  "event": "train.failed",
  "status": "failed",
  "environment_id": "example.adventure-v1",
  "environment_family": "action-rpg",
  "kind": "training",
  "stage": "trainer",
  "run_id": "run-75a523aa996f48deacc27a612ea9af09",
  "exit_code": 7,
  "reason": "trainer command exited with code 7",
  "occurred_at_ns": 1790011519592187500,
  "payload": {}
}
```

`kind` is the run dimension (`training`, `record`, `goal`, `runtime`,
`playback`, ...), `stage` is the phase inside it (`train`, `game-launch`,
`capture`, `trainer`, ...), and `reason` is the human-readable failure cause.

## Configuration

Hooks are declared in the project manifest, next to `trainer` and `capture`.

```toml
[hooks]
enabled = true
default_timeout_seconds = 5.0

[[hooks.subscriptions]]
event = "train.*"
action = "notify.message"
enabled = true

[hooks.subscriptions.when]
environment_family = "action-rpg"
kind = "training"
status = "failed"
exit_code_min = 1

[hooks.subscriptions.config]
outbox = "hooks/messages.jsonl"
message = "{event} {status} at {stage}: {reason}"
```

| Field | Meaning |
| --- | --- |
| `enabled` | Master switch; when false nothing is dispatched. |
| `default_timeout_seconds` | Budget applied to subscriptions that do not set one. |
| `subscriptions[].event` | Exact event name, or a namespace wildcard such as `train.*`. |
| `subscriptions[].action` | Registered action name. |
| `subscriptions[].when` | Optional filters: `environment_id`, `environment_family`, `kind`, `stage`, `status`, `exit_code_min`, `exit_code_max`. |
| `subscriptions[].config` | Action-owned configuration, validated by the action itself. |
| `subscriptions[].timeout_seconds` | Per-substitution budget. |
| `subscriptions[].enabled` | Disable one subscription without deleting it. |

A subscription list is capped at `MAX_HOOK_SUBSCRIPTIONS` (64), and every budget
must be finite and between 0.1 s and 300 s. An unset filter matches everything,
so a subscription with no `when` block fires for every matching event.

Message templates support `{event}`, `{status}`, `{kind}`, `{stage}`,
`{run_id}`, `{environment_id}`, `{environment_family}`, `{exit_code}`, and
`{reason}`. They are rendered by literal replacement, never by `str.format`, and
an unknown placeholder is a configuration error.

## Built-in actions

| Action | Effect | Configuration |
| --- | --- | --- |
| `notify.log` | Emit one structured log record | `message`, `level` (`debug`, `info`, `warning`, `error`) |
| `notify.message` | Append one JSON Lines message to a project-relative outbox | `outbox`, `message` |
| `notify.webhook` | POST the structured event as JSON to one URL | `url`, `timeout_seconds`, `message` |

The outbox is the offline lane: an agent, scheduler, or mailer reads
`hooks/messages.jsonl` without GLR knowing the transport. The webhook lane
accepts only `http`/`https` URLs without embedded credentials, because secrets
do not belong in a project manifest.

## Custom actions

Anything else is registered in code; nothing about the built-ins is special.

```python
from game_learning_runtime.hooks import CallableHookAction, HookEvent, HookRegistry


def publish(event: HookEvent, config):
    ticket_system.post(config["queue"], event.to_mapping())
    return {"queued": event.name}


registry = HookRegistry()
registry.register_action(
    CallableHookAction(
        name="ticket.create",
        handler=publish,
        config_validator=lambda config: {"queue": str(config["queue"])},
    )
)
```

A class works just as well: implement `name`, `validate_config(config)`, and
`__call__(event, config)`. The returned mapping, if any, is recorded as the
result detail.

## Inspecting and exercising hooks

```bash
glr hooks list --format json     # actions, subscriptions, predefined events
glr hooks emit --event train.failed --status failed --exit-code 7 --reason "oom" --dry-run
glr hooks emit --event train.failed --status failed --exit-code 7 --reason "oom"
```

`hooks list` is strict: an unknown action or a malformed configuration is an
error, so a mistake surfaces when you edit the manifest instead of during a
training run. `hooks emit` publishes a synthetic event through the same
dispatcher and exits `1` when any action failed or timed out, which makes it
usable as a configuration smoke test.

## Failure isolation

A hook is observability, never a dependency:

- An action that raises is reported as a `failed` result with
  `TypeName: message` in `error`; the exception never reaches the run.
- An action that exceeds its budget is reported as `timeout`. The dispatcher
  stops waiting and the thread is left to finish as a daemon, so one slow
  notification cannot stall or crash a training run.
- An unknown action or an unusable configuration makes the run fall back to a
  registry with no subscriptions. The reason is logged, and `glr hooks list`
  still reports the error.
- A failed, timed-out, or misconfigured hook never changes the exit code of
  `glr train`, `glr goal run`, `glr runtime start`, or `glr play`.

## Observability

Every dispatch that produced at least one result is appended to the run as a
`hook.dispatched` event:

```json
{
  "schema_version": "glr.hooks.v1",
  "event": "train.failed",
  "enabled": true,
  "dry_run": false,
  "dispatched": 1,
  "failed": 0,
  "timed_out": 0,
  "results": [
    {
      "event": "train.failed",
      "action": "notify.message",
      "status": "succeeded",
      "duration_ms": 0.4,
      "error": null,
      "detail": { "outbox": "hooks/messages.jsonl", "bytes": 417 }
    }
  ]
}
```

Read it with `glr runs show <run-id> --format json`. Each result carries its
status, wall-clock duration, and error, so a delivered notification and a broken
one are distinguishable from the run record alone.

## Machine-readable failures for agents

A failed run reports the same facts in the command envelope, following the
`--format json` convention:

```bash
glr train --format json
```

```json
{
  "schema_version": "glr.cli-output.v1",
  "command": "train",
  "data": {
    "run_id": "run-75a523aa996f48deacc27a612ea9af09",
    "status": "failed",
    "exit_code": 7,
    "failure": {
      "stage": "trainer",
      "reason": "trainer command exited with code 7",
      "exit_code": 7
    }
  }
}
```

`failure.stage` names where the run broke (`train`, `game-launch`, `capture`,
`trainer`, `goal`, ...), `failure.reason` is the cause, and `failure.exit_code`
matches the process exit code. `glr goal run` reports `failure` the same way when
the success criteria are not satisfied, and `glr runtime start` / `glr play`
report it when the role command fails.

## Related

- ADR-0038 records the decision behind the registry and the isolation contract.
- [Agent-first CLI](agent-first-cli.md) documents the command surface.
- [Project output layout](project-output-layout.md) explains where runs and
  artifacts land.
