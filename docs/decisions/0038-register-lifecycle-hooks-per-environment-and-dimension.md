# ADR-0038: Register lifecycle hooks per environment and dimension

## Status

Accepted

## Context

Training runs end silently. A run that finishes at 03:00, or dies on step 40 000,
leaves its verdict in a log file nobody is watching, and an agent that wants to
react to a failure has to poll `glr runs show` and re-derive what happened. Users
asked for a hook mechanism that fires on training completion so a message can be
sent, on training failure so an agent can react, and — explicitly — for more
environments and dimensions than success and failure: start of training, start
and stop of video recording, and anything else a project needs later.

The control plane already knew every one of those moments. What was missing was a
place to declare "when this happens, do that" without putting a callback, an
import path, or a credential into a configuration file.

Three constraints shaped the design:

- The event space cannot be a closed enum. A hook model limited to
  success/failure would need a code change for every new lifecycle moment, and
  adapters must be able to publish their own.
- A notification must never be able to break a run. A hook that raises, hangs,
  or is misconfigured cannot change the exit code of the training it observes;
  the run's verdict belongs to the run.
- Configuration stays strict data. GLR validates and composes; it does not import
  entry points, run build hooks, or hold secrets.

## Decision

Add `game_learning_runtime.hooks` (contract, registry, dispatcher) and
`game_learning_runtime.hook_actions` (built-in notifications), exposed through
`glr hooks list` and `glr hooks emit`.

- **Events are open identifiers.** `HookEvent` carries `name`, `status`,
  `environment_id`, `environment_family`, `kind`, `stage`, `run_id`,
  `exit_code`, `reason`, `occurred_at_ns`, and a JSON-safe `payload`.
  `PREDEFINED_HOOK_EVENTS` documents what the control plane publishes
  (`train.start/complete/failed`, `record.start/stop`, `goal.*`, `runtime.*`,
  `play.*`) and is discovery metadata only — any lowercase dotted identifier is
  a legal event name.
- **Subscriptions select "environment x dimension".** A `HookSubscription` pairs
  an event selector with an action name and an optional `HookEventFilter`
  (`environment_id`, `environment_family`, `kind`, `stage`, `status`,
  `exit_code_min`/`exit_code_max`). A selector ending in `.*` matches its
  namespace; an unset filter matches everything. This is what makes "notify on
  every failed training run in the racing family, exit code non-zero"
  expressible without new code.
- **Actions are pluggable and self-validating.** `HookAction` is a `Protocol`
  with `name`, `validate_config`, and `__call__`; each action validates its own
  configuration block, so the registry carries no per-action business logic.
  `CallableHookAction` adapts a plain function.
- **Three actions ship so one lane works out of the box:** `notify.log`,
  `notify.message` (a project-relative JSON Lines outbox), and `notify.webhook`
  (one `http`/`https` POST with no embedded credentials). Anything else is
  registered by the caller.
- **Dispatch never propagates.** `HookRegistry.dispatch` evaluates matching
  inside a guard, runs each action on a daemon thread bounded by its
  `timeout_seconds`, and converts a raise into a `failed` result and an overrun
  into a `timeout` result. It returns a `HookDispatchReport`; it raises nothing.
- **Configuration lives in the project manifest** under `[hooks]`, parsed by
  `project.load_project` into `HookConfig`. It is bounded: at most 64
  subscriptions, budgets between 0.1 s and 300 s, and no arbitrary keys.
- **Run verbs degrade, inspection verbs fail.** `glr train`, `glr goal run`,
  `glr runtime start`, and `glr play` build the registry non-strictly: an
  unusable configuration is logged and the run proceeds with no subscriptions.
  `glr hooks list` and `glr hooks emit` are strict, so a mistake surfaces when
  the manifest is edited rather than during a run.
- **Results are recorded, not just logged.** Every dispatch that produced a
  result is appended to the run as a `hook.dispatched` event carrying status,
  duration, error, and detail per action.
- **Failures are machine-readable.** A failed `train`, `goal run`, `runtime
  start`, or `play` adds a `failure` object — `stage`, `reason`, `exit_code` —
  to its `--format json` envelope, so an agent can decide what to do next
  without parsing a log.

## Consequences

- "Notify when training finishes" is configuration, not code, and works on a
  fresh project.
- A notification backend that is down, slow, or misconfigured costs one visible
  `hook.dispatched` event and a log line. It cannot turn a successful run into a
  failure or hide a real failure.
- The event space can grow per project without a runtime change, at the cost of
  typos being silently unmatched — `glr hooks list` and `--dry-run` are the
  intended checks.
- Outbox paths are resolved inside the project root with a symlink check per
  component, so a hook cannot be used to write outside the project.
- Webhook URLs may not embed credentials. An authenticated transport is a
  registered custom action, keeping secrets out of the manifest.
- A hook action runs on a daemon thread, so it must be safe to abandon after its
  budget expires; a hung action's thread is not killed, only stopped being
  waited on.

## Rejected alternatives

- A fixed enum of lifecycle events: rejected because every new moment would
  require a runtime change, and adapters could not publish their own.
- Import paths or executable callbacks in configuration: rejected because it
  reintroduces arbitrary code execution through a data file, which the
  knowledge/reward configuration (ADR-0008) deliberately refuses.
- Propagating hook exceptions to the run: rejected because a notification must
  never be able to fail a training run or mask the run's own exit code.
- Running actions without a budget: rejected because a hung webhook would hang
  the training run that published the event.
- Failing a run on a broken hook configuration: rejected for the same reason —
  the run's verdict belongs to the run; `glr hooks list` reports the error
  instead.
- Generic templating or expression support in messages: rejected in favour of a
  fixed placeholder set rendered by literal replacement, so a value cannot
  introduce a new field or a format expression.

## Related

- [Lifecycle hooks guide](../guides/lifecycle-hooks.md).
- ADR-0008 configures knowledge and rewards as strict data; hooks follow the
  same "validated data, no callbacks" rule.
- ADR-0015 defines the agent-first local control plane that publishes these
  events.
- ADR-0029 bounds the runtime start readiness window, one of the moments hooks
  observe.
