# ADR-0029: Bound the runtime start readiness window in the project contract

## Status

Accepted

## Context

GLR defines readiness in three places. `ReadinessResult` names `ready`,
`not_ready`, and `unavailable`, and the module documents that callers decide
how to park and retry a not-ready host. The bridge turn path and the configured
game launcher already consume that model: the bridge polls a probe before
attaching and during an episode, and the launcher waits for a process or file
signal before it calls an instance started.

`glr runtime start` did not. It invoked the runtime role once and compressed
the framework's three-state model back into two: a non-zero exit code became
`failed`, whether the role crashed or reported that its host was still starting.
A role that brings up a cold GUI host legitimately refuses to drive that host
while it boots and refuses with a named non-zero code and a `not_ready` receipt.
Because the start verb owned no waiting window, an environment that recovered on
its own was recorded as a failure as soon as that refusal arrived, while another
command in the same project judged the same environment correctly inside its own
settle window. The project therefore had to re-implement the missing window as a
private settle constant that every other entrypoint could not reach.

## Decision

A project may declare a bounded startup readiness window for its runtime role:

```toml
[runtime]
argv = ["uv", "run", "--frozen", "python", "-m", "example_runtime"]

[runtime.readiness]
timeout_seconds = 300
poll_interval_seconds = 5
file = "runtime-readiness.json"
```

`timeout_seconds` is required; the poll interval and receipt name have defaults.
The receipt name is a portable project-relative path inside the run directory.
A window is never unbounded and never longer than one hour. An undeclared
window keeps the historical single-invocation behavior exactly.

Within a declared window, `runtime start` re-invokes the role and reads the
receipt the role publishes through `GLR_READINESS_PATH`, which is the
`ReadinessResult.to_mapping()` mapping already published as
`glr.environment-readiness.v1`. No new wire schema is introduced. The role also
receives `GLR_READINESS_ATTEMPT` so a re-probe can tell itself apart from a cold
launch.

Only an explicit `not_ready` receipt is retryable. A receipt that is missing,
unreadable, oversized, off-schema, `unavailable`, or that claims `ready` while
the role exited non-zero is terminal on first observation. A crash is therefore
never retried and never mistaken for a host that is still starting, and the
retry decision stays owned by the adapter that knows what its host can serve.

An exhausted window is a distinct verdict, not a generic failure. The run
stores one `readiness.attempt` event per invocation and one `readiness.outcome`
event before the run is finished, both carrying the receipt verbatim, and the
`runtime.start` envelope adds the same summary. `RunStatus` keeps its existing
vocabulary: a start that never reached its goal is still a failed run, and the
new information is explicitly about why. An exhausted window returns exit code
`78` instead of the role's refusal code, so a caller that only reads exit codes
can tell "retry when the host has settled" from "this is broken". Every other
verdict keeps the role's exit code. Both the Rust CLI and the Python CLI
implement the same contract.

## Consequences

### Positive

- The project contract owns the startup window once, so an adapter no longer
  needs a private settle constant and every entrypoint observes one verdict.
- `not_ready` and `unavailable` are named separately in the run's durable
  evidence, so a host that was still booting is auditable after the fact.
- Retrying is opt-in twice: the project declares a window, and the role declares
  a retryable receipt. Neither can happen implicitly.

### Negative

- A declared window can re-invoke a role, so a role with start-time side effects
  must be idempotent across attempts; `GLR_READINESS_ATTEMPT` exists so it can
  be. The framework cannot check this property for a project.
- Exit code `78` is a new documented code for one framework verdict.

### Neutral

- Run events remain the durable record channel, consistent with
  `context.selected`; no run-store schema version changes.
- `bridge.py` and `game_launcher.py` keep their existing single-probe
  semantics. An adapter that only needs one probe inside the role is unaffected.
- `doctor` still reports role executable availability; the declared window is
  configuration, and its effect appears in run evidence.

## Alternatives Considered

**Add a `not_ready` run status.** Rejected because `RunStatus` is a persisted
compatibility contract surfaced by `runs list --status`, and the start verb's
verdict is still "the command failed". The missing information was diagnosis,
not lifecycle, so it belongs to run evidence instead of a new state that every
existing reader would have to learn.

**Retry any non-zero start exit code inside the window.** Rejected because it
would retry genuine crashes and hide adapter defects, and it would break the
existing meaning of a named refusal.

**Give the runtime role a "park and re-probe" protocol instead of re-invoking
it.** Rejected because the role owns host semantics and already knows how to
judge them; asking it to stay resident would move process supervision back into
every adapter, which is the duplication this decision removes.

**Let each adapter keep publishing its own settle constant.** Rejected because
the two verbs would keep contradicting each other for the same environment, and
no entrypoint could observe or bound the window.

## References

- `src/game_learning_runtime/readiness.py`
- `src/game_learning_runtime/project.py`
- `crates/glr-cli/src/readiness.rs`
- `crates/glr-cli/src/project.rs`
- ADR-0015: Add an agent-first local control plane
- ADR-0016: Make the Rust CLI the distribution entrypoint
- ADR-0021: Resolve portable projects from one manifest
