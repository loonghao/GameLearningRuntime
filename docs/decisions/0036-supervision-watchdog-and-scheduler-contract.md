# ADR-0036: Add a policy-only supervision watchdog with a scheduler exit-code contract

## Status

Accepted

## Context

Unattended and scheduled training needs a way to prove a trainer is still alive
and to recover it when it is not. `ProcessSupervisor` (ADR-0032) already owns
process identity, exclusive leases, stop, and restart for one process, but
nothing evaluates *whether* a restart is warranted. Without that, an unattended
run either dies silently or is restarted forever by an external loop that cannot
tell recovery from failure.

The requirement is that a cron job, a systemd timer, or a Windows Scheduled Task
must be able to drive supervision with one command and one exit code. That rules
out a design that requires a daemon, shared state between invocations, or
output parsing.

## Decision

Add `game_learning_runtime.watchdog` with a `SupervisionWatchdog` that is
**policy-only**: it reads heartbeats, decides, and delegates the restart. It
never spawns a process and never probes OS liveness.

- A `Heartbeat` is one bounded liveness proof: `source`, `sequence`,
  `observed_at_ns`, `state`, `detail`, persisted as JSON Lines by
  `HeartbeatLog`. Any producer may write it; the watchdog only reads.
- `WatchdogPolicy` carries a finite budget: `heartbeat_timeout_seconds` (30),
  `max_missed_heartbeats` (3), `restart_attempt_limit` (3), `restart_backoff_seconds`
  (5), `restart_cooldown_seconds` (30), `recovery_timeout_seconds` (60).
  `starvation_seconds` is derived as timeout × missed.
- `restart_attempt_limit` counts **attempts**, not successes: every intervention
  increments it whether or not it worked. The name, the code, and this document
  use the same word so the exit semantics (an intervention budget) and the field
  cannot drift apart.
- Restart authority is delegated to an injected `ProcessSupervisor` or to a
  declared recovery command executed with `shell=False`, captured, and bounded by
  `recovery_timeout_seconds`.
- The attempt budget is finite and per source. Once exhausted the watchdog
  escalates and stops touching that source.
- **A failed intervention never reports exit code 3.** The attempt is counted,
  the decision is `FAILED` / `ESCALATE` / `restart-failed`, and the report exit
  code is `4`. A recovery that timed out is a failed attempt:
  `subprocess.TimeoutExpired` is caught, not raised.
- A starved source with **no** recovery wiring escalates rather than reporting
  healthy. A watchdog that cannot fix anything must not tell its scheduler that
  all is well.

The scheduler contract is three exit codes, exposed by
`WatchdogReport.exit_code` and `SupervisionWatchdog.run_once()`:

| Code | Meaning |
| --- | --- |
| `0` | Healthy or degraded. |
| `3` | Recovered — a recovery attempt was issued and succeeded. |
| `4` | Escalated — needs a human. |

Expose it as `glr watchdog tick`, with `--source` (repeatable), `--heartbeats`,
`--timeout`, `--max-missed`, `--restart-attempt-limit`, `--recovery-command` (argv
tokens, repeated), and optional `--interval` / `--max-ticks` for a long-lived
supervisor. `run()` returns early on escalation.

Every decision carries a reason string — `heartbeat-current`, `heartbeat-late`,
`no-heartbeat-yet`, `heartbeat-starved`, `detect-only-no-recovery-wiring`,
`restart-issued`, `restart-failed`, `restart-attempts-exhausted`,
`restart-backoff-active`, `awaiting-heartbeat-after-restart` — so an alert is
actionable without reading code. Clock, sleep, and recovery runner are injected,
so the whole policy is testable with a synthetic clock and no subprocess.

## Consequences

- One command and one exit code is enough for any scheduler; no daemon, no
  shared state, no output parsing.
- `SuccessExitStatus=0 3` in a systemd unit expresses the intended semantics
  directly: a recovery is successful supervision, and only `4` pages someone.
  This is only honest because `3` is unreachable when an intervention failed —
  both the code and the docs must keep that invariant.
- Preferring one pass per invocation over `--interval` means a scheduler-driven
  run cannot leak restart state and a watchdog crash cannot silence the
  schedule.
- Bounded restarts surface broken trainers instead of hiding them behind an
  infinite restart loop.
- The watchdog does not detect a hung-but-heartbeating trainer. Liveness is
  whatever the producer chooses to report; richer health belongs to the
  project's own metrics.

## Rejected alternatives

- Let the watchdog own process spawning and OS liveness probes: rejected because
  it would duplicate `ProcessSupervisor`, add a platform-specific rabbit hole,
  and make the decision untestable without a real process.
- Restart forever with unbounded attempts: rejected because it converts a real
  failure into permanent noise.
- Report starvation with no recovery wired as healthy (exit 0): rejected because
  the scheduler watching the run could never learn the trainer went silent.
- A daemon with a shared state file: rejected because it cannot be driven by a
  one-shot cron entry and it introduces state that survives a crash.
- Accept a single shell string for recovery: rejected because `shell=False` with
  an argv list removes the injection surface and the quoting problem.

## Related

- ADR-0032 hosts the externally driven loop that `ProcessSupervisor` governs.
- [Supervision and watchdog guide](../guides/supervision-watchdog.md).
