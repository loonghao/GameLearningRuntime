# Supervision and watchdog

Unattended training needs two things a normal test run does not: a way to prove
a trainer is still alive, and a bounded, auditable way to recover when it is
not. GLR splits these between `ProcessSupervisor` (identity, leases, stop and
restart of one process) and `SupervisionWatchdog` (heartbeat evaluation and the
recovery decision).

The watchdog is deliberately **policy-only**. It never spawns a process and
never probes OS liveness. It reads heartbeats, decides, and delegates the
restart either to an injected `ProcessSupervisor` or to a declared recovery
command. That keeps the decision testable without a real trainer and keeps
restart authority in one place.

## Heartbeats

A heartbeat is one bounded liveness proof from one named source:

```json
{"schema_version": "glr.heartbeat.v1", "source": "trainer", "sequence": 41, "observed_at_ns": 1770000000000, "state": "running", "detail": ""}
```

- `source` — a stable identifier (letters, digits, `.`, `_`, `-`).
- `sequence` — monotonically non-decreasing per source.
- `observed_at_ns` — `time.monotonic_ns()` at emission, or any monotonic clock
  in nanoseconds.
- `state` — free-form but an identifier; `running` is the default.
- `detail` — optional human note.

Heartbeats are appended to a JSON Lines log. `HeartbeatLog.read()` skips
unparsable lines rather than failing the whole pass, and
`latest_by_source()` returns the newest beat per source.

```python
import time
from pathlib import Path

from game_learning_runtime import Heartbeat, HeartbeatLog

log = HeartbeatLog(Path(".glr/heartbeats.jsonl"))
log.append(Heartbeat(source="trainer", sequence=41, observed_at_ns=time.monotonic_ns()))
```

Any producer can write the log: a training loop, a wrapper script, or an
external scheduler. The watchdog only reads it.

## Policy

| Field | Default | Meaning |
| --- | --- | --- |
| `heartbeat_timeout_seconds` | `30.0` | Gap tolerated before a heartbeat is *late*. |
| `max_missed_heartbeats` | `3` | Late intervals tolerated before the source is *starved*. |
| `restart_limit` | `3` | Total automatic restarts allowed per source, ever. |
| `restart_backoff_seconds` | `5.0` | Wait before another restart is attempted. |
| `restart_cooldown_seconds` | `30.0` | Grace period after a restart, before lateness counts again. |
| `recovery_timeout_seconds` | `60.0` | Bound on one recovery command. |

`starvation_seconds = heartbeat_timeout_seconds * max_missed_heartbeats` — with
defaults, a source silent for 90 seconds is starved.

The restart budget is the important number. It is finite and per source: once
exhausted the watchdog **escalates and stops touching that source**. A
supervisor that silently restarts forever hides a bug; this one reports it.

## Status, action, exit code

| Status | Meaning | Action | Report exit code |
| --- | --- | --- | --- |
| `HEALTHY` | Heartbeat inside the timeout. | `NONE` | `0` |
| `DEGRADED` | Heartbeat late, or none yet but inside the grace window. | `NOTIFY` | `0` |
| `STARVED` | Silent past starvation. | `RESTART` when recovery is wired, `ESCALATE` when it is not. | `3` / `4` |
| `RECOVERING` | Restart issued, or inside backoff/cooldown. | `RESTART` or `NONE` | `3` / `0` |
| `FAILED` | Restart failed or budget exhausted. | `ESCALATE` | `4` |

A starved source with **no** recovery wiring escalates rather than reporting
healthy: a watchdog that cannot fix anything must not tell its scheduler that
all is well.

The scheduler contract is three exit codes:

| Code | Meaning | What a scheduler should do |
| --- | --- | --- |
| `0` | Healthy or degraded. | Nothing. |
| `3` | Recovered — a restart was issued. | Log it; optionally alert. |
| `4` | Escalated — needs a human. | Alert and stop retrying. |

`WatchdogReport.exit_code` is derived, never guessed: `4` if any decision
escalated, `3` if any recovered, otherwise `0`.

## Running it

One pass, from the command line:

```bash
# Named sources only — with no heartbeat log every source is immediately starved.
vx just glr-watchdog --source trainer

# Read the newest beat per source from a log.
vx just glr-watchdog --source trainer --heartbeats .glr/heartbeats.jsonl

# Tune the budget and wire a recovery command.
vx just glr-watchdog --source trainer \
  --heartbeats .glr/heartbeats.jsonl \
  --timeout 30 --max-missed 3 --restart-limit 3 \
  --recovery-command glr --recovery-command train
```

`--recovery-command` takes argv **tokens**, repeated once per token. It is
executed with `shell=False`, captured, and bounded by
`recovery_timeout_seconds`. There is no shell, so there is no injection surface
and no quoting problem.

For a long-lived supervisor, pass `--interval` (and optionally `--max-ticks`)
to loop; the loop returns immediately on escalation.

Machine-readable output always uses the `glr.watchdog-report.v1` envelope:

```json
{"schema_version": "glr.watchdog-report.v1", "decisions": [...], "exit_code": 4}
```

## Recovery reasons

Every decision carries a reason string, so an alert can be actionable without
reading code:

| Reason | Meaning |
| --- | --- |
| `heartbeat-current` | Beat is inside the timeout. |
| `heartbeat-late` | Beat is older than the timeout but not starved. |
| `no-heartbeat-yet` | Source registered, no beat observed. |
| `heartbeat-starved` | Silence exceeded `starvation_seconds`. |
| `detect-only-no-recovery-wiring` | Starved, but neither a supervisor nor a recovery command is wired, so escalation is the only honest answer. |
| `restart-issued` | A restart was issued this pass. |
| `restart-failed` | The recovery command or supervisor restart failed. |
| `restart-budget-exhausted` | `restart_limit` reached; escalating from here on. |
| `restart-backoff-active` | Inside `restart_backoff_seconds`; wait. |
| `awaiting-heartbeat-after-restart` | Inside `restart_cooldown_seconds`; give the source a chance. |

## Scheduling it

The design constraint is that a cron job, a systemd timer, or a Windows
Scheduled Task must be able to drive this with **one command and one exit
code**. Nothing needs a daemon.

### cron

```cron
# Every 5 minutes: one pass. Exit code is the outcome.
*/5 * * * * cd /srv/GameLearningRuntime && vx just glr-watchdog --source trainer --heartbeats /srv/glr/heartbeats.jsonl --restart-limit 3 >> /var/log/glr-watchdog.log 2>&1

# Hourly: fail loudly if the checkout has drifted from canonical upstream.
7 * * * * cd /srv/GameLearningRuntime && git fetch --quiet origin && vx just glr-fork-gate >> /var/log/glr-fork-gate.log 2>&1
```

Prefer one pass per invocation over `--interval`. A scheduler that re-invokes a
fresh process cannot leak restart state, and a crash in the watchdog cannot
silence the schedule.

### systemd timer

```ini
[Unit]
Description=GLR supervision pass

[Timer]
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
[Unit]
Description=GLR supervision pass
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/srv/GameLearningRuntime
ExecStart=/usr/bin/env vx just glr-watchdog --source trainer --heartbeats /srv/glr/heartbeats.jsonl
SuccessExitStatus=0 3
```

`SuccessExitStatus=0 3` is the point: a recovery is *successful supervision*,
not a unit failure. Only `4` should page someone.

### Windows Scheduled Task

```powershell
$action  = New-ScheduledTaskAction -Execute 'vx' `
  -Argument 'just glr-watchdog --source trainer --heartbeats C:\glr\heartbeats.jsonl' `
  -WorkingDirectory 'C:\srv\GameLearningRuntime'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName 'GLR Supervision' -Action $action -Trigger $trigger
```

## Wiring a Python supervisor

When the supervised process is owned in-process, inject a `ProcessSupervisor`
instead of a command. The watchdog calls it and counts the result against the
same budget.

```python
from game_learning_runtime import SupervisionWatchdog, WatchdogPolicy, WatchdogTarget

watchdog = SupervisionWatchdog(policy=WatchdogPolicy())
watchdog.register(WatchdogTarget(name="trainer", policy=WatchdogPolicy(), supervisor=supervisor))
exit_code = watchdog.run_once()
```

## Related

- [Agent onboarding](agent-onboarding.md) — bootstrap and verify chain.
- [Anti-fork gate](fork-gate.md) — drift detection for derived checkouts.
- [Repository layout](repository-layout.md) — where automation lives.
- ADR-0036 in [the decision index](../decisions/README.md).
