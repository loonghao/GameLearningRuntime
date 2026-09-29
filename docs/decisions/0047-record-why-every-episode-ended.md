# ADR-0047: Record why every episode ended, and make an indeterminate outcome absorbing

## Status

Accepted (retrospective).

This record describes architecture that is **already implemented and shipped on
`main`**. It was written after the fact, because the change it describes landed as
a guide and a code change with no decision record, while two issues closed in the
same batch (#157 → ADR-0040, #159 → ADR-0042) each got one. Nothing here proposes
new behavior. Where this record and the code disagree, the code is right and this
record should be corrected.

Landed in [PR #174](https://github.com/loonghao/GameLearningRuntime/pull/174)
(2026-09-21), which closed
[#155](https://github.com/loonghao/GameLearningRuntime/issues/155) and
[#156](https://github.com/loonghao/GameLearningRuntime/issues/156). The
normative usage guide is
[Record why every episode ended](../guides/episode-termination.md).

## Context

A `done` tensor says an episode stopped. It does not say why, and the difference
between "reached the objective" and "ran out of budget" is the difference between
progress and noise.

Both problems below were observed in the same external-attach Windows
integration, and both share one failure mode: **a gate reading the run store sees
green while nothing is measured.**

### #156 — ordinary endings were never recorded

Nothing required an episode to say how it ended in the ordinary case. #57, #82
and #83 had each added a *failure* terminal state; the ordinary ones (step cap,
wall-clock cap, death cap) had no home. From that integration: 15 consecutive
rounds ended on a death cap or a wall-clock cap, and **not one** summary carried a
termination field. A round that spent its whole budget walking and a round that
advanced the objective were both just "a round that finished".

The consequence compounds, because a caller cannot answer "did this episode reach
its goal?" without parsing logs:

- a promotion decision cannot condition on it;
- a cross-round circuit breaker cannot distinguish "tried and failed" from "never
  tried";
- an unattended agent comparing two configurations compares two numbers that mean
  different things.

### #155 — the unknowable outcome had no home

In an exactly-once transport, an action can be dispatched and then become
unknowable: the acknowledgement never arrives, and the target's state after the
action cannot be established. #42 had given actions a typed outcome, but the
taxonomy could not express "the outcome is not knowable", so adapters mapped it
onto `rejected` and every downstream safeguard read that as a routine refusal.

The two are not the same, and the difference is expensive:

| Outcome | Meaning | Correct response |
| --- | --- | --- |
| `rejected` | this action did not happen | try another action |
| `indeterminate` | this action **may** have happened, and the environment can no longer be reasoned about | stop; supervised restart |

From the same integration: after a single transport timeout the bridge latched a
terminal fence, and from that moment on **every** subsequent action returned the
same refusal for the rest of the run. Because the outcome looked like an ordinary
rejection, nothing above it changed course. The collector kept recording ordinary
transitions from a fenced environment, the round ran to its full budget — 660
steps — wrote a complete summary, and its checkpoint was promoted and inherited by
the next round. The environment was not frozen (#83) and the host was not
unavailable (#82): it answered every call. It simply could not say what any action
had done.

## Decision

### D1. A closed `termination_reason` enum on every episode, and a missing value is a contract violation

Every episode carries a `termination_reason` from a closed enum plus a free-text
`termination_detail`, in `game_learning_runtime.termination`:

`goal_reached`, `failed`, `step_budget`, `time_budget`, `death_cap`, `stalled`,
`env_frozen`, `host_unavailable`, `env_indeterminate`, `caller_aborted`.

**A missing reason is a contract violation, not a warning.** The reasoning is the
same one that drives ADR-0040: a diagnostic nobody is required to act on is not a
gate, it is a comment. Warning about an unexplained episode in a log is
indistinguishable from not saying anything, because every downstream consumer —
promotion, circuit breakers, comparison — reads the absence of a failure state as
success. So closing without a reason raises `MissingTerminationReason`, which
subclasses `TerminationError` and `GLRError` and carries a `field` naming
`termination_reason`, so a caller can catch the contract failure without catching
environment faults.

The enum is closed for the same reason: **a reason nobody can gate on is worse
than no reason at all.** An unknown string is a contract violation rather than a
silent fallback to some default, because a fallback would put a value in the field
that no consumer can branch on — which is the original defect wearing a different
hat.

### D2. The runtime owns attribution; the adapter declares facts and caps

Attribution is a fixed, reproducible order, so the same episode always yields the
same reason:

1. a latched indeterminate outcome (D3);
2. an explicit `reason=` from the caller (`attributed_by="caller"`);
3. `termination_reason` in `TimeStep.info` (`attributed_by="adapter"`);
4. the caps the adapter already declared in `EnvironmentSpec.episode_caps`
   (`attributed_by="runtime"`), checked as death cap, then step budget, then time
   budget, then stall.

Which source supplied the reason is recorded as `attributed_by`, so a reader can
tell an adapter declaration from a runtime attribution.

The caps are the load-bearing part. A field that is present only when someone
remembers to set it will not be present; so an adapter declares its budgets once
(`EpisodeCaps(max_steps=…, death_cap=…, max_time_ns=…, stall_steps=…)`) and
reports only the underlying *facts* (`episode_deaths`, `episode_stall_steps`),
letting the runtime decide the reason. An adapter with no caps may still declare
the reason itself in `info`. **The requirement is that something sets it.**

### D3. `indeterminate` is distinct from `rejected`, and absorbing

`ActionOutcome.INDETERMINATE` is a third answer alongside `accepted` and
`rejected`: the action may have been applied and the consequence is unknown. It is
deliberately *not* a refusal — `refusals.py` does not route it — because routing it
as one would reproduce the original bug.

Because the environment can no longer be reasoned about, the episode ends
immediately with `env_indeterminate`, and the latch is **absorbing**:

- the latched step is dropped, and so is the transition whose successor
  observation is the untrustworthy one the latched step produced;
- the last surviving transition is marked truncated, so the learner does not
  bootstrap past an unknown boundary;
- the manifest records `latched_at_ns` and `last_known_sequence`, the evidence a
  supervised restart needs;
- collecting again raises `IndeterminateOutcomeError` until the caller calls
  `SyncCollector.reattach()`.

An indeterminate receipt cannot be built with `retryable=True`
(`ActionReceipt.__post_init__`): an implicit retry of a mutating action whose
first effect is unknown is unsafe, and #42 and #83 already forbid implicit
retries. The remedy is a supervised restart, never a retry.

**Steps taken after any termination are not training data.** `records_step()`
returns `False` once an episode is closed, and a step after a close raises
`EpisodeClosedError` instead of being admitted. This is the same rule #82 and #83
already apply to their own terminal states, now applied uniformly rather than per
terminal state.

### D4. The reason is readable without parsing logs

Three surfaces carry the same `glr.episode-termination.v1` payload, so an agent
reads one shape everywhere:

- `SyncCollector.terminations` and `last_termination()` during a run;
- the run store, as `episode.termination` events
  (`TrainingStore.record_episode_termination` / `list_episode_terminations`),
  which is the only surface that outlives the process;
- `glr.cli-output.v1` from `glr --project . --json runs show <run-id>`, which adds
  `terminations` and a `termination_summary` (counts per reason, and
  `goal_reached` / `indeterminate` totals) next to the run record.

The store surface took two passes, and the failure mode is worth recording because
it is the same one this ADR exists to prevent. The first pass added an optional
`on_termination` callback and `TrainingStore.termination_sink(run_id)`, but nothing
in the codebase reached it: every documented path passes `store` and `run_id`
instead, so those runs still reported `terminations: []` with `episode_count: 0` —
indistinguishable from a run that never collected an episode. The second pass made
**`store` and `run_id` together bind the sink by default**, with an explicit
callback still winning. Coverage that constructs store state by hand proves the API
and not the wiring, which is why the gap survived a first review.

`reached_goal()` is the predicate a scheduler actually asks for, defined once.

## Relationship to #82, #83 and #85

Three different failures, three different terminal states, three different
remedies. Keeping them separate is the point: collapsing them is what let a fenced
environment run to its full budget.

| Issue | Symptom | Terminal state | Remedy |
| --- | --- | --- | --- |
| #82 | the host cannot run this | `host_unavailable` | park, do not burn budget |
| #83 | the environment produces no new state | `env_frozen` | terminate the episode |
| #155 | the environment still answers, but no action's consequence is knowable | `env_indeterminate` | stop, then supervised restart |
| #85 | the target process needs lifecycle supervision | — | exclusive lease, ordered stop, artifact ownership |

#85 owns the restart. This ADR only decides that a restart is *required*; it does
not implement supervision. #42 defined the outcome taxonomy that #155 extends.

## Rejected alternatives

- **Warn instead of failing when the reason is missing.** Rejected: this is the
  defect, not a mitigation. A run that cannot say why it stopped cannot be
  compared to anything, and a warning reaches a log rather than a gate. The same
  reasoning is why ADR-0040 fails closed on a declared metric that is never
  emitted.
- **Let each adapter declare its own reason vocabulary.** Rejected: a
  free-form string cannot be gated on, so consumers would each re-implement a
  predicate over adapter-specific wording — the exact situation every scheduler
  was already in. A closed enum costs adapters one mapping and buys one shared
  predicate.
- **Let the adapter report the reason, with no runtime attribution.** Rejected
  because it makes the field present only when someone remembers. Declared caps
  invert that: the runtime fills the reason in from a declaration the adapter
  already had to make. Adapter-declared reasons remain supported, as the second
  source in the resolution order, not the only one.
- **Map `indeterminate` onto `rejected`.** Rejected: this is #155's bug. It makes
  "known not applied" and "may have been applied" indistinguishable, and the
  correct responses are opposites — one permits another action, the other forbids
  it. Mapping it onto `unknown` was equally rejected, because `unknown` carries no
  obligation to stop.
- **Make `indeterminate` advisory rather than absorbing.** Rejected: a run that
  continues to step after it has lost the ability to attribute outcomes is
  producing noise, and — as observed — will spend its whole budget, report
  normally, and promote a checkpoint trained on it.
- **Retry the action instead of ending the episode.** Rejected: the first
  attempt's effect is unknown, so a retry may double-apply a mutating action.
  Retrying is also already forbidden by #42 and #83. The remedy is a supervised
  restart (#85), which is why `retryable` cannot be `True` on an indeterminate
  receipt.
- **Keep the steps collected after the latch.** Rejected: they are transitions
  from an environment whose state after the last action is unknown, so their
  successor observations are untrustworthy. Dropping them is the same rule #82 and
  #83 already apply.

## Consequences

### Positive

- "Did this episode reach its goal?" is answerable from one JSON field, so
  promotion, circuit breakers and configuration comparison stop re-implementing
  their own predicate.
- An unexplained episode is now impossible rather than merely discouraged: the
  close fails, and it fails with a field name a caller can branch on.
- A fenced-but-responsive environment ends its episode instead of running to
  budget and promoting a checkpoint.
- The latch carries `latched_at_ns` and `last_known_sequence`, so a supervised
  restart has the evidence it needs rather than a guess.

### Negative

- Every adapter owes a migration: either declare caps or declare the reason.
  There is no third option, and an adapter that does neither now raises at close
  instead of silently producing unexplained episodes.
- The closed enum is a compatibility surface: a new reason is a schema change,
  not a string a project can invent locally.
- Attributing from caps means the runtime's ordering decides which reason wins
  when several caps are breached at once. The order is fixed and documented, so
  it is reproducible, but it is a policy an adapter cannot override except by
  declaring the reason itself.

### Neutral

- An adapter that never reports `indeterminate` is unaffected: the latch is new
  behavior on a path no existing adapter can reach.
- The runtime stays learner-neutral. It records why an episode ended; it does not
  decide what to do about it.

## Verification on `main`

Behavior this record claims, and where it is pinned:

| Claim | Evidence |
| --- | --- |
| An indeterminate outcome ends the episode immediately, drops the latched step and the transition whose successor observation is untrustworthy, and marks the last survivor truncated | `test_reporting_indeterminate_ends_the_episode_immediately` (`tests/test_termination.py`) |
| The manifest records the latch time and the last known sequence | `test_the_manifest_records_the_latch_time_and_last_known_sequence` |
| Latching on the first step records nothing and says why | `test_latching_on_the_first_step_records_nothing_and_says_why` |
| No step is admitted after the latch | `test_no_step_is_admitted_after_the_latch` |
| An ordinary `rejected` lets the same episode run to its own boundary, so the two outcomes are not confusable | `test_a_plain_rejection_lets_the_episode_continue` |
| An indeterminate receipt is not routed through the refusal retry/backoff funnel | `test_an_indeterminate_receipt_is_not_routed_through_the_refusal_funnel` |
| `glr runs show` carries every episode's reason plus a summary, with no log parsing | `test_cli_reports_why_each_episode_ended_without_parsing_logs` (`tests/test_cli.py`) |
| An episode closed without a reason raises a typed, fielded error | `MissingTerminationReason`, `field == "termination_reason"` |
| An indeterminate receipt cannot be built as retryable | `ActionReceipt.__post_init__` (`src/game_learning_runtime/contracts.py`) |
| Collecting again after the latch is refused until an explicit reattach | `test_no_step_is_admitted_after_the_latch`, `SyncCollector.collect` → `IndeterminateOutcomeError` |
| A step after a close is refused, not recorded | `EpisodeClosedError`, `EpisodeTerminationGuard.records_step` |
| A run bound with `store` and `run_id` writes its terminations to the store by default | `tests/test_termination.py` (bound-run end-to-end) |

## Related

- [Record why every episode ended](../guides/episode-termination.md) — the
  normative usage guide
- [ADR-0040: Fail closed on a declared metric that is never emitted](0040-fail-closed-on-a-declared-metric-that-is-never-emitted.md) —
  the same fail-closed argument, applied to a declared metric
- [ADR-0036: Add a policy-only supervision watchdog with a scheduler exit-code contract](0036-supervision-watchdog-and-scheduler-contract.md)
- [Issue #155](https://github.com/loonghao/GameLearningRuntime/issues/155),
  [Issue #156](https://github.com/loonghao/GameLearningRuntime/issues/156),
  [PR #174](https://github.com/loonghao/GameLearningRuntime/pull/174)
