# Record why every episode ended

A `done` tensor says an episode stopped. It does not say why, so schedulers each
re-implemented their own "did we reach the goal?" predicate and unexplained
episodes quietly reached the training dataset. GLR makes the reason part of the
episode: a closed `TerminationReason` plus a free-text `termination_detail`,
kept in `game_learning_runtime.termination`.

## The closed enum

| Reason | Meaning | Usually attributed by |
| --- | --- | --- |
| `goal_reached` | the episode's goal was met | adapter |
| `failed` | the episode failed, or a step raised | runtime / caller |
| `step_budget` | the declared `max_steps` was reached | runtime |
| `time_budget` | the declared `max_time_ns` elapsed | runtime |
| `death_cap` | the declared `death_cap` was reached | runtime |
| `stalled` | the declared `stall_steps` passed without progress | runtime |
| `env_frozen` | the environment stopped producing new observations | runtime / caller |
| `host_unavailable` | the game host or transport went away | runtime / caller |
| `env_indeterminate` | an action outcome was reported indeterminate | runtime |
| `caller_aborted` | the caller stopped the episode | caller |

The enum is closed: an unknown value is a contract violation, not a fallback.
A reason nobody can gate on is worse than no reason at all.

## Where the reason comes from

`EpisodeTerminationGuard.close()` resolves the reason in a fixed order so
attribution is reproducible:

1. a latched indeterminate outcome;
2. an explicit `reason=` from the caller (`attributed_by="caller"`);
3. `termination_reason` in `TimeStep.info` (`attributed_by="adapter"`);
4. the caps the adapter already declared in `EnvironmentSpec.episode_caps`
   (`attributed_by="runtime"`), checked as death cap, then step budget, time
   budget, and stall.

Nothing left means a contract violation: `MissingTerminationReason` is raised
with `field == "termination_reason"`. It subclasses `TerminationError` and
`GLRError`, so a caller can catch the contract failure without catching
environment faults.

An adapter therefore never reports a reason it already declared:

```python
from game_learning_runtime.specs import EnvironmentSpec
from game_learning_runtime.termination import EpisodeCaps

spec = EnvironmentSpec(
    ...,
    episode_caps=EpisodeCaps(max_steps=256, death_cap=3),
)
```

An adapter with no declared caps sets the reason itself, in `TimeStep.info` at
the episode boundary:

```python
info = {"termination_reason": "goal_reached", "termination_detail": "reached the shrine"}
```

## Read the reason without parsing logs

Three surfaces carry it:

- `SyncCollector.terminations` and `last_termination()` during a run;
- the run store, as `episode.termination` events
  (`TrainingStore.record_episode_termination` / `list_episode_terminations`),
  which stay readable after a run is finished;
- `glr.cli-output.v1` from `glr --project . --json runs show <run-id>`, which
  adds `terminations` and a `termination_summary` next to the run record.

`EpisodeTermination.to_mapping()` is the `glr.episode-termination.v1` payload
all three share, so an agent can read one shape everywhere.

`reached_goal()` is the predicate a scheduler asks for; it is `False` for an
open episode and for every non-goal reason.

## No step after the episode ended

A step offered after a close is refused with `EpisodeClosedError` instead of
being admitted to the dataset, continuing the post-boundary rules from episode
identity. Closing twice returns the one terminal state.

## An indeterminate outcome is absorbing

`ActionOutcome.INDETERMINATE` is distinct from `REJECTED`. A rejected action is
known *not* to have been applied; an indeterminate one *may* have been applied
and the environment consequence is unknown. The correct response is to stop and
restart under supervision, never to try another action, so an indeterminate
receipt cannot be built with `retryable=True`.

When a collector sees one, the episode ends immediately with
`env_indeterminate`:

- the latched step is dropped, and so is the transition whose successor
  observation is the untrustworthy one;
- the last surviving transition is marked truncated, so the learner does not
  bootstrap past an unknown boundary;
- the manifest records `latched_at_ns` and `last_known_sequence`, the evidence
  a supervised restart needs;
- collecting again is refused until the caller calls `SyncCollector.reattach()`.

`validate_timestep()` and `assert_transition_provenance()` reject an
indeterminate receipt outright, and provenance now carries
`termination_reason` / `termination_detail` so a replay row explains its own
boundary.

An adapter that never reports `indeterminate` is unaffected by all of this. The
only migration an existing adapter owes is a termination reason for its
episodes, either by declaring caps or by declaring the reason in `info`.
