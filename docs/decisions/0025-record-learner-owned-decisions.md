# ADR-0025: Record learner-owned dynamic decisions

## Status

Accepted

## Context

Dynamic environments expose action candidates whose parameters depend on the
current observation. A controller that silently substitutes a scripted choice
can collect transitions that do not represent the learner's policy. A successful
trainer process also does not establish policy updates or improved performance.

## Decision

Add an optional Python API in `game_learning_runtime.decisions`. Immutable
`Candidate` values identify commands and canonical JSON parameters. A `Decision`
selects exactly one unique candidate and identifies its state, policy digest,
and training or evaluation mode. Parameter reads return detached objects.

`execute_decision` calls the supplied executor once with the selected command
and parameters. It retains the executor receipt without choosing a fallback.
The executor remains responsible for runtime authorization, action masks,
freshness, binding and outcome validation; this API is not a new host protocol
and does not authenticate an executor or enforce how a trainer updates weights.

`learning_status` classifies caller-supplied transition/update counts and policy
digests without certifying improvement. Basic CLI training persists metadata
that identifies its status as process execution and leaves learning and
improvement unverified. Existing status and exit-code semantics remain compatible.

## Consequences

- Dynamic candidate selection can be audited without coupling adapters to an optimizer.
- Tensor policies and `SyncCollector` remain unchanged.
- Rejections and exceptions do not trigger a hidden fallback or retry.
- Frozen evaluation and comparable performance measurements remain trainer/evaluator duties.
- These are in-process Python values, not a replacement for versioned trajectory formats.

## Rejected alternatives

- Put route or combat rules in the runtime: this obscures policy ownership.
- Treat a successful process or changed digest as improved performance: neither proves it.
- Introduce a new engine wire envelope: the existing provider vocabulary is sufficient.
