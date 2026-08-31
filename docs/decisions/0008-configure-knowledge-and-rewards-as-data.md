# ADR-0008: Configure knowledge and rewards as strict data

## Status

Accepted

## Context

Game adapters benefit from external rules, patch notes, maintained wikis, and
strategy guides when defining observations, actions, masks, and reward
hypotheses. Those sources can be stale, contradictory, copyrighted, or
untrusted. Runtime telemetry has a different role: when target-bound and
verified after an action, it may be authoritative for reward and terminal
state.

Each adapter also needs tunable reward weights and clips. A generic expression
language, import path, or callback loaded from configuration would turn a data
file into a code-execution surface, make reward behavior hard to audit, and
couple the core to game-specific semantics.

## Decision

Add a strict JSON schema family identified by `glr.training.v1` and parsed with
the Python standard library. Unknown fields, invalid identifiers, duplicate
sources or terms, non-finite values, and invalid bounds fail closed.

The configuration contains:

- collection lifecycle (`reset` or `attach`) and terminal behavior;
- bridge capabilities that a deployment must require;
- bounded knowledge-source declarations with `advisory` or `authoritative`
  authority, required status, maximum age, and maximum payload size;
- named reward terms with a fixed source, weight, raw clip, required status,
  minimum authority, and optional total clip.

Configuration contains no URLs, endpoints, credentials, local paths,
expressions, imports, or executable callbacks. Adapters own acquisition and
emit reviewed, named scalar `RewardSignal` values. `RewardComposer` validates
the signal set and source, clips and weights it, and returns an immutable total
and contribution breakdown. Reward terms require authoritative sources by
default. An adapter may deliberately allow an advisory shaping term, but must
declare that weaker authority on the individual term and validate it through
ablation.

Gameplay web research is a separate design-time
`glr.knowledge-research.v1` manifest owned by the project Agent Skill. It keeps
public URLs, short paraphrased claims, timestamps, confidence, volatility, and
verification status. It is not imported by the runtime. Research starts as
advisory; a claim becomes runtime-verified only through bounded authorized
evidence.

## Consequences

### Positive

- Reward behavior is deterministic, reviewable, and safe to diff.
- The same config shape works for turn-based, real-time, BC, PPO, IMPALA,
  offline collection, and evaluation lanes.
- Strategy research accelerates adapter design without receiving action or
  reward authority by accident.
- Strict identifiers and a separate research manifest reduce the chance of
  publishing local endpoints, credentials, or raw copyrighted content.

### Negative

- Adapters must implement game-specific signal extraction in code.
- Freshness and payload limits are policy declarations; each acquisition
  adapter must enforce them when it reads a knowledge snapshot.
- Changing reward logic beyond weights and clipping requires reviewed code and
  tests rather than a quick expression edit.

### Neutral

- Learners may still normalize or transform returns on the learner side, but
  that does not change the environment's declared reward provenance.
- A future signed or remotely distributed config format can wrap the same
  semantic contract without adding executable fields.

## Alternatives Considered

**Embed Python or a reward expression DSL.** Rejected because it adds code
execution, ambiguous numeric semantics, and a second game-logic layer.

**Put full strategy documents in runtime configuration.** Rejected because it
mixes design evidence with runtime state, increases copyright/privacy risk,
and makes freshness impossible to audit.

**Treat all knowledge as equally trusted.** Rejected because a guide, cache,
and authoritative runtime receipt have fundamentally different failure modes.

**Let each learner define environment reward.** Rejected because datasets,
evaluators, PPO actors, and BC pipelines would no longer share one truthful
environment contract.

## References

- ADR-0001, ADR-0002, ADR-0006, and ADR-0007
- `src/game_learning_runtime/training.py`
- `.agents/skills/glr-adapter-builder`
