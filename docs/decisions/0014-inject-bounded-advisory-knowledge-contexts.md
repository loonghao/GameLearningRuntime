# ADR-0014: Inject bounded advisory knowledge contexts

## Status

Accepted

## Context

Learners need more than raw observations for games with item acquisition,
target selection, upgrades, and hazards. Projects already declare bounded
knowledge sources in `glr.training.v1`, but the core previously had no runtime
contract for validating a point-in-time knowledge snapshot or selecting the
small subset relevant to one decision.

Passing whole guides to a model is not acceptable: content can be stale,
oversized, contradictory, private, or prompt-like untrusted data. Embedding
game-specific rules in collectors would couple adapters to one learner and
would blur the authority boundary between advisory strategy and runtime truth.

## Decision

Add the strict `glr.knowledge-snapshot.v1` data contract and a learner-side
`KnowledgeInjector`.

Every context source must be declared with `provides_context: true`. The
injector validates its payload byte limit, timestamp freshness, source ID,
schema, item count, text lengths, identifiers, confidence, stage range, and
content hash. A required context source fails closed when absent.

Knowledge items use four game-neutral intents:

- `acquire`: resources, equipment, or objectives worth taking;
- `engage`: targets or encounters worth confronting;
- `upgrade`: capabilities worth improving;
- `avoid`: hazards or unfavorable interactions.

The learner supplies a `KnowledgeQuery` with its current stage, desired
intents, and optional tags. Selection is deterministic and bounded by the
training policy. The result is an immutable `KnowledgeContext` with exact
per-item source/snapshot provenance and a SHA-256 digest. Encoding text or
fields into model inputs remains learner-owned.

All knowledge context remains advisory. It cannot expand an environment action
mask, acknowledge an action, provide terminal truth, or satisfy a reward term
that requires authoritative runtime evidence.

## Consequences

### Positive

- PPO, IMPALA, BC, offline, and evaluation learners can consume the same
  game-neutral knowledge contract without entering adapter code.
- Training bundles can preserve the exact snapshots and digest used for a run.
- Freshness, size, confidence, stage, source, and item-count failures are
  explicit and testable.
- “What to acquire, engage, upgrade, or avoid” is data, not executable policy.

### Negative

- Projects must produce compact snapshots and map runtime observations to
  query stage/tags.
- Learners still need their own deterministic encoder or tokenizer and must
  record that implementation in reproduction evidence.
- The initial selector uses explicit tags and ranges, not semantic/vector
  retrieval.

### Neutral

- Source acquisition and storage remain deployment concerns; the core accepts
  already acquired bytes and contains no URL, credential, or database client.
- A future signed bundle can wrap the same snapshot without changing selection
  semantics.

## Alternatives Considered

**Append full documents to every observation.** Rejected because it is
unbounded, stale, hard to reproduce, and exposes learners to irrelevant text.

**Let each adapter inject its own model prompt.** Rejected because it couples
runtime integration to model architecture and makes cross-game behavior
unreviewable.

**Use vector search in the NumPy-only core.** Rejected for the first version
because it adds models, indexes, optional dependencies, and nondeterministic
ranking before real adapter requirements justify them.

## References

- ADR-0001: Keep the runtime contract learner-neutral
- ADR-0008: Configure knowledge and rewards as strict data
- `src/game_learning_runtime/knowledge.py`
- `docs/guides/knowledge-and-rewards.md`
