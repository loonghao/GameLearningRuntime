# ADR-0015: Add an agent-first local control plane

## Status

Accepted

## Context

GLR already defines a learner-neutral game/runtime boundary, training policy, knowledge snapshots,
demonstration gates, and reproducible model bundles. Operating those pieces still required
project-specific human commands. An agent could not discover one project contract, start its
reviewed bridge, launch training with review capture, query prior experience, or pursue a bounded
objective through research, planning, training, and authoritative evaluation.

A generic autonomous runner would create the wrong abstraction. GLR cannot choose one learner,
scraper, game launcher, recorder, or success evaluator for every genre. It also must not turn web
guides into action authority, transfer game coordinates between unrelated worlds, execute arbitrary
remote scripts, or treat a model hash as evidence that a gameplay objective succeeded.

## Decision

Add an agent-first, local-only `glr` CLI and strict `glr.project.v1` project contract.

The project declares exact environment/family/protocol identity, a project-relative bridge path,
and fixed-argv runtime, trainer, player, researcher, planner, evaluator, and optional recorder
roles. GLR executes each role without a shell and substitutes only fixed whole-argument
placeholders. Acquisition, game semantics, learning algorithms, and target binding remain owned by
reviewed project code.

The CLI persists a query projection in SQLite:

- run lifecycle, events, scalar metrics, logs, and checksummed artifacts;
- exact-environment observed entities and advisory waypoint routes;
- provenance-bound research sources and compact findings scoped to an environment, game family,
  or generic mechanics.

Large observations, transitions, video, and model bytes remain external artifacts. SQLite is not a
tensor store.

Add a bounded goal loop using three data-only contracts:

- `glr.agent-goal.v1` declares machine-readable success criteria and trial, step, time, and source
  budgets;
- `glr.research-bundle.v1` stores public source provenance and non-executable findings;
- `glr.trial-plan.v1` carries learner-neutral reward hypotheses as bounded numeric terms;
- `glr.goal-evidence.v1` carries evaluator claims that must match metrics newly persisted for the
  active trial.

Only authoritative runtime evidence can satisfy a goal. When a trial fails, later researcher and
planner roles receive previous research/evaluation paths and may adjust sources, curriculum, and
reward terms within the original global budgets.

Run an optional project-owned recorder concurrently with training. The recorder emits a compact
H.264 MP4 plus `glr.capture-frame.v1` records mapping environment steps to video frames. A
checksummed `glr.capture.v1` manifest binds both artifacts. This makes the same capture useful for
human review and later supervised-data selection without putting OS capture code in GLR core.

Spatial snapshots use `glr.spatial-knowledge.v1`. Export/import requires the exact environment and
protocol. Imported coordinates and routes are advisory until observed again. Research findings may
transfer at `family` scope, but models still require exact environment/protocol compatibility.

## Consequences

### Positive

- An agent has one stable JSON CLI for runtime startup, training, goal pursuit, querying, knowledge
  transfer, and playback.
- Goal success is machine-readable and bound to persisted runtime evidence instead of logs, video,
  survival, or process exit.
- Small-window capture is decoupled from learner throughput and remains useful for both review and
  supervised-learning curation.
- Similar genres can reuse cited strategies and reward hypotheses without contaminating coordinate,
  action, or model identity.
- Project roles remain replaceable; GLR does not acquire a learner, browser, video stack, or game
  launcher dependency.

### Negative

- Projects must implement and review several explicit roles and produce strict JSON files.
- SQLite is a single-machine control projection; distributed coordination and tensor indexing still
  require a separate data plane.
- A configured evaluator is trusted project code. GLR can bind its evidence to persisted metrics,
  but live adapter acceptance still proves whether those metrics reflect the real game.
- Capture synchronization quality depends on the project recorder and its step timestamps.

### Neutral

- `runtime start` proves only that the fixed command ran. A bridge handshake and live gameplay
  acceptance remain separate gates.
- Model bundle verification proves byte/config integrity, not policy quality or hardware-level
  determinism.
- Source discovery remains a deployment/agent responsibility. GLR validates acquired research and
  enforces budgets; it does not crawl the web or store credentials.

## Alternatives Considered

**Put orchestration in every game adapter.** Rejected because it couples runtime semantics to one
training and research workflow.

**Add an unrestricted command or script endpoint.** Rejected because it breaks the bounded local
trust model and makes agent configuration executable remote authority.

**Store frames and tensors directly in SQLite.** Rejected because it harms training throughput and
duplicates artifact/dataset formats. The database keeps queryable metadata and exact hashes.

**Transfer all learned state across games in one family.** Rejected because coordinates, action
semantics, and model inputs are not family-stable. Only provenance-bound advisory findings have
family scope.

## References

- ADR-0001: Keep the runtime contract learner-neutral
- ADR-0008: Configure knowledge and rewards as strict data
- ADR-0010: Add authorized loader plugins and reproducible model bundles
- ADR-0014: Inject bounded advisory knowledge contexts
- `docs/guides/agent-first-cli.md`
- `.agents/skills/glr-cli/SKILL.md`
