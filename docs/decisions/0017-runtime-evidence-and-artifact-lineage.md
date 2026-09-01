# ADR-0017: Keep runtime evidence and artifact lineage learner-neutral

## Status

Accepted

## Context

Adapters that drive live-attached games need to explain why a route edge was
accepted, distinguish a stall from oscillation, pause navigation while modal UI
is open, and correlate recordings with the exact episode and build. GLR has
episode IDs, capture manifests, and run-store artifact records, but no
learner-neutral contract for these runtime semantics and their cross-artifact
identity.

## Decision

Add immutable contracts in `runtime_evidence.py`:

- `RouteTransitionEvidence` requires a settled position/orientation,
  monotonically supplied `producer_state_seq`, and route/build hashes before
  an edge can be considered succeeded;
- `RouteHealthTelemetry` records displacement, heading change, stall ticks, and
  oscillation count without granting a recovery action;
- `ModalNavigationBoundary` gates resume on an authoritative close and an
  increasing producer sequence;
- `ArtifactLineage` binds episode, trajectory, recording, route, map SHA,
  game-image SHA, and opaque encounter identity. Display names are excluded.

Adapters remain responsible for producing authoritative values and for all
game semantics. GLR does not implement route following, angle normalization,
UI interaction, recording, or terminal/reward decisions. Existing capture and
run-store modules remain responsible for writing files and persisted runs;
lineage can be stored alongside those artifacts.

## Consequences

These records make stale, unsettled, mis-bound, and unverified evidence
explicit and reproducible. They can be attached to transitions or recordings
by adapters without coupling the core to an engine or learner. Existing
transition schemas remain unchanged; projects may serialize the records as
sidecar evidence until a future versioned envelope is justified.

## Alternatives Considered

**Infer success from displacement or screenshots.** Rejected because zero
movement, wraparound angles, and stale frames are ambiguous.

**Put route recovery and modal automation in GLR.** Rejected because those are
adapter/game semantics and would violate the learner-neutral boundary.

**Use display names for encounter identity.** Rejected because names can be
localized, reused, or mislabelled; opaque adapter-owned IDs are stable evidence.
