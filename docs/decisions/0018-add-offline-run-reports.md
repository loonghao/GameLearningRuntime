# ADR-0018: Add offline interactive run reports

## Status

Accepted

## Context

GLR already persists run metadata, metrics, namespaced events, spatial
knowledge, and checksummed capture artifacts. Reviewers still need to assemble
their own tooling to inspect training curves, routes, progression, match
outcomes, and post-run media. Adding these fields to learner transitions or
putting game-specific logic in the core would reduce reuse and increase the
size and sensitivity of replay records.

## Decision

Add a standalone `glr.run-report.v1` consumer in the Rust CLI:

```text
glr --project . --json report build <run-id>
```

The report builder reads the existing run-store projection, verifies registered
artifact bytes, emits an offline `index.html`, and registers it as a
`run-report` artifact. The bundled UI is data-only and uses bounded, escaped
runtime values. Namespaced adapter events provide optional projections for
routes (`navigation.route_*`), progression (`progression.*`), and matches
(`match.result`); unsupported projections remain visibly empty.

The report layer never starts a runtime, sends an action, infers semantic state
from pixels, or changes tensor/learner contracts. Adapter semantics and
authority remain the adapter's responsibility. Media remains a separate,
checksummed artifact referenced by a portable relative path.

## Consequences

- A reviewer can open one local HTML file to inspect a run and its evidence.
- Existing `glr.transition.v1`, capture, and run-store schemas remain stable.
- Reports are deterministic for a fixed run projection and renderer version,
  but a report build is not live-game acceptance.
- Large media is linked rather than embedded; reports fail closed if a
  registered artifact is missing or has changed.
- Projects must sanitize public exports and avoid account, host, process/window,
  absolute-path, credential, or proprietary runtime data in events.

## Alternatives considered

**Put HTML in each game adapter.** Rejected because every adapter would repeat
storage, provenance, escaping, and artifact verification.

**Embed screenshots and tensors in transition records.** Rejected because it
breaks bounded transport and learner-neutral replay portability.

**Infer routes, unlocks, or wins from rendered frames.** Rejected because
pixels cannot provide the authoritative post-state required for those claims.
