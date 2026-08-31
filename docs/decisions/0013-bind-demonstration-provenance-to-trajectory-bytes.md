# ADR-0013: Bind demonstration provenance to trajectory bytes

- Status: Accepted
- Date: 2026-09-01

## Context

`glr.transition.v1` makes learner-neutral transitions portable, while
`DemonstrationGate` decides whether declared origin and outcome evidence may
enter behavior cloning. Those contracts previously met only in application
code. A copied or edited JSONL trajectory could retain an unrelated provenance
object, and every adapter had to invent its own sidecar format.

Behavior cloning needs a portable boundary that proves which exact trajectory
bytes, environment, and episode were reviewed. It must reject path traversal,
mixed episodes, missing terminal outcomes, modified bytes, and provenance that
the configured gate does not admit.

## Decision

Add `glr.demonstration-artifact.v1`, a strict adjacent manifest containing:

- the environment ID and single episode UUID;
- a portable relative trajectory path, byte size, and SHA-256;
- immutable `DemonstrationProvenance` origin, outcome, and optional policy ID.

`build_demonstration_artifact` validates a bounded `glr.transition.v1` file,
requires contiguous steps from zero, and requires success or failure to end at
a terminal boundary before writing the manifest atomically.

`verify_demonstration_artifact` rereads the exact bytes, verifies size and hash,
checks the expected environment and episode structure, and finally applies a
caller-supplied `DemonstrationGate`. It returns the already parsed transitions
and configured sample weight so a learner does not reopen a mutable path after
verification.

The manifest does not infer provenance or outcome. The authorized collector is
still responsible for recording them truthfully, and the gate remains the
policy authority.

## Consequences

- BC ingestion has one reusable fail-closed disk boundary instead of
  game-specific sidecars.
- Existing unbound datasets require explicit migration and review; filename or
  action plausibility is never enough.
- The verifier reads at most 256 MiB into memory to bind parsing to the hashed
  bytes and avoid a path-level time-of-check/time-of-use gap.
- Raw proprietary trajectories remain private; public evidence should contain
  only aggregate acceptance/rejection counts and checksums where appropriate.

## Rejected alternatives

- Store provenance only in each transition's `info`: repeated mutable metadata
  does not bind the whole episode or authoritative outcome.
- Accept an arbitrary manifest path: portable relative paths constrained below
  the manifest directory are easier to package and cannot escape the artifact.
- Stream, hash, then reopen for training: the file could change after
  verification, so verified transitions are returned directly.
