# ADR-0037: Add a read-only anti-fork drift gate

## Status

Accepted

## Context

A fork or a copy of this repository does not break loudly. It drifts. The
`origin` remote is repointed, the branch quietly falls hundreds of commits
behind, or version and schema constants are edited locally until the copy can no
longer merge back. Nothing in the tree reports the divergence, so by the time a
human notices, the rebase is unrecoverable.

The drift is measurable from the checkout alone — remote URL, divergence against
the default branch, package version against the release manifest, and wire
schema constants — but nothing measured it, and nothing had authority to fail.

## Decision

Add `game_learning_runtime.fork_gate` and expose it as `glr fork-gate`. The gate
is read-only, bounded, and scheduler-friendly.

- `GitRepositoryProbe` reads `origin`, the current branch, and
  `--left-right` divergence against `origin/<default-branch>` through bounded
  (30 s) read-only git calls, plus package and release-manifest versions from
  files. `RepositoryProbe` is a `Protocol`, so the policy is testable without a
  repository; `StaticRepositoryProbe` supplies fixed values.
- Remote comparison is normalized: scheme, user, trailing slash, and `.git`
  suffix are stripped and the result is lowercased, so
  `git@github.com:org/repo.git` and `https://github.com/org/repo/` are the same
  upstream.
- `ForkGatePolicy` holds the expectation and the drift budget:
  `expected_origin_url`, `default_branch` (`main`), `max_commits_behind` (50),
  `max_commits_ahead` (200), and three switches that can downgrade a finding to
  an advisory (`require_origin_match`, `require_version_alignment`,
  `require_upstream_ref`).
- Required wire schema versions are held as **pinned literals** in
  `FORK_GATE_SCHEMA_VERSIONS`, while the probe reports what this checkout's own
  modules say. Comparing a fixed expectation against a locally observed value is
  what makes the check meaningful; comparing an imported constant against itself
  would always pass.
- Commits *ahead* is an advisory, not a blocker: unreviewed work is not drift.
  An unfetched upstream ref blocks by default, because divergence cannot be
  measured without it and "unknown" must not read as "aligned".
- Every check is a `ForkGateFinding` carrying `observed` and `expected`. The
  report is `glr.fork-gate-report.v1`. The gate returns `0` when it passes and
  `5` when it is blocked, and nothing else.

The gate is deliberately **not** wired into `just check`. Local quality gates
must stay independent of upstream reachability, and a legitimate fork may fail
this gate on purpose. Its interception points are scheduled jobs, pre-training
checks on long-lived checkouts, and a fork's own CI with
`--allow-foreign-origin`.

## Consequences

- Drift is caught while the rebase is still small, and the report names the
  remediation (`observed` versus `expected`) without opening the code.
- A scheduler needs no output parsing: `0` or `5`.
- A fork that intentionally tracks a different upstream can still use the gate
  by passing its own `--origin` and `--allow-foreign-origin`, so real drift —
  behind, version, schema — keeps failing.
- The gate detects declared drift, not semantic drift. A fork that keeps
  `origin` pointed upstream and rebases regularly can still diverge in intent;
  no mechanical check can catch that.
- Schema literals must be updated deliberately when a wire schema is bumped.
  That is the point: a schema bump is a reviewed change, not a side effect.

## Rejected alternatives

- Fetch, rebase, or push from the gate: rejected because a gate that mutates
  state cannot run unattended, and its failure modes become repository damage.
- Fail on commits ahead as well as behind: rejected because being ahead is
  normal, unreviewed work, not divergence.
- Treat an unfetched upstream as passing: rejected because "unknown" reads as
  "aligned" and hides the drift the gate exists to find.
- Compare imported schema constants against themselves: rejected because the
  check would be vacuous and could never detect a local edit.
- Hard-fail on a non-canonical origin with no override: rejected because a
  downstream fork would then have no way to use the gate at all.

## Related

- ADR-0036 defines the supervision contract that runs alongside this gate in
  scheduled jobs.
- [Anti-fork gate guide](../guides/fork-gate.md).
