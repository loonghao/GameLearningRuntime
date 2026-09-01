# ADR-0016: Make the Rust CLI the distribution entrypoint

## Status

Accepted

## Context

ADR-0015 introduced an agent-first local control plane as an unpublished Python
console script. That prototype proved the command and data contracts, but it
made every Agent install Python, the GLR package, NumPy, and an environment
manager before it could start a bridge, inspect prior runs, or diagnose a
project. It also separated the CLI from the native Runtime Host and repository
Skills that an Agent needs to deploy and operate GLR.

The learning ecosystem remains Python-heavy. TorchRL, Gymnasium, learner
objectives, tensor contracts, and many project trainers should therefore remain
available as a Python SDK. The distribution entrypoint has different
non-functional requirements: one executable, stable machine output and exit
codes, cross-platform process supervision, predictable SQLite behavior, and a
bounded update path.

Self-update adds a supply-chain boundary. A downloaded binary cannot be trusted
only because it came from an asset URL, and arbitrary project dependencies must
not be changed implicitly.

## Decision

Replace the unpublished Python `glr` console script with a Rust binary in
`crates/glr-cli`. The Rust CLI is the canonical deployment and Agent control
entrypoint. The Python distribution remains the learner-facing SDK and no
longer publishes a `glr` console script.

The Rust CLI owns:

- strict `glr.project.v1` loading, fixed whole-argument placeholders, and
  shell-free project role execution;
- stable `glr.cli-output.v1` JSON envelopes, documented exit codes, diagnostics,
  run queries, spatial knowledge transfer, and verified playback;
- concurrent project-owned capture orchestration and the bounded
  research/plan/train/evaluate loop;
- the SQLite metadata projection and compatibility with project roles that use
  the Python SDK;
- `glr update --check` and confirmed `glr update --yes` operations.

Each supported GitHub Release target publishes one
`glr-{version}-{rust-target}.zip`. The archive contains the `glr` CLI,
`glr-hostd`, the `glr-cli` and `glr-adapter-builder` Skills, a strict
`glr.release-bundle.v1` manifest, license, and concise install guide. The
updater selects the exact compiled Rust target, downloads the exact
`SHA256SUMS`, verifies the archive before extraction, rejects unsafe archive
paths and symlinks, and then replaces only those managed components. Applying
an update requires `--yes`. Skill synchronization targets the current project's
`.agents/skills` directory unless the caller supplies `--skills-dir` or
`--no-skills`.

`glr update` does not alter a game's trainer dependencies, virtual environment,
adapter code, credentials, or reward configuration. Projects continue to own
those dependencies and roles. Same-release SHA-256 proves integrity relative to
the Release metadata; it is not an independent publisher signature.

The Python implementation and Rust implementation do not coexist as two public
CLIs. Existing Python contract types remain available where they are useful to
trainers and adapters, while Rust black-box tests lock the CLI behavior and
SQLite compatibility.

## Consequences

### Positive

- An Agent can deploy and inspect GLR from one versioned native archive without
  first constructing a Python environment.
- CLI startup, process lifecycle, update target matching, and exit behavior no
  longer depend on a project's Python packages.
- `glr`, `glr-hostd`, and the operating Skills advance through one checksummed
  release unit.
- The Python SDK stays focused on learning and adapter integration instead of
  deployment orchestration.

### Negative

- JSON and SQLite compatibility now cross a Rust/Python boundary and require
  explicit conformance tests.
- Release CI must build and smoke four native targets in addition to Python and
  provider SDK artifacts.
- Same-release checksums do not provide code signing or notarization; those
  remain separate future gates.

### Neutral

- Capture, research acquisition, trainers, evaluators, players, and concrete
  game launchers remain project roles. Moving the composition root to Rust does
  not move game-specific policy into GLR core.
- A source build can run `update --check`, but the updater is not considered
  release-accepted until the first archive using the declared naming contract
  is published and smoke-tested from a fresh download.

## Alternatives Considered

**Keep the Python CLI and package it with PyInstaller.** Rejected because it
retains a Python-shaped deployment, produces large opaque bundles, and does not
align with the existing Rust Runtime Host release lane.

**Ship a thin Rust wrapper around the Python CLI.** Rejected because it keeps
two runtimes and two failure surfaces while giving the Agent only the appearance
of a standalone entrypoint.

**Rewrite the learning SDK in Rust.** Rejected because the distribution problem
does not justify moving learner, tensor, TorchRL, or Gymnasium integrations away
from their primary ecosystem.

**Allow the updater to run package-manager commands for every project.**
Rejected because arbitrary trainer and adapter dependencies are project-owned,
may be pinned for reproducibility, and must not be mutated by a framework
self-update.

## References

- ADR-0001: Keep the runtime contract learner-neutral
- ADR-0012: Use a Runtime Host and engine Provider SDKs
- ADR-0015: Add an agent-first local control plane
- `crates/glr-cli`
- `.github/workflows/release.yml`
- `.agents/skills/glr-cli/SKILL.md`
