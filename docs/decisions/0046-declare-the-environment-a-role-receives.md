# ADR-0046: Let a project declare the environment its roles receive

## Status

Accepted

## Context

Every GLR project that needs to pass configuration to its roles invents the same
mechanism. A trainer that needs a dataset root, a runtime that needs a device
selector, or a researcher that needs an endpoint reads `os.environ` itself, and
the project then documents — outside the manifest — which variables a caller
must export first. The knowledge lives in a README, a justfile, or the memory of
whoever set the project up, and it is rediscovered each time a role moves
between a workstation and a scheduler.

The injection half of this capability already existed and was correct:
`TrainingLauncher.run(..., environment=...)` accepts an environment mapping, and
`game.environment` already declares variables for launched game instances with
validated key shapes and reserved names. What was missing was the layer above
it: a project could not *declare* what its roles receive, so nothing could
resolve, report, or refuse those variables before a process started.

`glr.project.v1` was a closed set: a top-level `[environment]` table was rejected
as an unknown field, in both the Python loader and the Rust schema mirror that
must stay in step with it.

This is the other half of [ADR-0042: Pin one entry point per project](0042-pin-one-entry-point-per-project.md).
That ADR pinned *which command is live*; this one pins *what that command and
its roles are handed*, so an unattended agent does not have to reconstruct a
project's invocation contract from its source.

## Decision

**`glr.project.v1` accepts a project-wide `environment` table and a per-role
table that overrides it. The CLI resolves both against the process environment
before it starts anything, refuses a variable that cannot resolve, and records
what it handed each role.**

- The project-wide table applies to every declared role. A role table
  (`runtime.environment`, `trainer.environment`, …) wins key by key; keys it
  does not name keep the project-wide value.
- A literal value passes through unchanged. `${NAME}` is interpolated from the
  **process** environment. A reference to a variable the process environment
  does not define fails closed and names the offending key — it is never
  substituted with an empty string.
- The real process environment outranks the declared table. A name it already
  defines keeps its own value, so an operator's export is never shadowed by the
  manifest.
- Declared keys reuse the `game.environment` key shape
  (`^[A-Za-z_][A-Za-z0-9_]*$`). A key in the `GLR_*` namespace is rejected while
  the manifest is loaded, before any process starts: the CLI owns that
  namespace, clears inherited `GLR_*` variables before spawning a child, and
  republishes the values it owns.
- `glr doctor` lists, per configured role, the variables that were declared and
  resolved, and reports the ones that were not. A run is refused before its role
  starts when a declared variable cannot resolve, and `doctor` exits non-zero.
- Each run records the resolved variables of the roles it started, next to the
  run's configuration metadata. A name that looks like a credential
  (`SECRET`, `TOKEN`, `PASSWORD`, `CREDENTIAL`, `API_KEY`, …) is recorded as
  received and never as content; its value is absent from the run record,
  `doctor` output, and the receipt.
- A project that declares no `environment` table behaves exactly as it did
  before, on both entry points.

Both entry points implement this: `src/game_learning_runtime/role_environment.py`
and `crates/glr-cli/src/role_environment.rs` accept the same tables, reject the
same names, and resolve with the same precedence, so a manifest that loads on
one loads on the other.

## Consequences

- A project states its invocation contract in the manifest, where `glr doctor`
  can check it, instead of in prose a caller has to find and trust.
- A missing export is now a refusal with a named variable, at run start, instead
  of a role that starts and fails later — or silently proceeds with an empty
  string where a path was promised.
- Resolution happens once, before the run row is written, so an unresolvable
  variable cannot leave behind a half-executed run that launched a game and then
  discovered its trainer was missing an input.
- Recording resolved variables makes a run reproducible from its record: a
  caller can see which values a role received without consulting the machine
  that ran it. Secret values are deliberately excluded, so the record is enough
  to reproduce the shape of the invocation and not enough to leak a credential.
- The declared environment reaches the six manifest roles. The project-owned
  capture recorder receives the project-wide table, and `glr host` — whose
  command line is supplied by the caller, not the manifest — receives nothing,
  so a declared value cannot leak into an arbitrary program.

## Rejected alternatives

- **Let each project keep loading its own environment** (the status quo):
  rejected because it is the thing that makes every project rediscover the same
  resolution, precedence, and refusal rules, and because the resulting contract
  is invisible to `doctor`.
- **Read a `.env` file from the project root**: rejected. It would introduce a
  second machine-local configuration source whose contents depend on where the
  project is checked out, and it would put secret values in a file inside the
  project root, where packaging and backup tooling would copy them. The issue's
  own proposal asks for interpolation from the process environment, not a second
  config source.
- **Resolve lazily, when a role reads the variable**: rejected because the
  failure would surface inside the role, after the process started, as an empty
  string or a crashed child rather than as a named refusal.
- **Let the declared table override the process environment**: rejected because
  an operator's own export must win. A manifest is checked in; an export is how
  someone overrides it for one run.
- **Allow declared `GLR_*` keys**: rejected because the CLI clears and
  republishes that namespace for every child. A declared value there would
  either be silently dropped or survive as a forgery of a value the CLI owns.
- **Mark secrets explicitly in the manifest** (for example
  `secrets = ["API_TOKEN"]`): rejected because a forgotten entry then fails
  open — the value would be recorded in plaintext by default. The lexical test
  fails closed instead: a name that reads like a credential is never recorded,
  whether or not anyone remembered to mark it.

## Related

- [ADR-0021: Resolve portable projects from one manifest](0021-portable-project-manifests.md)
- [ADR-0042: Pin one entry point per project](0042-pin-one-entry-point-per-project.md)
- [ADR-0040: Fail closed on a declared metric that is never emitted](0040-fail-closed-on-a-declared-metric-that-is-never-emitted.md)
- [Declared role environment guide](../guides/declared-role-environment.md)
