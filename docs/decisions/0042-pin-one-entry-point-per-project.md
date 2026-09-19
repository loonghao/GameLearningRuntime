# ADR-0042: Pin one entry point per project

## Status

Accepted

## Context

An unattended agent arrives at every round with a fresh context and must answer
a question the runtime never asks: *which command is live here?*

One Windows external-attach integration measured the cost of leaving that
question unanswered. The project had grown **seven** plausible training
launchers — a campaign driver, a stage pipeline, an exploration learner, a
boss-mission runner, a night-round driver, an overnight loop, and a legacy
pixel-capture resume script — plus **eight** per-session learner copies
diverging at 33 KB to 139 KB, and **310** loose executables at the top of the
script directory. A fix applied to one learner copy never reached the other
seven. The project then trained a full day against an unreadable inherited
value table while `inherited_rows: 974` reported healthy every round.

Nothing raised an error. That is the failure: nobody could say which code had
actually run.

GLR already fails closed on what it owns — readiness (ADR-0029), liveness,
configuration fingerprinting, checkpoints, the anti-fork gate (ADR-0037). It
has no concept of *the entry point*. Entry drift is therefore the one fault
that silently voids every other check: a verified host, a live environment and
a matching fingerprint say nothing when the code that ran is not the code
anyone believes is running.

Every GLR-driven project will rebuild this guard, because the drift follows
from GLR's own model — long unattended runs driven by agents that get a fresh
context each time — not from carelessness.

## Decision

Add an optional capability, **`entry-point-v1`**, to the Rust `glr` CLI. It has
four parts: a declared entry point, a launch attestation, generic single-owner
invariants, and one aggregate `doctor` verdict.

### 1. Declared entry point

A project may declare one canonical entry point in `glr-project.toml`:

```toml
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
strict = false

[[entry_point.invariants]]
id = "single-learner"
root = "src"
suffix = ".py"
marker = "class Learner"
```

- `id` matches `^[a-z][a-z0-9_.-]*$` and names the entry.
- `command` is provenance text. **GLR never executes it.** The entry point
  describes the door; it is not a role.
- `version` is free-form printable text, declared alongside the command so
  "which command" and "which version of it" age together.
- `strict` defaults to **`false`**: a drifting run is recorded, not refused.

The declared entry is surfaced in `glr doctor` JSON and written into every run
record, so "what is the entry point" is a recorded fact rather than something
each session re-derives from seven plausible launchers.

### 2. Launch attestation

The invoking process declares itself through two environment variables,
`GLR_ENTRY_ID` and the optional `GLR_ENTRY_VERSION`. At run start the runtime
compares the observed entry with the declared one and records the result in the
run's `metadata_json` under `entry_point`.

The status vocabulary is deliberately small:

| Status | Meaning |
| --- | --- |
| `undeclared` | The project declares no entry point. Nothing is compared. |
| `matched` | The observed entry is the declared entry. |
| `entry_drift` | An entry point is declared and the observed entry is absent or different. |

Absent provenance against a declared entry point *is* drift, not an unknown:
the runtime was told which door exists and the run did not come through it.

When `strict = true`, a drifted run is refused **before** the run row is
created and before any role is attached, with exit code `79`. No run record, no
attach, no consumed budget — the refusal costs nothing, which is the whole
point of failing before rather than after the expensive part.

### 3. Single-owner invariants

Each `[[entry_point.invariants]]` entry asserts that **exactly one** file under
`root` contains the literal `marker`, optionally restricted to filenames ending
in `suffix`. The runtime walks the tree at run start and returns a typed error
on violation.

The check is unconditionally performed when declared — not only in CI, not only
in review. The filesystem work is bounded; GLR's contribution is that it always
runs and that a violation is a typed error instead of a review comment.

A violation names **every** matching path, not just the first two. "Two modules
define the learner" is only actionable when the report says which two.

`marker` is a case-sensitive literal substring, not a regular expression. The
capability must import nothing beyond the standard library so it can run in
pre-commit, in CI and at the top of an unattended round; adding a regex engine
would trade that property for expressive power the invariant does not need.

### 4. One aggregate `doctor` verdict

`glr doctor` merges role readiness, task readiness, the run context, the goal
binding, the entry-point attestation, the invariant results and the last run's
verdict into one report and one exit code: **`0` when every included check
passes, `4` when any of them fails**.

Which checks are *included* is where the design earns its keep, because
`doctor` is a diagnosis and not a run: it never carries `GLR_ENTRY_ID`.

- **Invariant results always decide the verdict.** They are facts about the
tree — two modules defining the learner is unreadiness whether or not anyone
launched anything.
- **The attestation decides the verdict only when the project declares
`strict = true`.** A non-strict project records `entry_drift` and reports it
in `doctor.data.entry_point.status`; gating on it there would fail every round
of every project that pins an entry point, which is the opposite of the
capability's purpose. For a strict project the attestation *is* a project
fact: `entry_point_gate` refuses the run with `79` before the run row exists,
so `doctor` reporting readiness would be a lie the scheduler pays for.
- The last run's verdict is **reported but not gating**. A failed run yesterday
is information for the scheduler, not evidence that the project is unready
today; making past failures fail `doctor` would change behaviour for projects
that declare nothing.

### Bounds

The drift check is **bounded**, not constant: cost grows with the tree, and the
bound is what keeps it inside a budget a pre-commit hook, a CI job and the top
of an unattended round can all afford. At most 32 invariants, 32 directory
levels deep, 20 000 files scanned, 64 MiB read in total and 1 MiB read per
file, and a skip-list for `.git`, `target`, `node_modules`, `.venv` and
`__pycache__`. Symlinks are never followed, so the walk cannot escape the
project root or loop. A tree at the bound costs seconds; typical project sizes
sit far below it, and the scan stops at the bound rather than growing without
limit.

### Backward compatibility

Every field is optional and every default preserves current behaviour. A
project with no `[entry_point]` table produces status `undeclared`, runs no
invariant, adds nothing to `doctor`'s `ready` computation, and behaves exactly
as it does today.

## Consequences

- "Which command is live" becomes a recorded fact instead of a per-session
  deduction, so an agent no longer has to choose among seven launchers.
- A drifted run is visible after the fact (`entry_drift` in the run record)
  even when `strict` is off, and costs nothing when it is on.
- Two learner definitions fail the run with both paths named, while the run is
  still cheap to abort.
- A scheduler branches on one `doctor` exit code instead of merging four
  outputs.
- The guard is cheap enough to run at the top of every round, so it does not
  get skipped.
- The check detects *declared* drift. A project that declares an entry point
  nobody sets `GLR_ENTRY_ID` for will report drift on every run — which is
  correct, and is the intended pressure to actually pin the door.
- GLR does not verify that the declared `command` is the thing that ran; it
  verifies that the caller *claimed* to be the declared entry. The capability
  makes drift visible and expensive to ignore, not impossible.

## Rejected alternatives

- **Execute the declared `command` and refuse anything else.** Rejected because
  GLR is a control plane, not a launcher; owning execution would move every
  project's process model into the runtime.
- **Infer the entry point from the process tree or `argv[0]`.** Rejected because
  it is platform-specific, fragile under wrappers and shells, and would report
  the launcher's parent rather than the launcher. An explicit declaration is
  both cheaper and honest about being a claim.
- **Accept a regular expression for `marker`.** Rejected because it would add a
  dependency to a check whose entire value is that it is cheap and portable
  enough to never be skipped.
- **Make the last run's failure fail `doctor`.** Rejected because it would
  change behaviour for projects that declare nothing, violating the
  compatibility guarantee, and because a past failure is not current
  unreadiness.
- **Report only the first two matching paths on an invariant violation.**
  Rejected because a project with eight learner copies needs all eight named to
  converge on one.
- **Introduce a new `entry-point` subcommand.** Rejected because the capability
  is a gate, not a workflow: it belongs at run start and inside `doctor`, which
  is already the aggregate command a scheduler calls.

## Related

- ADR-0029 bounds the runtime start readiness window; this gates whether the
  code about to run is the code anyone believes is running, a precondition for
  it.
- ADR-0037 gates drift against upstream; this gates drift within the project.
- ADR-0024 binds invocation-scoped run contexts, which the attestation is
  recorded alongside.
- [Entry point guide](../guides/entry-point.md).
