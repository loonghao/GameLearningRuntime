# Pin one entry point per project

An unattended agent gets a fresh context every round and must answer a question
the runtime never asks: *which command is live here?*

One Windows external-attach integration measured what happens when nothing
answers it. The project had grown **seven** plausible training launchers, **eight**
per-session learner copies diverging at 33 KB to 139 KB, and **310** loose
executables nobody dared delete. A fix applied to one learner copy never reached
the other seven. The project then trained a full day against an unreadable
inherited value table while `inherited_rows: 974` reported healthy every round.

Nothing raised an error. That is the failure: **nobody could say which code had
actually run.**

GLR already fails closed on host readiness, environment liveness, configuration
fingerprinting and checkpoints. None of those checks mean anything if the code
that ran is not the code anyone believes is running. `entry-point-v1` closes that
gap.

Everything here is optional. A project that declares nothing behaves exactly as
it did before.

## Declare the entry point

Add an `[entry_point]` table to `glr-project.toml` (or the same keys to
`glr-project.json`):

```toml
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
strict = false
```

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `schema_version` | yes | — | Must be `glr.entry-point.v1`. |
| `id` | yes | — | Stable name for the door. Matches `^[a-z][a-z0-9_.-]*$`. |
| `command` | yes | — | Provenance text. **GLR never executes it.** |
| `version` | yes | — | Which version of the entry is live. |
| `strict` | no | `false` | Refuse a drifting run instead of only recording it. |
| `invariants` | no | `[]` | Single-owner invariants, checked at run start. |

`command` is documentation the runtime can show you, not a role to run. GLR is a
control plane; it does not become your launcher.

## Claim the entry from the launcher

The invoking process declares which door it came through with two environment
variables:

```powershell
$env:GLR_ENTRY_ID = "campaign-driver"
$env:GLR_ENTRY_VERSION = "1.4.0"
glr --project . train
```

`GLR_ENTRY_VERSION` is optional. When it is set it must match the declared
`version`; when it is absent, only the id is compared.

The runtime compares the claim against the declaration and records the outcome in
the run's metadata:

| Status | Meaning |
| --- | --- |
| `undeclared` | The project declares no entry point. Nothing is compared. |
| `matched` | The run came through the declared door. |
| `entry_drift` | An entry point is declared and the run did not come through it. |

**A run that claims nothing at all is drift, not an unknown.** The project told
the runtime which door exists; a run with no provenance did not come through it.

## Choose strict or record-only

- `strict = false` (default) — a drifting run **runs** and is recorded as
  `entry_drift`. You find out afterwards, and you still have the run.
- `strict = true` — a drifting run is **refused before the run row is created**,
  with exit code `79`. No run record, no attached role, no consumed budget.

Failing before the expensive part is the point. Discovering a configuration fault
by burning a day of training budget is the most expensive way to find it.

## Declare single-owner invariants

An invariant asserts that **exactly one** file under a root contains a marker —
"exactly one module defines the learner", "shared modules have one home":

```toml
[[entry_point.invariants]]
id = "single-learner"
root = "src"
suffix = ".py"
marker = "class Learner"
```

| Key | Required | Default | Meaning |
| --- | --- | --- | --- |
| `id` | yes | — | Stable name, unique within the entry point. |
| `root` | yes | — | Project-relative directory to walk. |
| `marker` | yes | — | Literal, case-sensitive substring. |
| `suffix` | no | none | Restrict candidates to filenames ending in this, e.g. `.py`. |

Each invariant reports one of:

| Status | Meaning |
| --- | --- |
| `ok` | Exactly one file carries the marker. |
| `missing` | No file carries it. |
| `multiple` | Two or more files carry it — **all of them are named**. |
| `truncated` | The scan hit a bound, so "exactly one" could not be verified. |

`multiple` names **every** matching path, not just the first two. A project with
eight learner copies needs all eight named to converge on one.

These are checked at **run start**, unconditionally, whenever they are declared.
The filesystem work is trivial; the point is that it always runs and that a
violation is a typed error instead of a review comment.

`marker` is a literal substring, not a regular expression. The capability imports
nothing beyond the standard library so it stays cheap enough to run in
pre-commit, in CI, and at the top of every unattended round — **a guard that is
inconvenient gets skipped, and a skipped guard is not a guard.**

## One aggregate doctor

`glr doctor` merges role readiness, task readiness, the run context, the goal
binding, the entry-point attestation, the invariant results, and the last run's
verdict into one report and one exit code:

```powershell
glr --project . --json doctor
```

```jsonc
{
  "data": {
    "ready": true,
    "entry_point": {
      "status": "matched",
      "declared": { "id": "campaign-driver", "command": "python -m campaign.driver",
                    "version": "1.4.0", "strict": true },
      "observed": { "id": "campaign-driver", "version": "1.4.0" },
      "strict": true,
      "invariants": [
        { "id": "single-learner", "status": "ok", "matches": ["src/learner.py"] }
      ],
      "ready": true,
      "scanned_files": 42,
      "elapsed_ms": 3
    },
    "last_run": { "run_id": "run-...", "status": "succeeded",
                  "entry_point": { "status": "matched" } }
  }
}
```

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Every included check passed. |
| `4` | At least one check failed. |

The declared entry point is in the JSON directly — no log parsing, no
reconstructing it from a process table.

The last run's verdict is **reported, not gating**. A run that failed yesterday is
information for whoever reads the report; it is not evidence that the project is
unready today, and failing on it would change behaviour for projects that declare
nothing.

## Exit codes

| Code | Where | Meaning |
| --- | --- | --- |
| `0` | `doctor`, runs | Everything passed. |
| `2` | any command | A declared invariant failed — a typed contract violation. |
| `4` | `doctor` | At least one included check failed. |
| `79` | a strict run | The run was refused: entry drift, before attach. |

A refused run leaves no trace in the store, so a scheduler can retry it once the
launcher is fixed without paying for it twice.

## Bounds

The drift check is **bounded**, not constant: cost grows with the tree, and the
bound is what keeps it inside a budget a pre-commit hook, a CI job and the top of
an unattended round can all afford. At most 32 invariants, 32 directory levels
deep, 20 000 files scanned, 64 MiB read in total, 1 MiB read per file. `.git`,
`target`, `node_modules`, `.venv` and `__pycache__` are skipped, and symlinks are
never followed, so the walk cannot loop or leave the project root.

A tree at the bound costs seconds, not milliseconds: measure `elapsed_ms` in the
report if your tree is unusually large. Typical project sizes stay far below it —
1 000 source files scan in roughly 100 ms on a cold cache — and the scan stops at
the bound instead of growing without limit.

Exceeding a bound is reported as `truncated` and **fails closed** — an unverified
invariant must not read as a satisfied one. The report names the files that were
skipped so you can raise a bound deliberately rather than guess.

## Add it to a round

The guard is cheapest at the top of a round, before anything is spent:

```powershell
glr --project . doctor
if ($LASTEXITCODE -ne 0) { throw "project is not ready; refusing to start a round" }
```

`doctor` is a diagnosis, not a run, so it never carries `GLR_ENTRY_ID`. The
attestation therefore decides the verdict **only for a strict project**, which is
the one whose runs would be refused anyway:

| Project | Bare `doctor` verdict |
| --- | --- |
| No `[entry_point]` table | `undeclared` — never affects the exit code |
| Declared, `strict = false` | `entry_drift` is **reported, not gated** — exit `0` |
| Declared, `strict = true` | `entry_drift` fails with exit `4`, because the run would be refused with `79` |

Invariant results always decide the verdict: they are facts about the tree, not
about the caller. If you want the round guard to insist on the declared door, set
`strict = true` and let the launcher claim its entry.

## Troubleshooting

**Every run reports `entry_drift` and nothing has changed.** The launcher does not
set `GLR_ENTRY_ID`. Either set it in the launcher, or — if the launcher really is
the canonical one — correct the declared `id`.

**`multiple` names two files that look like copies.** That is the drift this
capability exists to catch: pick one home, delete the other, and consider an
invariant over the shared module so it cannot split again.

**`truncated` on a large tree.** A candidate exceeded the 1 MiB per-file bound or
the 64 MiB total. Narrow `suffix` or `root` so the scan covers only your own
source, rather than raising the bound.

## Related

- [ADR-0042](../decisions/0042-pin-one-entry-point-per-project.md) — the decision,
  rejected alternatives, and consequences.
- [Agent onboarding](agent-onboarding.md) — the command chain for a fresh agent.
- [Anti-fork gate](fork-gate.md) — drift against upstream; this is drift within
  the project.
