# Declare the environment a role receives

A project's roles usually need configuration from the outside: a dataset root, a
device selector, an endpoint, a licence server. Without a declaration, that
contract lives wherever the project happened to write it down — a README, a
justfile, or the memory of whoever set the machine up — and each role reads
`os.environ` and interprets a missing value on its own. One role treats an
unset variable as "use the default", another crashes, and a third proceeds with
an empty string where a path was promised.

`glr.project.v1` lets the project state that contract in the manifest instead, so
the CLI can resolve it, report it, and refuse a run that cannot supply it.

Everything here is optional. A project that declares no `environment` table
behaves exactly as it did before.

## Declare the environment

Add a top-level `[environment]` table for values every role receives, and a
`[role.environment]` table for the values one role overrides:

```toml
[environment]
RENDER_DEVICE = "cpu"
SYNTHETIC_DATASET_ROOT = "${LOCAL_DATA_CACHE}/synthetic-v1"
LICENCE_SERVER = "synthetic-licence.invalid:27000"

[trainer.environment]
RENDER_DEVICE = "cuda"

[researcher.environment]
RESEARCH_DEPTH = "2"
```

The same keys work in `glr-project.json`:

```json
{
  "environment": { "RENDER_DEVICE": "cpu" },
  "trainer": {
    "argv": ["python", "-m", "synthetic.trainer"],
    "environment": { "RENDER_DEVICE": "cuda" }
  }
}
```

The roles that may declare their own table are `runtime`, `trainer`, `player`,
`researcher`, `planner`, and `evaluator`. Role tables override the project table
**key by key**: with the manifest above, the trainer receives
`RENDER_DEVICE=cuda` and still receives `SYNTHETIC_DATASET_ROOT` and
`LICENCE_SERVER` from the project table.

## How a value is resolved

| Declared value | Result |
| --- | --- |
| `"cpu"` | Passed through unchanged, reported as `literal`. |
| `"${LOCAL_DATA_CACHE}/v1"` | `${LOCAL_DATA_CACHE}` is expanded from the **process** environment; the rest of the string is kept verbatim. |
| `"${MISSING}"` | Refusal. The run is stopped before its role starts and the error names both the key and what is missing. |

Resolution happens once, before the run row is written, so a variable that
cannot resolve never leaves behind a run that launched a game and then
discovered its trainer had no dataset.

**The real process environment outranks the manifest.** If the caller already
exported `RENDER_DEVICE=metal`, that value wins, whatever the manifest declares.
An export is how someone overrides a checked-in default for one run, so the
manifest must not be able to shadow it. In `doctor` output such a variable is
reported with `"source": "process"`.

Only `${NAME}` is interpolated, where `NAME` matches
`^[A-Za-z_][A-Za-z0-9_]*$`. A malformed reference such as `${}` or `${BAD-NAME}`
is rejected when the manifest is loaded. An unterminated `${` is an ordinary
literal.

## Reserved names

The `GLR_` namespace belongs to the CLI. It clears inherited `GLR_*` variables
before spawning a child and then publishes the values that child owns
(`GLR_RUN_ID`, `GLR_TRIAL_ID`, `GLR_MODEL_BUNDLE`, …). A declared key in that
namespace is therefore rejected at manifest load time, before any process
starts:

```text
invalid configuration: project.environment cannot declare 'GLR_RUN_ID': the GLR_* namespace belongs to the CLI
```

Use the variables the CLI publishes instead of redeclaring them. Declared keys
also reuse the `game.environment` key shape, so `RENDER-DEVICE` and `9LIVES` are
rejected the same way `game.environment` rejects them.

## Check it before a run

`glr doctor` resolves the declaration for every **configured** role and reports
what each one would receive:

```powershell
glr --project . --json doctor
```

```json
{
  "roles": [
    {
      "role": "trainer",
      "configured": true,
      "available": true,
      "environment": {
        "role": "trainer",
        "ready": true,
        "variables": [
          { "name": "RENDER_DEVICE", "source": "literal", "secret": false, "value": "cuda" },
          { "name": "SYNTHETIC_DATASET_ROOT", "source": "interpolated", "secret": false, "value": "C:/data/synthetic-v1" }
        ],
        "unresolved": []
      }
    }
  ]
}
```

A role that is not configured never runs, so it reports no environment at all —
an unresolvable variable on an absent role cannot stop a run.

When something does not resolve, `doctor` reports it and **exits non-zero**:

```json
"unresolved": [{ "name": "SYNTHETIC_DATASET_ROOT", "missing": ["LOCAL_DATA_CACHE"] }]
```

Fix it by exporting the variable, or by declaring a literal:

```powershell
$env:LOCAL_DATA_CACHE = "C:/data"
glr --project . doctor
```

## What a run records

Every run records the environment the roles it started actually received, next
to the run's other configuration metadata:

```json
{
  "role_environment": {
    "role": "trainer",
    "ready": true,
    "variables": [
      { "name": "RENDER_DEVICE", "source": "literal", "secret": false, "value": "cuda" },
      { "name": "LICENCE_TOKEN", "source": "interpolated", "secret": true }
    ],
    "unresolved": []
  }
}
```

A `goal` run records one entry per role it started (`researcher`, `planner`,
`trainer`, `evaluator`).

**Secrets are recorded as received, never as content.** A name that reads like a
credential — `SECRET`, `TOKEN`, `PASSWORD`, `PASSWD`, `CREDENTIAL`, `API_KEY`,
`ACCESS_KEY`, `PRIVATE_KEY`, or `KEY` as a whole word — keeps its `secret: true`
flag and has no `value` field, in the run record, in `doctor` output, and in the
emitted receipt. The role still receives the resolved value; only the record
omits it.

## Scope

The declared environment reaches the six manifest roles. The project-owned
capture recorder receives the project-wide table. `glr host` receives none: its
command line comes from the caller, not the manifest, so a declared value must
not leak into an arbitrary program.

Game instances are configured by `game.environment`, which is unchanged by this
declaration.

Reading a `.env` file is deliberately **not** supported. It would add a second
machine-local configuration source whose contents depend on where the project
is checked out, and it would place secret values inside the project root where
packaging and backup tooling would copy them. Interpolate from the process
environment instead.

## Related

- [ADR-0046: Let a project declare the environment its roles receive](../decisions/0046-declare-the-environment-a-role-receives.md)
- [ADR-0042: Pin one entry point per project](../decisions/0042-pin-one-entry-point-per-project.md)
- [Game launch guide](game-launch.md)
- [Project output layout](project-output-layout.md)
