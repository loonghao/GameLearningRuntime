# Extend GLR with declarative VX tasks

Use `glr.toml` for project workflows that compose GLR with Python or other
project tools but do not define a new GLR runtime contract. Keep runtime roles
and environment identity in `glr-project.toml` (or legacy
`glr-project.json`).

## Configure a VX-backed task

Declare Python and uv versions in the project's `vx.toml`, then add a strict
task registry at the project root:

```toml
schema_version = "glr.tasks.v1"

[tasks.prepare]
description = "Prepare the training inputs"
runner = "vx"
argv = ["uv", "run", "--no-sync", "python", "tools/prepare.py"]
timeout_seconds = 600

[tasks.season]
description = "Run one bounded training season"
runner = "vx"
argv = [
  "uv", "run", "--no-sync", "python", "tools/run_season.py",
  "--profile", "{profile}",
  "--max-matches", "{max_matches}",
  "--result", "{task_result}",
]
depends = ["prepare"]
timeout_seconds = 7200

[tasks.season.parameters.profile]
type = "string"
required = true

[tasks.season.parameters.max_matches]
type = "integer"
default = 20
minimum = 1
maximum = 100

[tasks.season.result]
schema = "glr.season-result.v1"
required = true
```

`runner = "vx"` prepends `vx` to the fixed argument array. In this example VX
resolves uv, Python, and the project environment; GLR does not install Python or
duplicate VX environment management. `--no-sync` is appropriate after project
setup or in a locked CI job. Omit it when the task intentionally permits uv to
synchronize the locked environment.

## Inspect and run

```powershell
glr --project . --json doctor
glr --project . --json task list
glr --project . --json task show season
glr --project . --json task run season `
  --set profile=league-legends/native-100024 `
  --set max_matches=20
```

The same commands work in POSIX shells without changing the TOML. Use one
`--set NAME=VALUE` per parameter. Supported types are `string`, `integer`,
`boolean`, and project-relative `path`. Integer bounds are enforced before any
process starts.

Dependencies run once in declared DAG order. GLR rejects cycles, unknown tasks,
unknown parameters, duplicate assignments, partial placeholders, unknown TOML
fields, symlinked task files, out-of-project working directories, and invalid
timeouts before execution.

## Runtime contract

Every process receives:

- `GLR_PROJECT_ROOT`
- `GLR_TASK_NAME`
- `GLR_TASK_DIR`
- `GLR_TASK_RESULT`

The placeholders `{project_root}`, `{task_dir}`, and `{task_result}` provide the
same values as complete argv entries. Parameter placeholders also occupy a
complete argv entry; string interpolation such as `--profile={profile}` is
rejected.

Each execution writes `.glr/tasks/<execution-id>/result.json` using
`glr.task-result.v1`, plus one `task.log` per executed task. A timeout returns
exit code 124. When a task declares `result`, GLR also requires `{task_result}`
to be regular JSON with the exact declared `schema_version`; an invalid result
returns 78. The first failed dependency stops the remaining graph.

Task success proves only that the configured processes exited successfully. It
does not prove a live bridge handshake, authoritative match outcome, completed
season, improved policy, or accepted model. Those claims still require the
corresponding GLR run-store and evaluator evidence.

Treat `glr.toml` as trusted local project configuration. Direct argv execution
removes shell parsing but does not sandbox the selected executable.
