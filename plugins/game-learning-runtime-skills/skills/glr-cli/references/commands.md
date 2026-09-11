# GLR CLI command and contract reference

## Install and inspect the standalone distribution

Download `glr-{version}-{rust-target}.zip` and `SHA256SUMS` from the matching
GitHub Release, verify the archive digest, extract it, and put `glr` plus
`glr-hostd` on `PATH`. The archive also contains both repository-owned Skills
under `skills/`; no Python installation is required to run the CLI.

Start every project operation with:

```powershell
glr --version
glr --project . --json doctor
glr --json update --check
```

`doctor` verifies the strict project file, bridge path, data directory, and
configured executable roles. It does not prove a live bridge handshake or game
acceptance.

Only after an explicit user update request, apply the exact-target release:

```powershell
glr --json update --yes
glr --json update --yes --skills-dir .agents/skills
glr --json update --yes --no-skills
```

The default update scope is the CLI, sibling Runtime Host, and project Skills.
`--skills-dir` must name an explicitly selected project Skills directory; it
does not update a user-level Agent Plugin installation. Use the host's plugin
manager to replace a plugin package, or copy the package's `skills/` payload
into the project directory intentionally before running a project update.
The updater requires HTTPS, a matching target manifest, and the published
`SHA256SUMS`; it never runs an installer script or changes project/trainer data.
Re-run `--version`, `doctor`, and `update --check` after an update.

The public release check uses GitHub's latest-release asset link instead of the
REST API, so it does not consume anonymous API quota. The updater derives the
version, exact target archive, and digest from the published `SHA256SUMS`.

## Project configuration

New projects use `glr-project.toml`, with the same strict `glr.project.v1` schema
as legacy `glr-project.json`. Discovery searches upward from any project
subdirectory and stops at the nearest manifest. Two manifests in one directory,
symlinked manifests, and unknown fields are rejected. In TOML, omit optional
roles or capture tables instead of writing JSON `null`.

Every command is a fixed argv array executed without a shell; placeholders
occupy a whole argv element. Paths are manifest-relative and remain inside the
project. The following JSON example is retained for existing configurations;
new projects should use the equivalent TOML layout below.

```json
{
  "schema_version": "glr.project.v1",
  "environment_id": "example.adventure-v1",
  "environment_family": "action-rpg",
  "protocol_version": "1.0",
  "data_dir": ".glr",
  "bridge_path": "bridge",
  "runtime": {"argv": ["python", "tools/runtime.py", "{bridge_path}"]},
  "trainer": {"argv": ["python", "tools/train.py"]},
  "player": {"argv": ["python", "tools/play.py", "{bundle}"]},
  "researcher": {"argv": ["python", "tools/research.py", "{research_path}"]},
  "planner": {"argv": ["python", "tools/plan.py", "{trial_path}"]},
  "evaluator": {"argv": ["python", "tools/evaluate.py", "{evaluation_path}"]},
  "capture": {
    "argv": ["python", "tools/record_window.py", "{capture_video}", "{capture_index}"],
    "required": true,
    "stop": "stdin-q",
    "video_file": "capture.mp4",
    "index_file": "capture-index.jsonl",
    "codec": "h264",
    "frame_rate": 12,
    "width": 640,
    "height": 360
  },
  "lifecycle": {
    "schema_version": "glr.lifecycle.v1",
    "configs": [
      {"owner": "training", "path": "training.json", "schema_version": "glr.training.v1"},
      {"owner": "reward", "path": "reward-safety.json", "schema_version": "glr.reward-safety.v1"}
    ],
    "modes": ["train", "goal-evaluate", "frozen-playback"]
  }
}
```

`lifecycle.configs` gives each project input exactly one owner. `doctor` rejects
missing files, schema drift, duplicate owners, shared paths with conflicting
owners, unsupported modes, and duplicate required role entrypoints. Its JSON
output includes a SHA-256 manifest of the files it actually loaded. Existing
`glr.project.v1` files without `lifecycle` remain compatible, but generated
projects always include it.

The example recorder argv is illustrative. A real recorder must target the reviewed game window
and write the paths provided by `GLR_CAPTURE_VIDEO` and `GLR_CAPTURE_INDEX`. Do not insert window
discovery, arbitrary scripts, secrets, or shell expressions into the config.

Project roles receive `GLR_PROJECT_ROOT`, `GLR_BRIDGE_PATH`, `GLR_RUN_ID`, `GLR_RUN_DIR`,
`GLR_STORE_PATH`, environment identity variables, and role-specific `GLR_*_PATH` variables.
They also receive `GLR_PROJECT_MANIFEST`, with the selected absolute manifest
path, and may use the whole-argument `{project_manifest}` placeholder. These
machine-local paths are process context, not publishable training evidence.
On Windows, canonical paths can use extended-length or UNC syntax. Compare
filesystem identity with `Path.samefile()` when files exist, not raw strings.
If a reviewed legacy launcher requires another spelling, adapt it only at that
launcher boundary. A path-format or identity-check failure is not evidence that
the game is absent and must never cause an automatic second launch.

### Portable layout, extensions, and local paths

```toml
schema_version = "glr.project.v1"
environment_id = "example.environment-v1"
environment_family = "example-family"
protocol_version = "1.0"
data_dir = ".glr"
bridge_path = "src/example_adapter"

[runtime]
argv = ["uv", "run", "--frozen", "python", "-m", "example_runtime"]
[trainer]
argv = ["uv", "run", "--frozen", "python", "-m", "example_training"]
[player]
argv = ["uv", "run", "--frozen", "python", "-m", "example_playback", "{bundle}"]

[extensions.example]
config = "config/runtime.toml"
```

Each extension has exactly one `config` field: an existing, portable relative,
non-symlink file inside the project. GLR checks the mount; the named extension
owns the file's strict schema and actions. Python exposes the resolved file as
`load_project(path).extensions["example"]`. No arbitrary plugin import or script
execution is loaded from an extension reference.

Keep a default game directory such as `game` in the extension config. If a
different installation is required, let that extension accept only an explicit
`[game] directory` override in `config/runtime.local.toml`, ignored by version
control. Do not recursively merge arbitrary local fields, commands, credentials,
policy, or reward settings. GLR does not automatically load local overrides.
`resolve_game_directory(project.root, configured_directory)` accepts an existing
project-owned relative directory or an explicitly configured absolute directory;
relative escapes are rejected. It never installs, discovers, or launches games.

Commit the root manifest, package metadata, dependency locks, generic default
config, tests, and project-owned setup instructions. Do not copy a virtualenv:
recreate it from the lock after clone. Keep adapters as importable semantic
packages, not owners of hidden shared environments. Before removing an old
environment, verify the new interpreter/module origins and run synthetic tests,
`doctor`, bounded training, and artifact verification from a different checkout
path and a nested working directory. Game payloads, local paths, credentials,
private traces, and licensed assets are not implied to be redistributable.

Fresh-clone acceptance has separate gates: dependency setup; config/doctor;
synthetic training/reproduction; authorized live runtime binding; whole-match
evidence. Report which gate ran. TOML source support must be built/released and
adopted by the consumer before claiming its installed GLR is compatible.

## Commands

Use `--json` for compact `glr.cli-output.v1` output.

```powershell
glr --project . --json doctor
glr --project . --json runtime start
glr --project . --context config/contexts/ranked.toml --json doctor
glr --project . --context config/contexts/ranked.toml --json train
glr --project . --json task list
glr --project . --json task show season
glr --project . --json task run season --set profile=example/default
glr --project . --json train
glr --project . --json train --no-capture
glr --project . --json goal run --goal goals/reach-destination.json
glr --project . --json runs list --status succeeded --limit 20
glr --project . --json runs show run-0123456789abcdef
glr --project . --json query entities --world forest --kind shrine --name 土地庙
glr --project . --json query routes --world forest --to-entity shrine.forest-1
glr --project . --json query research --tag navigation --category strategy
glr --project . --json query research --verified-only
glr --project . --json knowledge export --output artifacts/spatial-knowledge.json
glr --project . --json knowledge import --input artifacts/spatial-knowledge.json
glr --project . --json play --bundle artifacts/model-bundle
glr --project . --json report build run-0123456789abcdef
```

Project-local `glr.toml` tasks use strict `glr.tasks.v1`. Prefer
`runner = "vx"` and an argv beginning with `uv`, `run` for Python workflows so
VX resolves the locked toolchain and environment. GLR rejects shell strings,
partial placeholders, dependency cycles, unknown parameters, unsafe paths, and
unbounded timeouts. Task receipts under `.glr/tasks/` record process outcomes;
they are not authoritative gameplay or learning evidence.

Invocation-scoped `glr.run-context.v1` files bind generic labels and owned
JSON/TOML inputs to `doctor`, `runtime start`, `train`, `goal run`, or `play`.
GLR freezes and re-verifies source/input hashes, exports `GLR_RUN_CONTEXT` plus
`GLR_RUN_CONTEXT_SHA256`, and persists `run-context.json` with the run. Use
`load_inherited_run_context(project)` in Python roles. Keep product-specific
setup and status policy in VX tasks rather than adding core subcommands.

`train` and `goal run` record lifecycle, events, metrics, logs, capture artifacts, and hashes under
the configured data directory. Training tensors and transitions remain checksummed artifacts or
JSONL datasets; SQLite is the query projection, not the tensor store.

## Offline run reports

`report build` renders `glr.run-report.v1` as a self-contained HTML review
projection under the selected run directory:

```powershell
glr --project . --json report build <run-id>
glr --project . --json report build <run-id> --output review/report
```

The report builder rejects missing, symlinked, out-of-run, size-mismatched, or
SHA-256-mismatched evidence artifacts before writing. It omits prior
`run-report` outputs to avoid a self-referential hash, registers the generated
page as `run-report`, and remains safe to rerun. Its timeline and panels are driven only
by persisted events: `navigation.route_sample` for route traces,
`progression.item_unlocked` / `progression.catalog_snapshot` for unlocks, and
`match.result` for completed matches. Treat `match_kind=pvp` as meaningful only
when the adapter has an authoritative PvP result. Screenshot and video links
must point to authorized checksummed artifacts; no report panel is a claim of
live-game acceptance.

## Goal loop files

The input is `glr.agent-goal.v1`. It must declare:

- a stable goal ID and objective;
- the environment family;
- one or more metric/operator/target/source success criteria;
- maximum trials, training steps, wall seconds, and research sources;
- allowed media from official rules, text guides, video tutorials, or runtime traces.

The researcher writes `glr.research-bundle.v1`. On later failed trials it receives
`GLR_PREVIOUS_RESEARCH_PATH` and `GLR_PREVIOUS_EVALUATION_PATH` so it can gather additional allowed
evidence within the same global source budget.

The planner writes `glr.trial-plan.v1`. Reward terms contain only names, metric IDs, bounded numeric
weights, rationales, and referenced finding IDs. They cannot contain expressions or import paths.

The evaluator writes `glr.goal-evidence.v1`. Every evidence item must match a metric already saved
to `GLR_STORE_PATH` during the current trial, including value, source, and authority. Only
`authoritative` runtime evidence can satisfy a goal criterion.

## Reuse and reproduction gates

- `knowledge import` requires exact `environment_id` and `protocol_version`; imported entities and
  routes are advisory.
- `query research` returns exact-environment, same-family, and generic findings, excluding rejected
  findings. Use `--verified-only` when a decision requires prior runtime confirmation.
- `play` verifies every model-bundle byte and requires exact environment/protocol compatibility.
  Run live evaluation again in the new game instance; do not report reproduction from load success.

Stop when a declared budget is exhausted, a required role/capture fails, the runtime identity is
uncertain, or authoritative success evidence is missing. Return the failed gate and the relevant
run ID rather than relaxing the contract.
