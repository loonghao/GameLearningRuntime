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

The public release check is anonymous by default. On a GitHub API rate-limit
response, provide an existing token only as the process-scoped
`GLR_GITHUB_TOKEN`. Never echo it or store it in `glr-project.json`.

## Project configuration

`glr-project.json` is strict `glr.project.v1`. Every command is a fixed argv array executed with
`shell=False`; placeholders must occupy a whole argv element. Paths are project-relative and must
stay inside the project.

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
  }
}
```

The example recorder argv is illustrative. A real recorder must target the reviewed game window
and write the paths provided by `GLR_CAPTURE_VIDEO` and `GLR_CAPTURE_INDEX`. Do not insert window
discovery, arbitrary scripts, secrets, or shell expressions into the config.

Project roles receive `GLR_PROJECT_ROOT`, `GLR_BRIDGE_PATH`, `GLR_RUN_ID`, `GLR_RUN_DIR`,
`GLR_STORE_PATH`, environment identity variables, and role-specific `GLR_*_PATH` variables.

## Commands

Use `--json` for compact `glr.cli-output.v1` output.

```powershell
glr --project . --json doctor
glr --project . --json runtime start
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
SHA-256-mismatched artifacts before writing. It registers the generated page as
`run-report` and remains safe to rerun. Its timeline and panels are driven only
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
