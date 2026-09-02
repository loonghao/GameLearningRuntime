# Operate GLR as an agent-first control plane

Use the `glr` CLI when an authorized game project already exposes a reviewed GLR bridge. The CLI
starts fixed project roles, records queryable run evidence, captures small-window review video,
pursues bounded goals, and reuses exact-environment knowledge or model bundles.

Use `glr-adapter-builder` instead when you need to implement observations, actions, transport,
lifecycle, target binding, or post-action verification.

## Install the primary entrypoint

Download the matching `glr-{version}-{rust-target}.zip` and `SHA256SUMS` from
the same GitHub Release. Verify the checksum, extract the archive, and put its
`glr` and `glr-hostd` executables on `PATH`. The archive includes the
`glr-cli` and `glr-adapter-builder` Skills. The Rust CLI is standalone; install
the Python package only when project-owned roles use its SDK.

Inspect the deployment before operating a project:

```powershell
glr --version
glr --project . --json doctor
glr --json update --check
```

`doctor` validates the project contract, paths, and configured executables. It
does not prove a live bridge handshake or game acceptance.

## Update managed GLR components

`glr update --check` is read-only. When the user explicitly requests an update,
run `glr update --yes`. The updater requires HTTPS, downloads the exact Rust
target archive and `SHA256SUMS`, verifies the release manifest and digest, then
replaces the CLI, sibling Runtime Host, and project Skills.

```powershell
glr --json update --yes
glr --json update --yes --skills-dir .agents/skills
glr --json update --yes --no-skills
```

It never runs an installer script or modifies game code, role dependencies,
virtual environments, models, datasets, or `glr-project.json`. SHA-256 verifies
same-release integrity; it is not publisher signature verification. Re-run
`--version`, `doctor`, and `update --check` after applying an update.
Checks are anonymous by default. If GitHub rate-limits the request, provide an
existing token only through the process-scoped `GLR_GITHUB_TOKEN`; never print
or persist it in the project.

## Configure the project

Create `glr-project.json` at the project root. Use project-relative paths and fixed argv arrays.
GLR never invokes a shell.

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

The recorder is project-owned because only the project knows the reviewed game window and capture
API. For a small game window, 640×360 at 12 FPS is a practical starting profile. Tune it based on
UI readability and learner needs rather than presentation quality.

Project roles receive `GLR_PROJECT_ROOT`, `GLR_BRIDGE_PATH`, `GLR_RUN_ID`, `GLR_RUN_DIR`,
`GLR_STORE_PATH`, environment identity, capture output paths, and goal-loop paths through
environment variables. They must stay bounded and validate the exact game target independently.

## Start the runtime and train

```powershell
glr --project . --json doctor
glr --project . --json runtime start
glr --project . --json train
```

`train` starts the recorder before the trainer and stops it afterward. A complete capture contains:

- a small H.264 MP4 for human review;
- an NDJSON index mapping `(episode_id, step_id)` to frame index and presentation timestamp;
- a manifest with dimensions, FPS, codec, sizes, and SHA-256 digests.

The index lets later tooling select human-approved video segments and align them with recorded
actions. GLR does not label a policy rollout as an expert demonstration. Apply the existing
demonstration provenance gate before behavior cloning or supervised ingestion.

For recorders that opt into `glr.capture-session.v1`, add a session block:

```json
"session": {
  "status_file": "capture-status.jsonl",
  "startup_timeout_seconds": 5,
  "heartbeat_timeout_seconds": 5,
  "minimum_frames": 1,
  "minimum_steps": 1
}
```

The CLI writes a start receipt and passes `GLR_CAPTURE_STATUS` to the recorder. The recorder
appends strict NDJSON status records containing the receipt `session_id`, `state` (`healthy`,
`degraded`, `stopped`, `failed`, or `completed`), frame/step counters, the latest frame timestamp,
dropped frames, and an optional reason. Required capture succeeds only after a healthy handshake,
a fresh heartbeat, a `completed` terminal record, the configured minimums, and a valid
`glr.capture.v1` manifest. Optional capture remains non-blocking, but its lifecycle is recorded as
a structured `capture.lifecycle` run event and included in `--json` output. The receipt and status
file are retained as run artifacts; video/index/manifest artifacts are registered only after all
gates pass.

Use `--no-capture` only when review/supervised evidence is intentionally unnecessary.

## Pursue a bounded goal

Create a strict `glr.agent-goal.v1` file:

```json
{
  "schema_version": "glr.agent-goal.v1",
  "goal_id": "goal.reach-destination",
  "objective": "Reach the requested destination and verify arrival.",
  "environment_family": "action-rpg",
  "success_criteria": [
    {
      "metric": "objective.arrived",
      "operator": "gte",
      "target": 1,
      "source": "runtime.telemetry"
    }
  ],
  "budget": {
    "max_trials": 8,
    "max_training_steps": 50000,
    "max_wall_seconds": 14400,
    "max_research_sources": 64
  },
  "allowed_research_media": ["official-rules", "text-guide", "video-tutorial"]
}
```

Run the control loop:

```powershell
glr --project . --json goal run --goal goals/reach-destination.json
```

The sequence is:

```text
goal + budgets
  -> project researcher -> cited research bundle
  -> project planner    -> bounded trial and reward terms
  -> project trainer    -> run metrics + capture + artifacts
  -> project evaluator  -> evidence bound to persisted metrics
  -> satisfied? yes: stop / no: refresh research and adjust next trial
```

Later research cycles receive previous research and evaluation paths. This supports looking up a
text guide after a difficult trial or revising a video-derived hypothesis. The total number of
unique sources, trials, planned steps, and wall time still cannot exceed the original goal budget.

Official rules, text guides, and video tutorials produce advisory findings. Only evidence whose
value, source, authority, and run ID match a metric persisted during the current trial can be
evaluated. Only `authoritative` evidence can satisfy a criterion.

## Query previous experience

```powershell
glr --project . --json runs list --limit 20
glr --project . --json runs show RUN_ID
glr --project . --json query entities --world forest --kind shrine --name shrine
glr --project . --json query routes --world forest --to-entity shrine.forest-1
glr --project . --json query edges --world forest --from-node node.spawn --at-ns 0
glr --project . --json query research --tag navigation --category strategy
glr --project . --json query research --verified-only
```

Run output uses `glr.cli-output.v1`. `runs show` returns events, metrics, artifact roles, hashes, and
metadata. SQLite keeps this query projection; transitions, tensors, videos, and model files remain
ordinary checksummed artifacts suitable for training pipelines.

Research lookup combines exact-environment, same-family, and generic findings. It excludes rejected
findings. Treat every returned route and guide as advisory; observe the current world and verify each
action postcondition.

## Transfer knowledge and reproduce playback

Export previously observed entities and routes:

```powershell
glr --project . --json knowledge export --output artifacts/spatial-knowledge.json
```

Import them in another checkout or fresh game instance with the same environment and protocol:

```powershell
glr --project . --json knowledge import --input artifacts/spatial-knowledge.json
```

The optional `glr.spatial-knowledge.v2` graph uses the same import command. Query directed edges
with `query edges`; pass `--status traversable` to obtain frontier candidates while blocked and stale
edges remain visible only when explicitly requested. Negative traversal evidence is retained as
advisory provenance and never grants action authority.

GLR rejects a different environment or protocol and downgrades imported observations to advisory.
For another game in the same genre, query family-scoped research instead; never transfer world
coordinates or assume identical action semantics.

Load a checksummed model bundle:

```powershell
glr --project . --json play --bundle artifacts/model-bundle
```

`play` requires exact environment and protocol compatibility and verifies every bundled byte before
starting the project player. Loading is not reproduction proof. Run the authoritative evaluator in
the new instance and compare the declared goal criteria.

## Inspect failures

The CLI stops when a required role or capture fails, a strict file is invalid, a budget is exceeded,
research uses disallowed media, a reward references an unknown finding, a spatial/model identity
differs, or evaluator evidence lacks a matching persisted metric.

Use the returned run ID with `runs show`. Do not relax identity, authority, provenance, or budget
checks to make a run green.
