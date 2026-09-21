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
`glr-cli`, `glr-adapter-builder`, and `glr-qa` Skills. The Rust CLI is standalone; install
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
run `glr update`. The updater requires HTTPS, downloads the exact Rust
target archive and `SHA256SUMS`, verifies the release manifest and digest, then
replaces the CLI, sibling Runtime Host, and project Skills.

```powershell
glr --json update
glr --json update --skills-dir .agents/skills
glr --json update --no-skills
```

It never runs an installer script or modifies game code, role dependencies,
virtual environments, models, datasets, or `glr-project.json`. SHA-256 verifies
same-release integrity; it is not publisher signature verification. Re-run
`--version`, `doctor`, and `update --check` after applying an update.
Checks use GitHub's public latest-release asset link and do not consume the
anonymous REST API quota. The selected version and exact target archive are
derived from the published `SHA256SUMS` before any update is applied.
The former `--yes` form remains accepted for compatibility.
Current binaries still download and verify the matching release archive when a
project Skills directory must be synchronized.

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
  "progress": {
    "signal": "day_counter",
    "window_steps": 256,
    "max_stalled_rounds": 3
  },
  "capture": {
    "argv": ["python", "tools/record_window.py", "{capture_video}", "{capture_index}"],
    "required": true,
    "stop": "stdin-q",
    "video_file": "capture.mp4",
    "index_file": "capture-index.jsonl",
    "codec": "h264",
    "frame_rate": 30,
    "width": 1920,
    "height": 1080
  },
  "hooks": {
    "default_timeout_seconds": 5,
    "subscriptions": [
      {
        "event": "train.failed",
        "action": "notify.message",
        "config": {"outbox": "hooks/messages.jsonl"}
      }
    ]
  }
}
```

The recorder is project-owned because only the project knows the reviewed game window and capture
API. Run `glr capture preset` and apply the default `training-balanced` output arguments:
1920x1080, 30 FPS CFR, H.264/libx264, CRF 18, fast, yuv420p, GOP 30, MP4 fast-start,
and no audio. Verify the finalized stream and frame index; manifest dimensions alone
do not configure the recorder.

Project roles receive `GLR_PROJECT_ROOT`, `GLR_BRIDGE_PATH`, `GLR_RUN_ID`, `GLR_RUN_DIR`,
`GLR_STORE_PATH`, environment identity, capture output paths, and goal-loop paths through
environment variables. They must stay bounded and validate the exact game target independently.

The CLI owns the `GLR_` environment namespace. Every child process it starts — project roles,
capture sessions, hosted children, `glr task` children, and dashboard jobs — clears inherited
`GLR_*` variables before the CLI publishes the values that child owns, so a child never observes a
stale or forged binding from an outer run or the ambient environment. Declare inputs through the
project manifest context instead of the ambient environment. Each child still receives only its own
bindings: a task child gets a task identity and no run identity.

A role invocation always describes the trial it serves. `glr goal run` issues `GLR_TRIAL_ID` and
`GLR_TRIAL_PATH` to its planner and trainer, and `glr train` issues the same pair for its single
implicit trial (`trial-1` under `trials/trial-1/`), so `{trial_id}` and `{trial_path}` expand under
either command. `GLR_RUN_DIR` stays the run-scoped output root `.glr/runs/<run-id>/`; a trial
directory is a child of it. Roles that own no trial, such as `runtime start` and `play`, receive no
trial identity. `glr train` reserves the trial plan path but never writes plan content: only the
goal loop plans.

Progress detection is opt-in. When `progress` is declared, a completed trainer must include this
shape in `trainer.result.json`:

```json
{
  "schema_version": "glr.trainer-result.v1",
  "status": "completed",
  "metrics": {},
  "progress": {
    "signal": "day_counter",
    "first_value": 12,
    "last_value": 12,
    "steps_since_change": 256,
    "accepted_steps": 256
  }
}
```

The CLI marks a trial `stalled` when the declared value is unchanged for the configured window,
and aborts the goal with exit code `76` after the configured consecutive-round threshold. The
verdict includes the signal, first/last values, and unchanged-step count. Without `progress`, no
signal is invented and no stall detection runs.

Startup readiness is opt-in and belongs to the runtime role:

```json
"runtime": {
  "argv": ["python", "tools/runtime.py", "{bridge_path}"],
  "readiness": {"timeout_seconds": 300, "poll_interval_seconds": 5}
}
```

`timeout_seconds` is required, must be positive, and cannot exceed one hour. `poll_interval_seconds`
and `file` default to `5` and `runtime-readiness.json`. An undeclared window keeps today's behavior:
one invocation, and the role exit code decides the run.

While `runtime start` holds a declared window open, it re-invokes the role and reads the
`glr.environment-readiness.v1` receipt that the role publishes at `GLR_READINESS_PATH`, receiving
`GLR_READINESS_ATTEMPT` so a re-probe can tell itself apart from a cold launch. Only an explicit
`not_ready` receipt is retried; a missing, unreadable, off-schema, or `unavailable` receipt is
terminal on first observation, so a crash is never retried. This is how a role that brings up a cold
GUI host reports "still starting" instead of being recorded as broken.

The window is durable evidence. Each invocation appends a `readiness.attempt` event and the verdict
appends one `readiness.outcome` event, both carrying the receipt verbatim; `runtime.start` also
returns the same summary under `readiness`. An exhausted window returns exit code `78` — the host was
still parking and the caller should retry — while every other verdict keeps the role exit code.

## Notify on lifecycle events

Hooks attach a named action to a named lifecycle event. The control plane publishes
`train.start`, `train.complete`, `train.failed`, `record.start`, `record.stop`, `goal.*`,
`runtime.*`, and `play.*`; any other lowercase dotted identifier is also a legal event name, so an
adapter can publish its own. A subscription selects an event or a whole namespace (`train.*`) and
narrows it by environment, run kind, stage, status, and exit-code range.

```bash
glr hooks list --format json
glr hooks emit --event train.failed --status failed --exit-code 7 --reason oom --dry-run
```

`hooks list` fails on an unknown action or a malformed configuration, so a mistake surfaces when the
manifest is edited. `hooks emit` publishes a synthetic event and exits `1` when an action failed or
timed out.

A hook is observability, never a dependency: an action that raises or exceeds its budget is reported
as a `failed` or `timeout` result, and a run whose hook configuration is unusable proceeds with no
subscriptions. A hook never changes the exit code of the run that published its event, and every
dispatch is recorded on the run as a `hook.dispatched` event with status, duration, and error per
action. See [the lifecycle hooks guide](lifecycle-hooks.md) for the full contract.

## Start the runtime and train

```powershell
glr --project . --json doctor
glr --project . --json runtime start
glr --project . --json task list
glr --project . --json train
```

Project-specific preparation and training workflows can be declared in a
strict `glr.toml` and executed with `glr task run`. Prefer VX-backed tasks for
Python so tool versions and the project virtual environment remain
reproducible. See [Extend GLR with declarative VX
tasks](declarative-tasks.md). A task exit is orchestration evidence, not live
game or learning acceptance.

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
  "minimum_steps": 1,
  "content_liveness": {
    "enabled": false,
    "required": false,
    "sample_every": 4,
    "max_bad_fraction": 0.5
  }
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

When enabled, content liveness samples sparse consecutive frame pairs and emits normalized numeric
`inter_frame_diff_mean`, `inter_frame_diff_max`, `luminance_mean`, and `luminance_std` values. The
manifest and `--json` output include the bounded sample window, heartbeat, `content_static` /
`content_blank` state, and reason. A required capture is rejected before artifact registration when
the configured bad-content fraction is exceeded; an optional capture remains usable but is marked
`degraded`. The feature is disabled by default and does not retain frames.

Use `--no-capture` only when review/supervised evidence is intentionally unnecessary.

## Close the visual loop numerically

`game_learning_runtime.visual_acceptance` provides a host-neutral visual-acceptance contract for
agents that cannot inspect pixels directly. `write_capture_atomically` writes through a temporary
file, flushes and fsyncs it, then atomically replaces the destination and returns the echoed
`request_id`, dimensions, byte count, and SHA-256. `CaptureJobRegistry` provides bounded
`pending`/`completed`/`failed` polling with strict request correlation.

`compute_visual_metrics` and `evaluate_visual_acceptance` return JSON-safe measurements for
coverage, bounding box/aspect, distinct colours, chroma, luminance, and optional silhouette IoU.
Use `require_visual_acceptance` for a required gate; optional checks can retain the report and its
failure reasons for later review. No host screenshot side channel or image retention is implied by
this contract.

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

### Bind a default goal

Bind the goal once so `goal run` no longer needs `--goal`, and bind the run context that makes it
executable so `train` no longer needs `--context`:

```powershell
glr --project . --context config/contexts/ranked.toml --json goal set --goal goals/reach-destination.json
glr --project . --json goal show
glr --project . --json goal list
glr --project . --json goal use goal.reach-destination
glr --project . --json train
glr --project . --json goal run
```

Bindings live in `<data_dir>/goal-binding.json` (`glr.goal-binding.v1`) and store project-relative
paths plus a SHA-256 of the goal file, so `goal show` reports `source_status`
(`unchanged` / `changed` / `missing`) and `context_status` (`unbound` / `bound` / `unresolved`).
An explicit `--goal` or `--context` always wins over the saved default. A binding only selects a
goal and a context: the goal stays structured metadata, is copied into the run directory as an
auditable receipt, and never shapes rewards or judges completion.

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
glr --project . --json knowledge export --output .glr/exports/knowledge/spatial-knowledge.json
```

Import them in another checkout or fresh game instance with the same environment and protocol:

```powershell
glr --project . --json knowledge import --input .glr/exports/knowledge/spatial-knowledge.json
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
glr --project . --json play --bundle .glr/exports/model-bundles/model-bundle
```

Checkpoint contract preflight and explicit migration:

```shell
glr --json checkpoint migrate \
  --manifest checkpoints/policy.manifest.json \
  --contract contracts/live-checkpoint-contract.json
glr --json checkpoint migrate \
  --manifest checkpoints/policy.manifest.json \
  --contract contracts/live-checkpoint-contract.json \
  --force
```

The first command is a dry-run for compatible changes (exit `3`); action,
observation, protocol, or schema changes fail closed (exit `4`). `--force`
creates backups, preserves checkpoint bytes, and verifies the rewritten
manifest.

`play` requires exact environment and protocol compatibility and verifies every bundled byte before
starting the project player. Loading is not reproduction proof. Run the authoritative evaluator in
the new instance and compare the declared goal criteria.

## Inspect failures

The CLI stops when a required role or capture fails, a strict file is invalid, a budget is exceeded,
research uses disallowed media, a reward references an unknown finding, a spatial/model identity
differs, or evaluator evidence lacks a matching persisted metric.

Use the returned run ID with `runs show`. Do not relax identity, authority, provenance, or budget
checks to make a run green.

A failed `train`, `goal run`, `runtime start`, or `play` also reports a machine-readable `failure`
object in its `--format json` envelope, so an agent can decide what to do next without parsing a
log:

```json
{
  "failure": {
    "stage": "trainer",
    "reason": "trainer command exited with code 7",
    "exit_code": 7
  }
}
```

`stage` names where the run broke (`train`, `game-launch`, `capture`, `trainer`, `goal`, ...). The
process exit code still carries the verdict.
