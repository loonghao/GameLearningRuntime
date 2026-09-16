# Training dashboard and durable history

Bridge producers can use the `BridgeTelemetry` Python SDK, authenticated local
HTTP batches, or CLI JSON/JSONL ingestion. See the [bridge integration guide](bridge-telemetry.md)
and [versioned JSON Schema](../schemas/bridge-telemetry.schema.json) for source provenance,
idempotent receipts, state/progress panels, and shared Agent queries.

```powershell
glr --project . dashboard
```

Open the printed localhost URL (port 7432 by default; `--port 0` selects a free
port). Axum and all page assets are embedded in `glr`; no web runtime, build step,
CDN, or cloud account is required. Start a preset, inspect the resulting run,
and use **GLR operations** for reports, source packaging/import, playback,
knowledge queries, plugins and other CLI operations. Forms come from the same
clap definitions as agent commands. Paths refer to the machine running GLR.
The dashboard does not launch a browser automatically.

The interface uses React, TypeScript, shadcn/ui (Radix primitives), and Tailwind.
Vite produces hashed JavaScript/CSS assets that Rust embeds alongside Axum.
All assets are served from the same loopback origin under the existing CSP;
no inline scripts or external asset hosts are needed.

`glr train` and `glr goal run` also print a read-only observation URL by default.
Their server stops when the command exits; `--no-observe` disables it. If 7432 is
occupied, this automatic view uses a free port. Keep `glr dashboard` running
independently for persistent controls and observation across multiple CLI runs.
`glr observe` is an explicitly read-only alternative; `--archive DIRECTORY`
opens a verified backup with the current project's environment scope.

## Train from the browser or CLI

The built-in **Default training** preset uses the project's existing trainer
and capture configuration. The Dashboard does not install a game, choose a
model, or invent trainer settings. Parameterized project tasks can define
different algorithms, budgets or configurations using `task run ... --set`.

Select `train`, `goal run`, or `task run` in GLR operations, fill in parameters,
then save a named preset. Presets can be exported/imported as small JSON files:

```json
{
  "schema_version": "glr.training-preset.v1",
  "id": "train.experiment-a",
  "title": "Experiment A",
  "description": "Project-owned training configuration",
  "argv": ["train"]
}
```

The equivalent agent commands are:

```powershell
glr --json dashboard catalog
glr --json dashboard presets
glr --json dashboard save-preset --file experiment-a.json
glr --json dashboard run train.experiment-a
glr --json dashboard jobs
glr --json dashboard jobs --before JOB_ID
glr --json dashboard job-log JOB_ID --stream stderr
```

Jobs retain requested/expanded argv, start/end times, PID, exit status, and
stdout/stderr under `.glr/dashboard/jobs/<job-id>/`. Runs contain the originating
`dashboard_job_id`. Double-clicks/network retries with the same request ID return
the existing receipt. Different arguments require a new ID. Dashboard jobs are
serialized; an active project run blocks another training/runtime/playback/task
execution. A service crash does not authorize repeating a task. Inspect a
nonterminal receipt and reconcile the underlying run/process before restarting.

## Observe decisions, updates, routes and recording

Run history, searchable events, per-step evidence, selectable scalar curves and
route/episode groups all read persisted SQLite data. Route points and the scrubber
select recorded steps. XY/XZ/YZ planes use runtime-supplied coordinates. Select
`capture.log` for recorder/FFmpeg output, including carriage-return progress.
An empty panel means the producer has not emitted those records.

The optional Python SDK supplies a learner-neutral emitter:

```python
from game_learning_runtime import Telemetry
from game_learning_runtime.decisions import execute_decision

telemetry = Telemetry.from_env()  # GLR_RUN_ID + GLR_STORE_PATH, no new run
if telemetry is not None:
    telemetry.learning_update(step_id=40, metrics={"loss": 0.25, "entropy": 0.7})
    telemetry.route_sample([10.0, 4.0, 2.0], step_id=40, episode_id="episode-1")

# With a real project Decision and executor, selection and receipt are emitted
# automatically when the GLR run environment is present:
receipt = execute_decision(decision, executor, step_id=40)
```

Other languages can continue using the existing event/metric store contracts.
Recommended event names are `agent.decision`, `agent.execution`,
`agent.execution_failed`, `learning.update`, and `navigation.route_sample`.
Payloads remain learner-owned. Metrics emitted through Telemetry carry diagnostic
authority; they are not runtime reward or evaluation evidence.

Structured events print to child stderr after persistence. All managed role
output is mirrored to GLR stderr, leaving `--json` stdout intact. Set
`GLR_LOG_STDERR=0` to suppress terminal mirroring or `GLR_TELEMETRY_STDERR=0` to
suppress the SDK's event printing; persisted data is unaffected. Python roles
receive `PYTHONUNBUFFERED=1`. Recorders that privately pipe FFmpeg output must
forward its stderr (or use `-progress pipe:2`) into `capture.log`; GLR cannot
recover output a recorder discards. Logs should contain diagnostics, not secrets.

## Trace and export

```powershell
glr --json runs trace RUN_ID --events-after 1000 --metrics-after 500 --limit 250
glr --json runs log RUN_ID --path capture.log --offset 0
glr --json report build RUN_ID
```

`runs trace` returns independent next cursors for events and metrics. Keep
requesting pages while `more` is true. It does not stop at the first 1,000 records.
`runs log` returns `next_offset`, `more`, and `reset` for truncation. Omit the
offset to start with the last 64 KiB. Log text is UTF-8 with replacement for
invalid bytes; raw files remain exact. Offline reports now page through history
and explicitly refuse above 100,000 events or metrics rather than silently
omitting later records.

The browser keeps 5,000 events/metrics and 128 KiB of log text. Its **Export
window** button exports that window with an explicit scope/partial marker, not a
complete backup. SQLite and raw logs are not pruned. Use cursors or backup for
full history. Oversized event/metric metadata (>16 KiB) is marked
`observation_truncated` in the live projection; the original stays in SQLite.

## Backup, verify and restore

```powershell
glr --json backup create --output ../backups/training-2026-09-16
glr --json backup verify ../backups/training-2026-09-16
glr --json backup restore ../backups/training-2026-09-16 --output ../restored-history
glr observe --archive ../restored-history
glr --json runs trace RUN_ID --archive ../restored-history
```

A backup is a portable directory with `backup-manifest.json`, `runs.sqlite3`,
completed run directories (including their logs, capture and registered
checkpoints/reports), and completed Dashboard job logs. Presets/job receipts
are part of SQLite. Files outside run directories are not implicitly swept in.
Use source packages for project code/configuration, and explicitly register run
artifacts that must be archived as evidence.

The SQLite online backup includes committed WAL data while training continues.
Active runs are **database-only** and named in `active_runs_database_only`; their
changing recordings/checkpoints are excluded. Active job logs are also excluded.
For a complete run archive, create a new backup after the run finishes. Every
copied file gets byte size/SHA-256; registered evidence must still match its
original digest. Restore verifies the manifest, all files and SQLite integrity,
uses a new directory, and never overwrites live storage. It does not resume a
game or validate a model's compatibility. Backups and retention are explicit:
there is no scheduled backup, automatic deletion, or automatic recovery.

**Package project / Import project** reuse `glr package export/import` and their
explicit file selection, expected environment and contract digest checks.
Export preset JSON into that selection to transfer presets with source code.
Import does not execute roles or install dependencies. **Back up history /
Restore history** operate on recorded data using the contracts above.

## Media and evidence workspace

The run view includes MP4/WebM playback, an image gallery, Markdown notes, and
plain-text logs. Decision moments open the inspector. A registered capture manifest
enables episode/step navigation; ambiguous steps never seek. Without a manifest,
playback is review-only. Playback depends on browser codecs; no transcoding occurs.

Only registered files beneath `.glr/runs/RUN_ID` are served. Producers write files
there, then call `TrainingStore.register_artifact` with a run-relative `path`, actual
file `source`, `role`, and `media_type`. Registration records hashes and metadata;
it does not copy files. Existing archives include these run files.

**Preview local media** opens a browser-local video/image without uploading,
persisting, or associating it with the selected run. Markdown resolves relative
links only to loaded, registered artifacts. External resources and raw HTML are
not rendered; HTML/SVG are download-only.

Catalog pages contain 100 artifacts at most; text previews stop at 256 KiB.
Capture manifests are capped at 8 MiB and mappings at 25,000 frames. Manifest
checksums, run/environment binding, monotonic frames, and video registry identity
are checked. Browsing does not rehash the video: `not_reverified` is explicit;
archive/report verification remains separate. Media streams support single byte
ranges beyond the JSON response limit, using `no-store` without ETag validators.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/v1/media?run=ID&after=PATH` | Registered artifact catalog and cursor |
| GET/HEAD | `/api/v1/media/file?run=ID&path=PATH` | Original bytes and Range playback |
| GET | `/api/v1/media/document?run=ID&path=PATH` | Bounded text preview |
| GET | `/api/v1/media/frames?run=ID&manifest=PATH&video=PATH` | Capture-step mapping |

## Local API

JSON responses are versioned; JSON data reads support ETag/If-None-Match. No CORS is
enabled. Reads reject non-loopback Host and foreign Origin/Fetch Metadata;
POST requires exact local Origin and `Content-Type: application/json`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/v1/health` | Version, environment, read-only mode |
| GET | `/api/v1/runs?before=RUN_ID` | 100 runs per page, `next_before` cursor |
| GET | `/api/v1/snapshot?run=ID&events_after=-1&metrics_after=0&limit=250` | Events, metrics, run status, managed log names |
| GET | `/api/v1/log?run=ID&path=capture.log&offset=0` | Bounded log byte page |
| GET | `/api/v1/control/catalog` | CLI-derived operation forms |
| GET/POST | `/api/v1/control/presets` | Read or save a preset |
| GET/POST | `/api/v1/control/jobs` | 100 jobs per page (`?before=ID`) or submit one |
| GET | `/api/v1/control/job-log?id=ID&stream=stdout` | Last 64 KiB of job output |

Submit `{"request_id":"request-unique-id","preset":"train.default"}` or
`{"request_id":"request-unique-id","argv":["doctor"]}` to jobs. These endpoints
exist only in Dashboard mode. The project is fixed at startup. Nested dashboard
and observer jobs are rejected. Use agent CLI commands for local scripting;
the web service is not an authenticated remote multi-user API.

## Build the embedded frontend

From a source checkout, build the frontend before any Cargo command that compiles
`glr-cli` (including `just check`, `just build`, and Rust tests):

```powershell
vx just dashboard-build dashboard-check
vx cargo build --release --package glr-cli --locked
```

`vx.toml` pins Node.js; `npm ci` uses the checked-in lockfile. The frontend build
runs TypeScript checking and writes `dist/source.sha256` over its source and
configuration, normalizing CRLF so Windows and Linux agree. Cargo verifies this
fingerprint and fails with a rebuild instruction if assets are missing or stale.
Rebuild after editing frontend files. Generated `dist/` is not committed.

CI builds and tests the frontend in the **React dashboard** job, then passes the
`dashboard-ui` artifact to Rust and distribution jobs. Release CI builds from the
immutable release tag and supplies that same artifact to Linux, Windows, and macOS
compilations. The final executable includes the HTML, JavaScript, and CSS; it does
not read a frontend directory at runtime. `/api/v1/health` exposes
`dashboard_source_sha256` to identify the embedded source revision.

Component tests run in jsdom and exercise controls and observation state. They do
not replace a real-browser visual acceptance check.
