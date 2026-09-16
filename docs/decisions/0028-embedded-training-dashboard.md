# ADR-0028: Embed a durable training dashboard in GLR

## Status

Accepted

## Context

Learners, recorders and agent roles already persist events, metrics and files.
Humans need a live view and a way to start the same operations as agents,
without installing another service or losing history when a page closes.

## Decision

Embed an Axum loopback server and local HTML/CSS/JavaScript assets in the Rust
CLI. `glr dashboard` serves controls and observation; `glr observe` serves only
observation. `train` and `goal run` start an observation server for their own
lifetime by default, with an explicit `--no-observe` opt-out. An independently
started Dashboard observes CLI runs through the same SQLite database.

The web transport depends on a read-only observation projection and a separate
command application service. Event/metric IDs and log byte offsets are resumable
cursors. SQLite is the authority; browser memory and exported view windows are
bounded projections. No telemetry is deleted automatically.

Command forms are derived from clap. Jobs parse argv using the real CLI parser
and spawn the current GLR executable with a fixed project, without a shell.
Training presets contain only GLR argv; they cannot contain executables or
expand runtime authority. Presets and job receipts are additional SQLite tables
compatible with existing schema versions 1 and 2. Stable request IDs prevent
duplicate submissions. A project file lock serializes Dashboard jobs across
processes; live run records also block another game execution. A stale running
record requires reconciliation, never automatic replay.

The HTTP server binds IPv4 loopback only, checks Host/Origin/Fetch Metadata,
requires same-origin JSON for control writes, bounds body sizes and concurrent storage
work, and serves a restrictive CSP. The interface has no shell or generic file
server. Managed log paths reject traversal and links. A local user who can run
GLR remains the trust boundary; the service is not a multi-user remote service.

Python `Telemetry` records learner updates, diagnostic scalars and explicit
route samples through TrainingStore. `execute_decision` emits its selection and
execution receipt when GLR environment bindings exist. Telemetry failures never
retry actions. Diagnostic metrics cannot satisfy authoritative reward terms.
Role stdout/stderr remain durable files and are mirrored to terminal stderr;
recorders must forward FFmpeg stderr to expose it.

Bridge producers use `glr.bridge-telemetry.v1` through authenticated localhost
HTTP, CLI JSON/JSONL ingestion, or the Python `BridgeTelemetry` client. HTTP
ingestion accepts non-browser requests with a project-scoped bearer token but
rejects foreign origins; read-only observation mode has no ingestion endpoint.
The shared ingestion service validates bounded batches and commits events,
metrics, latest status/state/progress projections and deduplication receipts in
one transaction. Retry identity is `(run_id, source, batch_id)`; changed content
under that identity is refused. New batches require an existing running run.
All ingress data remains diagnostic, and producer reports never prove current
connectivity or grant action authority. These tables are included in backups.

Backups use SQLite's online backup API, then copy and verify completed-run
files and terminal Dashboard job logs. Active run files and active job logs are
excluded because they cannot share an atomic snapshot with SQLite; the manifest
lists active run IDs as database-only. Restore verifies hashes and integrity,
stages a directory, then atomically promotes without replacing any destination.
Source project packages remain the existing explicit, contract-bound format.

## Consequences

- Agent and human operations share contracts and failure semantics.
- Closing the browser does not stop a job or discard data.
- A killed service may leave a child process and an unverified receipt;
  restarting never implicitly resumes or repeats that job.
- Backup history includes data through the database snapshot, not a promise of
  resumable live game state. Checkpoint compatibility remains separately checked.
- Frontend refresh is cursor polling (one second normally, faster while catching
  up), not a second event bus. Binary distribution needs no Node/Python web runtime.
- Live game and encoder acceptance are independent of synthetic Dashboard tests.
