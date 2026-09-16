# ADR-0032: Host an externally driven loop inside a run

Status: Accepted with the `glr host` implementation

Related: issue #145, ADR-0017, ADR-0020, ADR-0028, the `glr.bridge-telemetry.v1`
contract.

## Decision

`glr host -- <command>` creates a GLR-owned run, starts the telemetry write
endpoint, runs the caller-supplied command as a child, and finishes the run with
the child's exit code. The child is not a project role: GLR does not consult
`glr-project.json` roles, expand placeholders in the caller's command line, or
require a trainer to be configured.

The child receives `GLR_RUN_ID`, `GLR_RUN_DIR`, `GLR_STORE_PATH`, `GLR_CLI_PATH`
and the environment identity exactly as a role does, plus `GLR_TELEMETRY_URL` and
`GLR_TELEMETRY_TOKEN` when ingest is enabled. `--no-telemetry` omits the binding
and does not open the endpoint, so a caller can use the run without accepting
writes. `--timeout-seconds` bounds a child that would otherwise run forever.

The run kind is `hosted`, distinct from `training` and `goal`. The emitted receipt
reports only whether ingest was available, never the credential.

## Why this is not the trainer role

The reporter's only prior route was to make the external loop *be* the project
trainer. That works, but it conflates "the trainer" with "the whole campaign" and
forces every adapter to invent the same opt-in switch. `glr train` drives one
trial and reports trainer semantics (metrics, progress, no-data); a hosted
campaign loop is a different object. Giving it its own command keeps `train`'s
contract narrow and lets an adapter's policy loop own its own lifecycle.

`glr play` verifies and loads a bundle but does not drive a loop, so it is not an
alternative host.

## Credential handling

The ingest token is generated inside the CLI and handed to exactly one child
through its environment. It is never read from the ambient environment for this
path, never written to the run record, the run context, the hosted log, a report,
a preset or a package. An operator may still pin `GLR_TELEMETRY_TOKEN` before
start, in which case the same value is validated and used; that is an explicit
deployment choice, not the default.

Because roles and hosted children now clear inherited `GLR_*` variables (ADR-0031),
a hosted child's telemetry binding is a value the CLI decided for that child. It
cannot be inherited from, or observed by, a sibling.

## Consequences

Telemetry ingest becomes reachable without the dashboard: a loop hosted by `glr
host` can publish `glr.bridge-telemetry.v1` batches for as long as its run is
running. This is the supported channel issue #145 asked for.

The run must still be running when a batch arrives. Terminal runs remain
immutable and reject new telemetry; this decision adds no backfill path. A loop
that ran entirely outside a GLR run still cannot publish afterwards, and that
remains deliberate: rewriting a finished run's history from outside is not
evidence.

Hosting is not authorization. A hosted child is an ordinary process with the
caller's privileges; GLR does not sandbox it, does not grant it game-action
authority, and does not treat its exit code as proof of learning. Publishing
telemetry is diagnostic, consistent with the existing authority rules.

An adapter that wants planner, evaluator, promotion or checkpoint semantics still
needs the goal loop. `glr host` provides a run and an ingest channel, nothing more.
