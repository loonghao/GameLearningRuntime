# ADR-0030: Give each workbench server an instance identity

## Status

Accepted

## Context

The workbench servers were already per-project. `Dashboard::new(project)`
captures one `root`, `data_dir`, and `environment_id`; job receipts and
`dashboard/job.lock` live under that project's `data_dir`; a spawned job is
handed `GLR_TELEMETRY_URL` that points back at its own instance. Nothing about
that is shared between projects.

What was missing was every layer above it.

`--port` defaulted to the same literal `7432` for `dashboard` and `observe`, and
`observe::serve` reached a bare `TcpListener::bind`, so a second project's
dashboard on defaults failed with a raw `os error 10048` and no hint that the
flag existed. A few functions away, the automatic training sidecar already
degraded gracefully by falling back to an OS-assigned port — so the machinery
existed and only the human-facing entry point did not use it. That asymmetry was
itself a defect: the automatic server could land on an ephemeral port that
nothing recorded, leaving its URL alive only in one process's stderr.

`/api/v1/health` returned no identity. Two `glr dashboard` processes for the
same project on two ports returned byte-identical payloads, so given a port
nothing could say which project directory it served, and given a project nothing
could find its port. There was no command anywhere in the crate that enumerated
running servers: `dashboard jobs` reports job receipts, not servers.

Finally, nothing owned their lifetime. Servers outlived the work they were
started for, and on Windows a running image locks its executable, so `glr update`
failed on the replacement step. The workbench's own metadata was the missing
input to that diagnostic.

The reasoning already applied to external target processes — record an instance
lease, fail closed on ambiguity, never kill implicitly — had simply never been
applied to GLR's own long-lived servers.

## Decision

A running workbench server publishes an *instance lease*: a JSON record holding
`instance_id`, `environment_id`, `project_root`, `data_dir`, `data_dir_sha256`,
`executable`, `pid`, `port`, `url`, `read_only`, `version`, `started_at_ms`, and
`started_at`, under `glr.workbench-instance.v1`. The same object is returned by
`/api/v1/health` under `instance`, so a caller can prove which server answered
instead of inferring it from a port number.

The registry lives in **one per-user directory**, not in each project's
`<data_dir>/dashboard/`. Enumeration is the reason: `glr dashboard instances
--all` has to answer "what is running for this user" without being told a
project path, and a per-project location cannot do that without a global index
that would then need its own consistency story. `GLR_STATE_DIR` overrides the
location so tests and CI never touch a real operator profile. A lease is
published for `dashboard` and for the read-only `observe`/sidecar servers alike,
so an automatic observation URL is discoverable after the fact.

**A lease is a hint, not the truth.** Liveness is decided by asking the port's
health endpoint and comparing the returned `instance_id`. A lease that does not
answer is `stale`; a port that answers as a different instance is `foreign`.
Neither is ever listed as the caller's, and neither is ever stopped. This is what
makes port reuse safe: a recycled port cannot be mistaken for the original
server, and a server from a release that predates this feature is reported as
stale rather than impersonated.

**Ports become a candidate list.** Without `--port`, a server tries the preferred
7432, then a stable slot derived from the canonical `data_dir`, then any free
port. The derived slot keeps 7432 first so a single-project operator's existing
bookmark keeps working, and 256 slots make a birthday collision need roughly
nineteen concurrent projects — and even then the collision only advances to the
next candidate. An explicit `--port` is used **exactly as given and never
moved**; if it is taken, the command fails and names the alternatives. Silently
rebinding a number the caller is about to paste into a browser is worse than
failing.

**Same-project multi-instance stays allowed, and becomes distinguishable.** The
rule is stated rather than newly enforced: two servers for one project may run,
they share `environment_id` and `data_dir_sha256` but never an `instance_id` or
port, and `job.lock` keeps its existing semantics — only job submission is
serialized, and the loser still gets its receipt back. Stopping is where
ambiguity must fail closed: with more than one live server for the project,
`dashboard stop` refuses unless `--instance` or `--all` says which one.

**Stopping is a request, never a kill.** `dashboard stop` probes the target,
refuses a target that is not answering or no longer holds the instance, then
POSTs to the server's own `/api/v1/control/shutdown`, which sets the same
shutdown flag the owning command already uses, so the server retires through the
graceful path it would have taken anyway. Because that route is reachable only
in control mode, `glr observe` keeps its read-only guarantee.

When a binary replacement is blocked, the error reports which instances hold the
executable, by path, PID, and URL, together with what the update did manage to
apply. It is an upgrade diagnostic, not an auto-stop.

## Consequences

### Positive

- Every server is attributable from the CLI: project, port, version, mode, and
  age, whether or not the caller knows a project path.
- Two projects start on defaults without colliding, and each prints the address
  it actually bound — including a note when it had to move.
- Port reuse and pre-identity servers cannot be mistaken for the intended
  instance, so `stop` cannot be pointed at a stranger.
- A blocked self-replacement names its holder instead of reporting `os error 5`.
- Instance metadata is now available for the launcher contract to consume.

### Negative

- A lease directory is state outside every project. It is per-user, prunable, and
  never authoritative, but it is still a new location with new failure modes; an
  unwritable registry degrades to a warning so a server never fails to start
  because its lease could not be written.
- `--port` changed from a defaulted `u16` to an `Option<u16>`, so anything that
  read the old default from `--help` sees different text.

### Neutral

- The lease does not replace `job.lock`; job serialization is unchanged.
- Making ports depend on the data directory means the same project keeps its
  address across restarts, and a moved project gets a new one.
- Duplicate detection is unnecessary by construction: a second server for one
  project is a supported state, not a race to prevent.

## Alternatives Considered

**Derive the port and drop 7432 as the first candidate.** Rejected because it
changes the address of every existing single-project setup to fix a problem that
only appears at two or more projects.

**Always let the OS choose, and let the registry carry the URL.** Rejected
because a stable default address is what makes the dashboard bookmarkable; the
registry is a recovery path for when the default is taken, not a replacement for
having one.

**Write the lease into the existing `<data_dir>/dashboard/`.** Rejected because
it cannot satisfy enumeration without a global index, and the index would then
have to solve the same staleness problem the lease registry already solves.

**Treat a live same-project second server as an error.** Rejected because two
instances on two ports are genuinely useful — a read-only view beside a control
view, or an operator inspecting an old run — and forbidding it would break the
existing behavior for no safety gain that explicit stop targets do not already
provide.

**Kill the PID to stop a server.** Rejected because a PID is not an identity: it
can be recycled, and the operator cannot tell from the CLI whether the process
still is the server that answered. The server owns its own shutdown.

**Heartbeat the lease file and treat a missing heartbeat as dead.** Rejected in
favor of probing the port. A heartbeat is a second liveness notion that can
disagree with the socket, and the socket is the thing the caller actually cares
about.

## References

- `crates/glr-cli/src/instance.rs`
- `crates/glr-cli/src/observe.rs`
- `crates/glr-cli/src/dashboard.rs`
- `crates/glr-cli/src/update.rs`
- `crates/glr-cli/tests/dashboard_contract.rs`
- `docs/guides/dashboard.md`
- ADR-0015: Add an agent-first local control plane
- ADR-0016: Make the Rust CLI the distribution entrypoint
- ADR-0028: Embed a durable training dashboard
