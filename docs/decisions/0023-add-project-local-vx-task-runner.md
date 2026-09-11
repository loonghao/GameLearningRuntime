# ADR-0023: Add a project-local VX task runner

## Status

Accepted

## Context

GLR exposes stable, typed commands for runtime control, training, evidence, and
playback. Game projects also need repeatable project-specific workflows such as
dataset preparation, one training season, evaluation, or packaging. Adding each
workflow to the Rust command enum would couple GLR releases to project policy.
An unrestricted shell field would instead weaken the existing fixed-argument
execution contract.

Training commonly needs Python and native tools with reproducible versions.
The repository already uses VX to resolve those runtimes and the project-owned
Python environment.

## Decision

Add a strict `glr.tasks.v1` file named `glr.toml` and the commands `glr task
list`, `glr task show`, and `glr task run`. Tasks declare fixed argument arrays,
typed named parameters, bounded timeouts, project-relative working directories,
and an acyclic dependency graph.

Tasks may use `runner = "vx"`. GLR then executes `vx` with the declared `argv`,
so VX owns runtime/version/environment resolution while GLR owns validation,
ordering, timeout, logging, and the `glr.task-result.v1` receipt. Python training
tasks should normally use `runner = "vx"` with `argv = ["uv", "run", ...]`.

Arguments are passed directly to the child process without a shell. Parameter
and GLR path placeholders must occupy a complete argument. Task configuration,
working directories, and path parameters remain under the project root. Task
logs and receipts are written below `.glr/tasks/<execution-id>/`.

Project tasks do not become top-level GLR commands. `glr task run season` is a
project workflow, while a future `glr season` would be a versioned core contract.
Process success remains distinct from authoritative gameplay or learning
evidence.

## Consequences

### Positive

- Projects can extend the CLI without rebuilding GLR.
- VX supplies reproducible Python and tool environments across local and CI use.
- Task inputs, dependencies, duration, logs, and exit status are inspectable.
- Core GLR commands and project workflows remain distinguishable.

### Negative

- VX-backed tasks require the `vx` executable to be available.
- A configured executable still has the current user's OS permissions; fixed
  argv is not an operating-system sandbox.
- Dynamic parameters use explicit `--set NAME=VALUE` rather than generated Clap
  flags.

### Neutral

- `glr-project.toml` or legacy `glr-project.json` remains the runtime contract;
  `glr.toml` is only the project task registry.
- Task receipts do not replace run-store metrics, evaluator evidence, capture
  manifests, or live-game acceptance.

## Alternatives Considered

**Add every workflow as a Rust subcommand.** Rejected because project policy
would require coordinated GLR releases and would grow the core command surface.

**Execute shell command strings.** Rejected because quoting differs across
platforms and shell composition creates avoidable injection and audit risks.

**Implement Python environment management inside GLR.** Rejected because VX and
uv already own tool and Python environment resolution.

**Load native or Python CLI plugins.** Deferred because an ABI/plugin lifecycle
is unnecessary for declarative local workflows and materially increases trust
and compatibility costs.

## References

- ADR-0015: Add an agent-first local control plane
- ADR-0016: Make the Rust CLI the distribution entrypoint
- `docs/guides/declarative-tasks.md`
