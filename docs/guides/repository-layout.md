# Repository layout and tool ownership

This guide is the "what goes where" contract for this repository. It exists so
that neither a human nor an agent has to guess where a new file belongs, and so
that reviewers have a mechanical answer when a change lands in the wrong place.

Three rules govern the layout:

1. **Capability domains own directories.** A directory is named after the
   capability it delivers, not after the kind of file it holds.
2. **Every automation entry point is registered.** A script that is not in
   `tools/registry.toml` does not exist as far as the repository is concerned.
3. **The gate is mechanical, not social.** `just layout-check` fails the build
   when a tool is unregistered, orphaned, or parked in the wrong domain.

## Top-level map

| Path | Owns | Admits |
| --- | --- | --- |
| `src/game_learning_runtime/` | The installable Python runtime package | Runtime modules exported by the wheel. Nothing here may mutate `sys.path`. |
| `crates/` | The Rust workspace (`glr-cli`, `glr-host`) | Cargo crates only. |
| `sdk/` | Engine provider SDKs (C#, C++) | Provider-side sources and their samples. |
| `protocol/` | Versioned wire contracts | Schema definitions shared across languages. |
| `dashboard-ui/` | The React training dashboard | Frontend sources; its own `scripts/` is frontend-owned. |
| `plugins/` | Distributable agent plugin payload | Generated skill payloads, verified by `package_agent_plugin.py`. |
| `.agents/skills/` | Authoritative agent skills | Skill sources; a skill's internal `scripts/` is skill-owned. |
| `tests/`, `tests_optional/` | Test suites | Tests for the installed package. `tests_optional/` holds extra-gated lanes. |
| `benchmarks/` | Performance harness | Benchmark code and its reports. |
| `docs/` | Documentation | `guides/`, `decisions/`, `architecture/`, `runbooks/`, `schemas/`, `planning/`. |
| `tools/<domain>/` | Repository automation, by capability | Executable entry points and only those. |
| `.github/workflows/` | CI definitions | Workflow YAML; `actionlint` lints them. |

There is deliberately **no top-level `scripts/` directory**. It is the one
layout rule with a mechanical enforcement hook: `tools/governance/check_tool_registry.py`
fails when any `scripts/*.py` reappears.

## `tools/` domains

`tools/` is grouped by the capability a tool serves, not by the language it is
written in or the team that happens to run it.

| Domain | Contents | Question it answers |
| --- | --- | --- |
| `tools/ci/` | Quality gates and CI entry points | "How do I run the checks?" |
| `tools/packaging/` | Distribution assembly and verification | "How do I build the payload?" |
| `tools/providers/` | Engine provider smoke tests | "Does this provider still bind?" |
| `tools/release/` | Release-time verification | "Is this tag releasable?" |
| `tools/docs/` | Documentation asset generation | "How do I regenerate this artifact?" |
| `tools/demo/` | Self-contained demonstrations | "How do I show this without a game?" |
| `tools/governance/` | Repository self-checks | "Does this repo still follow its own rules?" |

## Decision rules: where does my new thing go?

Walk these in order and stop at the first match.

1. **Is it imported by the wheel?** → `src/game_learning_runtime/`. If it is not
   imported by the package it must not live there.
2. **Is it reusable logic that a script would otherwise inline?** → the package
   first, then a thin tool that calls it. Never grow a tool into a logic home.
3. **Does it run in CI, or is it a wrapper around other checks?** → `tools/ci/`.
4. **Does it build or verify a distributable payload?** → `tools/packaging/`.
5. **Does it exercise an engine or provider SDK?** → `tools/providers/`.
6. **Does it only run at release time against a tag?** → `tools/release/`.
7. **Does it produce a documentation asset?** → `tools/docs/`.
8. **Is it a demonstration or sample run?** → `tools/demo/`.
9. **Does it check the repository against its own conventions?** → `tools/governance/`.
10. **None of the above?** → you are probably about to add a one-off. Reconsider:
    a one-off that nobody registers is exactly what this layout forbids. Add a
    new domain only when the capability is genuinely new, and register it in the
    same commit.

## Anti-patterns

Reviewers should reject these outright.

| Anti-pattern | Why it fails |
| --- | --- |
| `scripts/foo.py` at any depth of the repository root | There is no generic dumping ground. Pick a domain. |
| `tools/misc/`, `tools/utils/`, `tools/helpers/` | A domain named after "other stuff" is not a capability. |
| An unregistered `*.py` under `tools/` | `just layout-check` fails; the tool has no owner and no discoverable caller. |
| A registry entry whose file no longer exists | Stale contract; the gate fails. |
| A tool that duplicates business logic instead of importing the package | Violates the Python package baseline in `CONTRIBUTING.md`. |
| `sys.path` mutation, `PYTHONPATH`, or `parents[...]` guessing to find the repo root | Install the package; derive the root once, explicitly. |
| A one-off script committed next to the code it patches | Ad-hoc automation is invisible to review and rots silently. |
| A shell snippet pasted into a workflow that should be a tested tool | Untested automation cannot fail loudly. |
| Renaming a domain directory without updating `tools/registry.toml` | The gate fails on domain mismatch. |

## Adding a tool: the standard five steps

1. **Choose the owning domain** using the decision rules above. If no domain
   fits, propose a new one in the PR description and say why.
2. **Create the file** at `tools/<domain>/<name>.py`. Keep it thin: parse
   arguments, call into `game_learning_runtime`, print a result, return an exit
   code. Reusable logic belongs in the package.
3. **Register it** in `tools/registry.toml` in the same commit:

   ```toml
   [[entries]]
   id = "docs.render-readme-demo"
   path = "tools/docs/render_readme_demo.py"
   domain = "docs"
   purpose = "Render the README GIF from a real synthetic GLR collection run."
   entrypoints = ["manual"]
   ```

   `id` is `<domain>.<kebab-name>` and must be unique. `domain` must equal the
   first path segment. `purpose` is one sentence. `entrypoints` lists the `just`
   recipe or workflow that calls the tool; use `["manual"]` when only a human
   runs it.
4. **Wire an entry point.** Prefer a `just` recipe so the tool is discoverable
   and reproducible; reference it from a workflow only when CI owns the call.
5. **Run the gate.**

   ```powershell
   vx just layout-check
   ```

   A clean run prints `layout-check ok: <n> registered tools`.

## The gate itself

`tools/governance/check_tool_registry.py` is a pure reader: it walks `tools/`,
parses the registry, and reports problems. It never edits. It fails when:

- any `scripts/*.py` exists at any depth (skill payloads under `plugins/` and
  `.agents/` are products, not automation, and are exempt), or a Python file
  sits outside an allowed source root such as `src/`, `tests/`, `tools/`;
- a discovered tool has no registry entry — this covers `.sh` and `.ps1` helpers,
  not just Python;
- an entry points at a missing file;
- two entries share an `id` or a `path`;
- an entry's `domain` is not one of the declared domains, or disagrees with its
  path;
- an entry's `id` is not `<domain>.<kebab-name>` with the domain as prefix;
- an entry has an empty or non-string `purpose`, or a path outside `tools/`;
- `entrypoints` is missing, empty, not a list of strings, or names a `just`
  recipe that does not exist, a workflow file that is missing, or a workflow that
  never invokes the tool.

Entry point verification is what keeps the registry honest: an entry claiming
`just build` fails once that recipe is renamed, instead of quietly pointing at
nothing. `manual` remains valid for a tool only a human runs.

It runs as part of `just check` and as its own CI job, so an unregistered or
mis-declared tool cannot merge.

## Related

- [Agent onboarding](agent-onboarding.md) — the one-command bootstrap chain.
- [Supervision and watchdog](supervision-watchdog.md) — scheduled runs.
- [Anti-fork gate](fork-gate.md) — drift detection for derived checkouts.
- [Pin one entry point per project](entry-point.md) — launch attestation and
  single-owner invariants.
- ADR-0035, ADR-0036, ADR-0037, ADR-0042 in [the decision
  index](../decisions/README.md).
