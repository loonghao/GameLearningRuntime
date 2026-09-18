# ADR-0035: Own repository automation by capability domain and register it

## Status

Accepted

## Context

Repository automation accumulated in a single top-level `scripts/` directory.
Sixteen executables — CI runners, packaging assemblers, provider smoke tests, a
release verifier, a docs renderer, and a demo — sat side by side with no
statement of ownership, no discoverable caller, and no way to tell whether a
file was still used. The directory name describes the *file kind* (`scripts`),
not the capability, so it cannot answer "where does a new tool go?" and it
offers no seam for review.

The problem is structural, not cosmetic. An ad-hoc script is invisible to
review: it has no owner, no declared purpose, no registered entry point, and no
test obligation. Over time this produces duplicated logic that should live in
the installable package, and one-off automation that rots silently.

## Decision

Replace `scripts/` with `tools/<domain>/`, where the domain names the capability
a tool serves: `ci`, `packaging`, `providers`, `release`, `docs`, `demo`,
`governance`. Move all existing executables into their owning domain.

Declare every tool in `tools/registry.toml`. Each entry carries:

- `id` — `<domain>.<kebab-name>`, unique.
- `path` — repository-relative, must start with `tools/<domain>/`.
- `domain` — must equal the first path segment.
- `purpose` — one sentence stating what the tool is for.
- `entrypoints` — the `just` recipe or workflow that calls it; `["manual"]` when
  only a human runs it.

Enforce the contract mechanically with
`tools/governance/check_tool_registry.py`, exposed as `just layout-check` and
included in `just check`. The checker is a pure reader — it never edits — and it
fails when:

- any `scripts/*.py` exists at the repository root level;
- a discovered tool has no registry entry;
- an entry points at a missing file;
- two entries share an `id` or a `path`;
- an entry's `domain` disagrees with its path;
- an entry has no `purpose`, no `entrypoints`, or a path outside `tools/`.

Document the decision rules, the anti-pattern list, and the five steps for
adding a tool in [the repository layout guide](../guides/repository-layout.md).
The anti-pattern list is normative review criteria, not advice.

Tools stay thin: they parse arguments, call into `game_learning_runtime`, print
a result, and return an exit code. Reusable logic belongs in the installable
package, per the Python package baseline in `CONTRIBUTING.md`.

## Consequences

- "Where does this go?" has a written answer, and a new domain is a reviewable
  decision rather than an accident.
- An unregistered tool cannot merge, because the gate runs in `just check`.
- A caller can be found for every tool, so a tool nobody calls is visible.
- Existing references had to move with the files: three workflows, four
  documents, and four test paths. The gate now prevents that drift from
  recurring silently.
- A tool is still only as good as its tests. Registration guarantees ownership
  and discoverability, not correctness.

## Rejected alternatives

- Keep `scripts/` and add a README listing the files: rejected because a
  hand-maintained list drifts immediately and cannot fail a build.
- Group by language or by invoking team: rejected because both change
  independently of what the tool does, which is what a reader needs to know.
- One `tools/` directory without domains: rejected because it recreates the
  dumping ground one level down.
- Enforce by reviewer discipline only: rejected because it does not survive a
  busy week; the value here is that the check is mechanical.

## Related

- ADR-0016 makes the Rust CLI the distribution entrypoint; this ADR governs
  repository automation, not shipped binaries.
- [Repository layout guide](../guides/repository-layout.md).
