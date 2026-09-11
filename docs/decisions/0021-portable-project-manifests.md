# ADR-0021: Resolve portable projects from one manifest

## Status

Accepted

## Context

Role scripts can accidentally depend on a particular adapter directory, current
working directory, or copied virtual environment. A relocated checkout then
selects the wrong configuration or cannot run. Knowledge-file presence also
does not reveal whether a decision actually performed a relevant lookup.

## Decision

New scaffolds use `glr-project.toml` with the existing `glr.project.v1` contract.
Legacy JSON remains readable. Both loaders search upward for the nearest
manifest, reject simultaneous JSON and TOML manifests at the same level, and
reject symlinked manifests. Optional TOML roles are omitted rather than null.

Expose the selected manifest to project roles through `GLR_PROJECT_MANIFEST`
and the whole-argument `{project_manifest}` placeholder. Relative configuration
paths are anchored to the manifest parent. The Python resolver is reusable by
generated scripts without relying on a fixed directory depth.

An optional `extensions.<namespace>.config` reference mounts one existing,
project-relative, non-symlink configuration file. GLR validates the reference;
the extension owns its schema and behavior. The reference does not load code or
expand the action vocabulary. No unrestricted local override merge is added.

Provide a path-only helper for an existing project-owned game directory or an
explicit absolute installation directory. Keep machine-specific overrides,
virtual environments, private traces, and licensed payloads out of shared
configuration and published artifacts. Preserve canonical Windows path identity;
legacy launchers adapt path spelling only at their reviewed boundary.

Knowledge injection reports trigger, match, selected, and first-filter rejection
counts alongside its existing query fingerprint. A valid miss, no invocation,
and a rejected source remain distinct. These counters neither change ranking
nor establish action authority or successful learning.

## Consequences

- A project can be relocated and invoked from nested directories without
  changing role source code.
- Dependency environments are recreated from project locks, not copied from
  an adapter subtree. Generated training output uses the run directory supplied
  by GLR.
- Older installed clients need a compatible release before consuming TOML or
  extension references. Source tests do not upgrade an installed client.
- Configuration checks and synthetic reproduction remain separate from
  authorized live-runtime and whole-game acceptance.
