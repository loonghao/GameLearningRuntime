# ADR-0022: Freeze explicit season and ruleset context before roles

## Status

Accepted

## Context

An implicit active profile can select a different ruleset in the CLI, adapter,
trainer or recorder. Reusing data or weights after such drift is not a reproducible
training result. New seasons also need a non-executing initialization boundary.

## Decision

Add an optional first-class project `seasons.config` reference to a strict
`glr.seasons.v1` TOML catalog. Entries bind an explicit pair to a root-relative
`glr.season.v1` declaration. The declaration binds project environment/protocol,
pending/ready status and optional extension config references. These identifiers
are opaque; game semantics and input interpretation remain project-owned.

Add `season list`, `season show` and `season init`. Initialization creates pending
TOML only and registers it without overriding existing declarations or executing
hooks. It does not edit the root manifest. Reject unknown selection and schema
fields, ambiguous identity, path escapes and linked inputs under bounded limits.

Global `--season` and `--ruleset` select one immutable in-memory context. Configured
projects require explicit selection before roles. Train/goal/play require ready;
runtime startup permits pending so the project can probe the actual ruleset.
Doctor splits installation and training-config readiness and never claims live
runtime verification. Projects without a catalog retain existing behavior.

Freeze project, catalog, declaration and mounted-input byte digests in the
portable `glr.season-context.v1` wire object. Hash canonical JSON and pass the same
context to every role; clear inherited values for unselected legacy projects.
Revalidate frozen input bytes before every spawn and persist a checksummed context
artifact plus event before the first role. Python project loading validates the
inherited envelope rather than ignoring the CLI selection.

The detailed wire schema and operating contract live in the packaged
[season reference](../../.agents/skills/glr-cli/references/seasons.md).

## Consequences

- Version/ruleset selection is observable in run evidence and cannot silently
  change between configured roles.
- Nested input files must be mounted explicitly; GLR does not recursively discover
  dependencies or import extension code.
- Changing any frozen byte, including catalog edits and line endings, requires a
  new context. Machine-specific overrides stay outside portable configuration.
- Ready configuration, process completion and stored hashes are not actual-runtime
  identity, successful action readback, model compatibility or learning quality.
- The role consumer still owns parsing verified bytes and enforcing live guards.
  This is not a filesystem sandbox, a cluster protocol or a model migration engine.
