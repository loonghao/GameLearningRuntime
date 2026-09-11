# ADR-0024: Bind invocation-scoped run contexts

## Status

Accepted

## Context

Projects need to select a coherent set of training, reward, roster, or policy
inputs for one invocation. Encoding one product concept such as a season in the
core CLI would duplicate project workflow policy and make every new domain need
another command family.

## Decision

Add the global `--context PATH` option for commands that launch or inspect
project roles. `PATH` names a strict, project-relative `glr.run-context.v1`
TOML file. It declares the exact environment and protocol, generic labels, and
owned JSON or TOML inputs with expected schema versions.

Before execution GLR rejects unknown fields, identity mismatches, links, path
escapes, duplicate owners or paths, unsupported input formats, and bounded-size
violations. It freezes SHA-256 and size identities for the source and every
input, verifies them before each role spawn, sends the compact receipt through
`GLR_RUN_CONTEXT` and `GLR_RUN_CONTEXT_SHA256`, and records the receipt as a
run artifact and event. Python roles can verify the same receipt with
`load_inherited_run_context()`.

Project workflow discovery and setup stay in strict `glr.toml` tasks, normally
with `runner = "vx"`. A season, league, ruleset, experiment, or campaign is a
label or project task policy, not a GLR core abstraction.

## Consequences

- One reusable contract covers product-specific configuration groupings.
- Every role sees the same immutable, auditable input identity.
- Context selection does not claim readiness, learning quality, or live-game
  success; those require authoritative runtime and evaluator evidence.
- Commands that do not launch or inspect roles reject `--context`.

## Rejected alternatives

- Add `glr season` and global `--season`/`--ruleset`: rejected as product policy.
- Load arbitrary environment variables or executable plugins: rejected because
  they weaken validation, portability, and auditability.
- Copy VX environment management into GLR: rejected because VX already owns
  tool and Python environment resolution.
