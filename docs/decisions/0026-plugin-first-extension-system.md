# ADR-0026: Add a plugin-first extension system

## Status

Accepted

## Context

GLR already has strict ports for game adapters, learners, the Runtime Host, and
the optional DeepSeek Harness provider.  Those ports are reusable, but a
project still has to hand-wire every learner, recorder, evaluator, model
provider, and harness integration.  We want the composability of DeepSeek
Harness (DSH) plugins without turning GLR configuration into arbitrary code
execution.

The extension surface must also preserve the learner-neutral boundary.  A
TorchRL learner and a Sample Factory learner should be selectable implementations,
not assumptions embedded in the environment or bridge contract.

## Decision

Introduce two versioned, declarative contracts:

- `glr.plugin.v1` describes one local bundle: identity, semantic version, kind,
  entrypoint metadata, capabilities, dependencies, supported platforms,
  isolation hint, declared permissions, and optional file digests.
- `glr.profile.v1` describes an ordered composition of plugin references.  A
  profile chooses version ranges, explicitly grants a subset of the plugin's
  declared permissions, and supplies bounded JSON configuration.

The Rust `glr` executable and the Python SDK expose the same control-plane
operations: inspect, install, list, health, profile list/show/resolve,
profile enable/disable, and remove.  Local directory bundles are the first
acquisition source.  Installation is atomic and re-inspected after copying.

Inspection and installation are deliberately no-exec operations.  They do not
import an entrypoint, evaluate a hook, invoke a shell, access a network, or
start a process.  A future host-specific runner may consume a resolved profile
only after applying trust, authorization, and runtime-specific policy.

## Security and compatibility rules

- Unknown manifest/profile fields, malformed semantic versions, unsafe (ASCII-only)
  paths,
  symlinks/reparse points, duplicate paths, oversized files, and digest
  mismatches fail closed.
- A profile grant cannot exceed the permissions declared by its plugin.
- Digest-bearing profile configuration accepts finite JSON numbers (plus
  booleans, strings, arrays, and objects); Rust enables correctly-rounded
  float parsing to preserve Python/Rust digest parity at decimal boundaries.
- Text field limits are measured in UTF-8 bytes across the Python SDK and Rust
  CLI, keeping non-ASCII metadata validation at the same boundary.
- Dependencies are resolved deterministically, with cycle and version-conflict
  detection. Explicit profile grants/configuration are retained when a bundle
  is first discovered through a dependency; incompatible repeated requests fail
  closed. Output paths are project-relative and profile digests are stable.
- The Rust CLI checks `requires.glr` against its compiled version. The Python
  SDK accepts an explicit `glr_version` for the same check and intentionally
  permits it to be omitted for source-side inventory without a runtime binding.
- `in-process` is metadata, not an authorization bypass.  The default
  isolation hint is `process`; no runner is implied by the contract.
- Existing `glr.project.v1` role commands and `glr.extensions` remain valid.
  Plugin profiles add an explicit composition layer and do not replace fixed
  project roles or the learner-neutral environment contract.

## Framework mapping

The recommended first-party learner profile is a TorchRL-backed plugin because
TorchRL is already the maintained optional integration and its 0.13 contract is
CI-tested.  A Sample Factory plugin can be added as an optional high-throughput
backend, particularly for Linux workloads, without changing the GLR protocol.
This ADR defines the slot and lifecycle; it does not claim that either
framework is bundled as a runtime plugin yet.

## Consequences

Projects gain a reviewable, reproducible extension inventory and a DSH-like
profile composition workflow while the control plane remains safe to inspect in
an untrusted checkout.  Registry/network acquisition, signed trust receipts,
and host runners remain follow-up work and must reuse these contracts rather
than adding an implicit import path.

## References

- [DeepSeek Harness](https://deepseek.com/harness/en/)
- [DeepSeek plugin packaging](https://deepseekplugin.org/en/docs/package-install)
- [ADR-0019: optional DeepSeek Harness provider](0019-add-optional-deepseek-harness-provider.md)
- [ADR-0023: project-local VX task runner](0023-add-project-local-vx-task-runner.md)
