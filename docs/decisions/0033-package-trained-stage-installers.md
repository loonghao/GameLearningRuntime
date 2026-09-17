# ADR-0033: Package trained-stage installers from reviewed runtime payloads

## Status

Accepted

## Decision

Keep source exports, model bundles, wheels and user installers separate. Add
build-time `game_learning_runtime.user_release` APIs to prepare and compile a
Windows x64 Inno installer from an explicitly staged application runtime. Reuse
model-bundle verification and record stage, environment/protocol, evidence kind,
and file inventory under `glr.user-release.v1`. Recheck files before compilation.

The caller supplies the frozen inference launcher, runtime dependencies, matching
GLR binaries, licenses, model card, evaluation receipt, and trusted compiler.
The APIs do not execute the payload, freeze Python, download prerequisites, or
implement a parallel GLR CLI. Installation is per-user and versioned, without
automatic execution, PATH changes or recursive uninstall deletion.

## Consequences

Compilation is distinct from signing, clean-machine startup, frozen inference,
live acceptance, and data-preserving uninstall. Synthetic installer tests state
their limited scope. Automatic freezing and other targets remain unsupported.
