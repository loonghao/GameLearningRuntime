# ADR-0027: Offline source-only project packages

Status: Proposed for review with the source-only implementation

Related: issue #116, ADR-0020, ADR-0021, model-bundle and checkpoint contracts.

Stage 2 of the plan below now ships a reporting gate: `glr package conformance`
runs an offline synthetic conformance check on an already materialized package.
It reports package validity, materialization, recipient-local overrides, declared
dependency locks and prerequisites as separate axes, and it never resolves or
installs dependencies, runs doctor or training, or accesses the network.
Recreating a locked environment stays an explicit, authorized recipient step and
is never an import side effect.

## Decision

Start with a deterministic, offline source envelope owned by the standalone Rust
CLI. `glr package plan --manifest selection.json` inventories only an explicit
file list. `export --manifest selection.json --output source.zip` writes a new
archive. `inspect source.zip` checks it without materializing files. `import`
requires an expected environment and project-defined contract fingerprint and
promotes a fully checked staging directory to a new destination atomically.

The strict `glr.source-package.v1` selection records package version, required
GLR range, environment/protocol identity, an opaque SHA-256 contract fingerprint,
source revision, redistribution license, and exact portable source paths. The
project owns construction of the fingerprint over observation/action/reward/
knowledge contracts and content/ruleset identity. GLR never infers compatibility
from game names or interprets ruleset identifiers.

The archive manifest adds creation-tool version, per-file byte counts and SHA-256
digests, and a SHA-256 over the ordered selection/tool/inventory tuple. No wall
clock or machine path is included. Archives are uncompressed ZIPs with normalized
permissions and timestamps, enabling reproducible offline inspection without a
registry or archive expansion amplification. Exactly one project manifest and
at least one lock file are required. This does not establish lock completeness
or runnable dependencies. `glr package conformance` now reports the state of a
materialized package on those axes; recreating dependencies, doctor and a live
synthetic reproduction remain separate, explicitly authorized steps.

Limits: 1,024 source files, 16 MiB per file, 128 MiB archive/expanded payload,
1 MiB manifest, 240-byte ASCII paths and depth 16. Reject traversal, device names,
absolute/drive/UNC paths, case collisions (including directories), links and
reparse points, conflicting file/directory prefixes, unexpected members, size or
digest mismatch, unknown schema/fields, and incompatible GLR requirements.
OS no-replace rename prevents a racing existing import destination being
overwritten. A rejected or interrupted operation never replaces an existing
project. A killed process may leave an isolated temporary directory for cleanup.

Selection is an explicit redistribution decision. Denied roots/extensions exclude
environments, caches, raw logs, recordings, datasets and binaries. An extension
allowlist cannot prove that a source file contains no secret or licensed content:
the exporter must review selected source contents, local overrides, credentials,
endpoints, and redistribution rights before export. The package is not a trusted
publisher attestation. Import does not run scripts, hooks, installers, pickle/model
loaders, doctor, training, or network operations. Only package validation is claimed.

## Deliberately separate stages of #116

1. Review this source-only wire contract and negative test corpus.
2. Test locked setup, ignored recipient-local overrides, doctor and synthetic
   reproduction on every supported OS. These are explicit execution operations,
   never import side effects.
3. Add optional model-bundle, dataset and knowledge entry groups with independent
   allowlists, provenance, redistribution authorization and checkpoint migration
   gates. The source envelope does not claim to transport those artifacts.
4. Define remote role admission as a separate contract: authenticated identity,
   package/policy digest, scoped capability, expiring lease and fencing token;
   monotonically ordered attempt IDs and idempotent result ingestion; bounded
   queue/lag/deadlines; cancellation and reconnect reconciliation. Unknown action
   outcomes are never replayed merely because a transport reconnects. Checkpoint
   ownership must be unique. Credentials remain deployment-local. No scheduler,
   learner or cloud provider becomes a mandatory dependency.
5. Require remote conformance for duplicate deliveries, stale ownership, dropped
   results, lease expiry, policy mismatch and checkpoint conflict before claiming
   authenticated multi-machine execution.

The source-only commands do not complete issue #116's model/dataset, locked
reproduction or cluster acceptance criteria. Keep that umbrella issue open.
