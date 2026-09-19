# ADR-0041: Ship portable training packages as group-scoped, non-executing envelopes

## Status

Proposed for design review. This record decides contracts and boundaries; it does
not implement them. Command verbs and schema identifiers below are **candidates
for review**, not settled names — see *Open questions for review*.

Related: issue #116, ADR-0005, ADR-0010, ADR-0013, ADR-0014, ADR-0015, ADR-0017,
ADR-0020, ADR-0021, ADR-0024, ADR-0027, ADR-0031, ADR-0033, and
[the phased plan](../planning/training-package-phases.md).

## Context

Issue #116 asks for a safe, portable training package and staged
developer/cluster distribution: export an explicitly selected project, deliver it
to another developer or an authorized training environment, inspect it without
execution, and recreate compatible dependencies without copying machine-specific
state. It is explicitly a proposal with phased acceptance, and it must preserve
the learner-neutral core.

ADR-0027 deliberately covers only the first slice of #116 and names four things
it does not deliver: locked reproduction, optional model/dataset/knowledge
groups, cluster admission, and remote conformance. Those are the subject of this
record.

### What already exists on `origin/main` (`e6d132e`)

ADR-0027 and `crates/glr-cli/src/package.rs` implement a deterministic, offline,
source-only envelope:

| Fact | Location |
| --- | --- |
| `glr.source-package.v1` selection; envelope manifest `glr-package.json` | `crates/glr-cli/src/package.rs:18,22` |
| Limits: 1,024 files, 16 MiB per file, 128 MiB total, 1 MiB manifest | `package.rs:19-21`, `:356` |
| Portable path rules: 240 bytes ASCII, depth 16, no drive/UNC/device/traversal forms | `package.rs:64-90` |
| Source-only extension allowlist plus denied roots (`.local`, `datasets`, `recordings`, `logs`, caches, secrets) | `package.rs:92-121` |
| Uncompressed (`Stored`) entries, `0o644`, selection sorted before hashing | `package.rs:290,413-415` |
| Content identity = SHA-256 over `(selection, tool_version, entries)` | `package.rs:253-259` |
| Inventory entries carry `path`, `size_bytes`, `sha256` — no role, no group | `package.rs:39-45` |
| Import refuses unless `--expected-environment` and `--expected-contract` match | `package.rs:445` |
| Staging directory in the destination parent, then `promote()` (no-replace `MoveFileW`/`rename`) | `package.rs:458,468`, `crates/glr-cli/src/filesystem.rs:7` |
| No execution: a `train.py` that raises on import survives import byte-identical | `package.rs:486,539-542` |
| Import reports `executed: false`, `training_ready: false` | `package.rs:471` |
| CLI round trip creates no run store in source or destination | `crates/glr-cli/tests/cli_contract.rs:140` |

### What #116 still needs

1. **Entry groups.** Issue #116 §1 wants project source, model artifacts,
   approved datasets, knowledge snapshots, aggregate reports, and optional
   redistribution-approved assets as *separate declared groups*, with
   source-only packages supported. Today there is exactly one group and one
   allowlist; every non-source path is refused.
2. **Role and authorization per file.** Issue #116 §2 wants every file to carry a
   declared role, and wants sensitive dataset/media export gated behind a
   separate reviewed allowlist plus redistribution authorization. Today a file
   is a bare path string.
3. **Season/content and ruleset binding.** Issue #116 §1 wants the season/content
   revision and ruleset fingerprint bound into the package. Today they are rolled
   into one opaque `contract_sha256` with no separate record, which makes a
   rejected handoff hard to diagnose.
4. **Locked reproduction and cluster admission.** ADR-0027 stages 2 and 4.

Two defects surfaced while reading the implemented envelope are recorded below
because they change the contract, not just the code:

- **Content identity is not stable across tool versions.** `identity()` folds
  `tool_version` into the hash (`package.rs:253-259`), so two exports of
  byte-identical content by different CLI versions disagree on
  `content_sha256`. A stable content identifier must exclude the producer's
  version.
- **Inspection buffers the entire archive.** `inspect()` reads the whole file
  with `read_file(archive, MAX_TOTAL)` (`package.rs:319`). That is deliberate and
  correct at 128 MiB, but it does not scale to the binary groups this ADR adds.

## Decision

### D1. One envelope, declared entry groups; source-only stays a profile

A package declares `entry_groups`: a non-empty subset of a **closed** vocabulary
— `source`, `model`, `dataset`, `knowledge`, `report`, `redistributable`. Every
path belongs to exactly one declared group. A path whose group is not declared is
a refusal, not a warning.

A package that declares only `source` is a **source-only package** and must
satisfy every ADR-0027 obligation unchanged. This ADR does not relax ADR-0027.
Interop rule:

- a `glr.source-package.v1` archive is treated as a single-group `source`
  package;
- a group-scoped package that declares only `source` is accepted by a
  source-only consumer when the shared fields agree;
- any other group in a package presented to a source-only consumer is a refusal.

Unknown group names, unknown fields, and unknown schema versions fail closed
(`deny_unknown_fields` is retained on every struct).

### D2. One opaque compatibility gate; revisions recorded but never compared

Required identity fields: `environment_id`, `protocol_version`, `required_glr`,
`required_api`, `target_platforms`, and `contract_sha256` — the project-owned
opaque SHA-256 over observation, action, reward, and knowledge contracts plus
content/ruleset identity.

Add two **advisory** opaque fields, `content_revision` and `ruleset_fingerprint`,
recorded in the manifest and echoed in receipts, **never compared by GLR**.

A mismatch — unknown schema version, incompatible `required_glr`/`required_api`,
or a `contract_sha256` that differs from the reviewer-supplied expectation —
fails **before** training setup begins. Crossing an incompatible contract never
silently reuses data or weights. Migration stays an explicit, recorded,
project-owned operation that produces a new digest, reusing the existing
checkpoint-contract migration rules (#54, #58); GLR never infers compatibility
from game names, season names, or ruleset identifiers.

### D3. Per-group admission: allowlist, role, provenance, authorization

Each group carries its own admission predicate. Defaults are deny-by-default for
everything except `source`.

| Group | Default | Admission predicate | Additional gate |
| --- | --- | --- | --- |
| `source` | allowed | existing extension allowlist + denied roots (`package.rs:92-121`) | exactly one project manifest and at least one lock file |
| `model` | denied | weight/config/metrics extensions only | must verify as `glr.model-bundle.v1` (ADR-0010) |
| `dataset` | denied | separate reviewed allowlist | redistribution authorization + demonstration provenance (ADR-0013) |
| `knowledge` | denied | snapshot files only | freshness-aware snapshot validation (ADR-0014) |
| `report` | denied | aggregate outputs only | no raw logs, recordings, or trajectories |
| `redistributable` | denied | explicit per-file license + authorization | provenance and license recorded per file |

Every included file declares a `role` from a closed, group-scoped vocabulary
(for example `project-manifest`, `dependency-lock`, `role-source`, `weights`,
`model-card`, `dataset-index`, `knowledge-snapshot`, `aggregate-report`). Roles
exist for receipts, per-group policy, and audit; they grant no behavior and are
never an execution hint.

Export stays allowlist-driven over an explicit file list. **Discovery never
recurses a workspace**: there is no "package this directory" mode, so a package
can never become an unintended workspace upload. `plan` (dry run) emits the exact
inventory with sizes and digests, and every refusal uses a stable error category
with a nonzero exit.

Excluded by default, for every group: local overrides (`*.local.*`, `*.local`),
credentials, account/process/window identifiers, absolute host paths, private
endpoints, raw logs, recordings, screenshots, real trajectories, and proprietary
payloads. A `dataset` or `redistributable` export additionally requires an
explicit redistribution authorization record — approver, scope, license, date —
carried in the manifest; without it the export is refused. Receipts and
diagnostics use portable selected paths only and never echo a source-machine
root, and no package or receipt embeds a credential, endpoint, or machine
binding.

### D4. Stable, deterministic content identity

Content identity is a SHA-256 over the **selection and the inventory entries
only**. `tool_version` is recorded in the manifest but excluded from the hash.
Nothing else enters it: no wall clock, no host path, no environment variable, no
filesystem ordering. Selection is sorted by documented byte order before hashing,
and entries keep normalized permissions and timestamps so identical content
yields identical bytes and an identical identifier.

Because this changes an existing computation, it needs an explicit migration
decision at implementation time (see *Open questions*): either a new source-only
schema revision or a parallel identity field with a documented transition.

### D5. Per-group limits, and archive-bomb defense for compressed groups

Limits stay per group, so the source-only profile keeps today's exact numbers and
today's behavior. Candidate bounds for review:

| Group | Max files | Max per file | Max group bytes |
| --- | --- | --- | --- |
| `source` | 1,024 | 16 MiB | 128 MiB |
| `model` | 64 | 1 GiB | 4 GiB |
| `dataset` | 4,096 | 1 GiB | 4 GiB |
| `knowledge` | 256 | 64 MiB | 256 MiB |
| `report` | 64 | 16 MiB | 64 MiB |
| `redistributable` | 256 | 256 MiB | 1 GiB |

A package also has one hard ceiling across all groups. The ceiling, not the
sum of the per-group caps, is the enforcement point.

The `source` group stays uncompressed (`Stored`): determinism and the absence of
expansion amplification are worth more than size there. Binary groups may be
compressed, but then each manifest entry must declare both compressed and
expanded sizes, and import must enforce:

- a per-entry and per-group expansion ratio cap (candidate: 200:1);
- the expanded cap **before** any expanded byte is written, using a running
  counter rather than the declared header;
- a streaming inspector, so a multi-GiB package is never buffered whole.

Every entry is still checked against its declared size and digest.

### D6. Import validates and materializes; it never executes

Import keeps and extends ADR-0027's guarantees. Validate everything first, then
materialize: reject traversal, absolute, drive/UNC and device forms, symlinks,
hard links, reparse points, duplicate and case-colliding entries, file-vs-
directory prefix conflicts, unexpected members, malformed metadata, and size or
digest mismatches. After normalization every staged path must remain inside the
staging root.

Materialization uses a dedicated staging directory in the destination's parent —
same filesystem, so promotion is one atomic no-replace rename — followed by
`promote()` (`filesystem.rs:7`). Consequences that must hold for the new groups
too:

- no automatic overwrite of an existing project; replacing one is a separate,
  explicit operation;
- an interrupted or refused import leaves the original destination intact and
  emits a bounded diagnostic receipt;
- a killed process may leave an isolated temporary directory for cleanup.

Import does not run scripts, hooks, installers, dependency resolution, doctor,
training, or any network operation, and it does not deserialize payloads. Model
and dataset files are the new hazard this ADR admits, so the rule is explicit:
no weight, checkpoint, or tensor deserializer with execution behavior (for
example `pickle`, `torch.load`, `numpy.load(allow_pickle=True)`) is invoked
during import; bytes are hashed and written, never loaded. Setup is a separate
authorized step taken after trust and compatibility checks, never an import side
effect.

### D7. Offline developer delivery, with prerequisites reported as blockers

A package is delivered as a local file and needs no registry and no network. The
recipient workflow is fixed and ordered:

1. `inspect` — verify the envelope offline;
2. `validate` — compare environment and contract against the reviewed handoff,
   never against values copied from the untrusted package itself;
3. materialize to a new directory;
4. supply recipient-local overrides. These stay out of the package and are never
   merged by unrestricted override merge (ADR-0021);
5. recreate the environment from the project's locks;
6. run doctor and a bounded synthetic conformance check.

Missing external prerequisites — a licensed game binary, a private dataset, a
provider SDK, a GPU/driver baseline — are reported as blockers. Nothing is
downloaded or installed implicitly. Clean-checkout and nested-working-directory
behavior follow ADR-0021.

### D8. Report the axes separately

Package validity, dependency setup, synthetic reproduction, live acceptance,
policy quality, and distribution adoption are independent results. The import
receipt already reports `executed: false` and `training_ready: false`; keep that
vocabulary and extend it to the new groups. **A valid package is never reported
as a successful training run.**

### D9. Cluster integration is a separate contract, outside the core

The package contract ends at bytes, identity, and provenance. Authenticated
multi-machine execution is a separate adapter/control-plane proposal with its own
review; it is not on this ADR's acceptance path and it never becomes a core
dependency (ADR-0005, ADR-0015, ADR-0032). No scheduler, learner implementation,
cloud vendor, or remote script endpoint is required to build, inspect, or import
a package.

That separate proposal must specify:

- authenticated learner and actor roles, distinct from the unauthenticated local
  control plane;
- package, contract, and policy identity carried as references — a deployment
  resolves them, the package does not carry them;
- scoped capabilities, expiring leases, and fencing tokens;
- explicit, unique ownership of checkpoints and outputs;
- disconnect/reconnect reconciliation, deadlines, cancellation, bounded
  backpressure, and policy-version lag;
- unique rollout and attempt identity, and idempotent result ingestion so a
  duplicate delivery cannot become a duplicate optimizer update;
- credentials and machine bindings that stay deployment-local.

It should reuse ADR-0020's local primitives while stating their limits: the
lease barrier is in-process, durable attempt metadata does not restore an
in-memory queue after exit, and cross-process ownership and policy publication
coordination remain future work. Unknown in-flight action outcomes are never
replayed merely because a transport reconnected.

## Non-functional requirements

- **Correctness:** one compatibility gate; unknown groups, fields, and schema
  versions fail closed before training, never during it.
- **Security:** no execution on import, no recursive discovery, no credential or
  machine binding in a package, bounded resources, atomic no-replace promotion.
- **Reproducibility:** content identity excludes the producer's version;
  environments are recreated from locks, never shipped as virtual environments,
  interpreters, or private caches.
- **Portability:** relative package-root paths, no host path, account, process,
  window, endpoint, or game-installation identifier in any manifest or receipt.
- **Performance:** the source profile stays uncompressed and fully buffered as
  today; binary groups require streaming inspection and pre-write caps.
- **Operability:** stable error categories, nonzero refusal exits,
  machine-readable receipts, and a dry-run inventory before every export.

## Failure modes and mitigation

| Failure | Mitigation |
| --- | --- |
| Package crosses an incompatible ruleset | `contract_sha256` mismatch fails before setup; no data or weight reuse |
| Producer version changes the package identity | identity excludes `tool_version` (D4) |
| Sensitive dataset exported by habit | `dataset` group is deny-by-default and needs a recorded redistribution authorization |
| Archive bomb in a compressed group | declared compressed/expanded sizes, ratio cap, and pre-write expanded cap |
| Import interrupted mid-materialization | staging + atomic no-replace rename; original destination untouched |
| Recipient imports blindly from untrusted input | `--expected-environment` / `--expected-contract` come from the reviewed handoff |
| Duplicate delivery in a cluster | idempotent ingestion keyed on unique rollout/attempt identity |
| Reconnect replays an unknown action | never replay an unknown in-flight outcome (ADR-0010, ADR-0020) |
| "Package imported" read as training success | separate reporting axes, `training_ready: false` (D8) |

## Consequences

### Positive

- #116 gains one envelope with declared groups instead of a second, parallel
  archive format, and the source-only guarantees do not weaken.
- Per-file roles and per-group authorization make an export reviewable as a
  redistribution decision, not just as a file list.
- A version-independent content identifier makes receipts, dedup, and
  "did this exact package already arrive?" questions answerable.
- The cluster boundary is written down before any code exists, so a future
  adapter cannot quietly make a scheduler a core dependency.

### Negative

- Six admission predicates and per-group limits are real surface to maintain and
  test; each new group needs its own negative corpus.
- Streaming inspection and pre-write expansion caps are a substantial change from
  the current read-it-all `inspect()`.
- Excluding `tool_version` from identity is a breaking change for any consumer
  that already stored a `content_sha256` (D4 migration).
- Binary groups admit payloads whose safety depends on a deserializer that must
  never run on import.

### Neutral

- The learner-neutral core is unchanged: no learner, optimizer, scheduler, or
  cloud vendor is added.
- `glr.model-bundle.v1`, trained-stage installers (ADR-0033), and the checkpoint
  migration contract keep their existing roles; this envelope transports, it does
  not replace them.
- Checksums still prove integrity, not publisher trust or model quality. Optional
  attestations need a separate trust policy.

## Alternatives considered

**One opaque digest vs. structured season/ruleset fields.** Chosen: one opaque
`contract_sha256` as the only gate, with `content_revision` and
`ruleset_fingerprint` recorded but never compared. Rejected: making ruleset
identity a first-class comparison — GLR must not interpret game or ruleset
semantics (ADR-0008, ADR-0021), and two gates leave it ambiguous which one wins.

**One schema with optional group fields.** Rejected. Optional fields make "what
does this package contain" ambiguous, and a source-only consumer must be able to
refuse an unknown group fail-closed. Chosen: explicit declared groups with a
documented single-group interop rule (D1).

**Loosen the source extension allowlist to carry weights.** Rejected. That
allowlist is the privacy and authorization boundary; loosening it silently admits
licensed binaries and executed deserializers through the group that is allowed by
default.

**Compress everything.** Rejected for `source`: compression breaks
byte-deterministic offline inspection and reintroduces expansion amplification.
Allowed for binary groups under declared sizes and a ratio cap (D5).

**Run setup and doctor automatically on import.** Rejected. Setup is an
authorized execution step; folding it into import turns a validation tool into an
execution vector (issue #116 §3, ADR-0027 stage 2).

**Recursive workspace discovery with a deny list.** Rejected. Deny lists fail
open; selection must stay an explicit file list (D3).

**Make the cluster control plane part of the package.** Rejected. Credentials and
machine bindings must stay deployment-local, and no scheduler or learner may
become a mandatory core dependency (D9, ADR-0005).

**Treat matching digests as publisher trust.** Rejected. Integrity is not
attestation, and neither is a claim about policy quality (ADR-0027).

## Open questions for review

1. **Names.** Final command verbs and schema identifiers are deliberately
   unresolved. Candidates in this record: `glr.training-package.v1` for the
   envelope, `entry_groups` / `role` / `content_revision` /
   `ruleset_fingerprint` / `redistribution_authorization` for fields, and
   `plan` / `export` / `inspect` / `validate` / `import` / `setup` for stages.
2. **D4 migration.** New source-only schema revision, or a parallel identity
   field with a documented transition?
3. **Group vocabulary.** Are `report` and `redistributable` distinct groups, or
   should `report` be derived output excluded from packages entirely?
4. **Limit numbers.** The per-group table in D5 needs a decision against real
   model and dataset sizes.
5. **Knowledge freshness.** Should `knowledge` snapshots be revalidated at
   import, or only recorded and validated at setup?
6. **M4 ordering.** Should `model` land before `dataset`, or together behind one
   authorization gate?

## References

- [Issue #116: Define safe portable training packages and staged developer/cluster distribution](https://github.com/loonghao/GameLearningRuntime/issues/116)
- [ADR-0010: Authorized loader plugins and reproducible model bundles](0010-add-authorized-loader-plugins-and-reproducible-model-bundles.md)
- [ADR-0013: Bind demonstration provenance to trajectory bytes](0013-bind-demonstration-provenance-to-trajectory-bytes.md)
- [ADR-0020: Rollout attempts and local queue barriers](0020-rollout-attempts-and-queue-barriers.md)
- [ADR-0021: Resolve portable projects from one manifest](0021-portable-project-manifests.md)
- [ADR-0027: Offline source-only project packages](0027-offline-source-packages.md)
- [ADR-0033: Package trained-stage installers](0033-package-trained-stage-installers.md)
- [Phased acceptance plan](../planning/training-package-phases.md)
