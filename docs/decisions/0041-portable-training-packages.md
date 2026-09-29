# ADR-0041: Ship portable training packages as group-scoped, non-executing envelopes

## Status

Accepted for design review. M0 (this record and the phased plan), M1, M2, M3
and M5 are complete; **M4 — optional model, dataset, knowledge and report
groups — is implemented** by the change that records the decisions below.

Names, schema identifiers and the limit table were **candidates** when this
record was written. M4 settles the ones it implements: see
*Decisions settled by M4*. The open questions that remain are deliberately
unresolved and do not block M4.

## Decisions settled by M4

1. **Envelope name.** `glr.training-package.v1` for the group-scoped profile.
   `glr.source-package.v1` (ADR-0027) is unchanged and stays a supported
   profile, not a legacy format: its selection, entries, wire shape and content
   identity are byte-for-byte what ADR-0027 shipped. The two are distinguished
   by `selection.schema_version`, which remains a top-level field in both. An
   unknown schema version is refused at parse time.
2. **Group vocabulary.** Five groups: `source`, `model`, `dataset`,
   `knowledge`, `report`. `redistributable` from the original table is
   **deferred**, not dropped: the four implemented groups cover every payload
   issue #116 names, and a sixth admission predicate with its own per-file
   license record needs its own review and its own negative corpus. The
   vocabulary stays closed, so adding it later is an additive schema change.
   Open question 3 is therefore half-answered: `report` is a distinct group;
   `redistributable` is still open.
3. **Every group-scoped package declares `source`.** The project manifest and
   dependency lock bind the environment and protocol a package is valid for, so
   a payload-only package would have nothing to bind them to. A `source`-only
   group-scoped package satisfies every ADR-0027 obligation unchanged.
4. **Group ownership is a path root.** `models/`, `data/`, `knowledge/` and
   `reports/` belong to their group; every other path is `source`. A path under
   an undeclared group's root is a refusal, which is what makes deny-by-default
   mechanical rather than a matter of policy interpretation. The `dataset` root
   is `data/` rather than `datasets/`: the latter is already a denied
   cache/output root in the export allowlist.
5. **Limit table** (replaces the candidate table in D5):

   | Group | Max files | Max per file | Max group bytes | Compression |
   | --- | --- | --- | --- |
   | `source` | 1,024 | 16 MiB | 128 MiB | `Stored` |
   | `model` | 64 | 1 GiB | 4 GiB | `Deflated` |
   | `dataset` | 512 | 1 GiB | 4 GiB | `Deflated` |
   | `knowledge` | 256 | 64 MiB | 256 MiB | `Deflated` |
   | `report` | 64 | 16 MiB | 64 MiB | `Deflated` |

   Two ceilings sit above the per-group caps and are the real enforcement
   points: **1,024 files** and **4 GiB** expanded per package. The file ceiling
   equals the archive member gate, so a package can never contain more files
   than an archive may carry — which is why `dataset` is 512 and not the 4,096
   originally proposed. The package ceiling, not the sum of the group caps, is
   what a recipient pays for.
6. **Knowledge freshness** (open question 5): a snapshot is validated at both
   export and import, against the wall clock and a **declared** `max_age_days`
   that the knowledge group must carry. Without a declared budget nothing is
   reviewable, so the export is refused; a snapshot stamped in the future is
   refused too. Freshness is therefore a property of the verification moment,
   which is the point: a package that was valid last quarter may legitimately
   be refused today.
7. **M4 ordering** (open question 6): `model`, `dataset`, `knowledge` and
   `report` landed together behind one envelope and separate admission
   predicates. Shipping `model` first would have built the authorization gate
   for `dataset` last, where it is cheapest to omit.
8. **Roles** are declared per file by the exporter and cross-checked against a
   derived role (`project-manifest`, `dependency-lock`, `source-file`,
   `model-manifest`, `model-input`, `model-artifact`, `dataset-manifest`,
   `dataset-payload`, `knowledge-snapshot`, `aggregate-report`). A declared role
   that disagrees with the path is a refusal: a role is an assertion someone
   must make, not a label GLR infers.
9. **Content identity excludes `compressed_size_bytes`.** A dry run hashes
   bytes but never compresses them, so the compressed size does not exist when
   a plan is computed. Excluding that one transport field from the hash keeps a
   dry run and the export it previews on the same identifier; the compressed
   size is still verified against the archive that carries the entry.

Open question 2 (the D4 migration) is unaffected by M4: the source-only identity
computation is unchanged, so no consumer of `glr.source-package.v1` migrates.

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
— `source`, `model`, `dataset`, `knowledge`, `report`. Every path belongs to
exactly one declared group, by package-root directory (`models/`, `data/`,
`knowledge/`, `reports/`; everything else is `source`). A path whose group is not
declared is a refusal, not a warning. Every group-scoped package declares
`source`.

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
| `source` | allowed | existing extension allowlist + denied roots | exactly one project manifest and at least one lock file |
| `model` | denied | weight/config/metrics extensions only; no deserializer-only format (`pkl`, `pickle`, `joblib`, `npy`, `npz`, `dill`) | must verify as `glr.model-bundle.v1` (ADR-0010), and every model file must be declared by that bundle |
| `dataset` | denied | separate reviewed allowlist naming each path | redistribution authorization + demonstration provenance binding every payload (ADR-0013) |
| `knowledge` | denied | snapshot files only | freshness-aware snapshot validation against a declared `max_age_days` (ADR-0014) |
| `report` | denied | aggregate outputs only (`json`, `md`, `csv`) | no raw logs, recordings, or trajectories, by extension *and* by path component |

`redistributable` — explicit per-file license plus authorization — is deferred
from the implemented vocabulary (see *Decisions settled by M4*, item 2).

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
today's behavior. The settled bounds are in *Decisions settled by M4*, item 5:
1,024 files and 4 GiB expanded per package are the two ceilings that actually
apply, above the per-group caps.

The `source` group stays uncompressed (`Stored`): determinism and the absence of
expansion amplification are worth more than size there. Binary groups are
compressed, and each manifest entry declares both compressed and expanded sizes
(`compressed_size_bytes` is absent for a stored entry). Both export and import
enforce:

- a per-entry expansion ratio cap of **200:1**, checked on the declared sizes at
  import and on the sizes the encoder actually produced at export, so an
  archive bomb is refused before it leaves the sender as well as before it is
  expanded by the recipient;
- the expanded cap **before** any expanded byte is written, using a running
  counter rather than the declared header, per file, per group and per package;
- a streaming inspector and a streaming materializer, so a multi-GiB package is
  never buffered whole — only a declared manifest or snapshot is materialized at
  all, and only under an 8 MiB inspection cap.

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

The contract that answers this list is
[ADR-0045: Admit remote roles with scoped capabilities, epoch-scoped fences, and
coordinator-owned checkpoints](0045-remote-role-admission.md)
(`glr.remote-admission.v1`; accepted 2026-09-28 at Phase 0, no code yet). Its D2–D11
specify the roles, admission claims, fencing, ingestion, reconciliation, state taxonomy,
checkpoint ownership, and trust boundary named above, and its conformance checklist
carries the tests. Two cross-references worth keeping in view: it takes `package_digest`
from §D4 above (and so inherits this ADR's open question 2 on the D4 migration, which it
answers by following §D4 and reading identity through an injected identity function), and
it answers this section's "learner and actor roles" as **actor-only in v1**, deferring
remote learner admission rather than designing it away. The accepted contract supersedes
[the original proposal record](../planning/remote-role-admission.md), which is retained
for its baseline survey and review history.

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

M4 answered 1, 4, 5 and 6 and half of 3; see *Decisions settled by M4*. What
remains open:

1. **Command verbs and the two advisory identity fields.**
   `plan` / `export` / `inspect` / `import` / `conformance` are the stages that
   exist. `validate` and `setup` from the original candidate list are still
   unnamed work, and `content_revision` / `ruleset_fingerprint` (D2) are still
   unimplemented: they are advisory and recorded-only, so nothing in M4 depends
   on them.
2. **D4 migration.** New source-only schema revision, or a parallel identity
   field with a documented transition? Unchanged by M4: the source-only identity
   computation is untouched, so no existing consumer migrates.
3. **Group vocabulary.** `report` is settled as a distinct group.
   `redistributable` — explicit per-file license plus authorization — is still
   open, and deferred until a payload needs it.
4. **Limit numbers.** Settled for M4 (item 5 above) and still open to revision
   against real model and dataset sizes: the numbers live in one table in
   `crate::package_groups`, and the corpus mirrors them, so changing them is one
   reviewed commit and a matching corpus update.
5. **Knowledge freshness.** Settled: validated at both export and import.
   Whether a *stale-but-declared* snapshot may also be revalidated at setup is
   the remaining half, and setup does not exist yet.
6. **M4 ordering.** Settled: all four groups landed together.

## Wire schemas

- [`docs/schemas/source-package.schema.json`](../schemas/source-package.schema.json) and
  [`source-package-selection.schema.json`](../schemas/source-package-selection.schema.json) —
  `glr.source-package.v1`, unchanged by M4.
- [`docs/schemas/training-package.schema.json`](../schemas/training-package.schema.json) and
  [`training-package-selection.schema.json`](../schemas/training-package-selection.schema.json) —
  `glr.training-package.v1`, the group-scoped profile M4 adds.
- [`docs/schemas/package-audit.schema.json`](../schemas/package-audit.schema.json) —
  `glr.package-audit.v1`, the aggregate audit receipt that names every admitted
  non-source file.

## References

- [Issue #116: Define safe portable training packages and staged developer/cluster distribution](https://github.com/loonghao/GameLearningRuntime/issues/116)
- [ADR-0010: Authorized loader plugins and reproducible model bundles](0010-add-authorized-loader-plugins-and-reproducible-model-bundles.md)
- [ADR-0013: Bind demonstration provenance to trajectory bytes](0013-bind-demonstration-provenance-to-trajectory-bytes.md)
- [ADR-0020: Rollout attempts and local queue barriers](0020-rollout-attempts-and-queue-barriers.md)
- [ADR-0021: Resolve portable projects from one manifest](0021-portable-project-manifests.md)
- [ADR-0027: Offline source-only project packages](0027-offline-source-packages.md)
- [ADR-0033: Package trained-stage installers](0033-package-trained-stage-installers.md)
- [Phased acceptance plan](../planning/training-package-phases.md)
- [Planning: Optional authenticated cluster distribution — remote role admission](../planning/remote-role-admission.md)
