# Phased plan: portable training packages (issue #116)

This document describes future work, not current capability. It turns issue
#116's phased acceptance into ordered milestones with exit conditions. The
contract each milestone builds is
[ADR-0041](../decisions/0041-portable-training-packages.md); the source-only
slice it starts from is [ADR-0027](../decisions/0027-offline-source-packages.md).

## Standing constraints

Every milestone inherits these rules. A milestone that violates one is not done.

- **Generic synthetic examples only.** No real game binaries, licensed payloads,
  private datasets, recordings, or customer data in fixtures or receipts.
- **No execution on import.** Import validates and materializes bytes. It never
  runs a script, hook, installer, deserializer, doctor, training job, or network
  operation.
- **No implicit downloads or installs.** Missing prerequisites are reported as
  blockers.
- **Report the axes separately.** Package validity, dependency setup, synthetic
  reproduction, live acceptance, and policy quality are independent results. A
  valid package is never a successful training run.
- **Each milestone ships its own negative corpus.** Positive round-trip tests do
  not demonstrate a refusal.

## Milestones

### M0 — ADR and wire schema review

**Goal.** Agree the contract before any code.

**Depends on.** Nothing.

**Exit criteria.**

- [ ] ADR-0041 reviewed and accepted, including the six open questions in it.
- [ ] Candidate wire shapes reviewed against `glr.project.v1`,
      `glr.source-package.v1`, `glr.model-bundle.v1`, and
      `glr.checkpoint-contract.v1`; conflicts and reuse recorded.
- [ ] Group vocabulary, per-group admission predicates, and limit table agreed.
- [ ] D4 identity migration decision recorded.
- [ ] Command verbs and schema identifiers agreed.

**Explicitly not claimed.** No implementation, no CLI surface, no compatibility
with an unreleased format.

**Primary artifacts.** `docs/decisions/0041-*.md`, schema drafts, this document.

### M1 — Deterministic source-only export and offline inspect/import

**Goal.** Prove the source-only round trip is deterministic, offline, and
non-executing at the CLI level.

**Depends on.** M0.

**Exit criteria.**

- [ ] Export is byte-deterministic for identical inputs.
- [ ] Clean-directory round trip: export → inspect → import into a fresh
      directory reproduces the selected files exactly.
- [ ] Import into an existing destination is refused; the existing tree is
      unchanged.
- [ ] No run store or execution artifact appears in the source or the
      destination.
- [ ] No network access during plan, export, inspect, or import.

**Already covered today** (verify, do not rebuild): `crates/glr-cli/src/package.rs`
unit tests and `package_cli_plans_exports_and_imports_without_creating_a_run_store`
in `crates/glr-cli/tests/cli_contract.rs` cover determinism, no-execution, and
no-run-store for the source-only profile.

**Explicitly not claimed.** No dependency setup, no reproduction, no remote
delivery, no model or dataset transport.

### M2 — Negative security corpus

**Goal.** Prove the refusals, not just the happy path.

**Depends on.** M1.

**Exit criteria.** One or more failing cases for each of:

- [ ] path traversal, absolute, drive/UNC and device forms;
- [ ] symlinks, hard links, reparse points;
- [ ] duplicate and case-colliding entries, file-vs-directory prefix conflicts;
- [ ] unexpected members and missing declared files;
- [ ] oversized files and archives, and expansion beyond the cap;
- [ ] corruption: size mismatch, digest mismatch, identity mismatch;
- [ ] malformed metadata, unknown fields, unknown schema version;
- [ ] incompatible `required_glr`, environment, or contract expectation;
- [ ] interruption and overwrite safety: a refused or interrupted import leaves
      the original destination intact and emits a bounded receipt;
- [ ] privacy defaults: local overrides, credentials, logs, recordings, and
      trajectories are refused.

**Explicitly not claimed.** No claim that an extension allowlist detects secrets
or licensed content inside an allowed file. Content review stays a human step.

### M3 — Locked setup, doctor, and synthetic reproduction

**Goal.** Make a received package runnable by an explicit, authorized recipient
action — and prove it reproduces.

**Depends on.** M2.

**Exit criteria.**

- [ ] Recipient can recreate the environment from the project's locks on every
      supported OS.
- [ ] Recipient-local overrides stay out of the package and are applied without
      an unrestricted override merge (ADR-0021).
- [ ] `doctor` and a bounded synthetic conformance check pass after setup.
- [ ] Missing external prerequisites are reported as blockers, with no implicit
      download or install.
- [ ] Setup produces a report distinct from package validity.

**Explicitly not claimed.** Live acceptance, policy quality, and cross-machine
execution.

### M4 — Optional model, dataset, knowledge, and report groups

**Goal.** Admit non-source payloads without weakening the source-only
guarantees.

**Depends on.** M3 (so reproduction is proven before payloads are added).

**Exit criteria.**

- [ ] Each new group is deny-by-default and refuses when its group is not
      declared.
- [ ] `model` entries verify under `glr.model-bundle.v1` (ADR-0010).
- [ ] `dataset` export requires a separate reviewed allowlist plus a recorded
      redistribution authorization and provenance binding (ADR-0013).
- [ ] `knowledge` snapshots pass freshness-aware validation (ADR-0014).
- [ ] `report` admits aggregates only; raw logs, recordings, and trajectories are
      refused.
- [ ] Streaming inspection and pre-write expansion caps hold for compressed
      groups; the ratio cap is enforced.
- [ ] Import still deserializes nothing, including weights and datasets.
- [ ] Aggregate audit receipts cover every admitted non-source file.

**Explicitly not claimed.** Model quality, dataset usefulness, or any claim that
a checksum attests to a publisher.

### M5 — Multi-process and multi-machine conformance (separately reviewed)

**Goal.** Prove the optional cluster adapter, not the core.

**Depends on.** M4. Reviewed as its own proposal; see ADR-0041 D9.

**Exit criteria.**

- [ ] Authenticated learner and actor roles, distinct from the local control
      plane.
- [ ] Scoped capabilities, expiring leases, and fencing tokens.
- [ ] Unique ownership of checkpoints and outputs.
- [ ] Disconnect and reconnect reconciliation; no blind replay of an unknown
      in-flight outcome.
- [ ] Deadlines, cancellation, bounded backpressure, and policy-version lag.
- [ ] Idempotent result ingestion: a duplicate delivery is not a duplicate
      optimizer update.
- [ ] Unique rollout/attempt identity across machines.
- [ ] Credentials and machine bindings stay deployment-local and absent from
      every package and receipt.
- [ ] Core builds, inspects, and imports packages with no scheduler, learner,
      cloud vendor, or remote endpoint available.

**Explicitly not claimed.** Scheduler performance, learner quality, or a
reference production deployment.

### M6 — Skills and documentation alignment

**Goal.** Make the documented workflow identical to the executable one.

**Depends on.** M1..M5 as each lands; M6 is the closing gate.

**Exit criteria.**

- [ ] `glr-cli`, adapter-builder, and QA skills describe the same commands,
      receipts, and error categories the CLI implements.
- [ ] Skills state the privacy and trust gates, local-override handling, and the
      distinction between package validity, local reproduction, remote
      execution, and model quality.
- [ ] Doctor, onboarding, and downstream-quality guides link the package
      workflow.
- [ ] Documentation uses only synthetic examples.

**Explicitly not claimed.** Adoption by installed clients, live acceptance, or
learning quality.

## Mapping to issue #116's acceptance checkboxes

| #116 checkbox | Milestone |
| --- | --- |
| ADR and wire schemas reviewed against existing contracts; synthetic examples only | M0 |
| Deterministic source-only export and offline inspect/import, round-trip clean-directory tests, no execution/network side effects | M1 |
| Negative security corpus | M2 |
| Locked setup plus doctor and synthetic reproduction on supported OS | M3 |
| Optional model/dataset/knowledge inclusion with provenance, contract gates, allowlists, audit receipts | M4 |
| Separately reviewed multi-process/multi-machine conformance | M5 |
| Skills and documentation use the same executable contract | M6 |

## Closing issue #116

Issue #116 is the umbrella and stays open until M0 through M6 are complete.
M1 alone does not close it: a source-only package deliberately does not transport
models or datasets, does not set up dependencies, and does not prove
reproduction. Each milestone is reported on its own, and distribution, installed
adoption, live acceptance, and learning quality are reported independently of one
another.
