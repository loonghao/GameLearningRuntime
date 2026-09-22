# Offline source packages

Use this workflow for an explicitly selected source-only project handoff. It
does not package models, real datasets, recordings, game binaries or accounts.

For a trained-stage installer, follow the project's `USER_RELEASE.md` or the
sibling adapter-builder `references/user-releases.md`. It uses a separately
prepared runtime and model bundle; this source-only contract remains unchanged.

1. Select individual files and review their contents and redistribution rights.
   Exclude secrets, host paths, private endpoints and recipient-local overrides.
   Extension checks are not a secret scanner. Include exactly one project manifest,
   dependency locks and the source needed to explain/reproduce the project.
2. Write a selection file using the strict shape below. The project owns the
   contract SHA-256 over observation/action/reward/knowledge and ruleset/content
   identity; do not invent a fingerprint from the package name.
3. Run `glr --project PROJECT --json package plan --manifest selection.json`.
   Review the exact inventory, sizes and digests before export.
4. Run `glr --project PROJECT --json package export --manifest selection.json --output source.zip`.
   An existing output is refused. A new archive is deterministic for identical inputs.
5. Recipient runs `glr --json package inspect source.zip` offline. Inspection
   verifies all entries and does not execute or install anything.
6. Recipient runs `glr --json package import source.zip --destination NEW_DIRECTORY --expected-environment ENVIRONMENT_ID --expected-contract SHA256`.
   Destination parent must already exist. Environment/contract expectations must
   come from the reviewed handoff, not be blindly copied from untrusted input.
7. Recreate dependencies from the lock through VX as a separate, authorized step.
   The package is a file list; it never carries a virtual environment, an
   interpreter or a cache, and GLR never resolves, downloads or installs anything.
8. Recipient runs `glr --project NEW_DIRECTORY --json package conformance source.zip`
   offline. Add `--expected-environment ENVIRONMENT_ID` and
   `--expected-contract SHA256` to re-assert the reviewed handoff at this gate.
   A nested working directory resolves upward to the project manifest. Exit code
   `0` means the synthetic conformance check passed; `4` means reproduction is
   blocked and `blockers` explains why. Nothing is executed, installed or fetched.

## Recipient conformance

`glr package conformance` checks a package that is already materialized on disk.
It reads and hashes bytes; it never runs a role, hook, installer or trainer. The
report keeps its axes independent, so a passing check is a statement about the
package and the destination tree — never about training:

| Field | Meaning |
| --- | --- |
| `package.valid` | The archive verified offline: identity, inventory, sizes and digests. |
| `materialization` | Every declared file is present and intact, and nothing undeclared is present. |
| `local_overrides` | Recipient-local files matching `*.local.*` or `*.local`, reported and never merged. |
| `artifacts.run_store` | A denied cache, output or run-store path is present in the destination. |
| `dependency_setup` | `declared` or `missing`; `performed` is always `false`. |
| `prerequisites` | Declared role and task programs that must already exist; never fetched. |
| `axes.training`, `axes.live_acceptance`, `axes.policy_quality` | Always `not-evaluated`. |
| `claims.training_succeeded` | Always `false`. |

**Local overrides.** `*.local.*` and `*.local` paths are refused by the export
allowlist, so a package never carries them. In the materialized destination the
conformance scan reports them under `local_overrides.present_in_destination` and
ignores them; it never merges them into the project (ADR-0021). Supplying them is
a separate, explicit recipient action.

**Missing prerequisites.** An unavailable role or task program is a blocker with
a `remediation` string, not permission to download a game, install a provider SDK
or launch training. A blocked reproduction is still reported as a valid,
completely materialized package.

```json
{
  "schema_version": "glr.source-package.v1",
  "package_version": "1.0.0",
  "required_glr": ">=0.18.0, <1.0.0",
  "environment_id": "synthetic.example",
  "protocol_version": "1.0",
  "contract_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_revision": "reviewed-source-revision",
  "redistribution_license": "MIT",
  "files": ["glr-project.toml", "pyproject.toml", "uv.lock", "train.py"]
}
```

The `a` fingerprint above is a synthetic placeholder, not a usable compatibility
receipt. Read CLI help from the installed version: source changes do not update
an installed CLI. Public receipts use portable selected paths, never source-machine
roots. Report package validity, dependency setup, synthetic reproduction, live
acceptance and policy quality as separate results.

The initial implementation supports only source packages. Optional model/data
groups and authenticated cluster deployment remain tracked by issue #116.
