# Offline source packages

Use this workflow for an explicitly selected source-only project handoff. It
does not package models, real datasets, recordings, game binaries or accounts.

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
7. Separately configure ignored local overrides, inspect dependency locks and
   obtain authorization for setup/execution. Run doctor and a bounded synthetic
   conformance check after recreating dependencies through VX. Missing prerequisites
   are blockers, not permission to download games or launch training.

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
