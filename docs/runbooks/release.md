# Release runbook

## Prerequisites

- Clean `main` at the intended release commit.
- Green required GitHub Actions checks.
- Version and changelog updated.
- `gh` authenticated as a maintainer.

## Steps

1. Run the complete local development runbook.
2. Create an annotated semantic-version tag:

   ```powershell
   git tag -a v0.1.0 -m "release: v0.1.0"
   git push origin v0.1.0
   ```

3. The release workflow verifies the tag against `pyproject.toml`, rebuilds the
   package, checks distributions, produces provenance attestations, and creates
   a GitHub Release with wheel and source archive.
4. Verify the release assets and install from the immutable tag in a clean
   temporary project.

## PyPI gate

PyPI publication is intentionally disabled in v0.1. Enable it only after the
`game-learning-runtime` project and GitHub trusted publisher are configured.
Add a separate OIDC publishing job; never add an API token to repository files.

## Recovery

Do not move or overwrite a published tag. Fix code/version, create the next
patch version, and document the superseded release.

