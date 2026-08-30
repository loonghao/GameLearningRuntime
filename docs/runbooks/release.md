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
   package, checks distributions, and uploads a short-lived workflow artifact.
4. Separate least-privilege jobs create the immutable GitHub Release and publish
   the same workflow artifact to PyPI through trusted publishing. No PyPI API
   token is stored in GitHub.
5. Verify the GitHub release assets, PyPI project page, and a PyPI installation
   in a clean temporary project.

## PyPI trusted publishing

PyPI trusts `.github/workflows/release.yml` in this repository when its
`pypi-publish` job runs in the GitHub `pypi` environment. The job receives only
the `id-token: write` permission and publishes through OIDC.

To publish an existing immutable tag whose GitHub Release already exists, run
the Release workflow manually from `main` and supply the tag. The workflow
checks out that tag, verifies its version, rebuilds and checks its distributions,
skips GitHub Release creation, and publishes to PyPI. PyPI rejects attempts to
overwrite an existing filename or release.

## Recovery

Do not move or overwrite a published tag. Fix code/version, create the next
patch version, and document the superseded release.
