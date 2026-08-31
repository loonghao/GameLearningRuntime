# Release runbook

Game Learning Runtime uses Release Please to turn English Conventional Commits
on `main` into a reviewed release pull request. Maintainers do not edit the
version, changelog, or tag by hand.

## Normal release

1. Merge feature and fix pull requests only after all required checks pass.
2. The `Release` workflow creates or updates a Release Please pull request. Its
   version, changelog, manifest, package metadata, and README workflow pins must
   move together.
3. Review the release pull request like any other change. Verify the proposed
   semantic version, included commits, generated changelog, and green required
   checks.
4. Merge the Release Please pull request. The same workflow creates the
   immutable tag and GitHub Release, then:

   - checks out the tag rather than mutable `main`;
   - verifies the tag against `pyproject.toml`;
   - runs quality checks and core tests;
   - builds and checks one wheel/sdist pair;
   - attaches build provenance and the distributions to the GitHub Release;
   - publishes those same distributions to PyPI through Trusted Publishing.

5. Verify the workflow conclusion, tag/commit identity, non-draft GitHub
   Release, attached assets, PyPI project page, and installation in a clean
   temporary project.

The release-impacting Conventional Commit types are:

| Commit | Version impact |
| --- | --- |
| `fix: ...` | Patch |
| `feat: ...` | Minor |
| `feat!: ...` or a `BREAKING CHANGE:` footer | Minor while the project is pre-1.0 |
| `docs: ...` | Patch, so user-facing documentation is versioned with the package |
| `test:`, `ci:`, `chore:` | No release by themselves |

## PyPI Trusted Publishing

PyPI trusts `.github/workflows/release.yml` in this repository when its
`pypi-publish` job runs in the GitHub `pypi` environment. That path and
environment name are compatibility contracts. The publish job receives only
`id-token: write`; no PyPI API token is stored in GitHub.

`PERSONAL_ACCESS_TOKEN` is optional but recommended for the Release Please job.
GitHub suppresses workflows triggered by pull requests created with the default
`github.token`; a repository-scoped token lets the generated release pull
request receive the normal required checks. The publishing jobs never receive
that token.

## Recover an existing release

First rerun failed jobs from the original workflow run. If its artifacts have
expired, manually run the `Release` workflow from `main` and supply an existing
immutable tag such as `v0.2.0`. The workflow checks out and verifies that tag,
rebuilds the distributions, replaces the GitHub Release assets, and attempts
the same Trusted Publishing path.

Do not use recovery to move a tag, change released source, or overwrite a PyPI
file. PyPI rejects an existing distribution filename. If released source is
wrong, fix it on `main` and publish the next patch release.
