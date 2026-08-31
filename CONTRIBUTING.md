# Contributing

Thanks for helping make game-learning infrastructure reusable.

## Development

Prerequisites: Git and [vx](https://github.com/loonghao/vx). The committed
`vx.toml` and `vx.lock` select Python, uv, and just for local and CI use.

```powershell
git clone https://github.com/loonghao/GameLearningRuntime.git
cd GameLearningRuntime
vx setup
vx just ci
```

Optional integration contracts:

```powershell
vx just ci-torchrl
vx just ci-gymnasium
```

## Change contract

- Keep game adapters independent from learning algorithms.
- Add or update an ADR when changing a public boundary or wire format.
- Treat protocol and dataset schemas as versioned compatibility contracts.
- Add adversarial tests for lifecycle, shapes, dtypes, bounds, masks, and stale
  episode/step identity.
- Use Conventional Commits in English.
- Use `fix:` for user-visible corrections, `feat:` for compatible capability
  additions, and a `!` or `BREAKING CHANGE:` footer for incompatible contracts.
  Release Please derives versions and changelog entries from this history; do
  not edit release versions or tags by hand.
- Deny incidental metadata by default; never commit local paths, hostnames,
  process/window identifiers, credentials, or private runtime data.
- Do not add game instrumentation unless it is legal, authorized, and isolated
  behind an adapter.

Open a pull request only after local checks pass. A maintainer review and green
required checks are necessary before merge.

Performance claims must use the synthetic benchmark contract in
[`benchmarks/README.md`](benchmarks/README.md). Do not publish hostnames, user
paths, process identifiers, or private datasets in benchmark reports.
