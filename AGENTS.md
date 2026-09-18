# Repository development baseline

Follow first principles, SOLID, explicit contracts, and Clean Architecture.
Keep responsibilities and dependency boundaries clear; avoid code smells.
Read [CONTRIBUTING.md](CONTRIBUTING.md) before implementing changes.

## Start here: bootstrap, run, test, verify

Run this chain from a clean clone. If all five steps succeed the checkout is
bootstrapped, tested, runnable, and still aligned with upstream.

```bash
git clone https://github.com/loonghao/GameLearningRuntime.git
cd GameLearningRuntime
vx setup                 # pinned Python 3.12.13, Node.js, uv, just, Rust 1.98.0, .NET 10.0.400
vx just layout-check     # every tool under tools/ is registered and domain-owned
vx just core-check       # ruff + ruff format + mypy + pytest (90% coverage gate)
vx just glr-doctor       # the control plane runs against this project
vx just glr-fork-gate    # exit 0 aligned, exit 5 drifted from canonical upstream
```

Agents must use `--format json` for machine-readable output; the entry point is
`python -m game_learning_runtime`. Full platform, environment, and
failure-criteria detail is in
[the agent onboarding guide](docs/guides/agent-onboarding.md).

## Python packaging and imports

- All Python business logic must belong to an explicitly named, installable
  package that can be distributed as a standard wheel (`.whl`). Follow the
  [Python package baseline](CONTRIBUTING.md#python-package-baseline).
- Put GLR runtime logic under `src/game_learning_runtime/`. Separate Python
  distributions must declare their own build metadata and import namespace.
- Never repair imports with `sys.path.append`, `sys.path.insert`, equivalent
  path mutation, `PYTHONPATH`, working-directory changes, or file-based loading
  of sibling source modules. Install the owning package and declare dependencies.
- Keep scripts and examples thin. Reusable logic belongs in the package, not
  in loose scripts or test helpers imported by production code.
- Editable installs support development; acceptance of packaging changes
  requires installing the built wheel in a clean environment and exercising
  imports and affected entry points outside the source checkout.
- When touching legacy code that violates this baseline, migrate the affected
  logic to its owning package; do not extend the workaround.

## Downstream training quality

The Python package baseline also applies to downstream projects using GLR.
Follow [the downstream quality guide](docs/guides/downstream-quality.md) when
creating or modifying adapters, trainers, evaluators, and training utilities.
Use standard `logging` with application-owned configuration, bounded queued
output and single-writer rotation; keep error aggregation optional. Require
offline behavioral regression tests and installed-wheel acceptance. Diagnostic
logs never replace authoritative learner metrics or terminal evidence.
Framework upgrades must follow the code/config/data/checkpoint migration contract
in that guide: inventory versions, back up consistently, dry-run to a new
destination, verify integrity, and retain a tested rollback.
Follow the adapter-builder module-boundary and user-release references for
downstream architecture and trained-stage distribution. A user installer includes
the inference runtime and immutable model, with installed-app acceptance evidence.

## Git identity

Use `loonghao <hal.long@outlook.com>` for GitHub commits and concise English
Conventional Commit messages (`<type>: <message>`).

## Repository layout and tool ownership

Automation must not be added ad hoc. Every executable belongs to a capability
domain under `tools/<domain>/` (`ci`, `packaging`, `providers`, `release`,
`docs`, `demo`, `governance`) and must be registered in `tools/registry.toml`
in the same commit. There is no `scripts/` directory, and `just layout-check`
fails when one reappears or when a tool is unregistered or orphaned.

Follow [the repository layout guide](docs/guides/repository-layout.md) for the
decision rules, the anti-pattern list, and the steps to add a tool. Keep tools
thin: reusable logic belongs in `src/game_learning_runtime/`.
