# Contributing

Thanks for helping make game-learning infrastructure reusable.

## Development

Prerequisites: Git and [vx](https://github.com/loonghao/vx). The committed
`vx.toml`, `vx.lock`, `rust-toolchain.toml`, and `global.json` select Python,
Node.js, uv, just, rustup/Rust, and .NET inputs for local and CI use.

```powershell
git clone https://github.com/loonghao/GameLearningRuntime.git
cd GameLearningRuntime
vx setup
vx just dashboard-build dashboard-check
vx just ci
```

Optional integration contracts:

```powershell
vx just ci-torchrl
vx just ci-gymnasium
vx just rust-check
vx just provider-sdk-check
```

New to this repository, or arriving as an agent? Start with
[the agent onboarding guide](docs/guides/agent-onboarding.md): one command chain
bootstraps, tests, runs, and verifies upstream alignment.

## Python package baseline

This baseline also applies to downstream projects using GLR. The
[downstream quality guide](docs/guides/downstream-quality.md) extends it with
logging, error aggregation, training regression, and CI acceptance contracts.

Every Python business component must be developed as part of a standard,
installable Python package that can be distributed as a wheel (`.whl`). A
successful run from the repository root is not evidence that the package works
after installation. This baseline applies to runtime code, adapters,
integrations, and reusable logic used by scripts, examples, or benchmarks.

### Package ownership and namespaces

- GLR runtime code belongs under `src/game_learning_runtime/`, with imports
  rooted in `game_learning_runtime`. The distribution name
  `game-learning-runtime` and import name `game_learning_runtime` serve different
  purposes; use each consistently.
- A separately distributed component must have its own `pyproject.toml`,
  declared build backend, supported Python versions, dependencies, and explicit
  package discovery. Use a `src/<import_namespace>/` layout. Do not create a
  separate distribution for every module; cohesive logic shares its owning
  package.
- Use regular packages with `__init__.py` by default. Use PEP 420 namespace
  packages only for an intentionally shared namespace across distributions,
  with documented ownership and discovery rules.
- Use explicit imports such as
  `from game_learning_runtime.protocol import ...` for package boundaries.
  Explicit relative imports within one package are allowed. Do not depend on
  ambiguous top-level names such as `utils`, `common`, or `src`, wildcard imports,
  or import order to locate modules.
- Declare external runtime dependencies in project metadata and optional
  integrations in extras. Development dependencies must not silently satisfy
  undeclared runtime requirements. Keep public imports and dependency direction
  aligned with the architecture; avoid circular imports and module shadowing.

### Execution and resources

- Do not use `sys.path.append`, `sys.path.insert`, other search-path mutation,
  `PYTHONPATH`, or working-directory changes to make project imports work.
  Loading sibling source files with `importlib` or `exec` is not a substitute
  for packaging. Importing installed plugin modules by their qualified names
  or declared entry points is allowed.
- Install the package for development, including editable installs when useful.
  Run package commands through declared console entry points or
  `python -m <qualified.module>`. Do not execute files inside `src/` directly.
- Keep repository automation and example scripts thin: argument parsing and
  orchestration may live there; reusable business logic must live in an
  installable package. Production code must not import from `scripts`, `tests`,
  or repository-only benchmark helpers.
- Include required runtime resources in the wheel and access bundled resources
  through `importlib.resources`. Never require a source-checkout directory
  layout or write mutable state into an installed package directory. User
  configuration, datasets, and outputs use explicit external paths.
- Tests must import the installed package. Do not add test path configuration,
  `conftest.py` path injection, or source-loader fallbacks to hide missing
  package files or dependencies.

### Acceptance and migration

For changes to package structure, imports, runtime dependencies, resources, or
Python entry points, include the following evidence in review:

1. Build the wheel using the declared backend and check distribution metadata.
   The existing `vx just build` recipe builds the distributions; building alone
   does not prove installation works.
2. Install that exact wheel into a fresh virtual environment with its declared
   dependencies, without an editable install or injected source paths.
3. From a temporary directory outside the checkout, with `PYTHONPATH` unset,
   verify that affected public modules import from the installed distribution.
   Exercise affected entry points and bundled-resource reads there as well.
4. Validate the base install without optional extras. Test affected optional
   integrations with their declared extras separately; unrelated imports must
   not require optional dependencies.

Apply this baseline to all new Python logic. When modifying existing violations,
migrate the affected logic and its callers together. Do not add another path
workaround or claim repository-wide compliance from a limited migration.
Reviewers must reject new violations even when source-tree tests pass.

## Change contract

- Keep game adapters independent from learning algorithms.
- Every new script or tool must have an owning capability domain under
  `tools/<domain>/` and an entry in `tools/registry.toml` in the same commit.
  There is no `scripts/` directory. See
  [the repository layout guide](docs/guides/repository-layout.md).
- Do not add one-off automation next to the code it patches. Register it or do
  not add it.
- Add or update an ADR when changing a public boundary or wire format.
- Treat protocol and dataset schemas as versioned compatibility contracts.
- Keep engine providers behind the shared C# or C++ provider vocabulary; do
  not add another learner-facing wire envelope without an ADR and parity tests.
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
- Loader templates must keep unknown actions denied, use explicit upstream
  versions, stage rather than auto-install, and never add arbitrary reflection,
  script, dump, or generic call surfaces.
- Model examples must include `glr.model-bundle.v1` inputs and checksums; never
  publish source workstation paths or proprietary runtime traces.
- Runtime Host changes must preserve frame bounds, pre-dispatch fencing,
  no-retry mutation semantics, and truthful capabilities. A stdio smoke does
  not prove authentication, target binding, or live-engine behavior.

Open a pull request only after local checks pass. A maintainer review and green
required checks are necessary before merge.

Performance claims must use the synthetic benchmark contract in
[`benchmarks/README.md`](benchmarks/README.md). Do not publish hostnames, user
paths, process identifiers, or private datasets in benchmark reports.
