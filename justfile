set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
export UV_PROJECT_ENVIRONMENT := ".venv-glr"

default: check

# Show the exact checkout backing the active project environment.
origin:
    vx uv run python -c "import game_learning_runtime as glr; print(glr.__file__)"

# Install the locked project dependencies into the project-owned environment.
setup:
    vx uv sync --python 3.12.13 --frozen --all-groups --no-install-project
    vx uv sync --python 3.12.13 --frozen --all-groups --no-build-isolation

# Prove that dependency resolution has not drifted.
lock-check:
    vx uv lock --check

lint:
    vx uv run python -m ruff check .

format-check:
    vx uv run python -m ruff format --check .

typecheck:
    vx uv run python -m mypy

workflow-check:
    vx actionlint

test:
    vx uv run python -m pytest -m "not torchrl" --cov=game_learning_runtime --cov-report=term-missing

core-check:
    vx uv run python tools/ci/run_core_checks.py

rust-format-check:
    vx cargo fmt --all -- --check

# Build once before Cargo. CI downloads this same checked artifact for every target.
dashboard-build:
    vx npm --prefix dashboard-ui ci --ignore-scripts
    vx npm --prefix dashboard-ui run build

dashboard-check:
    vx npm --prefix dashboard-ui run format:check
    vx npm --prefix dashboard-ui test

rust-clippy:
    vx cargo clippy --workspace --all-targets --locked -- -D warnings

rust-test:
    vx cargo test --workspace --locked

rust-build:
    vx cargo build --workspace --bins --locked

host-smoke: rust-build
    vx uv run python tools/ci/run_host_smoke.py

rust-check: rust-format-check rust-clippy rust-test host-smoke

csharp-check:
    vx dotnet build sdk/csharp/GameLearningRuntime.Provider.Smoke/GameLearningRuntime.Provider.Smoke.csproj --configuration Release
    vx dotnet run --project sdk/csharp/GameLearningRuntime.Provider.Smoke/GameLearningRuntime.Provider.Smoke.csproj --configuration Release --no-build

cpp-check:
    vx uv run python tools/providers/check_cpp_provider.py

provider-sdk-check: csharp-check cpp-check

agent-plugin-check:
    vx uv run python tools/packaging/package_agent_plugin.py --check

# Repository governance: every tool under tools/ must be registered and domain-owned.
layout-check:
    vx uv run --no-sync python tools/governance/check_tool_registry.py

# Standard project control-plane entry points. Human output is a table; add
# `--format json` when a script or CI job needs the stable JSON envelope.
glr-doctor project=".":
    vx uv run --no-sync python -m game_learning_runtime.cli --project "{{project}}" doctor

glr-train project=".":
    vx uv run --no-sync python -m game_learning_runtime.cli --project "{{project}}" train

glr-runs limit="20" project=".":
    vx uv run --no-sync python -m game_learning_runtime.cli --project "{{project}}" runs list --limit {{limit}}

glr-query world="default" project=".":
    vx uv run --no-sync python -m game_learning_runtime.cli --project "{{project}}" query entities --world {{world}}

# Anti-fork gate. Exits 5 when this checkout has drifted from the canonical upstream.
glr-fork-gate:
    vx uv run --no-sync python -m game_learning_runtime.cli --format json fork-gate

# One scheduler-friendly supervision pass. Exit 0 healthy, 3 recovered, 4 escalated.
glr-watchdog *args:
    vx uv run --no-sync python -m game_learning_runtime.cli --format json watchdog tick {{ args }}

# Full local integration suite for the project, runtime host, providers and package.
integration-suite: check build

check: setup lock-check workflow-check core-check rust-check provider-sdk-check agent-plugin-check layout-check

build:
    vx cargo build --release --workspace --bins --locked
    vx uv run --no-sync python -m build --no-isolation
    vx uv run python tools/packaging/check_dist.py

# Local pre-push equivalent of the core CI and package gates.
ci: check build

# GitHub Actions matrix lane. Each job owns its environment, so the selected
# interpreter can safely replace the baseline Python from vx.toml.
ci-core python_version:
    vx uv lock --check
    vx uv sync --python {{python_version}} --frozen --all-groups --no-install-project
    vx uv sync --python {{python_version}} --frozen --all-groups --no-build-isolation
    vx uv run --no-sync python tools/ci/run_core_checks.py

ci-gymnasium:
    vx uv sync --python 3.12.13 --frozen --all-groups --extra gymnasium --no-install-project
    vx uv sync --python 3.12.13 --frozen --all-groups --extra gymnasium --no-build-isolation
    vx uv run --no-sync python -m mypy src/game_learning_runtime/integrations/gymnasium.py
    vx uv run --no-sync python -m pytest tests_optional/test_gymnasium.py

ci-torchrl:
    vx uv sync --python 3.12.13 --frozen --all-groups --extra torchrl --no-install-project
    vx uv sync --python 3.12.13 --frozen --all-groups --extra torchrl --no-build-isolation
    vx uv run --no-sync python -m mypy src/game_learning_runtime/integrations/torch_objectives.py src/game_learning_runtime/integrations/torchrl.py
    vx uv run --no-sync python -m pytest tests_optional/test_torch_objectives.py tests_optional/test_torchrl.py

ci-package: setup workflow-check build

ci-runtime-host: setup lock-check rust-check provider-sdk-check
    vx uv run --no-sync python tools/ci/check_store_interop.py

# Windows counterpart of the Rust lane. It exercises the platform-specific
# branches that never execute on Linux: exclusive `share_mode(0)` opens and
# `FILE_FLAG_BACKUP_SEMANTICS` directory handles. Formatting and clippy stay
# with the Ubuntu lane: both are host independent, and running them twice only
# lengthens the signal.
ci-runtime-host-windows:
    vx cargo test --locked -p glr-cli -p glr-host

release-check tag:
    vx uv run python tools/release/verify_release.py {{tag}}
    vx just check
    vx just build
