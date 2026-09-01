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
    vx uv run python scripts/run_core_checks.py

rust-format-check:
    vx cargo fmt --all -- --check

rust-clippy:
    vx cargo clippy --workspace --all-targets --locked -- -D warnings

rust-test:
    vx cargo test --workspace --locked

rust-build:
    vx cargo build --workspace --bins --locked

host-smoke: rust-build
    vx uv run python scripts/run_host_smoke.py

rust-check: rust-format-check rust-clippy rust-test host-smoke

csharp-check:
    vx dotnet build sdk/csharp/GameLearningRuntime.Provider.Smoke/GameLearningRuntime.Provider.Smoke.csproj --configuration Release
    vx dotnet run --project sdk/csharp/GameLearningRuntime.Provider.Smoke/GameLearningRuntime.Provider.Smoke.csproj --configuration Release --no-build

cpp-check:
    vx uv run python scripts/check_cpp_provider.py

provider-sdk-check: csharp-check cpp-check

check: setup lock-check workflow-check core-check rust-check provider-sdk-check

build:
    vx cargo build --release --workspace --bins --locked
    vx uv run --no-sync python -m build --no-isolation
    vx uv run python scripts/check_dist.py

# Local pre-push equivalent of the core CI and package gates.
ci: check build

# GitHub Actions matrix lane. Each job owns its environment, so the selected
# interpreter can safely replace the baseline Python from vx.toml.
ci-core python_version:
    vx uv lock --check
    vx uv sync --python {{python_version}} --frozen --all-groups --no-install-project
    vx uv sync --python {{python_version}} --frozen --all-groups --no-build-isolation
    vx uv run --no-sync python scripts/run_core_checks.py

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

release-check tag:
    vx uv run python scripts/verify_release.py {{tag}}
    vx just check
    vx just build
