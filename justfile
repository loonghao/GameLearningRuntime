set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
export UV_PROJECT_ENVIRONMENT := ".venv-glr"

default: check

# Show the exact checkout backing the active project environment.
origin:
    vx uv run python -c "import game_learning_runtime as glr; print(glr.__file__)"

# Install the locked project dependencies into the project-owned environment.
setup:
    vx uv sync --python 3.12.13 --frozen --all-groups

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

check: setup lock-check workflow-check core-check

build:
    vx uv build
    vx uv run python scripts/check_dist.py

# Local pre-push equivalent of the core CI and package gates.
ci: check build

# GitHub Actions matrix lane. Each job owns its environment, so the selected
# interpreter can safely replace the baseline Python from vx.toml.
ci-core python_version:
    vx uv lock --check
    vx uv run --frozen --all-groups --python {{python_version}} python scripts/run_core_checks.py

ci-gymnasium:
    vx uv sync --python 3.12.13 --frozen --all-groups --extra gymnasium
    vx uv run --extra gymnasium python -m mypy src/game_learning_runtime/integrations/gymnasium.py
    vx uv run --extra gymnasium python -m pytest tests_optional/test_gymnasium.py

ci-torchrl:
    vx uv sync --python 3.12.13 --frozen --all-groups --extra torchrl
    vx uv run --extra torchrl python -m mypy src/game_learning_runtime/integrations/torch_objectives.py src/game_learning_runtime/integrations/torchrl.py
    vx uv run --extra torchrl python -m pytest tests_optional/test_torch_objectives.py tests_optional/test_torchrl.py

ci-package: setup workflow-check build

release-check tag:
    vx uv run python scripts/verify_release.py {{tag}}
    vx just check
    vx just build
