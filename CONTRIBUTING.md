# Contributing

Thanks for helping make game-learning infrastructure reusable.

## Development

Prerequisites: Git and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/loonghao/GameLearningRuntime.git
cd GameLearningRuntime
uv sync --frozen --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not torchrl" --cov=game_learning_runtime --cov-report=term-missing
uv build
uv run twine check dist/*
```

Optional integration contracts:

```powershell
uv sync --frozen --all-groups --extra torchrl
uv run --extra torchrl pytest tests_optional/test_torchrl.py
uv sync --frozen --all-groups --extra gymnasium
uv run --extra gymnasium pytest tests_optional/test_gymnasium.py
```

## Change contract

- Keep game adapters independent from learning algorithms.
- Add or update an ADR when changing a public boundary or wire format.
- Treat protocol and dataset schemas as versioned compatibility contracts.
- Add adversarial tests for lifecycle, shapes, dtypes, bounds, masks, and stale
  episode/step identity.
- Use Conventional Commits in English.
- Deny incidental metadata by default; never commit local paths, hostnames,
  process/window identifiers, credentials, or private runtime data.
- Do not add game instrumentation unless it is legal, authorized, and isolated
  behind an adapter.

Open a pull request only after local checks pass. A maintainer review and green
required checks are necessary before merge.
