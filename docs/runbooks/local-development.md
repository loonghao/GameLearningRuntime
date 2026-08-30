# Local development runbook

## Prerequisites

- Git
- uv 0.11 or newer
- Python 3.10 through 3.13

## Steps

```powershell
uv sync --frozen --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not torchrl" --cov=game_learning_runtime --cov-report=term-missing
uv build
uv run twine check dist/*
```

For the optional integration:

```powershell
uv sync --frozen --all-groups --extra torchrl
uv run --extra torchrl pytest tests_optional -m torchrl
```

## Verify

All commands must exit zero. Confirm imports originate from this checkout when
debugging environment drift:

```powershell
uv run python -c "import game_learning_runtime as g; print(g.__file__)"
```

## Troubleshooting

| Problem | Resolution |
|---|---|
| Lock file changed unexpectedly | Run `uv lock --check`; inspect dependency inputs before accepting a relock |
| Torch wheel is large | Run the core suite without the `torchrl` extra; use the dedicated integration job for TorchRL |
| Contract violation | Compare the failing path, dtype, shape, and declared bounds; do not cast silently |
| Protocol test fails | Compile the packaged `runtime.proto`; do not test an unshipped duplicate |
