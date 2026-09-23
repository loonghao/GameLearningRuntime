"""Keep every `Path.read_text()` / `Path.write_text()` call locale-independent.

Both default to `locale.getpreferredencoding(False)`, which is a code page such
as GBK on a Chinese Windows host. A test that reads its own fixture back then
fails with `UnicodeDecodeError` on those machines while staying green in UTF-8
CI, which hides the defect until someone runs the checks locally.

`encoding="utf-8"` is a one-word fix that is easy to forget, so the rule is
asserted here instead of being re-discovered on the next non-UTF-8 machine.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Checkout-only noise: vendored trees, build output and tool caches. Every
# virtualenv is skipped by prefix, not by name: `just ci-core` builds
# `.venv-glr`, a local bootstrap builds `.venv`, and a third-party tree is not
# this repository's contract to enforce.
SKIP_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pi",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)
SKIP_PREFIXES = (".venv",)

ROOT = Path(__file__).resolve().parents[1]
TEXT_METHODS = frozenset({"read_text", "write_text"})


def _is_checkout_source(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return not SKIP_DIRECTORIES.intersection(parts) and not any(
        part.startswith(SKIP_PREFIXES) for part in parts
    )


def _python_sources() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.py") if _is_checkout_source(path))


def _encoding_less_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_bytes())
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in TEXT_METHODS
        and not any(keyword.arg == "encoding" for keyword in node.keywords)
    ]


def test_read_text_and_write_text_declare_their_encoding() -> None:
    """An omitted `encoding` decodes whatever code page the host happens to use."""

    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{lineno}"
        for path in _python_sources()
        for lineno in _encoding_less_calls(path)
    ]
    assert offenders == []
