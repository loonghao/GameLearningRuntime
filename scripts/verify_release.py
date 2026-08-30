"""Verify that a release tag exactly matches the package version."""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib


def main(tag: str) -> int:
    if not tag.startswith("v"):
        raise SystemExit(f"release tag must start with v: {tag!r}")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    if tag[1:] != version:
        raise SystemExit(f"tag {tag!r} does not match project version {version!r}")
    print(f"release version verified: {version}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_release.py <tag>")
    raise SystemExit(main(sys.argv[1]))
