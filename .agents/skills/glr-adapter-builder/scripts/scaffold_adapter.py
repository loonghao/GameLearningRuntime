#!/usr/bin/env python3
"""Create a safe, synthetic-first GLR adapter development lane."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SKILL_ROOT = Path(__file__).resolve().parents[1]
_ASSETS = _SKILL_ROOT / "assets"
_PACKAGE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENVIRONMENT_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _load_asset(name: str) -> dict[str, Any]:
    with (_ASSETS / name).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"asset {name} must contain a JSON object")
    return value


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _environment_module(start_mode: str) -> str:
    if start_mode == "reset":
        return '''"""Synthetic seam; replace its semantics with the authorized adapter."""

from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.examples import CounterEnvironment, always_increment
from game_learning_runtime.environment import GameEnvironment


def create_environment() -> GameEnvironment:
    """Return a trainable synthetic environment for the initial green baseline."""

    return CounterEnvironment(target=3, max_steps=5)


def synthetic_policy(timestep: TimeStep) -> TensorTree:
    return always_increment(timestep)


__all__ = ["create_environment", "synthetic_policy"]
'''

    return '''"""Synthetic live-attach seam; it makes no physical reset claim."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.environment import GameEnvironment
from game_learning_runtime.examples import CounterEnvironment, always_increment
from game_learning_runtime.errors import ContractViolation
from game_learning_runtime.specs import EnvironmentSpec


class SyntheticAttachEnvironment(CounterEnvironment):
    """Runnable attach baseline to replace with an authorized live adapter."""

    def __init__(self) -> None:
        super().__init__(target=3, max_steps=5)
        source = self._spec
        self._spec = EnvironmentSpec(
            environment_id=source.environment_id,
            observation=source.observation,
            action=source.action,
            reward=source.reward,
            done=source.done,
            action_mask=source.action_mask,
            protocol_version=source.protocol_version,
            capabilities=source.capabilities | {"live-attach"},
            metadata=source.metadata,
        )

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> TimeStep:
        del seed, options
        raise ContractViolation("physical reset is unavailable in attach mode")

    def attach(self, *, options: Mapping[str, Any] | None = None) -> TimeStep:
        return super().reset(options=options)


def create_environment() -> GameEnvironment:
    """Return a trainable synthetic live-attach baseline."""

    return SyntheticAttachEnvironment()


def synthetic_policy(timestep: TimeStep) -> TensorTree:
    return always_increment(timestep)


__all__ = ["SyntheticAttachEnvironment", "create_environment", "synthetic_policy"]
'''


def _test_module(package: str) -> str:
    return f"""from pathlib import Path

from game_learning_runtime import RewardComposer, RewardSignal, load_training_config
from game_learning_runtime.testing import run_environment_conformance

from {package}.environment import create_environment, synthetic_policy


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_contract_is_trainable() -> None:
    config = load_training_config(ROOT / "training.json")
    report = run_environment_conformance(
        create_environment(),
        synthetic_policy,
        steps=3,
        start_mode=config.lifecycle.start_mode,
    )
    assert report.transition_count == 3


def test_reward_configuration_is_executable_data_not_an_expression() -> None:
    config = load_training_config(ROOT / "training.json")
    result = RewardComposer(config).compose(
        [RewardSignal(name="progress", source="runtime", value=0.25)]
    )
    assert result.total == 0.25
"""


def _pyproject(package: str) -> str:
    project = package.replace("_", "-")
    return f'''[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "{project}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "game-learning-runtime>=0.2,<0.3",
]

[dependency-groups]
dev = ["pytest>=8.3"]

[tool.hatch.build.targets.wheel]
packages = ["src/{package}"]

[tool.pytest.ini_options]
testpaths = ["tests"]
'''


def _readme(package: str, environment_id: str, start_mode: str) -> str:
    return f"""# {package}

GLR adapter development lane for `{environment_id}` using `{start_mode}` start
semantics.

The initial environment is deliberately synthetic and trainable. Replace it
with an authorized runtime adapter while preserving the tests. Do not publish
local endpoints, paths, process/window identifiers, credentials, observations,
actions, or proprietary game data.

Research public gameplay sources into `knowledge/research-manifest.json` as
compact paraphrased claims with provenance. Guide research remains advisory;
reward authority comes from verified runtime signals in `training.json`.

```powershell
uv sync --all-groups
uv run pytest
```
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new or empty output directory")
    parser.add_argument("--package", required=True, help="lowercase Python package name")
    parser.add_argument("--environment-id", required=True, help="generic public ID")
    parser.add_argument("--start-mode", choices=("reset", "attach"), default="reset")
    args = parser.parse_args()
    if _PACKAGE.fullmatch(args.package) is None:
        parser.error("--package must match ^[a-z][a-z0-9_]*$")
    if _ENVIRONMENT_ID.fullmatch(args.environment_id) is None:
        parser.error("--environment-id must be a generic identifier without paths or endpoints")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is non-empty and will not be overwritten: {output}")
    args.output = output
    return args


def main() -> int:
    args = _parse_args()
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    training = _load_asset("training-config.json")
    training["lifecycle"]["start_mode"] = args.start_mode
    if args.start_mode == "attach":
        training["bridge"]["required_capabilities"] = [
            "authenticated",
            "live-attach",
            "postcondition-verified",
            "target-bound",
        ]
    research = _load_asset("knowledge-research.json")
    research["environment_id"] = args.environment_id
    research["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    _write(output / "training.json", json.dumps(training, indent=2) + "\n")
    _write(
        output / "knowledge/research-manifest.json",
        json.dumps(research, indent=2) + "\n",
    )
    _write(output / f"src/{args.package}/__init__.py", "")
    _write(
        output / f"src/{args.package}/environment.py",
        _environment_module(args.start_mode),
    )
    _write(output / "tests/test_adapter_contract.py", _test_module(args.package))
    _write(output / "pyproject.toml", _pyproject(args.package))
    _write(
        output / "README.md",
        _readme(args.package, args.environment_id, args.start_mode),
    )
    print(f"Created synthetic-first GLR adapter lane at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
