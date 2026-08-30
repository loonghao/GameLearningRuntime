from __future__ import annotations

import numpy as np
import pytest

from game_learning_runtime import (
    CompositeSpec,
    ContractViolation,
    EnvironmentSpec,
    SpaceKind,
    TensorSpec,
)


def test_composite_spec_validates_hierarchical_action() -> None:
    spec = CompositeSpec(
        {
            "combat": CompositeSpec(
                {
                    "verb": TensorSpec(
                        (1,), np.int64, kind=SpaceKind.DISCRETE, minimum=0, maximum=3
                    ),
                    "direction": TensorSpec((2,), np.float32, minimum=-1.0, maximum=1.0),
                }
            )
        }
    )

    spec.validate(
        {
            "combat": {
                "verb": np.array([2], dtype=np.int64),
                "direction": np.array([0.25, -0.5], dtype=np.float32),
            }
        }
    )
    assert set(spec.flatten()) == {"combat.verb", "combat.direction"}


def test_tensor_spec_rejects_wrong_dtype_shape_and_bounds() -> None:
    spec = TensorSpec((2,), np.float32, minimum=-1.0, maximum=1.0)

    with pytest.raises(ContractViolation, match="dtype"):
        spec.validate(np.array([0, 1], dtype=np.int64))
    with pytest.raises(ContractViolation, match="shape"):
        spec.validate(np.array([0.0], dtype=np.float32))
    with pytest.raises(ContractViolation, match="above"):
        spec.validate(np.array([0.0, 2.0], dtype=np.float32))


def test_dynamic_dimension_accepts_different_lengths() -> None:
    spec = TensorSpec((None, 4), np.float32)

    spec.validate(np.zeros((3, 4), dtype=np.float32))
    spec.validate(np.zeros((9, 4), dtype=np.float32))
    assert spec.is_dynamic


def test_composite_spec_rejects_missing_or_extra_fields() -> None:
    spec = CompositeSpec({"state": TensorSpec((1,), np.int64)})

    with pytest.raises(ContractViolation, match=r"missing=.*state"):
        spec.validate({})
    with pytest.raises(ContractViolation, match=r"unexpected=.*other"):
        spec.validate(
            {
                "state": np.array([1], dtype=np.int64),
                "other": np.array([2], dtype=np.int64),
            }
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"shape": (-1,), "dtype": np.float32}, "non-negative"),
        (
            {"shape": (1,), "dtype": np.float32, "kind": SpaceKind.DISCRETE},
            "integer dtype",
        ),
        (
            {"shape": (1,), "dtype": np.int64, "kind": SpaceKind.BINARY},
            "bool dtype",
        ),
        ({"shape": (1,), "dtype": np.float32, "minimum": 2, "maximum": 1}, "exceed"),
    ],
)
def test_tensor_spec_rejects_invalid_declarations(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TensorSpec(**arguments)  # type: ignore[arg-type]


def test_composite_spec_rejects_invalid_declarations() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CompositeSpec({})
    with pytest.raises(ValueError, match="cannot contain dots"):
        CompositeSpec({"bad.path": TensorSpec((1,), np.float32)})
    with pytest.raises(TypeError, match="not a TensorSpec"):
        CompositeSpec({"bad": object()})  # type: ignore[dict-item]


def test_environment_spec_rejects_invalid_identity_and_signal_specs() -> None:
    tree = CompositeSpec({"value": TensorSpec((1,), np.float32)})
    with pytest.raises(ValueError, match="whitespace"):
        EnvironmentSpec("bad id", observation=tree, action=tree)
    with pytest.raises(ValueError, match="reward"):
        EnvironmentSpec(
            "bad-reward",
            observation=tree,
            action=tree,
            reward=TensorSpec((1,), np.int64, kind=SpaceKind.DISCRETE),
        )
    with pytest.raises(ValueError, match="done"):
        EnvironmentSpec(
            "bad-done",
            observation=tree,
            action=tree,
            done=TensorSpec((1,), np.float32),
        )
