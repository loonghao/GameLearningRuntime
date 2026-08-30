"""TorchRL ``EnvBase`` adapter for any GLR game environment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from game_learning_runtime.contracts import TensorTree, TimeStep
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.errors import OptionalDependencyError
from game_learning_runtime.specs import CompositeSpec, TensorSpec

try:
    import torch
    from tensordict import TensorDict, TensorDictBase
    from torchrl.data import Bounded, Categorical, Composite, Unbounded
    from torchrl.envs import EnvBase
except ImportError as error:  # pragma: no cover - exercised without the optional extra
    raise OptionalDependencyError(
        "TorchRL support requires `uv add game-learning-runtime[torchrl]`"
    ) from error


def _torch_dtype(dtype: np.dtype[Any]) -> torch.dtype:
    return torch.from_numpy(np.empty((), dtype=dtype)).dtype


def _leaf_spec(spec: TensorSpec, *, device: torch.device) -> Any:
    if spec.is_dynamic:
        raise ValueError("TorchRL EnvBase specs require static shapes")
    shape = tuple(int(dimension) for dimension in spec.shape if dimension is not None)
    dtype = _torch_dtype(np.dtype(spec.dtype))
    if spec.dtype == np.dtype(np.bool_):
        return Categorical(n=2, shape=shape, dtype=torch.bool, device=device)
    if spec.minimum is not None and spec.maximum is not None:
        return Bounded(
            low=torch.as_tensor(spec.minimum, dtype=dtype, device=device),
            high=torch.as_tensor(spec.maximum, dtype=dtype, device=device),
            shape=shape,
            dtype=dtype,
            device=device,
        )
    return Unbounded(shape=shape, dtype=dtype, device=device)


def _composite_spec(spec: CompositeSpec, *, device: torch.device) -> Composite:
    values = {
        key: _composite_spec(value, device=device)
        if isinstance(value, CompositeSpec)
        else _leaf_spec(value, device=device)
        for key, value in spec.fields.items()
    }
    return Composite(values, shape=(), device=device)


def _tree_to_tensordict(tree: TensorTree, *, device: torch.device) -> TensorDict:
    values = {
        key: _tree_to_tensordict(value, device=device)
        if isinstance(value, Mapping)
        else torch.as_tensor(np.array(value, copy=True), device=device)
        for key, value in tree.items()
    }
    return TensorDict(values, batch_size=(), device=device)


def _action_to_tree(tensordict: TensorDictBase, spec: CompositeSpec) -> TensorTree:
    result: dict[str, Any] = {}
    for key, child_spec in spec.fields.items():
        value = tensordict.get(key)
        if isinstance(child_spec, CompositeSpec):
            if not isinstance(value, TensorDictBase):
                raise TypeError(f"TorchRL action field {key!r} must be a TensorDict")
            result[key] = _action_to_tree(value, child_spec)
        else:
            result[key] = value.detach().cpu().numpy()
    return result


class TorchRLEnvironment(EnvBase):  # type: ignore[misc]
    """Adapt the learner-neutral GLR port to the current TorchRL environment API."""

    batch_locked = True

    def __init__(
        self,
        environment: GameEnvironment,
        *,
        device: str | torch.device = "cpu",
        run_type_checks: bool = True,
    ) -> None:
        self._glr = (
            environment
            if isinstance(environment, ContractEnvironment)
            else ContractEnvironment(environment)
        )
        self._next_seed: int | None = None
        super().__init__(device=device, batch_size=(), run_type_checks=run_type_checks)
        torch_device = torch.device(device)
        observation_values: dict[str, Any] = {
            "observation": _composite_spec(self._glr.spec.observation, device=torch_device)
        }
        if self._glr.spec.action_mask is not None:
            observation_values["action_mask"] = _composite_spec(
                self._glr.spec.action_mask, device=torch_device
            )
        self.observation_spec = Composite(observation_values, shape=(), device=torch_device)
        self.action_spec = _composite_spec(self._glr.spec.action, device=torch_device)
        self.reward_spec = _leaf_spec(self._glr.spec.reward, device=torch_device)
        done_leaf = _leaf_spec(self._glr.spec.done, device=torch_device)
        self.done_spec = Composite(
            done=done_leaf,
            terminated=done_leaf.clone(),
            truncated=done_leaf.clone(),
            shape=(),
            device=torch_device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = _action_to_tree(tensordict, self._glr.spec.action)
        return self._from_timestep(self._glr.step(action), include_reward=True)

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs: Any) -> TensorDictBase:
        del tensordict, kwargs
        timestep = self._glr.reset(seed=self._next_seed)
        self._next_seed = None
        return self._from_timestep(timestep, include_reward=False)

    def _set_seed(self, seed: int | None) -> None:
        self._next_seed = seed

    def _close(self) -> None:
        self._glr.close()

    def _from_timestep(self, timestep: TimeStep, *, include_reward: bool) -> TensorDict:
        device = self.device or torch.device("cpu")
        terminated = torch.as_tensor(np.array(timestep.terminated, copy=True), device=device)
        truncated = torch.as_tensor(np.array(timestep.truncated, copy=True), device=device)
        values: dict[str, Any] = {
            "observation": _tree_to_tensordict(timestep.observation, device=device),
            "done": torch.logical_or(terminated, truncated),
            "terminated": terminated,
            "truncated": truncated,
        }
        if timestep.action_mask is not None:
            values["action_mask"] = _tree_to_tensordict(timestep.action_mask, device=device)
        if include_reward:
            values["reward"] = torch.as_tensor(np.array(timestep.reward, copy=True), device=device)
        return TensorDict(values, batch_size=(), device=device)


__all__ = ["TorchRLEnvironment"]
