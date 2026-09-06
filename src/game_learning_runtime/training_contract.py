"""Fail-closed boundaries for learner-facing GLR data.

Adapters may expose diagnostics in ``info``, but collectors must derive
episode boundaries and rewards from :class:`~game_learning_runtime.contracts.TimeStep`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .contracts import TimeStep, Transition


class TrainingContractError(ValueError):
    """Raised when a learner-facing sample violates the GLR contract."""


def validate_timestep(timestep: TimeStep) -> TimeStep:
    """Validate authoritative lifecycle signals before a sample is recorded."""
    if not isinstance(timestep, TimeStep):
        raise TypeError("training input must be a GLR TimeStep")
    if np.any(np.logical_and(timestep.terminated, timestep.truncated)):
        raise TrainingContractError("terminated and truncated cannot both be true")
    if timestep.done and timestep.info.get("strategy_outcome") in {"failed", "error"}:
        # A provider failure is a truncation boundary, never a terminal loss.
        if not np.any(timestep.truncated):
            raise TrainingContractError("failed infrastructure outcome must be truncated")
    return timestep


def transition_provenance(timestep: TimeStep, *, segment: int = 0) -> dict[str, Any]:
    """Return stable provenance copied from a GLR timestep.

    ``run_id`` is optional for synthetic environments, while episode and
    segment are always explicit so replay shards cannot silently join runs.
    """
    validate_timestep(timestep)
    if segment < 0:
        raise ValueError("segment cannot be negative")
    run_id = timestep.info.get("run_id")
    return {
        "source": "glr-timestep",
        "episode_id": str(timestep.episode_id),
        "step_id": timestep.step_id,
        "run_id": str(run_id) if run_id is not None else None,
        "segment": segment,
        "terminated": bool(np.all(timestep.terminated)),
        "truncated": bool(np.all(timestep.truncated)),
        "infrastructure_failure": bool(timestep.info.get("infrastructure_failure", False)),
    }


def assert_transition_provenance(transition: Transition) -> Transition:
    """Reject replay rows that are not explicitly bound to GLR identity."""
    if not isinstance(transition, Transition):
        raise TypeError("replay input must be a GLR Transition")
    provenance: Mapping[str, Any] = transition.provenance or {}
    if provenance.get("source") != "glr-timestep":
        raise TrainingContractError("transition provenance must be glr-timestep")
    if str(provenance.get("episode_id")) != str(transition.episode_id):
        raise TrainingContractError("transition episode_id does not match provenance")
    if int(provenance.get("step_id", -1)) != transition.step_id:
        raise TrainingContractError("transition step_id does not match provenance")
    if bool(provenance.get("terminated")) and bool(provenance.get("truncated")):
        raise TrainingContractError("provenance cannot terminate and truncate together")
    return transition


__all__ = ["TrainingContractError", "assert_transition_provenance", "transition_provenance", "validate_timestep"]
