"""Reusable conformance helpers for authorized game adapter test suites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from game_learning_runtime.collector import Policy, SyncCollector
from game_learning_runtime.environment import ContractEnvironment, GameEnvironment
from game_learning_runtime.errors import ContractViolation


@dataclass(frozen=True, slots=True)
class EnvironmentConformanceReport:
    """Privacy-safe aggregate evidence from one bounded adapter trace."""

    transition_count: int
    episode_count: int
    masked_transition_count: int
    event_count: int
    terminated_transition_count: int
    truncated_transition_count: int


def run_environment_conformance(
    environment: GameEnvironment,
    policy: Policy,
    *,
    steps: int,
    seed: int | None = None,
    start_mode: Literal["reset", "attach"] = "reset",
) -> EnvironmentConformanceReport:
    """Exercise an adapter through the GLR contract and return aggregate evidence.

    The report intentionally excludes environment IDs, metadata, actions,
    observations, paths, and timestamps so it can be attached to CI safely.
    The environment is closed whether collection succeeds or fails.
    """

    contract = (
        environment
        if isinstance(environment, ContractEnvironment)
        else ContractEnvironment(environment)
    )
    try:
        if steps <= 0:
            raise ValueError("steps must be positive")
        unroll = SyncCollector(
            contract,
            actor_id="conformance",
            start_mode=start_mode,
        ).collect(
            policy,
            steps=steps,
            seed=seed,
        )
        for transition in unroll.transitions:
            if np.logical_and(transition.terminated, transition.truncated).any():
                raise ContractViolation(
                    "a transition cannot be both terminated and truncated for the same participant"
                )
        return EnvironmentConformanceReport(
            transition_count=len(unroll.transitions),
            episode_count=len({transition.episode_id for transition in unroll.transitions}),
            masked_transition_count=sum(
                transition.action_mask is not None for transition in unroll.transitions
            ),
            event_count=sum(len(transition.events) for transition in unroll.transitions),
            terminated_transition_count=sum(
                bool(np.any(transition.terminated)) for transition in unroll.transitions
            ),
            truncated_transition_count=sum(
                bool(np.any(transition.truncated)) for transition in unroll.transitions
            ),
        )
    finally:
        contract.close()


__all__ = ["EnvironmentConformanceReport", "run_environment_conformance"]
