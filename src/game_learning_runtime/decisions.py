"""Learner-neutral choices and execution provenance for dynamic action sets.

Adapters enumerate feasible actions. Policies select their IDs. Executors may
reject a choice, but must never silently replace it with a scripted choice.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


@dataclass(frozen=True)
class Candidate:
    key: str
    command: str
    parameters_json: str = "{}"

    def __post_init__(self) -> None:
        _text(self.key, "candidate key")
        _text(self.command, "candidate command")
        _text(self.parameters_json, "candidate parameters")
        parameters = json.loads(self.parameters_json)
        if not isinstance(parameters, dict):
            raise ValueError("candidate parameters must be a JSON object")
        # Reject NaN/Infinity and canonicalize to make equality meaningful.
        object.__setattr__(
            self, "parameters_json", json.dumps(parameters, sort_keys=True, allow_nan=False)
        )

    @property
    def parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = json.loads(self.parameters_json)
        return parameters


@dataclass(frozen=True)
class Decision:
    state: str
    candidates: tuple[Candidate, ...]
    selected_key: str
    policy_digest: str
    mode: str

    def __post_init__(self) -> None:
        _text(self.state, "decision state")
        _text(self.selected_key, "selected key")
        _text(self.policy_digest, "policy digest")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(item, Candidate) for item in self.candidates
        ):
            raise ValueError("candidates must be an immutable tuple of Candidate values")
        keys = [item.key for item in self.candidates]
        if len(keys) != len(set(keys)) or self.selected_key not in keys:
            raise ValueError("policy must select one uniquely identified candidate")
        if not self.policy_digest or self.mode not in {"train", "evaluate"}:
            raise ValueError("policy digest and train/evaluate mode are required")

    @property
    def selected(self) -> Candidate:
        return next(item for item in self.candidates if item.key == self.selected_key)


def execute_decision(
    decision: Decision,
    execute: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Submit precisely the policy selection, retaining rejected receipts too."""
    selected = decision.selected
    receipt = dict(execute(selected.command, selected.parameters))
    return {
        "selected_key": selected.key,
        "command": selected.command,
        "parameters": selected.parameters,
        "policy_digest": decision.policy_digest,
        "mode": decision.mode,
        "candidate_count": len(decision.candidates),
        "receipt": receipt,
    }


def learning_status(
    *, transitions: int, updates: int, initial_digest: str, final_digest: str
) -> str:
    """A changed policy is evidence of an update, never of improvement."""
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (transitions, updates)
    ):
        raise ValueError("evidence counts must be integers")
    if transitions < 0 or updates < 0:
        raise ValueError("evidence counts cannot be negative")
    if not transitions:
        return "no_transitions"
    if not updates or not initial_digest or not final_digest:
        return "learning_unverified"
    if initial_digest == final_digest:
        return "policy_unchanged"
    return "policy_changed_improvement_unverified"
