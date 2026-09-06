"""Standard, provider-neutral GLR lifecycle command vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class ControlCommand(StrEnum):
    TRAIN = "train"
    STATUS = "status"
    FEEDBACK = "feedback"
    REFLECT = "reflect"
    STOP = "stop"
    RESTART = "restart"


SAFE_COMMANDS: Final[frozenset[ControlCommand]] = frozenset(ControlCommand)


@dataclass(frozen=True, slots=True)
class GLRCommand:
    """A lifecycle request that must be routed through a GLR service."""

    command: ControlCommand
    run_id: str | None = None
    contract: str = "glr-timestep-v1"

    def __post_init__(self) -> None:
        if self.command not in SAFE_COMMANDS:
            raise ValueError("unsupported GLR lifecycle command")
        if self.command in {ControlCommand.STATUS, ControlCommand.FEEDBACK,
                            ControlCommand.REFLECT, ControlCommand.STOP} and not self.run_id:
            raise ValueError(f"{self.command.value} requires run_id")
        if self.contract != "glr-timestep-v1":
            raise ValueError("unsupported GLR training contract")

    def argv(self) -> tuple[str, ...]:
        args = ("glr", self.command.value)
        return args + (("--run-id", self.run_id) if self.run_id else ())


def command(command: str, *, run_id: str | None = None) -> GLRCommand:
    """Parse a CLI lifecycle command without granting provider-side control."""
    try:
        parsed = ControlCommand(command)
    except ValueError as exc:
        raise ValueError(f"unknown GLR command: {command}") from exc
    return GLRCommand(parsed, run_id=run_id)


__all__ = ["ControlCommand", "GLRCommand", "SAFE_COMMANDS", "command"]
