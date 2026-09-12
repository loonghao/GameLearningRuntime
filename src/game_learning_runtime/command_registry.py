"""Typed, reloadable command registry shared by game adapters."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str = ""
    requires_foreground: bool = False
    params: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("command name must be non-empty and trimmed")


class CommandRegistry:
    """Deterministic registry; replacement is atomic and generation tracked."""
    def __init__(self, specs: tuple[CommandSpec, ...] = ()) -> None:
        self._generation = 0
        self._specs: dict[str, CommandSpec] = {}
        self.replace(specs)

    @property
    def generation(self) -> int:
        return self._generation

    def replace(self, specs: tuple[CommandSpec, ...] | list[CommandSpec]) -> None:
        incoming = {s.name: s for s in specs}
        if len(incoming) != len(specs):
            raise ValueError("duplicate command name")
        self._specs = incoming
        self._generation += 1

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def capabilities(self) -> dict[str, Any]:
        commands = [self._specs[n].__dict__ for n in self.names()]
        payload = json.dumps(commands, sort_keys=True, separators=(",", ":"))
        return {"commands": commands, "generation": self._generation,
                "registry_sha256": sha256(payload.encode()).hexdigest()}

    def require(self, name: str) -> CommandSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unknown command: {name}") from exc
