"""Declared per-role environment for ``glr.project.v1``.

A project manifest may declare the environment its roles receive:

.. code-block:: toml

    [environment]
    RENDER_DEVICE = "cpu"
    DATASET_ROOT = "${SYNTHETIC_DATASET_ROOT}"

    [trainer.environment]
    RENDER_DEVICE = "cuda"

Resolution is a pure function of the declaration and the process environment:

* a literal value is passed through unchanged;
* ``${NAME}`` is interpolated from the **process** environment;
* a reference to a variable the process environment does not define fails
  closed and names the offending key;
* a variable the process environment already defines wins over the declared
  table, so an operator override is never shadowed by the manifest.

The ``GLR_`` namespace belongs to the CLI, not to a project: the CLI clears
inherited ``GLR_*`` variables before it spawns a child and then publishes the
values it owns. A declared ``GLR_*`` key is therefore rejected while the
manifest is loaded, before any process starts.

This module owns the declaration, its validation, and its resolution. It
performs no I/O: callers supply the process environment to resolve against.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from game_learning_runtime.game_launcher import ENVIRONMENT_KEY

#: Namespace the CLI owns and republishes for every child it spawns.
RESERVED_ENVIRONMENT_PREFIX = "GLR_"

#: A process-environment reference inside a declared value.
ENVIRONMENT_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Any brace-delimited token, used to reject malformed references at load time.
_ANY_REFERENCE = re.compile(r"\$\{[^}]*\}")

#: Declared names that carry a credential, so their value is never recorded.
SECRET_NAME_PATTERN = re.compile(
    r"SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|(?:^|_)KEY(?:$|_)",
    re.IGNORECASE,
)

_VARIABLE_SOURCE = Literal["process", "interpolated", "literal"]


def is_secret_name(name: str) -> bool:
    """Report whether a declared name looks like it carries a credential.

    A conservative lexical test: the manifest does not mark secrets, so a name
    that reads like a credential is treated as one and its resolved value is
    kept out of run records and doctor output.
    """

    return SECRET_NAME_PATTERN.search(name) is not None


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} requires string keys")
    return value


def parse_environment_table(value: object, *, path: str) -> Mapping[str, str]:
    """Validate and freeze one declared ``environment`` table.

    Rejects non-string entries, malformed keys, keys inside the ``GLR_``
    namespace, and malformed ``${...}`` references. Whether a referenced
    variable exists is a property of the process environment, not of the
    manifest, so it is decided at resolution time instead.
    """

    if value is None:
        return MappingProxyType({})
    table = _mapping(value, path=path)
    parsed: dict[str, str] = {}
    for name, raw in table.items():
        if ENVIRONMENT_KEY.fullmatch(name) is None:
            raise ValueError(f"{path} keys must match {ENVIRONMENT_KEY.pattern!r}: {name!r}")
        if name.startswith(RESERVED_ENVIRONMENT_PREFIX):
            raise ValueError(
                f"{path} cannot declare {name!r}: the "
                f"{RESERVED_ENVIRONMENT_PREFIX}* namespace belongs to the CLI"
            )
        if not isinstance(raw, str) or any(ord(character) < 32 for character in raw):
            raise ValueError(f"{path}.{name} must be a printable string")
        for token in _ANY_REFERENCE.findall(raw):
            if ENVIRONMENT_REFERENCE.fullmatch(token) is None:
                raise ValueError(
                    f"{path}.{name} has a malformed reference {token!r}: "
                    "write ${NAME} with a valid variable name"
                )
        parsed[name] = raw
    return MappingProxyType(parsed)


def merge_declarations(
    project_table: Mapping[str, str] | None, role_table: Mapping[str, str] | None
) -> Mapping[str, str]:
    """Merge the project-wide table with one role's table; the role wins."""

    merged: dict[str, str] = dict(project_table or {})
    merged.update(role_table or {})
    return MappingProxyType(merged)


@dataclass(frozen=True, slots=True)
class ResolvedVariable:
    """One declared variable after resolution.

    ``source`` records where the value came from: ``process`` when the ambient
    environment already defined the name, ``interpolated`` when ``${NAME}`` was
    expanded from it, and ``literal`` when the manifest value passed through
    unchanged.
    """

    name: str
    value: str
    source: _VARIABLE_SOURCE
    secret: bool

    def to_mapping(self) -> dict[str, object]:
        """Report shape shared by ``doctor`` output and run records.

        The value is present only for non-secret names, so a credential that a
        role received is reported as received and never as content.
        """

        report: dict[str, object] = {"name": self.name, "source": self.source}
        if self.secret:
            report["secret"] = True
        else:
            report["secret"] = False
            report["value"] = self.value
        return report


@dataclass(frozen=True, slots=True)
class UnresolvedVariable:
    """A declared variable that could not be resolved, and why."""

    name: str
    missing: tuple[str, ...]

    @property
    def reason(self) -> str:
        names = ", ".join(f"${{{name}}}" for name in self.missing)
        return f"process environment defines no {names}"


@dataclass(frozen=True, slots=True)
class RoleEnvironment:
    """The declared environment one role receives after resolution."""

    role: str | None
    variables: tuple[ResolvedVariable, ...]
    unresolved: tuple[UnresolvedVariable, ...]

    @property
    def ready(self) -> bool:
        """True when every declared variable resolved."""

        return not self.unresolved

    @property
    def target(self) -> str:
        """Human-readable subject for a failure message."""

        return "the declared role environment" if self.role is None else f"role {self.role!r}"

    def refusal(self) -> str:
        """Fail-closed message naming the first variable that did not resolve."""

        first = self.unresolved[0]
        return f"{self.target} cannot resolve {first.name!r}: {first.reason}"

    def process_environment(self) -> dict[str, str]:
        """Resolved values, keyed by name. Callers apply process precedence."""

        return {variable.name: variable.value for variable in self.variables}

    def to_mapping(self) -> dict[str, object]:
        """Report shape shared by ``doctor`` output and run records."""

        return {
            "role": self.role,
            "ready": self.ready,
            "variables": [variable.to_mapping() for variable in self.variables],
            "unresolved": [
                {"name": item.name, "missing": list(item.missing)} for item in self.unresolved
            ],
        }


def resolve_environment(
    declared: Mapping[str, str],
    *,
    environ: Mapping[str, str],
    role: str | None = None,
) -> RoleEnvironment:
    """Resolve a declared table against a process environment.

    The real process environment outranks the declaration: a name it already
    defines keeps its own value, and a declared value that references a missing
    variable only fails when nothing else supplied that name.
    """

    variables: list[ResolvedVariable] = []
    unresolved: list[UnresolvedVariable] = []
    for name in sorted(declared):
        raw = declared[name]
        secret = is_secret_name(name)
        if name in environ:
            variables.append(ResolvedVariable(name, environ[name], "process", secret))
            continue
        references = tuple(dict.fromkeys(ENVIRONMENT_REFERENCE.findall(raw)))
        if not references:
            variables.append(ResolvedVariable(name, raw, "literal", secret))
            continue
        missing = tuple(reference for reference in references if reference not in environ)
        if missing:
            unresolved.append(UnresolvedVariable(name, missing))
            continue
        value = ENVIRONMENT_REFERENCE.sub(lambda match: environ[match.group(1)], raw)
        variables.append(ResolvedVariable(name, value, "interpolated", secret))
    return RoleEnvironment(
        role=role,
        variables=tuple(variables),
        unresolved=tuple(unresolved),
    )


__all__ = [
    "ENVIRONMENT_REFERENCE",
    "RESERVED_ENVIRONMENT_PREFIX",
    "SECRET_NAME_PATTERN",
    "ResolvedVariable",
    "RoleEnvironment",
    "UnresolvedVariable",
    "is_secret_name",
    "merge_declarations",
    "parse_environment_table",
    "resolve_environment",
]
