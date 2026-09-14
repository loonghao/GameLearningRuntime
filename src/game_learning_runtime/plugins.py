"""Safe, declarative GLR plugin bundles and profile composition.

The plugin layer is intentionally a control-plane contract.  Inspecting,
installing, and resolving a plugin never imports its entrypoint, runs a build
hook, starts a process, or grants runtime authority.  A host-specific runner
may consume :class:`ResolvedProfile` after it has applied its own trust and
permission policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType
from typing import Any, cast

from game_learning_runtime.errors import GLRError

PLUGIN_SCHEMA_VERSION = "glr.plugin.v1"
PROFILE_SCHEMA_VERSION = "glr.profile.v1"
PLUGIN_FILE_NAME = "glr-plugin.json"
PLUGIN_STORE_DIR = ".glr/plugins"
PROFILE_STORE_DIR = ".glr/profiles"

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_CAPABILITY = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?$"
)
_COMPARATOR = re.compile(r"^(==|=|>=|<=|>|<)\s*(.+)$")
_PARTIAL_VERSION = re.compile(r"^(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?(?:\.(0|[1-9][0-9]*))?$")
_WILDCARD_VERSION = re.compile(
    r"^(0|[1-9][0-9]*)(?:\.(\*|x|X|0|[1-9][0-9]*))?"
    r"(?:\.(\*|x|X|0|[1-9][0-9]*))?$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_PROFILE_BYTES = 256 * 1024
_MAX_FILES = 4096
_MAX_FILE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_PATH_DEPTH = 16
_MAX_SEMVER_COMPONENT = 2**64 - 1
_MIN_JSON_INTEGER = -(2**63)
_MAX_JSON_INTEGER = 2**64 - 1

_PLUGIN_KINDS = frozenset(
    {
        "environment",
        "learner",
        "recorder",
        "evaluator",
        "model-provider",
        "harness",
        "ui",
        "command",
    }
)
_PLATFORMS = frozenset({"windows", "linux", "macos"})
_ISOLATIONS = frozenset({"process", "in-process"})
_PERMISSIONS = frozenset(
    {
        "read:environment",
        "read:dataset",
        "read:artifact",
        "write:checkpoint",
        "write:dataset",
        "write:artifact",
        "spawn:worker",
        "network:outbound",
        "runtime:observe",
        "runtime:act",
        "events:emit",
        "ui:panel",
    }
)


class PluginError(GLRError):
    """Base error for plugin discovery, storage, and profile resolution."""


class PluginValidationError(PluginError):
    """Raised when a plugin or profile violates the versioned contract."""


class PluginTrustError(PluginError):
    """Raised when a caller tries to activate an untrusted plugin state."""


def _mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PluginValidationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise PluginValidationError(f"{path} requires string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, path: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PluginValidationError(f"{path} must be an array")
    return tuple(value)


def _strict_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    required: frozenset[str],
    path: str,
) -> None:
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - allowed)
    if missing or unexpected:
        raise PluginValidationError(
            f"{path} has missing={missing} and unexpected={unexpected} fields"
        )


def _text(value: object, *, path: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(
            unicodedata.category(character) == "Cc" or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    ):
        raise PluginValidationError(f"{path} must be non-empty printable text")
    return value


def _identifier(value: object, *, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise PluginValidationError(f"{path} must match {_IDENTIFIER.pattern!r}")
    return value


def _name(value: object, *, path: str) -> str:
    return _identifier(value, path=path)


def _set_of_strings(
    value: object,
    *,
    path: str,
    pattern: re.Pattern[str],
    allowed: frozenset[str] | None = None,
    minimum: int = 0,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PluginValidationError(f"{path} must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            raise PluginValidationError(f"{path} contains an invalid value")
        if allowed is not None and item not in allowed:
            raise PluginValidationError(f"{path} contains unsupported value: {item}")
        if item in result:
            raise PluginValidationError(f"{path} contains duplicate value: {item}")
        result.append(item)
    if len(result) < minimum:
        raise PluginValidationError(f"{path} must contain at least {minimum} value(s)")
    return tuple(result)


def _portable_path(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise PluginValidationError(f"{path} must be a portable relative path")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or candidate.as_posix() != value
        or len(candidate.parts) > _MAX_PATH_DEPTH
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or any(part.endswith((".", " ")) for part in candidate.parts)
        or any(not character.isascii() for character in value)
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise PluginValidationError(f"{path} must be a portable relative path")
    for part in candidate.parts:
        stem = part.split(".", 1)[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or (
            len(stem) == 4 and stem[:3] in {"COM", "LPT"} and stem[3].isdigit()
        ):
            raise PluginValidationError(f"{path} uses a reserved path component")
    return candidate.as_posix()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PluginValidationError("JSON object keys must be strings")
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode contract data with the same stable JSON form as the Rust CLI."""

    thawed = _thaw_json(value)
    _validate_rust_json_numbers(thawed)
    encoded = json.dumps(
        thawed,
        sort_keys=True,
        separators=(",", ":"),
        # serde_json emits UTF-8 for non-ASCII strings; retaining that form
        # keeps Python and Rust bundle/profile digests byte-identical.
        ensure_ascii=False,
        allow_nan=False,
    )
    # Python's encoder pads negative exponents (``1e-07``) and switches to
    # scientific notation at 1e-5, while serde_json/ryu emits ``1e-7`` and
    # ``0.00001`` respectively. Normalize those two notation differences
    # outside JSON strings so profile/plugin digests remain byte-identical.
    try:
        return _normalize_rust_number_tokens(encoded).encode("utf-8")
    except UnicodeEncodeError as error:
        raise PluginValidationError("JSON text must contain valid UTF-8 characters") from error


def _validate_rust_json_numbers(value: Any) -> None:
    """Reject numeric values serde_json cannot represent as a JSON ``Value``."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        if not _MIN_JSON_INTEGER <= value <= _MAX_JSON_INTEGER:
            raise PluginValidationError("JSON integer is outside Rust serde_json bounds")
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise PluginValidationError("JSON number must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PluginValidationError("JSON object keys must be strings")
            _validate_rust_json_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_rust_json_numbers(item)


_JSON_NUMBER_TOKEN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def _normalize_rust_number_tokens(encoded: str) -> str:
    """Apply serde_json/ryu's exponent spelling to an encoded JSON string."""

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(encoded):
        character = encoded[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        match = _JSON_NUMBER_TOKEN.match(encoded, index)
        if match is None:
            output.append(character)
            index += 1
            continue
        token = match.group(0)
        exponent_match = re.search(r"[eE]([+-]?)([0-9]+)$", token)
        if exponent_match is not None:
            sign, digits = exponent_match.groups()
            exponent = int(f"-{digits}" if sign == "-" else digits)
            if exponent == -5:
                from decimal import Decimal

                token = format(Decimal(token), "f")
            else:
                token = token[: exponent_match.start()] + "e" + sign + str(int(digits))
        output.append(token)
        index = match.end()
    return "".join(output)


def _json_object(value: Mapping[str, Any], *, path: str, maximum: int) -> Mapping[str, Any]:
    try:
        encoded_bytes = _canonical_json_bytes(value)
    except PluginValidationError as error:
        raise PluginValidationError(f"{path} must be JSON-serializable") from error
    except (TypeError, ValueError, UnicodeError) as error:
        raise PluginValidationError(f"{path} must be JSON-serializable") from error
    if len(encoded_bytes) > maximum:
        raise PluginValidationError(f"{path} exceeds the size limit")
    return cast(Mapping[str, Any], _freeze_json(json.loads(encoded_bytes)))


def _parse_semver(value: object, *, path: str) -> tuple[int, int, int, str]:
    if not isinstance(value, str):
        raise PluginValidationError(f"{path} must be a semantic version")
    match = _SEMVER.fullmatch(value)
    if match is None:
        raise PluginValidationError(f"{path} must be a semantic version")
    major, minor, patch, prerelease, build = match.groups()
    if prerelease:
        identifiers = prerelease.split(".")
        if any(not identifier for identifier in identifiers):
            raise PluginValidationError(f"{path} must be a semantic version")
        if any(
            identifier.isdigit() and len(identifier) > 1 and identifier.startswith("0")
            for identifier in identifiers
        ):
            raise PluginValidationError(f"{path} must be a semantic version")
    if build and any(not identifier for identifier in build.split(".")):
        raise PluginValidationError(f"{path} must be a semantic version")
    components = (int(major), int(minor), int(patch))
    if any(component > _MAX_SEMVER_COMPONENT for component in components):
        raise PluginValidationError(f"{path} must be a semantic version")
    return *components, prerelease or ""


def _version_key(value: str) -> tuple[int, int, int, tuple[tuple[int, int, str], ...]]:
    major, minor, patch, prerelease = _parse_semver(value, path="version")
    # Tag numeric and textual identifiers separately so Python never compares
    # unlike ``int``/``str`` values while ordering prereleases.
    parts: list[tuple[int, int, str]] = []
    if prerelease:
        for item in prerelease.split("."):
            if item.isdigit():
                parts.append((0, int(item), ""))
            else:
                parts.append((1, 0, item))
    else:
        parts.append((2, 0, ""))
    return major, minor, patch, tuple(parts)


def _version_order_key(value: str) -> tuple[object, ...]:
    """Order versions like Rust's ``semver::Version`` (including build tags).

    Build metadata is ignored by requirement matching, but the Rust resolver's
    candidate sort includes it in ``Version::Ord``.  Keep that tie-breaker out
    of :func:`_version_key`, which is also used for ``VersionReq`` evaluation.
    """

    major, minor, patch, prerelease = _parse_semver(value, path="version")
    pre_parts: list[tuple[int, int, str]] = []
    if prerelease:
        for item in prerelease.split("."):
            pre_parts.append((0, int(item), "") if item.isdigit() else (1, 0, item))
    else:
        pre_parts.append((2, 0, ""))
    build = value.split("+", 1)[1] if "+" in value else ""
    build_parts: list[tuple[int, int, str, int]] = []
    for item in build.split(".") if build else ():
        if item.isdigit():
            stripped = item.lstrip("0") or "0"
            build_parts.append((0, len(stripped), stripped, len(item)))
        else:
            build_parts.append((1, 0, item, 0))
    return major, minor, patch, tuple(pre_parts), tuple(build_parts)


_INFINITE_VERSION_KEY: tuple[int, int, int, tuple[tuple[int, int, str], ...]] = (
    _MAX_SEMVER_COMPONENT + 1,
    0,
    0,
    ((2, 0, ""),),
)


def _bound_key(value: str) -> tuple[int, int, int, tuple[tuple[int, int, str], ...]]:
    """Return a comparison key, treating internal bounds above u64::MAX as infinity."""

    try:
        return _version_key(value)
    except PluginValidationError:
        match = _SEMVER.fullmatch(value)
        if match is not None and any(
            int(component) > _MAX_SEMVER_COMPONENT for component in match.groups()[:3]
        ):
            return _INFINITE_VERSION_KEY
        raise


def _validate_requirement(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise PluginValidationError(f"{path} must be a bounded version requirement")
    try:
        value_bytes = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PluginValidationError(f"{path} must be valid Unicode text") from error
    if len(value_bytes) > 256 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise PluginValidationError(f"{path} must be a bounded version requirement")
    if value.strip() in {"*", "x", "X"}:
        return "*"
    parts = [item.strip() for item in value.split(",")]
    if any(not item for item in parts) or any(item in {"*", "x", "X"} for item in parts):
        raise PluginValidationError(f"{path} contains an invalid wildcard comparator")
    if len(parts) > 32:
        raise PluginValidationError(f"{path} contains too many comparators")
    canonical: list[str] = []
    for item in parts:
        match = re.fullmatch(r"^(==|=|>=|<=|>|<|\^|~)?\s*(.+)$", item)
        if match is None:
            raise PluginValidationError(f"{path} contains an invalid comparator")
        operator, operand = match.groups()
        operand = operand.strip()
        wildcard = _wildcard_parts(operand, path=path)
        if wildcard is not None:
            major, minor, supplied = wildcard
            if operator is None:
                canonical.append(f"{major}.*" if supplied == 1 else f"{major}.{minor}.*")
            else:
                # rust-semver accepts operators on wildcard requirements and
                # canonicalizes the wildcard to its partial version (e.g.
                # ">=1.*" becomes ">=1").
                if operator == "==":
                    operator = "="
                rendered = f"{major}" if supplied == 1 else f"{major}.{minor}"
                canonical.append(f"{operator}{rendered}")
            continue
        full_version = _SEMVER.fullmatch(operand)
        if full_version is not None:
            _parse_semver(operand, path=path)
            major_s, minor_s, patch_s, _pre, _build = full_version.groups()
            partial = (int(major_s), int(minor_s), int(patch_s), 3)
        else:
            partial = _parse_partial_version(operand, path=path)
        if operator is None:
            operator = "^"
        elif operator == "==":
            operator = "="
        major, minor, patch, supplied = partial
        rendered = f"{major}"
        if supplied >= 2:
            rendered += f".{minor}"
        if supplied >= 3:
            rendered += f".{patch}"
            # A full version may carry prerelease/build metadata; partial
            # parsing above intentionally excludes those, so parse it as a
            # complete semantic version when all three components are given.
            if full_version is not None:
                # Build metadata is ignored by VersionReq and omitted by its
                # canonical Display implementation.
                rendered = f"{major}.{minor}.{patch}"
                if _pre:
                    rendered += f"-{_pre}"
        canonical.append(f"{operator}{rendered}")
    return ", ".join(canonical)


def _wildcard_parts(value: str, *, path: str) -> tuple[int, int, int] | None:
    """Return ``(major, minor, supplied)`` for a wildcard requirement."""

    match = _WILDCARD_VERSION.fullmatch(value)
    if match is None or not any(marker in value for marker in ("*", "x", "X")):
        return None
    major, minor, patch = match.groups()
    major_value = int(major)
    if major_value > _MAX_SEMVER_COMPONENT:
        raise PluginValidationError(f"{path} contains an out-of-range version")
    if patch is not None and patch in {"*", "x", "X"}:
        if minor in {"*", "x", "X"}:
            return major_value, 0, 1
        if minor not in {"*", "x", "X"}:
            minor_value = int(minor)
            if minor_value > _MAX_SEMVER_COMPONENT:
                raise PluginValidationError(f"{path} contains an out-of-range version")
            return major_value, minor_value, 2
        raise PluginValidationError(f"{path} contains an invalid wildcard comparator")
    if patch is not None:
        raise PluginValidationError(f"{path} contains an invalid wildcard comparator")
    if minor in {"*", "x", "X"}:
        return major_value, 0, 1
    raise PluginValidationError(f"{path} contains an invalid wildcard comparator")


def _parse_partial_version(value: str, *, path: str) -> tuple[int, int, int, int]:
    match = _PARTIAL_VERSION.fullmatch(value)
    if match is None:
        raise PluginValidationError(f"{path} must be a semantic version requirement")
    major, minor, patch = match.groups()
    supplied = 1 + (minor is not None) + (patch is not None)
    components = (int(major), int(minor or 0), int(patch or 0))
    if any(component > _MAX_SEMVER_COMPONENT for component in components):
        raise PluginValidationError(f"{path} must be a semantic version requirement")
    return *components, supplied


def _range_bounds(operator: str, value: str, *, path: str) -> tuple[str, str]:
    full = _SEMVER.fullmatch(value)
    if full is not None:
        major, minor, patch, prerelease, _build = full.groups()
        major, minor, patch = int(major), int(minor), int(patch)
        supplied = 3
        lower = f"{major}.{minor}.{patch}"
        if prerelease:
            lower += f"-{prerelease}"
    else:
        major, minor, patch, supplied = _parse_partial_version(value, path=path)
        lower = f"{major}.{minor}.{patch}"
    if operator == "~":
        upper = f"{major + 1}.0.0" if supplied == 1 else f"{major}.{minor + 1}.0"
    elif major > 0:
        upper = f"{major + 1}.0.0"
    elif supplied == 1:
        # Rust's semver caret semantics treat ^0 as any 0.x release.
        upper = "1.0.0"
    elif supplied == 2 and minor == 0:
        # Likewise, ^0.0 means any patch release in the 0.0 series.
        upper = "0.1.0"
    elif minor > 0:
        upper = f"0.{minor + 1}.0"
    else:
        upper = f"0.0.{patch + 1}"
    return lower, upper


def _satisfies(version: str, requirement: str) -> bool:
    if requirement.strip() in {"*", "x", "X"}:
        return not bool(_parse_semver(version, path="version")[3])
    candidate = _version_key(version)
    candidate_major, candidate_minor, candidate_patch, candidate_pre = _parse_semver(
        version, path="version"
    )
    if candidate_pre:
        # SemVer requirements match prereleases only when a comparator anchors
        # the same major/minor/patch tuple with an explicit prerelease.
        anchored = False
        for raw_item in requirement.split(","):
            item = raw_item.strip()
            item = re.sub(r"^(?:==|=|>=|<=|>|<|\^|~)\s*", "", item)
            try:
                major, minor, patch, prerelease = _parse_semver(item, path="requirement")
            except PluginValidationError:
                continue
            if prerelease and (major, minor, patch) == (
                candidate_major,
                candidate_minor,
                candidate_patch,
            ):
                anchored = True
                break
        if not anchored:
            return False
    for item in requirement.split(","):
        stripped = item.strip()
        # rust-semver permits comparator operators on wildcard operands (for
        # example, ``>=1.*``). Normalize those to equivalent partial
        # comparators before the regular comparator handling below.
        wildcard_operator: str | None = None
        wildcard_operand = stripped
        operator_match = re.fullmatch(r"^(==|=|>=|<=|>|<|\^|~)\s*(.+)$", stripped)
        if operator_match is not None:
            wildcard_operator, wildcard_operand = operator_match.groups()
        wildcard = _wildcard_parts(wildcard_operand, path="requirement")
        if wildcard is not None:
            major, minor, supplied = wildcard
            if wildcard_operator is None:
                lower_key = _version_key(f"{major}.{minor if supplied == 2 else 0}.0")
                upper_key = _bound_key(
                    f"{major}.{minor + 1}.0" if supplied == 2 else f"{major + 1}.0.0"
                )
                if not (lower_key <= candidate < upper_key):
                    return False
                continue
            wildcard_operator = "=" if wildcard_operator == "==" else wildcard_operator
            wildcard_partial = f"{major}" if supplied == 1 else f"{major}.{minor}"
            if wildcard_operator in {"^", "~"}:
                lower, upper = _range_bounds(
                    wildcard_operator, wildcard_partial, path="requirement"
                )
                if not (_version_key(lower) <= candidate < _bound_key(upper)):
                    return False
                continue
            # Ordinary operators use the existing partial comparator logic.
            stripped = f"{wildcard_operator}{wildcard_partial}"
        if stripped.startswith(("^", "~")):
            lower, upper = _range_bounds(stripped[0], stripped[1:].strip(), path="requirement")
            if not (_version_key(lower) <= candidate < _bound_key(upper)):
                return False
            continue
        match = _COMPARATOR.fullmatch(stripped)
        if match is None:
            lower, upper = _range_bounds("^", stripped, path="requirement")
            if not (_version_key(lower) <= candidate < _bound_key(upper)):
                return False
            continue
        operator, requested = match.groups()
        if _SEMVER.fullmatch(requested) is None:
            partial = _parse_partial_version(requested, path="requirement")
            major, minor, patch, supplied = partial
            if operator == "=":
                if supplied == 3:
                    requested = f"{major}.{minor}.{patch}"
                else:
                    lower_key = _version_key(f"{major}.{minor if supplied >= 2 else 0}.0")
                    upper_key = _bound_key(
                        f"{major}.{minor + 1}.0" if supplied == 2 else f"{major + 1}.0.0"
                    )
                    if not (lower_key <= candidate < upper_key):
                        return False
                    continue
            elif operator in {">=", ">", "<=", "<"} and supplied < 3:
                if operator == ">=":
                    requested = (
                        f"{major}.{minor if supplied >= 2 else 0}.{patch if supplied >= 3 else 0}"
                    )
                elif operator == ">":
                    requested = f"{major}.{minor + 1}.0" if supplied == 2 else f"{major + 1}.0.0"
                    operator = ">="
                elif operator == "<=":
                    requested = f"{major}.{minor + 1}.0" if supplied == 2 else f"{major + 1}.0.0"
                    operator = "<"
                else:
                    requested = f"{major}.{minor if supplied >= 2 else 0}.0"
        other = _bound_key(requested)
        if operator in {"=", "=="} and candidate != other:
            return False
        if operator == ">=" and candidate < other:
            return False
        if operator == "<=" and candidate > other:
            return False
        if operator == ">" and candidate <= other:
            return False
        if operator == "<" and candidate >= other:
            return False
    return True


def _platform_name() -> str:
    system = platform.system().lower()
    return {"windows": "windows", "darwin": "macos", "linux": "linux"}.get(system, system)


def _load_json(path: Path, *, maximum: int, label: str) -> Mapping[str, Any]:
    _assert_no_link_components(path.absolute(), label=label)
    if _is_link_or_reparse(path) or not path.is_file():
        raise PluginValidationError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise PluginValidationError(f"{label} exceeds the size limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise PluginValidationError(f"{label} contains duplicate field: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PluginValidationError(f"{label} must be valid UTF-8 JSON") from error
    return _mapping(value, path=label)


def _is_link_or_reparse(path: Path) -> bool:
    """Return whether a path is a symlink or Windows reparse point."""

    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _assert_no_link_components(path: Path, *, label: str) -> None:
    """Reject a path that reaches its target through a link or reparse point."""

    current = path
    while True:
        if _is_link_or_reparse(current):
            raise PluginValidationError(f"{label} cannot contain symlinks or reparse points")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _ensure_directory(path: Path, *, label: str) -> None:
    """Create a directory tree without traversing a link at any component."""

    _assert_no_link_components(path.absolute(), label=label)
    if _is_link_or_reparse(path):
        raise PluginValidationError(f"{label} cannot contain symlinks or reparse points")
    if path.exists():
        if not path.is_dir():
            raise PluginValidationError(f"{label} must be a regular directory")
        return
    parent = path.parent
    if parent == path:
        raise PluginValidationError(f"cannot create {label}")
    _ensure_directory(parent, label=label)
    try:
        path.mkdir()
    except FileExistsError:
        if _is_link_or_reparse(path) or not path.is_dir():
            raise PluginValidationError(f"{label} must be a regular directory") from None


@dataclass(frozen=True, slots=True)
class PluginFile:
    """One checksummed payload file declared by a plugin manifest."""

    path: str
    sha256: str
    size_bytes: int
    role: str = "payload"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _portable_path(self.path, path="plugin file path"))
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise PluginValidationError("plugin file sha256 must be a lowercase SHA-256 digest")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise PluginValidationError("plugin file size_bytes must be a non-negative integer")
        if self.size_bytes > _MAX_FILE_BYTES:
            raise PluginValidationError("plugin file exceeds the size limit")
        object.__setattr__(self, "role", _identifier(self.role, path="plugin file role"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, path: str = "plugin.files[]") -> PluginFile:
        value = _mapping(value, path=path)
        _strict_fields(
            value,
            allowed=frozenset({"path", "sha256", "size_bytes", "role"}),
            required=frozenset({"path", "sha256", "size_bytes"}),
            path=path,
        )
        return cls(
            path=value["path"],
            sha256=value["sha256"],
            size_bytes=value["size_bytes"],
            role=value.get("role", "payload"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Immutable metadata for one installable GLR plugin bundle."""

    plugin_id: str
    version: str
    kind: str
    name: str
    description: str
    entrypoint: str
    capabilities: tuple[str, ...]
    requires: Mapping[str, str] = field(default_factory=dict)
    platforms: tuple[str, ...] = ("windows", "linux", "macos")
    isolation: str = "process"
    permissions: tuple[str, ...] = ()
    dependencies: Mapping[str, str] = field(default_factory=dict)
    files: tuple[PluginFile, ...] = ()
    schema_version: str = PLUGIN_SCHEMA_VERSION

    @property
    def id(self) -> str:
        """Compatibility alias matching the manifest field name."""

        return self.plugin_id

    def __post_init__(self) -> None:
        if self.schema_version != PLUGIN_SCHEMA_VERSION:
            raise PluginValidationError(f"unsupported plugin schema: {self.schema_version!r}")
        object.__setattr__(self, "plugin_id", _identifier(self.plugin_id, path="plugin.id"))
        _parse_semver(self.version, path="plugin.version")
        object.__setattr__(self, "kind", _identifier(self.kind, path="plugin.kind"))
        if self.kind not in _PLUGIN_KINDS:
            raise PluginValidationError(f"unsupported plugin kind: {self.kind}")
        object.__setattr__(self, "name", _text(self.name, path="plugin.name"))
        object.__setattr__(self, "description", _text(self.description, path="plugin.description"))
        object.__setattr__(self, "entrypoint", _entrypoint(self.entrypoint))
        object.__setattr__(
            self,
            "capabilities",
            _set_of_strings(
                self.capabilities,
                path="plugin.capabilities",
                pattern=_CAPABILITY,
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "platforms",
            _set_of_strings(
                self.platforms,
                path="plugin.platforms",
                pattern=_IDENTIFIER,
                allowed=_PLATFORMS,
                minimum=1,
            ),
        )
        if self.isolation not in _ISOLATIONS:
            raise PluginValidationError(f"unsupported plugin isolation: {self.isolation}")
        object.__setattr__(
            self,
            "permissions",
            _set_of_strings(
                self.permissions,
                path="plugin.permissions",
                pattern=_CAPABILITY,
                allowed=_PERMISSIONS,
            ),
        )
        for label, raw in (("requires", self.requires), ("dependencies", self.dependencies)):
            mapping = _mapping(raw, path=f"plugin.{label}")
            normalized: dict[str, str] = {}
            for key, requirement in mapping.items():
                normalized[_identifier(key, path=f"plugin.{label} key")] = _validate_requirement(
                    requirement, path=f"plugin.{label}.{key}"
                )
            object.__setattr__(self, label, MappingProxyType(normalized))
        files = _sequence(self.files, path="plugin.files")
        if len(files) > _MAX_FILES or any(not isinstance(item, PluginFile) for item in files):
            raise PluginValidationError("plugin.files contains invalid entries")
        if any(item.path == PLUGIN_FILE_NAME for item in files):
            raise PluginValidationError("plugin.files cannot declare the manifest")
        paths = [item.path.casefold() for item in files]
        if len(paths) != len(set(paths)):
            raise PluginValidationError("plugin.files contains duplicate or case-colliding paths")
        object.__setattr__(self, "files", files)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PluginManifest:
        value = _mapping(value, path="plugin")
        _strict_fields(
            value,
            allowed=frozenset(
                {
                    "schema_version",
                    "id",
                    "version",
                    "kind",
                    "name",
                    "description",
                    "entrypoint",
                    "capabilities",
                    "requires",
                    "platforms",
                    "isolation",
                    "permissions",
                    "dependencies",
                    "files",
                }
            ),
            required=frozenset(
                {
                    "schema_version",
                    "id",
                    "version",
                    "kind",
                    "name",
                    "description",
                    "entrypoint",
                    "capabilities",
                }
            ),
            path="plugin",
        )
        raw_files = _sequence(value.get("files", ()), path="plugin.files")
        return cls(
            schema_version=value["schema_version"],
            plugin_id=value["id"],
            version=value["version"],
            kind=value["kind"],
            name=value["name"],
            description=value["description"],
            entrypoint=value["entrypoint"],
            capabilities=_sequence(value["capabilities"], path="plugin.capabilities"),
            requires=_mapping(value.get("requires", {}), path="plugin.requires"),
            platforms=_sequence(
                value.get("platforms", ("windows", "linux", "macos")),
                path="plugin.platforms",
            ),
            isolation=value.get("isolation", "process"),
            permissions=_sequence(value.get("permissions", ()), path="plugin.permissions"),
            dependencies=_mapping(value.get("dependencies", {}), path="plugin.dependencies"),
            files=tuple(PluginFile.from_mapping(item) for item in raw_files),
        )

    @classmethod
    def load(cls, path: str | Path) -> PluginManifest:
        return cls.from_mapping(
            _load_json(Path(path), maximum=_MAX_MANIFEST_BYTES, label="plugin manifest")
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "id": self.plugin_id,
            "version": self.version,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "entrypoint": self.entrypoint,
            "capabilities": list(self.capabilities),
            "requires": dict(self.requires),
            "platforms": list(self.platforms),
            "isolation": self.isolation,
            "permissions": list(self.permissions),
            "dependencies": dict(self.dependencies),
            "files": [item.to_mapping() for item in self.files],
        }


def _entrypoint(value: object) -> str:
    text = _text(value, path="plugin.entrypoint", maximum=512)
    if any(character in text for character in ("\n", "\r", "$", "`", "&", "|", ";")):
        raise PluginValidationError("plugin.entrypoint contains executable shell syntax")
    if ":" in text:
        module, attribute = text.split(":", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", module) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", attribute
        ):
            raise PluginValidationError("plugin.entrypoint must be module:attribute")
        return text
    _portable_path(text, path="plugin.entrypoint")
    if text == PLUGIN_FILE_NAME:
        raise PluginValidationError("plugin.entrypoint cannot be the manifest")
    return text


@dataclass(frozen=True, slots=True)
class PluginProfileRef:
    """One profile-selected plugin and its explicitly granted permissions."""

    plugin_id: str
    version: str = "*"
    enabled: bool = True
    permissions: tuple[str, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plugin_id", _identifier(self.plugin_id, path="profile plugin id"))
        object.__setattr__(
            self, "version", _validate_requirement(self.version, path="profile plugin version")
        )
        if not isinstance(self.enabled, bool):
            raise PluginValidationError("profile plugin enabled must be a boolean")
        object.__setattr__(
            self,
            "permissions",
            _set_of_strings(
                self.permissions,
                path="profile plugin permissions",
                pattern=_CAPABILITY,
                allowed=_PERMISSIONS,
            ),
        )
        object.__setattr__(
            self,
            "config",
            _json_object(
                _mapping(self.config, path="profile plugin config"),
                path="profile plugin config",
                maximum=64 * 1024,
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PluginProfileRef:
        value = _mapping(value, path="profile.plugins[]")
        _strict_fields(
            value,
            allowed=frozenset({"id", "version", "enabled", "permissions", "config"}),
            required=frozenset({"id"}),
            path="profile.plugins[]",
        )
        return cls(
            plugin_id=value["id"],
            version=value.get("version", "*"),
            enabled=value.get("enabled", True),
            permissions=tuple(value.get("permissions", ())),
            config=_mapping(value.get("config", {}), path="profile.plugins[].config"),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.plugin_id,
            "version": self.version,
            "enabled": self.enabled,
            "permissions": list(self.permissions),
            "config": _thaw_json(self.config),
        }


# Short alias used by integrations that prefer the DSH terminology.
PluginRef = PluginProfileRef


@dataclass(frozen=True, slots=True)
class PluginProfile:
    """An ordered, deterministic composition of plugin references."""

    name: str
    plugins: tuple[PluginProfileRef, ...] = ()
    schema_version: str = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise PluginValidationError(f"unsupported profile schema: {self.schema_version!r}")
        object.__setattr__(self, "name", _name(self.name, path="profile.name"))
        entries = _sequence(self.plugins, path="profile.plugins")
        if any(not isinstance(item, PluginProfileRef) for item in entries):
            raise PluginValidationError("profile.plugins contains invalid entries")
        ids = [item.plugin_id for item in entries]
        if len(ids) != len(set(ids)):
            raise PluginValidationError("profile.plugins must not contain duplicate plugin IDs")
        object.__setattr__(self, "plugins", entries)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PluginProfile:
        value = _mapping(value, path="profile")
        _strict_fields(
            value,
            allowed=frozenset({"schema_version", "name", "plugins"}),
            required=frozenset({"schema_version", "name", "plugins"}),
            path="profile",
        )
        raw_plugins = _sequence(value["plugins"], path="profile.plugins")
        return cls(
            schema_version=value["schema_version"],
            name=value["name"],
            plugins=tuple(PluginProfileRef.from_mapping(item) for item in raw_plugins),
        )

    @classmethod
    def load(cls, path: str | Path) -> PluginProfile:
        return cls.from_mapping(
            _load_json(Path(path), maximum=_MAX_PROFILE_BYTES, label="plugin profile")
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "plugins": [item.to_mapping() for item in self.plugins],
        }

    def digest(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_mapping()))


@dataclass(frozen=True, slots=True)
class PluginInspection:
    """Pure-data inspection output; no plugin code is imported."""

    manifest: PluginManifest
    source_label: str
    files: tuple[PluginFile, ...]
    content_sha256: str
    manifest_sha256: str
    total_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.manifest.schema_version,
            "source": self.source_label,
            "plugin": self.manifest.to_mapping(),
            "files": [item.to_mapping() for item in self.files],
            "content_sha256": self.content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True, slots=True)
class PluginInstallation:
    """An installed bundle identity with a project-relative storage path."""

    manifest: PluginManifest
    content_sha256: str
    manifest_sha256: str
    installed_path: str
    source_kind: str = "local"

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.manifest.plugin_id,
            "version": self.manifest.version,
            "kind": self.manifest.kind,
            "content_sha256": self.content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "path": self.installed_path,
            "source_kind": self.source_kind,
            "isolation": self.manifest.isolation,
            "capabilities": list(self.manifest.capabilities),
        }


@dataclass(frozen=True, slots=True)
class ResolvedPlugin:
    manifest: PluginManifest
    permissions: tuple[str, ...]
    config: Mapping[str, Any]
    content_sha256: str
    root: Path
    project_relative_path: str = ""

    def to_mapping(self) -> dict[str, object]:
        return {
            "id": self.manifest.plugin_id,
            "version": self.manifest.version,
            "kind": self.manifest.kind,
            "permissions": list(self.permissions),
            "config": _thaw_json(self.config),
            "content_sha256": self.content_sha256,
            "path": self.project_relative_path or self.root.name,
        }


@dataclass(frozen=True, slots=True)
class ResolvedProfile:
    profile: PluginProfile
    plugins: tuple[ResolvedPlugin, ...]
    digest: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.profile.schema_version,
            "name": self.profile.name,
            "digest": self.digest,
            "plugins": [item.to_mapping() for item in self.plugins],
        }


def _walk_files(root: Path) -> tuple[PluginFile, ...]:
    if _is_link_or_reparse(root) or not root.is_dir():
        raise PluginValidationError("plugin source must be a regular directory")
    entries: list[PluginFile] = []
    seen: set[str] = set()
    normalized: set[str] = set()
    total = 0

    def visit(directory: Path, depth: int) -> None:
        nonlocal total
        if depth > _MAX_PATH_DEPTH:
            raise PluginValidationError("plugin source path is too deep")
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name.casefold())
        except OSError as error:
            raise PluginValidationError(
                f"cannot inspect plugin directory: {directory.name}"
            ) from error
        for child in children:
            path = Path(child.path)
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as error:
                raise PluginValidationError("cannot stat plugin entry") from error
            if child.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400):
                raise PluginValidationError(
                    "plugin source cannot contain symlinks or reparse points"
                )
            mode = info.st_mode
            if stat.S_ISDIR(mode):
                visit(path, depth + 1)
                continue
            if not stat.S_ISREG(mode) or info.st_nlink > 1:
                raise PluginValidationError("plugin source can contain only regular files")
            relative = _portable_path(path.relative_to(root).as_posix(), path="plugin file path")
            folded = relative.casefold()
            unicode_folded = unicodedata.normalize("NFC", relative).casefold()
            if folded in seen or unicode_folded in normalized:
                raise PluginValidationError("plugin source contains case-colliding paths")
            seen.add(folded)
            normalized.add(unicode_folded)
            size = int(info.st_size)
            if size > _MAX_FILE_BYTES:
                raise PluginValidationError("plugin file exceeds the size limit")
            total += size
            if total > _MAX_TOTAL_BYTES:
                raise PluginValidationError("plugin source exceeds the total size limit")
            entries.append(PluginFile(relative, _sha256_file(path), size))
            if len(entries) > _MAX_FILES:
                raise PluginValidationError("plugin source contains too many files")

    visit(root, 0)
    return tuple(sorted(entries, key=lambda item: item.path))


def _inspect_directory(source: Path) -> PluginInspection:
    # Check the user-supplied path before resolving it.  Resolving first would
    # make a symlinked bundle root indistinguishable from a trusted directory.
    source = _safe_directory_path(source)
    manifest_path = source / PLUGIN_FILE_NAME
    manifest = PluginManifest.load(manifest_path)
    files = _walk_files(source)
    payload_files = tuple(item for item in files if item.path != PLUGIN_FILE_NAME)
    if manifest.files:
        declared = {item.path: item for item in manifest.files}
        actual = {item.path: item for item in payload_files}
        if set(declared) != set(actual):
            raise PluginValidationError("plugin manifest files do not match the source inventory")
        for path, expected in declared.items():
            found = actual[path]
            if found.sha256 != expected.sha256 or found.size_bytes != expected.size_bytes:
                raise PluginValidationError(f"plugin file integrity mismatch: {path}")
    entrypoint = manifest.entrypoint
    if ":" not in entrypoint:
        entry_path = _portable_path(entrypoint, path="plugin.entrypoint")
        if entry_path not in {item.path for item in payload_files}:
            raise PluginValidationError("plugin.entrypoint file is missing")
    inventory = [item.to_mapping() for item in files]
    canonical = _canonical_json_bytes({"manifest": manifest.to_mapping(), "files": inventory})
    manifest_bytes = manifest_path.read_bytes()
    return PluginInspection(
        manifest=manifest,
        source_label=source.name,
        files=files,
        content_sha256=_sha256_bytes(canonical),
        manifest_sha256=_sha256_bytes(manifest_bytes),
        total_bytes=sum(item.size_bytes for item in files),
    )


def _safe_directory_path(path: Path) -> Path:
    """Return a canonical directory path after checking every path component."""

    absolute = path.absolute()
    _assert_no_link_components(absolute, label="plugin source")
    if not absolute.is_dir():
        raise PluginValidationError("plugin source must be a regular directory")
    return absolute.resolve()


class PluginManager:
    """Manage local plugin bundles and declarative profiles for one project.

    The manager deliberately supports local directories only in this first
    contract.  Git/npm/HTTP acquisition can be layered on later, but must feed
    the same no-exec inspector and explicit trust gate.
    """

    def __init__(self, project_root: str | Path, *, glr_version: str | None = None) -> None:
        raw_root = Path(project_root).absolute()
        _assert_no_link_components(raw_root, label="plugin project root")
        if raw_root.is_file():
            raw_root = raw_root.parent
            _assert_no_link_components(raw_root, label="plugin project root")
        self.project_root = raw_root.resolve()
        self.store_root = self.project_root / PLUGIN_STORE_DIR
        self.profile_root = self.project_root / PROFILE_STORE_DIR
        self.glr_version = glr_version
        self._lock = RLock()

    def inspect(self, source: str | Path) -> PluginInspection:
        """Inspect a local bundle without importing or executing it."""

        return _inspect_directory(Path(source))

    def install(
        self, source: str | Path, *, expected_sha256: str | None = None
    ) -> PluginInstallation:
        """Atomically install a verified local bundle into the project store."""

        source_root = _safe_directory_path(Path(source))
        inspection = _inspect_directory(source_root)
        if expected_sha256 is not None:
            if _SHA256.fullmatch(expected_sha256) is None:
                raise PluginValidationError("expected_sha256 must be a lowercase SHA-256 digest")
            if inspection.content_sha256 != expected_sha256:
                raise PluginValidationError("expected_sha256 does not match the inspected bundle")
        destination = self.store_root / inspection.manifest.plugin_id / inspection.manifest.version
        with self._lock:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    "plugin is already installed: "
                    f"{inspection.manifest.plugin_id}@{inspection.manifest.version}"
                )
            _ensure_directory(self.store_root, label="plugin store")
            parent = destination.parent
            _ensure_directory(parent, label="plugin store")
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{inspection.manifest.plugin_id}-", dir=parent)
            )
            try:
                for item in inspection.files:
                    source_file = source_root / PurePosixPath(item.path)
                    target = temporary / PurePosixPath(item.path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_file, target)
                copied = self.inspect(temporary)
                if copied.content_sha256 != inspection.content_sha256:
                    raise PluginValidationError("staged plugin digest changed during installation")
                temporary.replace(destination)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return PluginInstallation(
            manifest=inspection.manifest,
            content_sha256=inspection.content_sha256,
            manifest_sha256=inspection.manifest_sha256,
            installed_path=destination.relative_to(self.project_root).as_posix(),
        )

    def _installed(self) -> list[tuple[PluginInspection, Path]]:
        _assert_no_link_components(self.store_root.absolute(), label="plugin store")
        if _is_link_or_reparse(self.store_root):
            raise PluginValidationError("plugin store must be a regular directory")
        if not self.store_root.exists():
            return []
        if not self.store_root.is_dir():
            raise PluginValidationError("plugin store must be a regular directory")
        result: list[tuple[PluginInspection, Path]] = []
        for plugin_dir in sorted(self.store_root.iterdir(), key=lambda item: item.name.casefold()):
            if _is_link_or_reparse(plugin_dir) or not plugin_dir.is_dir():
                raise PluginValidationError("plugin store contains an invalid entry")
            for version_dir in sorted(plugin_dir.iterdir(), key=lambda item: item.name.casefold()):
                if _is_link_or_reparse(version_dir) or not version_dir.is_dir():
                    raise PluginValidationError("plugin store contains an invalid version entry")
                inspection = _inspect_directory(version_dir)
                if (
                    inspection.manifest.plugin_id != plugin_dir.name
                    or inspection.manifest.version != version_dir.name
                ):
                    raise PluginValidationError("installed plugin path does not match its manifest")
                result.append((inspection, version_dir))
        return result

    def list_installed(self) -> tuple[PluginInstallation, ...]:
        return tuple(
            PluginInstallation(
                manifest=inspection.manifest,
                content_sha256=inspection.content_sha256,
                manifest_sha256=inspection.manifest_sha256,
                installed_path=path.relative_to(self.project_root).as_posix(),
                source_kind="installed",
            )
            for inspection, path in self._installed()
        )

    def remove(self, plugin_id: str, version: str | None = None) -> None:
        plugin_id = _identifier(plugin_id, path="plugin.id")
        candidates = [
            (inspection, path)
            for inspection, path in self._installed()
            if inspection.manifest.plugin_id == plugin_id
        ]
        if version is not None:
            _parse_semver(version, path="plugin.version")
            candidates = [
                (inspection, path)
                for inspection, path in candidates
                if inspection.manifest.version == version
            ]
        if not candidates:
            raise PluginValidationError("plugin is not installed")
        if version is None and len(candidates) > 1:
            raise PluginValidationError(
                "version is required when multiple plugin versions are installed"
            )
        target = candidates[0][1]
        for profile in self.list_profiles():
            if any(ref.enabled and ref.plugin_id == plugin_id for ref in profile.plugins):
                raise PluginTrustError(f"plugin is enabled by profile {profile.name}")
        if target.is_symlink() or not target.is_dir() or not target.is_relative_to(self.store_root):
            raise PluginValidationError("refusing to remove an unsafe plugin path")
        shutil.rmtree(target)
        with suppress(OSError):
            target.parent.rmdir()

    def save_profile(self, profile: PluginProfile) -> Path:
        if not isinstance(profile, PluginProfile):
            raise PluginValidationError("profile must be a PluginProfile")
        _ensure_directory(self.profile_root, label="profile store")
        destination = self.profile_root / f"{profile.name}.json"
        if _is_link_or_reparse(destination):
            raise PluginValidationError("profile path cannot be a symlink")
        payload = (
            json.dumps(profile.to_mapping(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.profile_root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            with suppress(OSError):
                temporary.unlink()
            raise
        return destination

    def load_profile(self, name: str) -> PluginProfile:
        name = _name(name, path="profile.name")
        return PluginProfile.load(self.profile_root / f"{name}.json")

    def list_profiles(self) -> tuple[PluginProfile, ...]:
        _assert_no_link_components(self.profile_root.absolute(), label="profile store")
        if _is_link_or_reparse(self.profile_root):
            raise PluginValidationError("profile store must be a regular directory")
        if not self.profile_root.exists():
            return ()
        if not self.profile_root.is_dir():
            raise PluginValidationError("profile store must be a regular directory")
        profiles: list[PluginProfile] = []
        for path in sorted(self.profile_root.glob("*.json"), key=lambda item: item.name.casefold()):
            if _is_link_or_reparse(path) or not path.is_file():
                raise PluginValidationError("profile store contains an invalid entry")
            profiles.append(PluginProfile.load(path))
        return tuple(profiles)

    def resolve_profile(self, profile: str | PluginProfile) -> ResolvedProfile:
        selected = self.load_profile(profile) if isinstance(profile, str) else profile
        if not isinstance(selected, PluginProfile):
            raise PluginValidationError("profile must be a name or PluginProfile")
        installed = self._installed()
        by_id: dict[str, list[tuple[PluginInspection, Path]]] = {}
        for inspection, path in installed:
            by_id.setdefault(inspection.manifest.plugin_id, []).append((inspection, path))
        for values in by_id.values():
            values.sort(key=lambda item: _version_order_key(item[0].manifest.version), reverse=True)
        resolved: dict[str, ResolvedPlugin] = {}
        order: list[str] = []
        resolving: list[str] = []

        def select(
            plugin_id: str,
            requirement: str,
            permissions: tuple[str, ...],
            config: Mapping[str, Any],
            *,
            explicit: bool,
        ) -> None:
            frozen_config = _json_object(
                _mapping(config, path="profile plugin config"),
                path="profile plugin config",
                maximum=64 * 1024,
            )
            if plugin_id in resolving:
                cycle = " -> ".join((*resolving, plugin_id))
                raise PluginValidationError(f"plugin dependency cycle: {cycle}")
            existing = resolved.get(plugin_id)
            if existing is not None:
                if not _satisfies(existing.manifest.version, requirement):
                    raise PluginValidationError(f"plugin dependency conflict for {plugin_id}")
                requested_permissions = tuple(sorted(permissions))
                if requested_permissions:
                    denied = set(requested_permissions) - set(existing.manifest.permissions)
                    if denied:
                        names = ", ".join(sorted(denied))
                        raise PluginValidationError(
                            f"profile permission grant exceeds plugin declaration: {names}"
                        )
                    if existing.permissions and requested_permissions != existing.permissions:
                        raise PluginValidationError(f"plugin permission conflict for {plugin_id}")
                    if not existing.permissions and explicit:
                        existing = replace(existing, permissions=requested_permissions)
                if explicit and frozen_config:
                    if existing.config and _thaw_json(existing.config) != _thaw_json(frozen_config):
                        raise PluginValidationError(f"plugin config conflict for {plugin_id}")
                    if not existing.config:
                        existing = replace(existing, config=frozen_config)
                if existing is not resolved.get(plugin_id):
                    resolved[plugin_id] = existing
                return
            candidates = by_id.get(plugin_id, [])
            candidate = next(
                (item for item in candidates if _satisfies(item[0].manifest.version, requirement)),
                None,
            )
            if candidate is None:
                raise PluginValidationError(
                    f"plugin is not installed or version is unsatisfied: {plugin_id}@{requirement}"
                )
            inspection, path = candidate
            incompatible_glr = (
                self.glr_version is not None
                and "glr" in inspection.manifest.requires
                and not _satisfies(self.glr_version, inspection.manifest.requires["glr"])
            )
            if incompatible_glr:
                raise PluginValidationError(
                    f"plugin requires an incompatible GLR version: {plugin_id}"
                )
            if _platform_name() not in inspection.manifest.platforms:
                raise PluginValidationError(
                    f"plugin is not supported on this platform: {plugin_id}"
                )
            denied = set(permissions) - set(inspection.manifest.permissions)
            if denied:
                names = ", ".join(sorted(denied))
                raise PluginValidationError(
                    f"profile permission grant exceeds plugin declaration: {names}"
                )
            resolving.append(plugin_id)
            for dependency_id, dependency_range in sorted(inspection.manifest.dependencies.items()):
                select(dependency_id, dependency_range, (), {}, explicit=False)
            resolving.pop()
            resolved[plugin_id] = ResolvedPlugin(
                manifest=inspection.manifest,
                permissions=tuple(sorted(permissions)),
                config=frozen_config,
                content_sha256=inspection.content_sha256,
                root=path,
                project_relative_path=path.relative_to(self.project_root).as_posix(),
            )
            order.append(plugin_id)

        for reference in selected.plugins:
            if reference.enabled:
                select(
                    reference.plugin_id,
                    reference.version,
                    reference.permissions,
                    reference.config,
                    explicit=True,
                )
        plugins = tuple(resolved[item] for item in order)
        digest_payload = {
            "profile": selected.to_mapping(),
            "plugins": [
                {
                    "id": item.manifest.plugin_id,
                    "version": item.manifest.version,
                    "content_sha256": item.content_sha256,
                    "permissions": list(item.permissions),
                    "config": _thaw_json(item.config),
                }
                for item in plugins
            ],
        }
        digest = _sha256_bytes(_canonical_json_bytes(digest_payload))
        return ResolvedProfile(selected, plugins, digest)

    def health(self, profile: str | PluginProfile | None = None) -> tuple[dict[str, object], ...]:
        """Return static readiness; this method never imports or executes plugins."""

        if profile is not None:
            try:
                resolved = self.resolve_profile(profile)
            except PluginError as error:
                return ({"status": "blocked", "reason": str(error)},)
            return tuple(
                {
                    "id": item.manifest.plugin_id,
                    "version": item.manifest.version,
                    "kind": item.manifest.kind,
                    "status": "ready",
                    "isolation": item.manifest.isolation,
                    "capabilities": list(item.manifest.capabilities),
                    "permissions": list(item.permissions),
                    "content_sha256": item.content_sha256,
                    "profile_digest": resolved.digest,
                }
                for item in resolved.plugins
            )
        result: list[dict[str, object]] = []
        for inspection, _path in self._installed():
            status = "ready" if _platform_name() in inspection.manifest.platforms else "blocked"
            result.append(
                {
                    "id": inspection.manifest.plugin_id,
                    "version": inspection.manifest.version,
                    "kind": inspection.manifest.kind,
                    "status": status,
                    "reason": None if status == "ready" else "unsupported platform",
                    "isolation": inspection.manifest.isolation,
                    "capabilities": list(inspection.manifest.capabilities),
                    "content_sha256": inspection.content_sha256,
                }
            )
        return tuple(result)


def inspect_plugin(source: str | Path) -> PluginInspection:
    """Convenience wrapper for pure-data bundle inspection."""

    return _inspect_directory(Path(source))


def load_plugin_manifest(path: str | Path) -> PluginManifest:
    """Load a strict manifest without importing its entrypoint."""

    return PluginManifest.load(path)


__all__ = [
    "PLUGIN_FILE_NAME",
    "PLUGIN_SCHEMA_VERSION",
    "PLUGIN_STORE_DIR",
    "PROFILE_SCHEMA_VERSION",
    "PROFILE_STORE_DIR",
    "PluginError",
    "PluginFile",
    "PluginInspection",
    "PluginInstallation",
    "PluginManager",
    "PluginManifest",
    "PluginProfile",
    "PluginProfileRef",
    "PluginRef",
    "PluginTrustError",
    "PluginValidationError",
    "ResolvedPlugin",
    "ResolvedProfile",
    "inspect_plugin",
    "load_plugin_manifest",
]
