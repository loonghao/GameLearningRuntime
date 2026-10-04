"""Bounded offline parsing of source and legacy records, without executing source.

These internal data helpers do not infer game semantics, correlate receipts, or
turn sparse legacy records into authoritative runtime transitions.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import re
import tokenize
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, TypeVar, cast

JsonScalar: TypeAlias = bool | int | float | str | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]
_Node = TypeVar("_Node", bound=ast.AST)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


class OfflineParseError(ValueError):
    """A categorized failure that never includes source lines or record values."""

    def __init__(
        self,
        category: str,
        *,
        line_number: int | None = None,
        column: int | None = None,
        selector: str | None = None,
        match_count: int | None = None,
    ) -> None:
        self.category = category
        self.line_number = line_number
        self.column = column
        self.selector = selector
        self.match_count = match_count
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ParsedPythonSource:
    source_sha256: str
    tree: ast.Module


@dataclass(frozen=True, slots=True)
class StatementBlock:
    owner: ast.AST
    field: str
    statements: tuple[ast.stmt, ...]


@dataclass(frozen=True, slots=True)
class JsonlRecord:
    line_number: int
    data: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ParsedJsonl:
    source_sha256: str
    records: tuple[JsonlRecord, ...]


@dataclass(frozen=True, slots=True)
class ParsedJsonObject:
    source_sha256: str
    data: Mapping[str, JsonValue]


def _positive_limits(**limits: int) -> None:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in limits.values()
    ):
        raise OfflineParseError("invalid_limit")


def _bound_payload(payload: bytes, max_bytes: int, expected_sha256: str | None) -> str:
    _positive_limits(max_bytes=max_bytes)
    if not isinstance(payload, bytes):
        raise OfflineParseError("input_type")
    if len(payload) > max_bytes:
        raise OfflineParseError("byte_limit")
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
        ):
            raise OfflineParseError("invalid_digest")
        if digest != expected_sha256.lower():
            raise OfflineParseError("source_integrity")
    return digest


def parse_python_source(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = 8_388_608,
    max_nodes: int = 100_000,
) -> ParsedPythonSource:
    """Parse sealed source bytes respecting Python's encoding declaration.

    The returned AST is a normal mutable AST; its structural edits do not change
    the digest of the original bytes. No compilation, imports or execution occur.
    """
    digest = _bound_payload(payload, max_bytes, expected_sha256)
    _positive_limits(max_nodes=max_nodes)
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
        text = payload.decode(encoding)
    except (SyntaxError, UnicodeError, LookupError):
        raise OfflineParseError("source_encoding") from None
    try:
        tree = ast.parse(text, filename="<offline-source>")
    except SyntaxError as error:
        raise OfflineParseError(
            "python_syntax", line_number=error.lineno, column=error.offset
        ) from None
    except (RecursionError, MemoryError):
        raise OfflineParseError("python_structure_limit") from None
    if sum(1 for _ in ast.walk(tree)) > max_nodes:
        raise OfflineParseError("python_structure_limit")
    return ParsedPythonSource(digest, tree)


def statement_blocks(owner: ast.AST) -> tuple[StatementBlock, ...]:
    """Return only actual statement-list fields, including exception finalbody.

    Lambda and conditional-expression body fields are single expressions, so
    they remain traversal contexts and cannot be sliced as statement lists.
    """
    blocks = []
    for field, value in ast.iter_fields(owner):
        if field not in {"body", "orelse", "finalbody"} or not isinstance(value, list):
            continue
        if not all(isinstance(statement, ast.stmt) for statement in value):
            raise OfflineParseError("ast_shape")
        blocks.append(StatementBlock(owner, field, tuple(value)))
    return tuple(blocks)


def walk_scope(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk one subtree without entering nested function/class/lambda scopes.

    Nested scope declarations themselves are yielded. This structural boundary
    does not resolve symbols or prove control-flow dominance or runtime effects.
    """
    stack = [scope]
    while stack:
        node = stack.pop()
        yield node
        if node is not scope and isinstance(node, _SCOPES):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def unique_match(
    nodes: Iterable[_Node],
    predicate: Callable[[_Node], bool],
    *,
    selector: str = "selection",
) -> _Node:
    """Require exactly one match; never choose the first or fall back to a line."""
    matches = [node for node in nodes if predicate(node)]
    if len(matches) != 1:
        raise OfflineParseError("selection_not_unique", selector=selector, match_count=len(matches))
    return matches[0]


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OfflineParseError("duplicate_key")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise OfflineParseError("nonfinite_number")
    return number


def _bounded_int(value: str) -> int:
    # Apply the modern interpreter's default ceiling consistently on Python 3.10.
    if len(value.removeprefix("-")) > 4_300:
        raise OfflineParseError("number_limit")
    return int(value)


def _reject_constant(value: str) -> None:
    raise OfflineParseError("nonfinite_number")


def _freeze(value: object, *, depth: int, max_depth: int) -> JsonValue:
    if depth > max_depth:
        raise OfflineParseError("nesting_limit")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return MappingProxyType(
            {
                key: _freeze(item, depth=depth + 1, max_depth=max_depth)
                for key, item in mapping.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze(item, depth=depth + 1, max_depth=max_depth) for item in value)
    raise OfflineParseError("record_shape")


def _object_from_text(
    text: str, *, max_depth: int, line_number: int | None = None
) -> Mapping[str, JsonValue]:
    try:
        value: object = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_int=_bounded_int,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
        if not isinstance(value, dict):
            raise OfflineParseError("record_shape")
        frozen = _freeze(value, depth=0, max_depth=max_depth)
        return cast(Mapping[str, JsonValue], frozen)
    except OfflineParseError as error:
        raise OfflineParseError(error.category, line_number=line_number) from None
    except json.JSONDecodeError as error:
        raise OfflineParseError(
            "json_syntax", line_number=line_number or error.lineno, column=error.colno
        ) from None
    except RecursionError:
        raise OfflineParseError("nesting_limit", line_number=line_number) from None
    except ValueError:
        # Includes the interpreter's bounded integer-string conversion failure.
        raise OfflineParseError("number_limit", line_number=line_number) from None


def parse_jsonl(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = 8_388_608,
    max_line_bytes: int = 262_144,
    max_records: int = 100_000,
    max_depth: int = 64,
) -> ParsedJsonl:
    """Decode all JSONL object records, or raise without returning a partial set.

    Blank lines preserve physical numbering. A BOM is accepted only at the
    start of the payload. A valid final record needs no newline; a truncated
    JSON value is an error. Missing keys stay absent and JSON null stays None.
    """
    digest = _bound_payload(payload, max_bytes, expected_sha256)
    _positive_limits(max_line_bytes=max_line_bytes, max_records=max_records, max_depth=max_depth)
    records: list[JsonlRecord] = []
    for line_number, line in enumerate(payload.split(b"\n"), start=1):
        if len(line) > max_line_bytes:
            raise OfflineParseError("line_limit", line_number=line_number)
        try:
            text = line.decode("utf-8-sig" if line_number == 1 else "utf-8")
        except UnicodeError:
            raise OfflineParseError("json_encoding", line_number=line_number) from None
        if not text.strip():
            continue
        if len(records) >= max_records:
            raise OfflineParseError("record_limit", line_number=line_number)
        data = _object_from_text(text, max_depth=max_depth, line_number=line_number)
        records.append(JsonlRecord(line_number, data))
    return ParsedJsonl(digest, tuple(records))


def parse_json_object(
    payload: bytes,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = 8_388_608,
    max_depth: int = 64,
) -> ParsedJsonObject:
    """Decode one possibly multiline JSON object with the same strict rules."""
    digest = _bound_payload(payload, max_bytes, expected_sha256)
    _positive_limits(max_depth=max_depth)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError:
        raise OfflineParseError("json_encoding") from None
    return ParsedJsonObject(digest, _object_from_text(text, max_depth=max_depth))
