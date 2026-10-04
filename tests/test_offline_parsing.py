"""Independent synthetic source and JSON samples; no external dataset fixtures."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from game_learning_runtime.offline_parsing import (
    OfflineParseError,
    parse_json_object,
    parse_jsonl,
    parse_python_source,
    statement_blocks,
    unique_match,
    walk_scope,
)


def test_expression_body_is_not_a_statement_list() -> None:
    tree = ast.parse("identity = lambda item: item\nvalue = before if ready else after\n")
    expressions = [node for node in ast.walk(tree) if isinstance(node, (ast.Lambda, ast.IfExp))]

    assert len(expressions) == 2
    for expression in expressions:
        assert statement_blocks(expression) == ()


def test_field_aware_blocks_cover_nested_ifs_and_exception_cleanup() -> None:
    source = b"""def process():
    try:
        if enabled:
            first = 1
        else:
            second = 2
    except ValueError:
        failure = 3
    else:
        success = 4
    finally:
        cleanup = 5
"""
    parsed = parse_python_source(source)
    function = unique_match(parsed.tree.body, lambda node: isinstance(node, ast.FunctionDef))
    assignments = {
        statement.targets[0].id
        for node in walk_scope(function)
        for block in statement_blocks(node)
        for statement in block.statements
        if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name)
    }
    final_blocks = [
        block
        for node in walk_scope(function)
        for block in statement_blocks(node)
        if block.field == "finalbody"
    ]

    assert assignments == {"first", "second", "failure", "success", "cleanup"}
    assert len(final_blocks) == 1
    assert isinstance(final_blocks[0].owner, ast.Try)


def test_malformed_statement_list_is_an_explicit_shape_error() -> None:
    malformed = ast.If(test=ast.Constant(True), body=[ast.Name(id="expression")], orelse=[])
    with pytest.raises(OfflineParseError) as caught:
        statement_blocks(malformed)
    assert caught.value.category == "ast_shape"


def test_unique_selection_respects_scope_and_rejects_ambiguous_loops() -> None:
    source = b"""def process(first_batch, second_batch):
    for item in first_batch:
        first = item
    for record in second_batch:
        second = record
    def nested(other_batch):
        for record in other_batch:
            third = record
"""
    parsed = parse_python_source(source)
    function = unique_match(parsed.tree.body, lambda node: isinstance(node, ast.FunctionDef))
    assert len([node for node in ast.walk(function) if isinstance(node, ast.For)]) == 3
    assert len([node for node in walk_scope(function) if isinstance(node, ast.For)]) == 2
    with pytest.raises(OfflineParseError) as caught:
        unique_match(
            walk_scope(function), lambda node: isinstance(node, ast.For), selector="iteration"
        )
    assert caught.value.category == "selection_not_unique"
    assert caught.value.match_count == 2
    selected = unique_match(
        walk_scope(function),
        lambda node: (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "record"
        ),
        selector="record_iteration",
    )
    assert isinstance(selected, ast.For)
    assert isinstance(selected.iter, ast.Name)
    assert selected.iter.id == "second_batch"


@pytest.mark.parametrize("count", [0, 2])
def test_unique_selector_does_not_fall_back_on_absent_or_duplicate_targets(count: int) -> None:
    nodes = [ast.Pass() for _ in range(count)]
    with pytest.raises(OfflineParseError) as caught:
        unique_match(nodes, lambda node: isinstance(node, ast.Pass), selector="target")
    assert caught.value.match_count == count
    assert caught.value.selector == "target"


def test_nested_function_class_and_lambda_are_scope_boundaries() -> None:
    tree = ast.parse("""def process():
    outer = 1
    async def helper():
        hidden_function = 2
    class Helper:
        hidden_class = 3
    expression = lambda: hidden_lambda
""")
    function = tree.body[0]
    names = {node.id for node in walk_scope(function) if isinstance(node, ast.Name)}
    assert "outer" in names
    assert "hidden_function" not in names
    assert "hidden_class" not in names
    assert "hidden_lambda" not in names


def test_source_parser_never_executes_module_or_imports(tmp_path: Path) -> None:
    sentinel = tmp_path / "must-not-exist"
    source = (
        f"import module_that_does_not_exist\nopen({str(sentinel)!r}, 'w').write('x')\n"
    ).encode()
    parsed = parse_python_source(source, expected_sha256=hashlib.sha256(source).hexdigest())
    assert parsed.source_sha256 == hashlib.sha256(source).hexdigest()
    assert len(parsed.tree.body) == 2
    assert not sentinel.exists()


@pytest.mark.parametrize(
    ("payload", "kwargs", "category"),
    [
        (b"value = 1\n", {"expected_sha256": "0" * 64}, "source_integrity"),
        (b"value = 1\n", {"max_bytes": 2}, "byte_limit"),
        (b"def broken(:\n", {}, "python_syntax"),
        (b"value = '\xff'\n", {}, "source_encoding"),
        (b"value = 1\n", {"max_nodes": 1}, "python_structure_limit"),
    ],
)
def test_python_source_rejects_unbound_or_invalid_bytes(
    payload: bytes, kwargs: dict[str, object], category: str
) -> None:
    with pytest.raises(OfflineParseError) as caught:
        parse_python_source(payload, **kwargs)  # type: ignore[arg-type]
    assert caught.value.category == category


def test_python_source_honors_declared_encoding_without_executing() -> None:
    parsed = parse_python_source(b"# coding: latin-1\nlabel = '\xe9'\n")
    assert isinstance(parsed.tree.body[0], ast.Assign)
    assert isinstance(parsed.tree.body[0].value, ast.Constant)
    assert parsed.tree.body[0].value.value == "é"


def test_missing_null_zero_and_boolean_are_distinct_decoded_values() -> None:
    decoded = parse_jsonl(b'{"value":null,"enabled":true}\n{"value":0,"enabled":false}\n')
    first, second = [record.data for record in decoded.records]
    assert first["value"] is None
    assert second["value"] == 0
    assert type(second["value"]) is int
    assert type(first["enabled"]) is bool
    assert first["enabled"] is True
    assert second["enabled"] is False
    assert "before" not in first
    assert "before" not in second


def test_jsonl_keeps_physical_line_numbers_bom_crlf_and_opaque_text() -> None:
    payload = b'\xef\xbb\xbf{"value":null,"note":"unclosed { fragment"}\r\n\r\n{"value":0}\n'
    decoded = parse_jsonl(payload)
    assert decoded.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert [record.line_number for record in decoded.records] == [1, 3]
    assert decoded.records[0].data["value"] is None
    assert decoded.records[0].data["note"] == "unclosed { fragment"
    assert decoded.records[1].data["value"] == 0


def test_decoded_records_cannot_mutate_bound_evidence() -> None:
    decoded = parse_jsonl(b'{"nested":{"value":null},"values":[1,2]}\n')
    record = decoded.records[0].data
    with pytest.raises(TypeError):
        record["value"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        record["nested"]["value"] = 0  # type: ignore[index]
    assert record["values"] == (1, 2)


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        (b'{"value":1,"value":null}\n', "duplicate_key"),
        (b'{"nested":{"value":1,"value":null}}\n', "duplicate_key"),
        (b'{"value":NaN}\n', "nonfinite_number"),
        (b'{"value":Infinity}\n', "nonfinite_number"),
        (b'{"value":1e400}\n', "nonfinite_number"),
        (b"[]\n", "record_shape"),
        (b"null\n", "record_shape"),
        (b'{"value":\xff}\n', "json_encoding"),
        (b'{"value":', "json_syntax"),
        (b'{"value":1}\n\xef\xbb\xbf{"value":2}', "json_syntax"),
    ],
)
def test_jsonl_rejects_ambiguous_or_malformed_records(payload: bytes, category: str) -> None:
    with pytest.raises(OfflineParseError) as caught:
        parse_jsonl(payload)
    assert caught.value.category == category
    assert caught.value.line_number is not None


def test_jsonl_reports_failure_line_without_partial_success_or_raw_payload() -> None:
    with pytest.raises(OfflineParseError) as caught:
        parse_jsonl(b'{"value":1}\n{"do_not_echo_this_input":}\n')
    assert caught.value.line_number == 2
    assert "do_not_echo_this_input" not in str(caught.value)
    assert "do_not_echo_this_input" not in repr(caught.value)


@pytest.mark.parametrize(
    ("payload", "kwargs", "category"),
    [
        (b'{"value":1}\n', {"max_bytes": 2}, "byte_limit"),
        (b'{"value":1}\n', {"max_line_bytes": 2}, "line_limit"),
        (b"{}\n{}\n", {"max_records": 1}, "record_limit"),
        (b"{}\n", {"expected_sha256": "0" * 64}, "source_integrity"),
        (b'{"x":' + b"[" * 70 + b"0" + b"]" * 70 + b"}", {}, "nesting_limit"),
    ],
)
def test_jsonl_enforces_input_line_record_nesting_and_hash_limits(
    payload: bytes, kwargs: dict[str, object], category: str
) -> None:
    with pytest.raises(OfflineParseError) as caught:
        parse_jsonl(payload, **kwargs)  # type: ignore[arg-type]
    assert caught.value.category == category


def test_pretty_json_document_preserves_nested_arrays_and_unknown_values() -> None:
    payload = json.dumps({"rows": [{"value": 3.5}, {"value": None}]}, indent=2).encode()
    parsed = parse_json_object(payload, expected_sha256=hashlib.sha256(payload).hexdigest())
    rows = parsed.data["rows"]
    assert isinstance(rows, tuple)
    assert len(rows) == 2
    assert rows[0]["value"] == 3.5  # type: ignore[index]
    assert rows[1]["value"] is None  # type: ignore[index]
    assert "before" not in rows[0]  # type: ignore[operator]


@pytest.mark.parametrize(
    ("payload", "category"),
    [
        (b'{\n"value":1,\n"value":null\n}', "duplicate_key"),
        (b'{"nested":{"value":1,"value":null}}', "duplicate_key"),
        (b'{"value":NaN}', "nonfinite_number"),
        (b'{"value":1e400}', "nonfinite_number"),
        (b"[]", "record_shape"),
        (b'{"value":\xff}', "json_encoding"),
        (b'{\n"value":\n}', "json_syntax"),
    ],
)
def test_json_document_uses_same_fail_closed_rules_as_jsonl(payload: bytes, category: str) -> None:
    with pytest.raises(OfflineParseError) as caught:
        parse_json_object(payload)
    assert caught.value.category == category


def test_document_and_jsonl_share_limits_and_digest_binding() -> None:
    payload = json.dumps({"nested": {"values": [0, None]}}).encode()
    expected = hashlib.sha256(payload).hexdigest()
    document = parse_json_object(payload, expected_sha256=expected)
    records = parse_jsonl(payload, expected_sha256=expected)
    assert document.data == records.records[0].data
    assert document.source_sha256 == records.source_sha256
    for parser in (parse_json_object, parse_jsonl):
        for kwargs, category in (
            ({"expected_sha256": "0" * 64}, "source_integrity"),
            ({"max_bytes": 1}, "byte_limit"),
            ({"max_depth": 1}, "nesting_limit"),
        ):
            with pytest.raises(OfflineParseError) as caught:
                parser(payload, **kwargs)  # type: ignore[arg-type]
            assert caught.value.category == category


@pytest.mark.parametrize("parser", [parse_python_source, parse_jsonl, parse_json_object])
def test_input_type_limit_and_digest_configuration_are_explicit(parser: object) -> None:
    for payload, kwargs, category in (
        (bytearray(b"{}"), {}, "input_type"),
        (b"{}", {"max_bytes": True}, "invalid_limit"),
        (b"{}", {"expected_sha256": "not-a-digest"}, "invalid_digest"),
    ):
        with pytest.raises(OfflineParseError) as caught:
            parser(payload, **kwargs)  # type: ignore[operator]
        assert caught.value.category == category


@pytest.mark.parametrize("parser", [parse_jsonl, parse_json_object])
def test_integer_conversion_has_a_version_independent_bound(parser: object) -> None:
    payload = b'{"value":' + b"1" * 4_301 + b"}"
    with pytest.raises(OfflineParseError) as caught:
        parser(payload)  # type: ignore[operator]
    assert caught.value.category == "number_limit"
