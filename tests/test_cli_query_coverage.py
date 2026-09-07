from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_cli_standardization import _project

from game_learning_runtime.cli import (
    _cell,
    _human_output,
    _read_json_mapping,
    _table,
    main,
)
from game_learning_runtime.run_store import RunStatus, TrainingStore


def test_cli_rendering_helpers_cover_scalar_mapping_and_empty_rows() -> None:
    assert _cell(None) == ""
    assert _cell(True) == "true"
    assert _cell({"key": "value"}) == '{"key":"value"}'
    assert _cell("x" * 100).endswith("...")
    assert _table([]) == "(no rows)"
    table = _table([{"name": "one"}, {"name": "two", "extra": 2}])
    assert "name" in table and "extra" in table
    assert "items" in _human_output("items", [{"value": 1}])
    assert "field" in _human_output("mapping", {"ok": True})
    assert _human_output("scalar", "ready") == "scalar: ready"


def test_read_json_mapping_rejects_unsafe_and_invalid_inputs(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        _read_json_mapping(missing, label="payload")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        _read_json_mapping(invalid, label="payload")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError):
        _read_json_mapping(array, label="payload")


def test_cli_query_and_run_views_return_stable_json(tmp_path: Path, capsys: object) -> None:
    _project(tmp_path, trainer=[sys.executable, "-c", "print('train')"])
    store = TrainingStore(tmp_path / ".glr" / "runs.sqlite3")
    run = store.create_run(
        environment_id="example.adventure-v1", protocol_version="1.0", kind="training"
    )
    store.append_event(run.run_id, kind="episode.started", payload={"episode": 1})
    store.record_metric(run.run_id, name="reward", value=1.0)
    store.finish_run(run.run_id, status=RunStatus.SUCCEEDED, exit_code=0)

    assert main(["--project", str(tmp_path), "--json", "runs", "list"]) == 0
    assert json.loads(capsys.readouterr().out)["data"]
    assert main(["--project", str(tmp_path), "--json", "runs", "show", run.run_id]) == 0
    shown = json.loads(capsys.readouterr().out)["data"]
    assert shown["events"] and shown["metrics"]
    assert main(["--project", str(tmp_path), "--json", "query", "entities", "--world", "w"]) == 0
    assert json.loads(capsys.readouterr().out)["data"] == []
    assert main(["--project", str(tmp_path), "--json", "query", "routes", "--world", "w"]) == 0
    assert json.loads(capsys.readouterr().out)["data"] == []
    assert main(["--project", str(tmp_path), "--json", "query", "research"]) == 0
    assert json.loads(capsys.readouterr().out)["data"] == []


def test_cli_knowledge_export_and_import_empty_bundle(tmp_path: Path, capsys: object) -> None:
    _project(tmp_path, trainer=[sys.executable, "-c", "print('train')"])
    exported = tmp_path / "knowledge.json"
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "--json",
                "knowledge",
                "export",
                "--output",
                str(exported),
            ]
        )
        == 0
    )
    assert exported.is_file()
    assert json.loads(capsys.readouterr().out)["data"]["entities"] == 0
    assert (
        main(
            [
                "--project",
                str(tmp_path),
                "--json",
                "knowledge",
                "import",
                "--input",
                str(exported),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["data"]["entities"] == 0
