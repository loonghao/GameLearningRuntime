from pathlib import Path

from game_learning_runtime.qa import run_qa


def test_qa_creates_dated_json_and_html(tmp_path):
    result = run_qa(
        "inspect whole game",
        output_root=tmp_path,
        commands=[("smoke", ["python", "-c", "print('ok')"])],
    )
    assert result.status == "passed"
    files = list(Path(result.output_dir).iterdir())
    assert {path.name for path in files} == {"result.json", "index.html"}
    assert "inspect whole game" in (Path(result.output_dir) / "index.html").read_text()
