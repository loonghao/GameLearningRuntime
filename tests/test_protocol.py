from __future__ import annotations

from pathlib import Path

from grpc_tools import protoc

from game_learning_runtime import protocol_path


def test_packaged_protocol_compiles(tmp_path: Path) -> None:
    with protocol_path() as schema:
        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"--proto_path={schema.parents[2]}",
                f"--python_out={tmp_path}",
                f"--grpc_python_out={tmp_path}",
                str(schema),
            ]
        )

    assert result == 0
    assert (tmp_path / "glr" / "v1" / "runtime_pb2.py").is_file()
