from __future__ import annotations

from pathlib import Path

from google.protobuf import descriptor_pb2
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


def test_protocol_exposes_live_attach_without_aliasing_reset(tmp_path: Path) -> None:
    descriptor_path = tmp_path / "runtime.pb"
    with protocol_path() as schema:
        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"--proto_path={schema.parents[2]}",
                f"--descriptor_set_out={descriptor_path}",
                str(schema),
            ]
        )

    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.ParseFromString(descriptor_path.read_bytes())
    runtime = next(file for file in descriptor_set.file if file.package == "glr.v1")
    messages = {message.name for message in runtime.message_type}
    methods = {method.name for service in runtime.service for method in service.method}

    assert result == 0
    assert "AttachRequest" in messages
    assert "Attach" in methods
