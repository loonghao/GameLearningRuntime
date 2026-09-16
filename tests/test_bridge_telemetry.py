import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from game_learning_runtime import BridgeTelemetry
from game_learning_runtime.bridge_telemetry import SCHEMA


@pytest.fixture
def server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            batch = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((batch, self.headers["Authorization"]))
            if batch["batch_id"] == "batch-redirect":
                self.send_response(307)
                self.send_header("Location", "/forbidden")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            if batch["batch_id"] == "batch-large":
                self.wfile.write(b"x" * 65537)
            else:
                receipt = {
                    key: batch[key] for key in ("schema_version", "run_id", "source", "batch_id")
                }
                if batch["batch_id"] == "batch-wrong":
                    receipt["run_id"] = "run-other"
                self.wfile.write(json.dumps(receipt).encode())

    host = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=host.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{host.server_port}/api/v1/telemetry", requests
    host.shutdown()
    host.server_close()
    thread.join()


def test_http_batches_preserve_retry_identity_without_mutating_caller(server, monkeypatch):
    url, requests = server
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    bridge = BridgeTelemetry("run-1", "bridge.test", url=url, token="test")
    payload = {"state": "ready"}
    batch = bridge.prepare(events=[{"kind": "bridge.status", "payload": payload}])
    payload["state"] = "changed"
    assert bridge.send(batch)["batch_id"] == batch["batch_id"]
    assert bridge.send(batch)["batch_id"] == batch["batch_id"]
    assert requests[0] == requests[1]
    assert requests[0][0]["events"][0]["payload"]["state"] == "ready"
    assert requests[0][1] == "Bearer test"


def test_http_rejects_redirects_oversized_or_mismatched_receipts(server):
    from urllib.error import HTTPError

    url, requests = server
    bridge = BridgeTelemetry("run-1", "bridge.test", url=url, token="test")
    for name, error in [("redirect", HTTPError), ("large", ValueError), ("wrong", ValueError)]:
        batch = bridge.prepare(metrics=[{"name": "fps", "value": 60}], batch_id=f"batch-{name}")
        with pytest.raises(error):
            bridge.send(batch)
    assert len(requests) == 3


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/api/v1/telemetry",
        "http://localhost/other",
        "http://user@localhost/api/v1/telemetry",
        "http://localhost/api/v1/telemetry?token=x",
    ],
)
def test_nonlocal_or_ambiguous_endpoints_are_rejected(url):
    with pytest.raises(ValueError):
        BridgeTelemetry("run-1", "bridge.test", url=url, token="secret")


def test_configuration_limits_and_environment_binding(monkeypatch, tmp_path):
    for key in (
        "GLR_RUN_ID",
        "GLR_TELEMETRY_URL",
        "GLR_TELEMETRY_TOKEN",
        "GLR_CLI_PATH",
        "GLR_PROJECT_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    assert BridgeTelemetry.from_env("bridge.test") is None
    monkeypatch.setenv("GLR_RUN_ID", "run-1")
    assert BridgeTelemetry.from_env("bridge.test") is None
    for timeout in (0, 31, float("nan")):
        with pytest.raises(ValueError):
            BridgeTelemetry("run-1", "bridge.test", timeout=timeout)
    with pytest.raises(ValueError):
        BridgeTelemetry("run-1", "bridge.test", cli_path="relative", project_root=".")
    monkeypatch.setenv("GLR_CLI_PATH", str(tmp_path / "glr"))
    monkeypatch.setenv("GLR_PROJECT_ROOT", str(tmp_path))
    assert BridgeTelemetry.from_env("bridge.test").cli_path == str(tmp_path / "glr")
    monkeypatch.setenv("GLR_TELEMETRY_URL", "http://localhost:7432/api/v1/telemetry")
    with pytest.raises(ValueError):
        BridgeTelemetry.from_env("bridge.test")
    monkeypatch.setenv("GLR_TELEMETRY_TOKEN", "test")
    bridge = BridgeTelemetry.from_env("bridge.test")
    assert bridge.url is not None
    with pytest.raises(ValueError, match=r"1\.\.64"):
        bridge.prepare()
    with pytest.raises(ValueError):
        bridge.prepare(metrics=[{"name": "fps", "value": float("nan")}])
    with pytest.raises(ValueError, match="64 KiB"):
        bridge.prepare(events=[{"kind": "bridge.state", "payload": {"text": "a" * 65536}}])
    with pytest.raises(ValueError, match="run and source"):
        bridge.send({"run_id": "run-other"})


def test_cli_transport_uses_fixed_argv_stdin_and_matching_receipt(tmp_path, monkeypatch):
    calls = []

    def execute(argv, **kwargs):
        calls.append((argv, kwargs))
        batch = json.loads(kwargs["input"])
        receipt = {key: batch[key] for key in ("schema_version", "run_id", "source", "batch_id")}
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"data": {"receipts": [receipt]}}).encode()
        )

    monkeypatch.setattr(subprocess, "run", execute)
    bridge = BridgeTelemetry(
        "run-1", "bridge.test", cli_path=str(tmp_path / "glr"), project_root=str(tmp_path)
    )
    batch = bridge.prepare(metrics=[{"name": "fps", "value": 60}])
    assert bridge.send(batch)["schema_version"] == SCHEMA
    assert calls[0][0][-4:] == ["telemetry", "ingest", "--file", "-"]
    assert calls[0][1]["timeout"] == 5.0
    assert "shell" not in calls[0][1]
