use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use serde_json::Value;
use tempfile::TempDir;

/// A runtime role that parks on a retryable state until the host is usable.
///
/// The role owns the game semantics: it publishes one
/// `glr.environment-readiness.v1` receipt per invocation and refuses with a
/// named exit code while it cannot serve. GLR owns how long to keep asking.
const ROLE: &str = r#"
import json
import os
import sys
from pathlib import Path

ready_after = int(sys.argv[1])
mode = sys.argv[2]
run_dir = Path(os.environ["GLR_RUN_DIR"])
counter = run_dir / "attempts.txt"
attempt = int(counter.read_text(encoding="utf-8")) + 1 if counter.is_file() else 1
counter.write_text(str(attempt), encoding="utf-8")
configured = os.environ.get("GLR_READINESS_PATH")
print(f"attempt={attempt} mode={mode} window={os.environ.get('GLR_READINESS_ATTEMPT')}")
if configured is None:
    print("no declared readiness window")
    sys.exit(63)
receipt = Path(configured)
if mode == "unreported":
    sys.exit(17)
if mode == "unavailable":
    state, exit_code = "unavailable", 19
elif mode == "inconsistent":
    state, exit_code = "ready", 23
elif attempt >= ready_after:
    state, exit_code = "ready", 0
else:
    state, exit_code = "not_ready", 63
receipt.write_text(
    json.dumps(
        {
            "schema_version": "glr.environment-readiness.v1",
            "state": state,
            "reason": f"attempt {attempt}",
            "checked_at_ns": attempt,
        }
    ),
    encoding="utf-8",
)
sys.exit(exit_code)
"#;

fn project(mode: &str, ready_after: u32, window: Option<&str>) -> TempDir {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::create_dir(root.join("bridge")).unwrap();
    fs::write(root.join("runtime_role.py"), ROLE.trim_start()).unwrap();
    let window = window
        .map(|value| format!("\n[runtime.readiness]\n{value}\n"))
        .unwrap_or_default();
    fs::write(
        root.join("glr-project.toml"),
        format!(
            r#"
schema_version = "glr.project.v1"
environment_id = "example.environment-v1"
environment_family = "example-family"
protocol_version = "1.0"
data_dir = ".glr"
bridge_path = "bridge"
[runtime]
argv = ["python", "runtime_role.py", "{ready_after}", "{mode}"]
[trainer]
argv = ["python", "-c", "print('train')"]
[player]
argv = ["python", "play.py", "{{bundle}}"]
{window}"#
        ),
    )
    .unwrap();
    directory
}

fn start(root: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_glr"))
        .env("GLR_NO_UPDATE_CHECK", "1")
        .arg("--project")
        .arg(root)
        .args(["--json", "runtime", "start"])
        .output()
        .unwrap()
}

fn envelope(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap()
}

fn attempts(root: &Path, run_id: &str) -> u32 {
    fs::read_to_string(root.join(".glr/runs").join(run_id).join("attempts.txt"))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn runtime_start_parks_and_reinvokes_until_the_host_reports_ready() {
    let directory = project(
        "park",
        2,
        Some("timeout_seconds = 30\npoll_interval_seconds = 0.01"),
    );
    let output = start(directory.path());
    assert_eq!(output.status.code(), Some(0));
    let data = envelope(&output)["data"].clone();
    assert_eq!(data["status"], "succeeded");
    assert_eq!(data["readiness"]["verdict"], "succeeded");
    assert_eq!(data["readiness"]["exhausted"], false);
    assert_eq!(
        data["readiness"]["attempts"][0]["readiness"]["state"],
        "not_ready"
    );
    assert_eq!(
        data["readiness"]["attempts"][1]["readiness"]["state"],
        "ready"
    );
    assert_eq!(
        attempts(directory.path(), data["run_id"].as_str().expect("run id")),
        2
    );
}

#[test]
fn runtime_start_records_a_booting_host_apart_from_a_crash() {
    let parked = project(
        "park",
        99,
        Some("timeout_seconds = 0.05\npoll_interval_seconds = 0.01"),
    );
    let refused = project("unreported", 99, Some("timeout_seconds = 30"));

    let parked_output = start(parked.path());
    let refused_output = start(refused.path());
    assert_eq!(parked_output.status.code(), Some(78));
    assert_eq!(refused_output.status.code(), Some(17));
    let parked_data = envelope(&parked_output)["data"].clone();
    let refused_data = envelope(&refused_output)["data"].clone();
    assert_eq!(parked_data["status"], "failed");
    assert_eq!(parked_data["exit_code"], 78);
    assert_eq!(parked_data["readiness"]["verdict"], "not_ready");
    assert_eq!(parked_data["readiness"]["exhausted"], true);
    assert_eq!(refused_data["readiness"]["verdict"], "unreported");
    assert_eq!(refused_data["readiness"]["exhausted"], false);
    assert!(
        attempts(
            parked.path(),
            parked_data["run_id"].as_str().expect("run id")
        ) >= 1
    );
    // A crash inside a declared window is still never retried.
    assert_eq!(
        attempts(
            refused.path(),
            refused_data["run_id"].as_str().expect("run id")
        ),
        1
    );

    let shown = Command::new(env!("CARGO_BIN_EXE_glr"))
        .env("GLR_NO_UPDATE_CHECK", "1")
        .arg("--project")
        .arg(parked.path())
        .args([
            "--json",
            "runs",
            "show",
            parked_data["run_id"].as_str().unwrap(),
        ])
        .output()
        .unwrap();
    let data = envelope(&shown)["data"].clone();
    let events = data["events"].as_array().unwrap();
    assert_eq!(events.last().unwrap()["kind"], "readiness.outcome");
    assert_eq!(events.last().unwrap()["payload"]["verdict"], "not_ready");
    assert_eq!(events.last().unwrap()["payload"]["exhausted"], true);
    assert_eq!(
        events.last().unwrap()["payload"]["attempts"][0]["readiness"]["schema_version"],
        "glr.environment-readiness.v1"
    );
}

#[test]
fn runtime_start_without_a_declared_window_stays_one_invocation() {
    let directory = project("park", 99, None);
    let output = start(directory.path());
    assert_eq!(output.status.code(), Some(63));
    let data = envelope(&output)["data"].clone();
    assert_eq!(data["status"], "failed");
    assert_eq!(data["exit_code"], 63);
    assert!(data.get("readiness").is_none());
    let run_id = data["run_id"].as_str().expect("run id");
    assert_eq!(attempts(directory.path(), run_id), 1);

    let shown = Command::new(env!("CARGO_BIN_EXE_glr"))
        .env("GLR_NO_UPDATE_CHECK", "1")
        .arg("--project")
        .arg(directory.path())
        .args(["--json", "runs", "show", run_id])
        .output()
        .unwrap();
    let events = envelope(&shown)["data"]["events"].clone();
    assert_eq!(events.as_array().map(Vec::len), Some(0));
}

#[test]
fn runtime_start_never_retries_a_terminal_receipt() {
    for (mode, exit_code, verdict) in [
        ("unavailable", 19, "unavailable"),
        ("inconsistent", 23, "inconsistent"),
    ] {
        let directory = project(mode, 99, Some("timeout_seconds = 30"));
        let output = start(directory.path());
        assert_eq!(output.status.code(), Some(exit_code));
        let data = envelope(&output)["data"].clone();
        assert_eq!(data["readiness"]["verdict"], verdict);
        assert_eq!(
            attempts(directory.path(), data["run_id"].as_str().expect("run id")),
            1
        );
    }
}

#[test]
fn runtime_start_rejects_an_unbounded_or_inverted_window() {
    for (window, expected) in [
        (
            "timeout_seconds = 0",
            "project.runtime.readiness.timeout_seconds",
        ),
        (
            "timeout_seconds = 3601",
            "project.runtime.readiness.timeout_seconds",
        ),
        (
            "timeout_seconds = 10\npoll_interval_seconds = 11",
            "project.runtime.readiness.poll_interval_seconds",
        ),
        ("timeout_seconds = 10\nsettle = 3", "settle"),
    ] {
        let directory = project("park", 1, Some(window));
        let output = start(directory.path());
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(expected), "{error}");
    }
}
