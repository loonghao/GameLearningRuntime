//! `glr host` must let an externally driven loop publish into a GLR-owned run.
//!
//! An adapter whose policy loop is its own long-lived process needs a run and an
//! ingest binding without pretending to be the project trainer. These tests pin
//! the published binding, the fact that the token reaches exactly the hosted
//! child, and the fact that it never lands in the run record or the log.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use tempfile::TempDir;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_glr"))
}

fn create_project() -> TempDir {
    let temporary = tempfile::tempdir().unwrap();
    fs::create_dir(temporary.path().join("bridge")).unwrap();
    let executable = binary().to_string_lossy().into_owned();
    let project = json!({
        "schema_version": "glr.project.v1",
        "environment_id": "example.host-v1",
        "environment_family": "test",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": [executable, "--version"]},
        "trainer": {"argv": [executable, "--version"]},
        "player": {"argv": [executable, "--version"]},
        "researcher": null,
        "planner": null,
        "evaluator": null,
        "capture": null
    });
    fs::write(
        temporary.path().join("glr-project.json"),
        serde_json::to_vec_pretty(&project).unwrap(),
    )
    .unwrap();
    temporary
}

fn run(project: &Path, arguments: &[&str]) -> Output {
    Command::new(binary())
        .arg("--project")
        .arg(project)
        .arg("--json")
        .args(arguments)
        .output()
        .unwrap()
}

fn output_json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|_| {
        panic!(
            "stdout is not JSON: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn run_id_of(output: &Value) -> &str {
    output["data"]["run_id"].as_str().unwrap()
}

/// The recording child: a hosted loop that publishes one telemetry batch.
///
/// It is `#[ignore]`d so the harness can name it as the hosted program through
/// `--exact`, exactly like the role-contract probes.
#[test]
#[ignore]
fn hosted_child() {
    let url = std::env::var("GLR_TELEMETRY_URL").expect("GLR_TELEMETRY_URL");
    let token = std::env::var("GLR_TELEMETRY_TOKEN").expect("GLR_TELEMETRY_TOKEN");
    let run_id = std::env::var("GLR_RUN_ID").expect("GLR_RUN_ID");
    let run_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").expect("GLR_RUN_DIR"));

    let batch = json!({
        "schema_version": "glr.bridge-telemetry.v1",
        "run_id": run_id,
        "source": "loop.external",
        "batch_id": "batch-0001",
        "events": [
            {"kind": "bridge.progress", "step_id": 1, "payload": {"fraction": 0.5, "label": "hosted"}}
        ],
        "metrics": [{"name": "loop.steps", "value": 1.0, "step_id": 1}]
    })
    .to_string();

    let authority = url
        .strip_prefix("http://")
        .expect("host ingest url is http");
    let (host, path) = authority
        .split_once('/')
        .expect("host ingest url carries a path");
    let mut stream = TcpStream::connect(host).expect("connect to ingest endpoint");
    let request = format!(
        "POST /{path} HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {token}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{batch}",
        batch.len()
    );
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    fs::write(run_dir.join("hosted-response.txt"), &response).unwrap();
    assert!(
        response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.1 201"),
        "ingest refused the batch: {response}"
    );
}

/// A hosted program that records the binding it received and then exits.
///
/// The token itself is written to the run directory on purpose: it gives the
/// leak test a real secret to search for, so "the credential was not persisted"
/// is an assertion about bytes rather than about a length.
#[test]
#[ignore]
fn binding_probe_child() {
    let run_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").expect("GLR_RUN_DIR"));
    let token = std::env::var("GLR_TELEMETRY_TOKEN").expect("GLR_TELEMETRY_TOKEN");
    fs::write(
        run_dir.join("binding-observed.json"),
        serde_json::to_vec_pretty(&json!({
            "has_url": std::env::var_os("GLR_TELEMETRY_URL").is_some(),
            "token": token,
            "run_id": std::env::var("GLR_RUN_ID").unwrap(),
        }))
        .unwrap(),
    )
    .unwrap();
}

fn probe_argv(test_name: &str) -> Vec<String> {
    vec![
        std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        "--ignored".into(),
        "--exact".into(),
        test_name.into(),
    ]
}

fn host(project: &Path, test_name: &str, extra: &[&str]) -> Output {
    let probe = probe_argv(test_name);
    let mut arguments: Vec<&str> = vec!["host"];
    arguments.extend_from_slice(extra);
    arguments.push("--");
    Command::new(binary())
        .arg("--project")
        .arg(project)
        .arg("--json")
        .args(&arguments)
        .args(&probe)
        .output()
        .unwrap()
}

#[test]
fn an_external_loop_receives_an_ingest_binding_inside_a_glr_run() {
    let project = create_project();
    let output = host(project.path(), "binding_probe_child", &[]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt = output_json(&output);
    assert_eq!(receipt["data"]["kind"], "hosted");
    assert_eq!(receipt["data"]["telemetry"]["available"], true);
    let run_id = run_id_of(&receipt);

    let observed: Value = serde_json::from_slice(
        &fs::read(
            project
                .path()
                .join(".glr/runs")
                .join(run_id)
                .join("binding-observed.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(observed["has_url"], true);
    assert_eq!(observed["run_id"], run_id);
    // A generated token is 64 hex characters; the contract floor is 32.
    let token = observed["token"].as_str().unwrap();
    assert!(token.len() >= 32);
    assert!(
        token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
    );
}

#[test]
fn an_external_loop_can_publish_a_batch_into_a_live_run() {
    let project = create_project();
    let output = host(project.path(), "hosted_child", &[]);
    let receipt = output_json(&output);
    assert_eq!(receipt["data"]["telemetry"]["available"], true);
    let run_id = run_id_of(&receipt).to_string();

    let response = fs::read_to_string(
        project
            .path()
            .join(".glr/runs")
            .join(&run_id)
            .join("hosted-response.txt"),
    )
    .unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");

    // The published batch is durable and attributed to the hosted source.
    let trace = run(project.path(), &["runs", "trace", &run_id]);
    let trace = output_json(&trace);
    let rendered = serde_json::to_string(&trace).unwrap();
    assert!(
        rendered.contains("loop.external") || rendered.contains("loop.steps"),
        "the hosted batch is not visible in the run trace: {rendered}"
    );
}

#[test]
fn the_ingest_credential_never_reaches_the_run_record_or_the_log() {
    let project = create_project();
    let output = host(project.path(), "binding_probe_child", &[]);
    let receipt = output_json(&output);
    let run_id = run_id_of(&receipt).to_string();
    let run_dir = project.path().join(".glr/runs").join(&run_id);

    // The child hands us the real token, so we can search every surface GLR
    // writes for it: the CLI receipt, the hosted log, and the run record.
    let observed: Value =
        serde_json::from_slice(&fs::read(run_dir.join("binding-observed.json")).unwrap()).unwrap();
    let token = observed["token"].as_str().unwrap().to_string();
    assert!(token.len() >= 32, "the child received a real binding");

    let receipt_text = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        !receipt_text.contains(&token),
        "the ingest token leaked into the CLI receipt"
    );
    let log_text = fs::read_to_string(run_dir.join("hosted.log")).unwrap_or_default();
    assert!(
        !log_text.contains(&token),
        "the ingest token leaked into the hosted log"
    );
    let record = run(project.path(), &["runs", "list"]);
    let record_text = String::from_utf8_lossy(&record.stdout).to_string();
    assert!(
        !record_text.contains(&token),
        "the ingest token leaked into the run record"
    );
}

#[test]
fn hosting_can_be_run_without_opening_an_ingest_endpoint() {
    let project = create_project();
    let output = host(
        project.path(),
        "no_binding_probe_child",
        &["--no-telemetry"],
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt = output_json(&output);
    assert_eq!(receipt["data"]["telemetry"]["available"], false);
}

#[test]
#[ignore]
fn no_binding_probe_child() {
    assert!(std::env::var_os("GLR_TELEMETRY_URL").is_none());
    assert!(std::env::var_os("GLR_TELEMETRY_TOKEN").is_none());
    let run_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").expect("GLR_RUN_DIR"));
    fs::write(run_dir.join("binding-observed.json"), b"{}").unwrap();
}
