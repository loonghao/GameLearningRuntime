//! A project declares the environment its roles receive.
//!
//! `glr.project.v1` accepts a project-wide `environment` table and a per-role
//! table that overrides it key by key. Literals pass through, `${NAME}` is
//! interpolated from the process environment, and a reference that resolves to
//! nothing refuses the run instead of handing a role an empty string. These
//! tests pin the contract at the boundary a role actually observes: the
//! environment of the child process the CLI spawned.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use tempfile::TempDir;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_glr"))
}

/// Rewrite the manifest, preserving the probe argv written by `create_project`.
fn write_manifest(project: &Path, mutate: impl FnOnce(&mut Value)) {
    let manifest = project.join("glr-project.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    mutate(&mut value);
    fs::write(&manifest, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
}

fn create_project() -> TempDir {
    let temporary = tempfile::tempdir().unwrap();
    fs::create_dir(temporary.path().join("bridge")).unwrap();
    let probe = std::env::current_exe()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let argv = json!([
        probe,
        "--ignored",
        "--exact",
        "role_environment_probe_child"
    ]);
    let project = json!({
        "schema_version": "glr.project.v1",
        "environment_id": "example.declared-env-v1",
        "environment_family": "test",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": argv},
        "trainer": {"argv": argv},
        "player": {"argv": [binary().to_string_lossy(), "--version"]},
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
    run_with(project, arguments, &[])
}

fn run_with(project: &Path, arguments: &[&str], environment: &[(&str, &str)]) -> Output {
    let mut command = Command::new(binary());
    command.arg("--project").arg(project).arg("--json");
    for (name, value) in environment {
        command.env(name, value);
    }
    command.args(arguments).output().unwrap()
}

fn success(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn observed(project: &Path, run_id: &str) -> Value {
    let path = project
        .join(".glr/runs")
        .join(run_id)
        .join("role-environment-observed.json");
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn role<'a>(output: &'a Value, name: &str) -> &'a Value {
    output["roles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["role"] == name)
        .unwrap_or_else(|| panic!("doctor reported no {name} role"))
}

/// Records the declared environment the capture recorder actually received.
#[test]
#[ignore]
fn capture_environment_probe_child() {
    let run_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").expect("GLR_RUN_DIR"));
    let observed = json!({
        "SYNTHETIC_MODE": std::env::var("SYNTHETIC_MODE").ok(),
        "SYNTHETIC_DATASET": std::env::var("SYNTHETIC_DATASET").ok(),
    });
    fs::write(
        run_dir.join("capture-environment-observed.json"),
        serde_json::to_vec_pretty(&observed).unwrap(),
    )
    .unwrap();
}

/// Records the declared environment the runtime role actually received.
#[test]
#[ignore]
fn role_environment_probe_child() {
    let run_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").expect("GLR_RUN_DIR"));
    let observed = json!({
        "SYNTHETIC_MODE": std::env::var("SYNTHETIC_MODE").ok(),
        "SYNTHETIC_DATASET": std::env::var("SYNTHETIC_DATASET").ok(),
        "SYNTHETIC_ENDPOINT": std::env::var("SYNTHETIC_ENDPOINT").ok(),
    });
    fs::write(
        run_dir.join("role-environment-observed.json"),
        serde_json::to_vec_pretty(&observed).unwrap(),
    )
    .unwrap();
}

#[test]
fn a_role_receives_the_declared_environment() {
    let project = create_project();
    write_manifest(project.path(), |value| {
        value["environment"] = json!({
            "SYNTHETIC_MODE": "synthetic",
            "SYNTHETIC_DATASET": "${SYNTHETIC_HOST}/v1",
        });
    });

    let started = success(&run_with(
        project.path(),
        &["runtime", "start"],
        &[("SYNTHETIC_HOST", "datasets")],
    ));
    let run_id = started["data"]["run_id"].as_str().unwrap();
    let observed = observed(project.path(), run_id);

    assert_eq!(observed["SYNTHETIC_MODE"], "synthetic");
    assert_eq!(observed["SYNTHETIC_DATASET"], "datasets/v1");
}

#[test]
fn the_role_table_overrides_the_project_table_key_by_key() {
    let project = create_project();
    write_manifest(project.path(), |value| {
        value["environment"] = json!({"SYNTHETIC_MODE": "project", "SYNTHETIC_DATASET": "shared"});
        value["runtime"]["environment"] = json!({"SYNTHETIC_MODE": "runtime"});
    });

    let started = success(&run(project.path(), &["runtime", "start"]));
    let run_id = started["data"]["run_id"].as_str().unwrap();
    let observed = observed(project.path(), run_id);

    assert_eq!(observed["SYNTHETIC_MODE"], "runtime");
    assert_eq!(observed["SYNTHETIC_DATASET"], "shared");
}

#[test]
fn the_process_environment_outranks_the_declared_table() {
    let project = create_project();
    write_manifest(project.path(), |value| {
        value["environment"] = json!({"SYNTHETIC_MODE": "from-manifest"});
    });

    let started = success(&run_with(
        project.path(),
        &["runtime", "start"],
        &[("SYNTHETIC_MODE", "from-operator")],
    ));
    let run_id = started["data"]["run_id"].as_str().unwrap();

    assert_eq!(
        observed(project.path(), run_id)["SYNTHETIC_MODE"],
        "from-operator"
    );
}

#[test]
fn a_project_that_declares_nothing_changes_nothing() {
    let project = create_project();

    let started = success(&run(project.path(), &["runtime", "start"]));
    let run_id = started["data"]["run_id"].as_str().unwrap();
    let observed = observed(project.path(), run_id);

    assert_eq!(observed["SYNTHETIC_MODE"], Value::Null);
    assert_eq!(observed["SYNTHETIC_DATASET"], Value::Null);
}

#[test]
fn doctor_reports_the_resolved_environment_of_every_configured_role() {
    let project = create_project();
    write_manifest(project.path(), |value| {
        value["environment"] = json!({"SYNTHETIC_MODE": "synthetic"});
    });

    let output = success(&run(project.path(), &["doctor"]));

    let runtime = role(&output["data"], "runtime");
    assert_eq!(runtime["environment"]["ready"], true);
    assert_eq!(
        runtime["environment"]["variables"],
        json!([{"name": "SYNTHETIC_MODE", "source": "literal", "secret": false,
                "value": "synthetic"}])
    );
    assert_eq!(runtime["environment"]["unresolved"], json!([]));
    // An unconfigured role never runs, so it reports no environment at all.
    assert!(role(&output["data"], "planner")["environment"].is_null());
}

#[test]
fn doctor_fails_when_a_declared_variable_cannot_resolve() {
    let project = create_project();
    write_manifest(project.path(), |value| {
        value["environment"] = json!({"SYNTHETIC_DATASET": "${SYNTHETIC_MISSING_HOST}"});
    });

    let output = run(project.path(), &["doctor"]);

    assert_eq!(output.status.code(), Some(4));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["data"]["ready"], false);
    assert_eq!(
        role(&report["data"], "runtime")["environment"]["unresolved"],
        json!([{"name": "SYNTHETIC_DATASET", "missing": ["SYNTHETIC_MISSING_HOST"]}])
    );
}

#[test]
fn a_run_is_refused_before_its_role_starts() {
    let project = create_project();
    write_manifest(project.path(), |value| {
        value["trainer"]["environment"] = json!({"SYNTHETIC_DATASET": "${SYNTHETIC_MISSING_HOST}"});
    });

    let output = run(project.path(), &["train", "--no-capture"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("SYNTHETIC_DATASET") && stderr.contains("SYNTHETIC_MISSING_HOST"),
        "the refusal must name the variable and what is missing: {stderr}"
    );
    // Nothing was launched, so no run wrote a probe file.
    assert!(!project.path().join(".glr/runs").exists());
}

#[test]
fn the_manifest_rejects_a_reserved_glr_key_before_any_process_starts() {
    let project = create_project();
    write_manifest(project.path(), |value| {
        value["environment"] = json!({"GLR_RUN_ID": "forged"});
    });

    let output = run(project.path(), &["doctor"]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("GLR_RUN_ID"),
        "the refusal must name the reserved key: {stderr}"
    );
}

#[test]
fn the_capture_recorder_receives_the_project_wide_table() {
    let project = create_project();
    let probe = std::env::current_exe()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    write_manifest(project.path(), |value| {
        value["environment"] = json!({"SYNTHETIC_MODE": "synthetic"});
        value["runtime"]["environment"] = json!({"SYNTHETIC_MODE": "runtime"});
        value["capture"] = json!({
            "argv": [probe, "--ignored", "--exact", "capture_environment_probe_child"],
            "required": false,
            "stop": "terminate",
            "video_file": "capture.mp4",
            "index_file": "capture-index.jsonl",
            "codec": "h264",
            "frame_rate": 12,
            "width": 640,
            "height": 360
        });
    });

    let trained = success(&run(project.path(), &["train"]));
    let run_id = trained["data"]["run_id"].as_str().unwrap();
    let path = project
        .path()
        .join(".glr/runs")
        .join(run_id)
        .join("capture-environment-observed.json");
    let observed: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();

    // The recorder is not a manifest role, so it gets the project-wide table and
    // not the runtime override. Both entry points behave this way.
    assert_eq!(observed["SYNTHETIC_MODE"], "synthetic");
}

#[test]
fn a_run_records_the_environment_its_role_received() {
    let project = create_project();
    write_manifest(project.path(), |value| {
        value["environment"] = json!({
            "SYNTHETIC_MODE": "synthetic",
            "SYNTHETIC_TOKEN": "synthetic-secret-value",
        });
    });

    let started = success(&run(project.path(), &["runtime", "start"]));
    let run_id = started["data"]["run_id"].as_str().unwrap();
    let recorded = started["data"]["metadata"]["role_environment"].clone();

    assert_eq!(recorded["ready"], true);
    let variables = recorded["variables"].as_array().unwrap();
    let mode = variables
        .iter()
        .find(|entry| entry["name"] == "SYNTHETIC_MODE")
        .unwrap();
    assert_eq!(mode["value"], "synthetic");
    // A secret is recorded as received, never as content.
    let token = variables
        .iter()
        .find(|entry| entry["name"] == "SYNTHETIC_TOKEN")
        .unwrap();
    assert_eq!(token["secret"], true);
    assert!(token.get("value").is_none());
    assert!(!recorded.to_string().contains("synthetic-secret-value"));
    assert!(observed(project.path(), run_id)["SYNTHETIC_MODE"] == "synthetic");
}
