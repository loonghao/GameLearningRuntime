use std::fs;
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
    fs::create_dir_all(temporary.path().join("config/contexts")).unwrap();
    fs::write(
        temporary.path().join("config/training.json"),
        br#"{"schema_version":"glr.training.v1","algorithm":"ppo"}"#,
    )
    .unwrap();
    fs::write(
        temporary.path().join("config/contexts/native.toml"),
        r#"schema_version = "glr.run-context.v1"
context_id = "league-native-100024"
environment_id = "example.context-v1"
protocol_version = "1.0"

[labels]
season = "ranked-2026"
ruleset = "standard"

[[inputs]]
owner = "training"
path = "config/training.json"
schema_version = "glr.training.v1"
"#,
    )
    .unwrap();
    let executable = binary().to_string_lossy().into_owned();
    let probe = std::env::current_exe()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let project = json!({
        "schema_version": "glr.project.v1",
        "environment_id": "example.context-v1",
        "environment_family": "test",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": [executable, "--version"]},
        "trainer": {"argv": [probe, "--ignored", "--exact", "context_probe_child"]},
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

fn set_trainer(project: &Path, test_name: &str) {
    let manifest = project.join("glr-project.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    value["trainer"]["argv"] = json!([
        std::env::current_exe().unwrap().to_string_lossy(),
        "--ignored",
        "--exact",
        test_name
    ]);
    fs::write(manifest, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
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

#[test]
#[ignore]
fn context_probe_child() {
    let context = std::env::var("GLR_RUN_CONTEXT").expect("GLR_RUN_CONTEXT");
    let digest = std::env::var("GLR_RUN_CONTEXT_SHA256").expect("context digest");
    let context_value: Value = serde_json::from_str(&context).unwrap();
    assert_eq!(context_value["context_sha256"], digest);
    let run_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").expect("run dir"));
    fs::write(run_dir.join("context-observed.json"), context).unwrap();
}

#[test]
#[ignore]
fn no_context_probe_child() {
    assert!(std::env::var_os("GLR_RUN_CONTEXT").is_none());
    assert!(std::env::var_os("GLR_RUN_CONTEXT_SHA256").is_none());
}

#[test]
fn selected_context_reaches_the_role_and_is_persisted_with_the_run() {
    let project = create_project();
    let trained = success(&run(
        project.path(),
        &[
            "--context",
            "config/contexts/native.toml",
            "train",
            "--no-capture",
        ],
    ));
    let run_id = trained["data"]["run_id"].as_str().unwrap();
    let run_dir = project.path().join(".glr/runs").join(run_id);
    let persisted: Value =
        serde_json::from_slice(&fs::read(run_dir.join("run-context.json")).unwrap()).unwrap();
    let observed: Value =
        serde_json::from_slice(&fs::read(run_dir.join("context-observed.json")).unwrap()).unwrap();
    assert_eq!(persisted, observed);
    assert_eq!(persisted["schema_version"], "glr.run-context.v1");
    assert_eq!(persisted["context_id"], "league-native-100024");
    assert_eq!(persisted["labels"]["season"], "ranked-2026");
    assert_eq!(persisted["inputs"][0]["owner"], "training");
    assert_eq!(persisted["inputs"][0]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(persisted["context_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(
        persisted["context_sha256"],
        "8a5f395b6a8bdedf913ee9063c468a69e8027eb2f27b067c1862878272582140"
    );
}

#[test]
fn an_unselected_context_cannot_leak_from_the_parent_environment() {
    let project = create_project();
    set_trainer(project.path(), "no_context_probe_child");
    let output = Command::new(binary())
        .arg("--project")
        .arg(project.path())
        .arg("--json")
        .arg("train")
        .arg("--no-capture")
        .env("GLR_RUN_CONTEXT", r#"{"forged":true}"#)
        .env("GLR_RUN_CONTEXT_SHA256", "forged")
        .output()
        .unwrap();
    success(&output);
}
