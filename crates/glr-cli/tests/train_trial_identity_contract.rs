//! `glr train` must bind its trainer role to the implicit trial it drives.
//!
//! A standalone training run is exactly one trial, and `glr-project.json` may
//! declare the `{trial_id}` and `{trial_path}` placeholders for the trainer
//! command. The goal loop has always issued `GLR_TRIAL_ID` / `GLR_TRIAL_PATH` to
//! its roles, so a trainer written against the trial contract could not run under
//! `glr train`. These tests pin both the published identity and the fact that it
//! is absent unless a trial actually exists.

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
    let executable = binary().to_string_lossy().into_owned();
    let probe = std::env::current_exe()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let project = json!({
        "schema_version": "glr.project.v1",
        "environment_id": "example.trial-v1",
        "environment_family": "test",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": [executable, "--version"]},
        "trainer": {"argv": [probe, "--ignored", "--exact", "trial_probe_child"]},
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

fn set_runtime(project: &Path, test_name: &str) {
    let manifest = project.join("glr-project.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    value["runtime"]["argv"] = json!([
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

fn observable(project: &Path, run_id: &str) -> Value {
    let path = project
        .join(".glr/runs")
        .join(run_id)
        .join("trial-observed.json");
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

/// Records the trial identity the trainer actually received.
#[test]
#[ignore]
fn trial_probe_child() {
    let trial_id = std::env::var("GLR_TRIAL_ID").expect("GLR_TRIAL_ID");
    let trial_path = std::env::var("GLR_TRIAL_PATH").expect("GLR_TRIAL_PATH");
    let run_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").expect("run dir"));
    fs::write(
        run_dir.join("trial-observed.json"),
        serde_json::to_vec_pretty(&json!({
            "trial_id": trial_id,
            "trial_path": trial_path,
            "run_dir": run_dir.to_string_lossy(),
        }))
        .unwrap(),
    )
    .unwrap();
}

/// Fails the run if a trial identity leaks into a role that has no trial.
#[test]
#[ignore]
fn no_trial_probe_child() {
    assert!(std::env::var_os("GLR_TRIAL_ID").is_none());
    assert!(std::env::var_os("GLR_TRIAL_PATH").is_none());
}

#[test]
fn a_standalone_training_run_binds_its_trainer_to_one_implicit_trial() {
    let project = create_project();
    let trained = success(&run(project.path(), &["train", "--no-capture"]));
    let run_id = trained["data"]["run_id"].as_str().unwrap();
    let observed = observable(project.path(), run_id);

    assert_eq!(observed["trial_id"], "trial-1");
    let trial_path = observed["trial_path"].as_str().unwrap();
    assert!(
        trial_path.ends_with("trials/trial-1/plan.json")
            || trial_path.ends_with(r"trials\trial-1\plan.json"),
        "unexpected trial path: {trial_path}"
    );
    assert!(
        Path::new(trial_path).is_file() || Path::new(trial_path).parent().unwrap().is_dir(),
        "the trial directory must exist before the trainer runs"
    );
}

#[test]
fn the_run_directory_keeps_its_documented_run_scoped_meaning() {
    let project = create_project();
    let trained = success(&run(project.path(), &["train", "--no-capture"]));
    let run_id = trained["data"]["run_id"].as_str().unwrap();
    let observed = observable(project.path(), run_id);

    // `GLR_RUN_DIR` is the run-scoped output root, not the trial directory.
    let run_dir = observed["run_dir"].as_str().unwrap();
    assert!(
        run_dir
            .replace('\\', "/")
            .ends_with(&format!("/runs/{run_id}")),
        "GLR_RUN_DIR must remain the run root; got {run_dir}"
    );
}

#[test]
fn the_trial_identity_matches_between_a_training_run_and_the_goal_loop() {
    let project = create_project();
    let trained = success(&run(project.path(), &["train", "--no-capture"]));
    let run_id = trained["data"]["run_id"].as_str().unwrap();
    let observed = observable(project.path(), run_id);
    let trial_id = observed["trial_id"].as_str().unwrap();

    // The goal loop names its first trial `trial-1`; a standalone run must agree,
    // so a trainer needs one convention rather than one per entrypoint.
    assert_eq!(trial_id, "trial-1");
    assert_eq!(
        Path::new(observed["trial_path"].as_str().unwrap())
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy(),
        trial_id
    );
}

#[test]
fn a_role_without_a_trial_never_receives_a_trial_identity() {
    let project = create_project();
    set_runtime(project.path(), "no_trial_probe_child");
    // A forged parent-environment value must not survive into a role that owns no
    // trial: `runtime start` drives no trial, so it must not forward one.
    let output = Command::new(binary())
        .arg("--project")
        .arg(project.path())
        .arg("--json")
        .arg("runtime")
        .arg("start")
        .env("GLR_TRIAL_ID", "forged")
        .env("GLR_TRIAL_PATH", "forged")
        .output()
        .unwrap();
    success(&output);
}
