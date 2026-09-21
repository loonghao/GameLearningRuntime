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
    fs::create_dir_all(temporary.path().join("goals")).unwrap();
    fs::write(
        temporary.path().join("config/training.json"),
        br#"{"schema_version":"glr.training.v1","algorithm":"ppo"}"#,
    )
    .unwrap();
    for (name, context_id) in [
        ("native", "league-native-100024"),
        ("ranked", "ranked-2026"),
    ] {
        fs::write(
            temporary
                .path()
                .join(format!("config/contexts/{name}.toml")),
            format!(
                r#"schema_version = "glr.run-context.v1"
context_id = "{context_id}"
environment_id = "example.context-v1"
protocol_version = "1.0"

[labels]
season = "{name}"

[[inputs]]
owner = "training"
path = "config/training.json"
schema_version = "glr.training.v1"
"#
            ),
        )
        .unwrap();
    }
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
        "trainer": {"argv": [probe, "--ignored", "--exact", "goal_context_probe_child"]},
        "player": {"argv": [executable, "--version"]},
        "researcher": {"argv": [executable, "--version"]},
        "planner": {"argv": [executable, "--version"]},
        "evaluator": {"argv": [executable, "--version"]},
        "capture": null
    });
    fs::write(
        temporary.path().join("glr-project.json"),
        serde_json::to_vec_pretty(&project).unwrap(),
    )
    .unwrap();
    temporary
}

fn write_goal(root: &Path, name: &str, goal_id: &str) -> PathBuf {
    let path = root.join("goals").join(format!("{name}.json"));
    fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "glr.agent-goal.v1",
            "goal_id": goal_id,
            "objective": "reach the destination",
            "environment_family": "test",
            "success_criteria": [{
                "metric": "progress",
                "operator": "gte",
                "target": 1.0,
                "source": "evaluator"
            }],
            "budget": {
                "max_trials": 2,
                "max_training_steps": 100,
                "max_wall_seconds": 60,
                "max_research_sources": 4
            },
            "allowed_research_media": ["runtime-trace"]
        }))
        .unwrap(),
    )
    .unwrap();
    path
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

/// Rewrite the objective of a bound goal, keeping the goal id and budgets: the
/// stored SHA-256 then describes an older file.
fn rewrite_goal_objective(root: &Path, name: &str, objective: &str) {
    let path = root.join("goals").join(format!("{name}.json"));
    let mut goal: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    goal["objective"] = Value::String(objective.to_owned());
    fs::write(&path, serde_json::to_vec_pretty(&goal).unwrap()).unwrap();
}

/// The `goal_binding` receipt of the newest run.
///
/// The project researcher is `glr --version`, so a goal run fails after the run
/// row and its receipt already exist; the receipt is what these tests read.
fn goal_binding_receipt(project: &Path) -> Value {
    let listed = success(&run(project, &["runs", "list", "--limit", "1"]));
    let runs = listed["data"].as_array().unwrap();
    assert!(!runs.is_empty(), "goal run must record a run");
    runs[0]["metadata"]["goal_binding"].clone()
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

fn failure(output: &Output) -> String {
    assert!(
        !output.status.success(),
        "expected failure, stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    // The observation server writes a notice to stderr ahead of the error
    // envelope, so the JSON payload is found by line.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let line = stderr
        .lines()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or_else(|| panic!("no JSON error envelope in stderr: {stderr}"));
    let error: Value = serde_json::from_str(line).unwrap();
    error["error"]["message"].as_str().unwrap().to_owned()
}

#[test]
#[ignore]
fn goal_context_probe_child() {
    let context = std::env::var("GLR_RUN_CONTEXT").expect("GLR_RUN_CONTEXT");
    let run_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").expect("run dir"));
    fs::write(run_dir.join("goal-context-observed.json"), context).unwrap();
}

fn train_and_read_context(project: &Path, arguments: &[&str]) -> Value {
    let output = Command::new(binary())
        .arg("--project")
        .arg(project)
        .arg("--json")
        .args(arguments)
        .output()
        .unwrap();
    let trained = success(&output);
    let run_id = trained["data"]["run_id"].as_str().unwrap();
    let run_dir = project.join(".glr/runs").join(run_id);
    serde_json::from_slice(&fs::read(run_dir.join("goal-context-observed.json")).unwrap()).unwrap()
}

#[test]
fn goal_run_without_a_saved_default_goal_reports_the_gap() {
    let project = create_project();
    let error = failure(&run(project.path(), &["goal", "run"]));
    assert!(error.contains("saved default goal"), "{error}");
}

#[test]
fn a_saved_default_goal_is_read_when_the_flag_is_omitted() {
    let project = create_project();
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    success(&run(
        project.path(),
        &["goal", "set", "--goal", "goals/reach-destination.json"],
    ));
    // Overwrite the bound file with a goal whose budget is empty: reaching this
    // error proves `goal run` read the saved default instead of asking for
    // --goal again.
    fs::write(
        project.path().join("goals/reach-destination.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version": "glr.agent-goal.v1",
            "goal_id": "goal.reach-destination",
            "objective": "reach the destination",
            "environment_family": "test",
            "success_criteria": [{
                "metric": "progress",
                "operator": "gte",
                "target": 1.0,
                "source": "evaluator"
            }],
            "budget": {
                "max_trials": 0,
                "max_training_steps": 100,
                "max_wall_seconds": 60,
                "max_research_sources": 4
            },
            "allowed_research_media": ["runtime-trace"]
        }))
        .unwrap(),
    )
    .unwrap();
    let error = failure(&run(project.path(), &["goal", "run"]));
    assert!(error.contains("budgets must be positive"), "{error}");
}

#[test]
fn a_deleted_default_goal_asks_for_a_rebind() {
    let project = create_project();
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    success(&run(
        project.path(),
        &["goal", "set", "--goal", "goals/reach-destination.json"],
    ));
    fs::remove_file(project.path().join("goals/reach-destination.json")).unwrap();
    let error = failure(&run(project.path(), &["goal", "run"]));
    assert!(error.contains("no longer exists"), "{error}");
}

#[test]
fn an_explicit_goal_still_wins_over_the_saved_default() {
    let project = create_project();
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    write_goal(project.path(), "explicit", "goal.explicit");
    success(&run(
        project.path(),
        &["goal", "set", "--goal", "goals/reach-destination.json"],
    ));
    let shown = success(&run(project.path(), &["goal", "show"]));
    assert_eq!(shown["command"], "goal.show");
    assert_eq!(shown["data"]["active_goal_id"], "goal.reach-destination");
    // The explicit file is the one with the empty media list, so reaching that
    // error proves the default goal was not used.
    fs::write(
        project.path().join("goals/explicit.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version": "glr.agent-goal.v1",
            "goal_id": "goal.explicit",
            "objective": "reach the destination",
            "environment_family": "test",
            "success_criteria": [{
                "metric": "progress",
                "operator": "gte",
                "target": 1.0,
                "source": "evaluator"
            }],
            "budget": {
                "max_trials": 2,
                "max_training_steps": 100,
                "max_wall_seconds": 60,
                "max_research_sources": 4
            },
            "allowed_research_media": []
        }))
        .unwrap(),
    )
    .unwrap();
    let explicit = project.path().join("goals/explicit.json");
    let error = failure(&run(
        project.path(),
        &["goal", "run", "--goal", explicit.to_string_lossy().as_ref()],
    ));
    assert!(error.contains("allowed_research_media"), "{error}");
}

#[test]
fn the_bound_context_reaches_training_without_the_context_flag() {
    let project = create_project();
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    success(&run(
        project.path(),
        &[
            "--context",
            "config/contexts/native.toml",
            "goal",
            "set",
            "--goal",
            "goals/reach-destination.json",
        ],
    ));
    let observed = train_and_read_context(project.path(), &["train", "--no-capture"]);
    assert_eq!(observed["context_id"], "league-native-100024");
    assert_eq!(observed["schema_version"], "glr.run-context.v1");
}

#[test]
fn an_explicit_context_still_overrides_the_bound_default() {
    let project = create_project();
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    success(&run(
        project.path(),
        &[
            "--context",
            "config/contexts/native.toml",
            "goal",
            "set",
            "--goal",
            "goals/reach-destination.json",
        ],
    ));
    let observed = train_and_read_context(
        project.path(),
        &[
            "--context",
            "config/contexts/ranked.toml",
            "train",
            "--no-capture",
        ],
    );
    assert_eq!(observed["context_id"], "ranked-2026");
}

#[test]
fn a_saved_goal_can_be_switched_and_listed() {
    let project = create_project();
    write_goal(project.path(), "first", "goal.first");
    write_goal(project.path(), "second", "goal.second");
    success(&run(
        project.path(),
        &["goal", "set", "--goal", "goals/first.json"],
    ));
    success(&run(
        project.path(),
        &["goal", "set", "--goal", "goals/second.json"],
    ));
    let listed = success(&run(project.path(), &["goal", "list"]));
    assert_eq!(listed["data"]["goals"].as_array().unwrap().len(), 2);
    assert_eq!(listed["data"]["active_goal_id"], "goal.second");
    let switched = success(&run(project.path(), &["goal", "use", "goal.first"]));
    assert_eq!(switched["data"]["active_goal_id"], "goal.first");
    let shown = success(&run(project.path(), &["goal", "show", "goal.second"]));
    assert_eq!(shown["data"]["goal"]["goal_id"], "goal.second");
    assert_eq!(shown["data"]["goal"]["source_status"], "unchanged");
    assert_eq!(shown["data"]["goal"]["context_status"], "unbound");
}

#[test]
fn an_explicit_goal_receipt_records_the_inherited_context() {
    let project = create_project();
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    success(&run(
        project.path(),
        &[
            "--context",
            "config/contexts/native.toml",
            "goal",
            "set",
            "--goal",
            "goals/reach-destination.json",
        ],
    ));
    let explicit = project.path().join("goals/reach-destination.json");
    let _ = run(
        project.path(),
        &["goal", "run", "--goal", explicit.to_string_lossy().as_ref()],
    );
    // The run named its own goal but no `--context`, so the context came from
    // the binding of the active goal.
    let receipt = goal_binding_receipt(project.path());
    assert_eq!(receipt["source"], "explicit");
    assert_eq!(receipt["context_source"], "default");
    assert_eq!(receipt["context_path"], "config/contexts/native.toml");
}

#[test]
fn a_default_goal_receipt_records_the_goal_drift_status() {
    let project = create_project();
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    success(&run(
        project.path(),
        &["goal", "set", "--goal", "goals/reach-destination.json"],
    ));
    let _ = run(project.path(), &["goal", "run"]);
    let receipt = goal_binding_receipt(project.path());
    assert_eq!(receipt["source"], "default");
    assert_eq!(receipt["source_status"], "unchanged");
    assert_eq!(receipt["context_source"], "none");

    rewrite_goal_objective(
        project.path(),
        "reach-destination",
        "reach the other destination",
    );
    let _ = run(project.path(), &["goal", "run"]);
    assert_eq!(
        goal_binding_receipt(project.path())["source_status"],
        "changed"
    );
}

#[test]
fn a_hand_edited_store_cannot_point_goal_run_outside_the_project() {
    let project = create_project();
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    success(&run(
        project.path(),
        &["goal", "set", "--goal", "goals/reach-destination.json"],
    ));
    let store = project.path().join(".glr/goal-binding.json");
    let mut file: Value = serde_json::from_slice(&fs::read(&store).unwrap()).unwrap();
    file["goals"]["goal.reach-destination"]["goal_path"] = json!("../outside.json");
    fs::write(&store, serde_json::to_vec_pretty(&file).unwrap()).unwrap();

    let error = failure(&run(project.path(), &["goal", "run"]));
    assert!(error.contains("project-relative"), "{error}");
    let error = failure(&run(project.path(), &["goal", "show"]));
    assert!(error.contains("project-relative"), "{error}");
    // `doctor` stays usable and reports the gap instead of a bound goal.
    let report = success(&run(project.path(), &["doctor"]));
    assert!(
        report["data"]["goal_binding"]["error"]
            .as_str()
            .unwrap()
            .contains("project-relative"),
        "{}",
        report["data"]["goal_binding"]
    );
}

#[test]
fn doctor_reports_the_active_goal() {
    let project = create_project();
    let report = success(&run(project.path(), &["doctor"]));
    assert_eq!(report["data"]["goal_binding"]["goal_count"], 0);
    write_goal(
        project.path(),
        "reach-destination",
        "goal.reach-destination",
    );
    success(&run(
        project.path(),
        &["goal", "set", "--goal", "goals/reach-destination.json"],
    ));
    let report = success(&run(project.path(), &["doctor"]));
    assert_eq!(
        report["data"]["goal_binding"]["active_goal_id"],
        "goal.reach-destination"
    );
    assert_eq!(report["data"]["goal_binding"]["goal_count"], 1);
}
