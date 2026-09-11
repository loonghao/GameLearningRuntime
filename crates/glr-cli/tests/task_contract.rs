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
    let project = json!({
        "schema_version": "glr.project.v1",
        "environment_id": "example.tasks-v1",
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

fn success(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn task_graph_lists_inspects_and_runs_fixed_argv() {
    let project = create_project();
    let executable = binary().to_string_lossy().replace('\\', "\\\\");
    fs::write(
        project.path().join("glr.toml"),
        format!(
            r#"schema_version = "glr.tasks.v1"

[tasks.prepare]
description = "Prepare the bounded task"
argv = ["{executable}", "--version"]

[tasks.season]
description = "Run a synthetic season"
argv = ["{executable}", "--version"]
depends = ["prepare"]
timeout_seconds = 30

[tasks.season.parameters.profile]
type = "string"
required = true
"#
        ),
    )
    .unwrap();

    let listed = success(&run(project.path(), &["task", "list"]));
    assert_eq!(listed["command"], "task.list");
    assert_eq!(listed["data"].as_array().unwrap().len(), 2);

    let shown = success(&run(project.path(), &["task", "show", "season"]));
    assert_eq!(shown["data"]["task"]["depends"][0], "prepare");

    let executed = success(&run(
        project.path(),
        &[
            "task",
            "run",
            "season",
            "--set",
            "profile=league-legends/native-100024",
        ],
    ));
    assert_eq!(executed["command"], "task.run");
    assert_eq!(executed["data"]["status"], "succeeded");
    assert_eq!(executed["data"]["steps"].as_array().unwrap().len(), 2);
    let execution_id = executed["data"]["execution_id"].as_str().unwrap();
    assert!(
        project
            .path()
            .join(".glr/tasks")
            .join(execution_id)
            .join("result.json")
            .is_file()
    );

    let doctor = success(&run(project.path(), &["doctor"]));
    assert_eq!(doctor["data"]["tasks"]["schema_version"], "glr.tasks.v1");
    assert_eq!(doctor["data"]["tasks"]["task_count"], 2);
}

#[test]
fn task_contract_rejects_cycles_unknown_parameters_and_partial_placeholders() {
    let project = create_project();
    fs::write(
        project.path().join("glr.toml"),
        r#"schema_version = "glr.tasks.v1"

[tasks.first]
description = "First"
argv = ["echo", "prefix-{value}"]
depends = ["second"]

[tasks.first.parameters.value]
type = "string"
required = true

[tasks.second]
description = "Second"
argv = ["echo", "ok"]
depends = ["first"]
"#,
    )
    .unwrap();
    let rejected = run(project.path(), &["task", "list"]);
    assert_eq!(rejected.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        stderr.contains("placeholders must occupy a complete argv entry")
            || stderr.contains("dependency cycle")
    );

    let executable = binary().to_string_lossy().replace('\\', "\\\\");
    fs::write(
        project.path().join("glr.toml"),
        format!(
            r#"schema_version = "glr.tasks.v1"

[tasks.valid]
description = "Valid"
argv = ["{executable}", "--version"]
"#
        ),
    )
    .unwrap();
    let rejected = run(
        project.path(),
        &["task", "run", "valid", "--set", "unknown=value"],
    );
    assert_eq!(rejected.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("is not declared"));

    fs::write(
        project.path().join("glr.toml"),
        r#"schema_version = "glr.tasks.v1"

[tasks.untrusted-program]
description = "Reject a caller-selected executable"
argv = ["{program}", "--version"]

[tasks.untrusted-program.parameters.program]
type = "string"
required = true
"#,
    )
    .unwrap();
    let rejected = run(project.path(), &["task", "list"]);
    assert_eq!(rejected.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&rejected.stderr)
            .contains("argv[0] must be selected by trusted configuration")
    );
}

#[test]
fn task_contract_exposes_vx_as_an_explicit_runner() {
    let project = create_project();
    fs::write(
        project.path().join("glr.toml"),
        r#"schema_version = "glr.tasks.v1"

[tasks.python-train]
description = "Run Python training in the vx-managed environment"
runner = "vx"
argv = ["uv", "run", "--no-sync", "python", "tools/train.py"]
"#,
    )
    .unwrap();
    let shown = success(&run(project.path(), &["task", "show", "python-train"]));
    assert_eq!(shown["data"]["task"]["runner"], "vx");
    assert_eq!(shown["data"]["task"]["argv"][0], "uv");
}

#[test]
fn vx_backed_task_writes_a_schema_checked_result() {
    let project = create_project();
    fs::write(
        project.path().join("glr.toml"),
        r#"schema_version = "glr.tasks.v1"

[tasks.python-result]
description = "Write a result through VX-managed Python"
runner = "vx"
argv = ["python", "-c", "import json,os; open(os.environ['GLR_TASK_RESULT'],'w').write(json.dumps(dict(schema_version='glr.test-result.v1')))" ]
timeout_seconds = 60

[tasks.python-result.result]
schema = "glr.test-result.v1"
required = true
"#,
    )
    .unwrap();

    let executed = success(&run(project.path(), &["task", "run", "python-result"]));
    assert_eq!(executed["data"]["status"], "succeeded");
    assert_eq!(executed["data"]["steps"][0]["exit_code"], 0);
    assert!(executed["data"]["steps"][0]["result_path"].is_string());
}
