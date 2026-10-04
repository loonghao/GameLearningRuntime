//! Legacy goal-run stages independently evaluated candidates for trusted-host review.
//! The fixture roles operate on temporary JSON/SQLite files, without a game.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rusqlite::{Connection, params};
use serde_json::{Value, json};
use tempfile::TempDir;

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn create_project(case: &str, target: f64) -> TempDir {
    let temporary = tempfile::tempdir().unwrap();
    fs::create_dir(temporary.path().join("bridge")).unwrap();
    let executable = env!("CARGO_BIN_EXE_glr");
    let probe = std::env::current_exe().unwrap();
    let role = |name: &str| {
        json!({
            "argv": [probe, "--ignored", "--exact", format!("promotion_{name}_child")]
        })
    };
    fs::write(temporary.path().join("fixture.case"), case).unwrap();
    write_json(
        &temporary.path().join("glr-project.json"),
        &json!({
            "schema_version": "glr.project.v1",
            "environment_id": "example.promotion-v1", "environment_family": "test",
            "protocol_version": "1.0", "data_dir": ".glr", "bridge_path": "bridge",
            "runtime": {"argv": [executable, "--version"]},
            "player": {"argv": [executable, "--version"]},
            "researcher": role("researcher"), "planner": role("planner"),
            "trainer": role("trainer"), "evaluator": role("evaluator"), "capture": null
        }),
    );
    write_json(
        &temporary.path().join("goal.json"),
        &json!({
            "schema_version": "glr.agent-goal.v1", "goal_id": "goal.promotion",
            "objective": "increase verified victories", "environment_family": "test",
            "success_criteria": [{"metric": "victories", "operator": "gte",
                "target": target, "source": "referee"}],
            "budget": {"max_trials": 1, "max_training_steps": 1,
                "max_wall_seconds": 30, "max_research_sources": 1},
            "allowed_research_media": ["runtime-trace"],
            "promotion": {"metric": "victories", "mode": "max"}
        }),
    );
    let live = live_checkpoint(temporary.path());
    fs::create_dir_all(live.parent().unwrap()).unwrap();
    fs::write(live, b"incumbent").unwrap();
    temporary
}

fn live_checkpoint(root: &Path) -> PathBuf {
    root.join(".glr/checkpoints/example.promotion-v1/goal.promotion/best.checkpoint")
}

fn run(root: &Path) -> Output {
    run_with_capture(root, false)
}

fn run_with_capture(root: &Path, capture_enabled: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_glr"));
    let goal_path = root.join("goal.json");
    command.args([
        "--project",
        root.to_str().unwrap(),
        "--json",
        "goal",
        "run",
        "--goal",
        goal_path.to_str().unwrap(),
    ]);
    if !capture_enabled {
        command.arg("--no-capture");
    }
    command.env("CI", "1").output().unwrap()
}

fn evidence(role_dir: &Path, source: &str, authority: &str, value: f64, run_id: &str) {
    write_json(
        &role_dir.join("evaluation.json"),
        &json!({
            "schema_version": "glr.goal-evidence.v1", "goal_id": "goal.promotion",
            "trial_id": std::env::var("GLR_TRIAL_ID").unwrap(),
            "evidence": [{"metric": "victories", "value": value,
                "source": source, "authority": authority, "run_id": run_id}]
        }),
    );
}

fn append_metric(run_id: &str, source: &str, authority: &str, value: f64) {
    Connection::open(std::env::var_os("GLR_STORE_PATH").unwrap()).unwrap().execute(
        "INSERT INTO metrics(run_id, timestamp_ns, name, value, step_id, metadata_json) VALUES (?, 1, 'victories', ?, NULL, ?)",
        params![run_id, value, json!({"source": source, "authority": authority}).to_string()],
    ).unwrap();
}

#[test]
#[ignore]
fn promotion_researcher_child() {
    promotion_role_child("researcher");
}
#[test]
#[ignore]
fn promotion_planner_child() {
    promotion_role_child("planner");
}
#[test]
#[ignore]
fn promotion_trainer_child() {
    promotion_role_child("trainer");
}
#[test]
#[ignore]
fn promotion_evaluator_child() {
    promotion_role_child("evaluator");
}
fn promotion_role_child(role: &str) {
    let role_dir = PathBuf::from(std::env::var_os("GLR_RUN_DIR").unwrap());
    let root = PathBuf::from(std::env::var_os("GLR_PROJECT_ROOT").unwrap());
    let run_id = std::env::var("GLR_RUN_ID").unwrap();
    match role {
        "researcher" => write_json(
            Path::new(&std::env::var_os("GLR_RESEARCH_PATH").unwrap()),
            &json!({"schema_version": "glr.research-bundle.v1", "sources": [], "findings": []}),
        ),
        "planner" => write_json(
            Path::new(&std::env::var_os("GLR_TRIAL_PATH").unwrap()),
            &json!({"schema_version": "glr.trial-plan.v1", "trial_id": "trial-1",
                "goal_id": "goal.promotion", "seed": 1, "max_steps": 1,
                "reward_terms": [], "notes": "offline promotion regression"}),
        ),
        "trainer" => {
            assert!(std::env::var_os("GLR_PROMOTION_PATH").is_none());
            let baseline = PathBuf::from(std::env::var_os("GLR_CHECKPOINT_PATH").unwrap());
            assert_ne!(baseline, live_checkpoint(&root));
            assert_eq!(fs::read(&baseline).unwrap(), b"incumbent");
            fs::write(&baseline, b"worker-local-baseline-change").unwrap();
            assert_eq!(fs::read(live_checkpoint(&root)).unwrap(), b"incumbent");
            fs::write(role_dir.join("checkpoint.candidate"), b"candidate").unwrap();
            write_json(
                &role_dir.join("trainer.result.json"),
                &json!({
                    "schema_version": "glr.trainer-result.v1", "status": "completed",
                    "metrics": {"victories": 99.0}
                }),
            );
        }
        "evaluator" => {
            assert_eq!(
                fs::read(live_checkpoint(&root)).unwrap(),
                b"incumbent",
                "the fixed evaluator must see the incumbent before promotion"
            );
            fs::write(root.join("evaluator-saw-incumbent"), b"verified").unwrap();
            let case = fs::read_to_string(root.join("fixture.case")).unwrap();
            if case == "evaluator-fails" {
                panic!("fixed evaluator deliberately refused the trial");
            }
            if case == "budget-expired" {
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
            append_metric(&run_id, "referee", "authoritative", 4.0);
            if case == "same-source-conflict" {
                append_metric(&run_id, "referee", "authoritative", 0.0);
            }
            // Advisory metrics cannot define the selected authoritative value.
            append_metric(&run_id, "strategy", "advisory", 999.0);
            match case.as_str() {
                "unbacked" => evidence(&role_dir, "referee", "authoritative", 8.0, &run_id),
                "wrong-source" => evidence(&role_dir, "trainer", "authoritative", 99.0, &run_id),
                "wrong-run" => evidence(&role_dir, "referee", "authoritative", 4.0, "other-run"),
                "advisory" => {
                    append_metric(&run_id, "referee", "advisory", 5.0);
                    evidence(&role_dir, "referee", "advisory", 5.0, &run_id);
                }
                _ => evidence(&role_dir, "referee", "authoritative", 4.0, &run_id),
            }
        }
        other => panic!("unknown test role: {other}"),
    }
}

#[test]
fn evaluator_and_persisted_authoritative_bundle_only_stage_a_proposal() {
    let project = create_project("success", 4.0);
    let output = run(project.path());
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(project.path().join("evaluator-saw-incumbent").is_file());
    assert_eq!(
        receipt["data"]["promotion"]["status"],
        "staged-awaiting-review"
    );
    assert_eq!(receipt["data"]["promotion"]["promoted"], false);
    let connection = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
    let count: i64 = connection
        .query_row("SELECT COUNT(*) FROM checkpoint_promotions", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(count, 0);
    assert_eq!(
        receipt["data"]["promotion"]["final_measurement"]["value"],
        4.0
    );
    assert_eq!(
        receipt["data"]["promotion"]["final_measurement"]["source"],
        "referee"
    );
    assert_eq!(
        fs::read(live_checkpoint(project.path())).unwrap(),
        b"incumbent"
    );
}

#[test]
fn evaluated_progress_can_stage_before_the_final_goal_is_satisfied() {
    let project = create_project("success", 10.0);
    let output = run(project.path());
    assert_eq!(
        output.status.code(),
        Some(3),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(receipt["data"]["satisfied"], false);
    assert_eq!(receipt["data"]["promotion"]["promoted"], false);
    assert_eq!(
        fs::read(live_checkpoint(project.path())).unwrap(),
        b"incumbent"
    );
}

#[test]
fn failed_evaluation_or_invalid_evidence_never_replaces_incumbent() {
    for case in [
        "evaluator-fails",
        "unbacked",
        "wrong-source",
        "wrong-run",
        "advisory",
        "same-source-conflict",
    ] {
        let project = create_project(case, 4.0);
        let output = run(project.path());
        assert_eq!(
            output.status.code(),
            Some(2),
            "case: {case}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            project.path().join("evaluator-saw-incumbent").is_file(),
            "case: {case}"
        );
        assert_eq!(
            fs::read(live_checkpoint(project.path())).unwrap(),
            b"incumbent",
            "case: {case}"
        );
        let connection = Connection::open(project.path().join(".glr/runs.sqlite3")).unwrap();
        let promotions: i64 = connection
            .query_row("SELECT COUNT(*) FROM checkpoint_promotions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(promotions, 0, "case: {case}");
    }
}

#[test]
fn required_capture_failure_prevents_promotion_and_evaluation() {
    let project = create_project("success", 4.0);
    let manifest = project.path().join("glr-project.json");
    let mut value: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    value["capture"] = json!({
        "argv": [env!("CARGO_BIN_EXE_glr"), "--version"],
        "required": true, "stop": "terminate", "video_file": "capture.mp4",
        "index_file": "capture.jsonl", "codec": "h264", "frame_rate": 1.0,
        "width": 16, "height": 16
    });
    write_json(&manifest, &value);
    let output = run_with_capture(project.path(), true);
    assert_eq!(
        output.status.code(),
        Some(2),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("required capture failed"));
    assert!(!project.path().join("evaluator-saw-incumbent").exists());
    assert_eq!(
        fs::read(live_checkpoint(project.path())).unwrap(),
        b"incumbent"
    );
}

#[test]
fn wall_budget_expiration_preserves_incumbent() {
    let project = create_project("budget-expired", 4.0);
    let goal_path = project.path().join("goal.json");
    let mut goal: Value = serde_json::from_slice(&fs::read(&goal_path).unwrap()).unwrap();
    goal["budget"]["max_wall_seconds"] = json!(1);
    write_json(&goal_path, &goal);
    let output = run(project.path());
    assert_eq!(
        output.status.code(),
        Some(2),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("project command exceeded") || error.contains("wall-clock budget"),
        "stderr: {error}"
    );
    assert_eq!(
        fs::read(live_checkpoint(project.path())).unwrap(),
        b"incumbent"
    );
}
