//! `entry-point-v1`: one pinned entry point per project, with launch
//! attestation, single-owner invariants, and one aggregate `doctor` verdict.
//!
//! These tests pin the acceptance criteria from GLR #159 / ADR-0042 at the CLI
//! boundary, because that is where a scheduler and an unattended agent meet the
//! capability: exit codes and JSON, not library calls.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use tempfile::TempDir;

const ENTRY_REFUSED: i32 = 79;
const DOCTOR_FAILED: i32 = 4;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_glr"))
}

/// One learner definition, declared as a single-owner invariant.
fn learner_invariant() -> Value {
    json!([{
        "id": "single-learner",
        "root": "src",
        "suffix": ".py",
        "marker": "class Learner"
    }])
}

fn entry_point(strict: bool, invariants: Value) -> Value {
    json!({
        "schema_version": "glr.entry-point.v1",
        "id": "campaign-driver",
        "command": "python -m campaign.driver",
        "version": "1.4.0",
        "strict": strict,
        "invariants": invariants
    })
}

fn project(entry_point: Option<Value>, tree: &[(&str, &str)]) -> TempDir {
    let executable = binary().to_string_lossy().into_owned();
    project_with_trainer(
        entry_point,
        tree,
        json!({"argv": [executable, "--version"]}),
    )
}

/// A project whose trainer exits non-zero, so `doctor` has a real past failure
/// to report instead of a run that was asserted to have succeeded.
fn project_that_fails(entry_point: Option<Value>, tree: &[(&str, &str)]) -> TempDir {
    let executable = binary().to_string_lossy().into_owned();
    project_with_trainer(
        entry_point,
        tree,
        json!({"argv": [executable, "no-such-subcommand"]}),
    )
}

fn project_with_trainer(
    entry_point: Option<Value>,
    tree: &[(&str, &str)],
    trainer: Value,
) -> TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("bridge")).unwrap();
    for (relative, body) in tree {
        let path = root.path().join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }
    let executable = binary().to_string_lossy().into_owned();
    let mut manifest = json!({
        "schema_version": "glr.project.v1",
        "environment_id": "example.entry-point-v1",
        "environment_family": "test",
        "protocol_version": "1.0",
        "data_dir": ".glr",
        "bridge_path": "bridge",
        "runtime": {"argv": [executable, "--version"]},
        "trainer": trainer,
        "player": {"argv": [executable, "--version"]},
        "researcher": null,
        "planner": null,
        "evaluator": null,
        "capture": null
    });
    if let Some(entry_point) = entry_point {
        manifest["entry_point"] = entry_point;
    }
    fs::write(
        root.path().join("glr-project.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    root
}

/// Runs `glr`, with the provenance environment set exactly as given.
fn glr(root: &Path, arguments: &[&str], entry: Option<(&str, &str)>) -> Output {
    let mut command = Command::new(binary());
    command
        .arg("--project")
        .arg(root)
        .arg("--json")
        .args(arguments)
        .env_remove("GLR_ENTRY_ID")
        .env_remove("GLR_ENTRY_VERSION");
    if let Some((id, version)) = entry {
        command
            .env("GLR_ENTRY_ID", id)
            .env("GLR_ENTRY_VERSION", version);
    }
    command.output().unwrap()
}

fn train(root: &Path, entry: Option<(&str, &str)>) -> Output {
    glr(
        root,
        &["train", "--no-capture", "--no-observe", "--no-recording"],
        entry,
    )
}

fn doctor(root: &Path) -> Value {
    let output = glr(root, &["doctor"], None);
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "doctor stdout is JSON: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

fn matches_containing(value: &Value, suffix: &str) -> bool {
    value
        .as_array()
        .is_some_and(|paths| paths.iter().any(|path| path.as_str() == Some(suffix)))
}

/// How many runs the store actually holds. The store file itself is created
/// when the CLI dispatches, so its existence says nothing; the row count does.
fn run_count(root: &Path) -> usize {
    let output = glr(root, &["runs", "list", "--limit", "100"], None);
    let value: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "runs list is JSON: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    value["data"].as_array().map(Vec::len).unwrap_or(0)
}

#[test]
fn two_learner_definitions_fail_doctor_and_name_both_paths() {
    let root = project(
        Some(entry_point(false, learner_invariant())),
        &[
            ("src/a/learner.py", "class Learner: pass\n"),
            ("src/b/legacy_learner.py", "class Learner: pass\n"),
            ("src/b/util.py", "unrelated\n"),
        ],
    );
    let output = glr(root.path(), &["doctor"], None);
    assert_eq!(
        output.status.code(),
        Some(DOCTOR_FAILED),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = doctor(root.path());
    let invariant = &report["data"]["entry_point"]["invariants"][0];
    assert_eq!(invariant["status"], "multiple");
    // Both paths are named, not just the first offender.
    assert!(
        matches_containing(&invariant["matches"], "src/a/learner.py"),
        "{invariant}"
    );
    assert!(
        matches_containing(&invariant["matches"], "src/b/legacy_learner.py"),
        "{invariant}"
    );
    assert!(
        !matches_containing(&invariant["matches"], "src/b/util.py"),
        "only marker-bearing files are named: {invariant}"
    );
}

#[test]
fn one_learner_definition_passes_doctor() {
    let root = project(
        Some(entry_point(true, learner_invariant())),
        &[("src/learner.py", "class Learner: pass\n")],
    );
    let output = glr(root.path(), &["doctor"], Some(("campaign-driver", "1.4.0")));
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = doctor_of(&output);
    assert_eq!(report["entry_point"]["invariants"][0]["status"], "ok");
    assert_eq!(report["entry_point"]["status"], "matched");
}

fn doctor_of(output: &Output) -> Value {
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    value["data"].clone()
}

#[test]
fn a_project_declaring_nothing_behaves_as_before() {
    let root = project(None, &[]);
    let output = glr(root.path(), &["doctor"], None);
    assert_eq!(
        output.status.code(),
        Some(0),
        "an absent capability must not gate: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = doctor(root.path());
    assert_eq!(report["data"]["entry_point"]["status"], "undeclared");
    assert!(
        report["data"]["entry_point"]["invariants"]
            .as_array()
            .is_some_and(Vec::is_empty)
    );
    assert_eq!(report["data"]["last_run"]["status"], "none");
}

#[test]
fn the_declared_entry_is_visible_in_cli_json_without_parsing_logs() {
    let root = project(Some(entry_point(false, json!([]))), &[]);
    let report = doctor(root.path());
    let declared = &report["data"]["entry_point"]["declared"];
    assert_eq!(declared["id"], "campaign-driver");
    assert_eq!(declared["command"], "python -m campaign.driver");
    assert_eq!(declared["version"], "1.4.0");
}

#[test]
fn strict_mode_refuses_a_drifting_run_before_any_run_exists() {
    let root = project(Some(entry_point(true, json!([]))), &[]);
    let output = train(root.path(), Some(("legacy-pixel-capture", "0.9.0")));
    assert_eq!(
        output.status.code(),
        Some(ENTRY_REFUSED),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let refusal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(refusal["command"], "entry.drift");
    assert_eq!(refusal["data"]["refused"], true);
    assert_eq!(refusal["data"]["status"], "entry_drift");

    // Refused before attach: no run row, so no budget was consumed.
    assert_eq!(run_count(root.path()), 0, "a refused run is not recorded");
    let report = doctor(root.path());
    assert_eq!(report["data"]["last_run"]["status"], "none");
}

#[test]
fn strict_mode_refuses_a_run_that_claims_no_entry_at_all() {
    let root = project(Some(entry_point(true, json!([]))), &[]);
    let output = train(root.path(), None);
    assert_eq!(output.status.code(), Some(ENTRY_REFUSED));
    assert_eq!(run_count(root.path()), 0);
}

#[test]
fn a_non_strict_drifting_run_runs_but_is_recorded_as_entry_drift() {
    let root = project(Some(entry_point(false, json!([]))), &[]);
    let output = train(root.path(), None);
    assert_eq!(
        output.status.code(),
        Some(0),
        "non-strict drift is recorded, not refused: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = doctor(root.path());
    assert_eq!(
        report["data"]["last_run"]["entry_point"]["status"],
        "entry_drift"
    );
    assert_eq!(
        report["data"]["last_run"]["entry_point"]["declared"]["id"],
        "campaign-driver"
    );
}

#[test]
fn a_matching_run_is_recorded_as_matched() {
    let root = project(Some(entry_point(true, json!([]))), &[]);
    let output = train(root.path(), Some(("campaign-driver", "1.4.0")));
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = doctor(root.path());
    assert_eq!(
        report["data"]["last_run"]["entry_point"]["status"],
        "matched"
    );
    assert_eq!(report["data"]["last_run"]["status"], "succeeded");
    assert_eq!(run_count(root.path()), 1);
}

#[test]
fn an_invariant_violation_fails_the_run_and_names_every_path() {
    let root = project(
        Some(entry_point(false, learner_invariant())),
        &[
            ("src/a/learner.py", "class Learner: pass\n"),
            ("src/b/copy_learner.py", "class Learner: pass\n"),
        ],
    );
    let output = train(root.path(), Some(("campaign-driver", "1.4.0")));
    assert_ne!(
        output.status.code(),
        Some(0),
        "a duplicated learner must fail the run"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("src/a/learner.py"), "{stderr}");
    assert!(stderr.contains("src/b/copy_learner.py"), "{stderr}");
}

#[test]
fn doctor_does_not_fail_a_round_that_only_forgot_to_declare_its_entry() {
    // `doctor` is not a run and never carries `GLR_ENTRY_ID`. Counting the
    // attestation in its verdict failed every round of every project that pins
    // an entry point, which is the opposite of what the capability is for: a
    // non-strict project records drift, it does not gate on it.
    let root = project(
        Some(entry_point(false, learner_invariant())),
        &[("src/learner.py", "class Learner: pass\n")],
    );
    let output = glr(root.path(), &["doctor"], None);
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = doctor(root.path());
    let entry = &report["data"]["entry_point"];
    assert_eq!(entry["status"], "entry_drift");
    assert_eq!(entry["strict"], false);
    assert_eq!(entry["ready"], true, "non-strict drift is not a gate");
    assert_eq!(entry["invariants"][0]["status"], "ok");
    assert_eq!(report["data"]["ready"], true);
}

#[test]
fn a_strict_project_fails_doctor_without_its_declared_entry() {
    // The run-start gate would refuse the run with 79, so the aggregate
    // verdict must say so rather than reporting a healthy project.
    let root = project(Some(entry_point(true, json!([]))), &[]);
    let drifted = glr(root.path(), &["doctor"], None);
    assert_eq!(drifted.status.code(), Some(DOCTOR_FAILED));
    let report = doctor(root.path());
    assert_eq!(report["data"]["entry_point"]["ready"], false);

    let matched = glr(root.path(), &["doctor"], Some(("campaign-driver", "1.4.0")));
    assert_eq!(
        matched.status.code(),
        Some(0),
        "the declared entry satisfies a strict project: {}",
        String::from_utf8_lossy(&matched.stderr)
    );
}

#[test]
fn the_drift_check_stays_cheap_at_project_scale() {
    // The guarantee is a bound, not a constant: the scan stops at the
    // documented file and byte budgets and reports `truncated` when it gets
    // there. This pins the cost at a realistic project size — a thousand
    // modules — which is what a round-top guard actually pays.
    let modules = (0..1_000)
        .map(|index| {
            (
                format!("src/module_{index}.py"),
                format!("# module {index}\n"),
            )
        })
        .collect::<Vec<_>>();
    let mut tree = modules
        .iter()
        .map(|(path, body)| (path.as_str(), body.as_str()))
        .collect::<Vec<_>>();
    tree.push(("src/learner.py", "class Learner: pass\n"));
    let root = project(Some(entry_point(false, learner_invariant())), &tree);

    let report = doctor(root.path());
    let entry = &report["data"]["entry_point"];
    assert_eq!(entry["invariants"][0]["status"], "ok");
    assert!(
        entry["scanned_files"].as_u64().unwrap() >= 1_000,
        "every module was scanned: {entry}"
    );
    let elapsed = entry["elapsed_ms"]
        .as_u64()
        .expect("elapsed_ms is reported");
    assert!(elapsed < 1_000, "1,001 files took {elapsed} ms");
}

#[test]
fn the_last_run_verdict_is_reported_but_never_gates() {
    // A past failure is information, not current unreadiness: `doctor` reports
    // it without changing its own verdict for a healthy project.
    let root = project_that_fails(None, &[]);
    let failed = train(root.path(), None);
    assert_ne!(
        failed.status.code(),
        Some(0),
        "the trainer must actually fail for this case to mean anything"
    );
    let report = doctor(root.path());
    assert_eq!(
        report["data"]["last_run"]["status"], "failed",
        "the last run really did fail: {}",
        report["data"]["last_run"]
    );
    assert!(report["data"]["last_run"]["run_id"].is_string());
    assert_eq!(report["data"]["ready"], true);
}
