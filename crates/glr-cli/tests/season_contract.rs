use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/season_project")
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_glr"))
        .args(["--project", root.to_str().unwrap(), "--json"])
        .args(args)
        .output()
        .unwrap()
}

fn selected(root: &Path, args: &[&str]) -> Output {
    let mut arguments = vec!["--season", "example-season", "--ruleset", "standard"];
    arguments.extend(args);
    run(root, &arguments)
}

fn data(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice::<Value>(&output.stdout).unwrap()["data"].clone()
}

fn copy_fixture() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    for path in [
        "glr-project.toml",
        "config/seasons.toml",
        "config/training.toml",
        "config/preset.toml",
        "config/seasons/example-season/standard.toml",
    ] {
        let target = directory.path().join(path);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(fixture().join(path), target).unwrap();
    }
    let manifest = directory.path().join("glr-project.toml");
    let argv = serde_json::to_string(&[env!("CARGO_BIN_EXE_glr"), "--version"]).unwrap();
    let contents = fs::read_to_string(&manifest)
        .unwrap()
        .replace("[\"python\", \"-c\", \"pass\"]", &argv);
    fs::write(manifest, contents).unwrap();
    directory
}

#[test]
fn shared_wire_context_matches_golden_and_nested_discovery() {
    let expected: Value =
        serde_json::from_slice(&fs::read(fixture().join("context.json")).unwrap()).unwrap();
    assert_eq!(
        data(selected(
            &fixture().join("config/seasons"),
            &["season", "show"]
        )),
        expected
    );
    assert_eq!(
        data(run(&fixture(), &["season", "list"])),
        serde_json::json!([expected])
    );
    assert!(!run(&fixture(), &["season", "show"]).status.success());
    assert!(
        !run(
            &fixture(),
            &["--season", "example-season", "season", "show"]
        )
        .status
        .success()
    );
    assert!(
        !run(
            &fixture(),
            &[
                "--season",
                "unknown",
                "--ruleset",
                "standard",
                "season",
                "show"
            ]
        )
        .status
        .success()
    );
    assert!(!fixture().join(".glr").exists());
}

#[test]
fn pending_runtime_is_allowed_but_training_is_fenced_and_recorded() {
    let temporary = copy_fixture();
    let root = temporary.path();
    for args in [
        vec!["train"],
        vec!["goal", "run", "--goal", "missing.json"],
        vec!["play", "--bundle", "missing"],
    ] {
        assert!(!run(root, &args).status.success());
        let output = selected(root, &args);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("pending"));
    }
    let doctor = selected(root, &["doctor"]);
    assert!(!doctor.status.success());
    let diagnosis: Value = serde_json::from_slice(&doctor.stdout).unwrap();
    assert_eq!(diagnosis["data"]["installation_ready"], true);
    assert_eq!(diagnosis["data"]["training_config_ready"], false);
    assert_eq!(diagnosis["data"]["live_runtime_verified"], false);
    assert!(!root.join(".glr").exists());
    let context = data(selected(root, &["season", "show"]));
    let result = data(selected(root, &["runtime", "start"]));
    let run_id = result["run_id"].as_str().unwrap();
    let details = data(run(root, &["runs", "show", run_id]));
    assert_eq!(details["run"]["metadata"]["season_context"], context);
    assert!(
        details["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["role"] == "season-context")
    );
    assert!(
        details["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["kind"] == "season.selected")
    );
    let persisted: Value = serde_json::from_slice(
        &fs::read(
            root.join(".glr/runs")
                .join(run_id)
                .join("season-context.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(persisted, context);
    let declaration = root.join("config/seasons/example-season/standard.toml");
    fs::write(
        &declaration,
        fs::read_to_string(&declaration)
            .unwrap()
            .replace("\"pending\"", "\"ready\""),
    )
    .unwrap();
    assert_eq!(
        data(selected(root, &["train", "--no-capture"]))["status"],
        "succeeded"
    );
}

#[test]
fn initializer_never_overwrites_and_registers_only_pending_data() {
    let temporary = copy_fixture();
    let root = temporary.path();
    let args = [
        "--season",
        "next-season",
        "--ruleset",
        "standard",
        "season",
        "init",
    ];
    let initialized = data(run(root, &args));
    assert_eq!(initialized["status"], "pending");
    assert_eq!(initialized["extensions"], serde_json::json!({}));
    let before = fs::read(root.join("config/seasons.toml")).unwrap();
    assert!(!run(root, &args).status.success());
    assert_eq!(fs::read(root.join("config/seasons.toml")).unwrap(), before);
    fs::create_dir_all(root.join("config/seasons/owned")).unwrap();
    let owned = root.join("config/seasons/owned/standard.toml");
    fs::write(&owned, "user-owned").unwrap();
    assert!(
        !run(
            root,
            &[
                "--season",
                "owned",
                "--ruleset",
                "standard",
                "season",
                "init"
            ]
        )
        .status
        .success()
    );
    assert_eq!(fs::read_to_string(owned).unwrap(), "user-owned");
    assert_eq!(fs::read(root.join("config/seasons.toml")).unwrap(), before);
    assert!(!root.join("config/seasons.toml.lock").exists());
    assert!(!root.join(".glr").exists());
}

#[test]
fn malformed_or_escaping_declarations_fail_closed() {
    let temporary = copy_fixture();
    let root = temporary.path();
    let declaration = root.join("config/seasons/example-season/standard.toml");
    let original = fs::read_to_string(&declaration).unwrap();
    for (old, new) in [
        ("status = \"pending\"", "status = \"ready\"\nunknown = 1"),
        ("config/preset.toml", "../outside.toml"),
        ("config/preset.toml", "missing.toml"),
        ("ruleset_id = \"standard\"", "ruleset_id = \"other\""),
    ] {
        fs::write(&declaration, original.replace(old, new)).unwrap();
        assert!(!selected(root, &["season", "show"]).status.success());
    }
}
