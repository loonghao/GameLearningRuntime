//! Recipient conformance for a materialized offline source package.
//!
//! These tests cover the three scenarios #116 stage 4 names: a clean directory
//! round trip, a nested working directory, and a missing prerequisite. They use
//! synthetic fixtures only, and every command runs with an unroutable proxy so
//! an accidental network access fails instead of silently succeeding.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{Value, json};
use tempfile::TempDir;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_glr"))
}

/// Every command runs with a proxy that cannot be reached, and with the release
/// notice disabled, so any network attempt becomes a visible failure. This is a
/// best-effort offline assertion, not a syscall-level network sandbox.
fn glr(project: &Path, arguments: &[&str]) -> Output {
    Command::new(binary())
        .env("GLR_NO_UPDATE_CHECK", "1")
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("HTTPS_PROXY", "http://127.0.0.1:9")
        .env("ALL_PROXY", "http://127.0.0.1:9")
        .env("http_proxy", "http://127.0.0.1:9")
        .env("https_proxy", "http://127.0.0.1:9")
        .env("all_proxy", "http://127.0.0.1:9")
        .env("NO_PROXY", "")
        .arg("--project")
        .arg(project)
        .arg("--json")
        .args(arguments)
        .output()
        .unwrap()
}

fn data(output: &Output) -> Value {
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "expected a JSON envelope: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    envelope["data"].clone()
}

/// A synthetic source project whose materialized form is a loadable project.
fn source_project(trainer: &str) -> TempDir {
    let temporary = tempfile::tempdir().unwrap();
    let executable = binary().to_string_lossy().into_owned();
    fs::write(
        temporary.path().join("glr-project.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version": "glr.project.v1",
            "environment_id": "synthetic.package",
            "environment_family": "synthetic",
            "protocol_version": "1.0",
            "data_dir": ".glr",
            "bridge_path": "bridge",
            "runtime": {"argv": [executable, "--version"]},
            "trainer": {"argv": [trainer, "--version"]},
            "player": {"argv": [executable, "--version"]}
        }))
        .unwrap(),
    )
    .unwrap();
    fs::create_dir(temporary.path().join("bridge")).unwrap();
    fs::write(temporary.path().join("bridge/README.md"), b"bridge\n").unwrap();
    fs::write(temporary.path().join("uv.lock"), b"version = 1\n").unwrap();
    // A selected file that must never be executed by import or conformance.
    fs::write(
        temporary.path().join("train.py"),
        b"raise RuntimeError('must never execute')\n",
    )
    .unwrap();
    fs::write(
        temporary.path().join("selection.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version": "glr.source-package.v1",
            "package_version": "1.0.0",
            "required_glr": ">=0.18.0, <1.0.0",
            "environment_id": "synthetic.package",
            "protocol_version": "1.0",
            "contract_sha256": "a".repeat(64),
            "source_revision": "synthetic-conformance",
            "redistribution_license": "MIT",
            "files": ["glr-project.json", "bridge/README.md", "train.py", "uv.lock"]
        }))
        .unwrap(),
    )
    .unwrap();
    temporary
}

/// Export, then import into a directory that does not exist yet.
fn round_trip(trainer: &str) -> (TempDir, TempDir, PathBuf, PathBuf) {
    let source = source_project(trainer);
    let archive = source.path().join("source.zip");
    let output = glr(
        source.path(),
        &[
            "package",
            "export",
            "--manifest",
            "selection.json",
            "--output",
            archive.to_str().unwrap(),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let recipient = tempfile::tempdir().unwrap();
    let destination = recipient.path().join("imported");
    let output = glr(
        recipient.path(),
        &[
            "package",
            "import",
            archive.to_str().unwrap(),
            "--destination",
            destination.to_str().unwrap(),
            "--expected-environment",
            "synthetic.package",
            "--expected-contract",
            &"a".repeat(64),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (source, recipient, destination, archive)
}

fn collect(root: &Path, prefix: &str, output: &mut Vec<String>) {
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        let relative = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if entry.path().is_dir() {
            collect(&entry.path(), &relative, output);
        } else {
            output.push(relative);
        }
    }
}

#[test]
fn clean_directory_round_trip_materializes_only_declared_files_offline() {
    let executable = binary().to_string_lossy().into_owned();
    let (source, recipient, destination, archive) = round_trip(&executable);

    let mut present = Vec::new();
    collect(&destination, "", &mut present);
    present.sort();
    assert_eq!(
        present,
        vec![
            "bridge/README.md",
            "glr-project.json",
            "train.py",
            "uv.lock"
        ]
    );
    assert!(
        !destination.join(".glr").exists(),
        "run store in destination"
    );
    assert!(!destination.join("runs.sqlite3").exists());
    assert!(!source.path().join(".glr").exists(), "run store in source");
    assert!(
        !recipient.path().join(".glr").exists(),
        "run store in recipient"
    );

    let output = glr(
        &destination,
        &["package", "conformance", archive.to_str().unwrap()],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = data(&output);
    assert_eq!(report["schema_version"], "glr.package-conformance.v1");
    assert_eq!(report["axes"]["package_validity"], "valid");
    assert_eq!(report["axes"]["materialization"], "complete");
    assert_eq!(report["axes"]["dependency_setup"], "declared");
    assert_eq!(report["axes"]["synthetic_reproduction"], "pass");
    assert_eq!(report["axes"]["training"], "not-evaluated");
    assert_eq!(report["axes"]["live_acceptance"], "not-evaluated");
    assert_eq!(report["artifacts"]["run_store"], false);
    assert!(
        report["materialization"]["unexpected"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(report["dependency_setup"]["performed"], false);
    assert_eq!(report["claims"]["training_performed"], false);
    assert_eq!(report["claims"]["training_succeeded"], false);
    assert_eq!(report["executed"], false);
    assert_eq!(report["offline"], true);
}

#[test]
fn conformance_resolves_the_project_from_a_nested_working_directory() {
    let executable = binary().to_string_lossy().into_owned();
    let (_source, _recipient, destination, archive) = round_trip(&executable);
    let nested = destination.join("bridge");
    let output = glr(
        &nested,
        &["package", "conformance", archive.to_str().unwrap()],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = data(&output);
    assert_eq!(
        report["destination"],
        json!(fs::canonicalize(&destination).unwrap())
    );
    assert_eq!(report["axes"]["synthetic_reproduction"], "pass");
}

#[test]
fn missing_prerequisite_blocks_reproduction_and_is_reported_as_a_blocker() {
    let (_source, _recipient, destination, archive) =
        round_trip("glr-missing-synthetic-trainer-binary");
    let output = glr(
        &destination,
        &["package", "conformance", archive.to_str().unwrap()],
    );
    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(4));
    let report = data(&output);
    // A blocked reproduction is still a valid, completely materialized package.
    assert_eq!(report["package"]["valid"], true);
    assert_eq!(report["axes"]["package_validity"], "valid");
    assert_eq!(report["axes"]["materialization"], "complete");
    assert_eq!(report["axes"]["synthetic_reproduction"], "blocked");
    assert_eq!(report["claims"]["training_performed"], false);
    assert_eq!(report["claims"]["training_succeeded"], false);
    let blockers = report["blockers"].as_array().unwrap();
    assert!(
        blockers
            .iter()
            .any(|blocker| blocker["kind"] == "prerequisite"),
        "{blockers:?}"
    );
    assert!(
        blockers.iter().any(|blocker| blocker["remediation"]
            .as_str()
            .unwrap()
            .contains("never downloads or installs")),
        "{blockers:?}"
    );
    // Nothing installed or resolved the dependency as a side effect.
    assert_eq!(report["dependency_setup"]["performed"], false);
    assert_eq!(report["executed"], false);
}

#[test]
fn recipient_local_overrides_are_ignored_and_never_merged() {
    let executable = binary().to_string_lossy().into_owned();
    let (_source, _recipient, destination, archive) = round_trip(&executable);
    fs::write(destination.join("glr-project.local.json"), b"{}").unwrap();
    let output = glr(
        &destination,
        &["package", "conformance", archive.to_str().unwrap()],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = data(&output);
    assert_eq!(
        report["local_overrides"]["present_in_destination"],
        json!(["glr-project.local.json"])
    );
    assert_eq!(report["local_overrides"]["merged"], false);
    assert!(
        report["local_overrides"]["packaged"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert!(
        report["materialization"]["unexpected"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn conformance_rejects_a_reviewed_expectation_that_does_not_match() {
    let executable = binary().to_string_lossy().into_owned();
    let (_source, _recipient, destination, archive) = round_trip(&executable);
    let output = glr(
        &destination,
        &[
            "package",
            "conformance",
            archive.to_str().unwrap(),
            "--expected-environment",
            "other.environment",
        ],
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty(), "a refusal emits no receipt");
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["error"]["type"], "ContractViolation");
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("environment or contract fingerprint mismatch"),
        "{error}"
    );
}
