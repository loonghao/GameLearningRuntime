use std::fs;

use std::path::Path;
use std::process::{Command, Output};

fn doctor(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_glr"))
        .arg("--project")
        .arg(path)
        .args(["--json", "doctor"])
        .output()
        .unwrap()
}

fn read_project(path: &Path) -> serde_json::Value {
    let output = doctor(path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()["data"].clone()
}

const MANIFEST: &str = r#"
schema_version = "glr.project.v1"
environment_id = "example.environment-v1"
environment_family = "example-family"
protocol_version = "1.0"
data_dir = ".glr"
bridge_path = "bridge"
[runtime]
argv = ["python", "runtime.py", "{project_manifest}"]
[trainer]
argv = ["python", "train.py"]
[player]
argv = ["python", "play.py"]
"#;

fn manifest_text() -> String {
    // The standalone CLI contract must not depend on a Python installation.
    MANIFEST.replace(
        "\"python\"",
        &serde_json::to_string(env!("CARGO_BIN_EXE_glr")).unwrap(),
    )
}

#[test]
fn resolves_nearest_manifest_and_optional_roles() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::create_dir(root.join("bridge")).unwrap();
    fs::create_dir_all(root.join("src/nested")).unwrap();
    let manifest = root.join("glr-project.toml");
    fs::write(&manifest, manifest_text()).unwrap();
    let project = read_project(&root.join("src/nested"));
    assert_eq!(
        project["project_root"],
        fs::canonicalize(root).unwrap().to_string_lossy().as_ref()
    );
    assert_eq!(
        project["project_manifest"],
        fs::canonicalize(manifest)
            .unwrap()
            .to_string_lossy()
            .as_ref()
    );
    assert_eq!(project["extensions"], serde_json::json!({}));
}

#[test]
fn rejects_ambiguous_manifests_even_for_explicit_paths() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::write(root.join("glr-project.toml"), manifest_text()).unwrap();
    fs::write(root.join("glr-project.json"), "{}").unwrap();
    for input in [
        root.to_path_buf(),
        root.join("glr-project.toml"),
        root.join("glr-project.json"),
    ] {
        let output = doctor(&input);
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("multiple project manifests"), "{error}");
    }
}

#[test]
fn extension_mounts_are_explicit_strict_and_project_owned() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::create_dir(root.join("bridge")).unwrap();
    fs::create_dir(root.join("config")).unwrap();
    fs::write(
        root.join("config/runtime.toml"),
        "[game]\ndirectory = 'game'\n",
    )
    .unwrap();
    fs::write(
        root.join("glr-project.toml"),
        format!(
            "{}\n[extensions.example]\nconfig = 'config/runtime.toml'\n",
            manifest_text()
        ),
    )
    .unwrap();
    let project = read_project(root);
    assert_eq!(
        project["extensions"]["example"],
        fs::canonicalize(root.join("config/runtime.toml"))
            .unwrap()
            .to_string_lossy()
            .as_ref()
    );
    for entry in [
        "config = '../outside.toml'",
        "config = 'missing.toml'",
        "config = 'config/runtime.toml'\nextra = 1",
    ] {
        fs::write(
            root.join("glr-project.toml"),
            format!("{}\n[extensions.example]\n{entry}\n", manifest_text()),
        )
        .unwrap();
        assert!(!doctor(root).status.success());
    }
}
