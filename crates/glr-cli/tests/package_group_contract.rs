//! Negative corpus for the `glr.training-package.v1` optional groups (M4).
//!
//! Every case drives the real `glr` binary end to end and asserts a non-zero
//! exit code, a stable `--json` error category on stderr, and the absence of
//! every artifact the refusal should have prevented. It targets the gates in
//! `crates/glr-cli/src/package_groups.rs`: deny-by-default groups, the
//! `glr.model-bundle.v1`, `glr.demonstration-artifact.v1` and
//! `glr.knowledge-snapshot.v1` proofs, the reviewed dataset allowlist, the
//! aggregate-only report rule, and the expansion caps.
//!
//! Group work stays passive: the corpus carries a pickle payload that would run
//! if anything deserialized it, and asserts it arrives byte-identical with
//! `executed == false`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

const MANIFEST: &str = "glr-package.json";
const ENVIRONMENT: &str = "synthetic.package";
const PROTOCOL: &str = "1.0";
const REQUIRED_GLR: &str = ">=0.18.0, <1.0.0";
const CONTRACT: &str = "a";

/// A payload that only runs if something deserializes it. It never runs.
const PICKLE: &[u8] =
    b"\x80\x04\x95\x1c\x00\x00\x00\x00\x00\x00\x00\x8c\x02os\x94\x8c\x06system\x94\x93\x94.";

fn contract() -> String {
    CONTRACT.repeat(64)
}

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_glr"))
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// One synthetic project carrying one file in every optional group.
struct Corpus {
    root: TempDir,
}

impl Corpus {
    fn new() -> Self {
        let corpus = Self {
            root: tempfile::tempdir().unwrap(),
        };
        corpus.write(
            "glr-project.json",
            &serde_json::to_vec(&json!({
                "schema_version": "glr.project.v1",
                "environment_id": ENVIRONMENT,
                "protocol_version": PROTOCOL,
            }))
            .unwrap(),
        );
        corpus.write("uv.lock", b"version = 1\n");
        corpus.write("train.py", b"raise RuntimeError('must never execute')\n");
        corpus
    }

    fn path(&self) -> &Path {
        self.root.path()
    }

    fn write(&self, relative: &str, contents: &[u8]) -> PathBuf {
        let path = self.path().join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, contents).unwrap();
        path
    }

    fn select(&self, value: &Value) -> PathBuf {
        self.write("selection.json", &serde_json::to_vec(value).unwrap())
    }

    /// Writes a complete `glr.model-bundle.v1` under `models/reference/`.
    fn write_model(&self) {
        let config = b"{\"synthetic\": true}\n";
        let weights = PICKLE;
        self.write("models/reference/inputs/config.json", config);
        self.write("models/reference/artifacts/weights.safetensors", weights);
        self.write(
            "models/reference/manifest.json",
            &serde_json::to_vec(&json!({
                "schema_version": "glr.model-bundle.v1",
                "environment_id": ENVIRONMENT,
                "protocol_version": PROTOCOL,
                "algorithm": "synthetic",
                "framework": "synthetic",
                "framework_version": "1.0.0",
                "seeds": [7],
                "inputs": [{
                    "path": "config.json",
                    "sha256": digest(config),
                    "size_bytes": config.len(),
                }],
                "artifacts": [{
                    "path": "weights.safetensors",
                    "sha256": digest(weights),
                    "size_bytes": weights.len(),
                }],
            }))
            .unwrap(),
        );
    }

    /// Writes a demonstration artifact that binds one trajectory byte for byte.
    fn write_dataset(&self) -> Vec<u8> {
        let trajectory = b"{\"step\": 0, \"synthetic\": true}\n";
        self.write("data/demo/episode.jsonl", trajectory);
        let artifact = serde_json::to_vec(&json!({
            "schema_version": "glr.demonstration-artifact.v1",
            "environment_id": ENVIRONMENT,
            "episode_id": "00000000-0000-0000-0000-000000000001",
            "trajectory": {
                "path": "episode.jsonl",
                "sha256": digest(trajectory),
                "size_bytes": trajectory.len(),
            },
            "provenance": {"origin": "scripted-expert", "outcome": "success"},
        }))
        .unwrap();
        self.write("data/demo/artifact.json", &artifact);
        trajectory.to_vec()
    }

    /// Writes a knowledge snapshot stamped `days_ago` days in the past.
    fn write_knowledge(&self, days_ago: i64) {
        let created_at = rfc3339_days_ago(days_ago);
        self.write(
            "knowledge/snapshot.json",
            &serde_json::to_vec(&json!({
                "schema_version": "glr.knowledge-snapshot.v1",
                "snapshot_id": "snapshot.synthetic",
                "source_id": "source.synthetic",
                "created_at": created_at,
                "items": [{
                    "id": "item.synthetic",
                    "intent": "acquire",
                    "subject": "synthetic subject",
                    "summary": "synthetic advisory summary",
                }],
            }))
            .unwrap(),
        );
    }

    fn write_report(&self) {
        self.write(
            "reports/summary.json",
            &serde_json::to_vec(&json!({"episodes": 1, "synthetic": true})).unwrap(),
        );
    }
}

/// RFC 3339 UTC timestamp `days_ago` days before now.
fn rfc3339_days_ago(days_ago: i64) -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        - days_ago * 86_400;
    let days = seconds.div_euclid(86_400);
    let time = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        time / 3600,
        time % 3600 / 60,
        time % 60
    )
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    (if month <= 2 { year + 1 } else { year }, month, day)
}

// ---------------------------------------------------------------------------
// Selections
// ---------------------------------------------------------------------------

/// A `glr.training-package.v1` selection. `groups` maps a group name to its
/// list of `[path, role]` pairs.
fn selection(
    entry_groups: &[&str],
    groups: &BTreeMap<&str, Vec<(&str, &str)>>,
    extra: Value,
) -> Value {
    let mut value = json!({
        "schema_version": "glr.training-package.v1",
        "package_version": "1.0.0",
        "required_glr": REQUIRED_GLR,
        "environment_id": ENVIRONMENT,
        "protocol_version": PROTOCOL,
        "contract_sha256": contract(),
        "source_revision": "synthetic-groups",
        "redistribution_license": "MIT",
        "entry_groups": entry_groups,
        "groups": groups
            .iter()
            .map(|(group, files)| {
                (
                    group.to_string(),
                    json!({
                        "files": files
                            .iter()
                            .map(|(path, role)| json!({"path": path, "role": role}))
                            .collect::<Vec<_>>(),
                    }),
                )
            })
            .collect::<BTreeMap<_, _>>(),
    });
    for (key, nested) in extra.as_object().unwrap() {
        value[key] = nested.clone();
    }
    value
}

fn source_files() -> Vec<(&'static str, &'static str)> {
    vec![
        ("glr-project.json", "project-manifest"),
        ("train.py", "source-file"),
        ("uv.lock", "dependency-lock"),
    ]
}

fn model_files() -> Vec<(&'static str, &'static str)> {
    vec![
        ("models/reference/manifest.json", "model-manifest"),
        ("models/reference/inputs/config.json", "model-input"),
        (
            "models/reference/artifacts/weights.safetensors",
            "model-artifact",
        ),
    ]
}

fn knowledge_files() -> Vec<(&'static str, &'static str)> {
    vec![("knowledge/snapshot.json", "knowledge-snapshot")]
}

fn dataset_files() -> Vec<(&'static str, &'static str)> {
    vec![
        ("data/demo/artifact.json", "dataset-manifest"),
        ("data/demo/episode.jsonl", "dataset-payload"),
    ]
}

fn report_files() -> Vec<(&'static str, &'static str)> {
    vec![("reports/summary.json", "aggregate-report")]
}

fn authorization() -> Value {
    json!({
        "schema_version": "glr.redistribution-authorization.v1",
        "approver": "synthetic.approver",
        "scope": "synthetic redistribution review",
        "license": "MIT",
        "date": "2026-09-29",
    })
}

fn dataset_allowlist(entries: &[&str]) -> Value {
    json!({
        "schema_version": "glr.dataset-allowlist.v1",
        "reviewed_by": "synthetic.reviewer",
        "review_date": "2026-09-29",
        "entries": entries,
    })
}

/// A selection carrying every group, with the records a dataset export needs.
fn full_selection() -> Value {
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    groups.insert("model", model_files());
    groups.insert("dataset", dataset_files());
    groups.insert("knowledge", knowledge_files());
    groups.insert("report", report_files());
    let mut value = selection(
        &["source", "model", "dataset", "knowledge", "report"],
        &groups,
        json!({
            "dataset_allowlist": dataset_allowlist(&["data/demo/artifact.json", "data/demo/episode.jsonl"]),
            "redistribution_authorization": authorization(),
        }),
    );
    value["groups"]["knowledge"]["max_age_days"] = json!(30);
    value
}

// ---------------------------------------------------------------------------
// CLI driver
// ---------------------------------------------------------------------------

fn glr(project: &Path, arguments: &[&str]) -> Output {
    Command::new(binary())
        .env("GLR_NO_UPDATE_CHECK", "1")
        .arg("--project")
        .arg(project)
        .arg("--json")
        .args(arguments)
        .output()
        .unwrap()
}

fn plan(corpus: &Corpus, selection: &Path) -> Output {
    glr(
        corpus.path(),
        &[
            "package",
            "plan",
            "--manifest",
            &selection.to_string_lossy(),
        ],
    )
}

fn export(corpus: &Corpus, selection: &Path, output: &Path) -> Output {
    glr(
        corpus.path(),
        &[
            "package",
            "export",
            "--manifest",
            &selection.to_string_lossy(),
            "--output",
            &output.to_string_lossy(),
        ],
    )
}

fn inspect(corpus: &Corpus, archive: &Path) -> Output {
    glr(
        corpus.path(),
        &["package", "inspect", &archive.to_string_lossy()],
    )
}

fn import(corpus: &Corpus, archive: &Path, destination: &Path) -> Output {
    glr(
        corpus.path(),
        &[
            "package",
            "import",
            &archive.to_string_lossy(),
            "--destination",
            &destination.to_string_lossy(),
            "--expected-environment",
            ENVIRONMENT,
            "--expected-contract",
            &contract(),
        ],
    )
}

fn conformance(archive: &Path, destination: &Path) -> Output {
    glr(
        destination,
        &[
            "package",
            "conformance",
            &archive.to_string_lossy(),
            "--expected-environment",
            ENVIRONMENT,
            "--expected-contract",
            &contract(),
        ],
    )
}

/// Asserts a non-zero exit code, an empty stdout, and a stable error category.
#[track_caller]
fn refused(output: &Output, expected_type: &str, fragment: &str) {
    assert!(
        !output.status.success(),
        "expected a refusal, got status {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "a refusal must not emit stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let error: Value = serde_json::from_slice(&output.stderr).unwrap_or_else(|error| {
        panic!(
            "refusals must emit a JSON envelope on stderr: {error}\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert_eq!(error["command"], "error");
    let message = error["error"]["message"].as_str().unwrap_or_default();
    assert_eq!(
        error["error"]["type"].as_str().unwrap_or("<missing>"),
        expected_type,
        "unexpected category for message {message:?}"
    );
    assert!(
        message.contains(fragment),
        "expected {fragment:?} in {message:?}"
    );
}

/// Shorthand for the package refusals, whether the envelope or the group
/// policy raised them.
#[track_caller]
fn group_refusal(output: &Output, fragment: &str) {
    refused(output, "ContractViolation", "package:");
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    let message = error["error"]["message"].as_str().unwrap();
    assert!(
        message.contains(fragment),
        "expected {fragment:?} in {message:?}"
    );
}

/// The parsed `glr.cli-output.v1` result of a successful command.
fn result(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "expected success, got {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

// ---------------------------------------------------------------------------
// Archive helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize)]
struct ManifestMirror {
    selection: Value,
    tool_version: String,
    content_sha256: String,
    entries: Vec<EntryMirror>,
}

#[derive(Debug, Deserialize, Serialize)]
struct EntryMirror {
    path: String,
    size_bytes: u64,
    sha256: String,
    #[serde(default)]
    group: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    compression: Option<String>,
    #[serde(default)]
    compressed_size_bytes: Option<u64>,
}

type Members = BTreeMap<String, Vec<u8>>;

fn read_archive(path: &Path) -> Members {
    let mut archive = ZipArchive::new(Cursor::new(fs::read(path).unwrap())).unwrap();
    let mut members = BTreeMap::new();
    for index in 0..archive.len() {
        let mut file = archive.by_index(index).unwrap();
        let name = file.name().to_owned();
        let mut content = Vec::new();
        file.read_to_end(&mut content).unwrap();
        members.insert(name, content);
    }
    members
}

fn load_manifest(members: &Members) -> ManifestMirror {
    serde_json::from_slice(members.get(MANIFEST).expect("archive manifest")).unwrap()
}

fn store_manifest(members: &mut Members, manifest: &ManifestMirror) {
    members.insert(MANIFEST.into(), serde_json::to_vec(manifest).unwrap());
}

/// Writes members with the compression each one declares.
fn write_archive(path: &Path, members: &Members) {
    let mut writer = ZipWriter::new(fs::File::create(path).unwrap());
    let manifest = load_manifest(members);
    for (name, content) in members {
        let compressed = manifest
            .entries
            .iter()
            .find(|entry| &entry.path == name)
            .and_then(|entry| entry.compression.as_deref())
            == Some("deflated");
        let options = SimpleFileOptions::default()
            .compression_method(if compressed {
                CompressionMethod::Deflated
            } else {
                CompressionMethod::Stored
            })
            .unix_permissions(0o644);
        writer.start_file(name, options).unwrap();
        writer.write_all(content).unwrap();
    }
    writer.finish().unwrap();
}

/// A successful export of the full four-group package, plus its members.
fn exported_full_package() -> (Corpus, PathBuf, Members) {
    let corpus = Corpus::new();
    corpus.write_model();
    corpus.write_dataset();
    corpus.write_knowledge(1);
    corpus.write_report();
    let selection = corpus.select(&full_selection());
    let archive = corpus.path().join("package.zip");
    assert!(export(&corpus, &selection, &archive).status.success());
    let members = read_archive(&archive);
    (corpus, archive, members)
}

// ---------------------------------------------------------------------------
// 1. Deny by default
// ---------------------------------------------------------------------------

#[test]
fn group_content_without_its_declaration_is_refused() {
    let corpus = Corpus::new();
    corpus.write_model();
    corpus.write_knowledge(1);
    corpus.write_report();
    // Every group is present on disk; only `source` is declared. Each case
    // names one group in `groups` without declaring it in `entry_groups`.
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    for (group, files) in [
        ("model", model_files()),
        ("knowledge", knowledge_files()),
        ("report", report_files()),
    ] {
        let mut undeclared = groups.clone();
        undeclared.insert(group, files);
        let selection = corpus.select(&selection(&["source"], &undeclared, json!({})));
        group_refusal(&plan(&corpus, &selection), "is not declared");
    }
    // Declaring the group but omitting the file is a different failure: an
    // empty group is not a package.
    groups.insert("report", vec![]);
    let selection = corpus.select(&selection(&["source", "report"], &groups, json!({})));
    group_refusal(&plan(&corpus, &selection), "declares no files");
}

#[test]
fn unknown_groups_and_roles_fail_closed() {
    let corpus = Corpus::new();
    corpus.write_report();
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());

    let mut with_group = groups.clone();
    with_group.insert("report", report_files());
    let mut value = selection(&["source", "weights"], &with_group, json!({}));
    value["entry_groups"] = json!(["source", "weights"]);
    refused(
        &plan(&corpus, &corpus.select(&value)),
        "JsonError",
        "unknown variant",
    );

    let mut with_role = groups.clone();
    with_role.insert("report", vec![("reports/summary.json", "raw-log")]);
    let value = selection(&["source", "report"], &with_role, json!({}));
    group_refusal(&plan(&corpus, &corpus.select(&value)), "not a report role");

    // A source file declared with a role its path does not have.
    let mut mismatched = groups;
    mismatched.insert(
        "source",
        vec![
            ("glr-project.json", "project-manifest"),
            ("train.py", "dependency-lock"),
            ("uv.lock", "dependency-lock"),
        ],
    );
    let value = selection(&["source"], &mismatched, json!({}));
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "but its path is a source-file",
    );
}

// ---------------------------------------------------------------------------
// 2. model: glr.model-bundle.v1
// ---------------------------------------------------------------------------

#[test]
fn model_group_requires_a_verified_model_bundle() {
    let corpus = Corpus::new();
    corpus.write_model();
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    groups.insert("model", model_files());

    // No bundle manifest at all.
    let mut without = groups.clone();
    without.insert(
        "model",
        vec![(
            "models/reference/artifacts/weights.safetensors",
            "model-artifact",
        )],
    );
    let value = selection(&["source", "model"], &without, json!({}));
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "exactly one model-manifest",
    );

    // A manifest that is not a bundle.
    corpus.write(
        "models/reference/manifest.json",
        b"{\"schema_version\":\"nope\"}",
    );
    let value = selection(&["source", "model"], &groups, json!({}));
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "must be glr.model-bundle.v1",
    );

    // A bundle whose declared file the package omits.
    corpus.write_model();
    let mut partial = groups.clone();
    partial.insert(
        "model",
        vec![
            ("models/reference/manifest.json", "model-manifest"),
            ("models/reference/inputs/config.json", "model-input"),
        ],
    );
    let value = selection(&["source", "model"], &partial, json!({}));
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "which the package omits",
    );

    // A file the bundle does not declare.
    corpus.write("models/reference/artifacts/extra.bin", b"extra");
    let mut extra = groups.clone();
    let mut files = model_files();
    files.push(("models/reference/artifacts/extra.bin", "model-artifact"));
    extra.insert("model", files);
    let value = selection(&["source", "model"], &extra, json!({}));
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "is not declared by the model bundle",
    );
}

#[test]
fn model_group_refuses_a_weights_file_that_needs_a_deserializer() {
    let corpus = Corpus::new();
    corpus.write_model();
    corpus.write("models/reference/artifacts/weights.pkl", b"pickle");
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    groups.insert(
        "model",
        vec![
            ("models/reference/manifest.json", "model-manifest"),
            ("models/reference/artifacts/weights.pkl", "model-artifact"),
        ],
    );
    let value = selection(&["source", "model"], &groups, json!({}));
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "deserializer that import must never run",
    );
}

// ---------------------------------------------------------------------------
// 3. dataset: reviewed allowlist, authorization, provenance
// ---------------------------------------------------------------------------

#[test]
fn dataset_export_requires_an_allowlist_and_an_authorization() {
    let corpus = Corpus::new();
    corpus.write_dataset();
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    groups.insert("dataset", dataset_files());

    let value = selection(&["source", "dataset"], &groups, json!({}));
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "needs a separately reviewed dataset allowlist",
    );

    let value = selection(
        &["source", "dataset"],
        &groups,
        json!({"dataset_allowlist": dataset_allowlist(&["data/demo/artifact.json", "data/demo/episode.jsonl"])}),
    );
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "needs a recorded redistribution authorization",
    );

    // On the allowlist but not reviewed for redistribution is still a refusal:
    // the allowlist is a separate reviewed artifact, not a file list.
    let value = selection(
        &["source", "dataset"],
        &groups,
        json!({
            "dataset_allowlist": dataset_allowlist(&["data/demo/artifact.json"]),
            "redistribution_authorization": authorization(),
        }),
    );
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "is not on the reviewed dataset allowlist",
    );

    for (key, broken) in [
        ("schema_version", json!("glr.dataset-allowlist.v2")),
        ("review_date", json!("29/09/2026")),
        ("reviewed_by", json!("")),
    ] {
        let mut allowlist =
            dataset_allowlist(&["data/demo/artifact.json", "data/demo/episode.jsonl"]);
        allowlist[key] = broken;
        let value = selection(
            &["source", "dataset"],
            &groups,
            json!({
                "dataset_allowlist": allowlist,
                "redistribution_authorization": authorization(),
            }),
        );
        group_refusal(&plan(&corpus, &corpus.select(&value)), "training package:");
    }
}

#[test]
fn dataset_payloads_must_be_bound_by_a_demonstration_manifest() {
    let corpus = Corpus::new();
    let trajectory = corpus.write_dataset();
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    groups.insert("dataset", dataset_files());
    let records = json!({
        "dataset_allowlist": dataset_allowlist(&["data/demo/artifact.json", "data/demo/episode.jsonl"]),
        "redistribution_authorization": authorization(),
    });

    // The bound bytes changed after the manifest was written.
    corpus.write("data/demo/episode.jsonl", b"tampered\n");
    let value = selection(&["source", "dataset"], &groups, records.clone());
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "does not match the trajectory bytes",
    );

    // A payload no manifest binds.
    corpus.write("data/demo/episode.jsonl", &trajectory);
    corpus.write("data/demo/orphan.jsonl", b"orphan\n");
    let mut with_orphan = groups.clone();
    with_orphan.insert(
        "dataset",
        vec![
            ("data/demo/artifact.json", "dataset-manifest"),
            ("data/demo/episode.jsonl", "dataset-payload"),
            ("data/demo/orphan.jsonl", "dataset-payload"),
        ],
    );
    let mut allowlist = dataset_allowlist(&[
        "data/demo/artifact.json",
        "data/demo/episode.jsonl",
        "data/demo/orphan.jsonl",
    ]);
    let value = selection(
        &["source", "dataset"],
        &with_orphan,
        json!({
            "dataset_allowlist": allowlist,
            "redistribution_authorization": authorization(),
        }),
    );
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "no demonstration manifest binds",
    );
    let _ = &mut allowlist;

    // A manifest for another environment.
    corpus.write(
        "data/demo/artifact.json",
        &serde_json::to_vec(&json!({
            "schema_version": "glr.demonstration-artifact.v1",
            "environment_id": "other.environment",
            "episode_id": "00000000-0000-0000-0000-000000000001",
            "trajectory": {
                "path": "episode.jsonl",
                "sha256": digest(&trajectory),
                "size_bytes": trajectory.len(),
            },
            "provenance": {"origin": "scripted-expert", "outcome": "success"},
        }))
        .unwrap(),
    );
    let value = selection(&["source", "dataset"], &groups, records);
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "cannot join a package",
    );
}

// ---------------------------------------------------------------------------
// 4. knowledge: freshness-aware snapshots
// ---------------------------------------------------------------------------

#[test]
fn knowledge_snapshots_must_be_fresh_and_well_formed() {
    let corpus = Corpus::new();
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    groups.insert("knowledge", knowledge_files());

    // A freshness budget is mandatory.
    corpus.write_knowledge(1);
    let value = selection(&["source", "knowledge"], &groups, json!({}));
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "must declare max_age_days",
    );

    let mut value = selection(&["source", "knowledge"], &groups, json!({}));
    value["groups"]["knowledge"]["max_age_days"] = json!(30);

    // Older than the declared budget.
    corpus.write_knowledge(31);
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "past the 30 day freshness budget",
    );

    // Stamped after the verification clock.
    corpus.write_knowledge(-1);
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "stamped in the future",
    );

    // Not a knowledge snapshot.
    corpus.write("knowledge/snapshot.json", b"{\"schema_version\":\"nope\"}");
    group_refusal(
        &plan(&corpus, &corpus.select(&value)),
        "must be glr.knowledge-snapshot.v1",
    );

    // max_age_days belongs to the knowledge group alone.
    corpus.write_knowledge(1);
    let mut mismatched = selection(
        &["source", "report"],
        &BTreeMap::from([("source", source_files()), ("report", report_files())]),
        json!({}),
    );
    mismatched["groups"]["report"]["max_age_days"] = json!(30);
    corpus.write_report();
    group_refusal(
        &plan(&corpus, &corpus.select(&mismatched)),
        "only applies to the knowledge group",
    );
}

// ---------------------------------------------------------------------------
// 5. report: aggregates only
// ---------------------------------------------------------------------------

#[test]
fn report_group_refuses_raw_logs_recordings_and_trajectories() {
    let corpus = Corpus::new();
    corpus.write_report();
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    for path in [
        "reports/run.log",
        "reports/session.mp4",
        "reports/trajectory.jsonl",
        "reports/logs/aggregate.json",
        "reports/episodes/summary.csv",
        "reports/summary.yaml",
    ] {
        corpus.write(path, b"synthetic aggregate\n");
        let mut groups = groups.clone();
        groups.insert("report", vec![(path, "aggregate-report")]);
        let value = selection(&["source", "report"], &groups, json!({}));
        group_refusal(&plan(&corpus, &corpus.select(&value)), "training package:");
    }
}

// ---------------------------------------------------------------------------
// 6. Compressed groups: ratio and pre-write caps
// ---------------------------------------------------------------------------

#[test]
fn an_archive_bomb_is_refused_before_any_byte_is_expanded() {
    let corpus = Corpus::new();
    // 4 MiB of zeros deflates to roughly 4 KiB: far past the 200:1 ratio cap.
    corpus.write("reports/bomb.json", &vec![0_u8; 4 * 1024 * 1024]);
    let mut groups: BTreeMap<&str, Vec<(&str, &str)>> = BTreeMap::new();
    groups.insert("source", source_files());
    groups.insert("report", vec![("reports/bomb.json", "aggregate-report")]);
    let selection = corpus.select(&selection(&["source", "report"], &groups, json!({})));
    // Planning hashes bytes; the ratio only exists once they are compressed.
    assert!(plan(&corpus, &selection).status.success());
    let archive = corpus.path().join("bomb.zip");
    group_refusal(
        &export(&corpus, &selection, &archive),
        "past the 200:1 ratio cap",
    );
    assert!(
        !archive.exists(),
        "a refused export must leave no archive behind"
    );
}

#[test]
fn declared_expansion_past_the_group_cap_is_refused_before_writing() {
    let (corpus, _archive, mut members) = exported_full_package();
    let mut manifest = load_manifest(&members);
    // Declare a 1 GiB knowledge snapshot: past both its per-file and group caps.
    for entry in manifest.entries.iter_mut() {
        if entry.path == "knowledge/snapshot.json" {
            entry.size_bytes = 1024 * 1024 * 1024;
            entry.compressed_size_bytes = Some(4096);
        }
    }
    store_manifest(&mut members, &manifest);
    // The identity no longer matches the tampered inventory, so the refusal is
    // the identity gate for a sloppy forgery…
    let archive = corpus.path().join("inflated.zip");
    write_archive(&archive, &members);
    let destination = corpus.path().join("imported");
    let output = import(&corpus, &archive, &destination);
    assert!(!output.status.success());
    // …and nothing was materialized either way.
    assert!(
        !destination.exists(),
        "a refused import must leave no destination"
    );
}

#[test]
fn a_lying_compressed_size_is_refused() {
    let (corpus, _archive, mut members) = exported_full_package();
    let mut manifest = load_manifest(&members);
    for entry in manifest.entries.iter_mut() {
        if entry.path == "models/reference/artifacts/weights.safetensors" {
            entry.compressed_size_bytes = Some(1);
        }
    }
    store_manifest(&mut members, &manifest);
    let archive = corpus.path().join("lying.zip");
    write_archive(&archive, &members);
    let output = import(&corpus, &archive, &corpus.path().join("imported"));
    assert!(!output.status.success());
    assert!(!corpus.path().join("imported").exists());
}

// ---------------------------------------------------------------------------
// 7. Import deserializes nothing, and the audit receipt is aggregate
// ---------------------------------------------------------------------------

#[test]
fn every_group_round_trips_without_deserializing_and_with_a_complete_audit() {
    let (corpus, archive, _members) = exported_full_package();
    let inspected = result(&inspect(&corpus, &archive));
    assert_eq!(inspected["data"]["executed"], false);
    assert_eq!(inspected["data"]["status"], "verified-source-package");

    let audit = &inspected["data"]["audit"];
    assert_eq!(audit["schema_version"], "glr.package-audit.v1");
    let audited: BTreeSet<&str> = audit["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["path"].as_str().unwrap())
        .collect();
    let expected: BTreeSet<&str> = [
        "models/reference/manifest.json",
        "models/reference/inputs/config.json",
        "models/reference/artifacts/weights.safetensors",
        "data/demo/artifact.json",
        "data/demo/episode.jsonl",
        "knowledge/snapshot.json",
        "reports/summary.json",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        audited, expected,
        "the aggregate receipt must name every admitted non-source file"
    );
    // Source files are inventory, not a redistribution decision.
    assert!(!audited.contains("train.py"));
    for group in ["source", "model", "dataset", "knowledge", "report"] {
        assert_eq!(audit["groups"][group]["admission"], "verified", "{group}");
    }
    assert_eq!(
        audit["groups"]["model"]["checks"],
        json!(["glr.model-bundle.v1"])
    );
    assert_eq!(
        audit["groups"]["dataset"]["checks"],
        json!([
            "glr.dataset-allowlist.v1",
            "glr.demonstration-artifact.v1",
            "glr.redistribution-authorization.v1"
        ])
    );
    assert_eq!(audit["authorization"]["approver"], "synthetic.approver");
    assert_eq!(
        audit["dataset_allowlist"]["reviewed_by"],
        "synthetic.reviewer"
    );

    let destination = corpus.path().join("imported");
    let imported = result(&import(&corpus, &archive, &destination));
    assert_eq!(imported["data"]["executed"], false);
    assert_eq!(imported["data"]["training_ready"], false);
    // The pickle payload arrives byte-identical: nothing deserialized it.
    assert_eq!(
        fs::read(destination.join("models/reference/artifacts/weights.safetensors")).unwrap(),
        PICKLE
    );
    assert_eq!(
        fs::read(destination.join("data/demo/episode.jsonl")).unwrap(),
        fs::read(corpus.path().join("data/demo/episode.jsonl")).unwrap()
    );
    assert!(
        !destination.join(".glr").exists(),
        "package work must never create a run store"
    );
    // Conformance keeps the axes separate: this project has no role programs,
    // so reproduction is blocked and reported as a blocker, never as training.
    let output = conformance(&archive, &destination);
    assert_eq!(
        output.status.code(),
        Some(4),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["data"]["axes"]["package_validity"], "valid");
    assert_eq!(report["data"]["axes"]["materialization"], "complete");
    assert_eq!(report["data"]["axes"]["synthetic_reproduction"], "blocked");
    assert_eq!(report["data"]["claims"]["training_performed"], false);
    assert_eq!(report["data"]["claims"]["training_succeeded"], false);
    assert_eq!(
        report["data"]["audit"]["entries"].as_array().unwrap().len(),
        7
    );
}

#[test]
fn compressed_groups_are_compressed_and_source_stays_stored() {
    let (_corpus, _archive, members) = exported_full_package();
    let manifest = load_manifest(&members);
    for entry in &manifest.entries {
        match entry.group.as_deref() {
            Some("source") => assert_eq!(
                entry.compression.as_deref(),
                Some("stored"),
                "source stays uncompressed: {}",
                entry.path
            ),
            Some(_) => {
                assert_eq!(
                    entry.compression.as_deref(),
                    Some("deflated"),
                    "{}",
                    entry.path
                );
                assert!(
                    entry.compressed_size_bytes.is_some(),
                    "a compressed entry declares both sizes: {}",
                    entry.path
                );
            }
            other => panic!("unexpected group {other:?} for {}", entry.path),
        }
    }
}

#[test]
fn an_undeclared_group_in_the_manifest_is_refused_at_import() {
    let (corpus, _archive, mut members) = exported_full_package();
    let mut manifest = load_manifest(&members);
    // Drop the dataset group from the declaration but keep its entries: the
    // manifest and its inventory must agree.
    manifest.selection["entry_groups"] = json!(["source", "model", "knowledge", "report"]);
    manifest.selection["groups"]
        .as_object_mut()
        .unwrap()
        .remove("dataset");
    store_manifest(&mut members, &manifest);
    let archive = corpus.path().join("undeclared.zip");
    write_archive(&archive, &members);
    let output = import(&corpus, &archive, &corpus.path().join("imported"));
    assert!(!output.status.success());
    assert!(!corpus.path().join("imported").exists());
}
