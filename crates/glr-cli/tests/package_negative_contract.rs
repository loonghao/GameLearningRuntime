//! Negative security corpus for the `glr.source-package.v1` offline handoff.
//!
//! Every case drives the real `glr` binary end to end and asserts three things:
//! a non-zero exit code, a stable `--json` error category on stderr, and the
//! absence of every artifact the refusal should have prevented. The corpus
//! targets the `portable()` / `source_path()` / `no_links()` / `read_file()` /
//! `validate_selection()` / `inspect()` gates in `crates/glr-cli/src/package.rs`.
//!
//! Package work is passive: no role, installer, or hook ever runs. The corpus
//! proves that by carrying an executable-looking `train.py` through a full
//! export/import round trip and asserting `executed == false`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

// Mirrors of the production limits. Deliberately duplicated so the corpus fails
// loudly if the shipped contract moves without a matching corpus update.
const MAX_FILE: u64 = 16 * 1024 * 1024;
const MAX_TOTAL: u64 = 128 * 1024 * 1024;
const MAX_FILES: usize = 1024;
const MANIFEST: &str = "glr-package.json";
const ENVIRONMENT: &str = "synthetic.package";
const PROTOCOL: &str = "1.0";
const REQUIRED_GLR: &str = ">=0.18.0, <1.0.0";
const TRAIN: &[u8] = b"raise RuntimeError('must never execute')\n";

fn contract() -> String {
    "a".repeat(64)
}

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_glr"))
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct Corpus {
    root: TempDir,
}

impl Corpus {
    fn new() -> Self {
        let corpus = Self {
            root: tempfile::tempdir().unwrap(),
        };
        corpus.write("glr-project.json", &project_manifest(ENVIRONMENT, PROTOCOL));
        corpus.write("uv.lock", b"version = 1\n");
        corpus.write("train.py", TRAIN);
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

    /// Writes `selection.json` and returns its path.
    fn selection(&self, files: &[&str]) -> PathBuf {
        self.write_selection(&selection_value(files))
    }

    fn write_selection(&self, value: &Value) -> PathBuf {
        self.write("selection.json", &serde_json::to_vec(value).unwrap())
    }
}

fn project_manifest(environment: &str, protocol: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "schema_version": "glr.project.v1",
        "environment_id": environment,
        "protocol_version": protocol,
    }))
    .unwrap()
}

fn selection_value(files: &[&str]) -> Value {
    json!({
        "schema_version": "glr.source-package.v1",
        "package_version": "1.0.0",
        "required_glr": REQUIRED_GLR,
        "environment_id": ENVIRONMENT,
        "protocol_version": PROTOCOL,
        "contract_sha256": contract(),
        "source_revision": "negative-corpus",
        "redistribution_license": "MIT",
        "files": files,
    })
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

fn import(
    corpus: &Corpus,
    archive: &Path,
    destination: &Path,
    environment: &str,
    sha: &str,
) -> Output {
    glr(
        corpus.path(),
        &[
            "package",
            "import",
            &archive.to_string_lossy(),
            "--destination",
            &destination.to_string_lossy(),
            "--expected-environment",
            environment,
            "--expected-contract",
            sha,
        ],
    )
}

/// Asserts a non-zero exit code, an empty stdout, and a stable error category.
/// An empty `fragment` skips the message assertion.
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
    assert!(
        error["schema_version"].is_string(),
        "the error envelope must carry a schema version"
    );
    let message = error["error"]["message"].as_str().unwrap_or_default();
    assert_eq!(
        error["error"]["type"].as_str().unwrap_or("<missing>"),
        expected_type,
        "unexpected category for message {message:?}"
    );
    if !fragment.is_empty() {
        assert!(
            message.contains(fragment),
            "expected {fragment:?} in {message:?}"
        );
    }
}

/// Shorthand for the `contract violation: source package: …` family.
#[track_caller]
fn contract_refusal(output: &Output, fragment: &str) {
    refused(output, "ContractViolation", "source package:");
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    let message = error["error"]["message"].as_str().unwrap();
    assert!(
        message.contains(fragment),
        "expected {fragment:?} in {message:?}"
    );
}

/// For refusals whose category legitimately differs by platform or by how far
/// the malformed input gets before the operating system rejects it.
#[track_caller]
fn refused_one_of(output: &Output, expected_types: &[&str], fragment: &str) {
    let error: Value = serde_json::from_slice(&output.stderr).unwrap_or_else(|error| {
        panic!(
            "refusals must emit a JSON envelope on stderr: {error}\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    let kind = error["error"]["type"].as_str().unwrap_or("<missing>");
    assert!(
        expected_types.contains(&kind),
        "expected one of {expected_types:?}, got {kind}: {}",
        error["error"]["message"]
    );
    refused(output, kind, fragment);
}

// ---------------------------------------------------------------------------
// Side-effect assertions
// ---------------------------------------------------------------------------

fn assert_no_run_store(root: &Path) {
    assert!(
        !root.join(".glr").exists(),
        "package work must never create a run store"
    );
}

#[track_caller]
fn assert_only(dir: &Path, allowed: &[&str]) {
    let mut found: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    found.retain(|name| !allowed.iter().any(|allowed| allowed == name));
    found.sort();
    assert_eq!(
        found,
        Vec::<String>::new(),
        "unexpected leftovers in {}",
        dir.display()
    );
}

// ---------------------------------------------------------------------------
// Filesystem helpers (symlinks need privileges on some Windows hosts)
// ---------------------------------------------------------------------------

fn symlink(original: &Path, link: &Path) -> bool {
    #[cfg(windows)]
    let result = if original.is_dir() {
        std::os::windows::fs::symlink_dir(original, link)
    } else {
        std::os::windows::fs::symlink_file(original, link)
    };
    #[cfg(unix)]
    let result = std::os::unix::fs::symlink(original, link);
    result.is_ok()
}

// ---------------------------------------------------------------------------
// Archive helpers
// ---------------------------------------------------------------------------

fn read_archive(path: &Path) -> BTreeMap<String, Vec<u8>> {
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

fn archive_options() -> SimpleFileOptions {
    SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored)
        .unix_permissions(0o644)
}

fn write_archive(path: &Path, members: &BTreeMap<String, Vec<u8>>) {
    let mut writer = ZipWriter::new(fs::File::create(path).unwrap());
    for (name, content) in members {
        writer.start_file(name, archive_options()).unwrap();
        writer.write_all(content).unwrap();
    }
    writer.finish().unwrap();
}

/// Structural mirror of the production `Manifest`, used to recompute the
/// package identity so a corpus case can mutate one declared field while
/// keeping every earlier gate satisfied.
#[derive(Debug, Deserialize, Serialize)]
struct ManifestMirror {
    selection: SelectionMirror,
    tool_version: String,
    content_sha256: String,
    entries: Vec<EntryMirror>,
}

#[derive(Debug, Deserialize, Serialize)]
struct SelectionMirror {
    schema_version: String,
    package_version: String,
    required_glr: String,
    environment_id: String,
    protocol_version: String,
    contract_sha256: String,
    source_revision: String,
    redistribution_license: String,
    files: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct EntryMirror {
    path: String,
    size_bytes: u64,
    sha256: String,
}

fn identity(manifest: &ManifestMirror) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&(
                &manifest.selection,
                &manifest.tool_version,
                &manifest.entries
            ))
            .unwrap()
        )
    )
}

fn load_manifest(members: &BTreeMap<String, Vec<u8>>) -> ManifestMirror {
    serde_json::from_slice(members.get(MANIFEST).expect("archive manifest")).unwrap()
}

fn store_manifest(members: &mut BTreeMap<String, Vec<u8>>, manifest: &ManifestMirror) {
    let mut value = serde_json::to_value(manifest).unwrap();
    value["content_sha256"] = json!(identity(manifest));
    members.insert(MANIFEST.into(), serde_json::to_vec(&value).unwrap());
}

// ---------------------------------------------------------------------------
// 1. Path traversal, device names, and non-portable components
// ---------------------------------------------------------------------------

#[test]
fn selection_paths_that_escape_or_are_not_portable_are_refused() {
    let corpus = Corpus::new();
    let cases: &[(&str, &str)] = &[
        ("../outside.py", "non-portable path component"),
        ("a/../../outside.py", "non-portable path component"),
        ("/absolute.py", "non-portable path component"),
        ("//server/share/x.py", "non-portable path component"),
        ("C:/windows.py", "non-portable path component"),
        ("dir\\win.py", "non-portable path component"),
        ("CON.py", "non-portable path component"),
        ("NUL.py", "non-portable path component"),
        ("COM1.py", "non-portable path component"),
        ("LPT9.py", "non-portable path component"),
        ("trailing./x.py", "non-portable path component"),
        ("space /x.py", "non-portable path component"),
        ("trailing.py ", "non-portable path component"),
        ("pipe|.py", "non-portable path component"),
        ("quote\".py", "non-portable path component"),
        ("star*.py", "non-portable path component"),
        ("angle<.py", "non-portable path component"),
        ("angle>.py", "non-portable path component"),
        ("question?.py", "non-portable path component"),
        ("a//b.py", "non-portable path component"),
        ("./x.py", "non-portable path component"),
        ("a/./b.py", "non-portable path component"),
        ("a/b/..", "non-portable path component"),
        ("a\u{7}b.py", "non-portable path component"),
        ("a\tb.py", "non-portable path component"),
        ("a\0b.py", "non-portable path component"),
        ("café.py", "path exceeds portable limits"),
        ("train.py:stream", "non-portable path component"),
    ];
    for (path, fragment) in cases {
        let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock", path]);
        contract_refusal(&plan(&corpus, &selection), fragment);
    }
    assert_no_run_store(corpus.path());
}

#[test]
fn selection_paths_beyond_the_portable_envelope_are_refused() {
    let corpus = Corpus::new();
    // 16 components of 16 characters: 271 bytes, so only the length rule fires.
    let long: Vec<String> = (0..16).map(|_| "a".repeat(16)).collect();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock", &long.join("/")]);
    contract_refusal(&plan(&corpus, &selection), "path exceeds portable limits");

    // 17 short components: 33 bytes, so only the depth rule fires.
    let deep: Vec<String> = (0..17).map(|index| format!("d{index}")).collect();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock", &deep.join("/")]);
    contract_refusal(&plan(&corpus, &selection), "path exceeds portable limits");

    // Boundary control: 16 components is inside the envelope and must be
    // accepted, so the depth rule above is a real limit and not a blanket
    // rejection of nested paths.
    let nested: Vec<String> = (0..15).map(|index| format!("n{index}")).collect();
    let nested = format!("{}/leaf.py", nested.join("/"));
    corpus.write(&nested, b"pass\n");
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock", &nested]);
    assert!(
        plan(&corpus, &selection).status.success(),
        "a 16-component path is inside the portable envelope"
    );
}

// ---------------------------------------------------------------------------
// 2. Source-only allowlist
// ---------------------------------------------------------------------------

#[test]
fn non_source_and_machine_local_paths_are_refused() {
    let corpus = Corpus::new();
    let cases: &[&str] = &[
        // Not source.
        "model.bin",
        "image.png",
        "video.mp4",
        "data.csv",
        "archive.zip",
        "script.sh",
        "native.so",
        "native.dll",
        "notes",
        // Build outputs, captures, and datasets.
        "target/debug.py",
        "node_modules/dep.py",
        "recordings/clip.py",
        "screenshots/shot.py",
        "logs/run.py",
        "datasets/dataset.py",
        // Secrets and machine-local state.
        "secrets/key.py",
        "credentials/token.py",
        "cache/tmp.py",
        ".env",
        ".git/config",
        ".vscode/settings.json",
        "config.local.toml",
        "config.local.json",
        "bundle.local",
        "override.local.yaml",
        // A directory component that ends in `.local` is a local override too:
        // the rule matches per path component, not on the whole path string.
        "a.local/b.json",
        "a.local/sub/b.json",
        // The package manifest is generated, never selected.
        "glr-package.json",
    ];
    for path in cases {
        let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock", path]);
        contract_refusal(&plan(&corpus, &selection), "source-only allowlist");
    }
}

#[test]
fn selection_requires_exactly_one_manifest_and_a_lock() {
    let corpus = Corpus::new();
    let cases: &[&[&str]] = &[
        &["train.py", "uv.lock"],
        &[
            "glr-project.toml",
            "glr-project.json",
            "train.py",
            "uv.lock",
        ],
        &["glr-project.json", "train.py"],
        &[],
    ];
    for files in cases {
        let selection = corpus.selection(files);
        let output = plan(&corpus, &selection);
        refused(&output, "ContractViolation", "source package:");
        let message = serde_json::from_slice::<Value>(&output.stderr).unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .to_owned();
        assert!(
            message.contains("exactly one project manifest")
                || message.contains("unsupported schema or file count"),
            "unexpected message {message:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Link, hard link, and irregular file refusals
// ---------------------------------------------------------------------------

#[test]
fn hard_linked_sources_are_refused() {
    let corpus = Corpus::new();
    let alias = corpus.path().join("alias.py");
    if fs::hard_link(corpus.path().join("train.py"), &alias).is_err() {
        eprintln!("hard links unsupported on this filesystem; skipping");
        return;
    }
    for files in [
        ["glr-project.json", "uv.lock", "alias.py"],
        ["glr-project.json", "uv.lock", "train.py"],
    ] {
        let selection = corpus.selection(&files);
        contract_refusal(&plan(&corpus, &selection), "hard links are forbidden");
    }
    assert_no_run_store(corpus.path());
}

#[test]
fn symlinked_sources_selection_files_and_destinations_are_refused() {
    let corpus = Corpus::new();
    if !symlink(
        &corpus.path().join("train.py"),
        &corpus.path().join("link.py"),
    ) {
        eprintln!("symlinks unavailable on this host; skipping");
        return;
    }
    corpus.write("pkg/mod.py", b"pass\n");
    assert!(
        symlink(&corpus.path().join("pkg"), &corpus.path().join("alias")),
        "directory symlink creation failed"
    );

    let selection = corpus.selection(&["glr-project.json", "uv.lock", "link.py"]);
    contract_refusal(&plan(&corpus, &selection), "links are forbidden");

    let selection = corpus.selection(&["glr-project.json", "uv.lock", "alias/mod.py"]);
    contract_refusal(&plan(&corpus, &selection), "links are forbidden");

    // A symlinked selection file is refused before it is parsed.
    let real = corpus.write(
        "real-selection.json",
        &serde_json::to_vec(&selection_value(&[
            "glr-project.json",
            "train.py",
            "uv.lock",
        ]))
        .unwrap(),
    );
    assert!(symlink(&real, &corpus.path().join("linked-selection.json")));
    contract_refusal(
        &plan(&corpus, &corpus.path().join("linked-selection.json")),
        "links are forbidden",
    );

    // A symlinked output directory is refused before anything is staged.
    corpus.write("staged/keep.txt", b"keep\n");
    assert!(symlink(
        &corpus.path().join("staged"),
        &corpus.path().join("out")
    ));
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let output = export(&corpus, &selection, &corpus.path().join("out/package.zip"));
    contract_refusal(&output, "links are forbidden");
    assert!(!corpus.path().join("out/package.zip").exists());
}

#[test]
fn symlinked_project_manifest_is_refused_before_any_package_work() {
    let corpus = Corpus::new();
    let real = corpus.write(
        "real/glr-project.json",
        &project_manifest(ENVIRONMENT, PROTOCOL),
    );
    if !symlink(&real, &corpus.path().join("glr-project.json")) {
        eprintln!("symlinks unavailable on this host; skipping");
        return;
    }
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    refused(
        &plan(&corpus, &selection),
        "ValueError",
        "regular non-symlink file",
    );
}

#[test]
fn directories_selected_as_source_files_are_refused() {
    let corpus = Corpus::new();
    fs::create_dir(corpus.path().join("package.py")).unwrap();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock", "package.py"]);
    contract_refusal(&plan(&corpus, &selection), "file is not regular");
    assert_no_run_store(corpus.path());
}

// ---------------------------------------------------------------------------
// 4. Case collisions and file/directory conflicts
// ---------------------------------------------------------------------------

#[test]
fn case_collisions_and_file_directory_conflicts_are_refused() {
    let corpus = Corpus::new();
    let cases: &[(&[&str], &str)] = &[
        (
            &["glr-project.json", "uv.lock", "train.py", "A.py", "a.py"],
            "duplicate or case-colliding file",
        ),
        (
            &["glr-project.json", "uv.lock", "train.py", "train.py"],
            "duplicate or case-colliding file",
        ),
        (
            &["glr-project.json", "uv.lock", "Case/a.py", "case/b.py"],
            "case-colliding directory",
        ),
        (
            &[
                "glr-project.json",
                "uv.lock",
                "train.py",
                "train.py/nested.py",
            ],
            "file conflicts with a directory",
        ),
    ];
    for (files, fragment) in cases {
        let selection = corpus.selection(files);
        contract_refusal(&plan(&corpus, &selection), fragment);
    }
    assert_no_run_store(corpus.path());
}

// ---------------------------------------------------------------------------
// 5. Size and count limits
// ---------------------------------------------------------------------------

#[test]
fn oversized_files_and_payloads_are_refused() {
    let corpus = Corpus::new();
    corpus.write("big.py", &vec![0u8; MAX_FILE as usize + 1]);
    let selection = corpus.selection(&["glr-project.json", "uv.lock", "big.py"]);
    contract_refusal(&plan(&corpus, &selection), "exceeds size limit");

    // A byte over the 16 MiB per-file limit but inside the total budget still
    // has to be refused, so the per-file gate is real and not the total one.
    let corpus = Corpus::new();
    corpus.write("edge.py", &vec![0u8; MAX_FILE as usize + 1]);
    let selection = corpus.selection(&["glr-project.json", "uv.lock", "edge.py"]);
    contract_refusal(&plan(&corpus, &selection), "exceeds size limit");
    assert_no_run_store(corpus.path());
}

#[test]
fn payloads_over_the_total_budget_are_refused() {
    let corpus = Corpus::new();
    // Eight files of exactly MAX_FILE fill the 128 MiB budget; the project
    // manifest and the lock then push the expanded total past it.
    assert_eq!(
        MAX_FILE * 8,
        MAX_TOTAL,
        "this case assumes the budget is exactly eight files of MAX_FILE"
    );
    let mut files: Vec<String> = (0..8).map(|index| format!("bulk{index}.py")).collect();
    for name in &files {
        corpus.write(name, &vec![0u8; MAX_FILE as usize]);
    }
    files.push("glr-project.json".into());
    files.push("uv.lock".into());
    let selection = corpus.selection(&files.iter().map(String::as_str).collect::<Vec<_>>());
    contract_refusal(&plan(&corpus, &selection), "expanded size limit exceeded");
    assert_no_run_store(corpus.path());
}

#[test]
fn file_count_limits_are_enforced() {
    let corpus = Corpus::new();
    let mut files: Vec<String> = (0..MAX_FILES - 2)
        .map(|index| format!("f{index}.py"))
        .collect();
    for name in &files {
        corpus.write(name, b"pass\n");
    }
    files.push("glr-project.json".into());
    files.push("uv.lock".into());
    let selection = corpus.selection(&files.iter().map(String::as_str).collect::<Vec<_>>());
    let output = export(&corpus, &selection, &corpus.path().join("full.zip"));
    assert!(
        output.status.success(),
        "a selection of exactly MAX_FILES files is inside the contract: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(corpus.path().join("full.zip").exists());

    corpus.write("one_more.py", b"pass\n");
    files.push("one_more.py".into());
    let selection = corpus.selection(&files.iter().map(String::as_str).collect::<Vec<_>>());
    contract_refusal(
        &plan(&corpus, &selection),
        "unsupported schema or file count",
    );
}

#[test]
fn archives_with_too_many_members_are_refused() {
    let corpus = Corpus::new();
    let archive = corpus.path().join("inflated.zip");
    let mut members = BTreeMap::new();
    for index in 0..MAX_FILES + 2 {
        members.insert(format!("m{index}.py"), b"pass\n".to_vec());
    }
    write_archive(&archive, &members);
    contract_refusal(&inspect(&corpus, &archive), "archive file count exceeded");
}

// ---------------------------------------------------------------------------
// 6. Damaged and tampered archives
// ---------------------------------------------------------------------------

#[test]
fn truncated_and_rewritten_archives_are_refused() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let archive = corpus.path().join("valid.zip");
    assert!(export(&corpus, &selection, &archive).status.success());

    let bytes = fs::read(&archive).unwrap();
    let truncated = corpus.path().join("truncated.zip");
    fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();
    // A truncated archive is rejected while the ZIP reader is still reading
    // entries, so the failure surfaces as an I/O error on checksum and not as
    // a central-directory parse error.
    refused_one_of(
        &inspect(&corpus, &truncated),
        &["ArchiveError", "IoError"],
        "",
    );

    // Flipping a raw byte inside a stored entry breaks the ZIP checksum, so the
    // reader rejects the archive before the package digest is ever reached.
    let mut damaged = bytes.clone();
    let offset = damaged
        .windows(8)
        .position(|window| window == b"must nev")
        .expect("payload marker");
    damaged[offset] ^= 0xff;
    let damaged_path = corpus.path().join("damaged.zip");
    fs::write(&damaged_path, &damaged).unwrap();
    refused_one_of(
        &inspect(&corpus, &damaged_path),
        &["ArchiveError", "IoError"],
        "",
    );

    // Rewriting the same tampered payload through a ZIP writer repairs the
    // checksums, so this time the refusal has to come from the digest gate.
    let mut tampered = read_archive(&archive);
    tampered.insert("train.py".into(), b"silently changed\n".to_vec());
    let path = corpus.path().join("tampered.zip");
    write_archive(&path, &tampered);
    contract_refusal(&inspect(&corpus, &path), "file digest or size mismatch");
}

#[test]
fn archives_with_unexpected_members_are_refused() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let archive = corpus.path().join("valid.zip");
    assert!(export(&corpus, &selection, &archive).status.success());
    let members = read_archive(&archive);

    let mut extra = members.clone();
    extra.insert("unexpected.py".into(), b"smuggled\n".to_vec());
    let path = corpus.path().join("extra.zip");
    write_archive(&path, &extra);
    contract_refusal(
        &inspect(&corpus, &path),
        "package identity or inventory mismatch",
    );

    // Swapping one member's name for another keeps every count and set
    // consistent, so the refusal has to come from the per-entry lookup.
    let mut swapped = members.clone();
    let lock = swapped.remove("uv.lock").expect("lock member");
    swapped.insert("renamed.py".into(), lock);
    let path = corpus.path().join("swapped.zip");
    write_archive(&path, &swapped);
    contract_refusal(&inspect(&corpus, &path), "missing selected file");

    let mut bare = members.clone();
    bare.remove(MANIFEST);
    let path = corpus.path().join("bare.zip");
    write_archive(&path, &bare);
    contract_refusal(&inspect(&corpus, &path), "missing package manifest");

    let mut escaping = members.clone();
    escaping.insert("../escape.py".into(), b"escaped\n".to_vec());
    let path = corpus.path().join("escaping.zip");
    write_archive(&path, &escaping);
    contract_refusal(&inspect(&corpus, &path), "non-portable path component");
}

#[test]
fn archives_with_irregular_or_oversized_members_are_refused() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let archive = corpus.path().join("valid.zip");
    assert!(export(&corpus, &selection, &archive).status.success());
    let members = read_archive(&archive);
    let fragment = "archive has links, collisions, or oversized entries";

    // A case-differing member name is a collision on a case-insensitive
    // filesystem, even though the payload itself is a valid source file.
    let mut colliding = members.clone();
    colliding.insert("TRAIN.PY".into(), TRAIN.to_vec());
    let path = corpus.path().join("colliding.zip");
    write_archive(&path, &colliding);
    contract_refusal(&inspect(&corpus, &path), fragment);

    // A symlink member never becomes a symlink on disk.
    let path = corpus.path().join("symlink.zip");
    let mut writer = ZipWriter::new(fs::File::create(&path).unwrap());
    for (name, content) in &members {
        if name == "train.py" {
            writer
                .add_symlink(name, "uv.lock", archive_options())
                .unwrap();
        } else {
            writer.start_file(name, archive_options()).unwrap();
            writer.write_all(content).unwrap();
        }
    }
    writer.finish().unwrap();
    contract_refusal(&inspect(&corpus, &path), fragment);

    // A ZIP directory entry is not a source file. Its stored name always ends
    // in `/`, so the portable-path gate refuses it before anything is read.
    let path = corpus.path().join("directory.zip");
    let mut writer = ZipWriter::new(fs::File::create(&path).unwrap());
    for (name, content) in &members {
        writer.start_file(name, archive_options()).unwrap();
        writer.write_all(content).unwrap();
    }
    writer.add_directory("nested", archive_options()).unwrap();
    writer.finish().unwrap();
    contract_refusal(&inspect(&corpus, &path), "non-portable path component");

    // A declared member inflated past the per-file budget. The manifest is
    // re-signed so the inflated entry is internally consistent and the
    // per-file budget is the only gate left that can refuse it.
    let mut oversized = members.clone();
    let inflated = vec![0u8; MAX_FILE as usize + 1];
    oversized.insert("train.py".into(), inflated.clone());
    let mut manifest = load_manifest(&oversized);
    let entry = manifest
        .entries
        .iter_mut()
        .find(|entry| entry.path == "train.py")
        .unwrap();
    entry.size_bytes = inflated.len() as u64;
    entry.sha256 = format!("{:x}", Sha256::digest(&inflated));
    store_manifest(&mut oversized, &manifest);
    let path = corpus.path().join("oversized.zip");
    write_archive(&path, &oversized);
    contract_refusal(&inspect(&corpus, &path), fragment);
}

#[test]
fn declared_digests_and_sizes_must_match_the_payload() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let archive = corpus.path().join("valid.zip");
    assert!(export(&corpus, &selection, &archive).status.success());
    let mut members = read_archive(&archive);
    let mut manifest = load_manifest(&members);

    // Self-check: the mirror has to reproduce the shipped identity, otherwise
    // the mutations below would be testing the corpus and not the contract.
    assert_eq!(
        identity(&manifest),
        manifest.content_sha256,
        "corpus mirror must reproduce the production package identity"
    );

    let entry = manifest
        .entries
        .iter_mut()
        .find(|entry| entry.path == "train.py")
        .unwrap();
    entry.size_bytes += 1;
    store_manifest(&mut members, &manifest);
    let path = corpus.path().join("size.zip");
    write_archive(&path, &members);
    contract_refusal(&inspect(&corpus, &path), "file digest or size mismatch");

    let mut manifest = load_manifest(&read_archive(&archive));
    let entry = manifest
        .entries
        .iter_mut()
        .find(|entry| entry.path == "train.py")
        .unwrap();
    entry.sha256 = "b".repeat(64);
    store_manifest(&mut members, &manifest);
    let path = corpus.path().join("digest.zip");
    write_archive(&path, &members);
    contract_refusal(&inspect(&corpus, &path), "file digest or size mismatch");
}

// ---------------------------------------------------------------------------
// 7. Schema and provenance
// ---------------------------------------------------------------------------

#[test]
fn unknown_and_missing_schema_fields_are_refused() {
    let corpus = Corpus::new();
    let base = selection_value(&["glr-project.json", "train.py", "uv.lock"]);

    let mut value = base.clone();
    value["schema_version"] = json!("glr.source-package.v2");
    contract_refusal(
        &plan(&corpus, &corpus.write_selection(&value)),
        "unsupported schema or file count",
    );

    for field in ["contract_sha256", "files", "package_version"] {
        let mut value = base.clone();
        value.as_object_mut().unwrap().remove(field);
        refused(
            &plan(&corpus, &corpus.write_selection(&value)),
            "JsonError",
            "missing field",
        );
    }

    for field in ["smuggled", "post_install", "exec"] {
        let mut value = base.clone();
        value[field] = json!("anything");
        let output = plan(&corpus, &corpus.write_selection(&value));
        refused(&output, "JsonError", "unknown field");
    }
}

#[test]
fn malformed_provenance_and_fingerprints_are_refused() {
    let corpus = Corpus::new();
    let base = selection_value(&["glr-project.json", "train.py", "uv.lock"]);

    for value in [contract().to_uppercase(), "a".repeat(63), "a".repeat(65)] {
        let mut selection = base.clone();
        selection["contract_sha256"] = json!(value);
        contract_refusal(
            &plan(&corpus, &corpus.write_selection(&selection)),
            "contract fingerprint must be a lowercase SHA-256",
        );
    }

    for provenance in ["", &"e".repeat(300), "synthetic\npackage"] {
        for field in [
            "environment_id",
            "protocol_version",
            "source_revision",
            "redistribution_license",
        ] {
            let mut selection = base.clone();
            selection[field] = json!(provenance);
            contract_refusal(
                &plan(&corpus, &corpus.write_selection(&selection)),
                "invalid provenance or environment identity",
            );
        }
    }

    let mut selection = base.clone();
    selection["package_version"] = json!("1.0");
    refused(
        &plan(&corpus, &corpus.write_selection(&selection)),
        "VersionError",
        "",
    );

    let mut selection = base.clone();
    selection["required_glr"] = json!(">=999.0.0");
    contract_refusal(
        &plan(&corpus, &corpus.write_selection(&selection)),
        "incompatible GLR version",
    );

    let mut selection = base.clone();
    selection["required_glr"] = json!("not a requirement");
    refused(
        &plan(&corpus, &corpus.write_selection(&selection)),
        "VersionError",
        "",
    );
}

// ---------------------------------------------------------------------------
// 8. Environment, protocol, and contract mismatch
// ---------------------------------------------------------------------------

#[test]
fn project_manifest_identity_must_match_the_selection() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);

    corpus.write(
        "glr-project.json",
        &project_manifest("other.environment", PROTOCOL),
    );
    contract_refusal(
        &plan(&corpus, &selection),
        "project manifest identity does not match package",
    );

    corpus.write("glr-project.json", &project_manifest(ENVIRONMENT, "2.0"));
    contract_refusal(
        &plan(&corpus, &selection),
        "project manifest identity does not match package",
    );

    corpus.write("glr-project.json", &project_manifest(ENVIRONMENT, PROTOCOL));
    assert!(plan(&corpus, &selection).status.success());
}

#[test]
fn import_requires_matching_environment_and_contract_fingerprints() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let archive = corpus.path().join("valid.zip");
    assert!(export(&corpus, &selection, &archive).status.success());

    for (environment, sha) in [
        ("other.environment", contract()),
        (ENVIRONMENT, "b".repeat(64)),
    ] {
        let destination = corpus.path().join(format!("denied-{environment}-{sha}"));
        let output = import(&corpus, &archive, &destination, environment, &sha);
        contract_refusal(&output, "environment or contract fingerprint mismatch");
        assert!(!destination.exists(), "refused imports must not land");
    }

    let occupied = corpus.path().join("occupied");
    fs::create_dir(&occupied).unwrap();
    corpus.write("occupied/sentinel", b"keep\n");
    contract_refusal(
        &import(&corpus, &archive, &occupied, ENVIRONMENT, &contract()),
        "destination already exists",
    );
    assert_eq!(
        fs::read(occupied.join("sentinel")).unwrap(),
        b"keep\n".to_vec(),
        "an occupied destination must never be replaced"
    );
    assert_only(&occupied, &["sentinel"]);
}

// ---------------------------------------------------------------------------
// 9. No side effects, and packages never execute
// ---------------------------------------------------------------------------

#[test]
fn refused_package_work_leaves_nothing_behind() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock", "../escape.py"]);
    let output = corpus.path().join("never.zip");
    contract_refusal(&export(&corpus, &selection, &output), "non-portable");
    assert!(
        !output.exists(),
        "a refused export must not write an archive"
    );
    assert_only(
        corpus.path(),
        &["glr-project.json", "uv.lock", "train.py", "selection.json"],
    );
    assert_no_run_store(corpus.path());

    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let archive = corpus.path().join("valid.zip");
    assert!(export(&corpus, &selection, &archive).status.success());
    let staging_parent = corpus.path().join("landing");
    fs::create_dir(&staging_parent).unwrap();
    let destination = staging_parent.join("project");
    contract_refusal(
        &import(
            &corpus,
            &archive,
            &destination,
            "other.environment",
            &contract(),
        ),
        "environment or contract fingerprint mismatch",
    );
    assert!(!destination.exists());
    assert_only(&staging_parent, &[]);
    assert_only(
        corpus.path(),
        &[
            "glr-project.json",
            "uv.lock",
            "train.py",
            "selection.json",
            "valid.zip",
            "landing",
        ],
    );
}

#[test]
fn an_existing_archive_is_never_overwritten() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let archive = corpus.path().join("existing.zip");
    fs::write(&archive, b"do not replace").unwrap();
    refused(&export(&corpus, &selection, &archive), "IoError", "");
    assert_eq!(fs::read(&archive).unwrap(), b"do not replace".to_vec());
    assert_only(
        corpus.path(),
        &[
            "glr-project.json",
            "uv.lock",
            "train.py",
            "selection.json",
            "existing.zip",
        ],
    );
}

#[test]
fn a_verified_package_round_trips_without_executing_anything() {
    let corpus = Corpus::new();
    let selection = corpus.selection(&["glr-project.json", "train.py", "uv.lock"]);
    let archive = corpus.path().join("valid.zip");
    let output = export(&corpus, &selection, &archive);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let exported: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(exported["command"], "package");
    assert_eq!(exported["data"]["executed"], false);
    assert_eq!(exported["data"]["status"], "verified-source-inventory");

    let inspected: Value = serde_json::from_slice(&inspect(&corpus, &archive).stdout).unwrap();
    assert_eq!(inspected["data"]["training_ready"], false);
    assert_eq!(inspected["data"]["executed"], false);

    let destination = corpus.path().join("imported");
    let output = import(&corpus, &archive, &destination, ENVIRONMENT, &contract());
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(destination.join("train.py")).unwrap(),
        TRAIN.to_vec(),
        "the executable-looking source must arrive byte for byte and unrun"
    );
    assert!(
        !destination.join("selection.json").exists(),
        "the selection file is not part of the payload"
    );
    assert_eq!(
        fs::read(destination.join("glr-project.json")).unwrap(),
        project_manifest(ENVIRONMENT, PROTOCOL)
    );
    assert_no_run_store(corpus.path());
    assert_no_run_store(&destination);
}
