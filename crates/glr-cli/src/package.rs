//! Offline, explicit source-only packages. No role, installer, or hook execution.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use zip::{ZipArchive, ZipWriter, write::SimpleFileOptions};

use crate::args::PackageCommand;
use crate::commands::emit;
use crate::error::{Error, Result};
use crate::filesystem::promote;
use crate::process::executable_available;
use crate::project::{Project, find_project, load_project};

const SCHEMA: &str = "glr.source-package.v1";
const CONFORMANCE_SCHEMA: &str = "glr.package-conformance.v1";
const MAX_FILE: u64 = 16 * 1024 * 1024;
const MAX_TOTAL: u64 = 128 * 1024 * 1024;
const MAX_FILES: usize = 1024;
const MAX_DEPTH: usize = 16;
const MANIFEST: &str = "glr-package.json";
/// Exit code for a materialized package whose reproduction is blocked.
const BLOCKED_EXIT_CODE: i32 = 4;

/// Roots that must never appear inside a freshly materialized package: they are
/// caches, local run state, or captured/output artifacts, never redistributable
/// source. The denied roots of the export allowlist, plus the default run store.
const FORBIDDEN_ARTIFACT_DIRS: &[&str] = &[
    ".glr",
    "cache",
    "credentials",
    "datasets",
    "logs",
    "node_modules",
    "recordings",
    "screenshots",
    "secrets",
    "target",
];
const RUN_STORE_FILE: &str = "runs.sqlite3";

/// Recipient-local override forms. `source_path` refuses them in a package; a
/// conformance scan ignores them in the destination and never merges them.
const LOCAL_OVERRIDE_PATTERNS: &[&str] = &["*.local.*", "*.local"];

fn is_local_override(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains(".local.") || lower.ends_with(".local")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    schema_version: String,
    package_version: String,
    required_glr: String,
    environment_id: String,
    protocol_version: String,
    /// Project-owned fingerprint over observation/action/reward/knowledge/content rules.
    contract_sha256: String,
    source_revision: String,
    redistribution_license: String,
    files: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    selection: Selection,
    tool_version: String,
    content_sha256: String,
    entries: Vec<Entry>,
}

fn refusal(message: &str) -> Error {
    Error::Contract(format!("source package: {message}"))
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn portable(raw: &str) -> Result<()> {
    if raw.len() > 240 || raw.split('/').count() > 16 || !raw.is_ascii() {
        return Err(refusal("path exceeds portable limits"));
    }
    for part in raw.split('/') {
        let stem = part
            .split('.')
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase();
        if part.is_empty()
            || part == "."
            || part == ".."
            || part.ends_with(['.', ' '])
            || part
                .bytes()
                .any(|c| c < 32 || c == 127 || b"\\:<>\"|?*".contains(&c))
            || ["CON", "PRN", "AUX", "NUL"].contains(&stem.as_str())
            || ((stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.len() == 4
                && stem.as_bytes()[3].is_ascii_digit())
        {
            return Err(refusal("non-portable path component"));
        }
    }
    Ok(())
}

fn source_path(raw: &str) -> Result<()> {
    portable(raw)?;
    let lower = raw.to_ascii_lowercase();
    if lower.split('/').any(|part| {
        part.starts_with('.')
            || [
                "target",
                "node_modules",
                "recordings",
                "screenshots",
                "logs",
                "datasets",
                "secrets",
                "credentials",
                "cache",
            ]
            .contains(&part)
    }) || lower.contains(".local.")
        || lower.ends_with(".local")
        || lower == MANIFEST
        || ![
            "py", "rs", "toml", "json", "yaml", "yml", "md", "txt", "lock", "cs", "cpp", "h",
            "hpp", "gd",
        ]
        .contains(&lower.rsplit('.').next().unwrap_or_default())
    {
        return Err(refusal("source-only allowlist excludes this path"));
    }
    Ok(())
}

fn validate_selection(selection: &Selection) -> Result<()> {
    if selection.schema_version != SCHEMA
        || selection.files.is_empty()
        || selection.files.len() > MAX_FILES
    {
        return Err(refusal("unsupported schema or file count"));
    }
    Version::parse(&selection.package_version)?;
    let required = VersionReq::parse(&selection.required_glr)?;
    if !required.matches(&Version::parse(env!("CARGO_PKG_VERSION"))?) {
        return Err(refusal("incompatible GLR version"));
    }
    for value in [
        &selection.environment_id,
        &selection.protocol_version,
        &selection.source_revision,
        &selection.redistribution_license,
    ] {
        if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
            return Err(refusal("invalid provenance or environment identity"));
        }
    }
    if selection.contract_sha256.len() != 64
        || !selection
            .contract_sha256
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        return Err(refusal("contract fingerprint must be a lowercase SHA-256"));
    }
    let mut names = BTreeSet::new();
    let mut prefixes = std::collections::BTreeMap::new();
    for path in &selection.files {
        source_path(path)?;
        if !names.insert(path.to_ascii_lowercase()) {
            return Err(refusal("duplicate or case-colliding file"));
        }
        let mut prefix = String::new();
        for part in path.split('/') {
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(part);
            if let Some(previous) = prefixes.insert(prefix.to_ascii_lowercase(), prefix.clone())
                && previous != prefix
            {
                return Err(refusal("case-colliding directory"));
            }
        }
    }
    let manifests = ["glr-project.toml", "glr-project.json"]
        .iter()
        .filter(|name| names.contains(**name))
        .count();
    if manifests != 1 || !selection.files.iter().any(|path| path.ends_with(".lock")) {
        return Err(refusal(
            "exactly one project manifest and a dependency lock are required",
        ));
    }
    for name in &names {
        if name
            .split('/')
            .scan(String::new(), |prefix, part| {
                if !prefix.is_empty() {
                    prefix.push('/');
                }
                prefix.push_str(part);
                Some(prefix.clone())
            })
            .any(|prefix| prefix != *name && names.contains(&prefix))
        {
            return Err(refusal("file conflicts with a directory"));
        }
    }
    Ok(())
}

fn no_links(path: &Path) -> Result<()> {
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor)?;
        if metadata.is_symlink() {
            return Err(refusal("links are forbidden"));
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            if metadata.file_attributes() & 0x400 != 0 {
                return Err(refusal("reparse points are forbidden"));
            }
        }
    }
    Ok(())
}

fn read_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    no_links(path)?;
    // Reject non-regular and oversized entries before opening them. Opening a
    // directory fails with a platform-specific I/O error on Windows, while the
    // contract promises the same stable refusal on every platform.
    let entry = fs::symlink_metadata(path)?;
    if !entry.is_file() || entry.len() > limit {
        return Err(refusal("file is not regular or exceeds size limit"));
    }
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(refusal("file is not regular or exceeds size limit"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.nlink() != 1 {
            return Err(refusal("hard links are forbidden"));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        if info.nNumberOfLinks != 1 {
            return Err(refusal("hard links are forbidden"));
        }
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(refusal("file exceeds size limit"));
    }
    Ok(bytes)
}

fn identity(manifest: &Manifest) -> Result<String> {
    Ok(digest(&serde_json::to_vec(&(
        &manifest.selection,
        &manifest.tool_version,
        &manifest.entries,
    ))?))
}

type Payload = Vec<(String, Vec<u8>)>;

fn validate_project(selection: &Selection, payload: &Payload) -> Result<()> {
    let (name, bytes) = payload
        .iter()
        .find(|(name, _)| name == "glr-project.toml" || name == "glr-project.json")
        .ok_or_else(|| refusal("missing project manifest"))?;
    let value: Value = if name.ends_with(".json") {
        serde_json::from_slice(bytes)?
    } else {
        let text =
            std::str::from_utf8(bytes).map_err(|_| refusal("project manifest must be UTF-8"))?;
        toml::from_str(text)?
    };
    if value["schema_version"] != "glr.project.v1"
        || value["environment_id"] != selection.environment_id
        || value["protocol_version"] != selection.protocol_version
    {
        return Err(refusal("project manifest identity does not match package"));
    }
    Ok(())
}

fn plan(root: &Path, selection_path: &Path) -> Result<(Manifest, Payload)> {
    no_links(root)?;
    let root = root.canonicalize()?;
    let mut selection: Selection =
        serde_json::from_slice(&read_file(selection_path, 1024 * 1024)?)?;
    validate_selection(&selection)?;
    selection.files.sort();
    let mut entries = Vec::new();
    let mut payload = Vec::new();
    let mut total = 0;
    for path in &selection.files {
        let bytes = read_file(&root.join(path), MAX_FILE)?;
        total += bytes.len() as u64;
        if total > MAX_TOTAL {
            return Err(refusal("expanded size limit exceeded"));
        }
        entries.push(Entry {
            path: path.clone(),
            size_bytes: bytes.len() as u64,
            sha256: digest(&bytes),
        });
        payload.push((path.clone(), bytes));
    }
    let mut manifest = Manifest {
        selection,
        tool_version: env!("CARGO_PKG_VERSION").into(),
        content_sha256: String::new(),
        entries,
    };
    validate_project(&manifest.selection, &payload)?;
    manifest.content_sha256 = identity(&manifest)?;
    Ok((manifest, payload))
}

fn inspect(archive: &Path) -> Result<(Manifest, Payload)> {
    let bytes = read_file(archive, MAX_TOTAL)?;
    let mut zip = ZipArchive::new(std::io::Cursor::new(bytes))?;
    if zip.len() > MAX_FILES + 1 {
        return Err(refusal("archive file count exceeded"));
    }
    let mut payload = std::collections::BTreeMap::new();
    let mut names = BTreeSet::new();
    let mut total = 0;
    for index in 0..zip.len() {
        let mut file = zip.by_index(index)?;
        portable(file.name())?;
        if file.is_dir()
            || file
                .unix_mode()
                .is_some_and(|mode| mode & 0o170000 != 0o100000)
            || file.size() > MAX_FILE
            || !names.insert(file.name().to_ascii_lowercase())
        {
            return Err(refusal(
                "archive has links, collisions, or oversized entries",
            ));
        }
        total += file.size();
        if total > MAX_TOTAL {
            return Err(refusal("expanded size limit exceeded"));
        }
        let name = file.name().to_owned();
        let mut content = Vec::new();
        (&mut file).take(MAX_FILE + 1).read_to_end(&mut content)?;
        if content.len() as u64 != file.size() {
            return Err(refusal("entry size mismatch"));
        }
        payload.insert(name, content);
    }
    let encoded = payload
        .remove(MANIFEST)
        .ok_or_else(|| refusal("missing package manifest"))?;
    if encoded.len() > 1024 * 1024 {
        return Err(refusal("manifest size limit exceeded"));
    }
    let manifest: Manifest = serde_json::from_slice(&encoded)?;
    validate_selection(&manifest.selection)?;
    Version::parse(&manifest.tool_version)?;
    if manifest.content_sha256 != identity(&manifest)? || manifest.entries.len() != payload.len() {
        return Err(refusal("package identity or inventory mismatch"));
    }
    let declared: BTreeSet<_> = manifest.selection.files.iter().collect();
    let indexed: BTreeSet<_> = manifest.entries.iter().map(|entry| &entry.path).collect();
    if declared != indexed || indexed.len() != manifest.entries.len() {
        return Err(refusal("selection and inventory disagree"));
    }
    for entry in &manifest.entries {
        let bytes = payload
            .get(&entry.path)
            .ok_or_else(|| refusal("missing selected file"))?;
        if entry.sha256 != digest(bytes) || entry.size_bytes != bytes.len() as u64 {
            return Err(refusal("file digest or size mismatch"));
        }
    }
    let payload: Payload = payload.into_iter().collect();
    validate_project(&manifest.selection, &payload)?;
    Ok((manifest, payload))
}

fn absolute(path: &Path) -> Result<PathBuf> {
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    })
}

/// One materialized destination, inventoried without following any link.
#[derive(Debug, Default)]
struct Scan {
    files: BTreeMap<String, u64>,
    local_overrides: Vec<String>,
    forbidden: Vec<String>,
}

fn scan_tree(root: &Path, prefix: &str, depth: usize, scan: &mut Scan) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(refusal("materialized tree exceeds the depth limit"));
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let relative = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        no_links(&entry.path())?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.is_dir() {
            if FORBIDDEN_ARTIFACT_DIRS
                .iter()
                .any(|denied| *denied == name.to_ascii_lowercase())
            {
                scan.forbidden.push(relative);
                continue;
            }
            scan_tree(&entry.path(), &relative, depth + 1, scan)?;
        } else if metadata.is_file() {
            if name.to_ascii_lowercase() == RUN_STORE_FILE {
                scan.forbidden.push(relative.clone());
            }
            if relative.split('/').any(is_local_override) {
                scan.local_overrides.push(relative);
                continue;
            }
            if scan.files.len() >= MAX_FILES {
                return Err(refusal("materialized tree exceeds the file limit"));
            }
            scan.files.insert(relative, metadata.len());
        } else {
            return Err(refusal("materialized tree has a non-regular entry"));
        }
    }
    Ok(())
}

/// Declared roles whose executable must already exist on the recipient machine.
///
/// Unavailability is a blocker and a remediation hint. Nothing is fetched.
fn prerequisites(project: &Project) -> Result<Vec<Value>> {
    let roles = [
        ("runtime", Some(&project.runtime)),
        ("trainer", Some(&project.trainer)),
        ("player", Some(&project.player)),
        ("researcher", project.researcher.as_ref()),
        ("planner", project.planner.as_ref()),
        ("evaluator", project.evaluator.as_ref()),
    ];
    let mut results = Vec::new();
    for (name, command) in roles {
        let Some(command) = command else { continue };
        let Some(program) = command.argv.first() else {
            continue;
        };
        let available = executable_available(project, command);
        results.push(json!({
            "name": name,
            "kind": "role",
            "program": program,
            "available": available,
            "blocker": !available,
            "remediation": if available { Value::Null } else { json!(
                format!("make {program:?} available for the {name:?} role, then re-run conformance; GLR never downloads or installs a prerequisite")
            )},
        }));
    }
    if let Some(report) = crate::task::doctor_report(project)? {
        for name in report["unavailable"]
            .as_array()
            .cloned()
            .unwrap_or_default()
        {
            let name = name.as_str().unwrap_or_default().to_string();
            results.push(json!({
                "name": name,
                "kind": "task",
                "program": Value::Null,
                "available": false,
                "blocker": true,
                "remediation": format!(
                    "make the program required by the {name:?} task available, then re-run conformance; GLR never downloads or installs a prerequisite"
                ),
            }));
        }
    }
    Ok(results)
}

fn blocker(kind: &str, detail: String, remediation: &str) -> Value {
    json!({"kind": kind, "detail": detail, "remediation": remediation})
}

/// Offline synthetic conformance check for a package that is already on disk.
///
/// This validates and compares bytes. It never resolves or installs
/// dependencies, never runs a role, hook or trainer, and never touches the
/// network, so its result is a statement about the package and the materialized
/// tree only — never about training.
fn conformance(project: &Path, command: &PackageCommand) -> Result<Value> {
    let PackageCommand::Conformance {
        archive,
        expected_environment,
        expected_contract,
    } = command
    else {
        return Err(refusal("not a conformance command"));
    };
    let archive = absolute(archive)?;
    let (manifest, _) = inspect(&archive)?;
    if expected_environment
        .as_ref()
        .is_some_and(|value| *value != manifest.selection.environment_id)
        || expected_contract
            .as_ref()
            .is_some_and(|value| *value != manifest.selection.contract_sha256)
    {
        return Err(refusal("environment or contract fingerprint mismatch"));
    }
    let manifest_path = find_project(project)?;
    let root = fs::canonicalize(
        manifest_path
            .parent()
            .ok_or_else(|| refusal("missing project root"))?,
    )?;
    let mut scan = Scan::default();
    scan_tree(&root, "", 0, &mut scan)?;

    let mut missing = Vec::new();
    let mut mismatched = Vec::new();
    for entry in &manifest.entries {
        let size_matches = scan.files.get(&entry.path) == Some(&entry.size_bytes);
        let digest_matches = scan
            .files
            .contains_key(&entry.path)
            .then(|| read_file(&root.join(&entry.path), MAX_FILE))
            .transpose()?
            .is_some_and(|bytes| digest(&bytes) == entry.sha256);
        if size_matches && digest_matches {
            continue;
        }
        if scan.files.contains_key(&entry.path) {
            mismatched.push(entry.path.clone());
        } else {
            missing.push(entry.path.clone());
        }
    }
    let declared: BTreeSet<&String> = manifest.entries.iter().map(|entry| &entry.path).collect();
    let unexpected: Vec<String> = scan
        .files
        .keys()
        .filter(|path| !declared.contains(path))
        .cloned()
        .collect();

    let loaded = load_project(&manifest_path);
    let identity_matches = match &loaded {
        Ok(project) => {
            project.environment_id == manifest.selection.environment_id
                && project.protocol_version == manifest.selection.protocol_version
        }
        // A project that cannot be loaded is reported as its own blocker, not
        // additionally as an identity mismatch.
        Err(_) => true,
    };
    let prerequisites = match &loaded {
        Ok(project) => prerequisites(project)?,
        Err(_) => Vec::new(),
    };
    let lock_files: Vec<&String> = manifest
        .selection
        .files
        .iter()
        .filter(|path| path.ends_with(".lock"))
        .collect();

    let mut blockers = Vec::new();
    if !missing.is_empty() {
        blockers.push(blocker(
            "materialization",
            format!(
                "{} declared file(s) are absent from the destination",
                missing.len()
            ),
            "re-import the package into a new, empty directory",
        ));
    }
    if !mismatched.is_empty() {
        blockers.push(blocker(
            "materialization",
            format!(
                "{} declared file(s) differ from the package digests",
                mismatched.len()
            ),
            "re-import the package into a new, empty directory",
        ));
    }
    if !unexpected.is_empty() {
        blockers.push(blocker(
            "materialization",
            format!(
                "{} file(s) are not declared by the package",
                unexpected.len()
            ),
            "remove the undeclared files, or re-import into a new, empty directory",
        ));
    }
    if !scan.forbidden.is_empty() {
        blockers.push(blocker(
            "artifacts",
            format!(
                "{} denied cache, output or run-store path(s) are present",
                scan.forbidden.len()
            ),
            "remove the local run state; a package must materialize without it",
        ));
    }
    if !identity_matches {
        blockers.push(blocker(
            "identity",
            "the materialized project identity does not match the package selection".into(),
            "import the package that matches this project's environment and protocol",
        ));
    }
    if let Err(error) = &loaded {
        blockers.push(blocker(
            "project",
            error.to_string(),
            "supply the missing project file or directory, then re-run conformance",
        ));
    }
    if lock_files.is_empty() {
        blockers.push(blocker(
            "dependencies",
            "the package declares no dependency lock file".into(),
            "export a package that includes the project's dependency lock",
        ));
    }
    for prerequisite in &prerequisites {
        if prerequisite["blocker"].as_bool() == Some(true) {
            let kind = prerequisite["kind"].as_str().unwrap_or("prerequisite");
            let name = prerequisite["name"].as_str().unwrap_or_default();
            blockers.push(blocker(
                "prerequisite",
                format!("{kind} {name:?} has no available program"),
                prerequisite["remediation"]
                    .as_str()
                    .unwrap_or("supply the prerequisite, then re-run conformance"),
            ));
        }
    }

    // `inspect` above already refused an invalid envelope, so reaching this
    // point means the archive itself is valid and completely verified.
    let package_valid = true;
    let materialized =
        missing.is_empty() && mismatched.is_empty() && unexpected.is_empty() && identity_matches;
    let conformant =
        loaded.is_ok() && materialized && scan.forbidden.is_empty() && blockers.is_empty();
    let run_store = scan.forbidden.iter().any(|path| {
        let last = path
            .rsplit('/')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        last == RUN_STORE_FILE || last == ".glr"
    });
    Ok(json!({
        "schema_version": CONFORMANCE_SCHEMA,
        "status": "synthetic-conformance",
        "archive": archive,
        "destination": root,
        "executed": false,
        "offline": true,
        "package": {
            "valid": package_valid,
            "environment_id": manifest.selection.environment_id,
            "protocol_version": manifest.selection.protocol_version,
            "contract_sha256": manifest.selection.contract_sha256,
            "content_sha256": manifest.content_sha256,
            "package_version": manifest.selection.package_version,
            "file_count": manifest.entries.len(),
        },
        "materialization": {
            "status": if materialized { "complete" } else { "incomplete" },
            "missing": missing,
            "mismatched": mismatched,
            "unexpected": unexpected,
        },
        "local_overrides": {
            "packaged": [],
            "present_in_destination": scan.local_overrides,
            "merged": false,
            "ignored_patterns": LOCAL_OVERRIDE_PATTERNS,
        },
        "dependency_setup": {
            "performed": false,
            "status": if lock_files.is_empty() { "missing" } else { "declared" },
            "lock_files": lock_files,
            "remediation": "recreate dependencies from the lock with the project's own setup step; GLR never resolves, downloads or installs them",
        },
        "artifacts": {
            "run_store": run_store,
            "forbidden": scan.forbidden,
        },
        "prerequisites": prerequisites,
        "project": match &loaded {
            Ok(project) => json!({"loaded": true, "manifest": project.manifest_path}),
            Err(error) => json!({"loaded": false, "error": error.to_string()}),
        },
        "blockers": blockers,
        "axes": {
            "package_validity": if package_valid { "valid" } else { "invalid" },
            "materialization": if materialized { "complete" } else { "incomplete" },
            "dependency_setup": if lock_files.is_empty() { "missing" } else { "declared" },
            "synthetic_reproduction": if conformant { "pass" } else { "blocked" },
            "training": "not-evaluated",
            "live_acceptance": "not-evaluated",
            "policy_quality": "not-evaluated",
        },
        "claims": {
            "package_valid": package_valid,
            "materialized": materialized,
            "training_performed": false,
            "training_succeeded": false,
            "training_ready": false,
        },
    }))
}

pub(crate) fn execute(project: &Path, command: &PackageCommand, json: bool) -> Result<i32> {
    match command {
        PackageCommand::Plan { manifest } | PackageCommand::Export { manifest, .. } => {
            let project = find_project(project)?;
            let root = project
                .parent()
                .ok_or_else(|| refusal("missing project root"))?;
            let selection = if manifest.is_absolute() {
                manifest.clone()
            } else {
                root.join(manifest)
            };
            let (manifest, payload) = plan(root, &selection)?;
            if let PackageCommand::Export { output, .. } = command {
                let output = absolute(output)?;
                let parent = output
                    .parent()
                    .ok_or_else(|| refusal("missing output parent"))?;
                no_links(parent)?;
                let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
                {
                    let mut writer = ZipWriter::new(temporary.as_file_mut());
                    let options = SimpleFileOptions::default()
                        .compression_method(zip::CompressionMethod::Stored)
                        .unix_permissions(0o644);
                    writer.start_file(MANIFEST, options)?;
                    writer.write_all(&serde_json::to_vec(&manifest)?)?;
                    for (path, bytes) in payload {
                        writer.start_file(path, options)?;
                        writer.write_all(&bytes)?;
                    }
                    writer.finish()?;
                }
                temporary.as_file().sync_all()?;
                if temporary.as_file().metadata()?.len() > MAX_TOTAL {
                    return Err(refusal("archive size limit exceeded"));
                }
                temporary
                    .persist_noclobber(output)
                    .map_err(|error| Error::Io(error.error))?;
            }
            emit(
                "package",
                &json!({"status": "verified-source-inventory", "manifest": manifest, "executed": false}),
                json,
            )?;
            Ok(0)
        }
        PackageCommand::Inspect { archive } | PackageCommand::Import { archive, .. } => {
            let (manifest, payload) = inspect(&absolute(archive)?)?;
            if let PackageCommand::Import {
                destination,
                expected_environment,
                expected_contract,
                ..
            } = command
            {
                if *expected_environment != manifest.selection.environment_id
                    || *expected_contract != manifest.selection.contract_sha256
                {
                    return Err(refusal("environment or contract fingerprint mismatch"));
                }
                let destination = absolute(destination)?;
                if fs::symlink_metadata(&destination).is_ok() {
                    return Err(refusal("destination already exists"));
                }
                let parent = destination
                    .parent()
                    .ok_or_else(|| refusal("missing destination parent"))?;
                no_links(parent)?;
                let staging = tempfile::tempdir_in(parent)?;
                for (path, bytes) in payload {
                    let target = staging.path().join(path);
                    fs::create_dir_all(
                        target
                            .parent()
                            .ok_or_else(|| refusal("missing file parent"))?,
                    )?;
                    fs::write(target, bytes)?;
                }
                promote(staging.path(), &destination)?;
            }
            emit(
                "package",
                &json!({"status": "verified-source-package", "manifest": manifest, "executed": false, "training_ready": false}),
                json,
            )?;
            Ok(0)
        }
        PackageCommand::Conformance { .. } => {
            let report = conformance(project, command)?;
            let exit_code =
                if report["axes"]["synthetic_reproduction"] == Value::String("pass".into()) {
                    0
                } else {
                    BLOCKED_EXIT_CODE
                };
            emit("package", &report, json)?;
            Ok(exit_code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(root: &Path) -> PathBuf {
        fs::write(root.join("glr-project.json"), br#"{"schema_version":"glr.project.v1","environment_id":"synthetic.package","protocol_version":"1.0"}"#).unwrap();
        fs::write(root.join("uv.lock"), b"version = 1\n").unwrap();
        fs::write(
            root.join("train.py"),
            b"raise RuntimeError('must never execute')\n",
        )
        .unwrap();
        let path = root.join("selection.json");
        fs::write(
            &path,
            serde_json::to_vec(&Selection {
                schema_version: SCHEMA.into(),
                package_version: "1.0.0".into(),
                required_glr: ">=0.18.0, <1.0.0".into(),
                environment_id: "synthetic.package".into(),
                protocol_version: "1.0".into(),
                contract_sha256: "a".repeat(64),
                source_revision: "synthetic-fixture".into(),
                redistribution_license: "MIT".into(),
                files: vec![
                    "glr-project.json".into(),
                    "train.py".into(),
                    "uv.lock".into(),
                ],
            })
            .unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn source_package_is_deterministic_offline_and_never_executes() {
        let root = tempfile::tempdir().unwrap();
        let selection = fixture(root.path());
        let first = root.path().join("first.zip");
        let second = root.path().join("second.zip");
        for output in [&first, &second] {
            execute(
                root.path(),
                &PackageCommand::Export {
                    manifest: selection.clone(),
                    output: output.clone(),
                },
                false,
            )
            .unwrap();
        }
        assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
        let (manifest, _) = inspect(&first).unwrap();
        let destination = root.path().join("imported");
        let command = PackageCommand::Import {
            archive: first,
            destination: destination.clone(),
            expected_environment: "synthetic.package".into(),
            expected_contract: manifest.selection.contract_sha256,
        };
        assert_eq!(execute(root.path(), &command, false).unwrap(), 0);
        assert_eq!(
            fs::read(destination.join("train.py")).unwrap(),
            fs::read(root.path().join("train.py")).unwrap()
        );
        assert!(!destination.join("selection.json").exists());
        assert!(execute(root.path(), &command, false).is_err());
    }

    /// A package fixture whose materialized destination is a loadable project.
    ///
    /// `trainer` selects the trainer program so a test can make it unavailable
    /// without touching the network or the filesystem layout.
    fn conformance_fixture(root: &Path, trainer: &str) -> PathBuf {
        let executable = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        fs::write(
            root.join("glr-project.json"),
            serde_json::to_vec(&json!({
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
        fs::create_dir(root.join("bridge")).unwrap();
        fs::write(root.join("bridge/README.md"), b"bridge\n").unwrap();
        fs::write(root.join("uv.lock"), b"version = 1\n").unwrap();
        fs::write(
            root.join("train.py"),
            b"raise RuntimeError('must never execute')\n",
        )
        .unwrap();
        let path = root.join("selection.json");
        fs::write(
            &path,
            serde_json::to_vec(&Selection {
                schema_version: SCHEMA.into(),
                package_version: "1.0.0".into(),
                required_glr: ">=0.18.0, <1.0.0".into(),
                environment_id: "synthetic.package".into(),
                protocol_version: "1.0".into(),
                contract_sha256: "a".repeat(64),
                source_revision: "synthetic-conformance".into(),
                redistribution_license: "MIT".into(),
                files: vec![
                    "bridge/README.md".into(),
                    "glr-project.json".into(),
                    "train.py".into(),
                    "uv.lock".into(),
                ],
            })
            .unwrap(),
        )
        .unwrap();
        path
    }

    /// Export, then import into a fresh directory, and return the destination.
    ///
    /// The archive is written inside the recipient's temporary directory so it
    /// outlives the exported source tree.
    fn round_trip(trainer: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
        let recipient = tempfile::tempdir().unwrap();
        let archive = recipient.path().join("source.zip");
        let source = tempfile::tempdir().unwrap();
        let selection = conformance_fixture(source.path(), trainer);
        assert_eq!(
            execute(
                source.path(),
                &PackageCommand::Export {
                    manifest: selection,
                    output: archive.clone(),
                },
                false,
            )
            .unwrap(),
            0
        );
        let destination = recipient.path().join("imported");
        assert_eq!(
            execute(
                recipient.path(),
                &PackageCommand::Import {
                    archive: archive.clone(),
                    destination: destination.clone(),
                    expected_environment: "synthetic.package".into(),
                    expected_contract: "a".repeat(64),
                },
                false,
            )
            .unwrap(),
            0
        );
        (recipient, destination, archive)
    }

    fn conformance_command(archive: &Path) -> PackageCommand {
        PackageCommand::Conformance {
            archive: archive.to_owned(),
            expected_environment: None,
            expected_contract: None,
        }
    }

    #[test]
    fn conformance_passes_for_a_cleanly_materialized_package() {
        let executable = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (_recipient, destination, archive) = round_trip(&executable);
        let report = conformance(&destination, &conformance_command(&archive)).unwrap();
        assert_eq!(report["axes"]["package_validity"], "valid");
        assert_eq!(report["axes"]["materialization"], "complete");
        assert_eq!(report["axes"]["dependency_setup"], "declared");
        assert_eq!(report["axes"]["synthetic_reproduction"], "pass");
        assert_eq!(report["axes"]["training"], "not-evaluated");
        assert_eq!(report["artifacts"]["run_store"], false);
        assert!(
            report["materialization"]["unexpected"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(report["dependency_setup"]["performed"], false);
        assert_eq!(report["dependency_setup"]["lock_files"], json!(["uv.lock"]));
        assert_eq!(report["claims"]["training_performed"], false);
        assert_eq!(report["claims"]["training_succeeded"], false);
        assert_eq!(report["claims"]["training_ready"], false);
        assert_eq!(report["executed"], false);
    }

    #[test]
    fn conformance_resolves_a_nested_working_directory() {
        let executable = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (_recipient, destination, archive) = round_trip(&executable);
        let nested = destination.join("bridge");
        let report = conformance(&nested, &conformance_command(&archive)).unwrap();
        assert_eq!(
            report["destination"],
            json!(fs::canonicalize(&destination).unwrap())
        );
        assert_eq!(report["axes"]["synthetic_reproduction"], "pass");
    }

    #[test]
    fn conformance_ignores_recipient_local_overrides_and_never_merges_them() {
        let executable = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (_recipient, destination, archive) = round_trip(&executable);
        fs::write(destination.join("glr-project.local.json"), b"{}").unwrap();
        fs::create_dir(destination.join("bridge.local.d")).unwrap();
        fs::write(destination.join("bridge.local.d/override.json"), b"{}").unwrap();
        let report = conformance(&destination, &conformance_command(&archive)).unwrap();
        assert_eq!(
            report["local_overrides"]["present_in_destination"],
            json!(["bridge.local.d/override.json", "glr-project.local.json"])
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
        assert_eq!(report["axes"]["synthetic_reproduction"], "pass");
        // A local override is never acceptable inside the package itself.
        for path in ["glr-project.local.json", "secrets.local.toml", "a.local"] {
            assert!(source_path(path).is_err(), "{path}");
        }
    }

    #[test]
    fn conformance_blocks_a_missing_prerequisite_without_claiming_training() {
        let (_recipient, destination, archive) = round_trip("glr-missing-synthetic-trainer");
        let command = conformance_command(&archive);
        let report = conformance(&destination, &command).unwrap();
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
                .any(|blocker| blocker["kind"] == "prerequisite")
        );
        assert!(blockers.iter().any(|blocker| {
            blocker["remediation"]
                .as_str()
                .unwrap()
                .contains("never downloads or installs")
        }));
        assert_eq!(execute(&destination, &command, false).unwrap(), 4);
    }

    #[test]
    fn conformance_flags_a_run_store_undeclared_files_and_tampered_bytes() {
        let executable = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (_recipient, destination, archive) = round_trip(&executable);
        fs::create_dir(destination.join(".glr")).unwrap();
        fs::write(destination.join(".glr/runs.sqlite3"), b"store").unwrap();
        fs::write(destination.join("notes.txt"), b"undeclared").unwrap();
        fs::write(destination.join("train.py"), b"tampered").unwrap();
        fs::remove_file(destination.join("uv.lock")).unwrap();
        let report = conformance(&destination, &conformance_command(&archive)).unwrap();
        assert_eq!(report["artifacts"]["run_store"], true);
        assert_eq!(
            report["materialization"]["unexpected"],
            json!(["notes.txt"])
        );
        assert_eq!(report["materialization"]["mismatched"], json!(["train.py"]));
        assert_eq!(report["materialization"]["missing"], json!(["uv.lock"]));
        assert_eq!(report["materialization"]["status"], "incomplete");
        // The archive itself is untouched by a dirty destination.
        assert_eq!(report["package"]["valid"], true);
        assert_eq!(report["axes"]["package_validity"], "valid");
        assert_eq!(report["axes"]["materialization"], "incomplete");
        assert_eq!(report["axes"]["synthetic_reproduction"], "blocked");
    }

    #[test]
    fn conformance_refuses_a_reviewed_expectation_mismatch() {
        let executable = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (_recipient, destination, archive) = round_trip(&executable);
        for (environment, contract) in [
            (Some("other.environment".into()), None),
            (None, Some("b".repeat(64))),
        ] {
            let report = conformance(
                &destination,
                &PackageCommand::Conformance {
                    archive: archive.clone(),
                    expected_environment: environment,
                    expected_contract: contract,
                },
            );
            assert!(report.is_err());
        }
    }

    #[test]
    fn selection_rejects_unsafe_paths_and_incompatible_contracts() {
        let root = tempfile::tempdir().unwrap();
        let path = fixture(root.path());
        let source: Selection = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        for path in [
            "../outside.py",
            "C:/private.py",
            "a\\b.py",
            "CON.py",
            "secret.local.toml",
            ".env",
            "recordings/a.mp4",
            "x/../y.py",
            "foo./x.py",
        ] {
            let mut selection = source.clone();
            selection.files.push(path.into());
            assert!(validate_selection(&selection).is_err(), "{path}");
        }
        let mut selection = source.clone();
        selection
            .files
            .extend(["Case/a.py".into(), "case/b.py".into()]);
        assert!(validate_selection(&selection).is_err());
        selection = source.clone();
        selection.required_glr = ">=999.0.0".into();
        assert!(validate_selection(&selection).is_err());
        selection = source;
        selection.schema_version = "future".into();
        assert!(validate_selection(&selection).is_err());
    }

    #[test]
    fn rejects_hardlinks_and_atomic_promotion_preserves_existing_destination() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("source.py"), b"pass").unwrap();
        fs::hard_link(root.path().join("source.py"), root.path().join("alias.py")).unwrap();
        assert!(read_file(&root.path().join("source.py"), MAX_FILE).is_err());
        let source = root.path().join("staged");
        let destination = root.path().join("existing");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&destination).unwrap();
        fs::write(destination.join("sentinel"), b"keep").unwrap();
        assert!(promote(&source, &destination).is_err());
        assert_eq!(fs::read(destination.join("sentinel")).unwrap(), b"keep");
        fs::remove_file(destination.join("sentinel")).unwrap();
        assert!(promote(&source, &destination).is_err());
        assert!(destination.is_dir());
    }

    #[test]
    fn archive_rejects_unexpected_files_corruption_and_identity_mismatch() {
        let root = tempfile::tempdir().unwrap();
        let selection = fixture(root.path());
        let (manifest, payload) = plan(root.path(), &selection).unwrap();
        for bad_path in ["../escape.py", "unexpected.py", "TRAIN.py"] {
            let path = root.path().join("bad.zip");
            let mut writer = ZipWriter::new(File::create(&path).unwrap());
            let options = SimpleFileOptions::default().unix_permissions(0o644);
            writer.start_file(MANIFEST, options).unwrap();
            writer
                .write_all(&serde_json::to_vec(&manifest).unwrap())
                .unwrap();
            for (name, bytes) in &payload {
                writer.start_file(name, options).unwrap();
                writer.write_all(bytes).unwrap();
            }
            writer.start_file(bad_path, options).unwrap();
            writer.write_all(b"bad").unwrap();
            writer.finish().unwrap();
            assert!(inspect(&path).is_err());
        }
        let archive = root.path().join("valid.zip");
        execute(
            root.path(),
            &PackageCommand::Export {
                manifest: selection,
                output: archive.clone(),
            },
            false,
        )
        .unwrap();
        let destination = root.path().join("refused");
        assert!(
            execute(
                root.path(),
                &PackageCommand::Import {
                    archive,
                    destination: destination.clone(),
                    expected_environment: "wrong".into(),
                    expected_contract: "a".repeat(64)
                },
                false
            )
            .is_err()
        );
        assert!(!destination.exists());
    }

    #[test]
    fn archive_rejects_symlinks_oversized_entries_and_changed_payload() {
        let root = tempfile::tempdir().unwrap();
        let selection = fixture(root.path());
        let (manifest, payload) = plan(root.path(), &selection).unwrap();
        for mode in ["symlink", "oversized", "corrupt"] {
            let archive = root.path().join(format!("{mode}.zip"));
            let mut writer = ZipWriter::new(File::create(&archive).unwrap());
            let options = SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(0o644);
            writer.start_file(MANIFEST, options).unwrap();
            writer
                .write_all(&serde_json::to_vec(&manifest).unwrap())
                .unwrap();
            for (name, bytes) in &payload {
                if name == "train.py" && mode == "symlink" {
                    writer.add_symlink(name, "uv.lock", options).unwrap();
                } else {
                    writer.start_file(name, options).unwrap();
                    if name == "train.py" && mode == "oversized" {
                        writer.write_all(&vec![0; MAX_FILE as usize + 1]).unwrap();
                    } else if name == "train.py" && mode == "corrupt" {
                        writer.write_all(b"changed").unwrap();
                    } else {
                        writer.write_all(bytes).unwrap();
                    }
                }
            }
            writer.finish().unwrap();
            assert!(inspect(&archive).is_err(), "{mode}");
        }
    }
}
