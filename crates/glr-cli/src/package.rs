//! Offline, explicit source-only packages. No role, installer, or hook execution.
//!
//! One envelope carries a project: the `glr.source-package.v1` source-only
//! profile (ADR-0027) and the `glr.training-package.v1` group-scoped profile
//! (ADR-0041), which adds optional `model`, `dataset`, `knowledge` and `report`
//! groups behind per-group admission predicates. Group policy lives in
//! [`crate::package_groups`]; this module owns the envelope, the archive and the
//! streaming limits.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use zip::{ZipArchive, ZipWriter, write::SimpleFileOptions};

use crate::args::PackageCommand;
use crate::commands::emit;
use crate::error::{Error, Result};
use crate::filesystem::promote;
use crate::package_groups::{
    Audit, AuditEntry, Content, DatasetAllowlist, Group, GroupAudit, GroupFile, GroupSelection,
    INSPECTION_LIMIT, Identity, MAX_EXPANSION_RATIO, MAX_PACKAGE_BYTES, MAX_PACKAGE_FILES,
    RedistributionAuthorization, admit, denied_component, group_of, validate_authorization,
    validate_dataset_allowlist, verify,
};
use crate::process::executable_available;
use crate::project::{Project, find_project, load_project};

const SCHEMA: &str = "glr.source-package.v1";
/// Group-scoped envelope (ADR-0041 D1).
const TRAINING_SCHEMA: &str = "glr.training-package.v1";
const CONFORMANCE_SCHEMA: &str = "glr.package-conformance.v1";
/// Package-wide file ceiling. It equals the archive member gate, so a package
/// can never contain more files than an archive may carry.
const MAX_FILES: usize = MAX_PACKAGE_FILES;
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

/// Size of one streamed chunk while hashing or materializing: a multi-GiB group
/// is never buffered whole.
const CHUNK: usize = 1024 * 1024;

pub(crate) fn is_local_override(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains(".local.") || lower.ends_with(".local")
}

/// True when any path component is a recipient-local override form.
///
/// Both gates must call this: whatever export refuses is exactly what the
/// conformance scan ignores. Keep it as the single definition.
fn is_local_override_path(raw: &str) -> bool {
    raw.split('/').any(is_local_override)
}

/// Which profile a selection declares.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Schema {
    /// `glr.source-package.v1`: one flat, source-only file list (ADR-0027).
    Source,
    /// `glr.training-package.v1`: declared entry groups (ADR-0041 D1).
    Training,
}

/// `glr.source-package.v1` selection: a flat source file list.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceSelection {
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

/// `glr.training-package.v1` selection: declared groups, each with its own
/// files, roles, and admission records.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingSelection {
    package_version: String,
    required_glr: String,
    environment_id: String,
    protocol_version: String,
    contract_sha256: String,
    source_revision: String,
    redistribution_license: String,
    /// The declared groups. A path whose group is not declared is refused.
    entry_groups: Vec<Group>,
    /// One entry per declared group, each carrying its files and policy.
    groups: BTreeMap<Group, GroupSelection>,
    /// Required by — and only meaningful for — a `dataset` export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dataset_allowlist: Option<DatasetAllowlist>,
    /// Required by — and only meaningful for — a `dataset` export.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    redistribution_authorization: Option<RedistributionAuthorization>,
}

/// One selection, in one of the two profiles.
///
/// Internally tagged so both keep `schema_version` at the top level and the
/// source-only wire shape is byte-for-byte what ADR-0027 shipped.
///
/// The variants differ in size because a group-scoped selection carries its
/// per-group records; both are small, stack-allocated, and parsed at most once
/// per command, so boxing would only add indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "schema_version")]
enum Selection {
    #[serde(rename = "glr.source-package.v1")]
    Source(SourceSelection),
    #[serde(rename = "glr.training-package.v1")]
    Training(TrainingSelection),
}

impl Selection {
    fn schema(&self) -> Schema {
        match self {
            Self::Source(_) => Schema::Source,
            Self::Training(_) => Schema::Training,
        }
    }

    fn package_version(&self) -> &str {
        match self {
            Self::Source(selection) => &selection.package_version,
            Self::Training(selection) => &selection.package_version,
        }
    }

    fn required_glr(&self) -> &str {
        match self {
            Self::Source(selection) => &selection.required_glr,
            Self::Training(selection) => &selection.required_glr,
        }
    }

    fn environment_id(&self) -> &str {
        match self {
            Self::Source(selection) => &selection.environment_id,
            Self::Training(selection) => &selection.environment_id,
        }
    }

    fn protocol_version(&self) -> &str {
        match self {
            Self::Source(selection) => &selection.protocol_version,
            Self::Training(selection) => &selection.protocol_version,
        }
    }

    fn contract_sha256(&self) -> &str {
        match self {
            Self::Source(selection) => &selection.contract_sha256,
            Self::Training(selection) => &selection.contract_sha256,
        }
    }

    fn identity(&self) -> Identity<'_> {
        Identity {
            environment_id: self.environment_id(),
            protocol_version: self.protocol_version(),
        }
    }
}

/// One inventory row.
///
/// A source-only entry carries `path`, `size_bytes` and `sha256` only, so the
/// ADR-0027 wire shape is unchanged. A group-scoped entry adds its group, its
/// role, and the compressed size a compressed group must declare (ADR-0041 D5).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    path: String,
    size_bytes: u64,
    sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    group: Option<Group>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compression: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compressed_size_bytes: Option<u64>,
}

/// The identity projection of one entry.
///
/// `compressed_size_bytes` is transport metadata: it is verified against the
/// archive it travels in, so it stays out of the content identity. That keeps a
/// dry run and the export it previews on the same identifier.
#[derive(Debug, Serialize)]
struct IdentityEntry<'a> {
    path: &'a str,
    size_bytes: u64,
    sha256: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    group: Option<Group>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compression: Option<&'a str>,
}

impl Entry {
    fn identity(&self) -> IdentityEntry<'_> {
        IdentityEntry {
            path: &self.path,
            size_bytes: self.size_bytes,
            sha256: &self.sha256,
            group: self.group,
            role: self.role.as_deref(),
            compression: self.compression.as_deref(),
        }
    }
}

/// One file as the envelope carries it: path, owning group, declared role.
#[derive(Debug, Clone)]
struct PlannedFile {
    path: String,
    group: Group,
    role: String,
}

impl PlannedFile {
    /// The wire form of this file, for the group verifiers.
    fn as_group_file(&self) -> GroupFile {
        GroupFile {
            path: self.path.clone(),
            role: self.role.clone(),
        }
    }
}

/// Canonical, profile-independent view of what a package carries.
///
/// The wire types are data transfer objects; everything downstream of
/// `resolve()` works on this.
#[derive(Debug, Clone)]
struct Plan {
    schema: Schema,
    /// Every file, ordered by path.
    files: Vec<PlannedFile>,
    /// Files per group, in the same order.
    groups: BTreeMap<Group, Vec<PlannedFile>>,
    /// Declared freshness budget per group. Only `knowledge` declares one.
    freshness: BTreeMap<Group, u32>,
    dataset_allowlist: Option<DatasetAllowlist>,
    authorization: Option<RedistributionAuthorization>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    selection: Selection,
    tool_version: String,
    content_sha256: String,
    entries: Vec<Entry>,
}

fn refusal(message: impl std::fmt::Display) -> Error {
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

/// The role a source-group path has, derived the same way for every package.
fn source_role(path: &str) -> &'static str {
    let file = path.rsplit('/').next().unwrap_or_default();
    if file == "glr-project.toml" || file == "glr-project.json" {
        "project-manifest"
    } else if file.ends_with(".lock") {
        "dependency-lock"
    } else {
        "source-file"
    }
}

fn source_path(raw: &str) -> Result<()> {
    portable(raw)?;
    let lower = raw.to_ascii_lowercase();
    if lower
        .split('/')
        .any(|part| part.starts_with('.') || denied_component(part))
        || is_local_override_path(&lower)
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

/// Validates the shared identity fields every profile carries.
impl Selection {
    /// Parses one selection file, refusing an unknown profile fail-closed.
    ///
    /// The tag is read before the variant is built so an unknown
    /// `schema_version` is a contract refusal with the same stable category as
    /// every other gate, not a parser error.
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let value: Value = serde_json::from_slice(bytes)?;
        match value.get("schema_version").and_then(Value::as_str) {
            Some(SCHEMA) | Some(TRAINING_SCHEMA) => Ok(serde_json::from_value(value)?),
            _ => Err(refusal("unsupported schema or file count")),
        }
    }
}

fn validate_common(selection: &Selection) -> Result<()> {
    Version::parse(selection.package_version())?;
    let required = VersionReq::parse(selection.required_glr())?;
    if !required.matches(&Version::parse(env!("CARGO_PKG_VERSION"))?) {
        return Err(refusal("incompatible GLR version"));
    }
    if selection.contract_sha256().len() != 64
        || !selection
            .contract_sha256()
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        return Err(refusal("contract fingerprint must be a lowercase SHA-256"));
    }
    let provenance = match selection {
        Selection::Source(selection) => [
            &selection.environment_id,
            &selection.protocol_version,
            &selection.source_revision,
            &selection.redistribution_license,
        ],
        Selection::Training(selection) => [
            &selection.environment_id,
            &selection.protocol_version,
            &selection.source_revision,
            &selection.redistribution_license,
        ],
    };
    for value in provenance {
        if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
            return Err(refusal("invalid provenance or environment identity"));
        }
    }
    Ok(())
}

/// Refuses duplicate, case-colliding, and file-versus-directory paths.
fn validate_paths(files: &[PlannedFile]) -> Result<()> {
    let mut names = BTreeSet::new();
    let mut prefixes = std::collections::BTreeMap::new();
    for file in files {
        if !names.insert(file.path.to_ascii_lowercase()) {
            return Err(refusal("duplicate or case-colliding file"));
        }
        let mut prefix = String::new();
        for part in file.path.split('/') {
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

/// The ADR-0027 obligation a `source` group always carries: exactly one project
/// manifest and at least one dependency lock.
fn validate_source_group(files: &[GroupFile]) -> Result<()> {
    let names: BTreeSet<&str> = files.iter().map(|file| file.path.as_str()).collect();
    let manifests = ["glr-project.toml", "glr-project.json"]
        .iter()
        .filter(|name| names.contains(**name))
        .count();
    if manifests != 1 || !names.iter().any(|path| path.ends_with(".lock")) {
        return Err(refusal(
            "exactly one project manifest and a dependency lock are required",
        ));
    }
    Ok(())
}

/// Resolves a wire selection into the canonical plan, refusing anything the
/// profile, the group vocabulary, or a group's admission predicate rejects.
fn resolve(selection: &Selection) -> Result<Plan> {
    validate_common(selection)?;
    let mut files: Vec<PlannedFile> = match selection {
        Selection::Source(selection) => {
            if selection.files.is_empty() || selection.files.len() > MAX_FILES {
                return Err(refusal("unsupported schema or file count"));
            }
            for path in &selection.files {
                source_path(path)?;
            }
            selection
                .files
                .iter()
                .map(|path| PlannedFile {
                    path: path.clone(),
                    group: Group::Source,
                    role: source_role(path).into(),
                })
                .collect()
        }
        Selection::Training(selection) => resolve_groups(selection)?,
    };
    files.sort_by(|left, right| left.path.cmp(&right.path));
    validate_paths(&files)?;
    // Every profile carries a project: the manifest and lock bind the
    // environment and protocol the package is valid for (ADR-0027).
    validate_source_group(
        &files
            .iter()
            .filter(|file| file.group == Group::Source)
            .map(PlannedFile::as_group_file)
            .collect::<Vec<_>>(),
    )?;
    let mut groups: BTreeMap<Group, Vec<PlannedFile>> = BTreeMap::new();
    for file in &files {
        groups.entry(file.group).or_default().push(file.clone());
    }
    let mut freshness: BTreeMap<Group, u32> = BTreeMap::new();
    if let Selection::Training(selection) = selection {
        for (group, group_selection) in &selection.groups {
            if let Some(days) = group_selection.max_age_days {
                freshness.insert(*group, days);
            }
        }
    }
    Ok(Plan {
        schema: selection.schema(),
        files,
        groups,
        freshness,
        dataset_allowlist: match selection {
            Selection::Training(selection) => selection.dataset_allowlist.clone(),
            Selection::Source(_) => None,
        },
        authorization: match selection {
            Selection::Training(selection) => selection.redistribution_authorization.clone(),
            Selection::Source(_) => None,
        },
    })
}

/// Resolves the group-scoped profile: declared groups, roles, per-group limits,
/// and the authorization a `dataset` export needs.
fn resolve_groups(selection: &TrainingSelection) -> Result<Vec<PlannedFile>> {
    if selection.entry_groups.is_empty()
        || selection.entry_groups.iter().collect::<BTreeSet<_>>().len()
            != selection.entry_groups.len()
    {
        return Err(refusal("entry_groups must be a non-empty, unique set"));
    }
    let declared: BTreeSet<Group> = selection.entry_groups.iter().copied().collect();
    if !declared.contains(&Group::Source) {
        return Err(refusal(
            "a group-scoped package declares the source group: it carries the project manifest and lock that bind environment and protocol identity",
        ));
    }
    let mut files: Vec<PlannedFile> = Vec::new();
    for (group, group_selection) in &selection.groups {
        if !declared.contains(group) {
            return Err(refusal(format!(
                "{} files are present but the group is not declared",
                group.as_str()
            )));
        }
        if group_selection.files.is_empty() {
            return Err(refusal(format!(
                "the {} group declares no files",
                group.as_str()
            )));
        }
        if group_selection.files.len() > group.policy().max_files {
            return Err(refusal(format!(
                "the {} group declares {} files, over its {} limit",
                group.as_str(),
                group_selection.files.len(),
                group.policy().max_files
            )));
        }
        if group_selection.max_age_days.is_some() && *group != Group::Knowledge {
            return Err(refusal(format!(
                "max_age_days only applies to the knowledge group, not {}",
                group.as_str()
            )));
        }
        for file in &group_selection.files {
            portable(&file.path)?;
            if file.path == MANIFEST {
                return Err(refusal("the package manifest is generated, never selected"));
            }
            if *group != group_of(&file.path, &declared)? {
                return Err(refusal(format!(
                    "{} does not belong to the {} group",
                    file.path,
                    group.as_str()
                )));
            }
            admit(*group, &file.path)?;
            if !group.valid_role(&file.role) {
                return Err(refusal(format!(
                    "{} declares role {:?}, which is not a {} role",
                    file.path,
                    file.role,
                    group.as_str()
                )));
            }
            if *group == Group::Source {
                let derived = source_role(&file.path);
                if derived != file.role {
                    return Err(refusal(format!(
                        "{} declares role {:?} but its path is a {derived}",
                        file.path, file.role
                    )));
                }
            }
            files.push(PlannedFile {
                path: file.path.clone(),
                group: *group,
                role: file.role.clone(),
            });
        }
    }
    if files.len() > MAX_FILES {
        return Err(refusal("unsupported schema or file count"));
    }
    let source: Vec<GroupFile> = selection
        .groups
        .get(&Group::Source)
        .map(|group| group.files.clone())
        .unwrap_or_default();
    validate_source_group(&source)?;
    if declared.contains(&Group::Dataset) {
        let dataset = selection
            .groups
            .get(&Group::Dataset)
            .map(|group| group.files.as_slice())
            .unwrap_or_default();
        let allowlist = selection.dataset_allowlist.as_ref().ok_or_else(|| {
            refusal("a dataset export needs a separately reviewed dataset allowlist")
        })?;
        validate_dataset_allowlist(allowlist, dataset)?;
        let authorization = selection
            .redistribution_authorization
            .as_ref()
            .ok_or_else(|| {
                refusal("a dataset export needs a recorded redistribution authorization")
            })?;
        validate_authorization(authorization)?;
    }
    Ok(files)
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

pub(crate) fn read_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
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

/// The expanded size and digest of a file, streamed in bounded chunks.
///
/// A 1 GiB weight is hashed without ever being buffered whole.
fn hash_file(path: &Path, limit: u64) -> Result<(u64, String)> {
    let file = open_bounded(path, limit)?;
    hash_stream(file.take(limit + 1), limit)
}

/// The compressed size of every written member, read back from the archive.
fn compressed_sizes(archive: &Path) -> Result<BTreeMap<String, u64>> {
    let mut zip = ZipArchive::new(BufReader::new(File::open(archive)?))?;
    let mut sizes = BTreeMap::new();
    for index in 0..zip.len() {
        let file = zip.by_index(index)?;
        sizes.insert(file.name().to_owned(), file.compressed_size());
    }
    Ok(sizes)
}

/// Streams one source file into the archive, refusing anything past `limit`.
fn copy_bounded(path: &Path, writer: &mut impl Write, limit: u64) -> Result<u64> {
    let mut file = open_bounded(path, limit)?;
    let mut buffer = vec![0_u8; CHUNK];
    let mut total = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        if total > limit {
            return Err(refusal("file exceeds size limit"));
        }
        writer.write_all(&buffer[..count])?;
    }
    Ok(total)
}

/// Opens a regular, unlinked, non-oversized file. Shared by the streaming and
/// the buffered readers so both apply the identical gate.
fn open_bounded(path: &Path, limit: u64) -> Result<File> {
    read_file(path, limit)?;
    File::open(path).map_err(Error::from)
}

/// Hashes a stream while counting its bytes, refusing anything past `limit`.
fn hash_stream(mut reader: impl Read, limit: u64) -> Result<(u64, String)> {
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; CHUNK];
    let mut total = 0_u64;
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        if total > limit {
            return Err(refusal("file exceeds size limit"));
        }
        digest.update(&buffer[..count]);
    }
    Ok((total, format!("{:x}", digest.finalize())))
}

fn identity(manifest: &Manifest) -> Result<String> {
    let entries: Vec<IdentityEntry> = manifest.entries.iter().map(Entry::identity).collect();
    Ok(digest(&serde_json::to_vec(&(
        &manifest.selection,
        &manifest.tool_version,
        &entries,
    ))?))
}

fn validate_project(selection: &Selection, content: &Content) -> Result<()> {
    let path = ["glr-project.json", "glr-project.toml"]
        .into_iter()
        .find(|name| content.declared(name).is_ok())
        .ok_or_else(|| refusal("missing project manifest"))?;
    let bytes = content.bytes(path, 1024 * 1024)?;
    let value: Value = if path.ends_with(".json") {
        serde_json::from_slice(&bytes)?
    } else {
        let text =
            std::str::from_utf8(&bytes).map_err(|_| refusal("project manifest must be UTF-8"))?;
        toml::from_str(text)?
    };
    if value["schema_version"] != "glr.project.v1"
        || value["environment_id"] != selection.environment_id()
        || value["protocol_version"] != selection.protocol_version()
    {
        return Err(refusal("project manifest identity does not match package"));
    }
    Ok(())
}

/// Bytes a group verifier needs, keyed by package path.
///
/// Capture is role-driven and bounded: only declared manifests and snapshots
/// are materialized, and only under `INSPECTION_LIMIT`.
fn capture_roles(file: &PlannedFile) -> bool {
    matches!(
        file.role.as_str(),
        "project-manifest" | "model-manifest" | "knowledge-snapshot" | "dataset-manifest"
    )
}

/// Per-group byte and file accounting while entries are planned or checked.
#[derive(Debug, Default)]
struct Tally {
    group_bytes: BTreeMap<Group, u64>,
    total: u64,
}

impl Tally {
    /// Adds one expanded file, refusing past the per-group cap and the package
    /// ceiling. A ceiling that is lower than the sum of the group caps is the
    /// point of the rule: the package, not one group, is what a recipient pays for.
    fn add(&mut self, group: Group, size: u64) -> Result<()> {
        let policy = group.policy();
        let group_bytes = self.group_bytes.entry(group).or_insert(0);
        *group_bytes += size;
        if *group_bytes > policy.max_group_bytes {
            return Err(refusal("expanded size limit exceeded"));
        }
        self.total += size;
        if self.total > MAX_PACKAGE_BYTES {
            return Err(refusal("package byte ceiling exceeded"));
        }
        Ok(())
    }
}

/// Plans an export: the manifest the archive will carry and its audit receipt.
///
/// Files are hashed by streaming them, so a 1 GiB weight is measured without
/// being buffered. Nothing is executed and nothing is compressed yet: a dry run
/// and the export it previews share one content identity.
fn plan(root: &Path, selection_path: &Path) -> Result<(Manifest, Audit)> {
    no_links(root)?;
    let root = root.canonicalize()?;
    let selection = Selection::from_bytes(&read_file(selection_path, 1024 * 1024)?)?;
    let plan = resolve(&selection)?;
    let mut entries = Vec::new();
    let mut declared: BTreeMap<String, (u64, String)> = BTreeMap::new();
    let mut tally = Tally::default();
    for file in &plan.files {
        let policy = file.group.policy();
        let (size, sha256) = hash_file(&root.join(&file.path), policy.max_file_bytes)?;
        tally.add(file.group, size)?;
        declared.insert(file.path.clone(), (size, sha256.clone()));
        entries.push(Entry {
            path: file.path.clone(),
            size_bytes: size,
            sha256,
            group: group_field(plan.schema, file.group),
            role: role_field(plan.schema, &file.role),
            compression: compression_field(plan.schema, file.group),
            compressed_size_bytes: None,
        });
    }
    let manifest = Manifest {
        selection,
        tool_version: env!("CARGO_PKG_VERSION").into(),
        content_sha256: String::new(),
        entries,
    };
    let audit = audit(&manifest, &plan, Content::for_tree(&root, &declared))?;
    let mut manifest = manifest;
    manifest.content_sha256 = identity(&manifest)?;
    Ok((manifest, audit))
}

/// Group and role are wire fields of the group-scoped profile only: a
/// source-only entry keeps the ADR-0027 shape byte for byte.
fn group_field(schema: Schema, group: Group) -> Option<Group> {
    match schema {
        Schema::Source => None,
        Schema::Training => Some(group),
    }
}

fn role_field(schema: Schema, role: &str) -> Option<String> {
    match schema {
        Schema::Source => None,
        Schema::Training => Some(role.into()),
    }
}

/// `source` stays uncompressed; binary groups are deflated (ADR-0041 D5).
fn compression_field(schema: Schema, group: Group) -> Option<String> {
    match schema {
        Schema::Source => None,
        Schema::Training if group.policy().compressed => Some("deflated".into()),
        Schema::Training => Some("stored".into()),
    }
}

/// Runs every group's admission predicate and builds the aggregate audit receipt.
fn audit(manifest: &Manifest, plan: &Plan, content: Content) -> Result<Audit> {
    validate_project(&manifest.selection, &content)?;
    let now = SystemTime::now();
    let mut groups: BTreeMap<String, GroupAudit> = BTreeMap::new();
    let mut entries: Vec<AuditEntry> = Vec::new();
    for (group, files) in &plan.groups {
        let declared_files: Vec<GroupFile> = files.iter().map(PlannedFile::as_group_file).collect();
        let max_age_days = plan.freshness.get(group).copied();
        let (group_entries, checks) = verify(
            *group,
            &declared_files,
            max_age_days,
            &content,
            &manifest.selection.identity(),
            now,
        )?;
        let bytes = group_entries.iter().map(|entry| entry.size_bytes).sum();
        groups.insert(
            group.as_str().into(),
            GroupAudit {
                declared: true,
                file_count: group_entries.len(),
                bytes,
                admission: "verified".into(),
                checks,
            },
        );
        entries.extend(group_entries);
    }
    // The receipt names every admitted non-source file; source files are
    // covered by the inventory, not by a redistribution decision.
    entries.retain(|entry| entry.group != Group::Source);
    Ok(Audit {
        groups,
        entries,
        authorization: plan.authorization.clone(),
        dataset_allowlist: plan.dataset_allowlist.clone(),
        ..Audit::default()
    })
}

/// Writes every entry into a staging directory, streaming and verifying as it
/// goes.
///
/// Import never deserializes: weights, checkpoints and trajectories are read as
/// bytes, hashed, and written. The expanded caps are enforced by a running
/// counter against the bytes actually read, so a header that understates an
/// entry cannot buy more disk than the contract allows.
fn materialize(archive: &Path, manifest: &Manifest, staging: &Path) -> Result<()> {
    let mut zip = ZipArchive::new(BufReader::new(File::open(archive)?))?;
    let mut tally = Tally::default();
    let mut buffer = vec![0_u8; CHUNK];
    for entry in &manifest.entries {
        let group = group_of_entry(entry);
        let policy = group.policy();
        let target = staging.join(&entry.path);
        fs::create_dir_all(
            target
                .parent()
                .ok_or_else(|| refusal("missing file parent"))?,
        )?;
        let mut file = zip.by_name(&entry.path)?;
        let mut written = 0_u64;
        let mut digest = Sha256::new();
        let mut output = File::create(&target)?;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            // The cap is applied before the chunk is written, not after.
            if written + count as u64 > policy.max_file_bytes {
                return Err(refusal("file exceeds size limit"));
            }
            digest.update(&buffer[..count]);
            output.write_all(&buffer[..count])?;
            written += count as u64;
        }
        drop(output);
        if written != entry.size_bytes || format!("{:x}", digest.finalize()) != entry.sha256 {
            return Err(refusal("file digest or size mismatch"));
        }
        tally.add(group, written)?;
    }
    Ok(())
}

/// One fully verified archive: its manifest and its audit receipt.
struct Inspected {
    manifest: Manifest,
    audit: Audit,
}

/// Validates an archive without buffering it.
///
/// The manifest is read first, its declared inventory is checked against every
/// per-group limit and the package ceiling, and only then is each member
/// streamed through a digest. A multi-GiB package costs one chunk of memory.
///
/// Nothing is deserialized: weights, checkpoints and trajectories are hashed,
/// never loaded.
fn inspect(archive: &Path) -> Result<Inspected> {
    no_links(archive)?;
    if fs::symlink_metadata(archive)?.len() > MAX_PACKAGE_BYTES {
        return Err(refusal("archive size limit exceeded"));
    }
    let mut zip = ZipArchive::new(BufReader::new(File::open(archive)?))?;
    if zip.len() > MAX_FILES + 1 {
        return Err(refusal("archive file count exceeded"));
    }
    let mut encoded = Vec::new();
    {
        let file = zip
            .by_name(MANIFEST)
            .map_err(|_| refusal("missing package manifest"))?;
        if file.size() > 1024 * 1024 {
            return Err(refusal("manifest size limit exceeded"));
        }
        file.take(1024 * 1024 + 1).read_to_end(&mut encoded)?;
    }
    if encoded.len() > 1024 * 1024 {
        return Err(refusal("manifest size limit exceeded"));
    }
    let manifest: Manifest = serde_json::from_slice(&encoded)?;
    let plan = resolve(&manifest.selection)?;
    Version::parse(&manifest.tool_version)?;
    let declared: BTreeSet<_> = plan.files.iter().map(|file| file.path.as_str()).collect();
    let indexed: BTreeSet<_> = manifest
        .entries
        .iter()
        .map(|entry| entry.path.as_str())
        .collect();
    if declared != indexed || indexed.len() != manifest.entries.len() {
        return Err(refusal("selection and inventory disagree"));
    }
    tally_entries(&manifest)?;

    let mut names = BTreeSet::new();
    let mut verified: BTreeSet<String> = BTreeSet::new();
    let mut captured: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut declared_sizes: BTreeMap<String, (u64, String)> = BTreeMap::new();
    for index in 0..zip.len() {
        let mut file = zip.by_index(index)?;
        if file.name() == MANIFEST {
            continue;
        }
        portable(file.name())?;
        if file.is_dir()
            || file
                .unix_mode()
                .is_some_and(|mode| mode & 0o170000 != 0o100000)
            || !names.insert(file.name().to_ascii_lowercase())
        {
            return Err(refusal(
                "archive has links, collisions, or oversized entries",
            ));
        }
        let Some(entry) = declared_index(&manifest).get(file.name()).copied() else {
            // An undeclared member leaves a declared file unmatched; the
            // inventory check below reports that in the stable category.
            continue;
        };
        let entry = &manifest.entries[entry];
        let policy = group_of_entry(entry).policy();
        if file.size() > policy.max_file_bytes {
            return Err(refusal(
                "archive has links, collisions, or oversized entries",
            ));
        }
        check_compression(entry, file.compressed_size(), file.size())?;
        // Only a declared manifest or snapshot is buffered, and only under
        // `INSPECTION_LIMIT`; every payload is streamed through a digest.
        let capture = plan
            .files
            .iter()
            .any(|planned| planned.path == entry.path && capture_roles(planned))
            && file.size() <= INSPECTION_LIMIT;
        let (size, sha256, content) = if capture {
            let mut content = Vec::new();
            (&mut file)
                .take(INSPECTION_LIMIT + 1)
                .read_to_end(&mut content)?;
            (content.len() as u64, digest(&content), Some(content))
        } else {
            let (size, sha256) = hash_stream(&mut file, policy.max_file_bytes)?;
            (size, sha256, None)
        };
        if size != entry.size_bytes || sha256 != entry.sha256 {
            return Err(refusal("file digest or size mismatch"));
        }
        if let Some(content) = content {
            captured.insert(entry.path.clone(), content);
        }
        declared_sizes.insert(entry.path.clone(), (entry.size_bytes, entry.sha256.clone()));
        verified.insert(entry.path.clone());
    }
    for entry in &manifest.entries {
        if !verified.contains(&entry.path) {
            return Err(refusal("missing selected file"));
        }
    }
    // Identity last: every structural gate has already refused the members it
    // owns, so this is what catches an added, removed or renamed entry.
    if manifest.content_sha256 != identity(&manifest)? || manifest.entries.len() + 1 != zip.len() {
        return Err(refusal("package identity or inventory mismatch"));
    }
    let audit = audit(
        &manifest,
        &plan,
        Content::for_archive(&captured, &declared_sizes),
    )?;
    Ok(Inspected { manifest, audit })
}

fn group_of_entry(entry: &Entry) -> Group {
    entry.group.unwrap_or(Group::Source)
}

/// Index of the manifest inventory by path.
fn declared_index(manifest: &Manifest) -> BTreeMap<String, usize> {
    manifest
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.path.clone(), index))
        .collect()
}

/// Checks the declared inventory against the per-group byte caps, the package
/// ceiling, and the expansion ratio — before a single entry is expanded.
fn tally_entries(manifest: &Manifest) -> Result<()> {
    let mut tally = Tally::default();
    for entry in &manifest.entries {
        let group = group_of_entry(entry);
        let policy = group.policy();
        let compression = entry.compression.as_deref();
        match (manifest.selection.schema(), compression) {
            (Schema::Source, None) => {}
            (Schema::Training, Some("stored")) if !policy.compressed => {}
            (Schema::Training, Some("deflated")) if policy.compressed => {}
            _ => {
                return Err(refusal(format!(
                    "{} declares compression {:?}, which the {} group does not allow",
                    entry.path,
                    compression,
                    group.as_str()
                )));
            }
        }
        if let Some(compressed) = entry.compressed_size_bytes {
            // A deflated member is never empty. Everything else about the
            // compressed size is verified against the archive that carries it;
            // the ratio cap is what bounds the expansion it can claim. A tiny
            // file can legitimately deflate to slightly more than its own size.
            if compressed == 0 {
                return Err(refusal(format!(
                    "{} declares an empty compressed size",
                    entry.path
                )));
            }
            check_ratio(entry, compressed)?;
        }
        tally.add(group, entry.size_bytes)?;
    }
    Ok(())
}

/// The declared compressed size must describe the archive it travels in.
fn check_compression(entry: &Entry, compressed: u64, expanded: u64) -> Result<()> {
    if expanded != entry.size_bytes {
        return Err(refusal("file digest or size mismatch"));
    }
    if let Some(declared) = entry.compressed_size_bytes
        && declared != compressed
    {
        return Err(refusal("entry size mismatch"));
    }
    check_ratio(entry, compressed)
}

/// Refuses an entry whose declared expansion exceeds the ratio cap.
///
/// Checked at export as well as at import: a package that would be an archive
/// bomb for the recipient is refused before it leaves the sender.
fn check_ratio(entry: &Entry, compressed: u64) -> Result<()> {
    if compressed == 0 {
        return Ok(());
    }
    let ratio = entry.size_bytes / compressed;
    if ratio > MAX_EXPANSION_RATIO {
        return Err(refusal(format!(
            "{} expands {ratio}x, past the {MAX_EXPANSION_RATIO}:1 ratio cap",
            entry.path
        )));
    }
    Ok(())
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

impl Scan {
    /// Puts every discovery-ordered list into a canonical order.
    ///
    /// The walk yields entries in `read_dir` order, which differs per host and
    /// per filesystem. `files` is already ordered because it is a `BTreeMap`,
    /// but `local_overrides` and `forbidden` are pushed in walk order and go
    /// straight into the machine-readable `glr.cli-output.v1` report, so
    /// consumers need a stable order for both.
    fn canonicalize(&mut self) {
        self.local_overrides.sort();
        self.forbidden.sort();
    }
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
            if is_local_override_path(&relative) {
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
    let inspected = inspect(&archive)?;
    let manifest = &inspected.manifest;
    if expected_environment
        .as_ref()
        .is_some_and(|value| *value != manifest.selection.environment_id())
        || expected_contract
            .as_ref()
            .is_some_and(|value| *value != manifest.selection.contract_sha256())
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
    scan.canonicalize();

    let mut missing = Vec::new();
    let mut mismatched = Vec::new();
    for entry in &manifest.entries {
        let size_matches = scan.files.get(&entry.path) == Some(&entry.size_bytes);
        let limit = group_of_entry(entry).policy().max_file_bytes;
        let digest_matches = scan
            .files
            .contains_key(&entry.path)
            .then(|| hash_file(&root.join(&entry.path), limit))
            .transpose()?
            .is_some_and(|(_, sha256)| sha256 == entry.sha256);
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
            project.environment_id == manifest.selection.environment_id()
                && project.protocol_version == manifest.selection.protocol_version()
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
        .entries
        .iter()
        .map(|entry| &entry.path)
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
            "environment_id": manifest.selection.environment_id(),
            "protocol_version": manifest.selection.protocol_version(),
            "contract_sha256": manifest.selection.contract_sha256(),
            "content_sha256": manifest.content_sha256,
            "package_version": manifest.selection.package_version(),
            "file_count": manifest.entries.len(),
            "groups": inspected.audit.groups,
        },
        "audit": inspected.audit,
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
            let (mut manifest, audit) = plan(root, &selection)?;
            if let PackageCommand::Export { output, .. } = command {
                let output = absolute(output)?;
                let parent = output
                    .parent()
                    .ok_or_else(|| refusal("missing output parent"))?;
                no_links(parent)?;
                let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
                {
                    let mut writer = ZipWriter::new(temporary.as_file_mut());
                    // Entries first: a compressed group declares both sizes,
                    // and the compressed size only exists once the entry has
                    // been written.
                    for entry in &manifest.entries {
                        let group = group_of_entry(entry);
                        let options = SimpleFileOptions::default()
                            .compression_method(if group.policy().compressed {
                                zip::CompressionMethod::Deflated
                            } else {
                                zip::CompressionMethod::Stored
                            })
                            .unix_permissions(0o644);
                        writer.start_file(&entry.path, options)?;
                        copy_bounded(
                            &root.join(&entry.path),
                            &mut writer,
                            group.policy().max_file_bytes,
                        )?;
                    }
                    writer.finish()?;
                }
                temporary.as_file().sync_all()?;
                // Read the sizes the encoder actually produced, then append the
                // manifest that declares them.
                let compressed_sizes = compressed_sizes(temporary.path())?;
                for entry in &mut manifest.entries {
                    let Some(compressed) = compressed_sizes.get(&entry.path).copied() else {
                        return Err(refusal(format!("{} was not written", entry.path)));
                    };
                    check_ratio(entry, compressed)?;
                    if entry.compression.is_some() {
                        entry.compressed_size_bytes = Some(compressed);
                    }
                }
                {
                    let mut writer = ZipWriter::new_append(temporary.as_file())?;
                    let options = SimpleFileOptions::default()
                        .compression_method(zip::CompressionMethod::Stored)
                        .unix_permissions(0o644);
                    writer.start_file(MANIFEST, options)?;
                    writer.write_all(&serde_json::to_vec(&manifest)?)?;
                    writer.finish()?;
                }
                temporary.as_file().sync_all()?;
                if temporary.as_file().metadata()?.len() > MAX_PACKAGE_BYTES {
                    return Err(refusal("archive size limit exceeded"));
                }
                temporary
                    .persist_noclobber(output)
                    .map_err(|error| Error::Io(error.error))?;
            }
            emit(
                "package",
                &json!({"status": "verified-source-inventory", "manifest": manifest, "audit": audit, "executed": false}),
                json,
            )?;
            Ok(0)
        }
        PackageCommand::Inspect { archive } | PackageCommand::Import { archive, .. } => {
            let archive = absolute(archive)?;
            let inspected = inspect(&archive)?;
            let manifest = inspected.manifest;
            if let PackageCommand::Import {
                destination,
                expected_environment,
                expected_contract,
                ..
            } = command
            {
                if *expected_environment != manifest.selection.environment_id()
                    || *expected_contract != manifest.selection.contract_sha256()
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
                materialize(&archive, &manifest, staging.path())?;
                promote(staging.path(), &destination)?;
            }
            emit(
                "package",
                &json!({"status": "verified-source-package", "manifest": manifest, "audit": inspected.audit, "executed": false, "training_ready": false}),
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
            serde_json::to_vec(&Selection::Source(SourceSelection {
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
            }))
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
        let manifest = inspect(&first).unwrap().manifest;
        let destination = root.path().join("imported");
        let command = PackageCommand::Import {
            archive: first,
            destination: destination.clone(),
            expected_environment: "synthetic.package".into(),
            expected_contract: manifest.selection.contract_sha256().into(),
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
            serde_json::to_vec(&Selection::Source(SourceSelection {
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
            }))
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
    fn scan_canonicalize_orders_every_discovery_ordered_list() {
        let mut scan = Scan {
            files: BTreeMap::new(),
            local_overrides: vec!["z.local.json".into(), "a.local.json".into()],
            forbidden: vec!["logs".into(), ".glr".into(), "cache".into()],
        };
        scan.canonicalize();
        assert_eq!(scan.local_overrides, vec!["a.local.json", "z.local.json"]);
        assert_eq!(scan.forbidden, vec![".glr", "cache", "logs"]);
    }

    #[test]
    fn conformance_reports_denied_artifact_paths_in_a_stable_order() {
        let executable = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let (_recipient, destination, archive) = round_trip(&executable);
        // Created in reverse sorted order so a walk-order report is visible.
        fs::create_dir(destination.join("logs")).unwrap();
        fs::create_dir(destination.join("cache")).unwrap();
        let report = conformance(&destination, &conformance_command(&archive)).unwrap();
        assert_eq!(report["artifacts"]["forbidden"], json!(["cache", "logs"]));
        assert_eq!(report["artifacts"]["run_store"], false);
        assert_eq!(report["materialization"]["status"], "complete");
        assert!(
            report["blockers"]
                .as_array()
                .unwrap()
                .iter()
                .any(|blocker| blocker["detail"].as_str().unwrap()
                    == "2 denied cache, output or run-store path(s) are present"),
            "{:?}",
            report["blockers"]
        );
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
        for path in [
            "glr-project.local.json",
            "secrets.local.toml",
            "a.local",
            "a.local/b.json",
            "a.local/sub/b.json",
        ] {
            assert!(source_path(path).is_err(), "{path}");
        }
    }

    /// Export and conformance must answer the same question with the same
    /// predicate: what export refuses is exactly what the scan ignores.
    #[test]
    fn export_and_conformance_agree_on_every_local_override_form() {
        // Every entry is otherwise source-legal, so the local-override rule is
        // the only thing that can make `source_path` refuse it.
        let cases: &[(&str, bool)] = &[
            ("train.py", false),
            ("pkg/mod.py", false),
            ("src/deep/nested/lib.rs", false),
            ("local.py", false),
            ("pkg/local/mod.py", false),
            ("docs/local.override.md", false),
            ("glr-project.local.json", true),
            ("secrets.local.toml", true),
            ("pkg/config.local.json", true),
            ("pkg/a.local.d/b.json", true),
            ("x/a.local.deep/b.py", true),
            ("a.local/b.json", true),
            ("a.local/sub/b.json", true),
            ("a.local/sub/deep/c.md", true),
            ("pkg/inner.local/nested/b.json", true),
            ("A.LOCAL/B.JSON", true),
        ];
        for (path, expected) in cases {
            assert_eq!(
                is_local_override_path(path),
                *expected,
                "is_local_override_path({path:?})"
            );
            assert_eq!(
                source_path(path).is_ok(),
                !is_local_override_path(path),
                "export and conformance disagree on {path:?}"
            );
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
            push_source_file(&mut selection, path);
            assert!(resolve(&selection).is_err(), "{path}");
        }
        let mut selection = source.clone();
        push_source_file(&mut selection, "Case/a.py");
        push_source_file(&mut selection, "case/b.py");
        assert!(resolve(&selection).is_err());
        let mut selection = source.clone();
        match &mut selection {
            Selection::Source(selection) => selection.required_glr = ">=999.0.0".into(),
            Selection::Training(selection) => selection.required_glr = ">=999.0.0".into(),
        }
        assert!(resolve(&selection).is_err());
        let unknown = serde_json::from_str::<Selection>(&format!(
            "{{\"schema_version\":\"glr.source-package.v9\",\"package_version\":\"1.0.0\",             \"required_glr\":\">=0.18.0\",\"environment_id\":\"e\",\"protocol_version\":\"1.0\",             \"contract_sha256\":\"{}\",\"source_revision\":\"s\",             \"redistribution_license\":\"MIT\",\"files\":[\"train.py\"]}}",
            "a".repeat(64)
        ));
        assert!(
            unknown.is_err(),
            "an unknown schema version must fail closed"
        );
    }

    /// Adds one file to a source-only selection, whatever its profile.
    fn push_source_file(selection: &mut Selection, path: &str) {
        match selection {
            Selection::Source(selection) => selection.files.push(path.into()),
            Selection::Training(selection) => selection
                .groups
                .entry(Group::Source)
                .or_default()
                .files
                .push(GroupFile {
                    path: path.into(),
                    role: source_role(path).into(),
                }),
        }
    }

    /// The bytes a planned fixture carries, read back from the source tree.
    fn fixture_payload(root: &Path, manifest: &Manifest) -> Vec<(String, Vec<u8>)> {
        manifest
            .entries
            .iter()
            .map(|entry| {
                (
                    entry.path.clone(),
                    fs::read(root.join(&entry.path)).unwrap(),
                )
            })
            .collect()
    }

    /// The per-file limit of the source group.
    fn source_file_limit() -> u64 {
        Group::Source.policy().max_file_bytes
    }

    /// A reader that yields `count` zero bytes, however many chunks it takes.
    struct Zeroes {
        remaining: u64,
    }

    impl Read for Zeroes {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            if self.remaining == 0 {
                return Ok(0);
            }
            let count = buffer.len().min(self.remaining as usize);
            self.remaining -= count as u64;
            Ok(count)
        }
    }

    /// A streamed read refuses the moment it passes its limit, so a header that
    /// understates an entry cannot buy more memory than the contract allows.
    #[test]
    fn hash_stream_stops_at_its_limit_without_buffering_the_payload() {
        let limit = Group::Report.policy().max_file_bytes;
        let error = hash_stream(
            Zeroes {
                remaining: u64::MAX,
            },
            limit,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("file exceeds size limit"), "{error}");
        // Exactly one chunk past the limit is the most it ever holds.
        let (size, sha256) = hash_stream(Zeroes { remaining: limit }, limit).unwrap();
        assert_eq!(size, limit);
        assert_eq!(sha256, digest(&vec![0_u8; limit as usize]));
    }

    /// The group cap and the package ceiling are enforced on the running total,
    /// so the package — not one group — is what a recipient pays for.
    #[test]
    fn tally_enforces_the_group_cap_and_the_package_ceiling() {
        let mut tally = Tally::default();
        let report = Group::Report.policy();
        assert!(tally.add(Group::Report, report.max_group_bytes).is_ok());
        assert!(
            tally
                .add(Group::Report, 1)
                .unwrap_err()
                .to_string()
                .contains("expanded size limit exceeded")
        );

        // Two groups each inside their own cap can still breach the ceiling.
        let mut tally = Tally::default();
        assert!(tally.add(Group::Model, 3 * 1024 * 1024 * 1024).is_ok());
        assert!(
            tally
                .add(Group::Dataset, 2 * 1024 * 1024 * 1024)
                .unwrap_err()
                .to_string()
                .contains("package byte ceiling exceeded")
        );
    }

    /// A compressed entry is admitted on its declared ratio, and refused when
    /// the ratio or the archive's own compressed size disagrees.
    #[test]
    fn expansion_ratio_is_enforced_on_the_declared_sizes() {
        let entry = Entry {
            path: "models/reference/artifacts/weights.safetensors".into(),
            size_bytes: 200 * 1024,
            sha256: "a".repeat(64),
            group: Some(Group::Model),
            role: Some("model-artifact".into()),
            compression: Some("deflated".into()),
            compressed_size_bytes: Some(1024),
        };
        assert!(check_ratio(&entry, 1024).is_ok());
        assert!(
            check_ratio(&entry, 512)
                .unwrap_err()
                .to_string()
                .contains("200:1 ratio cap")
        );
        // Stored entries declare no compressed size, so the cap cannot fire.
        let mut stored = entry.clone();
        stored.compression = Some("stored".into());
        stored.compressed_size_bytes = None;
        assert!(check_compression(&stored, stored.size_bytes, stored.size_bytes).is_ok());
    }

    #[test]
    fn rejects_hardlinks_and_atomic_promotion_preserves_existing_destination() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("source.py"), b"pass").unwrap();
        fs::hard_link(root.path().join("source.py"), root.path().join("alias.py")).unwrap();
        assert!(read_file(&root.path().join("source.py"), source_file_limit()).is_err());
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
        let (manifest, _) = plan(root.path(), &selection).unwrap();
        let payload = fixture_payload(root.path(), &manifest);
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
        let (manifest, _) = plan(root.path(), &selection).unwrap();
        let payload = fixture_payload(root.path(), &manifest);
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
                        writer
                            .write_all(&vec![0; source_file_limit() as usize + 1])
                            .unwrap();
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
