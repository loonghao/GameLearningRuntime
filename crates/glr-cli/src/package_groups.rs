//! Entry-group policy for portable training packages (ADR-0041 D3, D5, D6).
//!
//! This module decides *what* a non-source group may carry and *why* it was
//! admitted. It never opens an archive and never writes a package byte: the
//! envelope lives in [`crate::package`], and every read here goes through
//! [`Content`], which is bounded.
//!
//! Two rules shape the whole module:
//!
//! * **Deny by default.** `source` is the only group that is allowed without an
//!   extra gate; `model`, `dataset`, `knowledge` and `report` need a declared
//!   group, a closed role, an extension allowlist and a group-specific proof.
//! * **Nothing is deserialized.** A weight, checkpoint or trajectory payload is
//!   bound by size and digest only. The JSON that *is* parsed here — a model
//!   bundle manifest, a knowledge snapshot, a demonstration artifact — is a
//!   declared manifest, never a learner or framework payload.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::contracts::MODEL_BUNDLE_SCHEMA_VERSION;
use crate::error::{Error, Result};
use crate::package::{is_local_override, read_file};
use crate::project::validate_identifier;

/// Aggregate audit receipt emitted for every admitted non-source file.
pub const AUDIT_SCHEMA: &str = "glr.package-audit.v1";
pub const KNOWLEDGE_SNAPSHOT_SCHEMA: &str = "glr.knowledge-snapshot.v1";
pub const DEMONSTRATION_ARTIFACT_SCHEMA: &str = "glr.demonstration-artifact.v1";
pub const DATASET_ALLOWLIST_SCHEMA: &str = "glr.dataset-allowlist.v1";
pub const AUTHORIZATION_SCHEMA: &str = "glr.redistribution-authorization.v1";

/// Hard ceiling across every group, in expanded bytes (ADR-0041 D5).
///
/// The per-group caps sum to more than this, so the ceiling — not the sum — is
/// what a package actually runs into.
pub const MAX_PACKAGE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// File-count ceiling across every group.
///
/// Deliberately equal to the archive member gate in `crate::package`: the
/// package-wide file count is what a recipient pays for, not the sum of the
/// per-group caps.
pub const MAX_PACKAGE_FILES: usize = 1024;
/// Per-entry and per-group expansion ratio cap for compressed groups.
pub const MAX_EXPANSION_RATIO: u64 = 200;
/// Upper bound for a file whose *content* this module inspects. Every file read
/// here is a declared manifest or snapshot, never a payload.
pub const INSPECTION_LIMIT: u64 = 8 * 1024 * 1024;

/// Directory-name component that no group may carry: build output, captured
/// media, local run state, credentials, or an artifact root that belongs to
/// another tool.
pub fn denied_component(part: &str) -> bool {
    [
        "target",
        "node_modules",
        "recordings",
        "screenshots",
        "logs",
        "datasets",
        "secrets",
        "credentials",
        "cache",
        ".glr",
    ]
    .contains(&part)
}

fn refusal(message: String) -> Error {
    Error::Contract(format!("training package: {message}"))
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

/// The closed group vocabulary. `source` is allowed by default; every other
/// group is deny-by-default and needs a group-specific proof.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Group {
    Source,
    Model,
    Dataset,
    Knowledge,
    Report,
}

/// Per-group limits and compression policy (ADR-0041 D5).
pub struct Policy {
    pub max_files: usize,
    pub max_file_bytes: u64,
    pub max_group_bytes: u64,
    /// `source` stays uncompressed: determinism and the absence of expansion
    /// amplification are worth more than size there.
    pub compressed: bool,
}

impl Group {
    pub const ALL: [Group; 5] = [
        Group::Source,
        Group::Model,
        Group::Dataset,
        Group::Knowledge,
        Group::Report,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Model => "model",
            Self::Dataset => "dataset",
            Self::Knowledge => "knowledge",
            Self::Report => "report",
        }
    }

    /// Package-root directory that owns this group. `source` has no root: it is
    /// every path no other group claims.
    pub const fn root(self) -> Option<&'static str> {
        match self {
            Self::Source => None,
            Self::Model => Some("models"),
            Self::Dataset => Some("data"),
            Self::Knowledge => Some("knowledge"),
            Self::Report => Some("reports"),
        }
    }

    pub const fn policy(self) -> Policy {
        match self {
            Self::Source => Policy {
                max_files: 1024,
                max_file_bytes: 16 * 1024 * 1024,
                max_group_bytes: 128 * 1024 * 1024,
                compressed: false,
            },
            Self::Model => Policy {
                max_files: 64,
                max_file_bytes: 1024 * 1024 * 1024,
                max_group_bytes: 4 * 1024 * 1024 * 1024,
                compressed: true,
            },
            Self::Dataset => Policy {
                max_files: 512,
                max_file_bytes: 1024 * 1024 * 1024,
                max_group_bytes: 4 * 1024 * 1024 * 1024,
                compressed: true,
            },
            Self::Knowledge => Policy {
                max_files: 256,
                max_file_bytes: 64 * 1024 * 1024,
                max_group_bytes: 256 * 1024 * 1024,
                compressed: true,
            },
            Self::Report => Policy {
                max_files: 64,
                max_file_bytes: 16 * 1024 * 1024,
                max_group_bytes: 64 * 1024 * 1024,
                compressed: true,
            },
        }
    }

    /// Closed, group-scoped role vocabulary.
    ///
    /// Roles exist for receipts, per-group policy and audit. They grant no
    /// behavior and are never an execution hint.
    pub const fn roles(self) -> &'static [&'static str] {
        match self {
            Self::Source => &["project-manifest", "dependency-lock", "source-file"],
            Self::Model => &["model-manifest", "model-input", "model-artifact"],
            Self::Dataset => &["dataset-manifest", "dataset-payload"],
            Self::Knowledge => &["knowledge-snapshot"],
            Self::Report => &["aggregate-report"],
        }
    }

    pub fn valid_role(self, role: &str) -> bool {
        self.roles().contains(&role)
    }
}

/// Which group owns `path`, given the groups the package declares.
///
/// A path under a group root belongs to that group. When the owning group is
/// not declared the path is refused: content may never be smuggled into a
/// package through a group the manifest does not admit.
pub fn group_of(path: &str, declared: &BTreeSet<Group>) -> Result<Group> {
    let lower = path.to_ascii_lowercase();
    for group in Group::ALL {
        let Some(root) = group.root() else {
            continue;
        };
        if !lower.starts_with(&format!("{root}/")) {
            continue;
        }
        if declared.contains(&group) {
            return Ok(group);
        }
        return Err(refusal(format!(
            "{path} belongs to the {} group, which this package does not declare",
            group.as_str()
        )));
    }
    Ok(Group::Source)
}

/// Extension allowlist per group.
fn extensions(group: Group) -> &'static [&'static str] {
    match group {
        Group::Source => &[
            "py", "rs", "toml", "json", "yaml", "yml", "md", "txt", "lock", "cs", "cpp", "h",
            "hpp", "gd",
        ],
        Group::Model => &[
            "safetensors",
            "bin",
            "pt",
            "pth",
            "ckpt",
            "onnx",
            "json",
            "toml",
            "yaml",
            "yml",
            "md",
            "txt",
            "csv",
            "lock",
        ],
        Group::Dataset => &[
            "json", "jsonl", "csv", "parquet", "txt", "md", "yaml", "yml",
        ],
        Group::Knowledge => &["json"],
        Group::Report => &["json", "md", "csv"],
    }
}

/// Formats whose safe handling cannot be proven offline because reading them
/// means executing a deserializer. Refused at admission, never loaded.
const PICKLE_FAMILY: &[&str] = &["pkl", "pickle", "joblib", "npy", "npz", "dill"];

/// Component names a `report` file may never carry: aggregates only, so raw
/// logs, recordings and trajectories are excluded by name as well as by type.
const REPORT_DENIED_COMPONENTS: &[&str] = &[
    "trajectory",
    "trajectories",
    "episodes",
    "episode",
    "recordings",
    "logs",
    "runs",
    "raw",
];

/// Admission predicate for one path in one group.
pub fn admit(group: Group, path: &str) -> Result<()> {
    let lower = path.to_ascii_lowercase();
    for part in lower.split('/') {
        if part.is_empty()
            || part == "."
            || part == ".."
            || part.starts_with('.')
            || denied_component(part)
            || is_local_override(part)
        {
            return Err(refusal(format!("{path} carries a denied path component")));
        }
    }
    if let Some(root) = group.root()
        && !lower.starts_with(&format!("{root}/"))
    {
        return Err(refusal(format!(
            "{} files must live under {root}/, got {path}",
            group.as_str()
        )));
    }
    if group == Group::Report
        && lower
            .split('/')
            .any(|part| REPORT_DENIED_COMPONENTS.contains(&part))
    {
        return Err(refusal(format!(
            "{path} is a raw log, recording or trajectory, not an aggregate"
        )));
    }
    let file = lower.rsplit('/').next().unwrap_or_default();
    let extension = file.rsplit_once('.').map(|(_, extension)| extension);
    let Some(extension) = extension else {
        return Err(refusal(format!(
            "{path} has no extension and cannot be admitted"
        )));
    };
    if PICKLE_FAMILY.contains(&extension) {
        return Err(refusal(format!(
            "{path} needs a deserializer that import must never run"
        )));
    }
    if !extensions(group).contains(&extension) {
        return Err(refusal(format!(
            "{} allowlist excludes {path}",
            group.as_str()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Selection payload
// ---------------------------------------------------------------------------

/// One declared file: a portable path plus its group-scoped role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupFile {
    pub path: String,
    pub role: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupSelection {
    pub files: Vec<GroupFile>,
    /// Freshness budget for `knowledge` snapshots, in days. Required by and
    /// only valid for the `knowledge` group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_age_days: Option<u32>,
}

/// The separately reviewed allowlist a `dataset` export needs.
///
/// A dataset is never admitted because its extension is plausible: the reviewed
/// list names the exact paths a reviewer approved.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetAllowlist {
    pub schema_version: String,
    pub reviewed_by: String,
    pub review_date: String,
    pub entries: Vec<String>,
}

/// Recorded redistribution authorization for a `dataset` export (ADR-0041 D3).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedistributionAuthorization {
    pub schema_version: String,
    pub approver: String,
    pub scope: String,
    pub license: String,
    pub date: String,
}

fn labeled_string(value: &str, label: &str, limit: usize) -> Result<()> {
    if value.is_empty() || value.len() > limit || value.chars().any(char::is_control) {
        return Err(refusal(format!("invalid {label}")));
    }
    Ok(())
}

fn is_calendar_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[..4].iter().all(u8::is_ascii_digit)
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[8..].iter().all(u8::is_ascii_digit)
}

pub fn validate_dataset_allowlist(allowlist: &DatasetAllowlist, files: &[GroupFile]) -> Result<()> {
    if allowlist.schema_version != DATASET_ALLOWLIST_SCHEMA {
        return Err(refusal(format!(
            "dataset allowlist must be {DATASET_ALLOWLIST_SCHEMA}"
        )));
    }
    labeled_string(&allowlist.reviewed_by, "dataset allowlist reviewed_by", 256)?;
    if !is_calendar_date(&allowlist.review_date) {
        return Err(refusal(
            "dataset allowlist review_date must be YYYY-MM-DD".into(),
        ));
    }
    if allowlist.entries.is_empty() {
        return Err(refusal("dataset allowlist cannot be empty".into()));
    }
    let approved: BTreeSet<&str> = allowlist.entries.iter().map(String::as_str).collect();
    if approved.len() != allowlist.entries.len() {
        return Err(refusal("dataset allowlist has duplicate entries".into()));
    }
    for file in files {
        if !approved.contains(file.path.as_str()) {
            return Err(refusal(format!(
                "{} is not on the reviewed dataset allowlist",
                file.path
            )));
        }
    }
    Ok(())
}

pub fn validate_authorization(authorization: &RedistributionAuthorization) -> Result<()> {
    if authorization.schema_version != AUTHORIZATION_SCHEMA {
        return Err(refusal(format!(
            "redistribution authorization must be {AUTHORIZATION_SCHEMA}"
        )));
    }
    labeled_string(&authorization.approver, "authorization approver", 256)?;
    labeled_string(&authorization.scope, "authorization scope", 512)?;
    labeled_string(&authorization.license, "authorization license", 256)?;
    if !is_calendar_date(&authorization.date) {
        return Err(refusal(
            "redistribution authorization date must be YYYY-MM-DD".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Content access
// ---------------------------------------------------------------------------

/// Bounded view of the bytes a package declares, from an archive or a tree.
///
/// Only declared manifests and snapshots are ever materialized here, and only
/// under [`INSPECTION_LIMIT`]. A payload is represented by its size and digest,
/// never by its bytes.
pub struct Content<'a> {
    root: Option<&'a Path>,
    captured: Option<&'a BTreeMap<String, Vec<u8>>>,
    declared: &'a BTreeMap<String, (u64, String)>,
}

impl<'a> Content<'a> {
    /// Content of an archive: bytes captured while it was streamed.
    pub fn for_archive(
        captured: &'a BTreeMap<String, Vec<u8>>,
        declared: &'a BTreeMap<String, (u64, String)>,
    ) -> Self {
        Self {
            root: None,
            captured: Some(captured),
            declared,
        }
    }

    /// Content of a source tree: bytes read on demand, never buffered wholesale.
    pub fn for_tree(root: &'a Path, declared: &'a BTreeMap<String, (u64, String)>) -> Self {
        Self {
            root: Some(root),
            captured: None,
            declared,
        }
    }

    /// Declared `(size, digest)` for a path.
    pub fn declared(&self, path: &str) -> Result<(u64, String)> {
        self.declared
            .get(path)
            .cloned()
            .ok_or_else(|| refusal(format!("{path} is not declared by the package")))
    }

    /// The bytes of a declared manifest or snapshot, under a hard cap.
    pub fn bytes(&self, path: &str, limit: u64) -> Result<Vec<u8>> {
        let (size, _) = self.declared(path)?;
        if size > limit {
            return Err(refusal(format!(
                "{path} is too large to inspect ({size} bytes)"
            )));
        }
        if let Some(captured) = self.captured
            && let Some(bytes) = captured.get(path)
        {
            return Ok(bytes.clone());
        }
        let Some(root) = self.root else {
            return Err(refusal(format!("{path} was not captured from the archive")));
        };
        read_file(&root.join(path), limit)
    }

    /// Parsed JSON for a declared manifest or snapshot.
    ///
    /// `expected` is checked before the typed parse, so a file that is valid
    /// JSON but the wrong contract is reported as the wrong contract.
    pub fn json<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        limit: u64,
        expected: &str,
    ) -> Result<T> {
        let bytes = self.bytes(path, limit)?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| refusal(format!("{path} is not valid JSON for its role: {error}")))?;
        if value.get("schema_version").and_then(Value::as_str) != Some(expected) {
            return Err(refusal(format!("{path} must be {expected}")));
        }
        serde_json::from_value(value)
            .map_err(|error| refusal(format!("{path} does not satisfy {expected}: {error}")))
    }
}

/// Identity a group's contents must agree with.
pub struct Identity<'a> {
    pub environment_id: &'a str,
    pub protocol_version: &'a str,
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

/// One admitted file and the reason it was admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEntry {
    pub path: String,
    pub group: Group,
    pub role: String,
    pub size_bytes: u64,
    pub sha256: String,
    /// The contract or allowlist that admitted this file.
    pub admitted_by: String,
}

/// Per-group rollup for the audit receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupAudit {
    pub declared: bool,
    pub file_count: usize,
    pub bytes: u64,
    /// Every declared group is verified before the package is admitted; a group
    /// that fails never reaches the receipt.
    pub admission: String,
    /// Every check that ran for this group.
    pub checks: Vec<String>,
}

/// Aggregate audit receipt: one record per admitted file, plus the group gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Audit {
    pub schema_version: String,
    pub groups: BTreeMap<String, GroupAudit>,
    pub entries: Vec<AuditEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization: Option<RedistributionAuthorization>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_allowlist: Option<DatasetAllowlist>,
}

impl Default for Audit {
    fn default() -> Self {
        Self {
            schema_version: AUDIT_SCHEMA.into(),
            groups: BTreeMap::new(),
            entries: Vec::new(),
            authorization: None,
            dataset_allowlist: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Group verification
// ---------------------------------------------------------------------------

/// Verifies one group and returns its audit entries.
///
/// `source` is covered by the project-identity check in [`crate::package`];
/// every other group needs the proof its row of the ADR-0041 D3 table names.
pub fn verify(
    group: Group,
    files: &[GroupFile],
    max_age_days: Option<u32>,
    content: &Content,
    identity: &Identity,
    now: SystemTime,
) -> Result<(Vec<AuditEntry>, Vec<String>)> {
    match group {
        Group::Source => Ok((
            audit_entries(group, files, content, "source-allowlist")?,
            vec!["glr.project.v1".into()],
        )),
        Group::Model => {
            let bundle = verify_model_bundle(files, content, identity)?;
            Ok((bundle.entries, vec![MODEL_BUNDLE_SCHEMA_VERSION.into()]))
        }
        Group::Dataset => {
            verify_demonstration_provenance(files, content, identity)?;
            Ok((
                audit_entries(
                    group,
                    files,
                    content,
                    &format!("{DATASET_ALLOWLIST_SCHEMA}+{DEMONSTRATION_ARTIFACT_SCHEMA}"),
                )?,
                vec![
                    DATASET_ALLOWLIST_SCHEMA.into(),
                    DEMONSTRATION_ARTIFACT_SCHEMA.into(),
                    AUTHORIZATION_SCHEMA.into(),
                ],
            ))
        }
        Group::Knowledge => {
            let checks = verify_knowledge(files, content, max_age_days, now)?;
            Ok((
                audit_entries(group, files, content, KNOWLEDGE_SNAPSHOT_SCHEMA)?,
                checks,
            ))
        }
        Group::Report => Ok((
            audit_entries(group, files, content, "report-aggregate-allowlist")?,
            vec!["report-aggregate-allowlist".into()],
        )),
    }
}

fn audit_entries(
    group: Group,
    files: &[GroupFile],
    content: &Content,
    admitted_by: &str,
) -> Result<Vec<AuditEntry>> {
    files
        .iter()
        .map(|file| {
            let (size_bytes, sha256) = content.declared(&file.path)?;
            Ok(AuditEntry {
                path: file.path.clone(),
                group,
                role: file.role.clone(),
                size_bytes,
                sha256,
                admitted_by: admitted_by.into(),
            })
        })
        .collect()
}

/// `glr.model-bundle.v1` verification over the package inventory (ADR-0010).
///
/// The bundle is verified against declared sizes and digests, so no weight is
/// opened, parsed or deserialized. Every model-group file must be accounted for
/// by the bundle manifest: a bundle cannot carry an undeclared extra file.
fn verify_model_bundle(
    files: &[GroupFile],
    content: &Content,
    identity: &Identity,
) -> Result<ModelVerification> {
    let manifests: Vec<&GroupFile> = files
        .iter()
        .filter(|file| file.role == "model-manifest")
        .collect();
    let [manifest] = manifests.as_slice() else {
        return Err(refusal(format!(
            "a model group needs exactly one model-manifest, got {}",
            manifests.len()
        )));
    };
    if !manifest.path.ends_with("/manifest.json") && manifest.path != "manifest.json" {
        return Err(refusal(format!(
            "{} must be named manifest.json at the bundle root",
            manifest.path
        )));
    }
    let root = manifest
        .path
        .rsplit_once('/')
        .map(|(prefix, _)| prefix)
        .unwrap_or_default()
        .to_owned();
    let bundle: crate::contracts::ModelBundleManifest = content.json(
        &manifest.path,
        INSPECTION_LIMIT,
        MODEL_BUNDLE_SCHEMA_VERSION,
    )?;
    if bundle.environment_id != identity.environment_id
        || bundle.protocol_version != identity.protocol_version
    {
        return Err(refusal(format!(
            "{} was trained for {}/{} and cannot join a package for {}/{}",
            manifest.path,
            bundle.environment_id,
            bundle.protocol_version,
            identity.environment_id,
            identity.protocol_version
        )));
    }
    if bundle.seeds.is_empty() || bundle.inputs.is_empty() || bundle.artifacts.is_empty() {
        return Err(refusal(format!(
            "{} needs seeds, inputs and artifacts",
            manifest.path
        )));
    }
    let mut accounted: BTreeSet<String> = BTreeSet::new();
    accounted.insert(manifest.path.clone());
    for (kind, entries) in [("inputs", &bundle.inputs), ("artifacts", &bundle.artifacts)] {
        for entry in entries {
            let path = format!("{root}/{kind}/{}", entry.path);
            if !is_relative(&entry.path) {
                return Err(refusal(format!(
                    "{} declares a non-portable {kind} path",
                    manifest.path
                )));
            }
            let (size, sha256) = content.declared(&path).map_err(|_| {
                refusal(format!(
                    "{} declares {path}, which the package omits",
                    manifest.path
                ))
            })?;
            if size != entry.size_bytes || sha256 != entry.sha256 {
                return Err(refusal(format!(
                    "{path} does not match the size or digest {manifest_path} declares",
                    manifest_path = manifest.path
                )));
            }
            if !accounted.insert(path.clone()) {
                return Err(refusal(format!("{path} is declared twice by the bundle")));
            }
        }
    }
    for file in files {
        if !accounted.contains(&file.path) {
            return Err(refusal(format!(
                "{} is not declared by the model bundle",
                file.path
            )));
        }
        let expected = match file.role.as_str() {
            "model-manifest" => continue,
            role @ ("model-input" | "model-artifact") => {
                let kind = if role == "model-input" {
                    "inputs"
                } else {
                    "artifacts"
                };
                file.path.starts_with(&format!("{root}/{kind}/"))
            }
            other => {
                return Err(refusal(format!(
                    "{} has an unknown model role {other:?}",
                    file.path
                )));
            }
        };
        if !expected {
            return Err(refusal(format!(
                "{} does not live in the {} layout its role declares",
                file.path, file.role
            )));
        }
    }
    Ok(ModelVerification {
        entries: audit_entries(Group::Model, files, content, MODEL_BUNDLE_SCHEMA_VERSION)?,
    })
}

struct ModelVerification {
    entries: Vec<AuditEntry>,
}

fn is_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && !path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
}

/// Demonstration-provenance binding for the `dataset` group (ADR-0013).
///
/// Every dataset payload must be bound by a `glr.demonstration-artifact.v1`
/// manifest that names the same environment and the exact trajectory bytes.
fn verify_demonstration_provenance(
    files: &[GroupFile],
    content: &Content,
    identity: &Identity,
) -> Result<()> {
    let manifests: Vec<&GroupFile> = files
        .iter()
        .filter(|file| file.role == "dataset-manifest")
        .collect();
    if manifests.is_empty() {
        return Err(refusal(
            "a dataset group needs at least one glr.demonstration-artifact.v1 manifest".into(),
        ));
    }
    let mut bound: BTreeSet<String> = BTreeSet::new();
    for manifest in manifests {
        let artifact: DemonstrationArtifact = content.json(
            &manifest.path,
            INSPECTION_LIMIT,
            DEMONSTRATION_ARTIFACT_SCHEMA,
        )?;
        if artifact.environment_id != identity.environment_id {
            return Err(refusal(format!(
                "{} binds environment {} and cannot join a package for {}",
                manifest.path, artifact.environment_id, identity.environment_id
            )));
        }
        validate_identifier(&artifact.provenance.origin, "demonstration origin")
            .map_err(|error| refusal(format!("{}: {error}", manifest.path)))?;
        if !DEMONSTRATION_ORIGINS.contains(&artifact.provenance.origin.as_str()) {
            return Err(refusal(format!(
                "{} declares an unsupported demonstration origin {:?}",
                manifest.path, artifact.provenance.origin
            )));
        }
        if !DEMONSTRATION_OUTCOMES.contains(&artifact.provenance.outcome.as_str()) {
            return Err(refusal(format!(
                "{} declares an unsupported demonstration outcome {:?}",
                manifest.path, artifact.provenance.outcome
            )));
        }
        // A policy demonstration names the policy; any other origin must not.
        match (
            artifact.provenance.origin.as_str(),
            &artifact.provenance.policy_id,
        ) {
            ("policy", None) => {
                return Err(refusal(format!(
                    "{} is a policy demonstration and needs a policy_id",
                    manifest.path
                )));
            }
            ("policy", Some(policy_id)) => validate_identifier(policy_id, "policy_id")
                .map_err(|error| refusal(format!("{}: {error}", manifest.path)))?,
            (_, Some(_)) => {
                return Err(refusal(format!(
                    "{} forbids policy_id unless the origin is policy",
                    manifest.path
                )));
            }
            (_, None) => {}
        }
        uuid::Uuid::parse_str(&artifact.episode_id).map_err(|_| {
            refusal(format!(
                "{} has an episode_id that is not a UUID",
                manifest.path
            ))
        })?;
        let root = manifest
            .path
            .rsplit_once('/')
            .map(|(prefix, _)| prefix)
            .unwrap_or_default();
        let path = format!("{root}/{}", artifact.trajectory.path);
        let (size, sha256) = content.declared(&path).map_err(|_| {
            refusal(format!(
                "{} binds {path}, which the package omits",
                manifest.path
            ))
        })?;
        if size != artifact.trajectory.size_bytes || sha256 != artifact.trajectory.sha256 {
            return Err(refusal(format!(
                "{path} does not match the trajectory bytes {manifest_path} binds",
                manifest_path = manifest.path
            )));
        }
        bound.insert(path);
    }
    for file in files {
        if file.role == "dataset-manifest" {
            continue;
        }
        if file.role != "dataset-payload" {
            return Err(refusal(format!(
                "{} has an unknown dataset role {:?}",
                file.path, file.role
            )));
        }
        if !bound.contains(&file.path) {
            return Err(refusal(format!(
                "{} is a dataset payload that no demonstration manifest binds",
                file.path
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DemonstrationArtifact {
    /// Checked by [`Content::json`] against the expected contract before this
    /// struct is built; kept so the wire shape is written down in one place.
    #[allow(dead_code)]
    schema_version: String,
    environment_id: String,
    episode_id: String,
    trajectory: DemonstrationFile,
    provenance: DemonstrationProvenance,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DemonstrationFile {
    path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DemonstrationProvenance {
    origin: String,
    outcome: String,
    #[serde(default)]
    policy_id: Option<String>,
}

/// Freshness-aware `glr.knowledge-snapshot.v1` validation (ADR-0014).
///
/// A snapshot is refused when its schema, identifiers, item count or timestamp
/// fail, and when it is older than the declared freshness budget or stamped in
/// the future. Item semantics stay learner-owned; the package gate proves the
/// envelope is bounded, well-formed and fresh.
fn verify_knowledge(
    files: &[GroupFile],
    content: &Content,
    max_age_days: Option<u32>,
    now: SystemTime,
) -> Result<Vec<String>> {
    let Some(max_age_days) = max_age_days else {
        return Err(refusal(
            "a knowledge group must declare max_age_days so freshness is reviewable".into(),
        ));
    };
    if max_age_days == 0 {
        return Err(refusal("knowledge max_age_days must be positive".into()));
    }
    let now = now
        .duration_since(UNIX_EPOCH)
        .map_err(|_| refusal("system clock precedes the epoch".into()))?
        .as_secs();
    let now = i64::try_from(now).map_err(|_| refusal("system clock is out of range".into()))?;
    let budget = i64::from(max_age_days) * 86_400;
    for file in files {
        if file.role != "knowledge-snapshot" {
            return Err(refusal(format!(
                "{} has an unknown knowledge role {:?}",
                file.path, file.role
            )));
        }
        let snapshot: KnowledgeSnapshot =
            content.json(&file.path, INSPECTION_LIMIT, KNOWLEDGE_SNAPSHOT_SCHEMA)?;
        validate_identifier(&snapshot.snapshot_id, "knowledge snapshot_id")
            .map_err(|error| refusal(format!("{}: {error}", file.path)))?;
        validate_identifier(&snapshot.source_id, "knowledge source_id")
            .map_err(|error| refusal(format!("{}: {error}", file.path)))?;
        if snapshot.items.len() > MAX_KNOWLEDGE_ITEMS {
            return Err(refusal(format!(
                "{} carries {} items, over the {MAX_KNOWLEDGE_ITEMS} limit",
                file.path,
                snapshot.items.len()
            )));
        }
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for item in &snapshot.items {
            let item = item
                .as_object()
                .ok_or_else(|| refusal(format!("{} has a non-object item", file.path)))?;
            let identifier = item
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| refusal(format!("{} has an item without an id", file.path)))?;
            validate_identifier(identifier, "knowledge item id")
                .map_err(|error| refusal(format!("{}: {error}", file.path)))?;
            if !seen.insert(identifier) {
                return Err(refusal(format!(
                    "{} repeats knowledge item {identifier:?}",
                    file.path
                )));
            }
            let intent = item
                .get("intent")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !KNOWLEDGE_INTENTS.contains(&intent) {
                return Err(refusal(format!(
                    "{} has an item with unsupported intent {intent:?}",
                    file.path
                )));
            }
        }
        let created_at = parse_rfc3339(&snapshot.created_at).map_err(|_| {
            refusal(format!(
                "{} has a created_at that is not an RFC 3339 timestamp",
                file.path
            ))
        })?;
        let age = now - created_at;
        if age < 0 {
            return Err(refusal(format!("{} is stamped in the future", file.path)));
        }
        if age > budget {
            return Err(refusal(format!(
                "{} is {} days old, past the {} day freshness budget",
                file.path,
                age / 86_400,
                max_age_days
            )));
        }
    }
    Ok(vec![
        KNOWLEDGE_SNAPSHOT_SCHEMA.into(),
        format!("freshness<= {max_age_days}d"),
    ])
}

const MAX_KNOWLEDGE_ITEMS: usize = 256;
const KNOWLEDGE_INTENTS: &[&str] = &["acquire", "engage", "upgrade", "avoid"];
const DEMONSTRATION_ORIGINS: &[&str] = &["human", "scripted-expert", "policy", "unknown"];
const DEMONSTRATION_OUTCOMES: &[&str] = &["success", "failure", "neutral", "unknown"];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct KnowledgeSnapshot {
    /// Checked by [`Content::json`] before this struct is built.
    #[allow(dead_code)]
    schema_version: String,
    snapshot_id: String,
    source_id: String,
    created_at: String,
    items: Vec<Value>,
}

/// Seconds since the Unix epoch for an RFC 3339 timestamp with an offset.
///
/// Hand-rolled rather than pulled from a date crate: the package gate needs one
/// conversion, and import must not depend on a system locale or a timezone
/// database to decide whether a snapshot is fresh.
fn parse_rfc3339(value: &str) -> Result<i64> {
    let bytes = value.as_bytes();
    if bytes.len() < 20 || !value.is_ascii() {
        return Err(refusal("timestamp is not RFC 3339".into()));
    }
    let (date, rest) = value.split_at(10);
    if !is_calendar_date(date) {
        return Err(refusal("timestamp has no YYYY-MM-DD date".into()));
    }
    let rest = rest
        .strip_prefix('T')
        .or_else(|| rest.strip_prefix(' '))
        .ok_or_else(|| refusal("timestamp needs a T separator".into()))?;
    if rest.as_bytes().get(2) != Some(&b':') || rest.as_bytes().get(5) != Some(&b':') {
        return Err(refusal("timestamp needs HH:MM:SS".into()));
    }
    let year: i64 = date[0..4].parse().map_err(|_| numeric())?;
    let month: i64 = date[5..7].parse().map_err(|_| numeric())?;
    let day: i64 = date[8..10].parse().map_err(|_| numeric())?;
    let hour: i64 = rest[0..2].parse().map_err(|_| numeric())?;
    let minute: i64 = rest[3..5].parse().map_err(|_| numeric())?;
    let second: i64 = rest[6..8].parse().map_err(|_| numeric())?;
    if !(1..=12).contains(&month) || hour > 23 || minute > 59 || second > 59 {
        return Err(refusal("timestamp has an out-of-range field".into()));
    }
    if day < 1 || day > days_in_month(year, month) {
        return Err(refusal("timestamp has an out-of-range day".into()));
    }
    let mut rest = &rest[8..];
    let mut fraction = 0_i64;
    if let Some(stripped) = rest.strip_prefix('.') {
        let digits = stripped.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return Err(numeric());
        }
        let (whole, tail) = stripped.split_at(digits);
        fraction = whole.parse::<i64>().map_err(|_| numeric())?;
        for _ in 0..digits.saturating_sub(1) {
            fraction /= 10;
        }
        rest = tail;
    }
    let offset = match rest.as_bytes().first() {
        Some(b'Z') | Some(b'z') if rest.len() == 1 => 0,
        Some(b'+') | Some(b'-') => {
            if rest.len() != 6
                || rest.as_bytes().get(3) != Some(&b':')
                || !rest[1..3].chars().all(|c| c.is_ascii_digit())
                || !rest[4..6].chars().all(|c| c.is_ascii_digit())
            {
                return Err(refusal("timestamp has a malformed offset".into()));
            }
            let hours: i64 = rest[1..3].parse().map_err(|_| numeric())?;
            let minutes: i64 = rest[4..6].parse().map_err(|_| numeric())?;
            if hours > 23 || minutes > 59 {
                return Err(refusal("timestamp has an out-of-range offset".into()));
            }
            let offset = hours * 3600 + minutes * 60;
            if rest.as_bytes()[0] == b'-' {
                -offset
            } else {
                offset
            }
        }
        _ => return Err(refusal("timestamp needs a Z or ±HH:MM offset".into())),
    };
    Ok(
        days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second + fraction
            - offset,
    )
}

fn numeric() -> Error {
    refusal("timestamp has a non-numeric field".into())
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days from 1970-01-01 to a proleptic Gregorian date (Hinnant's algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let shifted = if month <= 2 { year - 1 } else { year };
    let era = shifted.div_euclid(400);
    let year_of_era = shifted - era * 400;
    let day_of_year = (153 * if month > 2 { month - 3 } else { month + 9 } + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Captured manifest bytes, keyed by package path.
    type Captured = BTreeMap<String, Vec<u8>>;
    /// Declared `(size, digest)` per package path.
    type Declared = BTreeMap<String, (u64, String)>;

    fn declared(entries: &[(&str, u64, &str)]) -> Declared {
        entries
            .iter()
            .map(|(path, size, sha)| ((*path).to_string(), (*size, (*sha).to_string())))
            .collect()
    }

    fn identity() -> Identity<'static> {
        Identity {
            environment_id: "synthetic.package",
            protocol_version: "1.0",
        }
    }

    fn now() -> SystemTime {
        UNIX_EPOCH + std::time::Duration::from_secs(1_800_000_000)
    }

    #[test]
    fn every_group_is_deny_by_default_until_declared() {
        let mut declared_groups = BTreeSet::new();
        declared_groups.insert(Group::Source);
        for (path, group) in [
            ("models/reference/manifest.json", Group::Model),
            ("data/demo/artifact.json", Group::Dataset),
            ("knowledge/snapshot.json", Group::Knowledge),
            ("reports/summary.json", Group::Report),
        ] {
            let error = group_of(path, &declared_groups).unwrap_err().to_string();
            assert!(
                error.contains(&format!(
                    "{} belongs to the {} group, which this package does not declare",
                    path,
                    group.as_str()
                )),
                "{error}"
            );
        }
        declared_groups.extend(Group::ALL);
        assert_eq!(
            group_of("models/reference/manifest.json", &declared_groups).unwrap(),
            Group::Model
        );
        assert_eq!(
            group_of("train.py", &declared_groups).unwrap(),
            Group::Source
        );
    }

    #[test]
    fn report_refuses_raw_logs_recordings_and_trajectories() {
        for path in [
            "reports/aggregate.json",
            "reports/summary.md",
            "reports/metrics.csv",
        ] {
            assert!(admit(Group::Report, path).is_ok(), "{path}");
        }
        for path in [
            "reports/run.log",
            "reports/session.mp4",
            "reports/trajectory.jsonl",
            "reports/logs/aggregate.json",
            "reports/trajectories/a.md",
            "reports/raw/metrics.csv",
            "reports/aggregate.yaml",
        ] {
            assert!(admit(Group::Report, path).is_err(), "{path}");
        }
    }

    #[test]
    fn model_refuses_deserializer_formats_and_foreign_roots() {
        for path in [
            "models/reference/manifest.json",
            "models/reference/artifacts/weights.safetensors",
            "models/reference/inputs/config.toml",
        ] {
            assert!(admit(Group::Model, path).is_ok(), "{path}");
        }
        for path in [
            "models/reference/artifacts/weights.pkl",
            "models/reference/inputs/data.npy",
            "models/reference/inputs/data.npz",
            "models/reference/artifacts/weights",
            "models/reference/artifacts/notes.pdf",
            "knowledge/reference/manifest.json",
        ] {
            assert!(admit(Group::Model, path).is_err(), "{path}");
        }
    }

    #[test]
    fn admission_refuses_local_overrides_and_denied_components_everywhere() {
        for group in Group::ALL {
            let root = group.root().unwrap_or("src");
            for path in [
                format!("{root}/config.local.json"),
                format!("{root}/cache/x.json"),
                format!("{root}/.hidden/x.json"),
            ] {
                assert!(admit(group, &path).is_err(), "{group:?} {path}");
            }
        }
    }

    #[test]
    fn rfc3339_offsets_and_calendar_edges_convert() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z").unwrap(), 0);
        assert_eq!(parse_rfc3339("1970-01-01T01:00:00+01:00").unwrap(), 0);
        assert_eq!(
            parse_rfc3339("2024-02-29T12:00:00Z").unwrap(),
            1_709_208_000
        );
        assert!(parse_rfc3339("2023-02-29T12:00:00Z").is_err());
        for value in [
            "1970-01-01",
            "1970-01-01T00:00:00",
            "1970-01-01 00:00:00+0000",
            "1970-13-01T00:00:00Z",
            "1970-01-01T24:00:00Z",
            "1970-01-01T00:00:60Z",
            "not-a-timestamp",
        ] {
            assert!(parse_rfc3339(value).is_err(), "{value}");
        }
    }

    fn knowledge_json(created_at: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": KNOWLEDGE_SNAPSHOT_SCHEMA,
            "snapshot_id": "snapshot.synthetic",
            "source_id": "source.synthetic",
            "created_at": created_at,
            "items": [{"id": "item.one", "intent": "acquire", "subject": "x", "summary": "y"}],
        }))
        .unwrap()
    }

    fn knowledge_selection() -> GroupSelection {
        GroupSelection {
            files: vec![GroupFile {
                path: "knowledge/snapshot.json".into(),
                role: "knowledge-snapshot".into(),
            }],
            max_age_days: Some(30),
        }
    }

    #[test]
    fn knowledge_freshness_gate_accepts_fresh_and_refuses_stale_or_future() {
        let captured: BTreeMap<String, Vec<u8>> = BTreeMap::from([(
            "knowledge/snapshot.json".to_string(),
            knowledge_json("2026-11-15T00:00:00Z"),
        )]);
        let declared = declared(&[("knowledge/snapshot.json", 10, "a".repeat(64).as_str())]);
        let content = Content::for_archive(&captured, &declared);
        let checks = verify_knowledge(
            &knowledge_selection().files,
            &content,
            Some(30),
            UNIX_EPOCH + std::time::Duration::from_secs(1_797_000_000),
        )
        .unwrap();
        assert_eq!(checks[0], KNOWLEDGE_SNAPSHOT_SCHEMA);

        // One day past the declared budget.
        let stale = knowledge_json("2026-01-01T00:00:00Z");
        let captured: BTreeMap<String, Vec<u8>> =
            BTreeMap::from([("knowledge/snapshot.json".to_string(), stale)]);
        let content = Content::for_archive(&captured, &declared);
        let error = verify_knowledge(&knowledge_selection().files, &content, Some(30), now())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("past the 30 day freshness budget"),
            "{error}"
        );

        // Stamped after the verification clock.
        let future = knowledge_json("2030-01-01T00:00:00Z");
        let captured: BTreeMap<String, Vec<u8>> =
            BTreeMap::from([("knowledge/snapshot.json".to_string(), future)]);
        let content = Content::for_archive(&captured, &declared);
        let error = verify_knowledge(&knowledge_selection().files, &content, Some(30), now())
            .unwrap_err()
            .to_string();
        assert!(error.contains("stamped in the future"), "{error}");

        // A freshness budget is mandatory: without one nothing is reviewable.
        let error = verify_knowledge(&knowledge_selection().files, &content, None, now())
            .unwrap_err()
            .to_string();
        assert!(error.contains("must declare max_age_days"), "{error}");
    }

    #[test]
    fn knowledge_refuses_unknown_intents_schema_and_duplicate_items() {
        let declared = declared(&[("knowledge/snapshot.json", 10, "a".repeat(64).as_str())]);
        for payload in [
            serde_json::json!({
                "schema_version": "glr.knowledge-snapshot.v2",
                "snapshot_id": "snapshot.synthetic",
                "source_id": "source.synthetic",
                "created_at": "2027-01-01T00:00:00Z",
                "items": [],
            }),
            serde_json::json!({
                "schema_version": KNOWLEDGE_SNAPSHOT_SCHEMA,
                "snapshot_id": "snapshot.synthetic",
                "source_id": "source.synthetic",
                "created_at": "2027-01-01T00:00:00Z",
                "items": [{"id": "item.one", "intent": "exploit", "subject": "x", "summary": "y"}],
            }),
            serde_json::json!({
                "schema_version": KNOWLEDGE_SNAPSHOT_SCHEMA,
                "snapshot_id": "snapshot.synthetic",
                "source_id": "source.synthetic",
                "created_at": "2027-01-01T00:00:00Z",
                "items": [
                    {"id": "item.one", "intent": "avoid", "subject": "x", "summary": "y"},
                    {"id": "item.one", "intent": "avoid", "subject": "x", "summary": "y"}
                ],
            }),
            serde_json::json!({
                "schema_version": KNOWLEDGE_SNAPSHOT_SCHEMA,
                "snapshot_id": "Snapshot.synthetic",
                "source_id": "source.synthetic",
                "created_at": "2027-01-01T00:00:00Z",
                "items": [],
            }),
        ] {
            let captured: BTreeMap<String, Vec<u8>> = BTreeMap::from([(
                "knowledge/snapshot.json".to_string(),
                serde_json::to_vec(&payload).unwrap(),
            )]);
            let content = Content::for_archive(&captured, &declared);
            assert!(
                verify_knowledge(&knowledge_selection().files, &content, Some(30), now()).is_err(),
                "{payload}"
            );
        }
    }

    fn bundle_manifest() -> Value {
        serde_json::json!({
            "schema_version": MODEL_BUNDLE_SCHEMA_VERSION,
            "environment_id": "synthetic.package",
            "protocol_version": "1.0",
            "algorithm": "ppo",
            "framework": "torch",
            "framework_version": "2.0.0",
            "seeds": [7],
            "inputs": [{"path": "config.json", "sha256": "b".repeat(64), "size_bytes": 4}],
            "artifacts": [{"path": "weights.safetensors", "sha256": "c".repeat(64), "size_bytes": 8}],
        })
    }

    fn model_selection() -> GroupSelection {
        GroupSelection {
            files: vec![
                GroupFile {
                    path: "models/reference/manifest.json".into(),
                    role: "model-manifest".into(),
                },
                GroupFile {
                    path: "models/reference/inputs/config.json".into(),
                    role: "model-input".into(),
                },
                GroupFile {
                    path: "models/reference/artifacts/weights.safetensors".into(),
                    role: "model-artifact".into(),
                },
            ],
            max_age_days: None,
        }
    }

    fn model_declared() -> BTreeMap<String, (u64, String)> {
        declared(&[
            (
                "models/reference/manifest.json",
                16,
                "a".repeat(64).as_str(),
            ),
            (
                "models/reference/inputs/config.json",
                4,
                "b".repeat(64).as_str(),
            ),
            (
                "models/reference/artifacts/weights.safetensors",
                8,
                "c".repeat(64).as_str(),
            ),
        ])
    }

    /// Captured bytes plus the declared inventory for one model bundle.
    fn model_content(manifest: Value) -> (Captured, Declared) {
        let declared = model_declared();
        let captured = BTreeMap::from([(
            "models/reference/manifest.json".to_string(),
            serde_json::to_vec(&manifest).unwrap(),
        )]);
        (captured, declared)
    }

    #[test]
    fn model_group_verifies_under_the_model_bundle_contract() {
        let (captured, declared) = model_content(bundle_manifest());
        let content = Content::for_archive(&captured, &declared);
        let (entries, checks) = verify(
            Group::Model,
            &model_selection().files,
            None,
            &content,
            &identity(),
            now(),
        )
        .unwrap();
        assert_eq!(checks, vec![MODEL_BUNDLE_SCHEMA_VERSION]);
        assert_eq!(entries.len(), 3);
        assert!(
            entries
                .iter()
                .all(|entry| entry.admitted_by == MODEL_BUNDLE_SCHEMA_VERSION)
        );
        // The audit receipt names every admitted file, weights included.
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.path.as_str())
                .collect::<Vec<_>>(),
            vec![
                "models/reference/manifest.json",
                "models/reference/inputs/config.json",
                "models/reference/artifacts/weights.safetensors"
            ]
        );
    }

    #[test]
    fn model_group_refuses_a_bundle_that_does_not_match_the_package() {
        let mut cases = Vec::new();
        for mutation in [
            json!({"schema_version": "glr.model-bundle.v2"}),
            json!({"environment_id": "other.environment"}),
            json!({"seeds": []}),
            json!({"artifacts": [
                {"path": "weights.safetensors", "sha256": "c".repeat(64), "size_bytes": 8},
                {"path": "missing.bin", "sha256": "d".repeat(64), "size_bytes": 1}
            ]}),
            json!({"inputs": [
                {"path": "config.json", "sha256": "e".repeat(64), "size_bytes": 4}
            ]}),
            json!({"inputs": [{"path": "../escape.json", "sha256": "b".repeat(64), "size_bytes": 4}]}),
        ] {
            let mut manifest = bundle_manifest();
            let object = manifest.as_object_mut().unwrap();
            for (key, value) in mutation.as_object().unwrap() {
                object.insert(key.clone(), value.clone());
            }
            cases.push(manifest);
        }
        for manifest in cases {
            let (captured, declared) = model_content(manifest);
            let content = Content::for_archive(&captured, &declared);
            let error = verify(
                Group::Model,
                &model_selection().files,
                None,
                &content,
                &identity(),
                now(),
            )
            .unwrap_err();
            assert!(error.to_string().contains("training package:"), "{error}");
        }
    }

    #[test]
    fn model_group_refuses_an_undeclared_file_and_a_missing_manifest() {
        let (captured, declared) = model_content(bundle_manifest());
        let content = Content::for_archive(&captured, &declared);

        let mut selection = model_selection();
        selection.files.push(GroupFile {
            path: "models/reference/artifacts/extra.bin".into(),
            role: "model-artifact".into(),
        });
        let error = verify(
            Group::Model,
            &selection.files,
            None,
            &content,
            &identity(),
            now(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("is not declared by the model bundle"),
            "{error}"
        );

        let only_input = GroupSelection {
            files: vec![GroupFile {
                path: "models/reference/inputs/config.json".into(),
                role: "model-input".into(),
            }],
            max_age_days: None,
        };
        let error = verify(
            Group::Model,
            &only_input.files,
            None,
            &content,
            &identity(),
            now(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exactly one model-manifest"), "{error}");
    }

    fn demonstration_manifest(sha: &str, size: u64, environment: &str) -> Value {
        json!({
            "schema_version": DEMONSTRATION_ARTIFACT_SCHEMA,
            "environment_id": environment,
            "episode_id": "00000000-0000-0000-0000-000000000001",
            "trajectory": {"path": "episode.jsonl", "sha256": sha, "size_bytes": size},
            "provenance": {"origin": "scripted-expert", "outcome": "success"},
        })
    }

    fn dataset_selection() -> GroupSelection {
        GroupSelection {
            files: vec![
                GroupFile {
                    path: "data/demo/artifact.json".into(),
                    role: "dataset-manifest".into(),
                },
                GroupFile {
                    path: "data/demo/episode.jsonl".into(),
                    role: "dataset-payload".into(),
                },
            ],
            max_age_days: None,
        }
    }

    fn dataset_declared() -> BTreeMap<String, (u64, String)> {
        declared(&[
            ("data/demo/artifact.json", 32, "a".repeat(64).as_str()),
            ("data/demo/episode.jsonl", 12, "d".repeat(64).as_str()),
            ("data/demo/orphan.jsonl", 12, "d".repeat(64).as_str()),
        ])
    }

    #[test]
    fn dataset_group_binds_every_payload_to_a_demonstration_manifest() {
        let declared = dataset_declared();
        let captured = BTreeMap::from([(
            "data/demo/artifact.json".to_string(),
            serde_json::to_vec(&demonstration_manifest(
                &"d".repeat(64),
                12,
                "synthetic.package",
            ))
            .unwrap(),
        )]);
        let content = Content::for_archive(&captured, &declared);
        let (entries, checks) = verify(
            Group::Dataset,
            &dataset_selection().files,
            None,
            &content,
            &identity(),
            now(),
        )
        .unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(checks[1], DEMONSTRATION_ARTIFACT_SCHEMA);
    }

    #[test]
    fn dataset_group_refuses_unbound_and_mismatched_provenance() {
        let declared = dataset_declared();
        for (payload, fragment) in [
            (
                demonstration_manifest(&"e".repeat(64), 12, "synthetic.package"),
                "does not match the trajectory bytes",
            ),
            (
                demonstration_manifest(&"d".repeat(64), 12, "other.environment"),
                "cannot join a package",
            ),
        ] {
            let captured = BTreeMap::from([(
                "data/demo/artifact.json".to_string(),
                serde_json::to_vec(&payload).unwrap(),
            )]);
            let content = Content::for_archive(&captured, &declared);
            let error = verify(
                Group::Dataset,
                &dataset_selection().files,
                None,
                &content,
                &identity(),
                now(),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(fragment), "{error}");
        }

        // A payload no manifest binds is refused, however plausible its name.
        let captured = BTreeMap::from([(
            "data/demo/artifact.json".to_string(),
            serde_json::to_vec(&demonstration_manifest(
                &"d".repeat(64),
                12,
                "synthetic.package",
            ))
            .unwrap(),
        )]);
        let content = Content::for_archive(&captured, &declared);
        let mut selection = dataset_selection();
        selection.files.push(GroupFile {
            path: "data/demo/orphan.jsonl".into(),
            role: "dataset-payload".into(),
        });
        let error = verify(
            Group::Dataset,
            &selection.files,
            None,
            &content,
            &identity(),
            now(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("no demonstration manifest binds"), "{error}");
    }

    #[test]
    fn dataset_allowlist_and_authorization_are_strict() {
        let files = dataset_selection().files;
        let allowlist = DatasetAllowlist {
            schema_version: DATASET_ALLOWLIST_SCHEMA.into(),
            reviewed_by: "synthetic.reviewer".into(),
            review_date: "2026-09-29".into(),
            entries: vec![
                "data/demo/artifact.json".into(),
                "data/demo/episode.jsonl".into(),
            ],
        };
        assert!(validate_dataset_allowlist(&allowlist, &files).is_ok());

        let mut missing = allowlist.clone();
        missing.entries.pop();
        assert!(
            validate_dataset_allowlist(&missing, &files)
                .unwrap_err()
                .to_string()
                .contains("not on the reviewed dataset allowlist")
        );
        for (schema_version, reviewed_by, review_date, entries) in [
            (
                "glr.dataset-allowlist.v2",
                "r",
                "2026-09-29",
                vec!["data/demo/episode.jsonl".to_string()],
            ),
            (
                DATASET_ALLOWLIST_SCHEMA,
                "",
                "2026-09-29",
                vec!["data/demo/episode.jsonl".to_string()],
            ),
            (
                DATASET_ALLOWLIST_SCHEMA,
                "r",
                "29-09-2026",
                vec!["data/demo/episode.jsonl".to_string()],
            ),
            (DATASET_ALLOWLIST_SCHEMA, "r", "2026-09-29", vec![]),
        ] {
            let allowlist = DatasetAllowlist {
                schema_version: schema_version.into(),
                reviewed_by: reviewed_by.into(),
                review_date: review_date.into(),
                entries,
            };
            assert!(validate_dataset_allowlist(&allowlist, &files).is_err());
        }

        let authorization = RedistributionAuthorization {
            schema_version: AUTHORIZATION_SCHEMA.into(),
            approver: "synthetic.approver".into(),
            scope: "synthetic redistribution review".into(),
            license: "MIT".into(),
            date: "2026-09-29".into(),
        };
        assert!(validate_authorization(&authorization).is_ok());
        for (schema_version, date) in [
            ("glr.redistribution-authorization.v2", "2026-09-29"),
            (AUTHORIZATION_SCHEMA, "2026-9-29"),
        ] {
            let authorization = RedistributionAuthorization {
                schema_version: schema_version.into(),
                approver: "a".into(),
                scope: "s".into(),
                license: "MIT".into(),
                date: date.into(),
            };
            assert!(validate_authorization(&authorization).is_err());
        }
    }
}
