//! Declared entry points and single-owner invariants (`entry-point-v1`).
//!
//! An unattended agent gets a fresh context every round and must therefore
//! answer a question the runtime never asks: *which command is live here?* This
//! module lets a project pin that answer in its manifest, and makes a run that
//! came through some other door visible (`entry_drift`) instead of silent.
//!
//! See ADR-0042. Every declaration here is optional; a project that declares
//! nothing behaves exactly as it did before this module existed.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::project::{Project, inside_project, validate_identifier, validate_text};

pub const SCHEMA_VERSION: &str = "glr.entry-point.v1";

/// How the invoking process claims to be the declared entry point.
pub const ENTRY_ID_ENV: &str = "GLR_ENTRY_ID";
/// Optional version claim; when present it must match the declared version.
pub const ENTRY_VERSION_ENV: &str = "GLR_ENTRY_VERSION";

/// Exit code for a run refused by a strict entry point, before any attach.
pub const ENTRY_DRIFT_REFUSED_EXIT_CODE: i32 = 79;

const MAX_INVARIANTS: usize = 32;
const MAX_DEPTH: usize = 32;
const MAX_SCANNED_FILES: usize = 20_000;
const MAX_TOTAL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// Directories that are never part of a project's own source tree. Skipping
/// them is what keeps a cold-cache scan inside its cost budget.
const SKIP_DIRECTORIES: &[&str] = &[".git", "target", "node_modules", ".venv", "__pycache__"];

/// The bounds one invariant scan is allowed to consume.
///
/// The scan is bounded rather than fast: cost grows with the tree, and the
/// bound is what keeps it inside a budget a pre-commit hook, a CI job and the
/// top of an unattended round can all afford. Exceeding a bound is `truncated`,
/// which fails closed, so a tree too large to verify never reads as verified.
///
/// The default is the documented contract. Tests inject smaller budgets so the
/// fail-closed paths are pinned without writing twenty thousand files.
#[derive(Debug, Clone, Copy)]
struct ScanBudget {
    max_files: usize,
    max_depth: usize,
    max_total_bytes: u64,
    max_file_bytes: u64,
}

impl Default for ScanBudget {
    fn default() -> Self {
        Self {
            max_files: MAX_SCANNED_FILES,
            max_depth: MAX_DEPTH,
            max_total_bytes: MAX_TOTAL_BYTES,
            max_file_bytes: MAX_FILE_BYTES,
        }
    }
}

/// One bounded search: what to look for, under which budget, and how to report
/// the paths it names.
struct Scan<'a> {
    /// The invariant root to walk.
    root: &'a Path,
    /// The project root, so every reported path stays project-relative.
    project_root: &'a Path,
    suffix: Option<&'a str>,
    marker: &'a str,
    /// Named in errors so a failure says which invariant could not be verified.
    invariant_id: &'a str,
    budget: ScanBudget,
}

/// The single-owner invariant declared by one `[[entry_point.invariants]]` entry.
///
/// Exactly one file under `root` may contain the literal `marker`. `marker` is a
/// case-sensitive substring, not a regular expression: the capability must import
/// nothing beyond the standard library so it stays cheap enough to never be
/// skipped.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvariantConfig {
    pub id: String,
    /// Project-relative directory to walk.
    pub root: String,
    /// Literal substring exactly one matching file must contain.
    pub marker: String,
    /// Optional filename suffix filter, for example `.py`.
    #[serde(default)]
    pub suffix: Option<String>,
}

/// The optional `[entry_point]` table of `glr-project.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntryPointConfig {
    pub schema_version: String,
    pub id: String,
    /// Provenance text describing the command. GLR never executes it.
    pub command: String,
    pub version: String,
    /// Refuse a drifting run before attach instead of only recording it.
    #[serde(default)]
    pub strict: bool,
    /// Single-owner invariants. Empty by default: pinning an entry point does
    /// not require asserting anything about the tree.
    #[serde(default)]
    pub invariants: Vec<InvariantConfig>,
}

impl EntryPointConfig {
    pub(crate) fn validate(&self, root: &Path) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            return Err(Error::Invalid(format!(
                "project.entry_point.schema_version must be {SCHEMA_VERSION:?}"
            )));
        }
        validate_identifier(&self.id, "project.entry_point.id")?;
        validate_text(&self.command, "project.entry_point.command")?;
        validate_text(&self.version, "project.entry_point.version")?;
        if self.invariants.len() > MAX_INVARIANTS {
            return Err(Error::Invalid(format!(
                "project.entry_point.invariants cannot exceed {MAX_INVARIANTS} entries"
            )));
        }
        let mut identifiers = BTreeSet::new();
        for invariant in &self.invariants {
            validate_identifier(&invariant.id, "project.entry_point.invariants[].id")?;
            if !identifiers.insert(invariant.id.as_str()) {
                return Err(Error::Invalid(format!(
                    "duplicate entry point invariant id: {}",
                    invariant.id
                )));
            }
            invariant.validate(root)?;
        }
        Ok(())
    }

    /// The declared entry, as recorded in run metadata and `doctor` output.
    pub fn declaration(&self) -> DeclaredEntry {
        DeclaredEntry {
            id: self.id.clone(),
            command: self.command.clone(),
            version: self.version.clone(),
            strict: self.strict,
        }
    }
}

impl InvariantConfig {
    fn validate(&self, root: &Path) -> Result<()> {
        validate_text(&self.marker, "project.entry_point.invariants[].marker")?;
        if self.suffix.as_deref().is_some_and(|suffix| {
            suffix.is_empty()
                || suffix.contains('/')
                || suffix.contains('\\')
                || suffix.chars().any(char::is_control)
        }) {
            return Err(Error::Invalid(
                "project.entry_point.invariants[].suffix must be a single filename suffix".into(),
            ));
        }
        let directory = inside_project(root, &self.root, "project.entry_point.invariants[].root")?;
        if !directory.is_dir() {
            return Err(Error::Missing(directory));
        }
        Ok(())
    }

    /// Walk the declared root and assert exactly one file carries the marker.
    fn check(&self, project_root: &Path) -> Result<InvariantResult> {
        self.check_with_budget(project_root, &ScanBudget::default())
    }

    /// Same walk under an explicit budget, so the bounds are testable without
    /// writing twenty thousand files.
    fn check_with_budget(
        &self,
        project_root: &Path,
        budget: &ScanBudget,
    ) -> Result<InvariantResult> {
        let directory = inside_project(
            project_root,
            &self.root,
            "project.entry_point.invariants[].root",
        )?;
        // Re-checked here and not only at load: a root deleted between load and
        // run start must not read as "no file declares the learner".
        if !directory.is_dir() {
            return Err(Error::Missing(directory));
        }
        let outcome = scan(&Scan {
            root: &directory,
            project_root,
            suffix: self.suffix.as_deref(),
            marker: &self.marker,
            invariant_id: &self.id,
            budget: *budget,
        })?;
        let matches = outcome
            .matches
            .iter()
            .map(|path| portable_path(project_root, path))
            .collect::<Vec<_>>();
        let skipped_files = outcome
            .too_large
            .iter()
            .map(|path| portable_path(project_root, path))
            .collect::<Vec<_>>();
        // A budget we could not finish is never "exactly one": fail closed, so
        // an unverified invariant cannot read as a satisfied one.
        let status = if outcome.truncated || !skipped_files.is_empty() {
            "truncated"
        } else {
            match matches.len() {
                1 => "ok",
                0 => "missing",
                _ => "multiple",
            }
        };
        Ok(InvariantResult {
            id: self.id.clone(),
            status,
            root: self.root.clone(),
            marker: self.marker.clone(),
            matches,
            skipped_files,
            scanned_files: outcome.scanned_files,
        })
    }
}

/// One invariant's verdict.
#[derive(Debug, Clone, Serialize)]
pub struct InvariantResult {
    pub id: String,
    /// `ok`, `missing`, `multiple`, or `truncated`.
    pub status: &'static str,
    pub root: String,
    pub marker: String,
    /// Every matching path, project-relative with forward slashes.
    pub matches: Vec<String>,
    /// Candidates too large to read, so the invariant could not be verified.
    pub skipped_files: Vec<String>,
    pub scanned_files: usize,
}

impl InvariantResult {
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }

    /// One-line, path-naming explanation used as the typed error message.
    pub fn summary(&self) -> String {
        match self.status {
            "ok" => format!("invariant {:?} holds in {}", self.id, self.matches[0]),
            "missing" => format!(
                "invariant {:?} found no file under {:?} containing marker {:?}",
                self.id, self.root, self.marker
            ),
            "multiple" => format!(
                "invariant {:?} expected exactly one file containing marker {:?} but found {} under {:?}: {}",
                self.id,
                self.marker,
                self.matches.len(),
                self.root,
                self.matches.join(", ")
            ),
            _ => format!(
                "invariant {:?} could not be verified: scan budget exhausted or candidates skipped ({})",
                self.id,
                self.skipped_files.join(", ")
            ),
        }
    }
}

/// The entry point a project declares.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeclaredEntry {
    pub id: String,
    pub command: String,
    pub version: String,
    pub strict: bool,
}

/// The entry the invoking process claims to be, read from the environment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservedEntry {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// The declared entry compared against the caller's claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attestation {
    /// `undeclared`, `matched`, or `entry_drift`.
    pub status: &'static str,
    pub declared: Option<DeclaredEntry>,
    pub observed: Option<ObservedEntry>,
}

/// Everything `doctor` reports and every run records about its entry point.
#[derive(Debug, Clone, Serialize)]
pub struct EntryPointReport {
    /// `undeclared`, `matched`, or `entry_drift`.
    pub status: &'static str,
    pub declared: Option<DeclaredEntry>,
    pub observed: Option<ObservedEntry>,
    pub strict: bool,
    pub invariants: Vec<InvariantResult>,
    /// Whether this report blocks a caller.
    ///
    /// Every invariant must hold. The attestation only blocks when the project
    /// declares `strict = true`: `doctor` is not a run and never carries
    /// `GLR_ENTRY_ID`, so gating on it there would fail every round of every
    /// project that pins an entry point. Non-strict drift stays visible in
    /// `status` and in the run record, and is never gated here.
    pub ready: bool,
    pub scanned_files: usize,
    pub elapsed_ms: u128,
}

impl EntryPointReport {
    /// Merge into a run's metadata so provenance needs no log parsing.
    pub fn as_metadata(&self) -> Result<Value> {
        Ok(json!({
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "declared": self.declared,
            "observed": self.observed,
            "strict": self.strict,
        }))
    }

    /// Non-empty when `doctor` must fail and a strict run must be refused.
    pub fn drift_summary(&self) -> Option<String> {
        if self.status != "entry_drift" {
            return None;
        }
        let declared = self.declared.as_ref()?;
        Some(match &self.observed {
            Some(observed) => format!(
                "run entered through entry {:?} but the project declares {:?} ({})",
                observed.id, declared.id, declared.command
            ),
            None => format!(
                "run claimed no entry point ({ENTRY_ID_ENV} unset) but the project declares {:?} ({})",
                declared.id, declared.command
            ),
        })
    }
}

/// Read the invoking process's claim from the environment.
pub fn observe() -> Option<ObservedEntry> {
    let id = std::env::var(ENTRY_ID_ENV).ok()?;
    let id = id.trim().to_owned();
    if id.is_empty() {
        return None;
    }
    let version = std::env::var(ENTRY_VERSION_ENV)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    Some(ObservedEntry { id, version })
}

/// Compare what the project declares against a supplied claim.
///
/// Absent provenance against a declared entry point *is* drift, not an unknown:
/// the runtime was told which door exists and the run did not come through it.
pub fn attest_with(project: &Project, observed: Option<ObservedEntry>) -> Attestation {
    let Some(config) = &project.entry_point else {
        return Attestation {
            status: "undeclared",
            declared: None,
            observed,
        };
    };
    let declared = config.declaration();
    let matched = observed.as_ref().is_some_and(|observed| {
        observed.id == declared.id
            && observed
                .version
                .as_ref()
                .is_none_or(|version| *version == declared.version)
    });
    Attestation {
        status: if matched { "matched" } else { "entry_drift" },
        declared: Some(declared),
        observed,
    }
}

/// Run the declared invariants and summarize the entry point.
pub fn report_with(project: &Project, observed: Option<ObservedEntry>) -> Result<EntryPointReport> {
    let started = Instant::now();
    let attestation = attest_with(project, observed);
    let mut invariants = Vec::new();
    let mut scanned_files = 0;
    if let Some(config) = &project.entry_point {
        for invariant in &config.invariants {
            let result = invariant.check(&project.root)?;
            scanned_files += result.scanned_files;
            invariants.push(result);
        }
    }
    let strict = project
        .entry_point
        .as_ref()
        .is_some_and(|config| config.strict);
    // `doctor` is not a run: it never carries `GLR_ENTRY_ID`, so counting the
    // attestation here would fail every round for every project that pins an
    // entry point. The attestation gates only when the project declared
    // `strict = true` — the same test the run-start gate applies — because then
    // the report is telling the truth a run would be refused.
    let attestation_gates = strict && attestation.status == "entry_drift";
    let ready = invariants.iter().all(InvariantResult::is_ok) && !attestation_gates;
    Ok(EntryPointReport {
        status: attestation.status,
        declared: attestation.declared,
        observed: attestation.observed,
        strict,
        invariants,
        ready,
        scanned_files,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// Same as [`report_with`], reading the claim from the environment.
pub fn report(project: &Project) -> Result<EntryPointReport> {
    report_with(project, observe())
}

/// Check the declared invariants at run start and return the report to record.
///
/// The filesystem work is trivial; the value is that this runs unconditionally
/// and that a violation is a typed error rather than a review comment.
pub fn enforce_with(
    project: &Project,
    observed: Option<ObservedEntry>,
) -> Result<EntryPointReport> {
    let report = report_with(project, observed)?;
    for invariant in &report.invariants {
        if !invariant.is_ok() {
            return Err(Error::Contract(format!(
                "{SCHEMA_VERSION}: {}",
                invariant.summary()
            )));
        }
    }
    Ok(report)
}

/// Same as [`enforce_with`], reading the claim from the environment.
pub fn enforce_at_run_start(project: &Project) -> Result<EntryPointReport> {
    enforce_with(project, observe())
}

#[derive(Debug, Default)]
struct ScanOutcome {
    matches: Vec<PathBuf>,
    too_large: Vec<PathBuf>,
    scanned_files: usize,
    truncated: bool,
}

/// Bounded, symlink-free, std-only walk over one invariant root.
fn scan(scan: &Scan<'_>) -> Result<ScanOutcome> {
    let Scan {
        root,
        project_root,
        suffix,
        marker,
        invariant_id,
        budget,
    } = scan;
    let marker = marker.as_bytes();
    let mut outcome = ScanOutcome::default();
    let mut total_bytes: u64 = 0;
    let mut stack = vec![(root.to_path_buf(), 0usize)];

    'walk: while let Some((directory, depth)) = stack.pop() {
        if outcome.scanned_files >= budget.max_files {
            outcome.truncated = true;
            break;
        }
        // A directory the OS will not list is a typed contract error naming
        // the invariant and the path, never a silent skip. Skipping it would
        // let the walk report "exactly one" over a subtree it never saw, which
        // is the one thing this module exists to refuse (see
        // [`InvariantConfig::check_with_budget`]).
        let entries = match fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) => {
                return Err(Error::Contract(format!(
                    "{SCHEMA_VERSION}: invariant {invariant_id:?} could not list {}: {error}",
                    portable_path(project_root, &directory)
                )));
            }
        };
        for entry in entries.flatten() {
            if outcome.scanned_files >= budget.max_files {
                outcome.truncated = true;
                break 'walk;
            }
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => continue,
            };
            // Never follow a symlink: the walk cannot loop or leave the root.
            if file_type.is_symlink() {
                continue;
            }
            let path = entry.path();
            if file_type.is_dir() {
                if depth + 1 > budget.max_depth {
                    outcome.truncated = true;
                    continue;
                }
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default();
                if SKIP_DIRECTORIES.contains(&name) {
                    continue;
                }
                stack.push((path, depth + 1));
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            outcome.scanned_files += 1;
            if suffix.is_some_and(|suffix| !path.to_string_lossy().ends_with(suffix)) {
                continue;
            }
            let size = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            if size > budget.max_file_bytes {
                outcome.too_large.push(path);
                continue;
            }
            if total_bytes + size > budget.max_total_bytes {
                outcome.truncated = true;
                break 'walk;
            }
            total_bytes += size;
            // A file that vanished between `read_dir` and `read`, or one the OS
            // will not open, is a typed contract error naming the invariant and
            // the path — never a bare OS error from inside the walk.
            let bytes = fs::read(&path).map_err(|error| {
                Error::Contract(format!(
                    "{SCHEMA_VERSION}: invariant {invariant_id:?} could not read {}: {error}",
                    portable_path(project_root, &path)
                ))
            })?;
            if contains(&bytes, marker) {
                outcome.matches.push(path);
            }
        }
    }

    outcome.matches.sort();
    outcome.too_large.sort();
    Ok(outcome)
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn portable_path(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use serde_json::json;
    use tempfile::TempDir;

    use super::{
        ENTRY_DRIFT_REFUSED_EXIT_CODE, InvariantConfig, Scan, ScanBudget, attest_with, contains,
        enforce_with, report_with, scan,
    };

    const MANIFEST: &str = r#"
schema_version = "glr.project.v1"
environment_id = "example.environment-v1"
environment_family = "example"
protocol_version = "1.0"
data_dir = "data"
bridge_path = "bridge.json"

[runtime]
argv = ["python", "-c", "pass"]

[trainer]
argv = ["python", "-c", "pass"]

[player]
argv = ["python", "-c", "pass"]
"#;

    /// Writes a real project manifest so parsing is exercised end to end.
    fn project_with(entry_point: &str) -> (TempDir, crate::project::Project) {
        project_with_tree(entry_point, &[])
    }

    /// Same, with a source tree written before the project loads. A declared
    /// invariant root must already exist, so the tree cannot come afterwards.
    fn project_with_tree(
        entry_point: &str,
        tree: &[(&str, &str)],
    ) -> (TempDir, crate::project::Project) {
        let root = TempDir::new().expect("tempdir is creatable");
        fs::write(root.path().join("bridge.json"), "{}").expect("bridge is writable");
        for (relative, body) in tree {
            write(root.path(), relative, body);
        }
        fs::write(
            root.path().join("glr-project.toml"),
            format!("{MANIFEST}\n{entry_point}\n"),
        )
        .expect("manifest is writable");
        let project = crate::project::load_project(root.path()).expect("project loads");
        (root, project)
    }

    fn write(root: &Path, relative: &str, body: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("file has a parent")).expect("directory creatable");
        fs::write(path, body).expect("file is writable");
    }

    fn claimed(id: &str, version: Option<&str>) -> Option<super::ObservedEntry> {
        Some(super::ObservedEntry {
            id: id.into(),
            version: version.map(str::to_owned),
        })
    }

    /// A scan over one directory, reported relative to that same directory.
    fn scan_of<'a>(root: &'a Path, budget: ScanBudget) -> Scan<'a> {
        Scan {
            root,
            project_root: root,
            suffix: Some(".py"),
            marker: "class Learner",
            invariant_id: "single-learner",
            budget,
        }
    }

    /// A budget with one field shrunk, so a bound is reachable in a test.
    fn budget_with_files(max_files: usize) -> ScanBudget {
        ScanBudget {
            max_files,
            ..ScanBudget::default()
        }
    }

    #[test]
    fn substring_search_is_literal_and_total() {
        assert!(contains(b"class Learner:", b"class Learner"));
        assert!(!contains(b"class learner:", b"class Learner"));
        assert!(!contains(b"", b"class Learner"));
        assert!(!contains(b"ab", b"abc"));
        assert!(contains(b"a", b"a"));
    }

    #[test]
    fn exactly_one_match_is_ok() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/learner.py", "class Learner:\n    pass\n");
        write(
            root.path(),
            "src/trainer.py",
            "from learner import helper\n",
        );
        let result = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check(root.path())
        .expect("check succeeds");
        assert_eq!(result.status, "ok");
        assert_eq!(result.matches, vec!["src/learner.py"]);
        assert!(result.is_ok());
    }

    #[test]
    fn two_definitions_fail_and_name_both_paths() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/a/learner.py", "class Learner: pass\n");
        write(
            root.path(),
            "src/b/legacy_learner.py",
            "class Learner: pass\n",
        );
        write(root.path(), "src/b/util.py", "unrelated\n");
        let result = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check(root.path())
        .expect("check succeeds");
        assert_eq!(result.status, "multiple");
        assert_eq!(
            result.matches,
            vec!["src/a/learner.py", "src/b/legacy_learner.py"]
        );
        let summary = result.summary();
        assert!(summary.contains("src/a/learner.py"), "{summary}");
        assert!(summary.contains("src/b/legacy_learner.py"), "{summary}");
    }

    #[test]
    fn no_match_is_missing() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/util.py", "unrelated\n");
        let result = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check(root.path())
        .expect("check succeeds");
        assert_eq!(result.status, "missing");
        assert!(result.matches.is_empty());
    }

    #[test]
    fn suffix_filter_restricts_candidates() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/notes.txt", "class Learner");
        write(root.path(), "src/learner.py", "class Learner");
        let result = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check(root.path())
        .expect("check succeeds");
        assert_eq!(result.status, "ok");
        assert_eq!(result.matches, vec!["src/learner.py"]);
    }

    #[test]
    fn scan_skips_excluded_directories() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "tree/learner.py", "class Learner");
        write(root.path(), "tree/.git/learner.py", "class Learner");
        write(root.path(), "tree/target/learner.py", "class Learner");
        write(root.path(), "tree/node_modules/learner.py", "class Learner");
        let outcome = scan(&scan_of(&root.path().join("tree"), ScanBudget::default()))
            .expect("scan succeeds");
        assert_eq!(outcome.matches.len(), 1, "{outcome:?}");
        assert!(!outcome.truncated);
    }

    #[test]
    fn an_oversized_candidate_fails_closed() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/learner.py", "class Learner");
        write(
            root.path(),
            "src/huge.py",
            &"x".repeat((super::MAX_FILE_BYTES as usize) + 1),
        );
        let result = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check(root.path())
        .expect("check succeeds");
        assert_eq!(result.status, "truncated");
        assert_eq!(result.skipped_files, vec!["src/huge.py"]);
        assert!(!result.is_ok());
    }

    #[test]
    fn the_file_budget_fails_closed() {
        let root = TempDir::new().expect("tempdir");
        for index in 0..3 {
            write(
                root.path(),
                &format!("src/module_{index}.py"),
                "unrelated contents\n",
            );
        }
        write(root.path(), "src/learner.py", "class Learner");
        let result = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check_with_budget(root.path(), &budget_with_files(3))
        .expect("check succeeds");
        // The fourth candidate was never scanned, so the tree is unverified
        // even if the marker was already seen: fail closed.
        assert_eq!(result.status, "truncated");
        assert!(!result.is_ok(), "{result:?}");
        assert_eq!(result.scanned_files, 3);
    }

    #[test]
    fn the_total_bytes_budget_fails_closed() {
        let root = TempDir::new().expect("tempdir");
        let body = "x".repeat(4 * 1024);
        for index in 0..4 {
            write(root.path(), &format!("src/module_{index}.py"), &body);
        }
        let result = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check_with_budget(
            root.path(),
            &ScanBudget {
                max_total_bytes: 8 * 1024,
                ..ScanBudget::default()
            },
        )
        .expect("check succeeds");
        assert_eq!(result.status, "truncated");
        assert!(result.scanned_files < 4, "{result:?}");
    }

    #[test]
    fn the_depth_budget_fails_closed() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/a/learner.py", "class Learner");
        write(root.path(), "src/a/b/deeper_learner.py", "class Learner");
        let result = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check_with_budget(
            root.path(),
            &ScanBudget {
                max_depth: 1,
                ..ScanBudget::default()
            },
        )
        .expect("check succeeds");
        // `src/a/b` was never entered, so the second definition was never seen.
        assert_eq!(result.status, "truncated");
        assert!(!result.is_ok(), "{result:?}");
    }

    #[test]
    fn the_default_budget_is_the_documented_contract() {
        let budget = ScanBudget::default();
        assert_eq!(budget.max_files, super::MAX_SCANNED_FILES);
        assert_eq!(budget.max_depth, super::MAX_DEPTH);
        assert_eq!(budget.max_total_bytes, super::MAX_TOTAL_BYTES);
        assert_eq!(budget.max_file_bytes, super::MAX_FILE_BYTES);
    }

    /// Makes `path` unreadable for as long as the returned guard is held.
    ///
    /// Unix denies the read bit and has nothing to hold, so it returns `None`;
    /// Windows holds a handle with share mode 0, so no other open can succeed
    /// while it is held. Both arms return the same type: the caller binds one
    /// guard either way, and neither platform gets a `let_unit_value` or a
    /// `dropping_copy_types` lint that the other cannot see.
    #[cfg(unix)]
    fn deny_read(path: &Path) -> Option<fs::File> {
        use std::os::unix::fs::PermissionsExt;

        fs::set_permissions(path, fs::Permissions::from_mode(0o000)).expect("deniable");
        None
    }

    #[cfg(windows)]
    fn deny_read(path: &Path) -> Option<fs::File> {
        use std::os::windows::fs::OpenOptionsExt;

        // Share mode 0: every other open of this file fails while it is held.
        Some(
            fs::OpenOptions::new()
                .read(true)
                .share_mode(0)
                .open(path)
                .expect("an exclusive handle is openable"),
        )
    }

    /// A candidate the OS will not open must be a typed contract error naming
    /// the invariant and the path, not a bare OS error from inside the walk.
    #[test]
    fn an_unreadable_candidate_is_a_typed_contract_error() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/learner.py", "class Learner");
        let unreadable = root.path().join("src/locked.py");
        fs::write(&unreadable, "class Learner").expect("file is writable");

        let lock = deny_read(&unreadable);
        if fs::read(&unreadable).is_ok() {
            // A privileged process can read anything, so there is no failure
            // to assert on. Skipping beats a false red.
            return;
        }

        let error = scan(&Scan {
            root: &root.path().join("src"),
            project_root: root.path(),
            suffix: Some(".py"),
            marker: "class Learner",
            invariant_id: "single-learner",
            budget: ScanBudget::default(),
        })
        .expect_err("an unreadable candidate must fail");
        drop(lock);
        let message = error.to_string();
        assert!(
            matches!(error, crate::error::Error::Contract(_)),
            "the failure must be typed, not a bare OS error: {message}"
        );
        assert!(message.contains("single-learner"), "{message}");
        assert!(
            message.contains("src/locked.py"),
            "the path is project-relative, as every other path in the report is: {message}"
        );
    }

    /// Puts back whatever [`deny_list`] took away.
    ///
    /// One type on both platforms, so the test body that binds it is identical
    /// and neither platform gets a lint the other cannot see. The handle is
    /// released before the mode bits go back, because on Windows the handle is
    /// the only thing denying access.
    struct DenyListGuard {
        /// Windows: the exclusive handle that stops the directory being listed.
        handle: Option<fs::File>,
        /// Unix: the directory whose mode bits must be restored, so the
        /// temporary tree can still be torn down afterwards.
        #[cfg(unix)]
        path: Option<std::path::PathBuf>,
    }

    impl Drop for DenyListGuard {
        fn drop(&mut self) {
            drop(self.handle.take());
            self.restore_mode();
        }
    }

    impl DenyListGuard {
        #[cfg(unix)]
        fn restore_mode(&mut self) {
            use std::os::unix::fs::PermissionsExt;

            if let Some(path) = self.path.take() {
                fs::set_permissions(path, fs::Permissions::from_mode(0o755)).ok();
            }
        }

        /// Windows denies access with the handle alone, so there is no mode to
        /// restore.
        #[cfg(windows)]
        fn restore_mode(&mut self) {}
    }

    /// Makes `path` unlistable for as long as the returned guard is held.
    ///
    /// Unix denies every bit on the directory; Windows holds a directory
    /// handle with share mode 0, which is what makes a later listing fail with
    /// a sharing violation. `FILE_FLAG_BACKUP_SEMANTICS` is what lets a
    /// directory be opened at all.
    #[cfg(unix)]
    fn deny_list(path: &Path) -> DenyListGuard {
        use std::os::unix::fs::PermissionsExt;

        fs::set_permissions(path, fs::Permissions::from_mode(0o000)).expect("deniable");
        DenyListGuard {
            handle: None,
            path: Some(path.to_path_buf()),
        }
    }

    #[cfg(windows)]
    fn deny_list(path: &Path) -> DenyListGuard {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        DenyListGuard {
            handle: Some(
                fs::OpenOptions::new()
                    .read(true)
                    .share_mode(0)
                    .attributes(FILE_FLAG_BACKUP_SEMANTICS)
                    .open(path)
                    .expect("an exclusive directory handle is openable"),
            ),
        }
    }

    /// A directory the walk cannot list must be a typed contract error naming
    /// the invariant and the path, not a silent skip.
    #[test]
    fn an_unlistable_directory_is_a_typed_contract_error() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/learner.py", "class Learner");
        write(root.path(), "src/hidden/copy_learner.py", "class Learner");
        let hidden = root.path().join("src/hidden");

        let guard = deny_list(&hidden);
        if fs::read_dir(&hidden).is_ok() {
            // A privileged process can list anything, so there is no failure
            // to assert on. Skipping beats a false red.
            return;
        }

        let error = scan(&Scan {
            root: &root.path().join("src"),
            project_root: root.path(),
            suffix: Some(".py"),
            marker: "class Learner",
            invariant_id: "single-learner",
            budget: ScanBudget::default(),
        })
        .expect_err("an unlistable directory must fail");
        drop(guard);
        let message = error.to_string();
        assert!(
            matches!(error, crate::error::Error::Contract(_)),
            "the failure must be typed, not a bare OS error: {message}"
        );
        assert!(message.contains("single-learner"), "{message}");
        assert!(
            message.contains("src/hidden"),
            "the path is project-relative, as every other path in the report is: {message}"
        );
    }

    /// The invariant survives a directory it could not read: it refuses
    /// instead of reporting exactly one learner over a subtree it never saw.
    #[test]
    fn an_unlistable_directory_never_reports_ok() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/learner.py", "class Learner");
        write(root.path(), "src/hidden/copy_learner.py", "class Learner");
        let hidden = root.path().join("src/hidden");

        let guard = deny_list(&hidden);
        if fs::read_dir(&hidden).is_ok() {
            return;
        }

        let checked = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check(root.path());
        drop(guard);
        // The scan saw `src/learner.py` and nothing else, so a silent skip
        // would have read as a satisfied invariant.
        let error = checked.expect_err("an unverified tree must not read as verified");
        assert!(
            matches!(error, crate::error::Error::Contract(_)),
            "the invariant must refuse, not report ok: {error}"
        );
    }

    /// The invariant root is the one directory whose failure changes the
    /// verdict a reader sees: an unlistable root is "could not be evaluated",
    /// not "no file declares the learner". Reporting `missing` there would tell
    /// an agent the project simply has no entry point, and `doctor` would exit
    /// `4` on a tree nothing ever looked at.
    #[test]
    fn an_unlistable_root_does_not_read_as_missing() {
        let root = TempDir::new().expect("tempdir");
        write(root.path(), "src/learner.py", "class Learner");
        let declared = root.path().join("src");

        let guard = deny_list(&declared);
        if fs::read_dir(&declared).is_ok() {
            return;
        }

        let checked = InvariantConfig {
            id: "single-learner".into(),
            root: "src".into(),
            marker: "class Learner".into(),
            suffix: Some(".py".into()),
        }
        .check(root.path());
        drop(guard);
        let error = checked.expect_err("an unlistable root is not an absent declaration");
        assert!(
            matches!(error, crate::error::Error::Contract(_)),
            "the root's verdict is an evaluation failure, not `missing`: {error}"
        );
        assert!(error.to_string().contains("src"), "{error}");
    }

    #[test]
    fn an_absent_root_is_a_typed_error() {
        let root = TempDir::new().expect("tempdir");
        let error = InvariantConfig {
            id: "single-learner".into(),
            root: "missing".into(),
            marker: "class Learner".into(),
            suffix: None,
        }
        .check(root.path())
        .expect_err("missing root must fail");
        assert!(
            matches!(error, crate::error::Error::Missing(_)),
            "{error:?}"
        );
    }

    #[test]
    fn a_project_declaring_nothing_is_undeclared_and_ready() {
        let (_root, project) = project_with("");
        assert!(project.entry_point.is_none());
        let report = report_with(&project, claimed("anything", None)).expect("report succeeds");
        assert_eq!(report.status, "undeclared");
        assert!(report.declared.is_none());
        assert!(report.invariants.is_empty());
        assert!(report.ready, "absent capability must not gate anything");
        assert!(!report.strict);
    }

    #[test]
    fn a_matching_claim_is_matched() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
"#,
        );
        let report =
            report_with(&project, claimed("campaign-driver", Some("1.4.0"))).expect("report");
        assert_eq!(report.status, "matched");
        assert!(report.ready);
        assert_eq!(
            report.declared.as_ref().map(|entry| entry.command.clone()),
            Some("python -m campaign.driver".to_owned())
        );
    }

    #[test]
    fn a_different_entry_is_entry_drift() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
"#,
        );
        let report = report_with(&project, claimed("legacy-pixel-capture", None)).expect("report");
        assert_eq!(report.status, "entry_drift");
        // Drift is recorded; only a strict project lets it block a caller.
        assert!(report.ready, "non-strict drift is not current unreadiness");
        let summary = report.drift_summary().expect("drift is summarized");
        assert!(summary.contains("legacy-pixel-capture"), "{summary}");
        assert!(summary.contains("campaign-driver"), "{summary}");
    }

    #[test]
    fn absent_provenance_against_a_declared_entry_is_drift() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
"#,
        );
        let report = report_with(&project, None).expect("report");
        assert_eq!(report.status, "entry_drift");
        assert!(report.ready, "non-strict drift is not current unreadiness");
        assert!(
            report
                .drift_summary()
                .expect("drift summary")
                .contains("GLR_ENTRY_ID")
        );
    }

    /// `doctor` is not a run: it never carries `GLR_ENTRY_ID`. Counting the
    /// attestation there failed every round of every project that pins an
    /// entry point, which is the opposite of what the capability is for.
    #[test]
    fn a_non_strict_attestation_is_reported_without_gating() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
"#,
        );
        let report = report_with(&project, None).expect("report");
        assert_eq!(report.status, "entry_drift");
        assert!(!report.strict);
        assert!(report.ready);
    }

    /// A strict project refuses the run at the gate, so the attestation is a
    /// fact about the project and does belong in the verdict.
    #[test]
    fn a_strict_attestation_gates() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
strict = true
"#,
        );
        let drifted = report_with(&project, None).expect("report");
        assert_eq!(drifted.status, "entry_drift");
        assert!(!drifted.ready, "a strict project refuses the run itself");
        let matched =
            report_with(&project, claimed("campaign-driver", Some("1.4.0"))).expect("report");
        assert!(matched.ready);
    }

    #[test]
    fn a_claimed_version_mismatch_is_drift() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
"#,
        );
        let report =
            report_with(&project, claimed("campaign-driver", Some("1.3.0"))).expect("report");
        assert_eq!(report.status, "entry_drift");
    }

    #[test]
    fn an_unclaimed_version_still_matches_on_id() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
"#,
        );
        let report = report_with(&project, claimed("campaign-driver", None)).expect("report");
        assert_eq!(report.status, "matched");
    }

    #[test]
    fn strict_mode_is_recorded_in_the_report() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
strict = true
"#,
        );
        let report = report_with(&project, None).expect("report");
        assert!(report.strict);
        assert_eq!(report.status, "entry_drift");
        assert!(
            !report.ready,
            "a strict project refuses the run, so the verdict must say so"
        );
    }

    #[test]
    fn a_duplicated_learner_fails_the_run_start_gate() {
        let (_root, project) = project_with_tree(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"

[[entry_point.invariants]]
id = "single-learner"
root = "src"
suffix = ".py"
marker = "class Learner"
"#,
            &[
                ("src/a/learner.py", "class Learner: pass\n"),
                ("src/b/copy_learner.py", "class Learner: pass\n"),
            ],
        );
        let error = enforce_with(&project, claimed("campaign-driver", Some("1.4.0")))
            .expect_err("two learners must fail");
        let message = error.to_string();
        assert!(message.contains("src/a/learner.py"), "{message}");
        assert!(message.contains("src/b/copy_learner.py"), "{message}");
    }

    #[test]
    fn a_single_learner_passes_the_run_start_gate() {
        let (_root, project) = project_with_tree(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"

[[entry_point.invariants]]
id = "single-learner"
root = "src"
suffix = ".py"
marker = "class Learner"
"#,
            &[("src/learner.py", "class Learner: pass\n")],
        );
        let report = enforce_with(&project, claimed("campaign-driver", Some("1.4.0")))
            .expect("one learner passes");
        assert!(report.ready);
        assert_eq!(report.invariants[0].status, "ok");
    }

    #[test]
    fn the_report_metadata_carries_the_declared_entry() {
        let (_root, project) = project_with(
            r#"
[entry_point]
schema_version = "glr.entry-point.v1"
id = "campaign-driver"
command = "python -m campaign.driver"
version = "1.4.0"
"#,
        );
        let report =
            report_with(&project, claimed("campaign-driver", Some("1.4.0"))).expect("report");
        let metadata = report.as_metadata().expect("metadata serializes");
        assert_eq!(metadata["status"], json!("matched"));
        assert_eq!(metadata["declared"]["id"], json!("campaign-driver"));
        // The declared entry is visible without parsing any log.
        assert_eq!(
            metadata["declared"]["command"],
            json!("python -m campaign.driver")
        );
    }

    #[test]
    fn an_invalid_schema_version_is_rejected_at_load() {
        let root = TempDir::new().expect("tempdir");
        fs::write(root.path().join("bridge.json"), "{}").expect("bridge");
        fs::write(
            root.path().join("glr-project.toml"),
            format!(
                "{MANIFEST}\n[entry_point]\nschema_version = \"glr.entry-point.v9\"\nid = \"a\"\ncommand = \"b\"\nversion = \"1\"\n"
            ),
        )
        .expect("manifest");
        let error = crate::project::load_project(root.path()).expect_err("bad schema must fail");
        assert!(error.to_string().contains("glr.entry-point.v1"), "{error}");
    }

    #[test]
    fn an_invariant_root_outside_the_project_is_rejected_at_load() {
        let root = TempDir::new().expect("tempdir");
        fs::write(root.path().join("bridge.json"), "{}").expect("bridge");
        fs::write(
            root.path().join("glr-project.toml"),
            format!(
                "{MANIFEST}\n[entry_point]\nschema_version = \"glr.entry-point.v1\"\nid = \"a\"\ncommand = \"b\"\nversion = \"1\"\n\n[[entry_point.invariants]]\nid = \"x\"\nroot = \"../elsewhere\"\nmarker = \"m\"\n"
            ),
        )
        .expect("manifest");
        let error = crate::project::load_project(root.path()).expect_err("escape must fail");
        assert!(
            error.to_string().contains("portable project-relative path"),
            "{error}"
        );
    }

    #[test]
    fn the_attestation_shape_is_stable() {
        let (_root, project) = project_with("");
        let attestation = attest_with(&project, claimed("x", None));
        assert_eq!(attestation.status, "undeclared");
        assert!(attestation.declared.is_none());
    }

    #[test]
    fn the_refusal_exit_code_is_stable() {
        assert_eq!(ENTRY_DRIFT_REFUSED_EXIT_CODE, 79);
    }
}
