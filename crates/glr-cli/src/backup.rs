//! Online SQLite snapshot plus checksum-verified, immutable completed-run files.
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::args::BackupCommand;
use crate::contracts::sha256_file;
use crate::error::{Error, Result};
use crate::observation::safe_child;
use crate::project::{load_project, validate_identifier};

const SCHEMA: &str = "glr.observation-backup.v1";
const MANIFEST: &str = "backup-manifest.json";
const MAX_FILES: usize = 100_000;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BackupFile {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    environment_id: String,
    created_at_unix_ms: u128,
    consistency: String,
    active_runs_database_only: Vec<String>,
    files: Vec<BackupFile>,
}

pub fn execute(project: &Path, command: &BackupCommand) -> Result<Value> {
    match command {
        BackupCommand::Create { output } => create(project, output),
        BackupCommand::Verify { archive } => verify(archive),
        BackupCommand::Restore { archive, output } => {
            verify(archive)?;
            let (staging, target) = staging(output)?;
            let manifest = read_manifest(archive)?;
            for file in &manifest.files {
                copy_file(
                    &safe_child(archive, Path::new(&file.path))?,
                    &staging.path().join(&file.path),
                )?;
            }
            copy_file(&archive.join(MANIFEST), &staging.path().join(MANIFEST))?;
            verify(staging.path())?;
            crate::filesystem::promote(staging.path(), &target)?;
            Ok(
                json!({"schema_version": SCHEMA, "restored": target, "overwrote_existing": false,
                "active_runs_database_only": manifest.active_runs_database_only}),
            )
        }
    }
}

fn staging(output: &Path) -> Result<(tempfile::TempDir, PathBuf)> {
    let target = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()?.join(output)
    };
    if target.exists() || target.is_symlink() {
        return Err(Error::Invalid("backup destination must not exist".into()));
    }
    let parent = target
        .parent()
        .ok_or_else(|| Error::Invalid("backup destination needs a parent".into()))?;
    fs::create_dir_all(parent)?;
    safe_child(parent, Path::new("backup-check"))?;
    let parent = fs::canonicalize(parent)?;
    let target = parent.join(
        target
            .file_name()
            .ok_or_else(|| Error::Invalid("backup destination must name a directory".into()))?,
    );
    Ok((
        tempfile::Builder::new()
            .prefix(".glr-backup-")
            .tempdir_in(&parent)?,
        target,
    ))
}

fn read_manifest(root: &Path) -> Result<Manifest> {
    let path = safe_child(root, Path::new(MANIFEST))?;
    if path.metadata()?.len() > 32 * 1024 * 1024 {
        return Err(Error::Invalid("backup manifest too large".into()));
    }
    let manifest: Manifest = serde_json::from_slice(&fs::read(path)?)?;
    if manifest.schema_version != SCHEMA || manifest.files.len() > MAX_FILES {
        return Err(Error::Contract(
            "unsupported or oversized backup manifest".into(),
        ));
    }
    Ok(manifest)
}

pub fn verify(root: &Path) -> Result<Value> {
    let manifest = read_manifest(root)?;
    let mut paths = std::collections::HashSet::new();
    for file in &manifest.files {
        if !paths.insert(&file.path) || file.path == MANIFEST {
            return Err(Error::Invalid("duplicate or reserved backup path".into()));
        }
        let path = safe_child(root, Path::new(&file.path))?;
        if !path.is_file()
            || path.metadata()?.len() != file.size_bytes
            || sha256_file(&path)? != file.sha256
        {
            return Err(Error::Contract(format!(
                "backup checksum mismatch: {}",
                file.path
            )));
        }
    }
    if !paths.contains(&"runs.sqlite3".to_string()) {
        return Err(Error::Contract("backup is missing the run database".into()));
    }
    let mut actual = Vec::new();
    inventory(root, root, &mut actual, false)?;
    if actual
        .iter()
        .any(|file| file.path != MANIFEST && !paths.contains(&file.path))
    {
        return Err(Error::Contract("backup contains unregistered files".into()));
    }
    crate::store::Store::read_only(root.join("runs.sqlite3"))?;
    let db =
        Connection::open_with_flags(root.join("runs.sqlite3"), OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let integrity: String = db.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    if integrity != "ok" {
        return Err(Error::Contract(
            "backup SQLite integrity check failed".into(),
        ));
    }
    Ok(
        json!({"schema_version": SCHEMA, "verified": true, "files": manifest.files.len(),
        "environment_id": manifest.environment_id, "consistency": manifest.consistency,
        "active_runs_database_only": manifest.active_runs_database_only}),
    )
}

fn create(project_path: &Path, output: &Path) -> Result<Value> {
    let project = load_project(project_path)?;
    let source_path = safe_child(&project.data_dir, Path::new("runs.sqlite3"))?;
    let source = Connection::open_with_flags(source_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let (staging, target) = staging(output)?;
    // A destination inside a run could recursively include itself.
    if target.starts_with(fs::canonicalize(&project.data_dir)?) {
        return Err(Error::Invalid(
            "backup destination must be outside project data_dir".into(),
        ));
    }
    let mut destination = Connection::open(staging.path().join("runs.sqlite3"))?;
    let backup = rusqlite::backup::Backup::new(&source, &mut destination)?;
    // Bound writer contention instead of retrying forever during live training.
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    loop {
        if matches!(backup.step(256)?, rusqlite::backup::StepResult::Done) {
            break;
        }
        if std::time::Instant::now() >= deadline {
            return Err(Error::Contract(
                "online backup exceeded 60 seconds; retry at a quieter point".into(),
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(backup);
    destination.execute_batch("PRAGMA journal_mode = DELETE;")?;
    let rows: Vec<(String, String)> = destination
        .prepare("SELECT run_id, status FROM runs")?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<std::result::Result<_, _>>()?;
    let mut active_runs = Vec::new();
    let mut count = 1;
    for (run_id, status) in &rows {
        validate_identifier(run_id, "backup run_id")?;
        if status == "running" {
            active_runs.push(run_id.clone());
            continue;
        }
        let relative = Path::new("runs").join(run_id);
        let root = safe_child(&project.data_dir, &relative)?;
        if root.is_dir() {
            copy_tree(&root, &staging.path().join(relative), &mut count, 0)?;
        }
        // Preserve registered evidence integrity, not just a hash of copied bytes.
        let artifacts: Vec<(String, String, u64)> = destination
            .prepare("SELECT path, sha256, size_bytes FROM artifacts WHERE run_id = ?")?
            .query_map([run_id], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<std::result::Result<_, _>>()?;
        for (path, digest, size) in artifacts {
            let copied = safe_child(&staging.path().join("runs").join(run_id), Path::new(&path))?;
            if copied.metadata()?.len() != size || sha256_file(&copied)? != digest {
                return Err(Error::Contract(format!(
                    "registered artifact changed: {run_id}/{path}"
                )));
            }
        }
    }
    let has_jobs: bool = destination.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='dashboard_jobs')",
        [],
        |r| r.get(0),
    )?;
    if has_jobs {
        let jobs: Vec<(String, String)> = destination
            .prepare("SELECT id, payload FROM dashboard_jobs")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<std::result::Result<_, _>>()?;
        for (id, payload) in jobs {
            validate_identifier(&id, "backup job id")?;
            let job: Value = serde_json::from_str(&payload)?;
            if !matches!(job["status"].as_str(), Some("succeeded" | "failed")) {
                continue;
            }
            let relative = Path::new("dashboard/jobs").join(id);
            let source = safe_child(&project.data_dir, &relative)?;
            if source.is_dir() {
                copy_tree(&source, &staging.path().join(relative), &mut count, 0)?;
            }
        }
    }
    drop(destination);
    let mut files = Vec::new();
    inventory(staging.path(), staging.path(), &mut files, true)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let manifest = Manifest {
        schema_version: SCHEMA.into(),
        environment_id: project.environment_id,
        created_at_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        consistency: "sqlite_online_snapshot_with_verified_completed_run_files".into(),
        active_runs_database_only: active_runs,
        files,
    };
    fs::write(
        staging.path().join(MANIFEST),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    verify(staging.path())?;
    crate::filesystem::promote(staging.path(), &target)?;
    Ok(
        json!({"schema_version": SCHEMA, "archive": target, "files": manifest.files.len(),
        "active_runs_database_only": manifest.active_runs_database_only, "verified": true}),
    )
}

fn copy_file(source: &Path, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut source = fs::File::open(source)?;
    let mut destination = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(destination)?;
    std::io::copy(&mut source, &mut destination)?;
    destination.sync_all()?;
    Ok(())
}

fn copy_tree(source: &Path, destination: &Path, count: &mut usize, depth: usize) -> Result<()> {
    if depth > 32 {
        return Err(Error::Invalid("backup directory nesting exceeds 32".into()));
    }
    for entry in fs::read_dir(source)? {
        *count += 1;
        if *count > MAX_FILES {
            return Err(Error::Invalid("backup exceeds 100000 entries".into()));
        }
        let entry = entry?;
        let path = safe_child(source, Path::new(&entry.file_name()))?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&path, &target, count, depth + 1)?;
        } else if entry.file_type()?.is_file() {
            copy_file(&path, &target)?;
        } else {
            return Err(Error::Invalid("backup supports regular files only".into()));
        }
    }
    Ok(())
}

fn inventory(root: &Path, current: &Path, files: &mut Vec<BackupFile>, hash: bool) -> Result<()> {
    if current
        .strip_prefix(root)
        .map_or(usize::MAX, |p| p.components().count())
        > 32
        || files.len() > MAX_FILES
    {
        return Err(Error::Invalid("backup inventory exceeds limits".into()));
    }
    for entry in fs::read_dir(current)? {
        if files.len() > MAX_FILES {
            return Err(Error::Invalid("backup inventory exceeds file limit".into()));
        }
        let entry = entry?;
        safe_child(
            root,
            entry
                .path()
                .strip_prefix(root)
                .map_err(|_| Error::Invalid("invalid backup child".into()))?,
        )?;
        if entry.file_type()?.is_dir() {
            inventory(root, &entry.path(), files, hash)?;
        } else {
            files.push(BackupFile {
                path: entry
                    .path()
                    .strip_prefix(root)
                    .expect("child of staging")
                    .to_string_lossy()
                    .replace('\\', "/"),
                size_bytes: entry.metadata()?.len(),
                sha256: if hash {
                    sha256_file(&entry.path())?
                } else {
                    String::new()
                },
            });
        }
    }
    Ok(())
}
