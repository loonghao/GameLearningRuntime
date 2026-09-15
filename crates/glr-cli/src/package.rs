//! Offline, explicit source-only packages. No role, installer, or hook execution.
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use zip::{ZipArchive, ZipWriter, write::SimpleFileOptions};

use crate::args::PackageCommand;
use crate::error::{Error, Result};
use crate::project::find_project;

const SCHEMA: &str = "glr.source-package.v1";
const MAX_FILE: u64 = 16 * 1024 * 1024;
const MAX_TOTAL: u64 = 128 * 1024 * 1024;
const MAX_FILES: usize = 1024;
const MANIFEST: &str = "glr-package.json";

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

// Atomic no-replace promotion. A racing destination must never be overwritten.
fn promote(source: &Path, destination: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
        let destination: Vec<u16> = destination
            .as_os_str()
            .encode_wide()
            .chain(Some(0))
            .collect();
        if unsafe {
            windows_sys::Win32::Storage::FileSystem::MoveFileW(
                source.as_ptr(),
                destination.as_ptr(),
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let source = std::ffi::CString::new(source.as_os_str().as_bytes())
            .map_err(|_| refusal("invalid destination"))?;
        let destination = std::ffi::CString::new(destination.as_os_str().as_bytes())
            .map_err(|_| refusal("invalid destination"))?;
        #[cfg(target_os = "linux")]
        let result = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                destination.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        #[cfg(target_os = "macos")]
        let result =
            unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
        if result != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}

pub(crate) fn execute(project: &Path, command: &PackageCommand) -> Result<Value> {
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
            Ok(
                json!({"status": "verified-source-inventory", "manifest": manifest, "executed": false}),
            )
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
            Ok(
                json!({"status": "verified-source-package", "manifest": manifest, "executed": false, "training_ready": false}),
            )
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
        execute(root.path(), &command).unwrap();
        assert_eq!(
            fs::read(destination.join("train.py")).unwrap(),
            fs::read(root.path().join("train.py")).unwrap()
        );
        assert!(!destination.join("selection.json").exists());
        assert!(execute(root.path(), &command).is_err());
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
                }
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
