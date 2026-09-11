use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::project::Project;

pub const SCHEMA_VERSION: &str = "glr.run-context.v1";
const MAX_CONTEXT_BYTES: usize = 1024 * 1024;
const MAX_CONTEXT_ENV_BYTES: usize = 24 * 1024;
const MAX_INPUTS: usize = 64;
const MAX_LABELS: usize = 32;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextFile {
    schema_version: String,
    context_id: String,
    environment_id: String,
    protocol_version: String,
    #[serde(default)]
    labels: BTreeMap<String, String>,
    inputs: Vec<InputReference>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputReference {
    owner: String,
    path: String,
    schema_version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileIdentity {
    owner: String,
    path: String,
    schema_version: String,
    sha256: String,
    size_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ContextIdentity {
    path: String,
    sha256: String,
    size_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct UnsignedContext {
    schema_version: &'static str,
    context_id: String,
    environment_id: String,
    protocol_version: String,
    labels: BTreeMap<String, String>,
    source: ContextIdentity,
    inputs: Vec<FileIdentity>,
}

#[derive(Debug, Clone)]
pub struct RunContext {
    unsigned: UnsignedContext,
    context_sha256: String,
}

impl RunContext {
    pub fn load(project: &Project, requested: &Path) -> Result<Self> {
        let (source_path, source_relative) = regular_project_file(&project.root, requested)?;
        let source_bytes = read_bounded(&source_path)?;
        let text = std::str::from_utf8(&source_bytes)
            .map_err(|_| invalid("run context must be UTF-8 TOML"))?;
        let value: ContextFile = toml::from_str(text)
            .map_err(|error| invalid(format!("invalid run context TOML: {error}")))?;
        if value.schema_version != SCHEMA_VERSION {
            return Err(invalid(format!(
                "run context schema_version must be {SCHEMA_VERSION:?}"
            )));
        }
        identifier(&value.context_id, "context_id")?;
        if value.environment_id != project.environment_id {
            return Err(invalid(
                "run context environment_id does not match the project",
            ));
        }
        if value.protocol_version != project.protocol_version {
            return Err(invalid(
                "run context protocol_version does not match the project",
            ));
        }
        if value.inputs.is_empty() || value.inputs.len() > MAX_INPUTS {
            return Err(invalid(format!(
                "run context must contain 1-{MAX_INPUTS} inputs"
            )));
        }
        if value.labels.len() > MAX_LABELS {
            return Err(invalid(format!(
                "run context cannot contain more than {MAX_LABELS} labels"
            )));
        }
        for (name, label) in &value.labels {
            identifier(name, "run context label name")?;
            printable(label, "run context label value", 256)?;
        }
        let mut owners = HashSet::new();
        let mut paths = HashSet::new();
        let mut inputs = Vec::with_capacity(value.inputs.len());
        for reference in value.inputs {
            identifier(&reference.owner, "run context input owner")?;
            printable(
                &reference.schema_version,
                "run context input schema_version",
                128,
            )?;
            if !owners.insert(reference.owner.clone()) {
                return Err(invalid(format!(
                    "duplicate run context input owner: {}",
                    reference.owner
                )));
            }
            let requested = Path::new(&reference.path);
            let (path, relative) = regular_project_file(&project.root, requested)?;
            if relative == source_relative || !paths.insert(relative.clone()) {
                return Err(invalid(format!(
                    "duplicate run context input path: {relative}"
                )));
            }
            let bytes = read_bounded(&path)?;
            verify_schema(&path, &bytes, &reference.schema_version)?;
            inputs.push(FileIdentity {
                owner: reference.owner,
                path: relative,
                schema_version: reference.schema_version,
                sha256: digest(&bytes),
                size_bytes: bytes.len(),
            });
        }
        let unsigned = UnsignedContext {
            schema_version: SCHEMA_VERSION,
            context_id: value.context_id,
            environment_id: value.environment_id,
            protocol_version: value.protocol_version,
            labels: value.labels,
            source: ContextIdentity {
                path: source_relative,
                sha256: digest(&source_bytes),
                size_bytes: source_bytes.len(),
            },
            inputs,
        };
        let context_sha256 = digest(&serde_json::to_vec(&unsigned)?);
        let context = Self {
            unsigned,
            context_sha256,
        };
        if context.json()?.len() > MAX_CONTEXT_ENV_BYTES {
            return Err(invalid(
                "run context exceeds the 24 KiB role environment limit",
            ));
        }
        Ok(context)
    }

    pub fn value(&self) -> Result<Value> {
        let mut value = serde_json::to_value(&self.unsigned)?;
        value
            .as_object_mut()
            .expect("run context serializes as an object")
            .insert(
                "context_sha256".into(),
                Value::String(self.context_sha256.clone()),
            );
        Ok(value)
    }

    pub fn json(&self) -> Result<String> {
        Ok(serde_json::to_string(&self.value()?)?)
    }

    pub fn digest(&self) -> &str {
        &self.context_sha256
    }

    pub fn verify(&self, root: &Path) -> Result<()> {
        verify_identity(
            root,
            &self.unsigned.source.path,
            &self.unsigned.source.sha256,
            self.unsigned.source.size_bytes,
        )?;
        for input in &self.unsigned.inputs {
            verify_identity(root, &input.path, &input.sha256, input.size_bytes)?;
        }
        let expected = digest(&serde_json::to_vec(&self.unsigned)?);
        if expected != self.context_sha256 {
            return Err(Error::Contract(
                "run context identity changed in memory".into(),
            ));
        }
        Ok(())
    }
}

fn verify_identity(
    root: &Path,
    relative: &str,
    expected: &str,
    expected_size: usize,
) -> Result<()> {
    let (path, normalized) = regular_project_file(root, Path::new(relative))?;
    if normalized != relative {
        return Err(Error::Contract("run context path identity changed".into()));
    }
    let bytes = read_bounded(&path)?;
    if bytes.len() != expected_size || digest(&bytes) != expected {
        return Err(Error::Contract(format!(
            "frozen run context input changed before role execution: {relative}"
        )));
    }
    Ok(())
}

fn verify_schema(path: &Path, bytes: &[u8], expected: &str) -> Result<()> {
    let actual = match path.extension().and_then(|value| value.to_str()) {
        Some("json") => serde_json::from_slice::<Value>(bytes)?
            .get("schema_version")
            .and_then(Value::as_str)
            .map(str::to_owned),
        Some("toml") => {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| invalid("run context TOML input must be UTF-8"))?;
            toml::from_str::<toml::Value>(text)
                .map_err(|error| invalid(format!("invalid TOML context input: {error}")))?
                .get("schema_version")
                .and_then(toml::Value::as_str)
                .map(str::to_owned)
        }
        _ => return Err(invalid("run context inputs must be JSON or TOML")),
    };
    if actual.as_deref() != Some(expected) {
        return Err(invalid(format!(
            "run context input {} schema_version must be {expected:?}",
            path.display()
        )));
    }
    Ok(())
}

fn regular_project_file(root: &Path, requested: &Path) -> Result<(PathBuf, String)> {
    if requested.as_os_str().is_empty()
        || requested.is_absolute()
        || requested.components().any(|component| {
            matches!(
                component,
                Component::CurDir
                    | Component::ParentDir
                    | Component::Prefix(_)
                    | Component::RootDir
            )
        })
    {
        return Err(invalid("run context paths must be project-relative"));
    }
    let relative = requested.to_string_lossy().replace('\\', "/");
    if relative.len() > 512 {
        return Err(invalid("run context path exceeds 512 bytes"));
    }
    let mut current = root.to_path_buf();
    for component in requested.components() {
        if let Component::Normal(value) = component {
            current.push(value);
            if current.exists() && linked(&current)? {
                return Err(invalid(
                    "run context paths cannot contain links or reparse points",
                ));
            }
        }
    }
    if !current.is_file() || linked(&current)? {
        return Err(Error::Missing(current));
    }
    Ok((current, relative))
}

fn linked(path: &Path) -> Result<bool> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Ok(true);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if bytes.len() > MAX_CONTEXT_BYTES {
        return Err(invalid(format!(
            "run context file exceeds the 1 MiB limit: {}",
            path.display()
        )));
    }
    Ok(bytes)
}

fn identifier(value: &str, label: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || !value.as_bytes()[0].is_ascii_lowercase()
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_' | b'.')
        })
    {
        return Err(invalid(format!(
            "{label} must start with a lowercase letter and contain lowercase ASCII letters, digits, '.', '_' or '-'"
        )));
    }
    Ok(())
}

fn printable(value: &str, label: &str, maximum: usize) -> Result<()> {
    if value.is_empty() || value.len() > maximum || value.chars().any(char::is_control) {
        return Err(invalid(format!(
            "{label} must be a bounded printable string"
        )));
    }
    Ok(())
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

pub fn metadata(project: &Project) -> Result<Value> {
    project
        .run_context
        .as_ref()
        .map(RunContext::value)
        .transpose()
        .map(|value| value.unwrap_or(Value::Null))
}

pub fn persist(
    project: &Project,
    store: &crate::store::Store,
    run_id: &str,
    run_dir: &Path,
) -> Result<()> {
    let Some(context) = &project.run_context else {
        return Ok(());
    };
    context.verify(&project.root)?;
    let path = run_dir.join("run-context.json");
    crate::contracts::write_json(&path, &context.value()?)?;
    store.register_artifact(
        run_id,
        "run-context.json",
        &path,
        "run-context",
        "application/json",
    )?;
    store.append_event(run_id, "context.selected", context.value()?)?;
    Ok(())
}
