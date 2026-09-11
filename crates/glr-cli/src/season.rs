use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::project::Project;

const LIMIT: u64 = 1024 * 1024;
pub const ENVIRONMENT_KEYS: [&str; 5] = [
    "GLR_SEASON_ID",
    "GLR_RULESET_ID",
    "GLR_SEASON_CONFIG_SHA256",
    "GLR_SEASON_CONTEXT_SHA256",
    "GLR_SEASON_CONTEXT",
];

fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}

fn identifier(value: &str) -> Result<()> {
    if value.len() > 64
        || !value.starts_with(|c: char| c.is_ascii_lowercase())
        || !value
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    {
        return Err(invalid(
            "season identifiers must be portable lowercase identifiers (1..64)",
        ));
    }
    Ok(())
}

pub fn config_path(root: &Path, relative: &str) -> Result<PathBuf> {
    if relative.len() > 512
        || relative.contains(['\\', ':'])
        || relative.chars().any(char::is_control)
        || relative
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(invalid(
            "season config must be a portable root-relative path",
        ));
    }
    let mut path = root.to_path_buf();
    for part in relative.split('/') {
        path.push(part);
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            #[cfg(windows)]
            let reparse = {
                use std::os::windows::fs::MetadataExt;
                metadata.file_attributes() & 0x400 != 0
            };
            #[cfg(not(windows))]
            let reparse = false;
            if metadata.is_symlink() || reparse {
                return Err(invalid(
                    "season config cannot contain links or reparse points",
                ));
            }
        }
    }
    Ok(path)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub config: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    season_id: String,
    ruleset_id: String,
    config: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    schema_version: String,
    entries: Vec<Entry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    schema_version: String,
    season_id: String,
    ruleset_id: String,
    environment_id: String,
    protocol_version: String,
    status: String,
    #[serde(default)]
    extensions: BTreeMap<String, Reference>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub path: String,
    pub sha256: String,
    pub size_bytes: usize,
}

fn read(root: &Path, relative: &str) -> Result<(Input, Vec<u8>)> {
    let path = config_path(root, relative)?;
    if !path.is_file() {
        return Err(invalid("season config must be an existing regular file"));
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(LIMIT + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        return Err(invalid("season input exceeds the 1 MiB limit"));
    }
    Ok((
        Input {
            path: relative.into(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            size_bytes: bytes.len(),
        },
        bytes,
    ))
}

fn read_toml<T: DeserializeOwned>(root: &Path, relative: &str) -> Result<(Input, T)> {
    if !relative.ends_with(".toml") {
        return Err(invalid("season declarations and catalogs must be TOML"));
    }
    let (input, bytes) = read(root, relative)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| invalid("season TOML must be UTF-8"))?;
    let value = toml::from_str(text)
        .map_err(|_| invalid("season TOML has invalid, missing or unknown fields"))?;
    Ok((input, value))
}

fn catalog(project: &Project) -> Result<(Input, Catalog)> {
    let relative = project
        .seasons
        .as_deref()
        .ok_or_else(|| invalid("project has no [seasons] config reference"))?;
    let (input, catalog): (_, Catalog) = read_toml(&project.root, relative)?;
    if catalog.schema_version != "glr.seasons.v1" || catalog.entries.len() > 256 {
        return Err(invalid(
            "unsupported season catalog schema or more than 256 entries",
        ));
    }
    let mut pairs = HashSet::new();
    for entry in &catalog.entries {
        identifier(&entry.season_id)?;
        identifier(&entry.ruleset_id)?;
        config_path(&project.root, &entry.config)?;
        if !pairs.insert((&entry.season_id, &entry.ruleset_id)) {
            return Err(invalid("duplicate season/ruleset selection"));
        }
    }
    Ok((input, catalog))
}

#[derive(Debug, Clone, Serialize)]
pub struct Context {
    schema_version: &'static str,
    pub season_id: String,
    pub ruleset_id: String,
    pub environment_id: String,
    pub protocol_version: String,
    pub status: String,
    pub project: Input,
    pub catalog: Input,
    pub declaration: Input,
    pub extensions: BTreeMap<String, Input>,
}

impl Context {
    pub fn value(&self) -> Result<Value> {
        // Value objects serialize with sorted keys, matching Python's canonical encoding.
        let mut value = serde_json::to_value(self)?;
        let hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?));
        value
            .as_object_mut()
            .expect("context is an object")
            .insert("context_sha256".into(), hash.into());
        Ok(value)
    }

    pub fn json(&self) -> Result<String> {
        let text = serde_json::to_string(&self.value()?)?;
        if text.len() > 24 * 1024 {
            return Err(invalid(
                "season context exceeds the 24 KiB role-environment limit",
            ));
        }
        Ok(text)
    }

    pub fn verify(&self, root: &Path) -> Result<()> {
        for reference in [&self.project, &self.catalog, &self.declaration]
            .into_iter()
            .chain(self.extensions.values())
        {
            if read(root, &reference.path)?.0 != *reference {
                return Err(invalid("frozen season input changed before role execution"));
            }
        }
        Ok(())
    }

    pub fn require_ready(&self) -> Result<()> {
        if self.status != "ready" {
            return Err(invalid(
                "selected season is pending; train/goal/play require ready configuration",
            ));
        }
        Ok(())
    }
}

pub fn select(
    project: &Project,
    season: Option<&str>,
    ruleset: Option<&str>,
) -> Result<Option<Context>> {
    let (season, ruleset) = match (season, ruleset) {
        (None, None) => return Ok(None),
        (Some(season), Some(ruleset)) => (season, ruleset),
        _ => return Err(invalid("--season and --ruleset must be supplied together")),
    };
    identifier(season)?;
    identifier(ruleset)?;
    let (catalog_input, catalog) = catalog(project)?;
    let entry = catalog
        .entries
        .iter()
        .find(|entry| entry.season_id == season && entry.ruleset_id == ruleset)
        .ok_or_else(|| invalid("unknown season/ruleset selection"))?;
    let (declaration_input, declaration): (_, Declaration) =
        read_toml(&project.root, &entry.config)?;
    if declaration.schema_version != "glr.season.v1"
        || declaration.season_id != season
        || declaration.ruleset_id != ruleset
        || declaration.environment_id != project.environment_id
        || declaration.protocol_version != project.protocol_version
    {
        return Err(invalid(
            "season declaration identity does not match project and selection",
        ));
    }
    if !matches!(declaration.status.as_str(), "pending" | "ready") {
        return Err(invalid("season status must be pending or ready"));
    }
    if declaration.extensions.len() > 32 {
        return Err(invalid("season extensions exceed 32 entries"));
    }
    let mut extensions = BTreeMap::new();
    for (namespace, reference) in declaration.extensions {
        identifier(&namespace)?;
        extensions.insert(namespace, read(&project.root, &reference.config)?.0);
    }
    let manifest_name = project
        .manifest_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("manifest filename must be UTF-8"))?;
    let project_input = read(&project.root, manifest_name)?.0;
    if project_input.sha256 != project.manifest_sha256 {
        return Err(invalid("project manifest changed while selecting season"));
    }
    let context = Context {
        schema_version: "glr.season-context.v1",
        season_id: season.into(),
        ruleset_id: ruleset.into(),
        environment_id: project.environment_id.clone(),
        protocol_version: project.protocol_version.clone(),
        status: declaration.status,
        project: project_input,
        catalog: catalog_input,
        declaration: declaration_input,
        extensions,
    };
    context.json()?;
    Ok(Some(context))
}

pub fn list(project: &Project) -> Result<Vec<Value>> {
    if project.seasons.is_none() {
        return Ok(Vec::new());
    }
    catalog(project)?
        .1
        .entries
        .iter()
        .map(|entry| {
            select(project, Some(&entry.season_id), Some(&entry.ruleset_id))?
                .expect("selected entry")
                .value()
        })
        .collect()
}

pub fn require_selection(project: &Project, ready: bool) -> Result<()> {
    if project.seasons.is_some() && project.season_context.is_none() {
        return Err(invalid(
            "project requires explicit --season and --ruleset before role execution",
        ));
    }
    if let Some(context) = &project.season_context {
        context.verify(&project.root)?;
        if ready {
            context.require_ready()?;
        }
    }
    Ok(())
}

pub fn initialize(project: &Project, season: &str, ruleset: &str) -> Result<Value> {
    identifier(season)?;
    identifier(ruleset)?;
    let relative = project
        .seasons
        .as_deref()
        .ok_or_else(|| invalid("first declare [seasons] config in the project"))?;
    if !relative.ends_with(".toml") {
        return Err(invalid("season catalog must be TOML"));
    }
    let path = config_path(&project.root, relative)?;
    let original = if path.exists() {
        Some(read(&project.root, relative)?.1)
    } else {
        None
    };
    let mut entries = if original.is_some() {
        catalog(project)?.1.entries
    } else {
        Vec::new()
    };
    if entries
        .iter()
        .any(|entry| entry.season_id == season && entry.ruleset_id == ruleset)
    {
        return Err(invalid("season/ruleset already registered"));
    }
    if entries.len() >= 256 {
        return Err(invalid("season catalog exceeds 256 entries"));
    }
    let declaration_relative = format!("config/seasons/{season}/{ruleset}.toml");
    if declaration_relative == relative
        || (cfg!(windows) && declaration_relative.eq_ignore_ascii_case(relative))
    {
        return Err(invalid("catalog and declaration paths must differ"));
    }
    let declaration_path = config_path(&project.root, &declaration_relative)?;
    fs::create_dir_all(declaration_path.parent().expect("declaration parent"))?;
    fs::create_dir_all(path.parent().expect("catalog parent"))?;
    let lock_path = path.with_extension("toml.lock");
    let lock = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)?;
    let result = (|| {
        config_path(&project.root, relative)?;
        let current = if path.exists() {
            Some(read(&project.root, relative)?.1)
        } else {
            None
        };
        if current != original {
            return Err(invalid("season catalog changed during initialization"));
        }
        entries.push(Entry {
            season_id: season.into(),
            ruleset_id: ruleset.into(),
            config: declaration_relative.clone(),
        });
        let mut catalog_text = "schema_version = \"glr.seasons.v1\"\n".to_owned();
        for entry in &entries {
            catalog_text.push_str(&format!(
                "\n[[entries]]\nseason_id = {}\nruleset_id = {}\nconfig = {}\n",
                serde_json::to_string(&entry.season_id)?,
                serde_json::to_string(&entry.ruleset_id)?,
                serde_json::to_string(&entry.config)?
            ));
        }
        let text = format!(
            "schema_version = \"glr.season.v1\"\nseason_id = {}\nruleset_id = {}\nenvironment_id = {}\nprotocol_version = {}\nstatus = \"pending\"\n",
            serde_json::to_string(season)?,
            serde_json::to_string(ruleset)?,
            serde_json::to_string(&project.environment_id)?,
            serde_json::to_string(&project.protocol_version)?
        );
        config_path(&project.root, &declaration_relative)?;
        let mut declaration_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&declaration_path)?;
        let write_result = (|| {
            declaration_file.write_all(text.as_bytes())?;
            drop(declaration_file);
            let mut temporary =
                tempfile::NamedTempFile::new_in(path.parent().expect("catalog parent"))?;
            temporary.write_all(catalog_text.as_bytes())?;
            temporary
                .persist(&path)
                .map_err(|error| Error::Io(error.error))?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&declaration_path);
        }
        write_result
    })();
    drop(lock);
    let _ = fs::remove_file(lock_path);
    result?;
    select(project, Some(season), Some(ruleset))?
        .expect("created selection")
        .value()
}
