//! Declarative, no-exec plugin control plane for the standalone glr CLI.
//!
//! A plugin is a local directory with a strict glr-plugin.json manifest.
//! Inspecting, installing, and resolving bundles only reads and copies files;
//! it never imports an entrypoint, runs a hook, or starts a child process.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use semver::{Version, VersionReq};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use crate::args::{PluginCommand, PluginProfileCommand};
use crate::commands::emit;
use crate::error::{Error, Result};

const PLUGIN_SCHEMA_VERSION: &str = "glr.plugin.v1";
const PROFILE_SCHEMA_VERSION: &str = "glr.profile.v1";
const PLUGIN_FILE_NAME: &str = "glr-plugin.json";
const PLUGIN_STORE_DIR: &str = ".glr/plugins";
const PROFILE_STORE_DIR: &str = ".glr/profiles";
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
const MAX_PROFILE_BYTES: u64 = 256 * 1024;
const MAX_FILES: usize = 4096;
const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 512 * 1024 * 1024;
const MAX_PATH_DEPTH: usize = 16;

const PLUGIN_KINDS: &[&str] = &[
    "environment",
    "learner",
    "recorder",
    "evaluator",
    "model-provider",
    "harness",
    "ui",
    "command",
];
const PERMISSIONS: &[&str] = &[
    "read:environment",
    "read:dataset",
    "read:artifact",
    "write:checkpoint",
    "write:dataset",
    "write:artifact",
    "spawn:worker",
    "network:outbound",
    "runtime:observe",
    "runtime:act",
    "events:emit",
    "ui:panel",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PluginFile {
    path: String,
    sha256: String,
    size_bytes: u64,
    #[serde(default = "default_role")]
    role: String,
}

fn default_role() -> String {
    "payload".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    id: String,
    version: String,
    kind: String,
    name: String,
    description: String,
    entrypoint: String,
    capabilities: Vec<String>,
    #[serde(default)]
    requires: BTreeMap<String, String>,
    #[serde(default = "default_platforms")]
    platforms: Vec<String>,
    #[serde(default = "default_isolation")]
    isolation: String,
    #[serde(default)]
    permissions: Vec<String>,
    #[serde(default)]
    dependencies: BTreeMap<String, String>,
    #[serde(default)]
    files: Vec<PluginFile>,
}

fn default_platforms() -> Vec<String> {
    vec!["windows".into(), "linux".into(), "macos".into()]
}

fn default_isolation() -> String {
    "process".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileRef {
    id: String,
    #[serde(default = "default_requirement")]
    version: String,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    permissions: Vec<String>,
    #[serde(default = "default_config")]
    config: Value,
}

fn default_requirement() -> String {
    "*".into()
}

fn default_true() -> bool {
    true
}

fn default_config() -> Value {
    Value::Object(Map::new())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Profile {
    schema_version: String,
    name: String,
    plugins: Vec<ProfileRef>,
}

#[derive(Debug, Clone)]
struct Inspection {
    manifest: Manifest,
    source: String,
    files: Vec<PluginFile>,
    content_sha256: String,
    manifest_sha256: String,
    total_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct Installation {
    id: String,
    version: String,
    kind: String,
    content_sha256: String,
    manifest_sha256: String,
    path: String,
    source_kind: String,
    isolation: String,
    capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ResolvedPlugin {
    id: String,
    version: String,
    kind: String,
    #[serde(skip_serializing)]
    isolation: String,
    #[serde(skip_serializing)]
    capabilities: Vec<String>,
    permissions: Vec<String>,
    config: Value,
    content_sha256: String,
    path: String,
}

struct Manager {
    project_root: PathBuf,
    store_root: PathBuf,
    profile_root: PathBuf,
}

impl Manager {
    fn new(project: &Path) -> Result<Self> {
        let requested = absolute_path(project)?;
        if path_has_link_component(&requested)? {
            return Err(plugin_error(
                "plugin project root cannot contain symlinks or reparse points",
            ));
        }
        let root = if requested.is_file() {
            requested
                .parent()
                .ok_or_else(|| plugin_error("project path has no parent"))?
                .to_path_buf()
        } else {
            requested
        };
        if path_has_link_component(&root)? {
            return Err(plugin_error(
                "plugin project root cannot contain symlinks or reparse points",
            ));
        }
        Ok(Self {
            store_root: root.join(PLUGIN_STORE_DIR),
            profile_root: root.join(PROFILE_STORE_DIR),
            project_root: root,
        })
    }

    fn inspect(&self, source: &Path) -> Result<Inspection> {
        inspect_bundle(&absolute_path(source)?)
    }

    fn install(&self, source: &Path, expected: Option<&str>) -> Result<Installation> {
        let source_root = canonical_regular_directory(source)?;
        let inspection = inspect_bundle(&source_root)?;
        if let Some(expected) = expected {
            ensure_digest(expected, "--sha256")?;
            if expected != inspection.content_sha256 {
                return Err(plugin_error("--sha256 does not match the inspected bundle"));
            }
        }
        let destination = self
            .store_root
            .join(&inspection.manifest.id)
            .join(&inspection.manifest.version);
        if destination.exists() || is_symlink_or_reparse(&destination)? {
            return Err(plugin_error(format!(
                "plugin is already installed: {}@{}",
                inspection.manifest.id, inspection.manifest.version
            )));
        }
        ensure_directory(&self.store_root, "plugin store")?;
        let parent = destination
            .parent()
            .ok_or_else(|| plugin_error("plugin destination has no parent"))?;
        ensure_directory(parent, "plugin store")?;
        let temporary = TempDir::new_in(parent)?;
        copy_bundle(&source_root, temporary.path(), &inspection.files)?;
        let copied = inspect_bundle(temporary.path())?;
        if copied.content_sha256 != inspection.content_sha256 {
            return Err(plugin_error(
                "staged plugin digest changed during installation",
            ));
        }
        let temporary_path = temporary.keep();
        fs::rename(&temporary_path, &destination)
            .map_err(|error| plugin_error(format!("cannot commit plugin installation: {error}")))?;
        Ok(installation(
            &inspection,
            &destination,
            &self.project_root,
            "local",
        ))
    }

    fn installed(&self) -> Result<Vec<(Inspection, PathBuf)>> {
        if path_has_link_component(&self.store_root)? {
            return Err(plugin_error(
                "plugin store cannot contain symlinks or reparse points",
            ));
        }
        if is_symlink_or_reparse(&self.store_root)? {
            return Err(plugin_error("plugin store must be a regular directory"));
        }
        if !self.store_root.exists() {
            return Ok(Vec::new());
        }
        if !self.store_root.is_dir() {
            return Err(plugin_error("plugin store must be a regular directory"));
        }
        let mut result = Vec::new();
        for plugin_dir in sorted_dirs(&self.store_root)? {
            if is_symlink_or_reparse(&plugin_dir)? || !plugin_dir.is_dir() {
                return Err(plugin_error("plugin store contains an invalid entry"));
            }
            for version_dir in sorted_dirs(&plugin_dir)? {
                if is_symlink_or_reparse(&version_dir)? || !version_dir.is_dir() {
                    return Err(plugin_error(
                        "plugin store contains an invalid version entry",
                    ));
                }
                let inspection = inspect_bundle(&version_dir)?;
                let plugin_name = plugin_dir.file_name().unwrap().to_string_lossy();
                let version_name = version_dir.file_name().unwrap().to_string_lossy();
                if inspection.manifest.id != plugin_name
                    || inspection.manifest.version != version_name
                {
                    return Err(plugin_error(
                        "installed plugin path does not match its manifest",
                    ));
                }
                result.push((inspection, version_dir));
            }
        }
        Ok(result)
    }

    fn list(&self) -> Result<Vec<Installation>> {
        Ok(self
            .installed()?
            .iter()
            .map(|(inspection, path)| {
                installation(inspection, path, &self.project_root, "installed")
            })
            .collect())
    }

    fn profile_path(&self, name: &str) -> Result<PathBuf> {
        ensure_identifier(name, "profile.name")?;
        Ok(self.profile_root.join(format!("{name}.json")))
    }

    fn load_profile(&self, name: &str) -> Result<Profile> {
        let path = self.profile_path(name)?;
        let mut profile: Profile = read_json(&path, MAX_PROFILE_BYTES, "plugin profile")?;
        normalize_profile_requirements(&mut profile)?;
        validate_profile(&profile)?;
        Ok(profile)
    }

    fn save_profile(&self, profile: &Profile) -> Result<()> {
        let mut profile = profile.clone();
        normalize_profile_requirements(&mut profile)?;
        validate_profile(&profile)?;
        if path_has_link_component(&self.profile_root)? {
            return Err(plugin_error(
                "profile store cannot contain symlinks or reparse points",
            ));
        }
        ensure_directory(&self.profile_root, "profile store")?;
        let destination = self.profile_path(&profile.name)?;
        if is_symlink_or_reparse(&destination)? {
            return Err(plugin_error("profile path cannot be a symlink"));
        }
        let temporary = destination.with_extension("json.tmp");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let mut bytes = serde_json::to_vec_pretty(&profile)?;
        bytes.push(b'\n');
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        if destination.exists() {
            fs::remove_file(&destination)?;
        }
        fs::rename(temporary, destination)?;
        Ok(())
    }

    fn profiles(&self) -> Result<Vec<Profile>> {
        if path_has_link_component(&self.profile_root)? {
            return Err(plugin_error(
                "profile store cannot contain symlinks or reparse points",
            ));
        }
        if is_symlink_or_reparse(&self.profile_root)? {
            return Err(plugin_error("profile store must be a regular directory"));
        }
        if !self.profile_root.exists() {
            return Ok(Vec::new());
        }
        if !self.profile_root.is_dir() {
            return Err(plugin_error("profile store must be a regular directory"));
        }
        let mut paths = fs::read_dir(&self.profile_root)?
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .collect::<Vec<_>>();
        paths.sort_by_key(|path| path.file_name().unwrap().to_string_lossy().to_lowercase());
        paths
            .into_iter()
            .map(|path| {
                if is_symlink_or_reparse(&path)? || !path.is_file() {
                    return Err(plugin_error("profile store contains an invalid entry"));
                }
                let mut profile: Profile = read_json(&path, MAX_PROFILE_BYTES, "plugin profile")?;
                normalize_profile_requirements(&mut profile)?;
                validate_profile(&profile)?;
                Ok(profile)
            })
            .collect()
    }

    fn resolve(&self, profile: &Profile) -> Result<(Vec<ResolvedPlugin>, String)> {
        let mut profile = profile.clone();
        normalize_profile_requirements(&mut profile)?;
        validate_profile(&profile)?;
        let installed = self.installed()?;
        let mut by_id: BTreeMap<String, Vec<(Inspection, PathBuf, Version)>> = BTreeMap::new();
        for (inspection, path) in installed {
            let version = Version::parse(&inspection.manifest.version)
                .map_err(|error| plugin_error(format!("invalid installed version: {error}")))?;
            by_id
                .entry(inspection.manifest.id.clone())
                .or_default()
                .push((inspection, path, version));
        }
        for candidates in by_id.values_mut() {
            candidates.sort_by(|left, right| right.2.cmp(&left.2));
        }
        let mut resolved = BTreeMap::<String, ResolvedPlugin>::new();
        let mut order = Vec::new();
        let mut resolving = Vec::new();
        {
            let mut state = ResolveState {
                by_id: &by_id,
                resolved: &mut resolved,
                order: &mut order,
                resolving: &mut resolving,
                project_root: &self.project_root,
            };
            for reference in profile.plugins.iter().filter(|item| item.enabled) {
                select_plugin(
                    &reference.id,
                    &reference.version,
                    &reference.permissions,
                    &reference.config,
                    true,
                    &mut state,
                )?;
            }
        }
        let plugins = order
            .into_iter()
            .map(|id| {
                resolved
                    .remove(&id)
                    .expect("resolved plugin order is complete")
            })
            .collect::<Vec<_>>();
        let digest = resolved_profile_digest(&profile, &plugins);
        Ok((plugins, digest))
    }

    fn health(&self, profile: Option<&str>) -> Result<Value> {
        if let Some(name) = profile {
            match self
                .load_profile(name)
                .and_then(|profile| self.resolve(&profile))
            {
                Ok((plugins, digest)) => Ok(Value::Array(
                    plugins
                        .into_iter()
                        .map(|plugin| {
                            json!({
                                "id": plugin.id,
                                "version": plugin.version,
                                "kind": plugin.kind,
                                "status": "ready",
                                "isolation": plugin.isolation,
                                "capabilities": plugin.capabilities,
                                "permissions": plugin.permissions,
                                "content_sha256": plugin.content_sha256,
                                "profile_digest": digest,
                            })
                        })
                        .collect(),
                )),
                Err(error) => Ok(json!([{"status": "blocked", "reason": error.to_string()}])),
            }
        } else {
            let platform = platform_name();
            Ok(Value::Array(
                self.installed()?
                    .into_iter()
                    .map(|(inspection, _)| {
                        let ready = inspection.manifest.platforms.iter().any(|item| item == platform);
                        json!({
                            "id": inspection.manifest.id,
                            "version": inspection.manifest.version,
                            "kind": inspection.manifest.kind,
                            "status": if ready { "ready" } else { "blocked" },
                            "reason": if ready { Value::Null } else { json!("unsupported platform") },
                            "isolation": inspection.manifest.isolation,
                            "capabilities": inspection.manifest.capabilities,
                            "content_sha256": inspection.content_sha256,
                        })
                    })
                    .collect(),
            ))
        }
    }

    fn remove(&self, id: &str, version: Option<&str>) -> Result<()> {
        ensure_identifier(id, "plugin.id")?;
        if let Some(version) = version {
            ensure_version(version, "plugin.version")?;
        }
        let mut candidates = self
            .installed()?
            .into_iter()
            .filter(|(inspection, _)| inspection.manifest.id == id)
            .collect::<Vec<_>>();
        if let Some(version) = version {
            candidates.retain(|(inspection, _)| inspection.manifest.version == version);
        }
        if candidates.is_empty() {
            return Err(plugin_error("plugin is not installed"));
        }
        if version.is_none() && candidates.len() > 1 {
            return Err(plugin_error(
                "version is required when multiple plugin versions are installed",
            ));
        }
        for profile in self.profiles()? {
            if profile
                .plugins
                .iter()
                .any(|reference| reference.enabled && reference.id == id)
            {
                return Err(plugin_error(format!(
                    "plugin is enabled by profile {}",
                    profile.name
                )));
            }
        }
        let target = &candidates[0].1;
        if is_symlink_or_reparse(target)?
            || !target.is_dir()
            || !target.starts_with(&self.store_root)
        {
            return Err(plugin_error("refusing to remove an unsafe plugin path"));
        }
        fs::remove_dir_all(target)?;
        if let Some(parent) = target.parent() {
            let _ = fs::remove_dir(parent);
        }
        Ok(())
    }
}

struct ResolveState<'a> {
    by_id: &'a BTreeMap<String, Vec<(Inspection, PathBuf, Version)>>,
    resolved: &'a mut BTreeMap<String, ResolvedPlugin>,
    order: &'a mut Vec<String>,
    resolving: &'a mut Vec<String>,
    project_root: &'a Path,
}

pub(crate) fn execute(project: &Path, command: &PluginCommand, compact: bool) -> Result<i32> {
    let manager = Manager::new(project)?;
    match command {
        PluginCommand::Inspect { source } => {
            let inspection = manager.inspect(source)?;
            emit("plugin.inspect", &inspection_value(&inspection), compact)?;
        }
        PluginCommand::Install { source, sha256 } => {
            let installation = manager.install(source, sha256.as_deref())?;
            emit("plugin.install", &installation, compact)?;
        }
        PluginCommand::List => {
            emit("plugin.list", &manager.list()?, compact)?;
        }
        PluginCommand::Health { profile } => {
            emit(
                "plugin.health",
                &manager.health(profile.as_deref())?,
                compact,
            )?;
        }
        PluginCommand::Remove { id, version } => {
            manager.remove(id, version.as_deref())?;
            emit(
                "plugin.remove",
                &json!({"id": id, "version": version}),
                compact,
            )?;
        }
        PluginCommand::Profile { command } => match command {
            PluginProfileCommand::List => {
                emit("plugin.profile.list", &manager.profiles()?, compact)?;
            }
            PluginProfileCommand::Show { name } => {
                let profile = manager.load_profile(name)?;
                let mut value = serde_json::to_value(&profile)?;
                value
                    .as_object_mut()
                    .expect("profile serializes as an object")
                    .insert("digest".into(), Value::String(profile_digest(&profile)));
                emit("plugin.profile.show", &value, compact)?;
            }
            PluginProfileCommand::Resolve { name } => {
                let profile = manager.load_profile(name)?;
                let (plugins, digest) = manager.resolve(&profile)?;
                emit(
                    "plugin.profile.resolve",
                    &json!({
                        "schema_version": profile.schema_version,
                        "name": profile.name,
                        "digest": digest,
                        "plugins": plugins,
                    }),
                    compact,
                )?;
            }
            PluginProfileCommand::Enable {
                name,
                id,
                version,
                grants,
            } => {
                let mut profile = match manager.load_profile(name) {
                    Ok(profile) => profile,
                    Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                        Profile {
                            schema_version: PROFILE_SCHEMA_VERSION.into(),
                            name: name.clone(),
                            plugins: Vec::new(),
                        }
                    }
                    Err(error) => return Err(error),
                };
                update_profile_ref(&mut profile, id, version.as_deref(), grants, true)?;
                manager.save_profile(&profile)?;
                emit("plugin.profile.enable", &profile, compact)?;
            }
            PluginProfileCommand::Disable { name, id } => {
                let mut profile = manager.load_profile(name)?;
                update_profile_ref(&mut profile, id, None, &[], false)?;
                manager.save_profile(&profile)?;
                emit("plugin.profile.disable", &profile, compact)?;
            }
        },
    }
    Ok(0)
}

fn update_profile_ref(
    profile: &mut Profile,
    id: &str,
    version: Option<&str>,
    grants: &[String],
    enabled: bool,
) -> Result<()> {
    ensure_identifier(id, "plugin.id")?;
    let position = profile
        .plugins
        .iter()
        .position(|reference| reference.id == id);
    if !enabled && position.is_none() {
        return Err(plugin_error(format!(
            "plugin {id:?} is not present in profile {:?}",
            profile.name
        )));
    }
    if let Some(index) = position {
        let current = &profile.plugins[index];
        profile.plugins[index] = ProfileRef {
            id: id.into(),
            version: canonical_requirement(version.unwrap_or(&current.version))?,
            enabled,
            permissions: if grants.is_empty() {
                current.permissions.clone()
            } else {
                grants.to_vec()
            },
            config: current.config.clone(),
        };
    } else {
        profile.plugins.push(ProfileRef {
            id: id.into(),
            version: canonical_requirement(version.unwrap_or("*"))?,
            enabled,
            permissions: grants.to_vec(),
            config: Value::Object(Map::new()),
        });
    }
    validate_profile(profile)
}

fn select_plugin(
    id: &str,
    requirement: &str,
    permissions: &[String],
    config: &Value,
    explicit: bool,
    state: &mut ResolveState<'_>,
) -> Result<()> {
    if state.resolving.iter().any(|item| item == id) {
        let mut cycle = state.resolving.clone();
        cycle.push(id.into());
        return Err(plugin_error(format!(
            "plugin dependency cycle: {}",
            cycle.join(" -> ")
        )));
    }
    let requested_config = validate_config(config)?;
    if let Some(existing) = state.resolved.get(id).cloned() {
        let requested = requirement_req(requirement)?;
        let actual =
            Version::parse(&existing.version).map_err(|error| plugin_error(error.to_string()))?;
        if !requested.matches(&actual) {
            return Err(plugin_error(format!("plugin dependency conflict for {id}")));
        }
        let mut requested_permissions = permissions.to_vec();
        requested_permissions.sort();
        let existing_inspection = state
            .by_id
            .get(id)
            .and_then(|candidates| {
                candidates
                    .iter()
                    .find(|(inspection, _, _)| inspection.manifest.version == existing.version)
            })
            .map(|(inspection, _, _)| inspection)
            .ok_or_else(|| plugin_error(format!("installed plugin disappeared: {id}")))?;
        if !requested_permissions.is_empty() {
            validate_runtime_compatibility(existing_inspection, permissions)?;
        }
        let mut updated = existing.clone();
        if !requested_permissions.is_empty() {
            if !existing.permissions.is_empty() && requested_permissions != existing.permissions {
                return Err(plugin_error(format!("plugin permission conflict for {id}")));
            }
            if explicit && existing.permissions.is_empty() {
                updated.permissions = requested_permissions;
            }
        }
        let requested_has_config = requested_config
            .as_object()
            .is_some_and(|object| !object.is_empty());
        let existing_has_config = existing
            .config
            .as_object()
            .is_some_and(|object| !object.is_empty());
        if requested_has_config {
            if existing_has_config && existing.config != requested_config {
                return Err(plugin_error(format!("plugin config conflict for {id}")));
            }
            if explicit && !existing_has_config {
                updated.config = requested_config;
            }
        }
        if updated.permissions != existing.permissions || updated.config != existing.config {
            state.resolved.insert(id.into(), updated);
        }
        return Ok(());
    }
    let requested = requirement_req(requirement)?;
    let candidates = state.by_id.get(id).ok_or_else(|| {
        plugin_error(format!(
            "plugin is not installed or version is unsatisfied: {id}@{requirement}"
        ))
    })?;
    let (inspection, path, _version) = candidates
        .iter()
        .find(|(_, _, version)| requested.matches(version))
        .ok_or_else(|| {
            plugin_error(format!(
                "plugin is not installed or version is unsatisfied: {id}@{requirement}"
            ))
        })?;
    validate_runtime_compatibility(inspection, permissions)?;
    state.resolving.push(id.into());
    for (dependency_id, dependency_range) in &inspection.manifest.dependencies {
        select_plugin(
            dependency_id,
            dependency_range,
            &[],
            &Value::Object(Map::new()),
            false,
            state,
        )?;
    }
    state.resolving.pop();
    let relative_path = path
        .strip_prefix(state.project_root)
        .map_err(|_| plugin_error("resolved plugin path escaped the project root"))?
        .to_string_lossy()
        .replace('\\', "/");
    state.resolved.insert(
        id.into(),
        ResolvedPlugin {
            id: inspection.manifest.id.clone(),
            version: inspection.manifest.version.clone(),
            kind: inspection.manifest.kind.clone(),
            isolation: inspection.manifest.isolation.clone(),
            capabilities: inspection.manifest.capabilities.clone(),
            permissions: sorted_unique(permissions),
            config: requested_config,
            content_sha256: inspection.content_sha256.clone(),
            path: relative_path,
        },
    );
    state.order.push(id.into());
    Ok(())
}

fn validate_runtime_compatibility(inspection: &Inspection, permissions: &[String]) -> Result<()> {
    if !inspection
        .manifest
        .platforms
        .iter()
        .any(|item| item == platform_name())
    {
        return Err(plugin_error(format!(
            "plugin is not supported on this platform: {}",
            inspection.manifest.id
        )));
    }
    if let Some(requirement) = inspection.manifest.requires.get("glr") {
        let current = Version::parse(env!("CARGO_PKG_VERSION"))
            .map_err(|error| plugin_error(format!("invalid GLR version: {error}")))?;
        if !requirement_req(requirement)?.matches(&current) {
            return Err(plugin_error(format!(
                "plugin requires an incompatible GLR version: {}",
                inspection.manifest.id
            )));
        }
    }
    let declared = inspection
        .manifest
        .permissions
        .iter()
        .collect::<HashSet<_>>();
    if let Some(permission) = permissions
        .iter()
        .find(|permission| !declared.contains(permission))
    {
        return Err(plugin_error(format!(
            "profile permission grant exceeds plugin declaration: {permission}"
        )));
    }
    Ok(())
}

fn inspect_bundle(source: &Path) -> Result<Inspection> {
    let source = canonical_regular_directory(source)?;
    let manifest_path = source.join(PLUGIN_FILE_NAME);
    let mut manifest: Manifest = read_json(&manifest_path, MAX_MANIFEST_BYTES, "plugin manifest")?;
    normalize_manifest_requirements(&mut manifest)?;
    validate_manifest(&manifest)?;
    let files = inventory(&source)?;
    let payload = files
        .iter()
        .filter(|item| item.path != PLUGIN_FILE_NAME)
        .cloned()
        .collect::<Vec<_>>();
    if !manifest.files.is_empty() {
        let declared = manifest
            .files
            .iter()
            .map(|item| (item.path.clone(), item))
            .collect::<BTreeMap<_, _>>();
        let actual = payload
            .iter()
            .map(|item| (item.path.clone(), item))
            .collect::<BTreeMap<_, _>>();
        if declared.keys().collect::<Vec<_>>() != actual.keys().collect::<Vec<_>>() {
            return Err(plugin_error(
                "plugin manifest files do not match the source inventory",
            ));
        }
        for (path, expected) in declared {
            let found = actual.get(&path).unwrap();
            if expected.sha256 != found.sha256 || expected.size_bytes != found.size_bytes {
                return Err(plugin_error(format!(
                    "plugin file integrity mismatch: {path}"
                )));
            }
        }
    }
    if !manifest.entrypoint.contains(':')
        && !payload.iter().any(|item| item.path == manifest.entrypoint)
    {
        return Err(plugin_error("plugin.entrypoint file is missing"));
    }
    let canonical = canonical_json(&json!({"manifest": &manifest, "files": &files}));
    let manifest_bytes = fs::read(&manifest_path)?;
    let total_bytes = files.iter().map(|item| item.size_bytes).sum();
    Ok(Inspection {
        source: source
            .file_name()
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_else(|| source.to_string_lossy().into_owned()),
        manifest,
        files,
        content_sha256: sha256_bytes(&canonical),
        manifest_sha256: sha256_bytes(&manifest_bytes),
        total_bytes,
    })
}

fn inspection_value(inspection: &Inspection) -> Value {
    json!({
        "schema_version": inspection.manifest.schema_version,
        "source": inspection.source,
        "plugin": inspection.manifest,
        "files": inspection.files,
        "content_sha256": inspection.content_sha256,
        "manifest_sha256": inspection.manifest_sha256,
        "total_bytes": inspection.total_bytes,
    })
}

fn installation(
    inspection: &Inspection,
    path: &Path,
    project_root: &Path,
    source_kind: &str,
) -> Installation {
    Installation {
        id: inspection.manifest.id.clone(),
        version: inspection.manifest.version.clone(),
        kind: inspection.manifest.kind.clone(),
        content_sha256: inspection.content_sha256.clone(),
        manifest_sha256: inspection.manifest_sha256.clone(),
        path: path
            .strip_prefix(project_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/"),
        source_kind: source_kind.into(),
        isolation: inspection.manifest.isolation.clone(),
        capabilities: inspection.manifest.capabilities.clone(),
    }
}

fn copy_bundle(source: &Path, destination: &Path, files: &[PluginFile]) -> Result<()> {
    for file in files {
        let source_file = source.join(file.path.replace('/', std::path::MAIN_SEPARATOR_STR));
        let destination_file =
            destination.join(file.path.replace('/', std::path::MAIN_SEPARATOR_STR));
        let metadata = fs::symlink_metadata(&source_file)?;
        if metadata.file_type().is_symlink()
            || is_reparse(&metadata)
            || !metadata.is_file()
            || is_hard_linked(&metadata)
        {
            return Err(plugin_error(
                "plugin source changed or contains a non-regular file",
            ));
        }
        if let Some(parent) = destination_file.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(source_file, destination_file)?;
    }
    Ok(())
}

fn inventory(root: &Path) -> Result<Vec<PluginFile>> {
    let mut entries = Vec::new();
    let mut total = 0_u64;
    walk_inventory(root, root, &mut entries, &mut total)?;
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(entries)
}

fn walk_inventory(
    root: &Path,
    directory: &Path,
    entries: &mut Vec<PluginFile>,
    total: &mut u64,
) -> Result<()> {
    let mut children = fs::read_dir(directory)?.collect::<std::result::Result<Vec<_>, _>>()?;
    children.sort_by_key(|entry| entry.file_name().to_string_lossy().to_lowercase());
    for child in children {
        let path = child.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || is_reparse(&metadata) {
            return Err(plugin_error(
                "plugin source cannot contain symlinks or reparse points",
            ));
        }
        if metadata.is_dir() {
            let relative = path.strip_prefix(root).unwrap();
            if relative.components().count() > MAX_PATH_DEPTH {
                return Err(plugin_error("plugin path is too deep"));
            }
            walk_inventory(root, &path, entries, total)?;
            continue;
        }
        if !metadata.is_file() || is_hard_linked(&metadata) {
            return Err(plugin_error("plugin source can contain only regular files"));
        }
        // `Path` uses the host separator, while the bundle contract always
        // stores `/`-separated paths.  Normalize the inventory spelling on
        // Windows before applying the portable-path checks; otherwise a
        // perfectly valid nested bundle (`sub/payload.py`) would be rejected
        // because its relative `Path` renders as `sub\\payload.py`.
        let relative_path = path.strip_prefix(root).unwrap();
        let relative_raw = relative_path.to_string_lossy();
        let relative_raw = if cfg!(windows) {
            relative_raw.replace('\\', "/")
        } else {
            relative_raw.into_owned()
        };
        let relative = portable_path_text(&relative_raw)?;
        let (sha256, size) = hash_file(&path)?;
        if size > MAX_FILE_BYTES {
            return Err(plugin_error("plugin file exceeds the size limit"));
        }
        *total = total
            .checked_add(size)
            .ok_or_else(|| plugin_error("plugin source size overflow"))?;
        if *total > MAX_TOTAL_BYTES {
            return Err(plugin_error("plugin source exceeds the total size limit"));
        }
        if entries
            .iter()
            .any(|entry: &PluginFile| entry.path.to_lowercase() == relative.to_lowercase())
        {
            return Err(plugin_error("plugin source contains case-colliding paths"));
        }
        entries.push(PluginFile {
            path: relative,
            sha256,
            size_bytes: size,
            role: default_role(),
        });
        if entries.len() > MAX_FILES {
            return Err(plugin_error("plugin source contains too many files"));
        }
    }
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> Result<()> {
    if manifest.schema_version != PLUGIN_SCHEMA_VERSION {
        return Err(plugin_error(format!(
            "unsupported plugin schema: {:?}",
            manifest.schema_version
        )));
    }
    ensure_identifier(&manifest.id, "plugin.id")?;
    ensure_version(&manifest.version, "plugin.version")?;
    ensure_identifier(&manifest.kind, "plugin.kind")?;
    if !PLUGIN_KINDS.contains(&manifest.kind.as_str()) {
        return Err(plugin_error(format!(
            "unsupported plugin kind: {}",
            manifest.kind
        )));
    }
    ensure_text(&manifest.name, "plugin.name", 4096)?;
    ensure_text(&manifest.description, "plugin.description", 4096)?;
    validate_entrypoint(&manifest.entrypoint)?;
    validate_capabilities(&manifest.capabilities)?;
    validate_unique_strings(&manifest.platforms, "plugin.platforms", true)?;
    if manifest
        .platforms
        .iter()
        .any(|platform| !["windows", "linux", "macos"].contains(&platform.as_str()))
    {
        return Err(plugin_error(
            "plugin.platforms contains an unsupported value",
        ));
    }
    if !["process", "in-process"].contains(&manifest.isolation.as_str()) {
        return Err(plugin_error(format!(
            "unsupported plugin isolation: {}",
            manifest.isolation
        )));
    }
    validate_permissions(&manifest.permissions, "plugin.permissions")?;
    validate_requirements(&manifest.requires, "plugin.requires")?;
    validate_requirements(&manifest.dependencies, "plugin.dependencies")?;
    let mut paths = HashSet::new();
    for file in &manifest.files {
        let path = portable_path(Path::new(&file.path))?;
        if path == PLUGIN_FILE_NAME || !paths.insert(path.to_lowercase()) {
            return Err(plugin_error("plugin.files contains duplicate paths"));
        }
        ensure_digest(&file.sha256, "plugin file sha256")?;
        if file.size_bytes > MAX_FILE_BYTES {
            return Err(plugin_error("plugin file exceeds the size limit"));
        }
        ensure_identifier(&file.role, "plugin file role")?;
    }
    Ok(())
}

fn validate_profile(profile: &Profile) -> Result<()> {
    if profile.schema_version != PROFILE_SCHEMA_VERSION {
        return Err(plugin_error(format!(
            "unsupported profile schema: {:?}",
            profile.schema_version
        )));
    }
    ensure_identifier(&profile.name, "profile.name")?;
    let mut ids = HashSet::new();
    for reference in &profile.plugins {
        ensure_identifier(&reference.id, "profile plugin id")?;
        requirement_req(&reference.version)?;
        validate_permissions(&reference.permissions, "profile plugin permissions")?;
        validate_config(&reference.config)?;
        if !ids.insert(reference.id.clone()) {
            return Err(plugin_error(
                "profile.plugins must not contain duplicate plugin IDs",
            ));
        }
    }
    Ok(())
}

fn validate_entrypoint(value: &str) -> Result<()> {
    ensure_text(value, "plugin.entrypoint", 512)?;
    if value
        .chars()
        .any(|character| ['\n', '\r', '$', char::from(96), '&', '|', ';'].contains(&character))
    {
        return Err(plugin_error(
            "plugin.entrypoint contains executable shell syntax",
        ));
    }
    if let Some((module, attribute)) = value.split_once(':') {
        let module_start_valid = module
            .chars()
            .next()
            .is_some_and(|character| character == '_' || character.is_ascii_alphabetic());
        let attribute_start_valid = attribute
            .chars()
            .next()
            .is_some_and(|character| character == '_' || character.is_ascii_alphabetic());
        if !module_start_valid
            || !module.chars().all(|character| {
                character == '_' || character == '.' || character.is_ascii_alphanumeric()
            })
            || !attribute_start_valid
            || !attribute
                .chars()
                .all(|character| character == '_' || character.is_ascii_alphanumeric())
        {
            return Err(plugin_error("plugin.entrypoint must be module:attribute"));
        }
        return Ok(());
    }
    if value == PLUGIN_FILE_NAME {
        return Err(plugin_error("plugin.entrypoint cannot be the manifest"));
    }
    portable_path(Path::new(value)).map(|_| ())
}

fn validate_requirements(values: &BTreeMap<String, String>, label: &str) -> Result<()> {
    for (key, value) in values {
        ensure_identifier(key, &format!("{label} key"))?;
        requirement_req(value)?;
    }
    Ok(())
}

fn validate_permissions(values: &[String], label: &str) -> Result<()> {
    validate_unique_strings(values, label, false)?;
    if values
        .iter()
        .any(|value| !PERMISSIONS.contains(&value.as_str()))
    {
        return Err(plugin_error(format!(
            "{label} contains an unsupported permission"
        )));
    }
    Ok(())
}

fn validate_capabilities(values: &[String]) -> Result<()> {
    validate_unique_strings(values, "plugin.capabilities", true)?;
    for value in values {
        let mut characters = value.chars();
        let valid_start = characters
            .next()
            .is_some_and(|character| character.is_ascii_lowercase());
        if !valid_start
            || !characters.all(|character| {
                character.is_ascii_lowercase()
                    || character.is_ascii_digit()
                    || ['.', ':', '-', '_'].contains(&character)
            })
        {
            return Err(plugin_error(
                "plugin.capabilities contains an invalid value",
            ));
        }
    }
    Ok(())
}

fn validate_unique_strings(values: &[String], label: &str, nonempty: bool) -> Result<()> {
    if nonempty && values.is_empty() {
        return Err(plugin_error(format!(
            "{label} must contain at least one value"
        )));
    }
    let mut seen = HashSet::new();
    for value in values {
        if value.is_empty() || value.len() > 128 || !seen.insert(value) {
            return Err(plugin_error(format!(
                "{label} contains an invalid or duplicate value"
            )));
        }
    }
    Ok(())
}

fn requirement_req(value: &str) -> Result<VersionReq> {
    if value.is_empty() || value.len() > 256 {
        return Err(plugin_error("version requirement is empty or too long"));
    }
    VersionReq::parse(&canonical_requirement(value)?)
        .map_err(|error| plugin_error(format!("invalid version requirement: {error}")))
}

fn canonical_requirement(value: &str) -> Result<String> {
    let mut parts = Vec::new();
    for raw in value.split(',') {
        let item = raw.trim();
        let item = item
            .strip_prefix("==")
            .map_or_else(|| item.to_string(), |rest| format!("={rest}"));
        parts.push(item);
    }
    let normalized = parts.join(", ");
    VersionReq::parse(&normalized)
        .map(|request| request.to_string())
        .map_err(|error| plugin_error(format!("invalid version requirement: {error}")))
}

fn normalize_manifest_requirements(manifest: &mut Manifest) -> Result<()> {
    for requirement in manifest.requires.values_mut() {
        *requirement = canonical_requirement(requirement)?;
    }
    for requirement in manifest.dependencies.values_mut() {
        *requirement = canonical_requirement(requirement)?;
    }
    Ok(())
}

fn normalize_profile_requirements(profile: &mut Profile) -> Result<()> {
    for reference in &mut profile.plugins {
        reference.version = canonical_requirement(&reference.version)?;
    }
    Ok(())
}

fn ensure_version(value: &str, label: &str) -> Result<()> {
    Version::parse(value)
        .map(|_| ())
        .map_err(|error| plugin_error(format!("{label} must be a semantic version: {error}")))
}

fn ensure_identifier(value: &str, label: &str) -> Result<()> {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return Err(plugin_error(format!(
            "{label} must be a non-empty identifier"
        )));
    };
    if !first.is_ascii_lowercase()
        || value.len() > 128
        || !characters.all(|character| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || ['_', '.', '-'].contains(&character)
        })
    {
        return Err(plugin_error(format!("{label} has an invalid identifier")));
    }
    Ok(())
}

fn ensure_text(value: &str, label: &str, maximum: usize) -> Result<()> {
    if value.is_empty()
        || value.len() > maximum
        || value.chars().any(|character| character.is_control())
    {
        return Err(plugin_error(format!(
            "{label} must be non-empty printable text"
        )));
    }
    Ok(())
}

fn portable_path(path: &Path) -> Result<String> {
    let raw = path.to_string_lossy();
    if path.is_absolute() {
        return Err(plugin_error("plugin paths must be portable relative paths"));
    }
    portable_path_text(&raw)
}

fn portable_path_text(raw: &str) -> Result<String> {
    if raw.is_empty()
        || raw.starts_with('/')
        || raw.contains('\\')
        || raw.chars().any(|character| "<>:\"|?*".contains(character))
    {
        return Err(plugin_error("plugin paths must be portable relative paths"));
    }
    let mut parts = Vec::new();
    for part in raw.split('/') {
        if part.is_empty()
            || part == "."
            || part == ".."
            || part.ends_with('.')
            || part.ends_with(' ')
            || !part.is_ascii()
            || part.chars().any(|character| character.is_control())
        {
            return Err(plugin_error("plugin paths must be portable relative paths"));
        }
        let stem = part.split('.').next().unwrap_or_default().to_uppercase();
        let reserved_port = (stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && stem.as_bytes()[3].is_ascii_digit();
        if ["CON", "PRN", "AUX", "NUL"].contains(&stem.as_str()) || reserved_port {
            return Err(plugin_error("plugin path uses a reserved component"));
        }
        parts.push(part.to_owned());
    }
    if parts.is_empty() || parts.len() > MAX_PATH_DEPTH {
        return Err(plugin_error("plugin paths must be bounded relative paths"));
    }
    let canonical = parts.join("/");
    if raw != canonical {
        return Err(plugin_error("plugin paths must be portable relative paths"));
    }
    Ok(canonical)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, maximum: u64, label: &str) -> Result<T> {
    let path = absolute_path(path)?;
    if path_has_link_component(&path)? {
        return Err(plugin_error(format!(
            "{label} path cannot contain symlinks or reparse points"
        )));
    }
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            Error::Io(error)
        } else {
            plugin_error(format!("cannot read {label}: {error}"))
        }
    })?;
    if metadata.file_type().is_symlink() || is_reparse(&metadata) || !metadata.is_file() {
        return Err(plugin_error(format!(
            "{label} must be a regular non-symlink file"
        )));
    }
    if metadata.len() > maximum {
        return Err(plugin_error(format!("{label} exceeds the size limit")));
    }
    let bytes = fs::read(&path)?;
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    let value = StrictJsonValue
        .deserialize(&mut deserializer)
        .map_err(|error| plugin_error(format!("invalid {label}: {error}")))?;
    deserializer
        .end()
        .map_err(|error| plugin_error(format!("invalid {label}: {error}")))?;
    serde_json::from_value(value).map_err(|error| plugin_error(format!("invalid {label}: {error}")))
}

/// Deserialize JSON while rejecting duplicate object keys.
///
/// `serde_json::Value` normally keeps the last value for a duplicate key.  A
/// plugin manifest or profile is a security-sensitive declaration, so silently
/// accepting that ambiguity would make the Rust CLI disagree with the Python
/// SDK and could hide an unexpected permission or entrypoint value.
struct StrictJsonValue;

impl<'de> DeserializeSeed<'de> for StrictJsonValue {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("JSON number must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::String(value.into()))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictJsonValue.deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictJsonValue)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.contains_key(&key) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON object field: {key}"
                )));
            }
            let value = map.next_value_seed(StrictJsonValue)?;
            object.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}

fn canonical_regular_directory(path: &Path) -> Result<PathBuf> {
    let absolute = absolute_path(path)?;
    if path_has_link_component(&absolute)? || !absolute.is_dir() {
        return Err(plugin_error("plugin source must be a regular directory"));
    }
    Ok(fs::canonicalize(absolute)?)
}

fn ensure_directory(path: &Path, label: &str) -> Result<()> {
    if path_has_link_component(path)? {
        return Err(plugin_error(format!(
            "{label} cannot contain symlinks or reparse points"
        )));
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || is_reparse(&metadata) || !metadata.is_dir() {
                return Err(plugin_error(format!("{label} must be a regular directory")));
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = path
                .parent()
                .ok_or_else(|| plugin_error(format!("cannot create {label}")))?;
            ensure_directory(parent, label)?;
            match fs::create_dir(path) {
                Ok(()) => Ok(()),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    ensure_directory(path, label)
                }
                Err(error) => Err(error.into()),
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn path_has_link_component(path: &Path) -> Result<bool> {
    let mut current = path;
    loop {
        if let Ok(metadata) = fs::symlink_metadata(current)
            && (metadata.file_type().is_symlink() || is_reparse(&metadata))
        {
            return Ok(true);
        }
        let Some(parent) = current.parent() else {
            return Ok(false);
        };
        if parent == current {
            return Ok(false);
        }
        current = parent;
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn is_symlink_or_reparse(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_symlink() || is_reparse(&metadata)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn sorted_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(root)?
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    paths.sort_by_key(|path| path.file_name().unwrap().to_string_lossy().to_lowercase());
    Ok(paths)
}

fn hash_file(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut size = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        size += count as u64;
    }
    Ok((format!("{:x}", digest.finalize()), size))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn ensure_digest(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase())
    {
        return Err(plugin_error(format!(
            "{label} must be a lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

fn canonical_json(value: &Value) -> Vec<u8> {
    fn normalize(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut sorted = BTreeMap::new();
                for (key, value) in object {
                    sorted.insert(key.clone(), normalize(value));
                }
                serde_json::to_value(sorted).expect("BTreeMap is serializable")
            }
            Value::Array(values) => Value::Array(values.iter().map(normalize).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_vec(&normalize(value)).expect("normalized JSON is serializable")
}

fn profile_digest(profile: &Profile) -> String {
    sha256_bytes(&canonical_json(
        &serde_json::to_value(profile).expect("profile is serializable"),
    ))
}

fn resolved_profile_digest(profile: &Profile, plugins: &[ResolvedPlugin]) -> String {
    let digest_plugins = plugins
        .iter()
        .map(|plugin| {
            json!({
                "id": plugin.id,
                "version": plugin.version,
                "content_sha256": plugin.content_sha256,
                "permissions": plugin.permissions,
                "config": plugin.config,
            })
        })
        .collect::<Vec<_>>();
    sha256_bytes(&canonical_json(
        &json!({"profile": profile, "plugins": digest_plugins}),
    ))
}

fn validate_config(config: &Value) -> Result<Value> {
    if !config.is_object() {
        return Err(plugin_error("profile plugin config must be an object"));
    }
    let bytes = canonical_json(config);
    if bytes.len() > 64 * 1024 {
        return Err(plugin_error("profile plugin config exceeds the size limit"));
    }
    Ok(config.clone())
}

fn sorted_unique(values: &[String]) -> Vec<String> {
    let mut result = values.to_vec();
    result.sort();
    result.dedup();
    result
}

fn platform_name() -> &'static str {
    if cfg!(windows) {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
}

fn plugin_error(message: impl Into<String>) -> Error {
    Error::Plugin(message.into())
}

fn is_reparse(metadata: &Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        let _ = metadata;
        false
    }
}

fn is_hard_linked(metadata: &Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        metadata.nlink() > 1
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Manager, PLUGIN_SCHEMA_VERSION, PROFILE_SCHEMA_VERSION, Profile, ProfileRef,
        canonical_json, canonical_requirement, ensure_text, portable_path, requirement_req,
        validate_config,
    };
    use serde_json::json;
    use tempfile::tempdir;

    fn bundle(root: &std::path::Path, id: &str, version: &str) -> std::path::PathBuf {
        let path = root.join(format!("{id}-{version}"));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("plugin.py"), b"def create():\n    return None\n").unwrap();
        std::fs::write(
            path.join("glr-plugin.json"),
            serde_json::to_vec(&json!({
                "schema_version": PLUGIN_SCHEMA_VERSION,
                "id": id,
                "version": version,
                "kind": "learner",
                "name": "Fixture",
                "description": "A test fixture.",
                "entrypoint": "plugin:create",
                "capabilities": ["learner.ppo"],
                "permissions": ["read:environment"]
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn inspect_install_and_resolve_are_data_only() {
        let root = tempdir().unwrap();
        let source = bundle(root.path(), "fixture-plugin", "1.2.3");
        let project = root.path().join("project");
        let manager = Manager::new(&project).unwrap();
        let inspection = manager.inspect(&source).unwrap();
        assert_eq!(inspection.manifest.id, "fixture-plugin");
        assert!(inspection.content_sha256.len() == 64);
        let installation = manager
            .install(&source, Some(&inspection.content_sha256))
            .unwrap();
        assert_eq!(installation.id, "fixture-plugin");
        assert_eq!(manager.list().unwrap().len(), 1);

        let profile = Profile {
            schema_version: PROFILE_SCHEMA_VERSION.into(),
            name: "training".into(),
            plugins: vec![ProfileRef {
                id: "fixture-plugin".into(),
                version: "^1.0".into(),
                enabled: true,
                permissions: vec!["read:environment".into()],
                config: json!({"batch": 32}),
            }],
        };
        manager.save_profile(&profile).unwrap();
        let (resolved, digest) = manager.resolve(&profile).unwrap();
        assert_eq!(resolved[0].id, "fixture-plugin");
        assert_eq!(resolved[0].path, ".glr/plugins/fixture-plugin/1.2.3");
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn nested_payload_paths_use_the_portable_separator() {
        let root = tempdir().unwrap();
        let source = bundle(root.path(), "nested-plugin", "1.0.0");
        std::fs::create_dir_all(source.join("nested")).unwrap();
        std::fs::write(source.join("nested").join("payload.bin"), b"nested").unwrap();
        let manager = Manager::new(&root.path().join("project")).unwrap();
        let inspection = manager.inspect(&source).unwrap();
        assert!(
            inspection
                .files
                .iter()
                .any(|file| file.path == "nested/payload.bin")
        );
        manager
            .install(&source, Some(&inspection.content_sha256))
            .unwrap();
    }

    #[test]
    fn malformed_manifest_and_permission_escalation_fail_closed() {
        let root = tempdir().unwrap();
        let source = bundle(root.path(), "fixture-plugin", "1.0.0");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(source.join("glr-plugin.json")).unwrap())
                .unwrap();
        manifest["unknown"] = json!(true);
        std::fs::write(
            source.join("glr-plugin.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let manager = Manager::new(&root.path().join("project")).unwrap();
        assert!(manager.inspect(&source).is_err());

        let source = bundle(root.path(), "fixture-plugin", "1.0.0");
        let manager = Manager::new(&root.path().join("project-2")).unwrap();
        manager.install(&source, None).unwrap();
        let profile = Profile {
            schema_version: PROFILE_SCHEMA_VERSION.into(),
            name: "bad".into(),
            plugins: vec![ProfileRef {
                id: "fixture-plugin".into(),
                version: "*".into(),
                enabled: true,
                permissions: vec!["runtime:act".into()],
                config: json!({}),
            }],
        };
        assert!(manager.resolve(&profile).is_err());
    }

    #[test]
    fn semver_requirement_forms_are_canonical_and_compatible() {
        for requirement in [
            "1.*", "1.2.*", "1.x", "1.2.X", "=1.2.3", "==1.2.3", "<=1.2", ">1",
        ] {
            assert!(requirement_req(requirement).is_ok(), "{requirement}");
        }
        assert_eq!(
            canonical_requirement(" >=1.0.0, <2.0.0 ").unwrap(),
            ">=1.0.0, <2.0.0"
        );
        assert_eq!(canonical_requirement("1.x").unwrap(), "1.*");
    }

    #[test]
    fn portable_paths_are_ascii_for_cross_platform_identity() {
        assert!(portable_path(std::path::Path::new("payload-é.py")).is_err());
        for path in [
            "bad<.py", "bad>.py", "bad:.py", "bad\".py", "bad|.py", "bad?.py", "bad*.py",
        ] {
            assert!(portable_path(std::path::Path::new(path)).is_err(), "{path}");
        }
    }

    #[test]
    fn text_limits_use_utf8_bytes_across_runtimes() {
        let boundary = "界".repeat(1365); // 4,095 UTF-8 bytes.
        assert!(ensure_text(&boundary, "text", 4096).is_ok());
        let over_limit = "界".repeat(1366); // 4,098 UTF-8 bytes.
        assert!(ensure_text(&over_limit, "text", 4096).is_err());
    }

    #[test]
    fn canonical_json_numbers_preserve_cross_language_spelling() {
        let parsed: serde_json::Value = serde_json::from_str("9.999999999999999e-06").unwrap();
        assert_eq!(
            canonical_json(&parsed),
            b"9.999999999999999e-6",
            "JSON float parsing must retain the correctly rounded value"
        );
        let value = json!({
            "a": 1e-7,
            "b": 1e-5,
            "c": 1e-6,
            "d": 1e-4,
            "e": 1e20,
            "f": 1e21,
            "g": 1.2345678901234568e-5,
            "h": 1.2345678901234567e20,
            "u": "é😀"
        });
        assert_eq!(
            canonical_json(&value),
            "{\"a\":1e-7,\"b\":0.00001,\"c\":1e-6,\"d\":0.0001,\"e\":1e+20,\"f\":1e+21,\"g\":0.000012345678901234568,\"h\":1.2345678901234567e+20,\"u\":\"é😀\"}".as_bytes()
        );
    }

    #[test]
    fn profile_config_accepts_roundtripped_floating_point_numbers() {
        assert!(validate_config(&json!({"nested": {"value": 9.999999999999999e-6}})).is_ok());
        assert!(validate_config(&json!({"count": 7})).is_ok());
    }

    #[test]
    fn duplicate_json_fields_are_rejected_before_manifest_deserialization() {
        let root = tempdir().unwrap();
        let source = bundle(root.path(), "duplicate-plugin", "1.0.0");
        std::fs::write(
            source.join("glr-plugin.json"),
            br#"{
                "schema_version": "glr.plugin.v1",
                "schema_version": "glr.plugin.v1",
                "id": "duplicate-plugin",
                "version": "1.0.0",
                "kind": "learner",
                "name": "Duplicate",
                "description": "Duplicate field fixture.",
                "entrypoint": "plugin.py:create",
                "capabilities": ["learner.ppo"]
            }"#,
        )
        .unwrap();
        let manager = Manager::new(&root.path().join("project")).unwrap();
        assert!(manager.inspect(&source).is_err());
    }
}
