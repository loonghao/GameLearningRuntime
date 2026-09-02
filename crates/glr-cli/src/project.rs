use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};

pub const PROJECT_FILE_NAME: &str = "glr-project.json";
pub const PROJECT_SCHEMA_VERSION: &str = "glr.project.v1";

const PLACEHOLDERS: &[&str] = &[
    "bridge_path",
    "bundle",
    "capture_index",
    "capture_video",
    "evaluation_path",
    "goal_path",
    "previous_evaluation_path",
    "previous_research_path",
    "project_root",
    "research_path",
    "run_dir",
    "run_id",
    "trial_id",
    "trial_path",
    "trainer_result_path",
];

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectCommand {
    pub argv: Vec<String>,
}

impl ProjectCommand {
    fn validate(&self, path: &str) -> Result<()> {
        if self.argv.is_empty() {
            return Err(Error::Invalid(format!("{path}.argv cannot be empty")));
        }
        let allowed = PLACEHOLDERS.iter().copied().collect::<HashSet<_>>();
        for argument in &self.argv {
            if argument.is_empty() || argument.chars().any(char::is_control) {
                return Err(Error::Invalid(format!(
                    "{path}.argv entries must be non-empty printable strings"
                )));
            }
            if argument.contains('{') || argument.contains('}') {
                let Some(placeholder) = argument
                    .strip_prefix('{')
                    .and_then(|value| value.strip_suffix('}'))
                else {
                    return Err(Error::Invalid(
                        "command placeholders must occupy a complete argv entry".into(),
                    ));
                };
                if !allowed.contains(placeholder) {
                    return Err(Error::Invalid(format!(
                        "unsupported command placeholder: {argument}"
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn expand(&self, values: &HashMap<String, PathBuf>) -> Result<Vec<String>> {
        self.argv
            .iter()
            .map(|argument| {
                if let Some(placeholder) = argument
                    .strip_prefix('{')
                    .and_then(|value| value.strip_suffix('}'))
                {
                    values
                        .get(placeholder)
                        .map(|value| value.to_string_lossy().into_owned())
                        .ok_or_else(|| {
                            Error::Invalid(format!(
                                "missing command placeholder value: {placeholder}"
                            ))
                        })
                } else {
                    Ok(argument.clone())
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureConfig {
    pub argv: Vec<String>,
    pub required: bool,
    pub stop: String,
    pub video_file: String,
    pub index_file: String,
    pub codec: String,
    pub frame_rate: f64,
    pub width: u32,
    pub height: u32,
}

impl CaptureConfig {
    pub fn command(&self) -> ProjectCommand {
        ProjectCommand {
            argv: self.argv.clone(),
        }
    }

    fn validate(&self) -> Result<()> {
        self.command().validate("project.capture")?;
        if !matches!(self.stop.as_str(), "stdin-q" | "terminate") {
            return Err(Error::Invalid(
                "project.capture.stop must be 'stdin-q' or 'terminate'".into(),
            ));
        }
        portable_relative(&self.video_file, "project.capture.video_file")?;
        portable_relative(&self.index_file, "project.capture.index_file")?;
        validate_identifier(&self.codec, "project.capture.codec")?;
        if !self.frame_rate.is_finite() || self.frame_rate <= 0.0 {
            return Err(Error::Invalid(
                "project.capture.frame_rate must be positive and finite".into(),
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err(Error::Invalid(
                "project.capture width and height must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectFile {
    schema_version: String,
    environment_id: String,
    environment_family: String,
    protocol_version: String,
    data_dir: String,
    bridge_path: String,
    runtime: ProjectCommand,
    trainer: ProjectCommand,
    player: ProjectCommand,
    researcher: Option<ProjectCommand>,
    planner: Option<ProjectCommand>,
    evaluator: Option<ProjectCommand>,
    capture: Option<CaptureConfig>,
}

#[derive(Debug, Clone)]
pub struct Project {
    pub root: PathBuf,
    pub environment_id: String,
    pub environment_family: String,
    pub protocol_version: String,
    pub data_dir: PathBuf,
    pub bridge_path: PathBuf,
    pub runtime: ProjectCommand,
    pub trainer: ProjectCommand,
    pub player: ProjectCommand,
    pub researcher: Option<ProjectCommand>,
    pub planner: Option<ProjectCommand>,
    pub evaluator: Option<ProjectCommand>,
    pub capture: Option<CaptureConfig>,
}

pub fn find_project(start: &Path) -> Result<PathBuf> {
    let mut current = if start.is_file() {
        start.parent().unwrap_or(start).to_path_buf()
    } else {
        start.to_path_buf()
    };
    if !current.is_absolute() {
        current = std::env::current_dir()?.join(current);
    }
    loop {
        let candidate = current.join(PROJECT_FILE_NAME);
        if candidate.is_file() && !candidate.is_symlink() {
            return Ok(candidate);
        }
        if !current.pop() {
            return Err(Error::Missing(start.join(PROJECT_FILE_NAME)));
        }
    }
}

pub fn load_project(requested: &Path) -> Result<Project> {
    let config_path = if requested.is_dir() {
        requested.join(PROJECT_FILE_NAME)
    } else if requested
        .file_name()
        .is_some_and(|name| name == PROJECT_FILE_NAME)
    {
        requested.to_path_buf()
    } else {
        find_project(requested)?
    };
    if config_path.is_symlink() || !config_path.is_file() {
        return Err(Error::Missing(config_path));
    }
    let root = fs::canonicalize(
        config_path
            .parent()
            .ok_or_else(|| Error::Invalid("project config has no parent".into()))?,
    )?;
    let bytes = fs::read(&config_path)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(Error::Invalid(
            "project config exceeds the 8 MiB limit".into(),
        ));
    }
    let value: ProjectFile = serde_json::from_slice(&bytes)?;
    if value.schema_version != PROJECT_SCHEMA_VERSION {
        return Err(Error::Invalid(format!(
            "project.schema_version must be {PROJECT_SCHEMA_VERSION:?}"
        )));
    }
    validate_identifier(&value.environment_id, "project.environment_id")?;
    validate_identifier(&value.environment_family, "project.environment_family")?;
    validate_text(&value.protocol_version, "project.protocol_version")?;
    value.runtime.validate("project.runtime")?;
    value.trainer.validate("project.trainer")?;
    value.player.validate("project.player")?;
    for (name, command) in [
        ("project.researcher", value.researcher.as_ref()),
        ("project.planner", value.planner.as_ref()),
        ("project.evaluator", value.evaluator.as_ref()),
    ] {
        if let Some(command) = command {
            command.validate(name)?;
        }
    }
    if let Some(capture) = &value.capture {
        capture.validate()?;
    }
    let data_dir = inside_project(&root, &value.data_dir, "project.data_dir")?;
    let bridge_path = inside_project(&root, &value.bridge_path, "project.bridge_path")?;
    if !bridge_path.exists() {
        return Err(Error::Missing(bridge_path));
    }
    Ok(Project {
        root,
        environment_id: value.environment_id,
        environment_family: value.environment_family,
        protocol_version: value.protocol_version,
        data_dir,
        bridge_path,
        runtime: value.runtime,
        trainer: value.trainer,
        player: value.player,
        researcher: value.researcher,
        planner: value.planner,
        evaluator: value.evaluator,
        capture: value.capture,
    })
}

fn inside_project(root: &Path, value: &str, label: &str) -> Result<PathBuf> {
    let relative = portable_relative(value, label)?;
    let joined = root.join(relative);
    let normalized = normalize_path(&joined);
    if !normalized.starts_with(root) {
        return Err(Error::Invalid(format!(
            "{label} must stay inside the project root"
        )));
    }
    Ok(normalized)
}

fn portable_relative(value: &str, label: &str) -> Result<PathBuf> {
    if value.is_empty() || value.contains('\\') || value.contains(':') {
        return Err(Error::Invalid(format!(
            "{label} must be a portable project-relative path"
        )));
    }
    let path = PathBuf::from(value);
    if path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(Error::Invalid(format!(
            "{label} must be a portable project-relative path"
        )));
    }
    Ok(path)
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

pub fn validate_identifier(value: &str, label: &str) -> Result<()> {
    let valid = !value.is_empty()
        && value.as_bytes()[0].is_ascii_lowercase()
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.-".contains(&byte)
        });
    if valid {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "{label} must match ^[a-z][a-z0-9_.-]*$"
        )))
    }
}

fn validate_text(value: &str, label: &str) -> Result<()> {
    if value.is_empty() || value.chars().any(char::is_control) {
        Err(Error::Invalid(format!(
            "{label} must be non-empty printable text"
        )))
    } else {
        Ok(())
    }
}
