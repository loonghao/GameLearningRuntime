//! Persisted default-goal bindings for one GLR project.
//!
//! A binding records which `glr.agent-goal.v1` file a project falls back to when
//! `goal run` is invoked without `--goal`, plus the optional `glr.run-context.v1`
//! file that freezes the training and reward inputs that make the goal
//! executable. Bindings only *select* a goal and a context: the goal stays
//! structured metadata, is copied into the run directory as an auditable
//! receipt, and never shapes rewards or judges completion.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::emit;
use crate::contracts::{AgentGoal, read_json, sha256_file, write_json};
use crate::error::{Error, Result};
use crate::project::Project;
use crate::run_context::{RunContext, regular_project_file};

pub const GOAL_BINDING_SCHEMA_VERSION: &str = "glr.goal-binding.v1";
pub const GOAL_BINDING_FILE_NAME: &str = "goal-binding.json";
const MAX_GOALS: usize = 64;

/// One saved goal reference and the context that makes it executable.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalBinding {
    pub goal_id: String,
    pub objective: String,
    pub environment_family: String,
    /// Project-relative POSIX path of the bound `glr.agent-goal.v1` file.
    pub goal_path: String,
    pub goal_sha256: String,
    /// Project-relative POSIX path of the bound `glr.run-context.v1` file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_path: Option<String>,
}

/// The on-disk binding store: every saved goal plus one active pointer.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoalBindingFile {
    pub schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_goal_id: Option<String>,
    #[serde(default)]
    pub goals: BTreeMap<String, GoalBinding>,
}

impl Default for GoalBindingFile {
    fn default() -> Self {
        Self {
            schema_version: GOAL_BINDING_SCHEMA_VERSION.to_owned(),
            active_goal_id: None,
            goals: BTreeMap::new(),
        }
    }
}

/// Whether the goal came from an explicit `--goal` or from the saved default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalSource {
    Explicit,
    Default,
}

impl GoalSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Default => "default",
        }
    }
}

pub fn path(project: &Project) -> PathBuf {
    project.data_dir.join(GOAL_BINDING_FILE_NAME)
}

pub fn load(project: &Project) -> Result<GoalBindingFile> {
    let path = path(project);
    if !path.exists() {
        return Ok(GoalBindingFile::default());
    }
    let file: GoalBindingFile = read_json(&path, "goal binding")?;
    if file.schema_version != GOAL_BINDING_SCHEMA_VERSION {
        return Err(Error::Invalid(format!(
            "goal binding schema_version must be {GOAL_BINDING_SCHEMA_VERSION:?}"
        )));
    }
    for (key, binding) in &file.goals {
        if key != &binding.goal_id {
            return Err(Error::Invalid(format!(
                "goal binding key {key:?} does not match goal_id {:?}",
                binding.goal_id
            )));
        }
    }
    if let Some(active) = &file.active_goal_id
        && !file.goals.contains_key(active)
    {
        return Err(Error::Invalid(format!(
            "goal binding active_goal_id {active:?} is not a saved goal"
        )));
    }
    Ok(file)
}

fn save(project: &Project, file: &GoalBindingFile) -> Result<()> {
    fs::create_dir_all(&project.data_dir)?;
    write_json(&path(project), file)
}

pub fn active(project: &Project) -> Result<Option<GoalBinding>> {
    let file = load(project)?;
    Ok(file
        .active_goal_id
        .as_ref()
        .and_then(|goal_id| file.goals.get(goal_id).cloned()))
}

/// Save (or re-save) a goal as this project's active default goal.
///
/// The goal is parsed and validated eagerly so an unusable goal is rejected at
/// bind time instead of at the moment training starts.
pub fn bind(project: &Project, requested: &Path, context: Option<&Path>) -> Result<GoalBinding> {
    let relative = project_relative(project, requested)?;
    let (goal_path, relative) = regular_project_file(&project.root, &relative, "goal")?;
    let goal: AgentGoal = read_json(&goal_path, "agent goal")?;
    goal.validate()?;
    if goal.environment_family != project.environment_family {
        return Err(Error::Contract(format!(
            "goal {:?} environment_family {:?} does not match project environment_family {:?}",
            goal.goal_id, goal.environment_family, project.environment_family
        )));
    }
    let context_path = match context {
        Some(requested) => {
            RunContext::load(project, requested)?;
            Some(requested.to_string_lossy().replace('\\', "/"))
        }
        None => None,
    };
    let binding = GoalBinding {
        goal_id: goal.goal_id.clone(),
        objective: goal.objective.clone(),
        environment_family: goal.environment_family.clone(),
        goal_path: relative,
        goal_sha256: sha256_file(&goal_path)?,
        context_path,
    };
    let mut file = load(project)?;
    if !file.goals.contains_key(&binding.goal_id) && file.goals.len() >= MAX_GOALS {
        return Err(Error::Invalid(format!(
            "a project cannot bind more than {MAX_GOALS} goals"
        )));
    }
    file.goals.insert(binding.goal_id.clone(), binding.clone());
    file.active_goal_id = Some(binding.goal_id.clone());
    save(project, &file)?;
    Ok(binding)
}

/// Make one already-saved goal the active default goal.
pub fn select(project: &Project, goal_id: &str) -> Result<GoalBinding> {
    let mut file = load(project)?;
    let binding = file.goals.get(goal_id).cloned().ok_or_else(|| {
        Error::Invalid(format!(
            "no saved goal {goal_id:?}; run `glr goal set --goal <path>` to bind it"
        ))
    })?;
    file.active_goal_id = Some(goal_id.to_owned());
    save(project, &file)?;
    Ok(binding)
}

/// Resolve the goal file for one invocation: an explicit `--goal` always wins.
pub fn resolve(project: &Project, requested: Option<&Path>) -> Result<(PathBuf, GoalSource)> {
    if let Some(requested) = requested {
        return Ok((absolute(requested)?, GoalSource::Explicit));
    }
    let binding = active(project)?.ok_or_else(|| {
        Error::Invalid(
            "goal run needs --goal or a saved default goal; run `glr goal set --goal <path>` first"
                .into(),
        )
    })?;
    let resolved = join_relative(&project.root, &binding.goal_path);
    if !resolved.is_file() {
        return Err(Error::Invalid(format!(
            "default goal {:?} points at {:?}, which no longer exists; run `glr goal set --goal <path>` to rebind",
            binding.goal_id, binding.goal_path
        )));
    }
    Ok((resolved, GoalSource::Default))
}

/// The context bound to the active goal, used only when `--context` is omitted.
pub fn active_context(project: &Project) -> Result<Option<PathBuf>> {
    Ok(active(project)?
        .and_then(|binding| binding.context_path)
        .map(PathBuf::from))
}

/// Join a stored POSIX-style relative path onto the canonical project root.
///
/// A canonical Windows root carries the `\?\` prefix, which does not accept
/// forward slashes, so components are pushed individually.
fn join_relative(root: &Path, relative: &str) -> PathBuf {
    let mut joined = root.to_path_buf();
    for component in Path::new(relative).components() {
        match component {
            Component::Normal(value) => joined.push(value),
            Component::ParentDir => {
                joined.pop();
            }
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir => {}
        }
    }
    joined
}

fn absolute(requested: &Path) -> Result<PathBuf> {
    if requested.is_absolute() {
        Ok(requested.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(requested))
    }
}

/// Rewrite a caller-supplied path as a project-relative one, rejecting escapes.
///
/// The path is read relative to the project root first and to the working
/// directory second, so `glr --project .` and an out-of-tree project both bind
/// the same file. Both sides are canonicalized: the project root already is, and
/// a goal reached through a link is attributed to its real location.
fn project_relative(project: &Project, requested: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    if !requested.is_absolute() {
        candidates.push(join_relative(&project.root, &requested.to_string_lossy()));
    }
    candidates.push(absolute(requested)?);
    let mut resolved = None;
    for candidate in candidates {
        if let Ok(canonical) = fs::canonicalize(&candidate) {
            resolved = Some(canonical);
            break;
        }
    }
    let resolved = match resolved {
        Some(resolved) => resolved,
        None => return Err(Error::Missing(absolute(requested)?)),
    };
    let relative = resolved.strip_prefix(&project.root).map_err(|_| {
        Error::Invalid(format!(
            "goal must stay inside the project root: {}",
            project.root.display()
        ))
    })?;
    if relative.as_os_str().is_empty() {
        return Err(Error::Invalid(
            "goal must name a file inside the project".into(),
        ));
    }
    Ok(relative.to_path_buf())
}

fn source_status(project: &Project, binding: &GoalBinding) -> &'static str {
    match sha256_file(&join_relative(&project.root, &binding.goal_path)) {
        Ok(digest) if digest == binding.goal_sha256 => "unchanged",
        Ok(_) => "changed",
        Err(_) => "missing",
    }
}

fn context_status(project: &Project, binding: &GoalBinding) -> &'static str {
    match &binding.context_path {
        None => "unbound",
        Some(path) => match RunContext::load(project, Path::new(path)) {
            Ok(_) => "bound",
            Err(_) => "unresolved",
        },
    }
}

fn describe(project: &Project, binding: &GoalBinding) -> Value {
    json!({
        "goal_id": binding.goal_id,
        "objective": binding.objective,
        "environment_family": binding.environment_family,
        "goal_path": binding.goal_path,
        "goal_sha256": binding.goal_sha256,
        "context_path": binding.context_path,
        "source_status": source_status(project, binding),
        "context_status": context_status(project, binding),
    })
}

pub fn set_command(
    project: &Project,
    requested: &Path,
    context: Option<&Path>,
    as_json: bool,
) -> Result<i32> {
    let binding = bind(project, requested, context)?;
    emit(
        "goal.set",
        &json!({
            "active_goal_id": binding.goal_id,
            "path": path(project),
            "goal": describe(project, &binding),
        }),
        as_json,
    )?;
    Ok(0)
}

pub fn show_command(project: &Project, goal_id: Option<&str>, as_json: bool) -> Result<i32> {
    let file = load(project)?;
    let binding = match goal_id {
        Some(goal_id) => {
            let binding = file.goals.get(goal_id).cloned().ok_or_else(|| {
                Error::Invalid(format!("no saved goal {goal_id:?} in this project"))
            })?;
            Some(binding)
        }
        None => active(project)?,
    };
    emit(
        "goal.show",
        &json!({
            "path": path(project),
            "active_goal_id": file.active_goal_id,
            "goal": binding.as_ref().map(|binding| describe(project, binding)),
        }),
        as_json,
    )?;
    Ok(0)
}

pub fn use_command(project: &Project, goal_id: &str, as_json: bool) -> Result<i32> {
    let binding = select(project, goal_id)?;
    emit(
        "goal.use",
        &json!({
            "active_goal_id": binding.goal_id,
            "goal": describe(project, &binding),
        }),
        as_json,
    )?;
    Ok(0)
}

pub fn list_command(project: &Project, as_json: bool) -> Result<i32> {
    let file = load(project)?;
    let goals = file
        .goals
        .values()
        .map(|binding| describe(project, binding))
        .collect::<Vec<_>>();
    emit(
        "goal.list",
        &json!({
            "path": path(project),
            "active_goal_id": file.active_goal_id,
            "goals": goals,
        }),
        as_json,
    )?;
    Ok(0)
}

/// Doctor-facing summary: what is bound, without failing on an empty store.
pub fn doctor_metadata(project: &Project) -> Value {
    match load(project) {
        Ok(file) => json!({
            "path": path(project),
            "active_goal_id": file.active_goal_id,
            "goal_count": file.goals.len(),
        }),
        Err(error) => json!({
            "path": path(project),
            "error": error.to_string(),
        }),
    }
}

/// Receipt metadata recorded with every `goal run`.
pub fn run_metadata(binding: Option<&GoalBinding>, source: GoalSource) -> Value {
    json!({
        "source": source.as_str(),
        "goal_id": binding.map(|binding| binding.goal_id.clone()),
        "goal_path": binding.map(|binding| binding.goal_path.clone()),
        "context_path": binding.and_then(|binding| binding.context_path.clone()),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        GOAL_BINDING_SCHEMA_VERSION, GoalSource, active, bind, doctor_metadata, load, resolve,
        select,
    };
    use crate::project::Project;
    use serde_json::json;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn project(root: &Path) -> Project {
        fs::create_dir_all(root.join("bridge")).unwrap();
        fs::write(
            root.join("glr-project.json"),
            serde_json::to_vec_pretty(&json!({
                "schema_version": "glr.project.v1",
                "environment_id": "example.context-v1",
                "environment_family": "test",
                "protocol_version": "1.0",
                "data_dir": ".glr",
                "bridge_path": "bridge",
                "runtime": {"argv": ["true"]},
                "trainer": {"argv": ["true"]},
                "player": {"argv": ["true"]},
                "researcher": null,
                "planner": null,
                "evaluator": null,
                "capture": null
            }))
            .unwrap(),
        )
        .unwrap();
        crate::project::load_project(root).unwrap()
    }

    fn goal(root: &Path, goal_id: &str) -> PathBuf {
        fs::create_dir_all(root.join("goals")).unwrap();
        let path = root.join("goals").join(format!("{goal_id}.json"));
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": "glr.agent-goal.v1",
                "goal_id": goal_id,
                "objective": "reach the destination",
                "environment_family": "test",
                "success_criteria": [{
                    "metric": "progress",
                    "operator": "gte",
                    "target": 1.0,
                    "source": "evaluator"
                }],
                "budget": {
                    "max_trials": 2,
                    "max_training_steps": 100,
                    "max_wall_seconds": 60,
                    "max_research_sources": 4
                },
                "allowed_research_media": ["runtime-trace"]
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn binding_persists_the_goal_and_activates_it() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        let goal = goal(root.path(), "goal.reach-destination");
        let binding = bind(&project, &goal, None).unwrap();
        assert_eq!(binding.goal_id, "goal.reach-destination");
        assert_eq!(binding.goal_path, "goals/goal.reach-destination.json");
        assert_eq!(binding.context_path, None);
        assert_eq!(active(&project).unwrap().unwrap().goal_id, binding.goal_id);
        let file = load(&project).unwrap();
        assert_eq!(file.schema_version, GOAL_BINDING_SCHEMA_VERSION);
        assert_eq!(
            file.active_goal_id.as_deref(),
            Some("goal.reach-destination")
        );
    }

    #[test]
    fn an_explicit_goal_beats_the_saved_default() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        bind(&project, &goal(root.path(), "goal.default"), None).unwrap();
        let explicit = goal(root.path(), "goal.explicit");
        let (resolved, source) = resolve(&project, Some(&explicit)).unwrap();
        assert_eq!(
            fs::canonicalize(&resolved).unwrap(),
            fs::canonicalize(&explicit).unwrap()
        );
        assert_eq!(source, GoalSource::Explicit);
        let (resolved, source) = resolve(&project, None).unwrap();
        assert_eq!(
            fs::canonicalize(&resolved).unwrap(),
            fs::canonicalize(root.path().join("goals/goal.default.json")).unwrap()
        );
        assert_eq!(source, GoalSource::Default);
    }

    #[test]
    fn a_missing_default_goal_reports_the_gap_instead_of_silently_failing() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        let error = resolve(&project, None).unwrap_err().to_string();
        assert!(error.contains("saved default goal"), "{error}");

        bind(&project, &goal(root.path(), "goal.default"), None).unwrap();
        fs::remove_file(root.path().join("goals/goal.default.json")).unwrap();
        let error = resolve(&project, None).unwrap_err().to_string();
        assert!(error.contains("no longer exists"), "{error}");
    }

    #[test]
    fn switching_between_saved_goals_moves_the_active_pointer() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        bind(&project, &goal(root.path(), "goal.first"), None).unwrap();
        bind(&project, &goal(root.path(), "goal.second"), None).unwrap();
        assert_eq!(active(&project).unwrap().unwrap().goal_id, "goal.second");
        select(&project, "goal.first").unwrap();
        assert_eq!(active(&project).unwrap().unwrap().goal_id, "goal.first");
        assert!(select(&project, "goal.missing").is_err());
    }

    #[test]
    fn a_goal_outside_the_project_root_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        let outside = tempfile::tempdir().unwrap();
        let goal = goal(outside.path(), "goal.elsewhere");
        let error = bind(&project, &goal, None).unwrap_err().to_string();
        assert!(error.contains("inside the project root"), "{error}");
    }

    #[test]
    fn doctor_metadata_reports_the_active_goal() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        assert_eq!(doctor_metadata(&project)["goal_count"], 0);
        bind(&project, &goal(root.path(), "goal.default"), None).unwrap();
        let metadata = doctor_metadata(&project);
        assert_eq!(metadata["active_goal_id"], "goal.default");
        assert_eq!(metadata["goal_count"], 1);
    }
}
