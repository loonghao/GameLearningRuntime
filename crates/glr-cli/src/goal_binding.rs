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

/// Where the run context of one invocation came from.
///
/// `source` alone cannot express this: an explicit `--goal` may still inherit
/// the context bound to the active default goal, so the receipt records the
/// context origin next to the goal origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSource {
    /// `--context` was passed on the command line.
    Explicit,
    /// No `--context` was passed, so the context bound to the active goal was used.
    Default,
    /// No run context applies to this invocation.
    None,
}

impl ContextSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Default => "default",
            Self::None => "none",
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
        stored_project_path("goal binding goal_path", &binding.goal_path)?;
        if let Some(context_path) = &binding.context_path {
            stored_project_path("goal binding context_path", context_path)?;
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
    let resolved = stored_goal_path(project, &binding.goal_path).map_err(|_| {
        Error::Invalid(format!(
            "default goal {:?} points at {:?}, which no longer exists; run `glr goal set --goal <path>` to rebind",
            binding.goal_id, binding.goal_path
        ))
    })?;
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

/// Reject a stored path that a hand-edited store turned into an escape.
///
/// `bind` only stores `strip_prefix` results, so a store written by the CLI
/// holds bare project-relative POSIX paths. A hand-edited `goal-binding.json`
/// can still name `..`, a root, or a Windows prefix, and the plain join used to
/// read stored paths pops parent directories. Checking the stored value when
/// the store is loaded keeps such an entry out of every command instead of
/// only out of `goal run`.
fn stored_project_path(field: &str, stored: &str) -> Result<()> {
    let path = Path::new(stored);
    let escapes = stored.is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::CurDir
                    | Component::ParentDir
                    | Component::Prefix(_)
                    | Component::RootDir
            )
        });
    if escapes {
        return Err(Error::Invalid(format!(
            "{field} must be a project-relative path, found {stored:?}; correct or remove {GOAL_BINDING_FILE_NAME}"
        )));
    }
    Ok(())
}

/// Resolve a stored goal path through the guard `bind` used when saving it.
///
/// Stored paths are re-checked with `regular_project_file` rather than joined,
/// so a goal that became a directory, a link, or an out-of-project file after
/// binding is refused instead of opened.
fn stored_goal_path(project: &Project, stored: &str) -> Result<PathBuf> {
    let (path, _) = regular_project_file(&project.root, Path::new(stored), "goal")?;
    Ok(path)
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

/// Compare the digest captured at bind time with the file the store names.
///
/// A stored path that no longer names a regular project file — deleted, or
/// replaced by a directory or a link after binding — is reported as `missing`.
fn source_status(project: &Project, binding: &GoalBinding) -> &'static str {
    match stored_goal_path(project, &binding.goal_path) {
        Ok(path) => match sha256_file(&path) {
            Ok(digest) if digest == binding.goal_sha256 => "unchanged",
            Ok(_) => "changed",
            Err(_) => "missing",
        },
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
///
/// `context_source` closes the gap an explicit `--goal` leaves open: such a run
/// can still inherit the context bound to the active default goal, which
/// `source` alone cannot express. `source_status` carries the `goal show` drift
/// check into the run receipt, so a run records whether its goal still matched
/// the digest captured when it was bound.
///
/// Both fields are additive: an older receipt keeps its `source`, `goal_id`,
/// `goal_path`, and `context_path` keys.
pub fn run_metadata(
    project: &Project,
    binding: Option<&GoalBinding>,
    source: GoalSource,
    context: Option<&RunContext>,
    context_source: ContextSource,
) -> Value {
    let context_path = context
        .map(RunContext::source_path)
        .map(str::to_owned)
        .or_else(|| binding.and_then(|binding| binding.context_path.clone()));
    json!({
        "source": source.as_str(),
        "goal_id": binding.map(|binding| binding.goal_id.clone()),
        "goal_path": binding.map(|binding| binding.goal_path.clone()),
        "context_path": context_path,
        "context_source": context_source.as_str(),
        "source_status": binding.map(|binding| source_status(project, binding)),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        ContextSource, GOAL_BINDING_SCHEMA_VERSION, GoalSource, active, bind, doctor_metadata,
        load, resolve, run_metadata, select,
    };
    use crate::project::Project;
    use crate::run_context::RunContext;
    use serde_json::{Value, json};
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

    /// A `glr.run-context.v1` file and the input it names, both inside the project.
    fn context(root: &Path, name: &str, context_id: &str) -> PathBuf {
        fs::create_dir_all(root.join("config/contexts")).unwrap();
        fs::write(
            root.join("config/training.json"),
            br#"{"schema_version":"glr.training.v1","algorithm":"ppo"}"#,
        )
        .unwrap();
        let path = root.join(format!("config/contexts/{name}.toml"));
        fs::write(
            &path,
            format!(
                r#"schema_version = "glr.run-context.v1"
context_id = "{context_id}"
environment_id = "example.context-v1"
protocol_version = "1.0"

[labels]
season = "{name}"

[[inputs]]
owner = "training"
path = "config/training.json"
schema_version = "glr.training.v1"
"#
            ),
        )
        .unwrap();
        path
    }

    /// Write `goal-binding.json` by hand, the way an editor or a restore would.
    fn write_store(root: &Path, goal_id: &str, goal_path: &str, context_path: Option<&str>) {
        fs::create_dir_all(root.join(".glr")).unwrap();
        fs::write(
            root.join(".glr/goal-binding.json"),
            serde_json::to_vec_pretty(&json!({
                "schema_version": GOAL_BINDING_SCHEMA_VERSION,
                "active_goal_id": goal_id,
                "goals": {
                    goal_id: {
                        "goal_id": goal_id,
                        "objective": "reach the destination",
                        "environment_family": "test",
                        "goal_path": goal_path,
                        "goal_sha256": "0".repeat(64),
                        "context_path": context_path,
                    }
                },
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn a_hand_edited_goal_path_outside_the_project_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        write_store(root.path(), "goal.escaped", "../outside.json", None);
        let error = load(&project).unwrap_err().to_string();
        assert!(error.contains("project-relative"), "{error}");
        assert!(error.contains("../outside.json"), "{error}");
        // Every command reads the store through `load`, so the entry cannot be
        // resolved either, and `doctor` reports the gap instead of failing.
        assert!(resolve(&project, None).is_err());
        assert!(active(&project).is_err());
        let metadata = doctor_metadata(&project);
        assert!(
            metadata["error"]
                .as_str()
                .unwrap()
                .contains("project-relative"),
            "{metadata}"
        );
    }

    #[test]
    fn a_hand_edited_context_path_outside_the_project_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        let goal = goal(root.path(), "goal.default");
        bind(&project, &goal, None).unwrap();
        write_store(
            root.path(),
            "goal.default",
            "goals/goal.default.json",
            Some("../../outside.toml"),
        );
        let error = load(&project).unwrap_err().to_string();
        assert!(error.contains("project-relative"), "{error}");
        assert!(error.contains("../../outside.toml"), "{error}");
        assert!(crate::goal_binding::active_context(&project).is_err());
    }

    #[test]
    fn a_store_edited_inside_the_project_still_resolves() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        goal(root.path(), "goal.default");
        write_store(root.path(), "goal.default", "goals/goal.default.json", None);
        let (resolved, source) = resolve(&project, None).unwrap();
        assert_eq!(
            fs::canonicalize(&resolved).unwrap(),
            fs::canonicalize(root.path().join("goals/goal.default.json")).unwrap()
        );
        assert_eq!(source, GoalSource::Default);
    }

    #[test]
    fn an_explicit_goal_receipt_records_the_inherited_context() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        let goal = goal(root.path(), "goal.default");
        context(root.path(), "native", "example.native-1");
        let binding = bind(
            &project,
            &goal,
            Some(Path::new("config/contexts/native.toml")),
        )
        .expect("bind the goal with its context");
        assert_eq!(
            binding.context_path.as_deref(),
            Some("config/contexts/native.toml")
        );
        let loaded =
            RunContext::load(&project, Path::new("config/contexts/native.toml")).expect("context");
        // An explicit `--goal` with no `--context` still inherits the bound
        // context, which the receipt must show.
        let receipt = run_metadata(
            &project,
            None,
            GoalSource::Explicit,
            Some(&loaded),
            ContextSource::Default,
        );
        assert_eq!(receipt["source"], "explicit");
        assert_eq!(receipt["goal_id"], Value::Null);
        assert_eq!(receipt["context_path"], "config/contexts/native.toml");
        assert_eq!(receipt["context_source"], "default");
        assert_eq!(receipt["source_status"], Value::Null);

        let receipt = run_metadata(
            &project,
            None,
            GoalSource::Explicit,
            Some(&loaded),
            ContextSource::Explicit,
        );
        assert_eq!(receipt["context_source"], "explicit");

        let receipt = run_metadata(
            &project,
            None,
            GoalSource::Explicit,
            None,
            ContextSource::None,
        );
        assert_eq!(receipt["context_path"], Value::Null);
        assert_eq!(receipt["context_source"], "none");
    }

    #[test]
    fn the_receipt_records_whether_the_bound_goal_drifted() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        let goal = goal(root.path(), "goal.default");
        let binding = bind(&project, &goal, None).unwrap();
        let receipt = run_metadata(
            &project,
            Some(&binding),
            GoalSource::Default,
            None,
            ContextSource::None,
        );
        assert_eq!(receipt["source"], "default");
        assert_eq!(receipt["source_status"], "unchanged");
        assert_eq!(receipt["context_source"], "none");

        // Same goal id, different content: the stored digest no longer matches.
        let path = root.path().join("goals/goal.default.json");
        let mut goal: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        goal["objective"] = json!("reach the other destination");
        fs::write(&path, serde_json::to_vec_pretty(&goal).unwrap()).unwrap();
        let receipt = run_metadata(
            &project,
            Some(&binding),
            GoalSource::Default,
            None,
            ContextSource::None,
        );
        assert_eq!(receipt["source_status"], "changed");

        fs::remove_file(&path).unwrap();
        let receipt = run_metadata(
            &project,
            Some(&binding),
            GoalSource::Default,
            None,
            ContextSource::None,
        );
        assert_eq!(receipt["source_status"], "missing");
    }

    #[test]
    fn a_default_goal_that_is_no_longer_a_regular_file_asks_for_a_rebind() {
        let root = tempfile::tempdir().unwrap();
        let project = project(root.path());
        let goal = goal(root.path(), "goal.default");
        bind(&project, &goal, None).unwrap();
        fs::remove_file(root.path().join("goals/goal.default.json")).unwrap();
        fs::create_dir_all(root.path().join("goals/goal.default.json")).unwrap();
        let error = resolve(&project, None).unwrap_err().to_string();
        assert!(error.contains("no longer exists"), "{error}");
    }
}
