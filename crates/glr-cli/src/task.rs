use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::args::TaskCommand;
use crate::commands::emit;
use crate::error::{Error, Result};
use crate::process::executable_available;
use crate::project::Project;
use crate::project::ProjectCommand;

const TASK_FILE_NAME: &str = "glr.toml";
const TASK_SCHEMA_VERSION: &str = "glr.tasks.v1";
const TASK_RESULT_SCHEMA_VERSION: &str = "glr.task-result.v1";
const MAX_TASK_FILE_BYTES: u64 = 1024 * 1024;
const MAX_TASKS: usize = 256;
const MAX_DEPENDENCY_DEPTH: usize = 32;
const DEFAULT_TIMEOUT_SECONDS: u64 = 3600;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskFile {
    schema_version: String,
    tasks: BTreeMap<String, TaskDefinition>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TaskDefinition {
    description: String,
    #[serde(default)]
    runner: TaskRunner,
    argv: Vec<String>,
    #[serde(default = "default_cwd")]
    cwd: String,
    #[serde(default)]
    depends: Vec<String>,
    #[serde(default = "default_timeout_seconds")]
    timeout_seconds: u64,
    #[serde(default)]
    parameters: BTreeMap<String, TaskParameter>,
    #[serde(default)]
    result: Option<TaskResultContract>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum TaskRunner {
    #[default]
    Direct,
    Vx,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TaskParameter {
    #[serde(rename = "type")]
    kind: ParameterType,
    #[serde(default)]
    required: bool,
    #[serde(default)]
    default: Option<toml::Value>,
    #[serde(default)]
    minimum: Option<i64>,
    #[serde(default)]
    maximum: Option<i64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ParameterType {
    String,
    Integer,
    Boolean,
    Path,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TaskResultContract {
    schema: String,
    #[serde(default = "default_true")]
    required: bool,
}

#[derive(Debug, Serialize)]
struct TaskSummary<'a> {
    name: &'a str,
    description: &'a str,
    runner: TaskRunner,
    depends: &'a [String],
    parameters: Vec<&'a str>,
}

#[derive(Debug, Serialize)]
struct StepResult {
    task: String,
    status: &'static str,
    exit_code: Option<i32>,
    duration_ms: u128,
    log_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

fn default_cwd() -> String {
    ".".into()
}

fn default_timeout_seconds() -> u64 {
    DEFAULT_TIMEOUT_SECONDS
}

fn default_true() -> bool {
    true
}

pub fn execute(project: &Project, command: TaskCommand, as_json: bool) -> Result<i32> {
    let tasks = load(project)?;
    match command {
        TaskCommand::List => {
            let summaries = tasks
                .tasks
                .iter()
                .map(|(name, task)| TaskSummary {
                    name,
                    description: &task.description,
                    runner: task.runner,
                    depends: &task.depends,
                    parameters: task.parameters.keys().map(String::as_str).collect(),
                })
                .collect::<Vec<_>>();
            emit("task.list", &summaries, as_json)?;
            Ok(0)
        }
        TaskCommand::Show { name } => {
            let task = tasks
                .tasks
                .get(&name)
                .ok_or_else(|| Error::Invalid(format!("unknown task: {name}")))?;
            emit("task.show", &json!({"name": name, "task": task}), as_json)?;
            Ok(0)
        }
        TaskCommand::Run { name, parameters } => run(project, &tasks, &name, &parameters, as_json),
    }
}

pub fn doctor_report(project: &Project) -> Result<Option<serde_json::Value>> {
    let path = project.root.join(TASK_FILE_NAME);
    if !path.exists() {
        return Ok(None);
    }
    let tasks = load(project)?;
    let unavailable = tasks
        .tasks
        .iter()
        .filter_map(|(name, task)| {
            let program = match task.runner {
                TaskRunner::Vx => "vx",
                TaskRunner::Direct => task.argv.first()?.as_str(),
            };
            let command = ProjectCommand {
                argv: vec![program.into()],
            };
            (!executable_available(project, &command)).then(|| name.clone())
        })
        .collect::<Vec<_>>();
    Ok(Some(json!({
        "schema_version": TASK_SCHEMA_VERSION,
        "path": path,
        "task_count": tasks.tasks.len(),
        "ready": unavailable.is_empty(),
        "unavailable": unavailable,
    })))
}

fn load(project: &Project) -> Result<TaskFile> {
    let path = project.root.join(TASK_FILE_NAME);
    if path.is_symlink() || !path.is_file() {
        return Err(Error::Missing(path));
    }
    if fs::metadata(&path)?.len() > MAX_TASK_FILE_BYTES {
        return Err(Error::Invalid(format!(
            "{TASK_FILE_NAME} exceeds the 1 MiB limit"
        )));
    }
    let content = fs::read_to_string(path)?;
    let tasks: TaskFile = toml::from_str(&content)?;
    if tasks.schema_version != TASK_SCHEMA_VERSION {
        return Err(Error::Invalid(format!(
            "glr.toml schema_version must be {TASK_SCHEMA_VERSION:?}"
        )));
    }
    validate(project, &tasks)?;
    Ok(tasks)
}

fn validate(project: &Project, tasks: &TaskFile) -> Result<()> {
    if tasks.tasks.is_empty() || tasks.tasks.len() > MAX_TASKS {
        return Err(Error::Invalid(format!(
            "glr.toml must define between 1 and {MAX_TASKS} tasks"
        )));
    }
    for (name, task) in &tasks.tasks {
        validate_identifier(name, "task name")?;
        validate_text(&task.description, &format!("tasks.{name}.description"))?;
        if task.argv.is_empty() {
            return Err(Error::Invalid(format!("tasks.{name}.argv cannot be empty")));
        }
        if whole_placeholder(&task.argv[0]).is_some() {
            return Err(Error::Invalid(format!(
                "tasks.{name}.argv[0] must be selected by trusted configuration"
            )));
        }
        if task.timeout_seconds == 0 || task.timeout_seconds > 86_400 {
            return Err(Error::Invalid(format!(
                "tasks.{name}.timeout_seconds must be between 1 and 86400"
            )));
        }
        resolve_cwd(project, &task.cwd)?;
        let mut dependency_names = HashSet::new();
        for dependency in &task.depends {
            validate_identifier(dependency, "task dependency")?;
            if dependency == name || !dependency_names.insert(dependency) {
                return Err(Error::Invalid(format!(
                    "tasks.{name}.depends must contain unique other task names"
                )));
            }
            if !tasks.tasks.contains_key(dependency) {
                return Err(Error::Invalid(format!(
                    "tasks.{name} depends on unknown task {dependency:?}"
                )));
            }
        }
        for (parameter_name, parameter) in &task.parameters {
            validate_identifier(parameter_name, "task parameter")?;
            validate_parameter_definition(name, parameter_name, parameter)?;
        }
        if let Some(result) = &task.result {
            validate_identifier(&result.schema, "task result schema")?;
        }
        for argument in &task.argv {
            validate_text(argument, &format!("tasks.{name}.argv"))?;
            if argument.contains('{') || argument.contains('}') {
                let placeholder = whole_placeholder(argument).ok_or_else(|| {
                    Error::Invalid(format!(
                        "tasks.{name} placeholders must occupy a complete argv entry"
                    ))
                })?;
                if !task.parameters.contains_key(placeholder)
                    && !matches!(placeholder, "project_root" | "task_dir" | "task_result")
                {
                    return Err(Error::Invalid(format!(
                        "tasks.{name} uses undeclared placeholder {{{placeholder}}}"
                    )));
                }
            }
        }
    }
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    for name in tasks.tasks.keys() {
        visit(tasks, name, &mut visiting, &mut visited, 0, &mut Vec::new())?;
    }
    Ok(())
}

fn validate_parameter_definition(task: &str, name: &str, parameter: &TaskParameter) -> Result<()> {
    if parameter.required && parameter.default.is_some() {
        return Err(Error::Invalid(format!(
            "tasks.{task}.parameters.{name} cannot be required and have a default"
        )));
    }
    if !matches!(parameter.kind, ParameterType::Integer)
        && (parameter.minimum.is_some() || parameter.maximum.is_some())
    {
        return Err(Error::Invalid(format!(
            "tasks.{task}.parameters.{name} bounds require type integer"
        )));
    }
    if parameter
        .minimum
        .zip(parameter.maximum)
        .is_some_and(|(min, max)| min > max)
    {
        return Err(Error::Invalid(format!(
            "tasks.{task}.parameters.{name} minimum exceeds maximum"
        )));
    }
    if let Some(default) = &parameter.default {
        parse_parameter(&toml_value_string(default)?, parameter, None)?;
    }
    Ok(())
}

fn visit(
    tasks: &TaskFile,
    name: &str,
    visiting: &mut HashSet<String>,
    visited: &mut HashSet<String>,
    depth: usize,
    order: &mut Vec<String>,
) -> Result<()> {
    if visited.contains(name) {
        return Ok(());
    }
    if depth > MAX_DEPENDENCY_DEPTH || !visiting.insert(name.into()) {
        return Err(Error::Invalid(format!(
            "task dependency cycle includes {name:?}"
        )));
    }
    for dependency in &tasks.tasks[name].depends {
        visit(tasks, dependency, visiting, visited, depth + 1, order)?;
    }
    visiting.remove(name);
    visited.insert(name.into());
    order.push(name.into());
    Ok(())
}

fn run(
    project: &Project,
    tasks: &TaskFile,
    name: &str,
    assignments: &[String],
    as_json: bool,
) -> Result<i32> {
    if !tasks.tasks.contains_key(name) {
        return Err(Error::Invalid(format!("unknown task: {name}")));
    }
    let provided = parse_assignments(assignments)?;
    let mut order = Vec::new();
    visit(
        tasks,
        name,
        &mut HashSet::new(),
        &mut HashSet::new(),
        0,
        &mut order,
    )?;
    for key in provided.keys() {
        if !order
            .iter()
            .any(|task_name| tasks.tasks[task_name].parameters.contains_key(key))
        {
            return Err(Error::Invalid(format!(
                "parameter {key:?} is not declared by this task graph"
            )));
        }
    }
    let execution_id = format!("task-{}", Uuid::new_v4().simple());
    let execution_dir = project.data_dir.join("tasks").join(&execution_id);
    fs::create_dir_all(&execution_dir)?;
    let started_at_ns = now_ns();
    let mut results = Vec::new();
    let mut exit_code = 0;
    for task_name in order {
        let task = &tasks.tasks[&task_name];
        let values = resolve_parameters(project, &task_name, task, &provided)?;
        let result = run_step(project, &execution_dir, &task_name, task, &values)?;
        exit_code = result.exit_code.unwrap_or(1);
        let succeeded = result.status == "succeeded";
        results.push(result);
        if !succeeded {
            break;
        }
    }
    let status = if exit_code == 0 {
        "succeeded"
    } else {
        "failed"
    };
    let result = json!({
        "schema_version": TASK_RESULT_SCHEMA_VERSION,
        "execution_id": execution_id,
        "task": name,
        "status": status,
        "started_at_ns": started_at_ns,
        "finished_at_ns": now_ns(),
        "steps": results,
    });
    fs::write(
        execution_dir.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    emit("task.run", &result, as_json)?;
    Ok(exit_code)
}

fn run_step(
    project: &Project,
    execution_dir: &Path,
    name: &str,
    task: &TaskDefinition,
    values: &HashMap<String, String>,
) -> Result<StepResult> {
    let task_dir = execution_dir.join(name);
    fs::create_dir_all(&task_dir)?;
    let result_path = task_dir.join("result.json");
    let mut expanded = Vec::with_capacity(task.argv.len() + 1);
    if matches!(task.runner, TaskRunner::Vx) {
        expanded.push("vx".into());
    }
    for argument in &task.argv {
        let value = match whole_placeholder(argument) {
            Some("project_root") => project.root.to_string_lossy().into_owned(),
            Some("task_dir") => task_dir.to_string_lossy().into_owned(),
            Some("task_result") => result_path.to_string_lossy().into_owned(),
            Some(parameter) => values.get(parameter).cloned().ok_or_else(|| {
                Error::Invalid(format!(
                    "task {name:?} requires a value for parameter {parameter:?}"
                ))
            })?,
            None => argument.clone(),
        };
        expanded.push(value);
    }
    let program = expanded.remove(0);
    let cwd = resolve_cwd(project, &task.cwd)?;
    let log_path = task_dir.join("task.log");
    let stdout = File::create(&log_path)?;
    let stderr = stdout.try_clone()?;
    let started = Instant::now();
    let mut child = Command::new(&program)
        .args(&expanded)
        .current_dir(cwd)
        .env("GLR_PROJECT_ROOT", &project.root)
        .env("GLR_TASK_NAME", name)
        .env("GLR_TASK_DIR", &task_dir)
        .env("GLR_TASK_RESULT", &result_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|error| {
            Error::Contract(format!(
                "could not start task {name:?} with {program:?}: {error}"
            ))
        })?;
    let deadline = started + Duration::from_secs(task.timeout_seconds);
    loop {
        if let Some(status) = child.try_wait()? {
            let mut code = status.code();
            let mut task_status = if status.success() {
                "succeeded"
            } else {
                "failed"
            };
            let mut reason = None;
            if status.success()
                && let Some(contract) = &task.result
                && let Err(error) = verify_task_result(&result_path, contract)
            {
                task_status = "failed";
                code = Some(78);
                reason = Some(error.to_string());
            }
            return Ok(StepResult {
                task: name.into(),
                status: task_status,
                exit_code: code,
                duration_ms: started.elapsed().as_millis(),
                log_path: relative(project, &log_path)?,
                result_path: result_path
                    .is_file()
                    .then(|| relative(project, &result_path))
                    .transpose()?,
                reason,
            });
        }
        if Instant::now() >= deadline {
            child.kill()?;
            let _ = child.wait();
            return Ok(StepResult {
                task: name.into(),
                status: "timed-out",
                exit_code: Some(124),
                duration_ms: started.elapsed().as_millis(),
                log_path: relative(project, &log_path)?,
                result_path: None,
                reason: Some(format!(
                    "task exceeded its {} second timeout",
                    task.timeout_seconds
                )),
            });
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn verify_task_result(path: &Path, contract: &TaskResultContract) -> Result<()> {
    if !path.exists() && !contract.required {
        return Ok(());
    }
    if path.is_symlink() || !path.is_file() {
        return Err(Error::Contract(format!(
            "task result is required and must be a regular file: {}",
            path.display()
        )));
    }
    let bytes = fs::read(path)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(Error::Contract(
            "task result exceeds the 8 MiB limit".into(),
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    if value["schema_version"].as_str() != Some(contract.schema.as_str()) {
        return Err(Error::Contract(format!(
            "task result schema_version must be {:?}",
            contract.schema
        )));
    }
    Ok(())
}

fn parse_assignments(assignments: &[String]) -> Result<HashMap<String, String>> {
    let mut values = HashMap::new();
    for assignment in assignments {
        let (name, value) = assignment.split_once('=').ok_or_else(|| {
            Error::Invalid(format!(
                "task parameter must use NAME=VALUE: {assignment:?}"
            ))
        })?;
        validate_identifier(name, "task parameter")?;
        if values.insert(name.into(), value.into()).is_some() {
            return Err(Error::Invalid(format!("duplicate task parameter: {name}")));
        }
    }
    Ok(values)
}

fn resolve_parameters(
    project: &Project,
    task_name: &str,
    task: &TaskDefinition,
    provided: &HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    let mut values = HashMap::new();
    for (name, parameter) in &task.parameters {
        let raw = if let Some(value) = provided.get(name) {
            Some(value.clone())
        } else if let Some(value) = &parameter.default {
            Some(toml_value_string(value)?)
        } else {
            None
        };
        let Some(raw) = raw else {
            if parameter.required {
                return Err(Error::Invalid(format!(
                    "task {task_name:?} requires parameter {name:?}"
                )));
            }
            continue;
        };
        values.insert(
            name.clone(),
            parse_parameter(&raw, parameter, Some(project))?,
        );
    }
    Ok(values)
}

fn parse_parameter(
    raw: &str,
    parameter: &TaskParameter,
    project: Option<&Project>,
) -> Result<String> {
    if raw.is_empty() || raw.chars().any(char::is_control) {
        return Err(Error::Invalid(
            "task parameter values must be non-empty printable strings".into(),
        ));
    }
    match parameter.kind {
        ParameterType::String => Ok(raw.into()),
        ParameterType::Boolean => raw
            .parse::<bool>()
            .map(|value| value.to_string())
            .map_err(|_| Error::Invalid(format!("expected boolean task parameter, got {raw:?}"))),
        ParameterType::Integer => {
            let value = raw.parse::<i64>().map_err(|_| {
                Error::Invalid(format!("expected integer task parameter, got {raw:?}"))
            })?;
            if parameter.minimum.is_some_and(|minimum| value < minimum)
                || parameter.maximum.is_some_and(|maximum| value > maximum)
            {
                return Err(Error::Invalid(format!(
                    "task parameter {value} is outside its declared bounds"
                )));
            }
            Ok(value.to_string())
        }
        ParameterType::Path => {
            if Path::new(raw).is_absolute()
                || Path::new(raw).components().any(|part| {
                    matches!(
                        part,
                        Component::ParentDir | Component::Prefix(_) | Component::RootDir
                    )
                })
            {
                return Err(Error::Invalid(format!(
                    "task path parameter must stay project-relative: {raw:?}"
                )));
            }
            if let Some(project) = project {
                Ok(project.root.join(raw).to_string_lossy().into_owned())
            } else {
                Ok(raw.into())
            }
        }
    }
}

fn toml_value_string(value: &toml::Value) -> Result<String> {
    match value {
        toml::Value::String(value) => Ok(value.clone()),
        toml::Value::Integer(value) => Ok(value.to_string()),
        toml::Value::Boolean(value) => Ok(value.to_string()),
        _ => Err(Error::Invalid(
            "task parameter defaults must be strings, integers, or booleans".into(),
        )),
    }
}

fn resolve_cwd(project: &Project, value: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|part| {
            matches!(
                part,
                Component::ParentDir | Component::Prefix(_) | Component::RootDir
            )
        })
    {
        return Err(Error::Invalid(format!(
            "task cwd must stay project-relative: {value:?}"
        )));
    }
    let joined = project.root.join(path);
    if joined.is_symlink() || !joined.is_dir() {
        return Err(Error::Missing(joined));
    }
    let canonical = fs::canonicalize(joined)?;
    if !canonical.starts_with(&project.root) {
        return Err(Error::Invalid("task cwd escapes the project root".into()));
    }
    Ok(canonical)
}

fn whole_placeholder(value: &str) -> Option<&str> {
    value
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
}

fn validate_identifier(value: &str, label: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(Error::Invalid(format!(
            "{label} must contain only ASCII letters, digits, '.', '_' or '-'"
        )));
    }
    Ok(())
}

fn validate_text(value: &str, label: &str) -> Result<()> {
    if value.is_empty() || value.len() > 4096 || value.chars().any(char::is_control) {
        return Err(Error::Invalid(format!(
            "{label} must be a non-empty printable string"
        )));
    }
    Ok(())
}

fn relative(project: &Project, path: &Path) -> Result<String> {
    Ok(path
        .strip_prefix(&project.root)
        .map_err(|_| Error::Invalid("task artifact escaped the project root".into()))?
        .to_string_lossy()
        .replace('\\', "/"))
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.as_nanos().min(u64::MAX as u128) as u64
        })
}
