use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::args::{
    Cli, Command as CliCommand, GoalCommand, KnowledgeCommand, QueryCommand, ReportCommand,
    RunsCommand, RuntimeCommand, UpdateArgs,
};
use crate::contracts::{
    AgentGoal, GoalEvaluation, GoalEvidenceBundle, ResearchBundle, SpatialKnowledgeBundle,
    TrialPlan, read_json, verify_model_bundle, write_json,
};
use crate::error::{Error, Result};
use crate::process::{
    CaptureSession, CommandInvocation, executable_available, finish_capture, relative_portable,
    run_command, start_capture,
};
use crate::project::{Project, ProjectCommand, find_project, load_project};
use crate::report;
use crate::store::{EntityQuery, RunRecord, Store};
use crate::update::Updater;

pub const CLI_OUTPUT_SCHEMA_VERSION: &str = "glr.cli-output.v1";
const TRAINER_NO_DATA_EXIT_CODE: i32 = 75;
const TRAINER_RESULT_SCHEMA_VERSION: &str = "glr.trainer-result.v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainerResult {
    schema_version: String,
    status: String,
    #[serde(default)]
    metrics: HashMap<String, f64>,
}

#[derive(Debug)]
struct TrainerOutcome {
    log: PathBuf,
    status: &'static str,
    metrics: HashMap<String, f64>,
}

pub fn execute(cli: Cli) -> Result<i32> {
    if let CliCommand::Update(arguments) = &cli.command {
        return run_update(&cli, arguments);
    }
    let project = load_project(&cli.project)?;
    let store = Store::open(project.data_dir.join("runs.sqlite3"))?;
    match cli.command {
        CliCommand::Doctor => doctor(&project, cli.json),
        CliCommand::Runtime {
            command: RuntimeCommand::Start,
        } => run_project_role(
            &project,
            &store,
            ProjectRoleInvocation {
                command: &project.runtime,
                kind: "runtime",
                output_command: "runtime.start",
                as_json: cli.json,
                bundle: None,
                metadata: json!({}),
            },
        ),
        CliCommand::Train { no_capture } => run_training(&project, &store, cli.json, !no_capture),
        CliCommand::Goal {
            command: GoalCommand::Run { goal, no_capture },
        } => run_goal(&project, &store, &absolute(&goal)?, cli.json, !no_capture),
        CliCommand::Play { bundle } => {
            let bundle = absolute(&bundle)?;
            let manifest = verify_model_bundle(&bundle)?;
            if manifest.environment_id != project.environment_id {
                return Err(Error::Contract(
                    "model bundle environment_id does not match the current GLR project".into(),
                ));
            }
            if manifest.protocol_version != project.protocol_version {
                return Err(Error::Contract(
                    "model bundle protocol_version does not match the current GLR project".into(),
                ));
            }
            run_project_role(
                &project,
                &store,
                ProjectRoleInvocation {
                    command: &project.player,
                    kind: "playback",
                    output_command: "play",
                    as_json: cli.json,
                    bundle: Some(&bundle),
                    metadata: json!({
                    "algorithm": manifest.algorithm,
                    "framework": manifest.framework,
                    "framework_version": manifest.framework_version,
                    }),
                },
            )
        }
        CliCommand::Runs { command } => match command {
            RunsCommand::List { status, limit } => {
                let runs = store.list_runs(
                    &project.environment_id,
                    status.map(|value| value.as_str()),
                    limit,
                )?;
                emit("runs.list", &runs, cli.json)?;
                Ok(0)
            }
            RunsCommand::Show { run_id } => {
                let run = store.get_run(&run_id)?;
                emit(
                    "runs.show",
                    &json!({
                        "run": run,
                        "events": store.list_events(&run_id)?,
                        "metrics": store.list_metrics(&run_id)?,
                        "artifacts": store.list_artifacts(&run_id)?,
                    }),
                    cli.json,
                )?;
                Ok(0)
            }
        },
        CliCommand::Report { command } => match command {
            ReportCommand::Build { run_id, output } => {
                report::build(&project, &store, &run_id, output.as_deref(), cli.json)
            }
        },
        CliCommand::Query { command } => match command {
            QueryCommand::Entities {
                world,
                kind,
                name,
                near,
                radius,
                limit,
            } => {
                let entities = store.query_entities(EntityQuery {
                    environment_id: &project.environment_id,
                    world_id: &world,
                    kind: kind.as_deref(),
                    name: name.as_deref(),
                    near: near.as_deref(),
                    radius,
                    limit,
                })?;
                emit("query.entities", &entities, cli.json)?;
                Ok(0)
            }
            QueryCommand::Routes {
                world,
                from_entity,
                to_entity,
                limit,
            } => {
                let routes = store.query_routes(
                    &project.environment_id,
                    &world,
                    from_entity.as_deref(),
                    to_entity.as_deref(),
                    limit,
                )?;
                let routes = routes
                    .into_iter()
                    .map(|route| {
                        let mut value = serde_json::to_value(route)?;
                        value
                            .as_object_mut()
                            .expect("SpatialRoute serializes as an object")
                            .insert("advisory".into(), Value::Bool(true));
                        Ok(value)
                    })
                    .collect::<Result<Vec<_>>>()?;
                emit("query.routes", &routes, cli.json)?;
                Ok(0)
            }
            QueryCommand::Research {
                tags,
                category,
                verified_only,
                limit,
            } => {
                let findings = store.query_research(
                    &project.environment_id,
                    &project.environment_family,
                    &tags,
                    category.map(|value| value.as_str()),
                    verified_only,
                    limit,
                )?;
                emit("query.research", &findings, cli.json)?;
                Ok(0)
            }
        },
        CliCommand::Knowledge { command } => match command {
            KnowledgeCommand::Export { output } => {
                export_knowledge(&project, &store, &absolute(&output)?, cli.json)
            }
            KnowledgeCommand::Import { source } => {
                import_knowledge(&project, &store, &absolute(&source)?, cli.json)
            }
        },
        CliCommand::Update(_) => unreachable!("update handled before project loading"),
    }
}

fn doctor(project: &Project, as_json: bool) -> Result<i32> {
    let roles = [
        ("runtime", Some(&project.runtime)),
        ("trainer", Some(&project.trainer)),
        ("player", Some(&project.player)),
        ("researcher", project.researcher.as_ref()),
        ("planner", project.planner.as_ref()),
        ("evaluator", project.evaluator.as_ref()),
    ];
    let mut reports = Vec::new();
    let mut ready = true;
    for (name, command) in roles {
        let configured = command.is_some();
        let available = command.is_none_or(|value| executable_available(project, value));
        if configured && !available {
            ready = false;
        }
        reports.push(
            json!({"role": name, "configured": configured, "executable_available": available}),
        );
    }
    if let Some(capture) = &project.capture {
        let available = executable_available(project, &capture.command());
        ready &= available;
        reports.push(
            json!({"role": "recorder", "configured": true, "executable_available": available}),
        );
    } else {
        reports
            .push(json!({"role": "recorder", "configured": false, "executable_available": true}));
    }
    emit(
        "doctor",
        &json!({
            "ready": ready,
            "version": env!("CARGO_PKG_VERSION"),
            "target": crate::update::BUILD_TARGET,
            "project_root": project.root,
            "environment_id": project.environment_id,
            "bridge_path": project.bridge_path,
            "bridge_exists": project.bridge_path.exists(),
            "store_path": project.data_dir.join("runs.sqlite3"),
            "roles": reports,
        }),
        as_json,
    )?;
    Ok(if ready { 0 } else { 4 })
}

fn run_update(cli: &Cli, arguments: &UpdateArgs) -> Result<i32> {
    let updater = Updater::github()?;
    let plan = updater.check()?;
    if !arguments.yes {
        emit(
            "update.check",
            &json!({
                "plan": plan,
                "applied": false,
                "confirmation_required": !arguments.check,
            }),
            cli.json,
        )?;
        return Ok(0);
    }
    let skills_dir = if arguments.no_skills {
        None
    } else if let Some(path) = &arguments.skills_dir {
        Some(absolute(path)?)
    } else {
        find_project(&cli.project)
            .ok()
            .and_then(|path| path.parent().map(|root| root.join(".agents/skills")))
    };
    let result = updater.apply(plan, skills_dir.as_deref())?;
    emit("update.apply", &result, cli.json)?;
    Ok(0)
}

fn run_training(project: &Project, store: &Store, as_json: bool, capture: bool) -> Result<i32> {
    let run = store.create_run(
        &project.environment_id,
        &project.protocol_version,
        "training",
        json!({"environment_family": project.environment_family}),
    )?;
    let run_dir = project.data_dir.join("runs").join(&run.run_id);
    fs::create_dir_all(&run_dir)?;
    let trainer_log = run_dir.join("trainer.log");
    let capture_session = if capture && project.capture.is_some() {
        Some(start_capture(project, &run.run_id, &run_dir)?)
    } else {
        None
    };
    let extra = HashMap::new();
    let result = run_command(CommandInvocation {
        command: &project.trainer,
        project,
        run_id: &run.run_id,
        run_dir: &run_dir,
        log_path: &trainer_log,
        bundle: None,
        extra: &extra,
        timeout: None,
    });
    let capture_complete = if let Some(session) = capture_session {
        finish_capture(project, store, &run.run_id, &run_dir, &run_dir, session)?
    } else {
        true
    };
    let trainer_exit = match result {
        Ok(code) => code,
        Err(error) => {
            let _ = store.finish_run(&run.run_id, "failed", Some(1));
            return Err(error);
        }
    };
    store.register_artifact(
        &run.run_id,
        "trainer.log",
        &trainer_log,
        "run-log",
        "text/plain",
    )?;
    let capture_required = project.capture.as_ref().is_some_and(|value| value.required);
    let succeeded = trainer_exit == 0 && (capture_complete || !capture_required);
    let exit_code = if succeeded {
        0
    } else if trainer_exit != 0 {
        trainer_exit
    } else {
        1
    };
    let finished = store.finish_run(
        &run.run_id,
        if succeeded { "succeeded" } else { "failed" },
        Some(exit_code),
    )?;
    emit("train", &finished, as_json)?;
    Ok(exit_code)
}

struct ProjectRoleInvocation<'a> {
    command: &'a ProjectCommand,
    kind: &'a str,
    output_command: &'a str,
    as_json: bool,
    bundle: Option<&'a Path>,
    metadata: Value,
}

fn run_project_role(
    project: &Project,
    store: &Store,
    invocation: ProjectRoleInvocation<'_>,
) -> Result<i32> {
    let mut combined_metadata = invocation.metadata.as_object().cloned().unwrap_or_default();
    combined_metadata.insert(
        "environment_family".into(),
        Value::String(project.environment_family.clone()),
    );
    let run = store.create_run(
        &project.environment_id,
        &project.protocol_version,
        invocation.kind,
        Value::Object(combined_metadata),
    )?;
    let run_dir = project.data_dir.join("runs").join(&run.run_id);
    fs::create_dir_all(&run_dir)?;
    let log = run_dir.join(format!("{}.log", invocation.kind));
    let extra = HashMap::new();
    let exit_code = match run_command(CommandInvocation {
        command: invocation.command,
        project,
        run_id: &run.run_id,
        run_dir: &run_dir,
        log_path: &log,
        bundle: invocation.bundle,
        extra: &extra,
        timeout: None,
    }) {
        Ok(value) => value,
        Err(error) => {
            let _ = store.finish_run(&run.run_id, "failed", Some(1));
            return Err(error);
        }
    };
    store.register_artifact(
        &run.run_id,
        log.file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| Error::Invalid("run log filename is not UTF-8".into()))?,
        &log,
        "run-log",
        "text/plain",
    )?;
    let finished = store.finish_run(
        &run.run_id,
        if exit_code == 0 {
            "succeeded"
        } else {
            "failed"
        },
        Some(exit_code),
    )?;
    emit(invocation.output_command, &finished, invocation.as_json)?;
    Ok(exit_code)
}

fn run_goal(
    project: &Project,
    store: &Store,
    goal_path: &Path,
    as_json: bool,
    capture_enabled: bool,
) -> Result<i32> {
    let (researcher, planner, evaluator) = match (
        project.researcher.as_ref(),
        project.planner.as_ref(),
        project.evaluator.as_ref(),
    ) {
        (Some(researcher), Some(planner), Some(evaluator)) => (researcher, planner, evaluator),
        _ => {
            return Err(Error::Contract(
                "goal run requires project researcher, planner, trainer, and evaluator commands"
                    .into(),
            ));
        }
    };
    let goal: AgentGoal = read_json(goal_path, "agent goal")?;
    goal.validate()?;
    if goal.environment_family != project.environment_family {
        return Err(Error::Contract(
            "goal environment_family does not match the current GLR project".into(),
        ));
    }
    let run = store.create_run(
        &project.environment_id,
        &project.protocol_version,
        "goal",
        json!({
            "environment_family": project.environment_family,
            "goal_id": goal.goal_id,
            "objective": goal.objective,
        }),
    )?;
    let run_dir = project.data_dir.join("runs").join(&run.run_id);
    fs::create_dir_all(&run_dir)?;
    let canonical_goal = run_dir.join("goal.json");
    let initial_research = run_dir.join("research.json");
    write_json(&canonical_goal, &goal)?;
    let deadline = Instant::now() + Duration::from_secs(goal.budget.max_wall_seconds);
    let result = run_goal_inner(GoalRunContext {
        project,
        store,
        run: &run,
        run_dir: &run_dir,
        canonical_goal: &canonical_goal,
        initial_research: &initial_research,
        goal: &goal,
        researcher,
        planner,
        evaluator,
        deadline,
        capture_enabled,
    });
    let GoalRunResult {
        satisfied,
        trials_completed,
        total_steps,
        evaluation,
        trainer_statuses,
        promotion,
    } = match result {
        Ok(value) => value,
        Err(error) => {
            let _ = store.finish_run(&run.run_id, "failed", Some(1));
            return Err(error);
        }
    };
    let exit_code = if satisfied { 0 } else { 3 };
    let finished = store.finish_run(
        &run.run_id,
        if satisfied { "succeeded" } else { "failed" },
        Some(exit_code),
    )?;
    emit(
        "goal.run",
        &json!({
            "run": finished,
            "goal_id": goal.goal_id,
            "satisfied": satisfied,
            "trials_completed": trials_completed,
            "training_steps_planned": total_steps,
            "evaluation": evaluation,
            "trainer_statuses": trainer_statuses,
            "promotion": promotion,
        }),
        as_json,
    )?;
    Ok(exit_code)
}

struct GoalRunContext<'a> {
    project: &'a Project,
    store: &'a Store,
    run: &'a RunRecord,
    run_dir: &'a Path,
    canonical_goal: &'a Path,
    initial_research: &'a Path,
    goal: &'a AgentGoal,
    researcher: &'a ProjectCommand,
    planner: &'a ProjectCommand,
    evaluator: &'a ProjectCommand,
    deadline: Instant,
    capture_enabled: bool,
}

struct GoalRunResult {
    satisfied: bool,
    trials_completed: u32,
    total_steps: u64,
    evaluation: Option<GoalEvaluation>,
    trainer_statuses: Vec<String>,
    promotion: Option<Value>,
}

fn run_goal_inner(context: GoalRunContext<'_>) -> Result<GoalRunResult> {
    let GoalRunContext {
        project,
        store,
        run,
        run_dir,
        canonical_goal,
        initial_research,
        goal,
        researcher,
        planner,
        evaluator,
        deadline,
        capture_enabled,
    } = context;
    let research_log = run_goal_role(
        project,
        researcher,
        "researcher",
        &run.run_id,
        run_dir,
        HashMap::from([
            ("goal_path".into(), canonical_goal.to_path_buf()),
            ("research_path".into(), initial_research.to_path_buf()),
        ]),
        deadline,
    )?;
    let mut research: ResearchBundle = read_json(initial_research, "research bundle")?;
    research.validate()?;
    let mut seen_sources = HashSet::new();
    validate_goal_research(goal, &research, &mut seen_sources)?;
    store.upsert_research_bundle(&research)?;
    let mut known_findings = research
        .findings
        .iter()
        .map(|finding| finding.finding_id.clone())
        .collect::<HashSet<_>>();
    store.append_event(
        &run.run_id,
        "research.completed",
        json!({"sources": research.sources.len(), "findings": research.findings.len()}),
    )?;
    for (path, role, media_type) in [
        (canonical_goal, "goal", "application/json"),
        (initial_research, "research", "application/json"),
        (&research_log, "run-log", "text/plain"),
    ] {
        register_goal_artifact(store, &run.run_id, run_dir, path, role, media_type)?;
    }

    let mut active_research = initial_research.to_path_buf();
    let mut previous_evaluation: Option<PathBuf> = None;
    let mut total_steps = 0_u64;
    let mut trials_completed = 0_u32;
    let mut last_evaluation = None;
    let mut trainer_statuses = Vec::new();
    let mut last_promotion = None;
    for trial_number in 1..=goal.budget.max_trials {
        remaining(deadline)?;
        let trial_id = format!("trial-{trial_number}");
        let trial_dir = run_dir.join("trials").join(&trial_id);
        fs::create_dir_all(&trial_dir)?;
        let trial_path = trial_dir.join("plan.json");
        let evaluation_path = trial_dir.join("evaluation.json");
        if trial_number > 1 {
            let refreshed = trial_dir.join("research.json");
            let mut context = HashMap::from([
                ("goal_path".into(), canonical_goal.to_path_buf()),
                ("research_path".into(), refreshed.clone()),
                ("previous_research_path".into(), active_research.clone()),
            ]);
            if let Some(previous) = &previous_evaluation {
                context.insert("previous_evaluation_path".into(), previous.clone());
            }
            let log = run_goal_role(
                project,
                researcher,
                "researcher",
                &run.run_id,
                &trial_dir,
                context,
                deadline,
            )?;
            research = read_json(&refreshed, "research bundle")?;
            research.validate()?;
            validate_goal_research(goal, &research, &mut seen_sources)?;
            store.upsert_research_bundle(&research)?;
            known_findings.extend(
                research
                    .findings
                    .iter()
                    .map(|finding| finding.finding_id.clone()),
            );
            active_research = refreshed.clone();
            store.append_event(
                &run.run_id,
                "research.refreshed",
                json!({"trial_id": trial_id, "sources_seen": seen_sources.len(), "findings": research.findings.len()}),
            )?;
            for (path, role, media_type) in [
                (&refreshed, "research", "application/json"),
                (&log, "run-log", "text/plain"),
            ] {
                register_goal_artifact(store, &run.run_id, run_dir, path, role, media_type)?;
            }
        }
        let mut context = HashMap::from([
            ("goal_path".into(), canonical_goal.to_path_buf()),
            ("research_path".into(), active_research.clone()),
            ("trial_path".into(), trial_path.clone()),
            ("evaluation_path".into(), evaluation_path.clone()),
            ("trial_id".into(), PathBuf::from(&trial_id)),
            (
                "trainer_result_path".into(),
                trial_dir.join("trainer.result.json"),
            ),
        ]);
        if goal.promotion.is_some() {
            let checkpoint_dir = project.data_dir.join("checkpoints");
            context.insert(
                "checkpoint_path".into(),
                checkpoint_dir.join(format!("{}.checkpoint", goal.goal_id)),
            );
            context.insert(
                "candidate_checkpoint_path".into(),
                trial_dir.join("checkpoint.candidate"),
            );
            context.insert(
                "promotion_path".into(),
                checkpoint_dir.join(format!("{}.best.json", goal.goal_id)),
            );
        }
        if let Some(previous) = &previous_evaluation {
            context.insert("previous_evaluation_path".into(), previous.clone());
        }
        let planner_log = run_goal_role(
            project,
            planner,
            "planner",
            &run.run_id,
            &trial_dir,
            context.clone(),
            deadline,
        )?;
        let trial: TrialPlan = read_json(&trial_path, "trial plan")?;
        trial.validate()?;
        if trial.goal_id != goal.goal_id || trial.trial_id != trial_id {
            return Err(Error::Contract(
                "trial plan goal_id or trial_id does not match control state".into(),
            ));
        }
        if total_steps + trial.max_steps > goal.budget.max_training_steps {
            return Err(Error::Contract(
                "trial plan exceeds the remaining training-step budget".into(),
            ));
        }
        for finding_id in trial
            .reward_terms
            .iter()
            .flat_map(|term| term.source_finding_ids.iter())
        {
            if !known_findings.contains(finding_id) {
                return Err(Error::Contract(format!(
                    "trial reward terms reference unknown research finding: {finding_id}"
                )));
            }
        }
        total_steps += trial.max_steps;
        store.append_event(
            &run.run_id,
            "trial.planned",
            json!({
                "trial_id": trial_id,
                "max_steps": trial.max_steps,
                "reward_terms": trial.reward_terms.iter().map(|term| &term.name).collect::<Vec<_>>(),
            }),
        )?;
        let capture: Option<CaptureSession> = if capture_enabled && project.capture.is_some() {
            Some(start_capture(project, &run.run_id, &trial_dir)?)
        } else {
            None
        };
        let metric_floor = store.latest_metric_id(&run.run_id)?;
        let trainer_result = run_trainer_role(
            project,
            &project.trainer,
            &run.run_id,
            &trial_dir,
            context.clone(),
            deadline,
        );
        let capture_complete = if let Some(session) = capture {
            finish_capture(project, store, &run.run_id, &trial_dir, run_dir, session)?
        } else {
            true
        };
        let trainer_outcome = trainer_result?;
        let trainer_log = trainer_outcome.log.clone();
        for (name, value) in &trainer_outcome.metrics {
            store.append_metric(
                &run.run_id,
                name,
                *value,
                None,
                json!({"source": "trainer", "authority": "authoritative"}),
            )?;
        }
        trainer_statuses.push(trainer_outcome.status.to_owned());
        store.append_event(
            &run.run_id,
            "trial.trainer",
            json!({"trial_id": trial_id, "status": trainer_outcome.status}),
        )?;
        let promotion = if let Some(config) = &goal.promotion {
            let candidate = trial_dir.join("checkpoint.candidate");
            let live = project
                .data_dir
                .join("checkpoints")
                .join(format!("{}.checkpoint", goal.goal_id));
            let metric = store
                .latest_metric_value(&run.run_id, &config.metric, metric_floor)?
                .ok_or_else(|| {
                    Error::Contract(format!(
                        "checkpoint promotion metric {:?} was not persisted for {trial_id}",
                        config.metric
                    ))
                })?;
            let (promoted, record) = store.promote_checkpoint(
                &goal.goal_id,
                &config.metric,
                config.mode,
                metric,
                &run.run_id,
                &trial_id,
                &candidate,
                &live,
            )?;
            let output = json!({
                "promoted": promoted,
                "metric": metric,
                "best_metric": record.best_metric,
                "checkpoint_sha256": record.checkpoint_sha256,
                "checkpoint_path": record.checkpoint_path,
            });
            store.append_event(&run.run_id, "checkpoint.promotion", output.clone())?;
            Some(output)
        } else {
            None
        };
        if !capture_complete
            && project
                .capture
                .as_ref()
                .is_some_and(|capture| capture.required)
        {
            return Err(Error::Contract(format!(
                "required capture failed for {trial_id}"
            )));
        }
        let evaluator_log = run_goal_role(
            project,
            evaluator,
            "evaluator",
            &run.run_id,
            &trial_dir,
            context,
            deadline,
        )?;
        let evidence: GoalEvidenceBundle = read_json(&evaluation_path, "goal evidence")?;
        evidence.validate()?;
        if evidence.goal_id != goal.goal_id || evidence.trial_id != trial_id {
            return Err(Error::Contract(
                "goal evidence goal_id or trial_id does not match control state".into(),
            ));
        }
        for item in &evidence.evidence {
            if item.run_id != run.run_id
                || !store.has_metric_evidence(&run.run_id, metric_floor, item)?
            {
                return Err(Error::Contract(format!(
                    "goal evidence {:?} is not backed by persisted runtime metrics",
                    item.metric
                )));
            }
        }
        let evaluation = goal.evaluate(&evidence.evidence)?;
        last_promotion = promotion.clone();
        trials_completed = trial_number;
        previous_evaluation = Some(evaluation_path.clone());
        store.append_event(
            &run.run_id,
            "trial.evaluated",
            json!({"trial_id": trial_id, "satisfied": evaluation.satisfied, "criteria": evaluation.criteria, "promotion": promotion}),
        )?;
        for (path, role, media_type) in [
            (&trial_path, "trial-plan", "application/json"),
            (&evaluation_path, "goal-evidence", "application/json"),
            (&planner_log, "run-log", "text/plain"),
            (&trainer_log, "run-log", "text/plain"),
            (&evaluator_log, "run-log", "text/plain"),
        ] {
            register_goal_artifact(store, &run.run_id, run_dir, path, role, media_type)?;
        }
        let trainer_result_path = trial_dir.join("trainer.result.json");
        if trainer_result_path.is_file() {
            register_goal_artifact(
                store,
                &run.run_id,
                run_dir,
                &trainer_result_path,
                "trainer-result",
                "application/json",
            )?;
        }
        if goal.promotion.is_some() {
            let candidate = trial_dir.join("checkpoint.candidate");
            if candidate.is_file() {
                register_goal_artifact(
                    store,
                    &run.run_id,
                    run_dir,
                    &candidate,
                    "checkpoint-candidate",
                    "application/octet-stream",
                )?;
            }
        }
        let satisfied = evaluation.satisfied;
        last_evaluation = Some(evaluation);
        if satisfied {
            break;
        }
    }
    Ok(GoalRunResult {
        satisfied: last_evaluation
            .as_ref()
            .is_some_and(|evaluation| evaluation.satisfied),
        trials_completed,
        total_steps,
        evaluation: last_evaluation,
        trainer_statuses,
        promotion: last_promotion,
    })
}

fn run_goal_role(
    project: &Project,
    command: &ProjectCommand,
    role: &str,
    run_id: &str,
    role_dir: &Path,
    context: HashMap<String, PathBuf>,
    deadline: Instant,
) -> Result<PathBuf> {
    let log = role_dir.join(format!("{role}.log"));
    let exit_code = run_command(CommandInvocation {
        command,
        project,
        run_id,
        run_dir: role_dir,
        log_path: &log,
        bundle: None,
        extra: &context,
        timeout: Some(remaining(deadline)?),
    })?;
    if exit_code != 0 {
        return Err(Error::Contract(format!(
            "goal {role} command failed with exit code {exit_code}"
        )));
    }
    Ok(log)
}

fn run_trainer_role(
    project: &Project,
    command: &ProjectCommand,
    run_id: &str,
    role_dir: &Path,
    context: HashMap<String, PathBuf>,
    deadline: Instant,
) -> Result<TrainerOutcome> {
    let log = role_dir.join("trainer.log");
    let exit_code = run_command(CommandInvocation {
        command,
        project,
        run_id,
        run_dir: role_dir,
        log_path: &log,
        bundle: None,
        extra: &context,
        timeout: Some(remaining(deadline)?),
    })?;
    let result_path = role_dir.join("trainer.result.json");
    let (parsed_status, metrics) = if result_path.is_file() {
        let result: TrainerResult = read_json(&result_path, "trainer result")?;
        if result.schema_version != TRAINER_RESULT_SCHEMA_VERSION {
            return Err(Error::Contract(
                "trainer result schema_version is unsupported".into(),
            ));
        }
        if result.metrics.values().any(|value| !value.is_finite()) {
            return Err(Error::Contract(
                "trainer result metrics must be finite".into(),
            ));
        }
        (Some(result.status), result.metrics)
    } else {
        (None, HashMap::new())
    };
    let status = match (exit_code, parsed_status.as_deref()) {
        (0, None) | (0, Some("completed")) => "completed",
        (_, Some("no_data")) | (TRAINER_NO_DATA_EXIT_CODE, None) => "no_data",
        (0, Some(_)) => {
            return Err(Error::Contract(
                "trainer result status must be completed or no_data".into(),
            ));
        }
        (_, _) => {
            return Err(Error::Contract(format!(
                "goal trainer command failed with exit code {exit_code}"
            )));
        }
    };
    Ok(TrainerOutcome {
        log,
        status,
        metrics,
    })
}

fn validate_goal_research(
    goal: &AgentGoal,
    research: &ResearchBundle,
    seen_sources: &mut HashSet<String>,
) -> Result<()> {
    seen_sources.extend(
        research
            .sources
            .iter()
            .map(|source| source.source_id.clone()),
    );
    if seen_sources.len() > goal.budget.max_research_sources {
        return Err(Error::Contract(
            "research cycles exceed goal max_research_sources".into(),
        ));
    }
    for source in &research.sources {
        if !goal.allowed_research_media.contains(&source.media_type) {
            return Err(Error::Contract(
                "research bundle uses a disallowed media type".into(),
            ));
        }
    }
    Ok(())
}

fn register_goal_artifact(
    store: &Store,
    run_id: &str,
    run_dir: &Path,
    path: &Path,
    role: &str,
    media_type: &str,
) -> Result<()> {
    store.register_artifact(
        run_id,
        &relative_portable(run_dir, path)?,
        path,
        role,
        media_type,
    )?;
    Ok(())
}

fn export_knowledge(project: &Project, store: &Store, output: &Path, as_json: bool) -> Result<i32> {
    if output.exists() || output.is_symlink() {
        return Err(Error::Invalid(format!(
            "knowledge export output already exists: {}",
            output.display()
        )));
    }
    let bundle = store.spatial_bundle(&project.environment_id, &project.protocol_version)?;
    write_json(output, &bundle)?;
    emit(
        "knowledge.export",
        &json!({
            "path": output,
            "environment_id": bundle.environment_id,
            "entities": bundle.entities.len(),
            "routes": bundle.routes.len(),
        }),
        as_json,
    )?;
    Ok(0)
}

fn import_knowledge(project: &Project, store: &Store, source: &Path, as_json: bool) -> Result<i32> {
    let bundle: SpatialKnowledgeBundle = read_json(source, "spatial knowledge")?;
    bundle.validate()?;
    if bundle.environment_id != project.environment_id {
        return Err(Error::Contract(
            "spatial knowledge environment_id does not match".into(),
        ));
    }
    if bundle.protocol_version != project.protocol_version {
        return Err(Error::Contract(
            "spatial knowledge protocol_version does not match".into(),
        ));
    }
    let run = store.create_run(
        &project.environment_id,
        &project.protocol_version,
        "knowledge-import",
        json!({"source": source.file_name().and_then(|value| value.to_str())}),
    )?;
    let run_dir = project.data_dir.join("runs").join(&run.run_id);
    fs::create_dir_all(&run_dir)?;
    let imported = run_dir.join("spatial-knowledge.json");
    write_json(&imported, &bundle)?;
    let (entities, routes) = match store.import_spatial(&bundle, &run.run_id) {
        Ok(value) => value,
        Err(error) => {
            let _ = store.finish_run(&run.run_id, "failed", Some(1));
            return Err(error);
        }
    };
    store.append_event(
        &run.run_id,
        "knowledge.imported",
        json!({"entities": entities, "routes": routes, "advisory": true}),
    )?;
    store.register_artifact(
        &run.run_id,
        "spatial-knowledge.json",
        &imported,
        "spatial-knowledge",
        "application/json",
    )?;
    let finished = store.finish_run(&run.run_id, "succeeded", Some(0))?;
    emit(
        "knowledge.import",
        &json!({
            "run": finished,
            "entities": entities,
            "routes": routes,
            "authority": "advisory",
        }),
        as_json,
    )?;
    Ok(0)
}

fn remaining(deadline: Instant) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|value| !value.is_zero())
        .ok_or_else(|| Error::Contract("goal wall-clock budget was exhausted".into()))
}

fn emit<T: Serialize>(command: &str, data: &T, compact: bool) -> Result<()> {
    let envelope = json!({
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "command": command,
        "data": data,
    });
    if compact {
        println!("{}", serde_json::to_string(&envelope)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&envelope)?);
    }
    Ok(())
}

fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}
