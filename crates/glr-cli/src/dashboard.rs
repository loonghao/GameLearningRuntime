//! Human and agent controls share validated CLI arguments and durable receipts.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use clap::{CommandFactory, Parser};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::args::{Cli, Command as CliCommand, DashboardCommand, GoalCommand};
use crate::error::{Error, Result};
use crate::observation::safe_child;
use crate::project::{Project, validate_identifier};
use crate::store::Store;

const SCHEMA: &str = "glr.dashboard.v1";
const PRESET_SCHEMA: &str = "glr.training-preset.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    schema_version: String,
    id: String,
    title: String,
    description: String,
    argv: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JobRequest {
    #[serde(default)]
    preset: Option<String>,
    #[serde(default)]
    argv: Option<Vec<String>>,
    /// A stable request ID makes double clicks and network retries idempotent.
    request_id: String,
}

#[derive(Clone)]
pub struct Dashboard {
    root: PathBuf,
    data_dir: PathBuf,
    environment_id: String,
    telemetry: Option<(String, String)>,
}

impl Dashboard {
    pub fn new(project: &Project) -> Result<Self> {
        Store::open(project.data_dir.join("runs.sqlite3"))?;
        let dashboard = Self {
            root: project.root.clone(),
            data_dir: project.data_dir.clone(),
            environment_id: project.environment_id.clone(),
            telemetry: None,
        };
        dashboard.connect()?.execute_batch("CREATE TABLE IF NOT EXISTS dashboard_presets (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS dashboard_jobs (id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, created_at_ms INTEGER NOT NULL, payload TEXT NOT NULL);")?;
        Ok(dashboard)
    }

    pub fn with_telemetry(mut self, url: String, token: String) -> Self {
        self.telemetry = Some((url, token));
        self
    }

    pub fn telemetry_authorized(&self, authorization: Option<&str>) -> bool {
        self.telemetry.as_ref().is_some_and(|(_, token)| {
            authorization.is_some_and(|value| value == format!("Bearer {token}"))
        })
    }

    fn connect(&self) -> Result<Connection> {
        let db = Connection::open(safe_child(&self.data_dir, Path::new("runs.sqlite3"))?)?;
        db.busy_timeout(Duration::from_secs(2))?;
        Ok(db)
    }

    pub fn route(&self, method: &str, url: &str, body: &[u8]) -> Result<Value> {
        let url = reqwest::Url::parse(&format!("http://localhost{url}"))
            .map_err(|_| Error::Invalid("invalid control URL".into()))?;
        let data = match (method, url.path()) {
            ("GET", "/api/v1/control/catalog") => catalog(),
            ("GET", "/api/v1/control/presets") => serde_json::to_value(self.presets()?)?,
            ("POST", "/api/v1/control/presets") => {
                self.save_preset(serde_json::from_slice(body)?)?
            }
            ("GET", "/api/v1/control/jobs") => {
                let before = url
                    .query_pairs()
                    .find(|(key, _)| key == "before")
                    .map(|(_, v)| v.into_owned());
                self.jobs(before.as_deref())?
            }
            ("POST", "/api/v1/control/jobs") => self.launch(serde_json::from_slice(body)?, true)?,
            ("GET", "/api/v1/control/job-log") => {
                let query: std::collections::HashMap<_, _> = url.query_pairs().collect();
                self.job_log(
                    query.get("id").map(|s| s.as_ref()).unwrap_or(""),
                    query.get("stream").map(|s| s.as_ref()).unwrap_or("stderr"),
                )?
            }
            _ => return Err(Error::Invalid("unknown dashboard endpoint".into())),
        };
        Ok(json!({"schema_version":SCHEMA,"data":data}))
    }

    fn presets(&self) -> Result<Vec<Preset>> {
        let mut presets = vec![Preset {
            schema_version: PRESET_SCHEMA.into(),
            id: "train.default".into(),
            title: "Default training".into(),
            description: "Run the configured learner with the project's capture policy.".into(),
            argv: vec!["train".into()],
        }];
        let db = self.connect()?;
        let mut query =
            db.prepare("SELECT payload FROM dashboard_presets ORDER BY id LIMIT 100")?;
        for payload in query.query_map([], |r| r.get::<_, String>(0))? {
            let preset: Preset = serde_json::from_str(&payload?)?;
            self.validate_preset(&preset)?;
            presets.push(preset);
        }
        Ok(presets)
    }

    fn validate_preset(&self, preset: &Preset) -> Result<()> {
        validate_identifier(&preset.id, "preset id")?;
        if preset.schema_version != PRESET_SCHEMA
            || preset.id == "train.default"
            || preset.title.trim().is_empty()
            || preset.title.len() > 120
            || preset.description.len() > 2000
        {
            return Err(Error::Invalid(
                "invalid preset schema, reserved id, or text limits".into(),
            ));
        }
        let cli = self.parse(&preset.argv)?;
        if !matches!(
            cli.command,
            CliCommand::Train { .. }
                | CliCommand::Goal { .. }
                | CliCommand::Task {
                    command: crate::args::TaskCommand::Run { .. }
                }
        ) {
            return Err(Error::Invalid(
                "training presets require train, goal run, or task run".into(),
            ));
        }
        Ok(())
    }

    fn save_preset(&self, preset: Preset) -> Result<Value> {
        self.validate_preset(&preset)?;
        let db = self.connect()?;
        let count: i64 = db.query_row(
            "SELECT COUNT(*) FROM dashboard_presets WHERE id != ?",
            [&preset.id],
            |r| r.get(0),
        )?;
        if count >= 100 {
            return Err(Error::Invalid("at most 100 saved presets".into()));
        }
        db.execute("INSERT INTO dashboard_presets(id,payload) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",params![preset.id,serde_json::to_string(&preset)?])?;
        Ok(serde_json::to_value(preset)?)
    }

    fn parse(&self, argv: &[String]) -> Result<Cli> {
        if argv.is_empty()
            || argv.len() > 128
            || argv.iter().any(|a| {
                a.len() > 4096
                    || a.contains('\0')
                    || a == "--project"
                    || a.starts_with("--project=")
            })
        {
            return Err(Error::Invalid(
                "invalid dashboard argv or project override".into(),
            ));
        }
        let mut command = vec![
            "glr".into(),
            "--project".into(),
            self.root.to_string_lossy().into_owned(),
            "--json".into(),
        ];
        command.extend_from_slice(argv);
        let cli = Cli::try_parse_from(command).map_err(|e| Error::Invalid(e.to_string()))?;
        if matches!(
            cli.command,
            CliCommand::Observe { .. } | CliCommand::Dashboard { .. }
        ) {
            return Err(Error::Invalid(
                "dashboard jobs cannot recursively start servers or jobs".into(),
            ));
        }
        Ok(cli)
    }

    fn launch(&self, request: JobRequest, background: bool) -> Result<Value> {
        validate_identifier(&request.request_id, "request_id")?;
        if request.request_id.len() > 100 || request.preset.is_some() == request.argv.is_some() {
            return Err(Error::Invalid(
                "provide exactly one preset or argv and a bounded request_id".into(),
            ));
        }
        let mut argv = match request.argv {
            Some(argv) => argv,
            None => {
                self.presets()?
                    .into_iter()
                    .find(|p| Some(&p.id) == request.preset.as_ref())
                    .ok_or_else(|| Error::Invalid("unknown training preset".into()))?
                    .argv
            }
        };
        let cli = self.parse(&argv)?;
        let normalized = serde_json::to_value(&argv)?;
        let db = self.connect()?;
        let existing = db.query_row(
            "SELECT payload FROM dashboard_jobs WHERE request_id = ?",
            [&request.request_id],
            |r| r.get::<_, String>(0),
        );
        if let Ok(payload) = existing {
            let job: Value = serde_json::from_str(&payload)?;
            if job["requested_argv"] != normalized {
                return Err(Error::Invalid(
                    "request_id was already used with different arguments".into(),
                ));
            }
            return Ok(job);
        }
        let jobs_dir = safe_child(&self.data_dir, Path::new("dashboard/jobs"))?;
        fs::create_dir_all(&jobs_dir)?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(safe_child(&self.data_dir, Path::new("dashboard/job.lock"))?)?;
        lock.try_lock().map_err(|_| {
            Error::Invalid(
                "another dashboard job is running; inspect its receipt before retrying".into(),
            )
        })?;
        if matches!(
            cli.command,
            CliCommand::Train { .. }
                | CliCommand::Goal { .. }
                | CliCommand::Play { .. }
                | CliCommand::Runtime { .. }
                | CliCommand::Task {
                    command: crate::args::TaskCommand::Run { .. }
                }
        ) {
            let active = Store::read_only(self.data_dir.join("runs.sqlite3"))?.list_runs(
                &self.environment_id,
                Some("running"),
                1,
            )?;
            if !active.is_empty() {
                return Err(Error::Invalid("a project run is still marked running; reconcile it before starting another execution".into()));
            }
        }
        if matches!(
            cli.command,
            CliCommand::Train {
                no_observe: false,
                ..
            } | CliCommand::Goal {
                command: GoalCommand::Run {
                    no_observe: false,
                    ..
                }
            }
        ) {
            argv.push("--no-observe".into());
        }
        let id = format!("job-{}", uuid::Uuid::new_v4().simple());
        let job_dir = jobs_dir.join(&id);
        fs::create_dir(&job_dir)?;
        let timestamp = now_ms();
        let mut job = json!({"id":id,"request_id":request.request_id,"created_at_ms":timestamp,"status":"starting","requested_argv":normalized,"argv":argv,"preset":request.preset,"exit_code":null});
        db.execute(
            "INSERT INTO dashboard_jobs(id,request_id,created_at_ms,payload) VALUES (?,?,?,?)",
            params![
                id,
                request.request_id,
                timestamp,
                serde_json::to_string(&job)?
            ],
        )?;
        let mut process = Command::new(std::env::current_exe()?);
        if let Some((url, token)) = &self.telemetry {
            process
                .env("GLR_TELEMETRY_URL", url)
                .env("GLR_TELEMETRY_TOKEN", token);
        } else {
            process
                .env_remove("GLR_TELEMETRY_URL")
                .env_remove("GLR_TELEMETRY_TOKEN");
        }
        let spawn = process
            .arg("--project")
            .arg(&self.root)
            .arg("--json")
            .args(&argv)
            .current_dir(&self.root)
            .env("GLR_NO_UPDATE_CHECK", "1")
            .env("GLR_DASHBOARD_JOB_ID", &id)
            .stdin(Stdio::null())
            .stdout(File::create(job_dir.join("stdout.log"))?)
            .stderr(File::create(job_dir.join("stderr.log"))?)
            .spawn();
        let mut child = match spawn {
            Ok(child) => child,
            Err(error) => {
                job["status"] = "failed".into();
                job["error"] = error.to_string().into();
                self.update_job(&id, &job)?;
                return Err(error.into());
            }
        };
        job["status"] = "running".into();
        job["pid"] = child.id().into();
        if let Err(error) = self.update_job(&id, &job) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        let initial = job.clone();
        let dashboard = self.clone();
        let wait = move || {
            let _lock = lock;
            match child.wait() {
                Ok(exit) => {
                    job["status"] = if exit.success() {
                        "succeeded"
                    } else {
                        "failed"
                    }
                    .into();
                    job["exit_code"] = exit.code().into();
                }
                Err(error) => {
                    job["status"] = "unverified".into();
                    job["error"] = error.to_string().into();
                }
            }
            job["finished_at_ms"] = now_ms().into();
            if let Err(error) = dashboard.update_job(&id, &job) {
                eprintln!("dashboard receipt write failed: {error}");
            }
            job
        };
        if background {
            std::thread::spawn(wait);
            Ok(initial)
        } else {
            Ok(wait())
        }
    }

    fn update_job(&self, id: &str, job: &Value) -> Result<()> {
        self.connect()?.execute(
            "UPDATE dashboard_jobs SET payload = ? WHERE id = ?",
            params![serde_json::to_string(job)?, id],
        )?;
        Ok(())
    }

    fn jobs(&self, before: Option<&str>) -> Result<Value> {
        if let Some(before) = before {
            validate_identifier(before, "job cursor")?;
        }
        let db = self.connect()?;
        let mut query = db.prepare(
            "SELECT payload FROM dashboard_jobs WHERE (?1 IS NULL OR (created_at_ms,id) < (SELECT created_at_ms,id FROM dashboard_jobs WHERE id=?1)) ORDER BY created_at_ms DESC, id DESC LIMIT 100",
        )?;
        let values = query
            .query_map([before], |r| r.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut jobs = values
            .iter()
            .map(|p| serde_json::from_str::<Value>(p))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        // An OS lock, not a recycled PID, proves an active job owner. A receipt
        // left by a crashed service is historical evidence with unknown outcome.
        let lock_path = safe_child(&self.data_dir, Path::new("dashboard/job.lock"))?;
        let active_owner = if lock_path.is_file() {
            let lock = OpenOptions::new().read(true).write(true).open(lock_path)?;
            lock.try_lock().is_err()
        } else {
            false
        };
        for job in &mut jobs {
            if !active_owner && matches!(job["status"].as_str(), Some("starting" | "running")) {
                job["observed_status"] = "unverified".into();
            }
        }
        Ok(
            json!({"jobs":jobs,"limit":100,"next_before":if jobs.len()==100 { jobs.last().map(|j|&j["id"]) } else { None },"status_scope":"process_execution","note":"A nonterminal receipt after service restart is unverified; jobs are never automatically replayed."}),
        )
    }

    fn job_log(&self, id: &str, stream: &str) -> Result<Value> {
        validate_identifier(id, "job id")?;
        if !["stdout", "stderr"].contains(&stream) {
            return Err(Error::Invalid("unknown log stream".into()));
        }
        let path = safe_child(
            &self.data_dir,
            &Path::new("dashboard/jobs")
                .join(id)
                .join(format!("{stream}.log")),
        )?;
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        let start = size.saturating_sub(65536);
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        file.take(65536).read_to_end(&mut bytes)?;
        Ok(
            json!({"id":id,"stream":stream,"size_bytes":size,"tail_truncated":start>0,"text":String::from_utf8_lossy(&bytes)}),
        )
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

/// Derive forms from clap so adding an agent command also adds its human form.
pub fn catalog() -> Value {
    fn collect(command: &clap::Command, path: Vec<String>, out: &mut Vec<Value>) {
        let children: Vec<_> = command
            .get_subcommands()
            .filter(|c| c.get_name() != "help")
            .collect();
        if !children.is_empty() {
            for child in children {
                if ["observe", "dashboard"].contains(&child.get_name()) && path.is_empty() {
                    continue;
                }
                let mut next = path.clone();
                next.push(child.get_name().into());
                collect(child, next, out);
            }
            return;
        }
        let arguments:Vec<_>=command.get_arguments().filter(|a| !["help","version","project","json"].contains(&a.get_id().as_str()) && !a.is_hide_set()).map(|a| {
            let takes_value=a.get_action().takes_values();
            let arity=a.get_num_args();
            json!({"id":a.get_id().as_str(),"long":a.get_long(),"required":a.is_required_set(),"takes_value":takes_value,
                "repeat":matches!(a.get_action(),clap::ArgAction::Append),
                "multiple":arity.is_some_and(|n|n.max_values()>1)||matches!(a.get_action(),clap::ArgAction::Append),
                "help":a.get_help().map(ToString::to_string),"defaults":a.get_default_values().iter().map(|s|s.to_string_lossy()).collect::<Vec<_>>(),
                "choices":a.get_value_parser().possible_values().map(|v|v.filter(|p|!p.is_hide_set()).map(|p|p.get_name().to_string()).collect::<Vec<_>>()).unwrap_or_default()})
        }).collect();
        out.push(json!({"path":path,"description":command.get_about().map(ToString::to_string),"arguments":arguments}));
    }
    let mut command = Cli::command();
    command.build();
    let mut commands = Vec::new();
    collect(&command, Vec::new(), &mut commands);
    json!({"commands":commands,"execution":"fixed glr executable; argv validation; no shell"})
}

pub fn execute(project: &Project, command: &DashboardCommand, as_json: bool) -> Result<i32> {
    // The lifecycle commands describe servers this process does not own, so
    // they run before `Dashboard::new` -- which opens the project store, and
    // creating tables is not an acceptable side effect of asking who is up.
    match command {
        DashboardCommand::Instances { all, prune } => {
            return lifecycle_instances(project, *all, *prune, as_json);
        }
        DashboardCommand::Stop {
            instance,
            port,
            all,
        } => return lifecycle_stop(project, instance.as_deref(), *port, *all, as_json),
        _ => {}
    }
    let dashboard = Dashboard::new(project)?;
    let value = match command {
        DashboardCommand::Catalog => catalog(),
        DashboardCommand::Presets => serde_json::to_value(dashboard.presets()?)?,
        DashboardCommand::SavePreset { file } => {
            dashboard.save_preset(serde_json::from_slice(&fs::read(file)?)?)?
        }
        DashboardCommand::Jobs { before } => dashboard.jobs(before.as_deref())?,
        DashboardCommand::JobLog { id, stream } => dashboard.job_log(id, stream)?,
        DashboardCommand::Run { preset } => dashboard.launch(
            JobRequest {
                preset: Some(preset.clone()),
                argv: None,
                request_id: format!("request-{}", uuid::Uuid::new_v4().simple()),
            },
            false,
        )?,
        // Routed above; listed so adding a lifecycle command cannot silently
        // fall through to the store-opening path.
        DashboardCommand::Instances { .. } | DashboardCommand::Stop { .. } => {
            unreachable!("lifecycle commands are handled before the store is opened")
        }
    };
    crate::commands::emit(
        "dashboard",
        &json!({"schema_version":SCHEMA,"data":value}),
        as_json,
    )?;
    Ok(value
        .get("exit_code")
        .and_then(Value::as_i64)
        .unwrap_or_else(|| {
            if matches!(value["status"].as_str(), Some("failed" | "unverified")) {
                1
            } else {
                0
            }
        }) as i32)
}

/// List live workbench servers: this project's by default, this user's with
/// `--all`.
fn lifecycle_instances(project: &Project, all: bool, prune: bool, as_json: bool) -> Result<i32> {
    let directory = crate::instance::lease_dir()?;
    let survey =
        crate::instance::survey(&directory, if all { None } else { Some(project) }, prune)?;
    crate::commands::emit("dashboard.instances", &survey, as_json)?;
    Ok(0)
}

/// Ask live servers to stop.
///
/// Selection is scoped to this project unless `--instance` or `--port` names a
/// server outright -- an id is a choice the caller made, while "the live server
/// for this project" must not silently pick one of several. Every target has to
/// prove its identity before a request is sent, and the request only asks: the
/// server retires itself, so nothing is killed from here.
fn lifecycle_stop(
    project: &Project,
    instance: Option<&str>,
    port: Option<u16>,
    all: bool,
    as_json: bool,
) -> Result<i32> {
    let directory = crate::instance::lease_dir()?;
    let targets = if let Some(id) = instance {
        vec![crate::instance::find(&directory, id)?]
    } else if let Some(port) = port {
        vec![crate::instance::find_by_port(&directory, port)?]
    } else {
        let live = crate::instance::live_for(&directory, project)?;
        if live.is_empty() {
            return Err(Error::Invalid(
                "no live workbench server serves this project; `glr dashboard instances --all` \
                 lists every server this user has"
                    .into(),
            ));
        }
        if !all && live.len() > 1 {
            return Err(Error::Invalid(format!(
                "{} live servers serve this project; name one with --instance <id>, or pass --all",
                live.len()
            )));
        }
        live
    };
    let mut requested = Vec::new();
    let mut failed = Vec::new();
    for target in &targets {
        if let Err(error) = crate::instance::request_stop(target) {
            failed.push(format!("{}: {error}", target.instance_id));
            continue;
        }
        // The server answers before it has actually retired, so the receipt
        // reports what was observed afterwards instead of asserting an exit
        // nobody saw.
        requested.push(json!({
            "instance_id": target.instance_id,
            "url": target.url,
            "pid": target.pid,
            "environment_id": target.environment_id,
            "confirmed": wait_for_exit(target),
        }));
    }
    let confirmed = requested
        .iter()
        .filter(|row| row["confirmed"] == true)
        .count();
    crate::commands::emit(
        "dashboard.stop",
        &json!({"schema_version": crate::instance::SCHEMA,
            "requested": requested,
            "failed": failed,
            "confirmed": confirmed,
            "note": "The server retires itself after answering this request; nothing is killed. \
                     `confirmed` means its address stopped answering health before the wait \
                     expired."}),
        as_json,
    )?;
    Ok(if failed.is_empty() && confirmed == requested.len() {
        0
    } else {
        1
    })
}

/// Wait until `instance`'s address stops answering as itself.
///
/// Probed rather than read from the lease file, so the confirmation holds even
/// for a server that never managed to publish one.
fn wait_for_exit(instance: &crate::instance::Instance) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if crate::instance::probe(instance).0 != crate::instance::State::Live {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
