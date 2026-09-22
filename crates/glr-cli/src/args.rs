use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "glr",
    version,
    about = "Agent-first Game Learning Runtime control plane"
)]
pub struct Cli {
    #[arg(long, global = true, default_value = ".")]
    pub project: PathBuf,
    /// Project-relative glr.run-context.v1 file to freeze for role execution.
    #[arg(long, global = true)]
    pub context: Option<PathBuf>,
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Publish or inspect passive, persistent bridge diagnostics.
    Telemetry {
        #[command(subcommand)]
        command: TelemetryCommand,
    },
    /// Open the local training dashboard, or manage its persisted presets/jobs.
    Dashboard {
        /// Loopback port. Omit to try 7432, then this project's own slot, then
        /// a free port; pass 0 to always let the operating system choose. An
        /// explicit port is used exactly as given and never moved.
        #[arg(long)]
        port: Option<u16>,
        #[command(subcommand)]
        command: Option<DashboardCommand>,
    },
    /// Validate a GLR project and its local deployment dependencies.
    Doctor,
    /// Start the configured game/runtime bridge.
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommand,
    },
    /// Run the configured trainer and persist its evidence.
    Train {
        #[arg(long)]
        no_capture: bool,
        /// Disable the automatic localhost observation server.
        #[arg(long)]
        no_observe: bool,
        /// Disable automatic recording of the trainer window.
        #[arg(long)]
        no_recording: bool,
    },
    /// Record the main window of a running process (Windows only).
    Recording {
        #[command(subcommand)]
        command: RecordingCommand,
    },
    /// Host a caller-owned long-lived loop inside a GLR run.
    ///
    /// The child is an ordinary process, not a project role: GLR owns the run and
    /// the telemetry ingest bindings, the caller owns the loop. Everything after
    /// `--` is the child command line.
    Host {
        /// Run the child without opening a telemetry ingest endpoint.
        #[arg(long)]
        no_telemetry: bool,
        /// Stop the child after this many seconds; omit to run until it exits.
        #[arg(long)]
        timeout_seconds: Option<f64>,
        /// The child command line; the first token is the program.
        #[arg(last = true, required = true)]
        argv: Vec<String>,
    },
    /// Serve the bundled live dashboard and read-only observation API.
    Observe {
        /// Loopback port. Omit to try 7432, then this project's own slot, then
        /// a free port; pass 0 to always let the operating system choose.
        #[arg(long)]
        port: Option<u16>,
        /// Inspect a verified backup without modifying or restoring it.
        #[arg(long)]
        archive: Option<PathBuf>,
    },
    /// Create, verify, or restore durable observation backups.
    Backup {
        #[command(subcommand)]
        command: BackupCommand,
    },
    /// Pursue a bounded agent-first learning objective.
    Goal {
        #[command(subcommand)]
        command: GoalCommand,
    },
    /// Query persisted runtime and training runs.
    Runs {
        #[command(subcommand)]
        command: RunsCommand,
    },
    /// Build an offline interactive report for a persisted run.
    Report {
        #[command(subcommand)]
        command: ReportCommand,
    },
    /// Inspect standard recording presets and the project storage layout.
    Capture {
        #[command(subcommand)]
        command: CaptureCommand,
    },
    /// Query learned and observed experience.
    Query {
        #[command(subcommand)]
        command: QueryCommand,
    },
    /// Move exact-environment spatial knowledge.
    Knowledge {
        #[command(subcommand)]
        command: KnowledgeCommand,
    },
    /// Verify and load a trained model bundle.
    Play {
        #[arg(long)]
        bundle: PathBuf,
    },
    /// Inspect and explicitly migrate a checkpoint contract manifest.
    Checkpoint {
        #[command(subcommand)]
        command: CheckpointCommand,
    },
    /// Start or resume a bounded durable multi-step command transaction.
    Transaction {
        #[command(subcommand)]
        command: TransactionCommand,
    },
    /// List, inspect, or run project-local declarative tasks from glr.toml.
    Task {
        #[command(subcommand)]
        command: TaskCommand,
    },
    /// Check or apply a checksum-verified GLR distribution update.
    Update(UpdateArgs),
    /// Plan, export, inspect, or import an offline source-only project package.
    Package {
        #[command(subcommand)]
        command: PackageCommand,
    },
    /// Inspect, install, and compose declarative project plugins.
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum TelemetryCommand {
    /// Print the versioned bridge batch JSON Schema.
    Schema,
    /// Ingest one JSON batch, or JSONL batches (atomic per line).
    Ingest {
        /// JSON file, or - for stdin. At most 64 KiB per batch.
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        jsonl: bool,
    },
    /// Inspect latest bridge status/state/progress by producer.
    State { run_id: String },
}

#[derive(Debug, Subcommand)]
pub enum PackageCommand {
    Plan {
        #[arg(long)]
        manifest: PathBuf,
    },
    Export {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Inspect {
        archive: PathBuf,
    },
    Import {
        archive: PathBuf,
        #[arg(long)]
        destination: PathBuf,
        #[arg(long)]
        expected_environment: String,
        #[arg(long)]
        expected_contract: String,
    },
    /// Offline synthetic conformance check for an already materialized package.
    ///
    /// Reports package validity, materialization, recipient-local overrides,
    /// declared dependency locks and prerequisites as separate axes. It never
    /// executes training, installs, downloads or resolves dependencies, and a
    /// conformant package is never reported as a successful training run.
    Conformance {
        archive: PathBuf,
        /// Re-assert the reviewed environment identity against the package.
        #[arg(long)]
        expected_environment: Option<String>,
        /// Re-assert the reviewed contract fingerprint against the package.
        #[arg(long)]
        expected_contract: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
pub enum CaptureCommand {
    /// List presets, or show one complete FFmpeg output configuration.
    Preset {
        #[arg(default_value = "training-balanced")]
        name: String,
        /// List the available preset summaries instead of selecting one.
        #[arg(long)]
        list: bool,
    },
    /// Show canonical logs, capture, dataset, checkpoint, and report paths.
    Layout,
}

#[derive(Debug, Clone, Subcommand)]
pub enum RecordingCommand {
    /// Record the main window of one process for a fixed duration.
    Run {
        /// Process id whose main window is recorded.
        #[arg(long)]
        pid: u32,
        /// Stop recording after this many seconds.
        #[arg(long, default_value_t = 10)]
        seconds: u64,
        /// Output directory; defaults to the configured recording directory.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum TaskCommand {
    /// List configured project tasks without executing them.
    List,
    /// Show one validated task definition.
    Show { name: String },
    /// Run one task and its declared dependencies.
    Run {
        name: String,
        /// Set a declared task parameter as NAME=VALUE.
        #[arg(long = "set", value_name = "NAME=VALUE")]
        parameters: Vec<String>,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum PluginCommand {
    /// Validate a local plugin bundle without executing it.
    Inspect {
        #[arg(long)]
        source: PathBuf,
    },
    /// Atomically copy a validated local bundle into the project store.
    Install {
        #[arg(long)]
        source: PathBuf,
        /// Expected SHA-256 digest of the inspected bundle.
        #[arg(long)]
        sha256: Option<String>,
    },
    /// List installed plugin bundles.
    List,
    /// Report static plugin readiness without starting plugins.
    Health {
        #[arg(long)]
        profile: Option<String>,
    },
    /// Remove a plugin bundle that is not enabled by a profile.
    Remove {
        id: String,
        #[arg(long)]
        version: Option<String>,
    },
    /// Manage explicit plugin profiles.
    Profile {
        #[command(subcommand)]
        command: PluginProfileCommand,
    },
}

#[derive(Debug, Clone, Subcommand)]
pub enum PluginProfileCommand {
    /// List saved plugin profiles.
    List,
    /// Show one saved profile.
    Show { name: String },
    /// Resolve one profile against installed bundles.
    Resolve { name: String },
    /// Enable or add a plugin in a profile.
    Enable {
        name: String,
        id: String,
        #[arg(long)]
        version: Option<String>,
        #[arg(long = "grant")]
        grants: Vec<String>,
    },
    /// Disable a plugin in a profile.
    Disable { name: String, id: String },
}

#[derive(Debug, Clone, Subcommand)]
pub enum TransactionCommand {
    /// Persist a bounded ordered step list for one running run.
    Begin {
        #[arg(long)]
        run_id: String,
        #[arg(long)]
        transaction_id: String,
        #[arg(long)]
        steps: PathBuf,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=16))]
        max_resume_attempts: u32,
    },
    /// Record one refusal or advance the next step after an accepted command.
    Resume {
        #[arg(long)]
        transaction_id: String,
        #[arg(long)]
        refusal: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
pub enum CheckpointCommand {
    /// Report or apply a contract migration after an explicit confirmation.
    Migrate {
        /// Checkpoint manifest to inspect or rewrite.
        #[arg(long)]
        manifest: PathBuf,
        /// JSON file containing the live checkpoint contract.
        #[arg(long)]
        contract: PathBuf,
        /// Confirm the migration and create adjacent backups.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum RuntimeCommand {
    Start,
}

#[derive(Debug, Subcommand)]
pub enum GoalCommand {
    /// Save a goal file as this project's default goal and make it active.
    ///
    /// Pair it with the global `--context` flag to bind the run context that
    /// freezes the goal's training and reward inputs.
    Set {
        /// Project-relative `glr.agent-goal.v1` file to bind.
        #[arg(long)]
        goal: PathBuf,
    },
    /// Show the active goal, or one saved goal by id.
    Show {
        /// Saved goal id. Omit to show the active goal.
        goal_id: Option<String>,
    },
    /// Make one saved goal the active default goal.
    Use { goal_id: String },
    /// List every goal saved in this project.
    List,
    /// Pursue a goal. Omit `--goal` to pursue the saved default goal.
    Run {
        #[arg(long)]
        goal: Option<PathBuf>,
        #[arg(long)]
        no_capture: bool,
        #[arg(long)]
        no_observe: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RunStatusArg {
    Running,
    Succeeded,
    Failed,
    Interrupted,
}

impl RunStatusArg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum RunsCommand {
    /// Read a resumable event/metric page, from disk or a verified backup.
    Trace {
        run_id: String,
        #[arg(long, default_value_t = -1, allow_hyphen_values = true, value_parser = clap::value_parser!(i64).range(-1..))]
        events_after: i64,
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i64).range(0..))]
        metrics_after: i64,
        #[arg(long, default_value_t = 250, value_parser = clap::value_parser!(u32).range(1..=250))]
        limit: u32,
        #[arg(long)]
        archive: Option<PathBuf>,
    },
    /// Tail a durable managed role log using a byte cursor.
    Log {
        run_id: String,
        #[arg(long, default_value = "trainer.log")]
        path: String,
        #[arg(long)]
        offset: Option<u64>,
        #[arg(long)]
        archive: Option<PathBuf>,
    },
    List {
        #[arg(long)]
        status: Option<RunStatusArg>,
        #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..=1000))]
        limit: u32,
    },
    Show {
        run_id: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum DashboardCommand {
    /// Show the same command forms exposed in the web dashboard.
    Catalog,
    /// List validated training presets.
    Presets,
    /// Save a declarative preset from a JSON file.
    SavePreset {
        #[arg(long)]
        file: PathBuf,
    },
    /// Page through durable dashboard job receipts.
    Jobs {
        #[arg(long)]
        before: Option<String>,
    },
    /// Read persisted stdout/stderr from one dashboard operation.
    JobLog {
        id: String,
        #[arg(long, default_value = "stdout", value_parser = ["stdout", "stderr"])]
        stream: String,
    },
    /// Run a preset in the foreground with a durable dashboard receipt.
    Run { preset: String },
    /// List live workbench servers, with the project, port, version and age of each.
    Instances {
        /// Every server this user has running, not only this project's.
        #[arg(long)]
        all: bool,
        /// Forget leases whose address stopped answering. Never terminates anything.
        #[arg(long)]
        prune: bool,
    },
    /// Ask a live server to stop. The server retires itself; nothing is killed.
    Stop {
        /// Instance id from `dashboard instances`.
        #[arg(long)]
        instance: Option<String>,
        /// Address of the server to stop.
        #[arg(long)]
        port: Option<u16>,
        /// Stop every live server for this project.
        #[arg(long, conflicts_with_all = ["instance", "port"])]
        all: bool,
    },
}

#[derive(Debug, Subcommand)]
pub enum BackupCommand {
    /// Snapshot SQLite and archive completed-run files; active runs are DB-only.
    Create {
        #[arg(long)]
        output: PathBuf,
    },
    /// Verify every recorded file digest and the SQLite integrity check.
    Verify { archive: PathBuf },
    /// Copy a verified archive to a new directory; never overwrite live storage.
    Restore {
        archive: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
pub enum ReportCommand {
    Build {
        /// Persisted run identifier to render.
        run_id: String,
        /// Optional report directory inside the run directory.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ResearchCategoryArg {
    Mechanic,
    Strategy,
    RewardHypothesis,
    Safety,
    Navigation,
}

impl ResearchCategoryArg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mechanic => "mechanic",
            Self::Strategy => "strategy",
            Self::RewardHypothesis => "reward-hypothesis",
            Self::Safety => "safety",
            Self::Navigation => "navigation",
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum QueryCommand {
    Entities {
        #[arg(long)]
        world: String,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        name: Option<String>,
        #[arg(long, num_args = 3)]
        near: Option<Vec<f64>>,
        #[arg(long)]
        radius: Option<f64>,
        #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..=1000))]
        limit: u32,
    },
    Routes {
        #[arg(long)]
        world: String,
        #[arg(long)]
        from_entity: Option<String>,
        #[arg(long)]
        to_entity: Option<String>,
        #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..=1000))]
        limit: u32,
    },
    Edges {
        #[arg(long)]
        world: String,
        #[arg(long)]
        from_node: Option<String>,
        #[arg(long)]
        to_node: Option<String>,
        #[arg(long)]
        status: Option<EdgeStatusArg>,
        #[arg(long, default_value_t = 0)]
        at_ns: u64,
        #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..=1000))]
        limit: u32,
    },
    Research {
        #[arg(long = "tag")]
        tags: Vec<String>,
        #[arg(long)]
        category: Option<ResearchCategoryArg>,
        #[arg(long)]
        verified_only: bool,
        #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..=1000))]
        limit: u32,
    },
}

#[derive(Debug, Subcommand)]
pub enum KnowledgeCommand {
    Export {
        #[arg(long)]
        output: PathBuf,
    },
    Import {
        #[arg(long = "input")]
        source: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum EdgeStatusArg {
    Unknown,
    Traversable,
    Blocked,
    Stale,
}

impl EdgeStatusArg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Traversable => "traversable",
            Self::Blocked => "blocked",
            Self::Stale => "stale",
        }
    }
}

#[derive(Debug, Args)]
pub struct UpdateArgs {
    /// Check the release and checksum contract without changing files.
    #[arg(long, conflicts_with = "yes")]
    pub check: bool,
    /// Backward-compatible alias; updates are applied by default.
    #[arg(long, hide = true)]
    pub yes: bool,
    /// Override the target directory that receives bundled GLR skills.
    #[arg(long, conflicts_with = "no_skills")]
    pub skills_dir: Option<PathBuf>,
    /// Update binaries only.
    #[arg(long)]
    pub no_skills: bool,
}

impl UpdateArgs {
    pub fn applies_update(&self) -> bool {
        !self.check
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{Cli, Command, GoalCommand};
    use clap::Parser;

    #[test]
    fn update_applies_by_default_and_check_remains_read_only() {
        let cli = Cli::try_parse_from(["glr", "update"]).unwrap();
        let Command::Update(arguments) = cli.command else {
            panic!("expected update command");
        };
        assert!(arguments.applies_update());

        let cli = Cli::try_parse_from(["glr", "update", "--check"]).unwrap();
        let Command::Update(arguments) = cli.command else {
            panic!("expected update command");
        };
        assert!(!arguments.applies_update());
    }

    #[test]
    fn legacy_yes_flag_still_applies_an_update() {
        let cli = Cli::try_parse_from(["glr", "update", "--yes"]).unwrap();
        let Command::Update(arguments) = cli.command else {
            panic!("expected update command");
        };
        assert!(arguments.yes);
        assert!(arguments.applies_update());
    }

    #[test]
    fn goal_run_accepts_an_explicit_goal_and_omits_it_for_the_default() {
        let cli = Cli::try_parse_from(["glr", "goal", "run"]).unwrap();
        let Command::Goal { command } = cli.command else {
            panic!("expected goal command");
        };
        let GoalCommand::Run { goal, .. } = command else {
            panic!("expected goal run");
        };
        assert_eq!(goal, None);

        let cli = Cli::try_parse_from(["glr", "goal", "run", "--goal", "goals/a.json"]).unwrap();
        let Command::Goal { command } = cli.command else {
            panic!("expected goal command");
        };
        let GoalCommand::Run { goal, .. } = command else {
            panic!("expected goal run");
        };
        assert_eq!(goal.as_deref(), Some(Path::new("goals/a.json")));
    }

    #[test]
    fn goal_set_binds_a_goal_with_the_global_context_flag() {
        let cli = Cli::try_parse_from([
            "glr",
            "--context",
            "config/contexts/ranked.toml",
            "goal",
            "set",
            "--goal",
            "goals/a.json",
        ])
        .unwrap();
        let Command::Goal { command } = cli.command else {
            panic!("expected goal command");
        };
        let GoalCommand::Set { goal } = command else {
            panic!("expected goal set");
        };
        assert_eq!(goal, Path::new("goals/a.json"));
        assert_eq!(
            cli.context.as_deref(),
            Some(Path::new("config/contexts/ranked.toml"))
        );
    }

    #[test]
    fn goal_show_takes_an_optional_saved_goal_id() {
        let cli = Cli::try_parse_from(["glr", "goal", "show"]).unwrap();
        let Command::Goal { command } = cli.command else {
            panic!("expected goal command");
        };
        let GoalCommand::Show { goal_id } = command else {
            panic!("expected goal show");
        };
        assert_eq!(goal_id, None);

        let cli = Cli::try_parse_from(["glr", "goal", "show", "goal.reach-destination"]).unwrap();
        let Command::Goal { command } = cli.command else {
            panic!("expected goal command");
        };
        let GoalCommand::Show { goal_id } = command else {
            panic!("expected goal show");
        };
        assert_eq!(goal_id.as_deref(), Some("goal.reach-destination"));
    }
}
