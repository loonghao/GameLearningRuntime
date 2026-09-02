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
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
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
    /// Check or apply a checksum-verified GLR distribution update.
    Update(UpdateArgs),
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
    Run {
        #[arg(long)]
        goal: PathBuf,
        #[arg(long)]
        no_capture: bool,
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
    /// Confirm replacement of managed GLR binaries and skills.
    #[arg(long)]
    pub yes: bool,
    /// Override the target directory that receives bundled GLR skills.
    #[arg(long, conflicts_with = "no_skills")]
    pub skills_dir: Option<PathBuf>,
    /// Update binaries only.
    #[arg(long)]
    pub no_skills: bool,
}
