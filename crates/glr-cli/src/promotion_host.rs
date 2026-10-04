//! Privileged, in-process host seam for checkpoint installation.
//!
//! Provision the host secret outside worker configuration and keep this object
//! in the supervisor. This is an API trust boundary, not OS authentication or a
//! sandbox: a process with direct write access to SQLite/files can corrupt them.
//! Worker metric/event APIs and the CLI cannot issue these ledger records.
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::contracts::{Authority, GoalEvidenceBundle, PromotionMode, read_json, sha256_file};
pub use crate::error::{Error, Result};
use crate::project::validate_identifier;
use crate::promotion_journal::{canonical_live_path, validate_ancestors};
use crate::store::{CheckpointPromotionRequest, Store};

pub const AUTHORIZATION_SCHEMA: &str = "glr.checkpoint-promotion-authorization.v1";
const LEDGER_SCHEMA: &str = "glr.host-promotion-ledger.v1";
pub const EVALUATION_ZERO_METRICS: [&str; 7] = [
    "evaluation.parameter_mutations",
    "evaluation.reset_identity_mismatches",
    "evaluation.stale_observation_updates",
    "evaluation.dead_or_loading_updates",
    "evaluation.illegal_action_bootstraps",
    "evaluation.reward_attribution_errors",
    "evaluation.action_interval_errors",
];
fn required_correctness() -> Vec<FixedCriterion> {
    EVALUATION_ZERO_METRICS
        .iter()
        .map(|name| FixedCriterion {
            name: (*name).into(),
            source: "evaluation.contract".into(),
            mode: Direction::Min,
            threshold: 0.0,
        })
        .collect()
}

/// Fixed external evaluator output. Numeric zero alone cannot imply that a
/// check was measured. This envelope is separate from legacy CLI evidence.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FixedEvaluationReport {
    pub schema_version: String,
    pub coverage: std::collections::BTreeMap<String, String>,
    pub evidence_bundle: GoalEvidenceBundle,
}
fn read_fixed_evaluation(path: &Path) -> Result<GoalEvidenceBundle> {
    let report: FixedEvaluationReport = read_json(path, "fixed checkpoint evaluation report")?;
    if report.schema_version != "glr.checkpoint-evaluation.v1"
        || report.coverage.len() != 7
        || EVALUATION_ZERO_METRICS
            .iter()
            .any(|name| report.coverage.get(*name).map(String::as_str) != Some("measured"))
    {
        return Err(Error::Contract(
            "fixed checkpoint evaluation coverage is missing, unknown or inapplicable".into(),
        ));
    }
    report.evidence_bundle.validate()?;
    Ok(report.evidence_bundle)
}

/// Secret possession grants the host role. This must be provisioned by the
/// embedding trusted application, never derived from worker metadata.
pub struct HostAuthority([u8; 32]);

impl HostAuthority {
    pub fn from_secret(secret: [u8; 32]) -> Result<Self> {
        if secret == [0; 32] {
            return Err(Error::Invalid(
                "host authority secret cannot be zero".into(),
            ));
        }
        Ok(Self(secret))
    }
    fn fingerprint(&self) -> String {
        format!("{:x}", Sha256::digest(self.0))
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum Direction {
    Max,
    Min,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FixedCriterion {
    pub name: String,
    pub source: String,
    pub mode: Direction,
    pub threshold: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HostPolicy {
    pub goal_id: String,
    pub environment_id: String,
    pub protocol_version: String,
    pub target_id: String,
    pub environment_config_sha256: String,
    pub evaluator_sha256: String,
    pub evaluation_suite_sha256: String,
    pub metric: String,
    pub source: String,
    pub mandatory_criteria: Vec<FixedCriterion>,
    pub mode: Direction,
    pub minimum_improvement: f64,
    pub proposer_id: String,
    pub worker_ids: Vec<String>,
    pub evaluator_id: String,
    pub supervisor_id: String,
    pub reviewer_id: String,
    pub max_wall_seconds: u64,
    pub max_training_steps: u64,
}

impl HostPolicy {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("goal", &self.goal_id),
            ("environment", &self.environment_id),
            ("target", &self.target_id),
            ("metric", &self.metric),
            ("source", &self.source),
            ("proposer", &self.proposer_id),
            ("evaluator", &self.evaluator_id),
            ("supervisor", &self.supervisor_id),
            ("reviewer", &self.reviewer_id),
        ] {
            validate_identifier(value, name)?;
        }
        if self.protocol_version.trim().is_empty()
            || self.protocol_version.len() > 64
            || self.max_wall_seconds == 0
            || self.max_wall_seconds > 86400 * 30
            || self.max_training_steps == 0
            || self.worker_ids.is_empty()
            || self.worker_ids.len() > 32
            || !self.minimum_improvement.is_finite()
            || self.minimum_improvement < 0.0
        {
            return Err(Error::Invalid(
                "invalid bounded host promotion policy".into(),
            ));
        }
        for digest in [
            &self.environment_config_sha256,
            &self.evaluator_sha256,
            &self.evaluation_suite_sha256,
        ] {
            validate_digest(digest)?;
        }
        let workers = self.worker_ids.iter().collect::<HashSet<_>>();
        if workers.len() != self.worker_ids.len() {
            return Err(Error::Invalid(
                "duplicate supervised worker identity".into(),
            ));
        }
        for worker in &self.worker_ids {
            validate_identifier(worker, "worker")?;
        }
        if self.mandatory_criteria.is_empty() || self.mandatory_criteria.len() > 64 {
            return Err(Error::Invalid(
                "fixed evaluation requires bounded mandatory correctness criteria".into(),
            ));
        }
        let mut keys = HashSet::new();
        for required in required_correctness() {
            if !self.mandatory_criteria.contains(&required) {
                return Err(Error::Contract(
                    "checkpoint policy lacks fixed exact-zero seven correctness checks".into(),
                ));
            }
        }
        for criterion in &self.mandatory_criteria {
            validate_identifier(&criterion.name, "mandatory metric")?;
            validate_identifier(&criterion.source, "mandatory source")?;
            if !criterion.threshold.is_finite()
                || !keys.insert((&criterion.name, &criterion.source))
            {
                return Err(Error::Invalid(
                    "invalid or duplicate mandatory criterion".into(),
                ));
            }
        }
        if self.reviewer_id == self.proposer_id
            || workers.contains(&self.reviewer_id)
            || self.reviewer_id == self.evaluator_id
            || self.reviewer_id == self.supervisor_id
            || workers.contains(&self.evaluator_id)
            || workers.contains(&self.supervisor_id)
            || self.evaluator_id == self.proposer_id
            || self.evaluator_id == self.supervisor_id
            || self.supervisor_id == self.proposer_id
        {
            return Err(Error::Contract(
                "review/evaluator/supervisor identities are not independent".into(),
            ));
        }
        Ok(())
    }
}

pub struct TrialRequest {
    pub run_id: String,
    pub trial_id: String,
    pub candidate_path: PathBuf,
    pub live_path: PathBuf,
    pub environment_config_path: PathBuf,
    pub evaluator_program: PathBuf,
    pub evaluator_args: Vec<String>,
    pub suite_path: PathBuf,
    pub evaluation_path: PathBuf,
    pub metric_floor: i64,
    pub training_steps: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FinalMeasurement {
    pub name: String,
    pub source: String,
    pub metric_id: i64,
    pub value: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PromotionAuthorization {
    pub schema_version: String,
    pub candidate_kind: String,
    pub evaluation_scope: String,
    pub authorization_id: String,
    pub trial_token: String,
    pub goal_id: String,
    pub artifact_sha256: String,
    pub environment_id: String,
    pub protocol_version: String,
    pub target_id: String,
    pub environment_config_sha256: String,
    pub run_id: String,
    pub trial_id: String,
    pub candidate_path: String,
    pub live_path: String,
    pub incumbent_sha256: Option<String>,
    pub evaluation_sha256: String,
    pub evaluator_sha256: String,
    pub evaluation_suite_sha256: String,
    pub final_measurement: FinalMeasurement,
    pub reviewer_id: String,
    pub proposer_id: String,
    pub worker_ids: Vec<String>,
    pub stop_receipt_sha256: String,
    pub host_fingerprint: String,
    pub host_epoch: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ObservedProcess {
    identity: String,
    process_id: u32,
    exit_code: Option<i32>,
    observed_terminal: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WorkerBinding {
    worker_token: String,
    identity: String,
    run_id: String,
    trial_id: String,
    host_fingerprint: String,
    host_epoch: String,
    host_session_id: String,
    environment_id: String,
    target_id: String,
    store_path: String,
    environment_config_sha256: String,
    program_sha256: String,
    argv: Vec<String>,
    process_id: Option<u32>,
    exit_code: Option<i32>,
    observed_terminal: bool,
}

/// A worker launched and registered by this host. A raw or foreign Child
/// cannot be substituted as evidence that the configured worker has stopped.
pub struct SupervisedWorker {
    binding: WorkerBinding,
    child: Option<Child>,
    deadline: Instant,
    _store_guard: fs::File,
}
impl Drop for SupervisedWorker {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let result = (|| -> std::io::Result<std::process::ExitStatus> {
                match child.try_wait()? {
                    Some(s) => Ok(s),
                    None => {
                        child.kill()?;
                        child.wait()
                    }
                }
            })();
            if let Ok(status) = result {
                self.binding.exit_code = status.code();
                self.binding.observed_terminal = true;
                if let Ok(connection) = Connection::open(&self.binding.store_path) {
                    let _=connection.execute("UPDATE host_checkpoint_promotion_workers SET state='stopped',worker_json=? WHERE worker_token=?",params![encode(&self.binding).unwrap_or_default(),self.binding.worker_token]);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct StopReceipt {
    schema_version: String,
    scope: String,
    trial_token: String,
    run_id: String,
    trial_id: String,
    artifact_sha256: String,
    supervisor_id: String,
    processes: Vec<ObservedProcess>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Ledger {
    schema_version: String,
    host_session_id: String,
    pub(crate) binding: PromotionAuthorization,
    policy: HostPolicy,
    store_path: String,
    config_path: String,
    evaluator_program: String,
    evaluator_args: Vec<String>,
    suite_path: String,
    evaluation_path: String,
    metric_floor: i64,
    started_at_ns: i64,
    training_steps: u64,
    processes: Vec<ObservedProcess>,
    workers: Vec<WorkerBinding>,
    measurements: Vec<FinalMeasurement>,
    stop: Option<StopReceipt>,
    review: Option<bool>,
}

/// Opaque host-owned process handles. No Deserialize or worker stop setter.
pub struct SupervisedTrial {
    token: String,
    ledger: Ledger,
    children: Vec<(String, Child)>,
    deadline: Instant,
    _store_guard: fs::File,
}
impl SupervisedTrial {
    pub fn trial_token(&self) -> &str {
        &self.token
    }
}
impl Drop for SupervisedTrial {
    fn drop(&mut self) {
        // Only owned handles are touched. A dropped/failed host session cannot
        // silently leave its evaluator running, and creates no stop approval.
        let mut observed = true;
        for (index, (_, child)) in self.children.iter_mut().enumerate() {
            let result = (|| -> std::io::Result<std::process::ExitStatus> {
                match child.try_wait()? {
                    Some(status) => Ok(status),
                    None => {
                        child.kill()?;
                        child.wait()
                    }
                }
            })();
            match result {
                Ok(status) => {
                    self.ledger.processes[index].exit_code = status.code();
                    self.ledger.processes[index].observed_terminal = true;
                }
                Err(_) => observed = false,
            }
        }
        if let Ok(connection) = Connection::open(&self.ledger.store_path) {
            let current: std::result::Result<(String, String), _> = connection.query_row(
                "SELECT fingerprint,epoch FROM promotion_host_authority WHERE singleton=1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            );
            if current.ok()
                == Some((
                    self.ledger.binding.host_fingerprint.clone(),
                    self.ledger.binding.host_epoch.clone(),
                ))
            {
                if observed && set_stop(self).is_ok() {
                    let _ = persist_failed_stop(&connection, self);
                } else {
                    let _=connection.execute("UPDATE host_checkpoint_promotion_trials SET state='quarantined' WHERE trial_token=? AND state='started'",[&self.token]);
                }
            }
        }
    }
}

pub struct PromotionHost {
    store: Store,
    fingerprint: String,
    epoch: String,
    session_id: String,
}

impl PromotionHost {
    /// Reopen a previously provisioned host ledger. This never adopts legacy
    /// run or metric records, nor creates authority for an unprovisioned store.
    pub fn open(path: PathBuf, authority: HostAuthority) -> Result<Self> {
        Self::open_mode(path, authority, false)
    }

    /// Explicitly provision a completely empty store for a trusted supervisor.
    /// No CLI option, worker metric, or existing run grants this authority.
    /// Existing populated stores require a separate reviewed migration.
    pub fn provision_empty(path: PathBuf, authority: HostAuthority) -> Result<Self> {
        Self::open_mode(path, authority, true)
    }

    fn open_mode(path: PathBuf, authority: HostAuthority, provision: bool) -> Result<Self> {
        if !provision && !path.is_file() {
            return Err(Error::Missing(path));
        }
        let store = Store::open(path)?;
        let fingerprint = authority.fingerprint();
        let mut connection = store.connect()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let stored: Option<(String, String)> = transaction
            .query_row(
                "SELECT fingerprint, epoch FROM promotion_host_authority WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let epoch = match stored {
            Some((expected, epoch)) if expected == fingerprint && !provision => epoch,
            Some((expected, _)) if expected == fingerprint => {
                return Err(Error::Contract(
                    "host is already provisioned; reopen its pinned ledger".into(),
                ));
            }
            Some(_) => {
                return Err(Error::Contract(
                    "host authority differs from provisioned owner".into(),
                ));
            }
            None => {
                if !provision {
                    return Err(Error::Contract("host ledger is unprovisioned; explicitly provision a completely empty store".into()));
                }
                let tables = transaction.prepare(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT GLOB 'sqlite_*'"
                )?.query_map([], |row| row.get::<_, String>(0))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                for table in tables {
                    // Quote the schema-owned identifier. Row values and table
                    // contents never become authority or a migration request.
                    let quoted = table.replace('"', "\"\"");
                    let populated: bool = transaction.query_row(
                        &format!("SELECT EXISTS(SELECT 1 FROM \"{quoted}\" LIMIT 1)"),
                        [],
                        |row| row.get(0),
                    )?;
                    if populated {
                        return Err(Error::Contract("populated store requires explicit reviewed migration; host provisioning refused".into()));
                    }
                }
                let epoch = Uuid::new_v4().simple().to_string();
                transaction.execute("INSERT INTO promotion_host_authority(singleton,fingerprint,epoch) VALUES (1,?,?)", params![fingerprint,epoch])?;
                epoch
            }
        };
        transaction.commit()?;
        Ok(Self {
            store,
            fingerprint,
            epoch,
            session_id: Uuid::new_v4().simple().to_string(),
        })
    }

    pub fn create_run(&self, policy: &HostPolicy) -> Result<String> {
        self.ensure_authority()?;
        policy.validate()?;
        let run = self.store.create_run(
            &policy.environment_id,
            &policy.protocol_version,
            "host-evaluation",
            serde_json::json!({"goal_id": policy.goal_id, "host_epoch": self.epoch,
                "host_fingerprint":self.fingerprint,"host_policy_sha256":hash_json(policy)?,"target_id":policy.target_id}),
        )?;
        self.store.connect()?.execute(
            "UPDATE runs SET environment_config_digest=? WHERE run_id=? AND status='running'",
            params![policy.environment_config_sha256, run.run_id],
        )?;
        Ok(run.run_id)
    }

    /// Snapshot the current metric boundary of this host's running policy.
    /// begin_trial rechecks this boundary before admitting fixed evaluation;
    /// this read neither authorizes old measurements nor grants worker access.
    pub fn trial_metric_floor(&self, policy: &HostPolicy, run_id: &str) -> Result<i64> {
        self.ensure_authority()?;
        policy.validate()?;
        self.check_run_policy(policy, run_id, true)?;
        self.store.latest_metric_id(run_id)
    }

    fn check_run_policy(
        &self,
        policy: &HostPolicy,
        run_id: &str,
        require_budget: bool,
    ) -> Result<crate::store::RunRecord> {
        let run = self.store.get_run(run_id)?;
        let config: Option<String> = self.store.connect()?.query_row(
            "SELECT environment_config_digest FROM runs WHERE run_id=?",
            [run_id],
            |row| row.get(0),
        )?;
        if run.status != "running"
            || run.environment_id != policy.environment_id
            || run.protocol_version != policy.protocol_version
            || config.as_deref() != Some(policy.environment_config_sha256.as_str())
            || run.metadata["goal_id"] != policy.goal_id
            || run.metadata["target_id"] != policy.target_id
            || run.metadata["host_policy_sha256"] != hash_json(policy)?
            || run.metadata["host_epoch"] != self.epoch
            || run.metadata["host_fingerprint"] != self.fingerprint
        {
            return Err(Error::Contract(
                "run is outside immutable host policy/config/target scope".into(),
            ));
        }
        if require_budget
            && crate::store::now_ns()? as i128 - run.started_at_ns as i128
                >= policy.max_wall_seconds as i128 * 1_000_000_000
        {
            return Err(Error::Contract(
                "host worker wall budget exhausted before dispatch".into(),
            ));
        }
        Ok(run)
    }

    fn ensure_authority(&self) -> Result<()> {
        let current: Option<(String, String)> = self
            .store
            .connect()?
            .query_row(
                "SELECT fingerprint,epoch FROM promotion_host_authority WHERE singleton=1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if current != Some((self.fingerprint.clone(), self.epoch.clone())) {
            return Err(Error::Contract(
                "provisioned host identity/epoch changed".into(),
            ));
        }
        Ok(())
    }

    pub fn spawn_worker(
        &self,
        policy: &HostPolicy,
        run_id: &str,
        trial_id: &str,
        identity: &str,
        config_path: &Path,
    ) -> Result<SupervisedWorker> {
        self.ensure_authority()?;
        policy.validate()?;
        validate_identifier(trial_id, "worker trial")?;
        if !policy.worker_ids.iter().any(|id| id == identity) {
            return Err(Error::Contract(
                "worker is outside operator-approved set".into(),
            ));
        }
        let run = self.check_run_policy(policy, run_id, true)?;
        let store_guard = self.store.instance_guard()?;
        if run.status != "running"
            || run.environment_id != policy.environment_id
            || run.protocol_version != policy.protocol_version
        {
            return Err(Error::Contract("worker run scope differs".into()));
        }
        verify_hash(config_path, &policy.environment_config_sha256)?;
        let config: serde_json::Value = read_json(config_path, "worker config")?;
        if config["environment_id"] != policy.environment_id
            || config["protocol_version"] != policy.protocol_version
            || config["target_id"] != policy.target_id
        {
            return Err(Error::Contract(
                "worker config target/environment/protocol differs".into(),
            ));
        }
        verify_dependencies(&config)?;
        let spec = &config["worker_commands"][identity];
        let argv: Vec<String> = serde_json::from_value(spec["argv"].clone())?;
        if argv.is_empty() || argv.len() > 128 || argv.iter().map(String::len).sum::<usize>() > 8192
        {
            return Err(Error::Invalid("fixed worker argv exceeds bound".into()));
        }
        let program = checked_file(Path::new(&argv[0]))?;
        verify_direct_inputs(&config, &argv[1..], &[config_path], None)?;
        let digest = spec["sha256"]
            .as_str()
            .ok_or_else(|| Error::Invalid("fixed worker executable hash is absent".into()))?;
        validate_digest(digest)?;
        verify_hash(&program, digest)?;
        let mut binding = WorkerBinding {
            worker_token: Uuid::new_v4().simple().to_string(),
            identity: identity.into(),
            run_id: run_id.into(),
            trial_id: trial_id.into(),
            host_fingerprint: self.fingerprint.clone(),
            host_epoch: self.epoch.clone(),
            host_session_id: self.session_id.clone(),
            environment_id: policy.environment_id.clone(),
            target_id: policy.target_id.clone(),
            store_path: text_path(&self.store.path().canonicalize()?)?,
            environment_config_sha256: policy.environment_config_sha256.clone(),
            program_sha256: digest.into(),
            argv: argv.clone(),
            process_id: None,
            exit_code: None,
            observed_terminal: false,
        };
        let mut connection = self.store.connect()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        verify_dispatch_run(&transaction, policy, run_id, &self.fingerprint, &self.epoch)?;
        let remaining_ns = policy.max_wall_seconds as i128 * 1_000_000_000
            - (crate::store::now_ns()? as i128 - run.started_at_ns as i128);
        if remaining_ns <= 0 {
            return Err(Error::Contract(
                "worker wall budget exhausted before launch intent".into(),
            ));
        }
        let deadline = Instant::now() + Duration::from_nanos(remaining_ns as u64);
        reserve_resource(
            &transaction,
            policy,
            run_id,
            trial_id,
            &self.epoch,
            &self.session_id,
        )?;
        transaction.execute("INSERT INTO host_checkpoint_promotion_workers(worker_token,run_id,trial_id,identity,state,worker_json) VALUES (?,?,?,?,'launch-pending',?)",params![binding.worker_token,run_id,trial_id,identity,encode(&binding)?])?;
        transaction.commit()?;
        let spawn = Command::new(program)
            .args(&argv[1..])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let mut child = match spawn {
            Ok(child) => child,
            Err(error) => {
                let _=self.store.connect()?.execute("UPDATE host_checkpoint_promotion_workers SET state='launch-failed' WHERE worker_token=?",[&binding.worker_token]);
                return Err(error.into());
            }
        };
        binding.process_id = Some(child.id());
        let result = (|| -> Result<usize> {
            Ok(self.store.connect()?.execute("UPDATE host_checkpoint_promotion_workers SET state='started',worker_json=? WHERE worker_token=? AND state='launch-pending'",params![encode(&binding)?,binding.worker_token])?)
        })();
        match result {
            Ok(1) => Ok(SupervisedWorker {
                binding,
                child: Some(child),
                deadline,
                _store_guard: store_guard,
            }),
            Ok(_) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(Error::Contract("worker launch record changed".into()))
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(error)
            }
        }
    }

    pub fn poll_worker(&self, worker: &mut SupervisedWorker) -> Result<bool> {
        self.ensure_authority()?;
        if worker.binding.host_fingerprint != self.fingerprint
            || worker.binding.host_epoch != self.epoch
            || worker.binding.host_session_id != self.session_id
            || worker.binding.store_path != text_path(&self.store.path().canonicalize()?)?
        {
            return Err(Error::Contract("foreign supervised worker handle".into()));
        }
        let child = worker
            .child
            .as_mut()
            .ok_or_else(|| Error::Contract("worker handle already transferred".into()))?;
        let status = match child.try_wait()? {
            Some(status) => status,
            None if Instant::now() >= worker.deadline => {
                child.kill()?;
                child.wait()?
            }
            None => return Ok(false),
        };
        worker.binding.exit_code = status.code();
        worker.binding.observed_terminal = true;
        self.store.connect()?.execute("UPDATE host_checkpoint_promotion_workers SET state='stopped',worker_json=? WHERE worker_token=? AND state IN ('started','stopped')",params![encode(&worker.binding)?,worker.binding.worker_token])?;
        Ok(true)
    }

    /// The supplied Child handles must cover the exact operator-approved set.
    /// The host owns every handle until it verifies termination. The fixed
    /// evaluator is launched by this seam, never claimed through JSON.
    pub fn begin_trial(
        &self,
        policy: HostPolicy,
        request: TrialRequest,
        supervised: &mut [SupervisedWorker],
    ) -> Result<SupervisedTrial> {
        self.ensure_authority()?;
        policy.validate()?;
        let run = self.check_run_policy(&policy, &request.run_id, true)?;
        let store_guard = self.store.instance_guard()?;
        check_resource(
            &self.store.connect()?,
            &policy.environment_id,
            &policy.target_id,
            &request.run_id,
            &request.trial_id,
            &self.epoch,
            &self.session_id,
        )?;
        check_resource_workers(
            &self.store.connect()?,
            &policy.environment_id,
            &policy.target_id,
            &policy.worker_ids,
        )?;
        validate_identifier(&request.trial_id, "trial")?;
        if request.evaluator_args.len() > 128
            || request
                .evaluator_args
                .iter()
                .map(String::len)
                .sum::<usize>()
                > 8192
        {
            return Err(Error::Invalid("fixed evaluator argv exceeds bound".into()));
        }
        if request.metric_floor != self.store.latest_metric_id(&request.run_id)? {
            return Err(Error::Contract(
                "evaluation metric floor must be the current host snapshot".into(),
            ));
        }
        if run.status != "running"
            || run.environment_id != policy.environment_id
            || run.protocol_version != policy.protocol_version
            || request.training_steps > policy.max_training_steps
            || request.metric_floor < 0
        {
            return Err(Error::Contract(
                "host trial run/config/budget binding differs".into(),
            ));
        }
        let ids = supervised
            .iter()
            .map(|worker| worker.binding.identity.clone())
            .collect::<HashSet<_>>();
        let process_ids = supervised
            .iter()
            .map(|worker| worker.binding.process_id)
            .collect::<HashSet<_>>();
        if ids != policy.worker_ids.iter().cloned().collect()
            || ids.len() != supervised.len()
            || process_ids.len() != supervised.len()
        {
            return Err(Error::Contract(
                "actual supervised handles do not cover all claimed workers".into(),
            ));
        }
        for worker in supervised.iter_mut() {
            if worker.binding.run_id != request.run_id
                || worker.binding.trial_id != request.trial_id
                || worker.binding.environment_config_sha256 != policy.environment_config_sha256
                || !self.poll_worker(worker)?
                || worker.binding.exit_code != Some(0)
            {
                return Err(Error::Contract(
                    "worker handle is foreign, live or failed".into(),
                ));
            }
            let row:String=self.store.connect()?.query_row("SELECT worker_json FROM host_checkpoint_promotion_workers WHERE worker_token=? AND state='stopped'",[&worker.binding.worker_token],|row|row.get(0))?;
            if serde_json::from_str::<WorkerBinding>(&row)? != worker.binding {
                return Err(Error::Contract(
                    "supervised worker startup identity changed".into(),
                ));
            }
        }
        let candidate = checked_file(&request.candidate_path)?;
        let config = checked_file(&request.environment_config_path)?;
        let program = checked_file(&request.evaluator_program)?;
        let suite = checked_file(&request.suite_path)?;
        verify_hash(&config, &policy.environment_config_sha256)?;
        verify_hash(&program, &policy.evaluator_sha256)?;
        verify_hash(&suite, &policy.evaluation_suite_sha256)?;
        let config_value: serde_json::Value = read_json(&config, "host environment config")?;
        if config_value["environment_id"] != policy.environment_id
            || config_value["protocol_version"] != policy.protocol_version
            || config_value["target_id"] != policy.target_id
        {
            return Err(Error::Contract(
                "operator config bytes do not match environment/protocol".into(),
            ));
        }
        let expected_argv: Vec<String> =
            serde_json::from_value(config_value["evaluator_argv"].clone())?;
        if expected_argv.len() != request.evaluator_args.len() + 1
            || checked_file(Path::new(&expected_argv[0]))? != program
            || expected_argv[1..] != request.evaluator_args
        {
            return Err(Error::Contract(
                "fixed evaluator argv differs from operator config".into(),
            ));
        }
        let suite_value: serde_json::Value = read_json(&suite, "fixed suite scope")?;
        if suite_value["environment_id"] != policy.environment_id
            || suite_value["protocol_version"] != policy.protocol_version
            || suite_value["target_id"] != policy.target_id
            || suite_value["applicability"] != "policy"
            || suite_value["mandatory_criteria"]
                != serde_json::to_value(&policy.mandatory_criteria)?
        {
            return Err(Error::Contract(
                "fixed suite scope/applicability/criteria differs".into(),
            ));
        }
        verify_dependencies(&config_value)?;
        verify_direct_inputs(
            &config_value,
            &request.evaluator_args,
            &[&candidate, &config, &suite],
            Some(&request.evaluation_path),
        )?;
        validate_ancestors(&request.evaluation_path)?;
        if request.evaluation_path.try_exists()? {
            return Err(Error::Contract(
                "host evaluator output must be fresh".into(),
            ));
        }
        if let Some(parent) = request.live_path.parent() {
            validate_ancestors(parent)?;
            fs::create_dir_all(parent)?;
        }
        let now = crate::store::now_ns()?;
        let remaining_ns = (policy.max_wall_seconds as i128) * 1_000_000_000
            - (now as i128 - run.started_at_ns as i128);
        if remaining_ns <= 0 {
            return Err(Error::Contract(
                "host evaluation wall budget exhausted".into(),
            ));
        }
        let token = Uuid::new_v4().simple().to_string();
        let mut ledger = Ledger {
            schema_version: LEDGER_SCHEMA.into(),
            host_session_id: self.session_id.clone(),
            binding: PromotionAuthorization {
                schema_version: AUTHORIZATION_SCHEMA.into(),
                candidate_kind: "checkpoint-policy".into(),
                evaluation_scope: "policy".into(),
                authorization_id: String::new(),
                trial_token: token.clone(),
                goal_id: policy.goal_id.clone(),
                artifact_sha256: sha256_file(&candidate)?,
                environment_id: policy.environment_id.clone(),
                protocol_version: policy.protocol_version.clone(),
                target_id: policy.target_id.clone(),
                environment_config_sha256: policy.environment_config_sha256.clone(),
                run_id: run.run_id,
                trial_id: request.trial_id,
                candidate_path: text_path(&candidate)?,
                live_path: text_path(&canonical_live_path(&request.live_path)?)?,
                incumbent_sha256: if request.live_path.try_exists()? {
                    Some(sha256_file(&request.live_path)?)
                } else {
                    None
                },
                evaluation_sha256: String::new(),
                evaluator_sha256: policy.evaluator_sha256.clone(),
                evaluation_suite_sha256: policy.evaluation_suite_sha256.clone(),
                final_measurement: FinalMeasurement {
                    name: policy.metric.clone(),
                    source: policy.source.clone(),
                    metric_id: 0,
                    value: 0.0,
                },
                reviewer_id: policy.reviewer_id.clone(),
                proposer_id: policy.proposer_id.clone(),
                worker_ids: policy.worker_ids.clone(),
                stop_receipt_sha256: String::new(),
                host_fingerprint: self.fingerprint.clone(),
                host_epoch: self.epoch.clone(),
            },
            policy,
            store_path: text_path(&self.store.path().canonicalize()?)?,
            config_path: text_path(&config)?,
            evaluator_program: text_path(&program)?,
            evaluator_args: request.evaluator_args,
            suite_path: text_path(&suite)?,
            evaluation_path: text_path(&request.evaluation_path)?,
            metric_floor: request.metric_floor,
            started_at_ns: run.started_at_ns,
            training_steps: request.training_steps,
            processes: supervised
                .iter()
                .map(|worker| ObservedProcess {
                    identity: worker.binding.identity.clone(),
                    process_id: worker.binding.process_id.expect("observed worker PID"),
                    exit_code: None,
                    observed_terminal: false,
                })
                .collect(),
            workers: supervised
                .iter()
                .map(|worker| worker.binding.clone())
                .collect(),
            measurements: Vec::new(),
            stop: None,
            review: None,
        };
        // Worker handles have been bound before evaluation starts. Training and
        // capture must stop first, so a still-running child cannot mutate the
        // artifact while the evaluator is examining it.
        // Reserve the identity and validate bounded encoding before spawning.
        // On any subsequent persistence failure reap the exact owned child.
        let mut connection = self.store.connect()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        verify_dispatch_run(
            &transaction,
            &ledger.policy,
            &ledger.binding.run_id,
            &self.fingerprint,
            &self.epoch,
        )?;
        let metric_floor: i64 = transaction.query_row(
            "SELECT COALESCE(MAX(metric_id),0) FROM metrics WHERE run_id=?",
            [&ledger.binding.run_id],
            |row| row.get(0),
        )?;
        if metric_floor != ledger.metric_floor {
            return Err(Error::Contract(
                "metric boundary changed before evaluator launch intent".into(),
            ));
        }
        check_resource(
            &transaction,
            &ledger.policy.environment_id,
            &ledger.policy.target_id,
            &ledger.binding.run_id,
            &ledger.binding.trial_id,
            &self.epoch,
            &self.session_id,
        )?;
        transaction.execute("INSERT INTO host_checkpoint_promotion_trials(trial_token,run_id,trial_id,state,ledger_json) VALUES (?,?,?,'launch-pending',?)", params![token,ledger.binding.run_id,ledger.binding.trial_id,encode(&ledger)?])?;
        transaction.commit()?;
        let mut workers = supervised
            .iter_mut()
            .map(|worker| {
                (
                    worker.binding.identity.clone(),
                    worker.child.take().expect("verified owned worker handle"),
                )
            })
            .collect::<Vec<_>>();
        let spawn = Command::new(&program)
            .args(&ledger.evaluator_args)
            .env("GLR_RUN_ID", &ledger.binding.run_id)
            .env("GLR_TRIAL_ID", &ledger.binding.trial_id)
            .env("GLR_STORE_PATH", self.store.path())
            .env("GLR_EVALUATION_PATH", &ledger.evaluation_path)
            .env(
                "GLR_CANDIDATE_CHECKPOINT_PATH",
                &ledger.binding.candidate_path,
            )
            .env("GLR_EVALUATION_SUITE_PATH", &ledger.suite_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let mut child = match spawn {
            Ok(child) => child,
            Err(error) => {
                let _=self.store.connect()?.execute("UPDATE host_checkpoint_promotion_trials SET state='launch-failed' WHERE trial_token=? AND state='launch-pending'",[&token]);
                return Err(error.into());
            }
        };
        ledger.processes.push(ObservedProcess {
            identity: ledger.policy.evaluator_id.clone(),
            process_id: child.id(),
            exit_code: None,
            observed_terminal: false,
        });
        let persist = (|| -> Result<()> {
            if self.store.connect()?.execute("UPDATE host_checkpoint_promotion_trials SET state='started',ledger_json=? WHERE trial_token=? AND state='launch-pending'", params![encode(&ledger)?,token])? != 1 { return Err(Error::Contract("host launch identity reservation changed".into())); }
            Ok(())
        })();
        if let Err(error) = persist {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        workers.push((ledger.policy.evaluator_id.clone(), child));
        Ok(SupervisedTrial {
            token,
            ledger,
            children: workers,
            _store_guard: store_guard,
            deadline: Instant::now() + Duration::from_nanos(remaining_ns as u64),
        })
    }

    /// Poll owned handles. No worker-provided stopped bool or receipt string is
    /// accepted. A live, failed or timed-out process cannot produce a receipt.
    pub fn finish_trial(&self, trial: &mut SupervisedTrial) -> Result<bool> {
        self.ensure_authority()?;
        self.check_trial(trial)?;
        if Instant::now() >= trial.deadline {
            self.cancel(trial)?;
            return Err(Error::Contract(
                "host evaluation wall budget exhausted; owned processes reaped".into(),
            ));
        }
        for (index, (_, child)) in trial.children.iter_mut().enumerate() {
            let Some(status) = child.try_wait()? else {
                return Ok(false);
            };
            trial.ledger.processes[index].exit_code = status.code();
            trial.ledger.processes[index].observed_terminal = true;
        }
        set_stop(trial)?;
        let evaluated = (|| -> Result<()> {
            if trial
                .ledger
                .processes
                .iter()
                .any(|p| p.exit_code != Some(0))
            {
                return Err(Error::Contract(
                    "supervised worker/evaluator failed after confirmed stop".into(),
                ));
            }
            let evidence = read_fixed_evaluation(Path::new(&trial.ledger.evaluation_path))?;
            evidence.validate()?;
            check_evidence(&trial.ledger.policy, &evidence)?;
            let b = &mut trial.ledger.binding;
            if evidence.goal_id != b.goal_id
                || evidence.trial_id != b.trial_id
                || evidence.evidence.iter().any(|item| item.run_id != b.run_id)
            {
                return Err(Error::Contract(
                    "fixed evaluator returned stale run/goal/trial identity".into(),
                ));
            }
            b.evaluation_sha256 = sha256_file(Path::new(&trial.ledger.evaluation_path))?;
            verify_files(&trial.ledger)?;
            let mut connection = self.store.connect()?;
            let transaction =
                connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            // The host persists its own final reading of the fixed evaluator.
            // Worker-declared source/authority strings cannot issue this row.
            let b = &trial.ledger.binding;
            for item in &evidence.evidence {
                let metadata = serde_json::json!({"source":item.source,"authority":"authoritative","host_trial_token":trial.token,"evaluation_sha256":b.evaluation_sha256,"environment_config_sha256":b.environment_config_sha256,"evaluator_sha256":b.evaluator_sha256,"evaluation_suite_sha256":b.evaluation_suite_sha256,"target_id":b.target_id,"measured":true});
                transaction.execute("INSERT INTO metrics(run_id,timestamp_ns,name,value,step_id,metadata_json,environment_config_digest) VALUES (?,?,?,?,NULL,?,?)",params![b.run_id,crate::store::now_ns()?,item.metric,item.value,encode(&metadata)?,b.environment_config_sha256])?;
                let metric_id = transaction.last_insert_rowid();
                transaction.execute("UPDATE metrics SET metadata_json=json_set(metadata_json,'$.metric_id',?) WHERE metric_id=?",params![metric_id,metric_id])?;
            }
            trial.ledger.measurements = evidence
                .evidence
                .iter()
                .map(|item| {
                    final_measurement(
                        &transaction,
                        &trial.ledger.binding.run_id,
                        trial.ledger.metric_floor,
                        &item.metric,
                        &item.source,
                        item.value,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            trial.ledger.binding.final_measurement = trial
                .ledger
                .measurements
                .iter()
                .find(|item| {
                    item.name == trial.ledger.policy.metric
                        && item.source == trial.ledger.policy.source
                })
                .ok_or_else(|| Error::Contract("fixed promotion measurement absent".into()))?
                .clone();
            let observed_at = crate::store::now_ns()?;
            for measurement in &trial.ledger.measurements {
                verify_measurement_rows(&transaction, &trial.ledger, measurement, observed_at)?;
            }
            verify_startup(&transaction, trial)?;
            if transaction.execute("UPDATE host_checkpoint_promotion_trials SET state='evaluated-stopped',ledger_json=? WHERE trial_token=? AND state='started'",params![encode(&trial.ledger)?,trial.token])?!=1 { return Err(Error::Contract("host trial is already terminal or conflicted".into())); }
            if transaction.execute("UPDATE runs SET status='succeeded',finished_at_ns=?,exit_code=0 WHERE run_id=? AND status='running'",params![crate::store::now_ns()?,trial.ledger.binding.run_id])?!=1 { return Err(Error::Contract("host run became terminal before supervised evaluation".into())); }
            release_resource(&transaction, &trial.ledger)?;
            transaction.commit()?;
            Ok(())
        })();
        if let Err(error) = evaluated {
            persist_failed_stop(&self.store.connect()?, trial)?;
            return Err(error);
        }
        Ok(true)
    }

    fn check_trial(&self, trial: &SupervisedTrial) -> Result<()> {
        if trial.ledger.binding.host_fingerprint != self.fingerprint
            || trial.ledger.binding.host_epoch != self.epoch
            || trial.ledger.host_session_id != self.session_id
            || trial.ledger.store_path != text_path(&self.store.path().canonicalize()?)?
        {
            return Err(Error::Contract(
                "supervised trial belongs to another host/store".into(),
            ));
        }
        Ok(())
    }

    /// Cancel and reap this opaque trial's owned handles. Confirmed failure is
    /// stored separately from evaluation quality. An unobserved stop remains
    /// quarantined and never becomes a review or an installation permission.
    pub fn cancel(&self, trial: &mut SupervisedTrial) -> Result<()> {
        self.ensure_authority()?;
        self.check_trial(trial)?;
        for (index, (_, child)) in trial.children.iter_mut().enumerate() {
            let status = match child.try_wait()? {
                Some(status) => status,
                None => {
                    child.kill()?;
                    child.wait()?
                }
            };
            trial.ledger.processes[index].exit_code = status.code();
            trial.ledger.processes[index].observed_terminal = true;
        }
        set_stop(trial)?;
        persist_failed_stop(&self.store.connect()?, trial)
    }

    /// Close a failed cohort before evaluation started. Only the current host
    /// session's exact opaque handles can prove launched children stopped.
    /// Persisted launch failures prove no child was created. Lost handles,
    /// unknown launch intents and evaluator ownership remain quarantined.
    pub fn cancel_pending_cohort(
        &self,
        policy: &HostPolicy,
        run_id: &str,
        trial_id: &str,
        workers: &mut [SupervisedWorker],
    ) -> Result<()> {
        self.ensure_authority()?;
        policy.validate()?;
        self.check_run_policy(policy, run_id, false)?;
        let mut connection = self.store.connect()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        check_resource(
            &transaction,
            &policy.environment_id,
            &policy.target_id,
            run_id,
            trial_id,
            &self.epoch,
            &self.session_id,
        )?;
        check_resource_workers(
            &transaction,
            &policy.environment_id,
            &policy.target_id,
            &policy.worker_ids,
        )?;
        let evaluator:Option<(String,String)>=transaction.query_row("SELECT state,ledger_json FROM host_checkpoint_promotion_trials WHERE run_id=? AND trial_id=?",params![run_id,trial_id],|row|Ok((row.get(0)?,row.get(1)?))).optional()?;
        if let Some((state, text)) = evaluator {
            let ledger: Ledger = serde_json::from_str(&text)?;
            if state != "launch-failed"
                || ledger.host_session_id != self.session_id
                || ledger.binding.host_epoch != self.epoch
                || ledger.binding.host_fingerprint != self.fingerprint
            {
                return Err(Error::Contract(
                    "pending cancellation does not own evaluator handle; reservation retained"
                        .into(),
                ));
            }
        }
        let rows = {
            let mut query=transaction.prepare("SELECT state,worker_json FROM host_checkpoint_promotion_workers WHERE run_id=? AND trial_id=? LIMIT 33")?;
            query
                .query_map(params![run_id, trial_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        if rows.len() > 32 {
            return Err(Error::Contract("worker cohort exceeds bound".into()));
        }
        let mut tokens = HashSet::new();
        for worker in workers.iter() {
            if !tokens.insert(worker.binding.worker_token.clone())
                || !policy.worker_ids.contains(&worker.binding.identity)
                || worker.binding.run_id != run_id
                || worker.binding.trial_id != trial_id
                || worker.binding.host_session_id != self.session_id
                || worker.binding.host_epoch != self.epoch
                || worker.binding.host_fingerprint != self.fingerprint
                || worker.binding.store_path != text_path(self.store.path())?
            {
                return Err(Error::Contract(
                    "foreign or duplicate pending worker handle".into(),
                ));
            }
        }
        // Validate the complete durable launch set before touching any child.
        let mut observed = Vec::new();
        for (state, text) in &rows {
            let persisted: WorkerBinding = serde_json::from_str(text)?;
            if !policy.worker_ids.contains(&persisted.identity)
                || persisted.host_session_id != self.session_id
                || persisted.host_epoch != self.epoch
                || persisted.host_fingerprint != self.fingerprint
            {
                return Err(Error::Contract(
                    "unknown pending worker owner; reservation retained".into(),
                ));
            }
            if state == "launch-failed" && persisted.process_id.is_none() {
                observed.push(serde_json::json!({"identity":persisted.identity,"worker_token":persisted.worker_token,"outcome":"not-launched"}));
                continue;
            }
            let worker = workers
                .iter()
                .find(|w| w.binding.worker_token == persisted.worker_token)
                .ok_or_else(|| {
                    Error::Contract("lost owned worker handle; reservation retained".into())
                })?;
            if persisted != worker.binding
                || !matches!(state.as_str(), "started" | "stopped")
                || (worker.child.is_none() && (state != "stopped" || !persisted.observed_terminal))
            {
                return Err(Error::Contract(
                    "unknown pending launch state; reservation retained".into(),
                ));
            }
        }
        if workers.iter().any(|w| {
            !rows.iter().any(|(_, text)| {
                serde_json::from_str::<WorkerBinding>(text)
                    .is_ok_and(|p| p.worker_token == w.binding.worker_token)
            })
        }) {
            return Err(Error::Contract("unregistered owned worker".into()));
        }
        for worker in workers.iter_mut() {
            if let Some(child) = worker.child.as_mut() {
                let status = match child.try_wait()? {
                    Some(s) => s,
                    None => {
                        child.kill()?;
                        child.wait()?
                    }
                };
                worker.binding.exit_code = status.code();
                worker.binding.observed_terminal = true;
            }
            if !worker.binding.observed_terminal {
                return Err(Error::Contract(
                    "unknown terminal worker status; reservation retained".into(),
                ));
            }
            transaction.execute("UPDATE host_checkpoint_promotion_workers SET state='stopped',worker_json=? WHERE worker_token=?",params![encode(&worker.binding)?,worker.binding.worker_token])?;
            observed.push(serde_json::json!({"identity":worker.binding.identity,"worker_token":worker.binding.worker_token,"process_id":worker.binding.process_id,"exit_code":worker.binding.exit_code}));
        }
        let receipt = serde_json::json!({"schema_version":"glr.pending-cohort-stop-receipt.v1","scope":"owned-direct-child-processes","run_id":run_id,"trial_id":trial_id,"host_epoch":self.epoch,"host_session_id":self.session_id,"supervisor_id":policy.supervisor_id,"declared_worker_ids":policy.worker_ids,"observed":observed});
        transaction.execute("INSERT INTO host_checkpoint_promotion_cohort_stops(run_id,trial_id,receipt_json) VALUES (?,?,?)",params![run_id,trial_id,encode(&receipt)?])?;
        transaction.execute("UPDATE runs SET status='failed',exit_code=1,finished_at_ns=? WHERE run_id=? AND status='running'",params![crate::store::now_ns()?,run_id])?;
        transaction.execute("DELETE FROM host_checkpoint_promotion_resources WHERE environment_id=? AND target_id=? AND host_session_id=?",params![policy.environment_id,policy.target_id,self.session_id])?;
        transaction.commit()?;
        Ok(())
    }

    /// Trusted operator review entry point. The embedding host authenticates
    /// the reviewer and supplies the decision; the pinned reviewer identity is
    /// persisted here. Calling this from an untrusted worker violates the host
    /// trust contract. Denial and approval are final and cannot be overwritten.
    pub fn review(
        &self,
        trial_token: &str,
        approve: bool,
    ) -> Result<Option<PromotionAuthorization>> {
        self.ensure_authority()?;
        let mut connection = self.store.connect()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut ledger = load(&transaction, trial_token)?;
        verify_ready(&transaction, &ledger, self.store.path())?;
        if ledger.binding.host_fingerprint != self.fingerprint
            || ledger.binding.host_epoch != self.epoch
            || ledger.review.is_some()
        {
            return Err(Error::Contract(
                "host review authority changed or decision already final".into(),
            ));
        }
        ledger.review = Some(approve);
        if approve {
            ledger.binding.authorization_id = Uuid::new_v4().simple().to_string();
        }
        if transaction.execute("UPDATE host_checkpoint_promotion_trials SET state=?,ledger_json=?,authorization_id=? WHERE trial_token=? AND state='evaluated-stopped'", params![if approve {"approved"} else {"rejected"},encode(&ledger)?,if approve {Some(&ledger.binding.authorization_id)} else {None},trial_token])? != 1 {
            return Err(Error::Contract("host review requires persisted evaluation and stop records".into()));
        }
        transaction.commit()?;
        Ok(approve.then_some(ledger.binding))
    }

    /// Install by an opaque persisted ID. No caller-supplied value/path/gates
    /// are used. Store rereads and checks the complete authorization under its
    /// write transaction before creating a durable filesystem intent.
    pub fn install(&self, authorization_id: &str) -> Result<InstallationReceipt> {
        self.ensure_authority()?;
        let connection = self.store.connect()?;
        let state: String = connection
            .query_row(
                "SELECT state FROM host_checkpoint_promotion_trials WHERE authorization_id=?",
                [authorization_id],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| Error::Contract("unknown host authorization".into()))?;
        if state != "approved" {
            return Err(Error::Contract(
                "authorization has already been consumed".into(),
            ));
        }
        let ledger = load_authorization(&connection, authorization_id)?;
        if ledger.binding.host_fingerprint != self.fingerprint
            || ledger.binding.host_epoch != self.epoch
        {
            return Err(Error::Contract(
                "authorization belongs to another host".into(),
            ));
        }
        let b = &ledger.binding;
        let (promoted, record) = self.store.install_authorized_checkpoint(
            CheckpointPromotionRequest {
                goal_id: &b.goal_id,
                metric: &b.final_measurement.name,
                mode: match ledger.policy.mode {
                    Direction::Max => PromotionMode::Max,
                    Direction::Min => PromotionMode::Min,
                },
                value: b.final_measurement.value,
                run_id: &b.run_id,
                trial_id: &b.trial_id,
                candidate: Path::new(&b.candidate_path),
                live: Path::new(&b.live_path),
            },
            authorization_id,
        )?;
        Ok(InstallationReceipt {
            promoted,
            authorization_id: authorization_id.into(),
            artifact_sha256: record.checkpoint_sha256,
            best_value: record.best_metric,
            live_path: record.checkpoint_path,
        })
    }
}

fn verify_dispatch_run(
    connection: &Connection,
    policy: &HostPolicy,
    run: &str,
    fingerprint: &str,
    epoch: &str,
) -> Result<()> {
    let (environment,protocol,status,started,config,metadata):(String,String,String,i64,Option<String>,String)=connection.query_row("SELECT environment_id,protocol_version,status,started_at_ns,environment_config_digest,metadata_json FROM runs WHERE run_id=?",[run],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?)))?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata)?;
    if environment != policy.environment_id
        || protocol != policy.protocol_version
        || status != "running"
        || config.as_deref() != Some(policy.environment_config_sha256.as_str())
        || metadata["goal_id"] != policy.goal_id
        || metadata["target_id"] != policy.target_id
        || metadata["host_policy_sha256"] != hash_json(policy)?
        || metadata["host_fingerprint"] != fingerprint
        || metadata["host_epoch"] != epoch
        || crate::store::now_ns()? as i128 - started as i128
            >= policy.max_wall_seconds as i128 * 1_000_000_000
    {
        return Err(Error::Contract(
            "dispatch run policy/status/budget changed before launch intent".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct InstallationReceipt {
    pub promoted: bool,
    pub authorization_id: String,
    pub artifact_sha256: String,
    pub best_value: f64,
    pub live_path: String,
}

pub(crate) fn initialize(connection: &mut Connection) -> Result<()> {
    let transaction =
        connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    for table in ["runs", "metrics"] {
        let mut statement = transaction.prepare(&format!("PRAGMA table_info({table})"))?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if !columns
            .iter()
            .any(|name| name == "environment_config_digest")
        {
            transaction.execute_batch(&format!(
                "ALTER TABLE {table} ADD COLUMN environment_config_digest TEXT"
            ))?;
        }
    }
    transaction.execute_batch("CREATE TABLE IF NOT EXISTS promotion_host_authority(singleton INTEGER PRIMARY KEY CHECK(singleton=1),fingerprint TEXT NOT NULL,epoch TEXT NOT NULL); CREATE TABLE IF NOT EXISTS host_checkpoint_promotion_trials(trial_token TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs(run_id),trial_id TEXT NOT NULL,state TEXT NOT NULL,ledger_json TEXT NOT NULL,authorization_id TEXT UNIQUE,UNIQUE(run_id,trial_id)); CREATE TABLE IF NOT EXISTS host_checkpoint_promotion_workers(worker_token TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs(run_id),trial_id TEXT NOT NULL,identity TEXT NOT NULL,state TEXT NOT NULL,worker_json TEXT NOT NULL,UNIQUE(run_id,trial_id,identity)); CREATE TABLE IF NOT EXISTS host_checkpoint_promotion_resources(environment_id TEXT NOT NULL,target_id TEXT NOT NULL,run_id TEXT NOT NULL,trial_id TEXT NOT NULL,host_epoch TEXT NOT NULL,host_session_id TEXT NOT NULL,worker_ids_json TEXT NOT NULL,PRIMARY KEY(environment_id,target_id));")?;
    transaction.execute_batch("CREATE TABLE IF NOT EXISTS host_checkpoint_promotion_cohort_stops(run_id TEXT NOT NULL,trial_id TEXT NOT NULL,receipt_json TEXT NOT NULL,PRIMARY KEY(run_id,trial_id));")?;
    transaction.commit()?;
    Ok(())
}
fn validate_digest(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        Err(Error::Invalid("expected lowercase SHA256 digest".into()))
    } else {
        Ok(())
    }
}
fn text_path(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::Invalid("host path must be UTF8".into()))
}
fn checked_file(path: &Path) -> Result<PathBuf> {
    validate_ancestors(path)?;
    if !path.is_file() {
        return Err(Error::Missing(path.into()));
    }
    Ok(path.canonicalize()?)
}
fn verify_hash(path: &Path, digest: &str) -> Result<()> {
    if sha256_file(&checked_file(path)?)? != digest {
        Err(Error::Contract(
            "bound artifact/config/evaluator/suite hash changed".into(),
        ))
    } else {
        Ok(())
    }
}
fn encode(value: &impl Serialize) -> Result<String> {
    let text = serde_json::to_string(value)?;
    if text.len() > 64 * 1024 {
        return Err(Error::Invalid("host ledger size exceeds bound".into()));
    }
    Ok(text)
}
fn hash_json(value: &impl Serialize) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(encode(value)?.as_bytes())))
}
fn load(connection: &Connection, token: &str) -> Result<Ledger> {
    let text: String = connection
        .query_row(
            "SELECT ledger_json FROM host_checkpoint_promotion_trials WHERE trial_token=?",
            [token],
            |row| row.get(0),
        )
        .optional()?
        .ok_or_else(|| Error::Contract("unknown host trial token".into()))?;
    if text.len() > 64 * 1024 {
        return Err(Error::Contract("host ledger exceeds bound".into()));
    }
    Ok(serde_json::from_str(&text)?)
}
pub(crate) fn load_authorization(connection: &Connection, id: &str) -> Result<Ledger> {
    let token: String=connection.query_row("SELECT trial_token FROM host_checkpoint_promotion_trials WHERE authorization_id=? AND state IN ('approved','installed')",[id],|row| row.get(0)).optional()?.ok_or_else(|| Error::Contract("checkpoint installation lacks persisted host approval".into()))?;
    let ledger = load(connection, &token)?;
    if ledger.review != Some(true) || ledger.binding.authorization_id != id {
        return Err(Error::Contract("authorization ID/decision conflict".into()));
    }
    Ok(ledger)
}
fn verify_files(ledger: &Ledger) -> Result<()> {
    for (path, digest) in [
        (
            &ledger.binding.candidate_path,
            &ledger.binding.artifact_sha256,
        ),
        (
            &ledger.config_path,
            &ledger.binding.environment_config_sha256,
        ),
        (&ledger.evaluator_program, &ledger.binding.evaluator_sha256),
        (&ledger.suite_path, &ledger.binding.evaluation_suite_sha256),
        (&ledger.evaluation_path, &ledger.binding.evaluation_sha256),
    ] {
        verify_hash(Path::new(path), digest)?;
    }
    let config: serde_json::Value = read_json(Path::new(&ledger.config_path), "host config")?;
    verify_dependencies(&config)?;
    verify_direct_inputs(
        &config,
        &ledger.evaluator_args,
        &[
            Path::new(&ledger.binding.candidate_path),
            Path::new(&ledger.config_path),
            Path::new(&ledger.suite_path),
        ],
        Some(Path::new(&ledger.evaluation_path)),
    )?;
    Ok(())
}
fn verify_ready(connection: &Connection, ledger: &Ledger, store: &Path) -> Result<()> {
    ledger.policy.validate()?;
    let b = &ledger.binding;
    let expected: (String, String) = connection.query_row(
        "SELECT fingerprint,epoch FROM promotion_host_authority WHERE singleton=1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    if ledger.schema_version != LEDGER_SCHEMA
        || b.schema_version != AUTHORIZATION_SCHEMA
        || b.candidate_kind != "checkpoint-policy"
        || b.evaluation_scope != "policy"
        || expected != (b.host_fingerprint.clone(), b.host_epoch.clone())
        || ledger.store_path != text_path(&store.canonicalize()?)?
        || b.environment_id != ledger.policy.environment_id
        || b.protocol_version != ledger.policy.protocol_version
        || b.target_id != ledger.policy.target_id
        || b.goal_id != ledger.policy.goal_id
        || b.reviewer_id != ledger.policy.reviewer_id
        || b.proposer_id != ledger.policy.proposer_id
        || b.worker_ids != ledger.policy.worker_ids
        || b.environment_config_sha256 != ledger.policy.environment_config_sha256
        || b.evaluator_sha256 != ledger.policy.evaluator_sha256
        || b.evaluation_suite_sha256 != ledger.policy.evaluation_suite_sha256
        || b.final_measurement.name != ledger.policy.metric
        || b.final_measurement.source != ledger.policy.source
        || ledger.training_steps > ledger.policy.max_training_steps
    {
        return Err(Error::Contract(
            "host ledger identity/policy/store binding changed".into(),
        ));
    }
    let stop = ledger
        .stop
        .as_ref()
        .ok_or_else(|| Error::Contract("host supervisor stop receipt is absent".into()))?;
    let ids = stop
        .processes
        .iter()
        .map(|p| p.identity.clone())
        .collect::<HashSet<_>>();
    let mut expected_ids = ledger
        .policy
        .worker_ids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    expected_ids.insert(ledger.policy.evaluator_id.clone());
    if stop.schema_version != "glr.supervisor-stop-receipt.v1"
        || stop.scope != "owned-direct-child-processes"
        || stop.trial_token != b.trial_token
        || stop.run_id != b.run_id
        || stop.trial_id != b.trial_id
        || stop.artifact_sha256 != b.artifact_sha256
        || stop.supervisor_id != ledger.policy.supervisor_id
        || stop.processes != ledger.processes
        || ids != expected_ids
        || ids.len() != stop.processes.len()
        || stop
            .processes
            .iter()
            .any(|p| !p.observed_terminal || p.exit_code != Some(0))
        || hash_json(stop)? != b.stop_receipt_sha256
    {
        return Err(Error::Contract(
            "host supervisor receipt/process coverage conflict".into(),
        ));
    }
    let worker_ids = ledger
        .workers
        .iter()
        .map(|worker| worker.identity.clone())
        .collect::<HashSet<_>>();
    if worker_ids != ledger.policy.worker_ids.iter().cloned().collect()
        || ledger.workers.len() != worker_ids.len()
    {
        return Err(Error::Contract("persisted worker coverage differs".into()));
    }
    for worker in &ledger.workers {
        let text:String=connection.query_row("SELECT worker_json FROM host_checkpoint_promotion_workers WHERE worker_token=? AND state='stopped'",[&worker.worker_token],|row|row.get(0))?;
        if serde_json::from_str::<WorkerBinding>(&text)? != *worker
            || worker.run_id != b.run_id
            || worker.trial_id != b.trial_id
            || worker.exit_code != Some(0)
            || worker.host_fingerprint != b.host_fingerprint
            || worker.host_epoch != b.host_epoch
            || worker.store_path != ledger.store_path
            || worker.environment_config_sha256 != b.environment_config_sha256
            || !ledger
                .processes
                .iter()
                .any(|p| p.identity == worker.identity && Some(p.process_id) == worker.process_id)
        {
            return Err(Error::Contract(
                "persisted worker startup/stop binding changed".into(),
            ));
        }
    }
    let (environment,protocol,status,exit,started,finished,config):(String,String,String,Option<i32>,i64,Option<i64>,Option<String>)=connection.query_row("SELECT environment_id,protocol_version,status,exit_code,started_at_ns,finished_at_ns,environment_config_digest FROM runs WHERE run_id=?",[&b.run_id],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?)))?;
    if environment != b.environment_id
        || protocol != b.protocol_version
        || status != "succeeded"
        || exit != Some(0)
        || started != ledger.started_at_ns
        || config.as_deref() != Some(b.environment_config_sha256.as_str())
        || !finished.is_some_and(|finished| {
            finished >= started
                && (finished as i128 - started as i128)
                    <= ledger.policy.max_wall_seconds as i128 * 1_000_000_000
        })
    {
        return Err(Error::Contract(
            "host worker run is not successful terminal within budget".into(),
        ));
    }
    let final_value = final_measurement(
        connection,
        &b.run_id,
        ledger.metric_floor,
        &b.final_measurement.name,
        &b.final_measurement.source,
        b.final_measurement.value,
    )?;
    let run_metadata: String = connection.query_row(
        "SELECT metadata_json FROM runs WHERE run_id=?",
        [&b.run_id],
        |row| row.get(0),
    )?;
    let run_metadata: serde_json::Value = serde_json::from_str(&run_metadata)?;
    if run_metadata["goal_id"] != b.goal_id
        || run_metadata["target_id"] != b.target_id
        || run_metadata["host_policy_sha256"] != hash_json(&ledger.policy)?
        || run_metadata["host_fingerprint"] != b.host_fingerprint
        || run_metadata["host_epoch"] != b.host_epoch
    {
        return Err(Error::Contract(
            "terminal run immutable host policy changed".into(),
        ));
    }
    if final_value != b.final_measurement {
        return Err(Error::Contract("host final measurement ID changed".into()));
    }
    let evidence = read_fixed_evaluation(Path::new(&ledger.evaluation_path))?;
    evidence.validate()?;
    check_evidence(&ledger.policy, &evidence)?;
    if evidence.goal_id != b.goal_id
        || evidence.trial_id != b.trial_id
        || evidence.evidence.iter().any(|item| item.run_id != b.run_id)
        || evidence.evidence.len() != ledger.measurements.len()
    {
        return Err(Error::Contract(
            "authorized mandatory evidence identity/coverage changed".into(),
        ));
    }
    for measurement in &ledger.measurements {
        verify_measurement_rows(
            connection,
            ledger,
            measurement,
            finished.expect("validated terminal run"),
        )?;
        if !evidence.evidence.iter().any(|item| {
            item.metric == measurement.name
                && item.source == measurement.source
                && item.value == measurement.value
        }) || final_measurement(
            connection,
            &b.run_id,
            ledger.metric_floor,
            &measurement.name,
            &measurement.source,
            measurement.value,
        )? != *measurement
        {
            return Err(Error::Contract(
                "mandatory evaluation measurement drifted".into(),
            ));
        }
        let (metadata,digest,timestamp):(String,Option<String>,i64)=connection.query_row("SELECT metadata_json,environment_config_digest,timestamp_ns FROM metrics WHERE run_id=? AND metric_id=?",params![b.run_id,measurement.metric_id],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata)?;
        if timestamp < started
            || timestamp > finished.expect("validated terminal run")
            || digest.as_deref() != Some(b.environment_config_sha256.as_str())
            || metadata["host_trial_token"] != b.trial_token
            || metadata["evaluation_sha256"] != b.evaluation_sha256
            || metadata["environment_config_sha256"] != b.environment_config_sha256
            || metadata["evaluator_sha256"] != b.evaluator_sha256
            || metadata["evaluation_suite_sha256"] != b.evaluation_suite_sha256
            || metadata["target_id"] != b.target_id
            || metadata["measured"] != true
            || metadata["metric_id"] != measurement.metric_id
        {
            return Err(Error::Contract(
                "mandatory measurement lacks fixed host provenance".into(),
            ));
        }
    }
    let metadata: String = connection.query_row(
        "SELECT metadata_json FROM metrics WHERE metric_id=? AND run_id=?",
        params![b.final_measurement.metric_id, b.run_id],
        |row| row.get(0),
    )?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata)?;
    if metadata["host_trial_token"] != b.trial_token
        || metadata["evaluation_sha256"] != b.evaluation_sha256
        || metadata["environment_config_sha256"] != b.environment_config_sha256
        || metadata["evaluator_sha256"] != b.evaluator_sha256
    {
        return Err(Error::Contract(
            "final measurement lacks host evaluation provenance".into(),
        ));
    }
    verify_files(ledger)
}
fn verify_measurement_rows(
    connection: &Connection,
    ledger: &Ledger,
    measurement: &FinalMeasurement,
    finished: i64,
) -> Result<()> {
    let b = &ledger.binding;
    let mut query=connection.prepare("SELECT metric_id,timestamp_ns,environment_config_digest,metadata_json FROM metrics WHERE run_id=? AND metric_id>? AND name=? AND json_extract(metadata_json,'$.source')=? AND json_extract(metadata_json,'$.authority')='authoritative' LIMIT 1025")?;
    let rows = query
        .query_map(
            params![
                b.run_id,
                ledger.metric_floor,
                measurement.name,
                measurement.source
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if rows.is_empty() || rows.len() > 1024 {
        return Err(Error::Contract(
            "fixed measurement row coverage is absent or exceeds bound".into(),
        ));
    }
    for (id, timestamp, digest, text) in rows {
        let metadata: serde_json::Value = serde_json::from_str(&text)?;
        if timestamp < ledger.started_at_ns
            || timestamp > finished
            || digest.as_deref() != Some(b.environment_config_sha256.as_str())
            || metadata["host_trial_token"] != b.trial_token
            || metadata["evaluation_sha256"] != b.evaluation_sha256
            || metadata["environment_config_sha256"] != b.environment_config_sha256
            || metadata["evaluator_sha256"] != b.evaluator_sha256
            || metadata["evaluation_suite_sha256"] != b.evaluation_suite_sha256
            || metadata["target_id"] != b.target_id
            || metadata["measured"] != true
            || metadata["metric_id"] != id
        {
            return Err(Error::Contract(
                "same-source measurement has foreign config/time/evaluation provenance".into(),
            ));
        }
    }
    Ok(())
}
pub(crate) fn verify_authorization(
    connection: &Connection,
    id: &str,
    store: &Path,
    request: &CheckpointPromotionRequest<'_>,
) -> Result<f64> {
    let state: String = connection.query_row(
        "SELECT state FROM host_checkpoint_promotion_trials WHERE authorization_id=?",
        [id],
        |row| row.get(0),
    )?;
    if state != "approved" {
        return Err(Error::Contract(
            "authorization has already been consumed".into(),
        ));
    }
    let ledger = load_authorization(connection, id)?;
    verify_ready(connection, &ledger, store)?;
    let b = &ledger.binding;
    if request.goal_id != b.goal_id
        || request.run_id != b.run_id
        || request.trial_id != b.trial_id
        || request.metric != b.final_measurement.name
        || request.value != b.final_measurement.value
        || checked_file(request.candidate)? != Path::new(&b.candidate_path)
        || canonical_live_path(request.live)? != Path::new(&b.live_path)
        || request.mode
            != match ledger.policy.mode {
                Direction::Max => PromotionMode::Max,
                Direction::Min => PromotionMode::Min,
            }
    {
        return Err(Error::Contract(
            "installation request differs from authorized host ledger".into(),
        ));
    }
    let actual_baseline = if request.live.try_exists()? {
        Some(sha256_file(request.live)?)
    } else {
        None
    };
    if actual_baseline != b.incumbent_sha256 {
        return Err(Error::Contract(
            "reviewed incumbent bytes changed before installation".into(),
        ));
    }
    Ok(ledger.policy.minimum_improvement)
}
pub(crate) fn verify_recovery(
    connection: &Connection,
    id: &str,
    store: &Path,
    record: &crate::store::CheckpointPromotionRecord,
    incumbent_sha256: Option<&str>,
) -> Result<String> {
    let ledger = load_authorization(connection, id)?;
    verify_ready(connection, &ledger, store)?;
    let b = &ledger.binding;
    if incumbent_sha256 != b.incumbent_sha256.as_deref()
        || record.goal_id != b.goal_id
        || record.run_id != b.run_id
        || record.trial_id != b.trial_id
        || record.metric != b.final_measurement.name
        || record.best_metric != b.final_measurement.value
        || record.checkpoint_sha256 != b.artifact_sha256
        || record.checkpoint_path != b.live_path
    {
        return Err(Error::Contract(
            "journal authorization binding differs; bytes retained".into(),
        ));
    }
    Ok(connection.query_row(
        "SELECT state FROM host_checkpoint_promotion_trials WHERE authorization_id=?",
        [id],
        |row| row.get(0),
    )?)
}
pub(crate) fn final_measurement(
    connection: &Connection,
    run: &str,
    floor: i64,
    name: &str,
    source: &str,
    value: f64,
) -> Result<FinalMeasurement> {
    let mut statement=connection.prepare("SELECT metric_id,value FROM metrics WHERE run_id=? AND metric_id>? AND name=? AND json_extract(metadata_json,'$.source')=? AND json_extract(metadata_json,'$.authority')='authoritative' ORDER BY metric_id DESC LIMIT 1025")?;
    let rows = statement
        .query_map(params![run, floor, name, source], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f64>(1)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let Some((id, actual)) = rows.first().copied() else {
        return Err(Error::Contract(
            "fixed evaluator measurement is not persisted".into(),
        ));
    };
    if rows.len() > 1024
        || !value.is_finite()
        || rows.iter().any(|(_, v)| {
            !v.is_finite()
                || if EVALUATION_ZERO_METRICS.contains(&name) {
                    *v != 0.0 || value != 0.0
                } else {
                    (*v - value).abs() > 1e-12_f64.max(1e-9 * value.abs())
                }
        })
    {
        return Err(Error::Contract(
            "conflicting same-source trial measurements; stale score selection refused".into(),
        ));
    }
    Ok(FinalMeasurement {
        name: name.into(),
        source: source.into(),
        metric_id: id,
        value: actual,
    })
}

fn verify_dependencies(config: &serde_json::Value) -> Result<()> {
    let files = config["evaluator_files"].as_array().ok_or_else(|| {
        Error::Invalid("operator config requires explicit evaluator_files manifest".into())
    })?;
    if files.len() > 128 {
        return Err(Error::Invalid("evaluator files exceed bound".into()));
    }
    let mut paths = HashSet::new();
    for file in files {
        let path = file["path"]
            .as_str()
            .ok_or_else(|| Error::Invalid("evaluator dependency path is absent".into()))?;
        let digest = file["sha256"]
            .as_str()
            .ok_or_else(|| Error::Invalid("evaluator dependency digest is absent".into()))?;
        validate_digest(digest)?;
        if !Path::new(path).is_absolute() || !paths.insert(path) {
            return Err(Error::Invalid(
                "evaluator manifest requires unique absolute paths".into(),
            ));
        }
        verify_hash(Path::new(path), digest)?;
    }
    Ok(())
}
fn verify_direct_inputs(
    config: &serde_json::Value,
    args: &[String],
    bound: &[&Path],
    output: Option<&Path>,
) -> Result<()> {
    let declared = config["evaluator_files"]
        .as_array()
        .ok_or_else(|| Error::Invalid("explicit evaluator files manifest is absent".into()))?;
    for argument in args {
        let value = if argument.starts_with('-') {
            argument
                .split_once('=')
                .map_or(argument.as_str(), |(_, value)| value)
        } else {
            argument.as_str()
        };
        let path = Path::new(value);
        if output.is_some_and(|output| {
            canonical_live_path(path).ok() == canonical_live_path(output).ok()
                && canonical_live_path(path).is_ok()
        }) {
            continue; // Only the exact host-approved evaluation output is writable.
        }
        if path.is_file() {
            if !path.is_absolute() {
                return Err(Error::Contract(
                    "fixed command file input must be absolute".into(),
                ));
            }
            let canonical = checked_file(path)?;
            if !bound
                .iter()
                .any(|p| p.canonicalize().ok().as_ref() == Some(&canonical))
                && !declared.iter().any(|item| {
                    item["path"].as_str().is_some_and(|p| {
                        Path::new(p).canonicalize().ok().as_ref() == Some(&canonical)
                    })
                })
            {
                return Err(Error::Contract(
                    "fixed evaluator/worker argv file is not hash bound".into(),
                ));
            }
        } else if path.is_absolute()
            || path.extension().is_some_and(|ext| {
                [
                    "py", "ps1", "sh", "js", "json", "csv", "yaml", "yml", "toml", "txt", "bin",
                    "pt", "pth",
                ]
                .iter()
                .any(|known| ext == *known)
            })
        {
            return Err(Error::Contract(
                "fixed command script input is absent or undeclared".into(),
            ));
        }
    }
    Ok(())
}

fn set_stop(trial: &mut SupervisedTrial) -> Result<()> {
    let b = &trial.ledger.binding;
    if trial.ledger.processes.iter().any(|p| !p.observed_terminal) {
        return Err(Error::Contract(
            "owned process stop has not been observed".into(),
        ));
    }
    let stop = StopReceipt {
        schema_version: "glr.supervisor-stop-receipt.v1".into(),
        scope: "owned-direct-child-processes".into(),
        trial_token: trial.token.clone(),
        run_id: b.run_id.clone(),
        trial_id: b.trial_id.clone(),
        artifact_sha256: b.artifact_sha256.clone(),
        supervisor_id: trial.ledger.policy.supervisor_id.clone(),
        processes: trial.ledger.processes.clone(),
    };
    trial.ledger.binding.stop_receipt_sha256 = hash_json(&stop)?;
    trial.ledger.stop = Some(stop);
    Ok(())
}
fn verify_startup(connection: &Connection, trial: &SupervisedTrial) -> Result<()> {
    let persisted = load(connection, &trial.token)?;
    let mut final_ledger = trial.ledger.clone();
    final_ledger.binding.final_measurement = persisted.binding.final_measurement.clone();
    final_ledger.binding.evaluation_sha256 = persisted.binding.evaluation_sha256.clone();
    final_ledger.binding.stop_receipt_sha256 = persisted.binding.stop_receipt_sha256.clone();
    final_ledger.stop = persisted.stop.clone();
    final_ledger.measurements = persisted.measurements.clone();
    for process in &mut final_ledger.processes {
        process.exit_code = None;
        process.observed_terminal = false;
    }
    if final_ledger != persisted {
        return Err(Error::Contract(
            "host process/policy/artifact startup record changed".into(),
        ));
    }
    Ok(())
}
fn persist_failed_stop(connection: &Connection, trial: &SupervisedTrial) -> Result<()> {
    let transaction =
        rusqlite::Transaction::new_unchecked(connection, rusqlite::TransactionBehavior::Immediate)?;
    verify_startup(&transaction, trial)?;
    if transaction.execute("UPDATE host_checkpoint_promotion_trials SET state='failed-stopped',ledger_json=? WHERE trial_token=? AND state='started'",params![encode(&trial.ledger)?,trial.token])?!=1 {return Err(Error::Contract("failed-stop record is already terminal or unknown".into()));}
    transaction.execute("UPDATE runs SET status='failed',finished_at_ns=?,exit_code=1 WHERE run_id=? AND status='running'",params![crate::store::now_ns()?,trial.ledger.binding.run_id])?;
    release_resource(&transaction, &trial.ledger)?;
    transaction.commit()?;
    Ok(())
}

fn check_evidence(policy: &HostPolicy, evidence: &GoalEvidenceBundle) -> Result<()> {
    let mut expected = policy
        .mandatory_criteria
        .iter()
        .map(|item| (&item.name, &item.source))
        .collect::<HashSet<_>>();
    expected.insert((&policy.metric, &policy.source));
    let actual = evidence
        .evidence
        .iter()
        .map(|item| (&item.metric, &item.source))
        .collect::<HashSet<_>>();
    if actual != expected
        || evidence
            .evidence
            .iter()
            .any(|item| item.authority != Authority::Authoritative)
    {
        return Err(Error::Contract(
            "fixed evaluator mandatory coverage/source/applicability failed".into(),
        ));
    }
    for criterion in &policy.mandatory_criteria {
        let item = evidence
            .evidence
            .iter()
            .find(|item| item.metric == criterion.name && item.source == criterion.source)
            .ok_or_else(|| Error::Contract("mandatory evaluation result is absent".into()))?;
        let passed = match criterion.mode {
            Direction::Max => item.value >= criterion.threshold,
            Direction::Min => item.value <= criterion.threshold,
        };
        if !passed
            || (EVALUATION_ZERO_METRICS.contains(&criterion.name.as_str()) && item.value != 0.0)
        {
            return Err(Error::Contract(
                "fixed evaluator reported a correctness failure".into(),
            ));
        }
    }
    Ok(())
}

// Journal fault fixtures seed an already reviewed host ledger to exercise the
// storage kernel. This function is absent from production builds. Separate
// host tests below exercise actual launch/stop/review issuance.
#[cfg(test)]
pub(crate) fn fixture_authorization(
    store: &Store,
    request: &CheckpointPromotionRequest<'_>,
) -> Result<String> {
    let parent = store.path().parent().expect("fixture parent");
    let config = parent.join("journal-fixture-config.json");
    let suite = parent.join("journal-fixture-suite.json");
    let program = std::env::current_exe()?;
    let run = store.get_run(request.run_id)?;
    fs::write(
        &config,
        serde_json::to_vec(
            &serde_json::json!({"environment_id":run.environment_id,"protocol_version":run.protocol_version,"target_id":"fixture-target","evaluator_files":[]}),
        )?,
    )?;
    fs::write(&suite, b"fixture-only-fixed-suite")?;
    let authority = HostAuthority::from_secret([31; 32])?;
    let fingerprint = authority.fingerprint();
    let connection = store.connect()?;
    let existing: Option<String> = connection
        .query_row(
            "SELECT epoch FROM promotion_host_authority WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let epoch = existing.unwrap_or_else(|| Uuid::new_v4().simple().to_string());
    connection.execute("INSERT OR IGNORE INTO promotion_host_authority(singleton,fingerprint,epoch) VALUES (1,?,?)",params![fingerprint,epoch])?;
    let token = Uuid::new_v4().simple().to_string();
    let authorization_id = Uuid::new_v4().simple().to_string();
    let evaluation = parent.join(format!(
        "{}.{}.fixture-evaluation.json",
        request.goal_id, request.trial_id
    ));
    let mut items = vec![
        serde_json::json!({"metric":request.metric,"value":request.value,"source":"referee","authority":"authoritative","run_id":request.run_id}),
        serde_json::json!({"metric":"violations","value":0.0,"source":"referee","authority":"authoritative","run_id":request.run_id}),
    ];
    items.extend(EVALUATION_ZERO_METRICS.iter().map(|name|serde_json::json!({"metric":name,"value":0.0,"source":"evaluation.contract","authority":"authoritative","run_id":request.run_id})));
    let evidence = serde_json::json!({"schema_version":"glr.goal-evidence.v1","goal_id":request.goal_id,"trial_id":request.trial_id,"evidence":items});
    let report = FixedEvaluationReport {
        schema_version: "glr.checkpoint-evaluation.v1".into(),
        coverage: EVALUATION_ZERO_METRICS
            .iter()
            .map(|name| ((*name).into(), "measured".into()))
            .collect(),
        evidence_bundle: serde_json::from_value(evidence)?,
    };
    fs::write(&evaluation, serde_json::to_vec(&report)?)?;
    let config_sha = sha256_file(&config)?;
    let eval_sha = sha256_file(&evaluation)?;
    let program_sha = sha256_file(&program)?;
    let floor = store.latest_metric_id(request.run_id)?;
    let mut measurements = Vec::new();
    for (name, value) in [(request.metric, request.value), ("violations", 0.0)]
        .into_iter()
        .chain(EVALUATION_ZERO_METRICS.iter().map(|name| (*name, 0.0)))
    {
        let source = if EVALUATION_ZERO_METRICS.contains(&name) {
            "evaluation.contract"
        } else {
            "referee"
        };
        let metadata = serde_json::json!({"source":source,"authority":"authoritative","host_trial_token":token,"evaluation_sha256":eval_sha,"environment_config_sha256":config_sha,"evaluator_sha256":program_sha,"evaluation_suite_sha256":sha256_file(&suite)?,"target_id":"fixture-target","measured":true});
        connection.execute(
            "INSERT INTO metrics(run_id,timestamp_ns,name,value,metadata_json,environment_config_digest) VALUES (?,?,?,?,?,?)",
            params![
                request.run_id,
                crate::store::now_ns()?,
                name,
                value,
                encode(&metadata)?, config_sha
            ],
        )?;
        let metric_id = connection.last_insert_rowid();
        connection.execute("UPDATE metrics SET metadata_json=json_set(metadata_json,'$.metric_id',?) WHERE metric_id=?",params![metric_id,metric_id])?;
        measurements.push(FinalMeasurement {
            name: name.into(),
            source: source.into(),
            metric_id: connection.last_insert_rowid(),
            value,
        });
    }
    let worker = WorkerBinding {
        worker_token: Uuid::new_v4().simple().to_string(),
        identity: "worker".into(),
        run_id: request.run_id.into(),
        trial_id: request.trial_id.into(),
        host_fingerprint: fingerprint.clone(),
        host_epoch: epoch.clone(),
        host_session_id: "fixture-session".into(),
        environment_id: run.environment_id.clone(),
        target_id: "fixture-target".into(),
        store_path: text_path(&store.path().canonicalize()?)?,
        environment_config_sha256: config_sha.clone(),
        program_sha256: program_sha.clone(),
        argv: vec![text_path(&program)?],
        process_id: Some(100),
        exit_code: Some(0),
        observed_terminal: true,
    };
    connection.execute("INSERT INTO host_checkpoint_promotion_workers(worker_token,run_id,trial_id,identity,state,worker_json) VALUES (?,?,?,?,'stopped',?)",params![worker.worker_token,request.run_id,request.trial_id,"worker",encode(&worker)?])?;
    let processes = vec![
        ObservedProcess {
            identity: "worker".into(),
            process_id: 100,
            exit_code: Some(0),
            observed_terminal: true,
        },
        ObservedProcess {
            identity: "evaluator".into(),
            process_id: 101,
            exit_code: Some(0),
            observed_terminal: true,
        },
    ];
    let digest = sha256_file(request.candidate)?;
    let stop = StopReceipt {
        schema_version: "glr.supervisor-stop-receipt.v1".into(),
        scope: "owned-direct-child-processes".into(),
        trial_token: token.clone(),
        run_id: request.run_id.into(),
        trial_id: request.trial_id.into(),
        artifact_sha256: digest.clone(),
        supervisor_id: "supervisor".into(),
        processes: processes.clone(),
    };
    let policy = HostPolicy {
        goal_id: request.goal_id.into(),
        environment_id: run.environment_id.clone(),
        protocol_version: run.protocol_version.clone(),
        target_id: "fixture-target".into(),
        environment_config_sha256: config_sha.clone(),
        evaluator_sha256: program_sha.clone(),
        evaluation_suite_sha256: sha256_file(&suite)?,
        metric: request.metric.into(),
        source: "referee".into(),
        mandatory_criteria: required_correctness()
            .into_iter()
            .chain([FixedCriterion {
                name: "violations".into(),
                source: "referee".into(),
                mode: Direction::Min,
                threshold: 0.0,
            }])
            .collect(),
        mode: match request.mode {
            PromotionMode::Max => Direction::Max,
            PromotionMode::Min => Direction::Min,
        },
        minimum_improvement: 0.0,
        proposer_id: "proposer".into(),
        worker_ids: vec!["worker".into()],
        evaluator_id: "evaluator".into(),
        supervisor_id: "supervisor".into(),
        reviewer_id: "reviewer".into(),
        max_wall_seconds: 300,
        max_training_steps: 100,
    };
    let ledger = Ledger {
        schema_version: LEDGER_SCHEMA.into(),
        host_session_id: "fixture-session".into(),
        binding: PromotionAuthorization {
            schema_version: AUTHORIZATION_SCHEMA.into(),
            candidate_kind: "checkpoint-policy".into(),
            evaluation_scope: "policy".into(),
            authorization_id: authorization_id.clone(),
            trial_token: token.clone(),
            goal_id: request.goal_id.into(),
            artifact_sha256: digest,
            environment_id: run.environment_id,
            protocol_version: run.protocol_version,
            target_id: policy.target_id.clone(),
            environment_config_sha256: config_sha,
            run_id: request.run_id.into(),
            trial_id: request.trial_id.into(),
            candidate_path: text_path(&request.candidate.canonicalize()?)?,
            live_path: text_path(&canonical_live_path(request.live)?)?,
            incumbent_sha256: if request.live.try_exists()? {
                Some(sha256_file(request.live)?)
            } else {
                None
            },
            evaluation_sha256: eval_sha,
            evaluator_sha256: program_sha,
            evaluation_suite_sha256: policy.evaluation_suite_sha256.clone(),
            final_measurement: measurements[0].clone(),
            reviewer_id: policy.reviewer_id.clone(),
            proposer_id: policy.proposer_id.clone(),
            worker_ids: policy.worker_ids.clone(),
            stop_receipt_sha256: hash_json(&stop)?,
            host_fingerprint: fingerprint,
            host_epoch: epoch,
        },
        policy,
        store_path: text_path(&store.path().canonicalize()?)?,
        config_path: text_path(&config)?,
        evaluator_program: text_path(&program)?,
        evaluator_args: Vec::new(),
        suite_path: text_path(&suite)?,
        evaluation_path: text_path(&evaluation)?,
        metric_floor: floor,
        started_at_ns: run.started_at_ns,
        training_steps: 1,
        processes,
        workers: vec![worker],
        measurements,
        stop: Some(stop),
        review: Some(true),
    };
    connection.execute("INSERT INTO host_checkpoint_promotion_trials(trial_token,run_id,trial_id,state,ledger_json,authorization_id) VALUES (?,?,?,'approved',?,?)",params![token,request.run_id,request.trial_id,encode(&ledger)?,authorization_id])?;
    connection.execute("UPDATE runs SET metadata_json=json_set(metadata_json,'$.goal_id',?,'$.target_id',?,'$.host_policy_sha256',?,'$.host_fingerprint',?,'$.host_epoch',?) WHERE run_id=?",params![ledger.binding.goal_id,ledger.binding.target_id,hash_json(&ledger.policy)?,ledger.binding.host_fingerprint,ledger.binding.host_epoch,request.run_id])?;
    connection.execute(
        "UPDATE runs SET status='succeeded',exit_code=0,finished_at_ns=?,environment_config_digest=? WHERE run_id=?",
        params![crate::store::now_ns()?, ledger.binding.environment_config_sha256, request.run_id],
    )?;
    Ok(authorization_id)
}

fn reserve_resource(
    connection: &Connection,
    policy: &HostPolicy,
    run: &str,
    trial: &str,
    epoch: &str,
    session: &str,
) -> Result<()> {
    let environment = &policy.environment_id;
    let target = &policy.target_id;
    let workers = &policy.worker_ids;
    connection.execute("INSERT OR IGNORE INTO host_checkpoint_promotion_resources(environment_id,target_id,run_id,trial_id,host_epoch,host_session_id,worker_ids_json) VALUES (?,?,?,?,?,?,?)",params![environment,target,run,trial,epoch,session,encode(&workers)?])?;
    check_resource(connection, environment, target, run, trial, epoch, session)?;
    check_resource_workers(connection, environment, target, workers)
}
fn check_resource(
    connection: &Connection,
    environment: &str,
    target: &str,
    run: &str,
    trial: &str,
    epoch: &str,
    session: &str,
) -> Result<()> {
    let row:Option<(String,String,String,String)>=connection.query_row("SELECT run_id,trial_id,host_epoch,host_session_id FROM host_checkpoint_promotion_resources WHERE environment_id=? AND target_id=?",params![environment,target],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?))).optional()?;
    if row != Some((run.into(), trial.into(), epoch.into(), session.into())) {
        return Err(Error::Contract(
            "target resource belongs to a live or unknown host session; no TTL/PID takeover".into(),
        ));
    }
    Ok(())
}
fn release_resource(connection: &Connection, ledger: &Ledger) -> Result<()> {
    let b = &ledger.binding;
    check_resource(
        connection,
        &b.environment_id,
        &b.target_id,
        &b.run_id,
        &b.trial_id,
        &b.host_epoch,
        &ledger.host_session_id,
    )?;
    if ledger.stop.is_none() {
        return Err(Error::Contract(
            "resource release lacks observed owned-process stop".into(),
        ));
    }
    connection.execute("DELETE FROM host_checkpoint_promotion_resources WHERE environment_id=? AND target_id=? AND run_id=? AND trial_id=? AND host_epoch=? AND host_session_id=?",params![b.environment_id,b.target_id,b.run_id,b.trial_id,b.host_epoch,ledger.host_session_id])?;
    Ok(())
}

fn check_resource_workers(
    connection: &Connection,
    environment: &str,
    target: &str,
    workers: &[String],
) -> Result<()> {
    let text:String=connection.query_row("SELECT worker_ids_json FROM host_checkpoint_promotion_resources WHERE environment_id=? AND target_id=?",params![environment,target],|row|row.get(0))?;
    if serde_json::from_str::<Vec<String>>(&text)? != workers {
        return Err(Error::Contract(
            "resource declared worker set changed".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const EVALUATOR: &str = "promotion_host::tests::fixed_evaluator_fixture";
    const WORKER: &str = "promotion_host::tests::worker_fixture";

    #[test]
    #[ignore]
    fn worker_fixture() {}
    #[test]
    #[ignore]
    fn failed_worker_fixture() {
        std::process::exit(2);
    }
    #[test]
    #[ignore]
    fn fixed_evaluator_fixture() {
        let candidate =
            fs::read_to_string(std::env::var_os("GLR_CANDIDATE_CHECKPOINT_PATH").unwrap()).unwrap();
        if candidate == "fail" {
            std::process::exit(4);
        }
        if candidate == "slow" {
            std::thread::sleep(Duration::from_millis(500));
        }
        let run = std::env::var("GLR_RUN_ID").unwrap();
        let trial = std::env::var("GLR_TRIAL_ID").unwrap();
        let score = candidate
            .strip_prefix("score:")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(4.0);
        if candidate == "conflict" {
            let connection = Connection::open(std::env::var_os("GLR_STORE_PATH").unwrap()).unwrap();
            for value in [100.0, 0.0] {
                connection.execute("INSERT INTO metrics(run_id,timestamp_ns,name,value,metadata_json) VALUES (?,1,'score',?,?)",params![run,value,"{\"source\":\"fixed\",\"authority\":\"authoritative\"}"]).unwrap();
            }
        }
        if candidate == "tiny-counter" {
            Connection::open(std::env::var_os("GLR_STORE_PATH").unwrap()).unwrap().execute("INSERT INTO metrics(run_id,timestamp_ns,name,value,metadata_json) VALUES (?,1,?,0.0000000000001,?)",params![run,EVALUATION_ZERO_METRICS[0],"{\"source\":\"evaluation.contract\",\"authority\":\"authoritative\"}"]).unwrap();
        }
        if candidate == "stale-same-zero" || candidate == "stale-same-score" {
            let (name, source, value) = if candidate == "stale-same-zero" {
                (EVALUATION_ZERO_METRICS[0], "evaluation.contract", 0.0)
            } else {
                ("score", "fixed", 4.0)
            };
            Connection::open(std::env::var_os("GLR_STORE_PATH").unwrap()).unwrap().execute("INSERT INTO metrics(run_id,timestamp_ns,name,value,metadata_json) VALUES (?,1,?,?,?)",params![run,name,value,json!({"source":source,"authority":"authoritative"}).to_string()]).unwrap();
        }
        let mut evidence = vec![
            json!({"metric":"score","value":score,"source":if candidate=="wrong-source" {"trainer"} else {"fixed"},"authority":"authoritative","run_id":if candidate=="stale-run" {"other-run"} else {&run}}),
        ];
        if candidate != "missing" {
            evidence.push(json!({"metric":"violations","value":if candidate=="violation" {1.0} else {0.0},"source":"fixed","authority":"authoritative","run_id":run}));
        }
        for (index, name) in EVALUATION_ZERO_METRICS.iter().enumerate() {
            if candidate == "missing-seven" && index == 6 {
                continue;
            }
            evidence.push(json!({"metric":name,"value":if candidate=="negative-seven" && index==0 {-1.0}else if candidate=="violation-seven" && index==0 {1.0}else{0.0},"source":if candidate=="source-seven" && index==0 {"trainer"}else{"evaluation.contract"},"authority":"authoritative","run_id":run}));
        }
        if candidate == "unknown" {
            evidence.push(json!({"metric":"unexpected","value":0.0,"source":"fixed","authority":"authoritative","run_id":run}));
        }
        let mut coverage = EVALUATION_ZERO_METRICS
            .iter()
            .map(|name| ((*name).to_string(), "measured".to_string()))
            .collect::<std::collections::BTreeMap<_, _>>();
        if candidate == "na-coverage" {
            coverage.insert(EVALUATION_ZERO_METRICS[0].into(), "not-applicable".into());
        }
        if candidate == "unknown-coverage" {
            coverage.insert(EVALUATION_ZERO_METRICS[0].into(), "unknown".into());
        }
        if candidate == "missing-coverage" {
            coverage.remove(EVALUATION_ZERO_METRICS[0]);
        }
        let report = json!({"schema_version":"glr.checkpoint-evaluation.v1","coverage":coverage,"evidence_bundle":{"schema_version":"glr.goal-evidence.v1","goal_id":"goal.host","trial_id":trial,"evidence":evidence}});
        fs::write(
            std::env::var_os("GLR_EVALUATION_PATH").unwrap(),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
    }

    struct Fixture {
        temp: tempfile::TempDir,
        host: PromotionHost,
        policy: HostPolicy,
        program: PathBuf,
        args: Vec<String>,
        config: PathBuf,
        suite: PathBuf,
        live: PathBuf,
        dependency: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let temp = tempfile::tempdir().unwrap();
            let program = temp.path().join(if cfg!(windows) {
                "fixed-evaluator.exe"
            } else {
                "fixed-evaluator"
            });
            fs::copy(std::env::current_exe().unwrap(), &program).unwrap();
            let args = vec![
                "--ignored".into(),
                "--exact".into(),
                EVALUATOR.into(),
                "--nocapture".into(),
            ];
            let config = temp.path().join("environment.json");
            let suite = temp.path().join("suite.json");
            let dependency = temp.path().join("evaluator-input.txt");
            fs::write(&dependency, b"fixed dependency").unwrap();
            let executable = std::env::current_exe().unwrap();
            fs::write(&config,serde_json::to_vec(&json!({"environment_id":"fixture.environment-v1","protocol_version":"1.0","target_id":"fixture-target","evaluator_argv":std::iter::once(text_path(&program).unwrap()).chain(args.iter().cloned()).collect::<Vec<_>>(),"evaluator_files":[{"path":dependency,"sha256":sha256_file(&dependency).unwrap()}],"worker_commands":{"worker":{"argv":[executable,"--ignored","--exact",WORKER],"sha256":sha256_file(&executable).unwrap()}}})).unwrap()).unwrap();
            let criteria = required_correctness()
                .into_iter()
                .chain([FixedCriterion {
                    name: "violations".into(),
                    source: "fixed".into(),
                    mode: Direction::Min,
                    threshold: 0.0,
                }])
                .collect::<Vec<_>>();
            fs::write(&suite,serde_json::to_vec(&json!({"environment_id":"fixture.environment-v1","protocol_version":"1.0","target_id":"fixture-target","applicability":"policy","mandatory_criteria":criteria})).unwrap()).unwrap();
            let policy = HostPolicy {
                goal_id: "goal.host".into(),
                environment_id: "fixture.environment-v1".into(),
                protocol_version: "1.0".into(),
                target_id: "fixture-target".into(),
                environment_config_sha256: sha256_file(&config).unwrap(),
                evaluator_sha256: sha256_file(&program).unwrap(),
                evaluation_suite_sha256: sha256_file(&suite).unwrap(),
                metric: "score".into(),
                source: "fixed".into(),
                mandatory_criteria: criteria,
                mode: Direction::Max,
                minimum_improvement: 0.5,
                proposer_id: "proposer".into(),
                worker_ids: vec!["worker".into()],
                evaluator_id: "evaluator".into(),
                supervisor_id: "supervisor".into(),
                reviewer_id: "reviewer".into(),
                max_wall_seconds: 60,
                max_training_steps: 100,
            };
            let host = PromotionHost::provision_empty(
                temp.path().join("runs.sqlite3"),
                HostAuthority::from_secret([7; 32]).unwrap(),
            )
            .unwrap();
            let live = temp.path().join("best.checkpoint");
            fs::write(&live, b"baseline").unwrap();
            Self {
                temp,
                host,
                policy,
                program,
                args,
                config,
                suite,
                live,
                dependency,
            }
        }
        fn request(&self, run: &str, trial: &str, contents: &str) -> TrialRequest {
            let candidate = self.temp.path().join(format!("{trial}.candidate"));
            fs::write(&candidate, contents).unwrap();
            TrialRequest {
                run_id: run.into(),
                trial_id: trial.into(),
                candidate_path: candidate,
                live_path: self.live.clone(),
                environment_config_path: self.config.clone(),
                evaluator_program: self.program.clone(),
                evaluator_args: self.args.clone(),
                suite_path: self.suite.clone(),
                evaluation_path: self.temp.path().join(format!("{trial}.evaluation.json")),
                metric_floor: self.host.store.latest_metric_id(run).unwrap(),
                training_steps: 10,
            }
        }
        fn worker(&self, run: &str, trial: &str) -> SupervisedWorker {
            let mut worker = self
                .host
                .spawn_worker(&self.policy, run, trial, "worker", &self.config)
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while !self.host.poll_worker(&mut worker).unwrap() {
                assert!(Instant::now() < deadline);
                std::thread::sleep(Duration::from_millis(2));
            }
            worker
        }
        fn begin(&self, trial: &str, contents: &str) -> SupervisedTrial {
            let run = self.host.create_run(&self.policy).unwrap();
            let worker = self.worker(&run, trial);
            self.host
                .begin_trial(
                    self.policy.clone(),
                    self.request(&run, trial, contents),
                    &mut [worker],
                )
                .unwrap()
        }
        fn finish(&self, trial: &mut SupervisedTrial) -> Result<()> {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                match self.host.finish_trial(trial) {
                    Ok(true) => return Ok(()),
                    Ok(false) => {
                        assert!(Instant::now() < deadline);
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        fn reviewed(&self, trial: &str, contents: &str) -> PromotionAuthorization {
            let mut running = self.begin(trial, contents);
            self.finish(&mut running).unwrap();
            self.host
                .review(running.trial_token(), true)
                .unwrap()
                .unwrap()
        }
        fn assert_no_install(&self) {
            assert_eq!(fs::read(&self.live).unwrap(), b"baseline");
            let count: i64 = self
                .host
                .store
                .connect()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM checkpoint_promotions", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn promotion_host_requires_complete_stop_review_and_single_consumption() {
        let fixture = Fixture::new();
        let mut trial = fixture.begin("trial-positive", "score:4");
        fixture
            .host
            .store
            .append_event(
                &trial.ledger.binding.run_id,
                "worker.fake-stop",
                json!({"stopped":true,"reviewed":true}),
            )
            .unwrap();
        assert!(fixture.host.review(trial.trial_token(), true).is_err());
        assert!(fixture.host.install("worker-supplied-review").is_err());
        fixture.assert_no_install();
        fixture.finish(&mut trial).unwrap();
        fixture.assert_no_install();
        let authorization = fixture
            .host
            .review(trial.trial_token(), true)
            .unwrap()
            .unwrap();
        assert_eq!(authorization.schema_version, AUTHORIZATION_SCHEMA);
        assert!(fixture.host.review(trial.trial_token(), false).is_err());
        assert!(
            fixture
                .host
                .install(&authorization.authorization_id)
                .unwrap()
                .promoted
        );
        assert_eq!(fs::read(&fixture.live).unwrap(), b"score:4");
        assert!(
            fixture
                .host
                .install(&authorization.authorization_id)
                .is_err()
        );
        let reopened = PromotionHost::open(
            fixture.host.store.path().into(),
            HostAuthority::from_secret([7; 32]).unwrap(),
        )
        .unwrap();
        assert!(reopened.install(&authorization.authorization_id).is_err());
        assert!(
            PromotionHost::open(
                fixture.host.store.path().into(),
                HostAuthority::from_secret([8; 32]).unwrap()
            )
            .is_err()
        );
        let connection = fixture.host.store.connect().unwrap();
        let schema: i64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(schema, 1);
        let digest: String = connection
            .query_row(
                "SELECT environment_config_digest FROM runs WHERE run_id=?",
                [authorization.run_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(digest, fixture.policy.environment_config_sha256);
    }

    #[test]
    fn promotion_host_preserves_strict_minimum_improvement_and_terminal_denial() {
        let fixture = Fixture::new();
        let first = fixture.reviewed("first", "score:4");
        fixture.host.install(&first.authorization_id).unwrap();
        let equal = fixture.reviewed("equal", "score:4.5");
        assert!(
            !fixture
                .host
                .install(&equal.authorization_id)
                .unwrap()
                .promoted
        );
        assert!(fixture.host.install(&equal.authorization_id).is_err());
        assert_eq!(fs::read(&fixture.live).unwrap(), b"score:4");
        let better = fixture.reviewed("better", "score:4.51");
        assert!(
            fixture
                .host
                .install(&better.authorization_id)
                .unwrap()
                .promoted
        );
        assert_eq!(fs::read(&fixture.live).unwrap(), b"score:4.51");
        let mut trial = fixture.begin("denied", "score:8");
        fixture.finish(&mut trial).unwrap();
        assert!(
            fixture
                .host
                .review(trial.trial_token(), false)
                .unwrap()
                .is_none()
        );
        assert!(fixture.host.review(trial.trial_token(), true).is_err());
    }

    #[test]
    fn promotion_host_rejects_conflicting_measurements_and_failed_correctness() {
        for scenario in [
            "conflict",
            "tiny-counter",
            "stale-same-zero",
            "stale-same-score",
            "violation",
            "unknown",
            "missing",
            "missing-seven",
            "na-coverage",
            "unknown-coverage",
            "missing-coverage",
            "negative-seven",
            "violation-seven",
            "source-seven",
            "wrong-source",
            "stale-run",
            "fail",
        ] {
            let fixture = Fixture::new();
            let mut trial = fixture.begin("negative", scenario);
            assert!(fixture.finish(&mut trial).is_err(), "{scenario}");
            assert!(
                fixture.host.review(trial.trial_token(), true).is_err(),
                "{scenario}"
            );
            fixture.assert_no_install();
            let state: String = fixture
                .host
                .store
                .connect()
                .unwrap()
                .query_row(
                    "SELECT state FROM host_checkpoint_promotion_trials WHERE trial_token=?",
                    [trial.trial_token()],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(state, "failed-stopped", "{scenario}");
            assert!(
                load(&fixture.host.store.connect().unwrap(), trial.trial_token())
                    .unwrap()
                    .stop
                    .is_some()
            );
        }
    }

    #[test]
    fn promotion_host_rejects_every_postreview_binding_drift() {
        for field in [
            "candidate",
            "baseline",
            "config",
            "suite",
            "evaluator",
            "dependency",
            "evaluation",
            "metric",
            "run-config",
            "authority",
        ] {
            let fixture = Fixture::new();
            let authorization = fixture.reviewed("drift", "score:4");
            match field {
                "candidate" => {
                    fs::write(&authorization.candidate_path, b"changed candidate").unwrap()
                }
                "baseline" => fs::write(&fixture.live, b"changed incumbent").unwrap(),
                "config" => fs::write(&fixture.config, b"{}").unwrap(),
                "suite" => fs::write(&fixture.suite, b"{}").unwrap(),
                "evaluator" => {
                    use std::io::Write;
                    fs::OpenOptions::new()
                        .append(true)
                        .open(&fixture.program)
                        .unwrap()
                        .write_all(b"changed executable")
                        .unwrap();
                }
                "dependency" => fs::write(&fixture.dependency, b"changed dependency").unwrap(),
                "evaluation" => {
                    fs::write(fixture.temp.path().join("drift.evaluation.json"), b"{}").unwrap()
                }
                "metric" => {
                    fixture
                        .host
                        .store
                        .connect()
                        .unwrap()
                        .execute(
                            "UPDATE metrics SET value=100 WHERE metric_id=?",
                            [authorization.final_measurement.metric_id],
                        )
                        .unwrap();
                }
                "run-config" => {
                    fixture
                        .host
                        .store
                        .connect()
                        .unwrap()
                        .execute(
                            "UPDATE runs SET environment_config_digest=NULL WHERE run_id=?",
                            [&authorization.run_id],
                        )
                        .unwrap();
                }
                "authority" => {
                    fixture
                        .host
                        .store
                        .connect()
                        .unwrap()
                        .execute(
                            "UPDATE promotion_host_authority SET epoch='different-instance'",
                            [],
                        )
                        .unwrap();
                }
                _ => unreachable!(),
            }
            assert!(
                fixture
                    .host
                    .install(&authorization.authorization_id)
                    .is_err(),
                "{field}"
            );
            let count: i64 = fixture
                .host
                .store
                .connect()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM checkpoint_promotions", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 0, "{field}");
            assert_eq!(
                fs::read(&fixture.live).unwrap(),
                if field == "baseline" {
                    b"changed incumbent".as_slice()
                } else {
                    b"baseline".as_slice()
                }
            );
            assert!(
                !fixture
                    .temp
                    .path()
                    .join(".best.checkpoint.glr-owner.json")
                    .exists(),
                "{field}"
            );
        }
    }

    #[test]
    fn promotion_host_cannot_claim_foreign_handles_missing_workers_or_role_aliases() {
        let fixture = Fixture::new();
        let foreign = Fixture::new();
        let run = fixture.host.create_run(&fixture.policy).unwrap();
        let foreign_run = foreign.host.create_run(&foreign.policy).unwrap();
        let worker = foreign.worker(&foreign_run, "foreign");
        assert!(
            fixture
                .host
                .begin_trial(
                    fixture.policy.clone(),
                    fixture.request(&run, "foreign", "score:4"),
                    &mut [worker]
                )
                .is_err()
        );
        let worker = fixture.worker(&run, "missing-worker");
        let mut policy = fixture.policy.clone();
        policy.worker_ids.push("capture".into());
        assert!(
            fixture
                .host
                .begin_trial(
                    policy,
                    fixture.request(&run, "missing-worker", "score:4"),
                    &mut [worker]
                )
                .is_err()
        );
        let mut policy = fixture.policy.clone();
        policy.supervisor_id = policy.proposer_id.clone();
        assert!(policy.validate().is_err());
        policy = fixture.policy.clone();
        policy.reviewer_id = policy.worker_ids[0].clone();
        assert!(policy.validate().is_err());
        fixture.assert_no_install();
    }

    #[test]
    fn promotion_host_rejects_target_diagnostic_suite_and_unbound_script_inputs() {
        let fixture = Fixture::new();
        let mut policy = fixture.policy.clone();
        policy.target_id = "other-target".into();
        let run = fixture.host.create_run(&policy).unwrap();
        assert!(
            fixture
                .host
                .spawn_worker(&policy, &run, "wrong-target", "worker", &fixture.config)
                .is_err()
        );
        for field in ["target_id", "applicability"] {
            let fixture = Fixture::new();
            let run = fixture.host.create_run(&fixture.policy).unwrap();
            let worker = fixture.worker(&run, "scope");
            let mut suite: serde_json::Value = read_json(&fixture.suite, "suite").unwrap();
            suite[field] = json!(if field == "target_id" {
                "other-target"
            } else {
                "diagnostic"
            });
            fs::write(&fixture.suite, serde_json::to_vec(&suite).unwrap()).unwrap();
            let mut policy = fixture.policy.clone();
            policy.evaluation_suite_sha256 = sha256_file(&fixture.suite).unwrap();
            assert!(
                fixture
                    .host
                    .begin_trial(
                        policy,
                        fixture.request(&run, "scope", "score:4"),
                        &mut [worker]
                    )
                    .is_err()
            );
            fixture.assert_no_install();
        }
        let fixture = Fixture::new();
        let script = fixture.temp.path().join("unbound-evaluator.py");
        fs::write(&script, b"print('unbound')").unwrap();
        let mut config: serde_json::Value = read_json(&fixture.config, "config").unwrap();
        let mut args = fixture.args.clone();
        args.push(text_path(&script).unwrap());
        config["evaluator_argv"] = json!(
            std::iter::once(text_path(&fixture.program).unwrap())
                .chain(args.iter().cloned())
                .collect::<Vec<_>>()
        );
        fs::write(&fixture.config, serde_json::to_vec(&config).unwrap()).unwrap();
        let mut policy = fixture.policy.clone();
        policy.environment_config_sha256 = sha256_file(&fixture.config).unwrap();
        let run = fixture.host.create_run(&policy).unwrap();
        let mut worker = fixture
            .host
            .spawn_worker(&policy, &run, "script", "worker", &fixture.config)
            .unwrap();
        while !fixture.host.poll_worker(&mut worker).unwrap() {
            std::thread::sleep(Duration::from_millis(2));
        }
        let mut request = fixture.request(&run, "script", "score:4");
        request.evaluator_args = args;
        assert!(
            fixture
                .host
                .begin_trial(policy, request, &mut [worker])
                .is_err()
        );
        fixture.assert_no_install();
    }

    #[test]
    fn promotion_host_timeout_cancel_and_drop_reap_owned_evaluator_and_record_stop() {
        for mode in ["timeout", "cancel", "drop"] {
            let fixture = Fixture::new();
            let mut trial = fixture.begin("stop", "slow");
            let token = trial.trial_token().to_owned();
            if mode == "timeout" {
                trial.deadline = Instant::now();
                assert!(fixture.host.finish_trial(&mut trial).is_err());
            } else if mode == "cancel" {
                fixture.host.cancel(&mut trial).unwrap();
            }
            drop(trial);
            let ledger = load(&fixture.host.store.connect().unwrap(), &token).unwrap();
            assert!(ledger.stop.is_some(), "{mode}");
            assert!(fixture.host.review(&token, true).is_err());
            fixture.assert_no_install();
            let state: String = fixture
                .host
                .store
                .connect()
                .unwrap()
                .query_row(
                    "SELECT state FROM host_checkpoint_promotion_trials WHERE trial_token=?",
                    [token],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(state, "failed-stopped");
        }
    }

    #[test]
    fn promotion_host_failed_worker_never_launches_evaluator() {
        let fixture = Fixture::new();
        let mut config: serde_json::Value = read_json(&fixture.config, "config").unwrap();
        config["worker_commands"]["worker"]["argv"][3] =
            json!("promotion_host::tests::failed_worker_fixture");
        fs::write(&fixture.config, serde_json::to_vec(&config).unwrap()).unwrap();
        let mut policy = fixture.policy.clone();
        policy.environment_config_sha256 = sha256_file(&fixture.config).unwrap();
        let run = fixture.host.create_run(&policy).unwrap();
        let mut worker = fixture
            .host
            .spawn_worker(&policy, &run, "failed", "worker", &fixture.config)
            .unwrap();
        while !fixture.host.poll_worker(&mut worker).unwrap() {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(worker.binding.exit_code, Some(2));
        assert!(
            fixture
                .host
                .begin_trial(
                    policy,
                    fixture.request(&run, "failed", "score:4"),
                    &mut [worker]
                )
                .is_err()
        );
        assert!(!fixture.temp.path().join("failed.evaluation.json").exists());
        fixture.assert_no_install();
    }

    #[test]
    fn promotion_host_refuses_legacy_bootstrap_foreign_store_and_recreated_database_owner() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("legacy.sqlite3")).unwrap();
        store
            .create_run("legacy.environment", "1.0", "goal", json!({}))
            .unwrap();
        assert!(
            PromotionHost::provision_empty(
                store.path().into(),
                HostAuthority::from_secret([7; 32]).unwrap()
            )
            .is_err()
        );
        let fixture = Fixture::new();
        let authorization = fixture.reviewed("foreign-store", "score:4");
        let connection = fixture.host.store.connect().unwrap();
        connection
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
        drop(connection);
        let copied = fixture.temp.path().join("copy.sqlite3");
        fs::copy(fixture.host.store.path(), &copied).unwrap();
        let foreign =
            PromotionHost::open(copied, HostAuthority::from_secret([7; 32]).unwrap()).unwrap();
        assert!(foreign.install(&authorization.authorization_id).is_err());
        fixture.assert_no_install();
        fixture
            .host
            .install(&authorization.authorization_id)
            .unwrap();
        let connection = fixture.host.store.connect().unwrap();
        connection
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
        drop(connection);
        let dbpath = fixture.host.store.path().to_path_buf();
        let mut request = fixture.request(&authorization.run_id, "new-instance", "score:8");
        #[cfg(windows)]
        assert!(
            fs::rename(
                &dbpath,
                fixture.temp.path().join("denied-replacement.sqlite3")
            )
            .is_err()
        );
        drop(fixture.host);
        fs::rename(
            &dbpath,
            fixture.temp.path().join("retained-original.sqlite3"),
        )
        .unwrap();
        let replacement =
            PromotionHost::provision_empty(dbpath, HostAuthority::from_secret([7; 32]).unwrap())
                .unwrap();
        let run = replacement.create_run(&fixture.policy).unwrap();
        let mut worker = replacement
            .spawn_worker(
                &fixture.policy,
                &run,
                "new-instance",
                "worker",
                &fixture.config,
            )
            .unwrap();
        while !replacement.poll_worker(&mut worker).unwrap() {
            std::thread::sleep(Duration::from_millis(2));
        }
        request.run_id = run.clone();
        request.metric_floor = 0;
        let mut trial = replacement
            .begin_trial(fixture.policy.clone(), request, &mut [worker])
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !replacement.finish_trial(&mut trial).unwrap() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        let authorization = replacement
            .review(trial.trial_token(), true)
            .unwrap()
            .unwrap();
        assert!(
            replacement
                .install(&authorization.authorization_id)
                .is_err()
        );
        assert_eq!(fs::read(&fixture.live).unwrap(), b"score:4");
    }

    #[test]
    #[ignore]
    fn long_worker_fixture() {
        std::thread::sleep(Duration::from_secs(5));
    }

    #[test]
    fn promotion_host_pretrial_cancellation_releases_only_complete_owned_cohort() {
        let fixture = Fixture::new();
        let run = fixture.host.create_run(&fixture.policy).unwrap();
        let mut workers = vec![fixture.worker(&run, "invalid-suite")];
        let mut request = fixture.request(&run, "invalid-suite", "score:4");
        request.suite_path = fixture.temp.path().join("missing-suite.json");
        assert!(
            fixture
                .host
                .begin_trial(fixture.policy.clone(), request, &mut workers)
                .is_err()
        );
        let reopened = PromotionHost::open(
            fixture.host.store.path().into(),
            HostAuthority::from_secret([7; 32]).unwrap(),
        )
        .unwrap();
        assert!(
            reopened
                .cancel_pending_cohort(&fixture.policy, &run, "invalid-suite", &mut workers)
                .is_err()
        );
        assert!(
            fixture
                .host
                .cancel_pending_cohort(&fixture.policy, &run, "invalid-suite", &mut [])
                .is_err()
        );
        fixture
            .host
            .cancel_pending_cohort(&fixture.policy, &run, "invalid-suite", &mut workers)
            .unwrap();
        let receipt: String = fixture
            .host
            .store
            .connect()
            .unwrap()
            .query_row(
                "SELECT receipt_json FROM host_checkpoint_promotion_cohort_stops WHERE run_id=?",
                [&run],
                |r| r.get(0),
            )
            .unwrap();
        assert!(receipt.contains("owned-direct-child-processes"));
        let authorization = fixture.reviewed("after-cancel", "score:4");
        assert!(
            fixture
                .host
                .install(&authorization.authorization_id)
                .unwrap()
                .promoted
        );
    }

    #[test]
    fn promotion_host_unknown_launch_blocks_restart_and_new_ids_without_spawning() {
        let fixture = Fixture::new();
        let run = fixture.host.create_run(&fixture.policy).unwrap();
        fixture.host.store.connect().unwrap().execute_batch("CREATE TRIGGER reject_worker_registration BEFORE UPDATE ON host_checkpoint_promotion_workers WHEN NEW.state='started' BEGIN SELECT RAISE(ABORT,'fixture registration failure'); END;").unwrap();
        assert!(
            fixture
                .host
                .spawn_worker(
                    &fixture.policy,
                    &run,
                    "unknown-launch",
                    "worker",
                    &fixture.config
                )
                .is_err()
        );
        let reopened = PromotionHost::open(
            fixture.host.store.path().into(),
            HostAuthority::from_secret([7; 32]).unwrap(),
        )
        .unwrap();
        let newrun = reopened.create_run(&fixture.policy).unwrap();
        assert!(
            reopened
                .spawn_worker(
                    &fixture.policy,
                    &newrun,
                    "new-ids",
                    "worker",
                    &fixture.config
                )
                .is_err()
        );
        assert!(
            reopened
                .cancel_pending_cohort(&fixture.policy, &run, "unknown-launch", &mut [])
                .is_err()
        );
        assert!(
            fixture
                .host
                .cancel_pending_cohort(&fixture.policy, &run, "unknown-launch", &mut [])
                .is_err()
        );
        let count: i64 = fixture
            .host
            .store
            .connect()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM host_checkpoint_promotion_workers",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        fixture.assert_no_install();
    }

    #[test]
    fn promotion_host_run_policy_and_expired_budget_fail_before_any_launch_intent() {
        let fixture = Fixture::new();
        let run = fixture.host.create_run(&fixture.policy).unwrap();
        for field in ["goal", "target", "config", "source", "reviewer"] {
            let mut foreign = fixture.policy.clone();
            match field {
                "goal" => foreign.goal_id = "other-goal".into(),
                "target" => foreign.target_id = "other-target".into(),
                "config" => foreign.environment_config_sha256 = "a".repeat(64),
                "source" => foreign.source = "other-source".into(),
                _ => foreign.reviewer_id = "other-reviewer".into(),
            };
            assert!(
                fixture
                    .host
                    .spawn_worker(&foreign, &run, "foreign-policy", "worker", &fixture.config)
                    .is_err(),
                "{field}"
            );
        }
        fixture
            .host
            .store
            .connect()
            .unwrap()
            .execute("UPDATE runs SET started_at_ns=1 WHERE run_id=?", [&run])
            .unwrap();
        assert!(
            fixture
                .host
                .spawn_worker(&fixture.policy, &run, "expired", "worker", &fixture.config)
                .is_err()
        );
        let connection = fixture.host.store.connect().unwrap();
        for table in [
            "host_checkpoint_promotion_workers",
            "host_checkpoint_promotion_resources",
            "host_checkpoint_promotion_trials",
        ] {
            let count: i64 = connection
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn promotion_host_owned_worker_wall_timeout_can_be_cancelled_and_next_run_started() {
        let fixture = Fixture::new();
        let mut config: serde_json::Value = read_json(&fixture.config, "fixture").unwrap();
        config["worker_commands"]["worker"]["argv"][3] =
            json!("promotion_host::tests::long_worker_fixture");
        fs::write(&fixture.config, serde_json::to_vec(&config).unwrap()).unwrap();
        let mut policy = fixture.policy.clone();
        policy.max_wall_seconds = 1;
        policy.environment_config_sha256 = sha256_file(&fixture.config).unwrap();
        let run = fixture.host.create_run(&policy).unwrap();
        let mut workers = vec![
            fixture
                .host
                .spawn_worker(&policy, &run, "long-worker", "worker", &fixture.config)
                .unwrap(),
        ];
        let deadline = Instant::now() + Duration::from_secs(3);
        while !fixture.host.poll_worker(&mut workers[0]).unwrap() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(workers[0].binding.observed_terminal);
        assert_ne!(workers[0].binding.exit_code, Some(0));
        fixture
            .host
            .cancel_pending_cohort(&policy, &run, "long-worker", &mut workers)
            .unwrap();
        let next = fixture.host.create_run(&policy).unwrap();
        let mut nextworkers = vec![
            fixture
                .host
                .spawn_worker(&policy, &next, "next-worker", "worker", &fixture.config)
                .unwrap(),
        ];
        fixture
            .host
            .cancel_pending_cohort(&policy, &next, "next-worker", &mut nextworkers)
            .unwrap();
    }

    #[test]
    fn promotion_host_concurrent_sessions_reserve_target_before_dispatch() {
        let fixture = Fixture::new();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles = (0..2)
            .map(|index| {
                let db = fixture.host.store.path().to_path_buf();
                let config = fixture.config.clone();
                let policy = fixture.policy.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let host =
                        PromotionHost::open(db, HostAuthority::from_secret([7; 32]).unwrap())
                            .unwrap();
                    let run = host.create_run(&policy).unwrap();
                    barrier.wait();
                    host.spawn_worker(&policy, &run, &format!("race-{index}"), "worker", &config)
                        .is_ok()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .filter(|ok| *ok)
                .count(),
            1
        );
        let count: i64 = fixture
            .host
            .store
            .connect()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM host_checkpoint_promotion_workers",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn promotion_host_mandatory_profile_cannot_be_weakened() {
        let fixture = Fixture::new();
        for case in ["missing", "threshold", "mode", "source"] {
            let mut policy = fixture.policy.clone();
            match case {
                "missing" => {
                    policy.mandatory_criteria.remove(0);
                }
                "threshold" => policy.mandatory_criteria[0].threshold = 1.0,
                "mode" => policy.mandatory_criteria[0].mode = Direction::Max,
                _ => policy.mandatory_criteria[0].source = "trainer".into(),
            };
            assert!(fixture.host.create_run(&policy).is_err());
        }
    }

    #[test]
    fn promotion_host_direct_argv_file_grammar_requires_bound_absolute_inputs() {
        let fixture = Fixture::new();
        let config: serde_json::Value = read_json(&fixture.config, "fixture").unwrap();
        let input = fixture.temp.path().join("unpinned-rules.json");
        fs::write(&input, b"{}").unwrap();
        assert!(
            verify_direct_inputs(
                &config,
                &[format!("--rules={}", input.display())],
                &[],
                None
            )
            .is_err()
        );
        assert!(
            verify_direct_inputs(
                &config,
                &["--rules=missing-relative.json".into()],
                &[],
                None
            )
            .is_err()
        );
        assert!(
            verify_direct_inputs(
                &config,
                &[format!("--rules={}", fixture.dependency.display())],
                &[],
                None
            )
            .is_ok()
        );
        fs::write(&fixture.dependency, b"changed dependency").unwrap();
        assert!(verify_dependencies(&config).is_err());
        let output = fixture.temp.path().join("fresh-output.json");
        assert!(
            verify_direct_inputs(
                &config,
                &[format!("--output={}", output.display())],
                &[],
                Some(&output)
            )
            .is_ok()
        );
        assert!(
            verify_direct_inputs(
                &config,
                &[format!(
                    "--other={}",
                    fixture.temp.path().join("other-output.json").display()
                )],
                &[],
                Some(&output)
            )
            .is_err()
        );
    }

    #[test]
    fn promotion_host_legacy_store_entry_never_installs() {
        let fixture = Fixture::new();
        let run = fixture.host.create_run(&fixture.policy).unwrap();
        let request = fixture.request(&run, "legacy", "score:4");
        assert!(
            fixture
                .host
                .store
                .promote_checkpoint(CheckpointPromotionRequest {
                    goal_id: &fixture.policy.goal_id,
                    metric: "score",
                    mode: PromotionMode::Max,
                    value: 100.0,
                    run_id: &run,
                    trial_id: "legacy",
                    candidate: &request.candidate_path,
                    live: &fixture.live
                })
                .is_err()
        );
        fixture.assert_no_install();
    }

    #[test]
    fn promotion_host_persistence_failure_leaves_launch_intent_and_no_permission() {
        let fixture = Fixture::new();
        let run = fixture.host.create_run(&fixture.policy).unwrap();
        let worker = fixture.worker(&run, "persist-failure");
        fixture.host.store.connect().unwrap().execute_batch("CREATE TRIGGER reject_evaluator_registration BEFORE UPDATE ON host_checkpoint_promotion_trials WHEN NEW.state='started' BEGIN SELECT RAISE(ABORT,'fixture persistence failure'); END;").unwrap();
        assert!(
            fixture
                .host
                .begin_trial(
                    fixture.policy.clone(),
                    fixture.request(&run, "persist-failure", "slow"),
                    &mut [worker]
                )
                .is_err()
        );
        let state: String = fixture
            .host
            .store
            .connect()
            .unwrap()
            .query_row(
                "SELECT state FROM host_checkpoint_promotion_trials WHERE run_id=?",
                [run],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(state, "launch-pending");
        fixture.assert_no_install();
    }

    #[test]
    fn host_provisioning_is_explicit_empty_only_and_reopen_preserves_identity() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("host.sqlite3");
        assert!(
            PromotionHost::open(path.clone(), HostAuthority::from_secret([7; 32]).unwrap())
                .is_err()
        );
        assert!(!path.exists());
        let store = Store::open(path.clone()).unwrap();
        assert!(
            PromotionHost::open(path.clone(), HostAuthority::from_secret([7; 32]).unwrap())
                .is_err()
        );
        let count: i64 = store
            .connect()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM promotion_host_authority", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
        drop(store);
        let host = PromotionHost::provision_empty(
            path.clone(),
            HostAuthority::from_secret([7; 32]).unwrap(),
        )
        .unwrap();
        let epoch = host.epoch.clone();
        assert!(
            PromotionHost::provision_empty(
                path.clone(),
                HostAuthority::from_secret([7; 32]).unwrap()
            )
            .is_err()
        );
        drop(host);
        let reopened =
            PromotionHost::open(path, HostAuthority::from_secret([7; 32]).unwrap()).unwrap();
        assert_eq!(reopened.epoch, epoch);
    }

    #[test]
    fn host_provisioning_cannot_claim_populated_projection_or_knowledge_tables() {
        for seed in [
            "INSERT INTO research_sources(source_id,media_type,accessed_at,source_json) VALUES ('fixture','runtime-trace','fixture','{}')",
            "CREATE TABLE project_projection(value TEXT); INSERT INTO project_projection(value) VALUES ('retained')",
            "CREATE TABLE sqliteXprojection(value TEXT); INSERT INTO sqliteXprojection(value) VALUES ('retained')",
        ] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("existing.sqlite3");
            let store = Store::open(path.clone()).unwrap();
            store.connect().unwrap().execute_batch(seed).unwrap();
            assert!(
                PromotionHost::provision_empty(path, HostAuthority::from_secret([7; 32]).unwrap())
                    .is_err()
            );
            let count: i64 = store
                .connect()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM promotion_host_authority", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 0);
            let populated: i64 = store
                .connect()
                .unwrap()
                .query_row(
                    if seed.starts_with("INSERT") {
                        "SELECT COUNT(*) FROM research_sources"
                    } else if seed.contains("sqliteX") {
                        "SELECT COUNT(*) FROM sqliteXprojection"
                    } else {
                        "SELECT COUNT(*) FROM project_projection"
                    },
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(populated, 1);
        }
    }

    #[test]
    fn host_extension_preserves_both_supported_store_versions_and_modern_records() {
        for version in [1, 2] {
            let temp = tempfile::tempdir().unwrap();
            let path = temp.path().join("schema.sqlite3");
            let connection = Connection::open(&path).unwrap();
            connection
                .pragma_update(None, "user_version", version)
                .unwrap();
            drop(connection);
            let host = PromotionHost::provision_empty(
                path.clone(),
                HostAuthority::from_secret([7; 32]).unwrap(),
            )
            .unwrap();
            let preserved: i64 = host
                .store
                .connect()
                .unwrap()
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .unwrap();
            assert_eq!(preserved, version);
            let run = host
                .store
                .create_run(
                    "fixture.environment-v1",
                    "1.0",
                    "goal",
                    json!({"modern": {"learning_stage": "trainer"}}),
                )
                .unwrap();
            assert_eq!(
                host.store.get_run(&run.run_id).unwrap().metadata,
                run.metadata
            );
            let digest: Option<String> = host
                .store
                .connect()
                .unwrap()
                .query_row(
                    "SELECT environment_config_digest FROM runs WHERE run_id=?",
                    [&run.run_id],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(digest, None);
            let authority_rows: i64 = host
                .store
                .connect()
                .unwrap()
                .query_row("SELECT COUNT(*) FROM promotion_host_authority", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(authority_rows, 1);
            drop(host);
            let read_only = Store::read_only(path).unwrap();
            assert_eq!(
                read_only.get_run(&run.run_id).unwrap().metadata,
                run.metadata
            );
        }
    }

    #[test]
    fn host_metric_floor_allows_training_metrics_but_never_reuses_a_stale_boundary() {
        let fixture = Fixture::new();
        let run = fixture.host.create_run(&fixture.policy).unwrap();
        let initial = fixture
            .host
            .trial_metric_floor(&fixture.policy, &run)
            .unwrap();
        assert_eq!(initial, 0);
        fixture
            .host
            .store
            .append_metric(
                &run,
                "training.reward",
                100.0,
                Some(1),
                json!({"source":"trainer","authority":"authoritative"}),
            )
            .unwrap();
        let floor = fixture
            .host
            .trial_metric_floor(&fixture.policy, &run)
            .unwrap();
        assert!(floor > initial);
        let mut foreign = fixture.policy.clone();
        foreign.goal_id = "goal.foreign".into();
        assert!(fixture.host.trial_metric_floor(&foreign, &run).is_err());
        let mut workers = vec![fixture.worker(&run, "metric-boundary")];
        let mut request = fixture.request(&run, "metric-boundary", "score:4");
        request.metric_floor = initial;
        assert!(
            fixture
                .host
                .begin_trial(fixture.policy.clone(), request, &mut workers)
                .is_err()
        );
        // The refused request does not create a trial or consume the worker
        // reservation; explicitly cancel the exact owned cohort and retry.
        let next = fixture.host.create_run(&fixture.policy).unwrap();
        assert!(
            fixture
                .host
                .spawn_worker(
                    &fixture.policy,
                    &next,
                    "still-reserved",
                    "worker",
                    &fixture.config
                )
                .is_err()
        );
        fixture
            .host
            .cancel_pending_cohort(&fixture.policy, &run, "metric-boundary", &mut workers)
            .unwrap();
        let mut next_workers = vec![fixture.worker(&next, "after-owned-stop")];
        fixture
            .host
            .cancel_pending_cohort(
                &fixture.policy,
                &next,
                "after-owned-stop",
                &mut next_workers,
            )
            .unwrap();
        fixture.assert_no_install();

        let positive = Fixture::new();
        let run = positive.host.create_run(&positive.policy).unwrap();
        positive
            .host
            .store
            .append_metric(
                &run,
                "training.reward",
                100.0,
                Some(1),
                json!({"source":"trainer","authority":"authoritative"}),
            )
            .unwrap();
        let worker = positive.worker(&run, "training-then-evaluate");
        let mut request = positive.request(&run, "training-then-evaluate", "score:4");
        request.metric_floor = positive
            .host
            .trial_metric_floor(&positive.policy, &run)
            .unwrap();
        let metric_floor = request.metric_floor;
        let mut trial = positive
            .host
            .begin_trial(positive.policy.clone(), request, &mut [worker])
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !positive.host.finish_trial(&mut trial).unwrap() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        let authorization = positive
            .host
            .review(trial.trial_token(), true)
            .unwrap()
            .unwrap();
        assert!(authorization.final_measurement.metric_id > metric_floor);
        assert_eq!(authorization.final_measurement.value, 4.0);
        assert!(
            positive
                .host
                .install(&authorization.authorization_id)
                .unwrap()
                .promoted
        );
        assert_eq!(fs::read(&positive.live).unwrap(), b"score:4");
        assert!(
            positive
                .host
                .trial_metric_floor(&positive.policy, &run)
                .is_err()
        );
    }
}
