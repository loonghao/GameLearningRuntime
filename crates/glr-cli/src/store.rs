use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::types::{Type, Value as SqlValue};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::contracts::{
    Authority, GoalEvidence, PromotionMode, ResearchBundle, RouteWaypoint, SpatialEntity,
    SpatialKnowledgeBundle, SpatialKnowledgeGraph, SpatialRoute, sha256_file,
};
use crate::error::{Error, Result};
use crate::project::validate_identifier;
use crate::promotion_journal::{
    Binding as CheckpointBinding, Journal as CheckpointJournal, Phase as CheckpointPhase,
};

pub const RUN_STORE_SCHEMA_VERSION: i64 = 1;
// Python v2 adds nullable digest columns and project-owned projections.
// CLI queries name their columns and can read/write both formats.
pub const MAX_READABLE_RUN_STORE_SCHEMA_VERSION: i64 = 2;
pub const MAX_TRANSACTION_STEPS: usize = 64;
pub const MAX_TRANSACTION_RESUME_ATTEMPTS: u32 = 16;

#[derive(Debug, Clone, Serialize)]
pub struct RunRecord {
    pub run_id: String,
    pub environment_id: String,
    pub protocol_version: String,
    pub kind: String,
    pub status: String,
    pub started_at_ns: i64,
    pub finished_at_ns: Option<i64>,
    pub exit_code: Option<i32>,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventRecord {
    pub run_id: String,
    pub sequence_id: i64,
    pub timestamp_ns: i64,
    pub kind: String,
    pub episode_id: Option<String>,
    pub step_id: Option<i64>,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricRecord {
    pub run_id: String,
    pub metric_id: i64,
    pub timestamp_ns: i64,
    pub name: String,
    pub value: f64,
    pub step_id: Option<i64>,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransactionRefusal {
    pub action_id: String,
    pub target_id: String,
    pub reason_class: String,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TransactionRecord {
    pub transaction_id: String,
    pub run_id: String,
    pub step_count: u32,
    pub next_step_index: u32,
    pub status: String,
    pub resume_attempts: u32,
    pub max_resume_attempts: u32,
    pub last_refusal: Option<Value>,
    pub updated_at_ns: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TransactionResume {
    pub transaction: TransactionRecord,
    pub outcome: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactRecord {
    pub run_id: String,
    pub path: String,
    pub role: String,
    pub media_type: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub metadata: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointPromotionRecord {
    pub goal_id: String,
    pub metric: String,
    pub mode: String,
    pub best_metric: f64,
    pub checkpoint_sha256: String,
    pub checkpoint_path: String,
    pub run_id: String,
    pub trial_id: String,
    pub updated_at_ns: i64,
}

pub struct CheckpointPromotionRequest<'a> {
    pub goal_id: &'a str,
    pub metric: &'a str,
    pub mode: PromotionMode,
    pub value: f64,
    pub run_id: &'a str,
    pub trial_id: &'a str,
    pub candidate: &'a Path,
    pub live: &'a Path,
}

pub struct EntityQuery<'a> {
    pub environment_id: &'a str,
    pub world_id: &'a str,
    pub kind: Option<&'a str>,
    pub name: Option<&'a str>,
    pub near: Option<&'a [f64]>,
    pub radius: Option<f64>,
    pub limit: u32,
}

pub struct Store {
    path: PathBuf,
    read_only: bool,
    _instance: fs::File,
}

impl Store {
    pub fn open(path: PathBuf) -> Result<Self> {
        if path.is_symlink() {
            return Err(Error::Invalid("run store cannot be a symlink".into()));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        crate::promotion_journal::validate_ancestors(&path)?;
        drop(Connection::open(&path)?);
        let path = path.canonicalize()?;
        let mut options = fs::OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(0x1 | 0x2); // FILE_SHARE_READ | WRITE; no DELETE
        }
        let store = Self {
            _instance: options.open(&path)?,
            path,
            read_only: false,
        };
        store.initialize()?;
        store.reconcile_checkpoint_promotions()?;
        Ok(store)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
    pub(crate) fn instance_guard(&self) -> Result<fs::File> {
        Ok(self._instance.try_clone()?)
    }

    pub(crate) fn connect(&self) -> Result<Connection> {
        crate::promotion_journal::validate_ancestors(&self.path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let pinned = self._instance.metadata()?;
            let current = fs::metadata(&self.path)?;
            if pinned.dev() != current.dev() || pinned.ino() != current.ino() {
                return Err(Error::Contract(
                    "cached store database instance was replaced".into(),
                ));
            }
        }
        let connection = if self.read_only {
            Connection::open_with_flags(&self.path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?
        } else {
            Connection::open(&self.path)?
        };
        connection.busy_timeout(Duration::from_secs(if self.read_only { 1 } else { 30 }))?;
        connection.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(connection)
    }

    /// Observation never creates, migrates, or writes the training database.
    pub fn read_only(path: PathBuf) -> Result<Self> {
        if path.is_symlink() || !path.is_file() {
            return Err(Error::Missing(path));
        }
        crate::promotion_journal::validate_ancestors(&path)?;
        let path = path.canonicalize()?;
        let mut options = fs::OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            options.share_mode(0x1 | 0x2);
        }
        let store = Self {
            _instance: options.open(&path)?,
            path,
            read_only: true,
        };
        let version: i64 = store
            .connect()?
            .query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if !(1..=MAX_READABLE_RUN_STORE_SCHEMA_VERSION).contains(&version) {
            return Err(Error::Contract(format!(
                "unsupported run store schema version: {version}"
            )));
        }
        Ok(store)
    }

    /// Bounded, resumable projection. Oversized records retain their cursor with
    /// an explicit marker rather than silently dropping data or allocating it.
    pub fn observation_page(
        &self,
        run_id: &str,
        table: &str,
        after: i64,
        limit: u32,
    ) -> Result<Vec<Value>> {
        let (sql, id, payload) = match table {
            "events" => (
                "SELECT sequence_id, timestamp_ns, kind, episode_id, step_id, CASE WHEN length(CAST(payload_json AS BLOB)) <= 16384 THEN payload_json ELSE '{\"observation_truncated\":true}' END AS data FROM events WHERE run_id = ? AND sequence_id > ? ORDER BY sequence_id LIMIT ?",
                "sequence_id",
                "payload",
            ),
            "metrics" => (
                "SELECT metric_id, timestamp_ns, name, value, step_id, CASE WHEN length(CAST(metadata_json AS BLOB)) <= 16384 THEN metadata_json ELSE '{\"observation_truncated\":true}' END AS data FROM metrics WHERE run_id = ? AND metric_id > ? ORDER BY metric_id LIMIT ?",
                "metric_id",
                "metadata",
            ),
            _ => return Err(Error::Invalid("unknown observation table".into())),
        };
        let connection = self.connect()?;
        let mut query = connection.prepare(sql)?;
        Ok(query
            .query_map(params![run_id, after, limit.clamp(1, 250)], |row| {
                let mut item = json!({"run_id": run_id, id: row.get::<_, i64>(id)?,
                "timestamp_ns": row.get::<_, i64>("timestamp_ns")?,
                "step_id": row.get::<_, Option<i64>>("step_id")?,
                payload: parse_json_row(row.get::<_, String>("data")?)?});
                if table == "events" {
                    item["kind"] = row.get::<_, String>("kind")?.into();
                    item["episode_id"] = row.get::<_, Option<String>>("episode_id")?.into();
                } else {
                    item["name"] = row.get::<_, String>("name")?.into();
                    item["value"] = row.get::<_, f64>("value")?.into();
                }
                Ok(item)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn initialize(&self) -> Result<()> {
        let mut connection = self.connect()?;
        connection.execute_batch("PRAGMA journal_mode = WAL;")?;
        let connection =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if !(0..=MAX_READABLE_RUN_STORE_SCHEMA_VERSION).contains(&version) {
            return Err(Error::Contract(format!(
                "unsupported run store schema version: {version} at {}; this CLI reads versions 0..={MAX_READABLE_RUN_STORE_SCHEMA_VERSION}; upgrade GLR before opening this store (do not lower PRAGMA user_version)",
                self.path.display()
            )));
        }
        connection.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                environment_id TEXT NOT NULL,
                protocol_version TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at_ns INTEGER NOT NULL,
                finished_at_ns INTEGER,
                exit_code INTEGER,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS runs_environment_started
                ON runs(environment_id, started_at_ns DESC);
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                sequence_id INTEGER NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                episode_id TEXT,
                step_id INTEGER,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(run_id, sequence_id)
            );
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                timestamp_ns INTEGER NOT NULL,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                step_id INTEGER,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS metrics_run_name_step
                ON metrics(run_id, name, step_id);
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                steps_json TEXT NOT NULL,
                next_step_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                resume_attempts INTEGER NOT NULL,
                max_resume_attempts INTEGER NOT NULL,
                last_refusal_json TEXT,
                updated_at_ns INTEGER NOT NULL,
                CHECK (status IN ('pending', 'completed', 'abandoned'))
            );
            CREATE INDEX IF NOT EXISTS transactions_run_status
                ON transactions(run_id, status, updated_at_ns DESC);
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                role TEXT NOT NULL,
                media_type TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY(run_id, path)
            );
            CREATE TABLE IF NOT EXISTS spatial_entities (
                environment_id TEXT NOT NULL,
                world_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                label TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                coordinate_frame TEXT NOT NULL,
                authority TEXT NOT NULL,
                confidence REAL NOT NULL,
                observed_at_ns INTEGER NOT NULL,
                source_run_id TEXT NOT NULL REFERENCES runs(run_id),
                metadata_json TEXT NOT NULL,
                PRIMARY KEY(environment_id, world_id, entity_id)
            );
            CREATE INDEX IF NOT EXISTS spatial_entities_lookup
                ON spatial_entities(environment_id, world_id, kind, label);
            CREATE TABLE IF NOT EXISTS spatial_routes (
                environment_id TEXT NOT NULL,
                world_id TEXT NOT NULL,
                route_id TEXT NOT NULL,
                name TEXT NOT NULL,
                from_entity_id TEXT,
                to_entity_id TEXT,
                coordinate_frame TEXT NOT NULL,
                confidence REAL NOT NULL,
                verified_at_ns INTEGER NOT NULL,
                source_run_id TEXT NOT NULL REFERENCES runs(run_id),
                metadata_json TEXT NOT NULL,
                PRIMARY KEY(environment_id, world_id, route_id)
            );
            CREATE INDEX IF NOT EXISTS spatial_routes_lookup
                ON spatial_routes(environment_id, world_id, from_entity_id, to_entity_id);
            CREATE TABLE IF NOT EXISTS route_waypoints (
                environment_id TEXT NOT NULL,
                world_id TEXT NOT NULL,
                route_id TEXT NOT NULL,
                waypoint_index INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                tolerance REAL NOT NULL,
                label TEXT,
                PRIMARY KEY(environment_id, world_id, route_id, waypoint_index),
                FOREIGN KEY(environment_id, world_id, route_id)
                    REFERENCES spatial_routes(environment_id, world_id, route_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS spatial_graphs (
                environment_id TEXT NOT NULL,
                protocol_version TEXT NOT NULL,
                graph_id TEXT NOT NULL,
                exported_at_ns INTEGER NOT NULL,
                source_run_id TEXT NOT NULL REFERENCES runs(run_id),
                graph_json TEXT NOT NULL,
                PRIMARY KEY(environment_id, graph_id)
            );
            CREATE INDEX IF NOT EXISTS spatial_graphs_lookup
                ON spatial_graphs(environment_id, protocol_version, exported_at_ns DESC);
            CREATE TABLE IF NOT EXISTS research_sources (
                source_id TEXT PRIMARY KEY,
                media_type TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                source_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_findings (
                finding_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_id TEXT,
                finding_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS research_findings_lookup
                ON research_findings(scope, scope_id, category, status);
            CREATE TABLE IF NOT EXISTS research_finding_sources (
                finding_id TEXT NOT NULL REFERENCES research_findings(finding_id) ON DELETE CASCADE,
                source_id TEXT NOT NULL REFERENCES research_sources(source_id),
                ordinal INTEGER NOT NULL,
                PRIMARY KEY(finding_id, source_id)
            );
            CREATE TABLE IF NOT EXISTS research_finding_tags (
                finding_id TEXT NOT NULL REFERENCES research_findings(finding_id) ON DELETE CASCADE,
                tag TEXT NOT NULL,
                PRIMARY KEY(finding_id, tag)
            );
            CREATE INDEX IF NOT EXISTS research_finding_tags_lookup
                ON research_finding_tags(tag, finding_id);
            CREATE TABLE IF NOT EXISTS checkpoint_promotion_locations (
                goal_id TEXT PRIMARY KEY,
                environment_id TEXT NOT NULL,
                protocol_version TEXT NOT NULL,
                target_id TEXT NOT NULL,
                environment_config_sha256 TEXT NOT NULL,
                host_fingerprint TEXT NOT NULL, host_epoch TEXT NOT NULL,
                store_path TEXT NOT NULL,
                live_path TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS checkpoint_promotions (
                goal_id TEXT PRIMARY KEY,
                metric TEXT NOT NULL,
                mode TEXT NOT NULL,
                best_metric REAL NOT NULL,
                checkpoint_sha256 TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                trial_id TEXT NOT NULL,
                updated_at_ns INTEGER NOT NULL
            );
            "#,
        )?;
        connection.pragma_update(None, "user_version", version.max(RUN_STORE_SCHEMA_VERSION))?;
        connection.commit()?;
        crate::promotion_host::initialize(&mut self.connect()?)?;
        Ok(())
    }

    pub fn create_run(
        &self,
        environment_id: &str,
        protocol_version: &str,
        kind: &str,
        metadata: Value,
    ) -> Result<RunRecord> {
        validate_identifier(environment_id, "environment_id")?;
        validate_identifier(kind, "run kind")?;
        let run_id = format!("run-{}", Uuid::new_v4().simple());
        let started_at_ns = now_ns()?;
        self.connect()?.execute(
            "INSERT INTO runs(run_id, environment_id, protocol_version, kind, status, started_at_ns, finished_at_ns, exit_code, metadata_json) VALUES (?, ?, ?, ?, 'running', ?, NULL, NULL, ?)",
            params![run_id, environment_id, protocol_version, kind, started_at_ns, compact_json(&metadata)?],
        )?;
        self.get_run(&run_id)
    }

    pub fn finish_run(
        &self,
        run_id: &str,
        status: &str,
        exit_code: Option<i32>,
    ) -> Result<RunRecord> {
        if !matches!(status, "succeeded" | "failed" | "interrupted") {
            return Err(Error::Invalid(
                "finish_run requires a terminal status".into(),
            ));
        }
        let changed = self.connect()?.execute(
            "UPDATE runs SET status = ?, finished_at_ns = ?, exit_code = ? WHERE run_id = ? AND status = 'running'",
            params![status, now_ns()?, exit_code, run_id],
        )?;
        if changed != 1 {
            return Err(Error::Contract("run is missing or already terminal".into()));
        }
        self.get_run(run_id)
    }

    pub fn begin_transaction(
        &self,
        run_id: &str,
        transaction_id: &str,
        steps: &[Value],
        max_resume_attempts: u32,
    ) -> Result<TransactionRecord> {
        validate_identifier(transaction_id, "transaction_id")?;
        if steps.is_empty() || steps.len() > MAX_TRANSACTION_STEPS {
            return Err(Error::Invalid(format!(
                "transaction steps must contain 1-{MAX_TRANSACTION_STEPS} entries"
            )));
        }
        if !(1..=MAX_TRANSACTION_RESUME_ATTEMPTS).contains(&max_resume_attempts) {
            return Err(Error::Invalid(format!(
                "max_resume_attempts must be between 1 and {MAX_TRANSACTION_RESUME_ATTEMPTS}"
            )));
        }
        let steps_json = compact_json(&steps)?;
        if steps_json.len() > 64 * 1024 {
            return Err(Error::Invalid(
                "transaction steps exceed the 64 KiB limit".into(),
            ));
        }
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        let status: Option<String> = transaction
            .query_row(
                "SELECT status FROM runs WHERE run_id = ?",
                [run_id],
                |row| row.get(0),
            )
            .optional()?;
        if status.as_deref() != Some("running") {
            return Err(Error::Contract(
                "transactions require an existing running run".into(),
            ));
        }
        transaction.execute(
            "INSERT INTO transactions(transaction_id, run_id, steps_json, next_step_index, status, resume_attempts, max_resume_attempts, last_refusal_json, updated_at_ns) VALUES (?, ?, ?, 0, 'pending', 0, ?, NULL, ?)",
            params![transaction_id, run_id, steps_json, max_resume_attempts, now_ns()?],
        )?;
        append_event_transaction(
            &transaction,
            run_id,
            "transaction.started",
            json!({
                "transaction_id": transaction_id,
                "step_count": steps.len(),
                "max_resume_attempts": max_resume_attempts,
            }),
        )?;
        transaction.commit()?;
        self.get_transaction(transaction_id)
    }

    pub fn get_transaction(&self, transaction_id: &str) -> Result<TransactionRecord> {
        validate_identifier(transaction_id, "transaction_id")?;
        self.connect()?
            .query_row(
                "SELECT transaction_id, run_id, steps_json, next_step_index, status, resume_attempts, max_resume_attempts, last_refusal_json, updated_at_ns FROM transactions WHERE transaction_id = ?",
                [transaction_id],
                transaction_from_row,
            )
            .optional()?
            .ok_or_else(|| Error::Contract(format!("unknown transaction_id: {transaction_id}")))
    }

    pub fn resume_transaction(
        &self,
        transaction_id: &str,
        refusal: Option<&TransactionRefusal>,
    ) -> Result<TransactionResume> {
        validate_identifier(transaction_id, "transaction_id")?;
        if let Some(refusal) = refusal {
            validate_transaction_refusal(refusal)?;
        }
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        let mut record = transaction
            .query_row(
                "SELECT transaction_id, run_id, steps_json, next_step_index, status, resume_attempts, max_resume_attempts, last_refusal_json, updated_at_ns FROM transactions WHERE transaction_id = ?",
                [transaction_id],
                transaction_from_row,
            )
            .optional()?
            .ok_or_else(|| Error::Contract(format!("unknown transaction_id: {transaction_id}")))?;
        if record.status != "pending" {
            transaction.commit()?;
            return Ok(TransactionResume {
                transaction: record,
                outcome: "already_terminal".into(),
            });
        }
        let steps_json: String = transaction.query_row(
            "SELECT steps_json FROM transactions WHERE transaction_id = ?",
            [transaction_id],
            |row| row.get(0),
        )?;
        let steps: Vec<Value> = serde_json::from_str(&steps_json)
            .map_err(|error| Error::Contract(format!("transaction steps are corrupt: {error}")))?;
        let step_count = u32::try_from(steps.len())
            .map_err(|_| Error::Contract("transaction step count overflows u32".into()))?;
        if let Some(refusal) = refusal {
            let refusal_json = compact_json(refusal)?;
            let structural = refusal.reason_class == "structural";
            let attempts = if structural {
                record.resume_attempts.saturating_add(1)
            } else {
                record.resume_attempts
            };
            let abandon = structural && attempts >= record.max_resume_attempts;
            let status = if abandon { "abandoned" } else { "pending" };
            transaction.execute(
                "UPDATE transactions SET status = ?, resume_attempts = ?, last_refusal_json = ?, updated_at_ns = ? WHERE transaction_id = ? AND status = 'pending'",
                params![status, attempts, refusal_json, now_ns()?, transaction_id],
            )?;
            let kind = if abandon {
                "transaction.abandoned"
            } else {
                "transaction.refused"
            };
            append_event_transaction(
                &transaction,
                &record.run_id,
                kind,
                json!({
                    "transaction_id": transaction_id,
                    "next_step_index": record.next_step_index,
                    "step_count": step_count,
                    "resume_attempts": attempts,
                    "max_resume_attempts": record.max_resume_attempts,
                    "refusal": refusal,
                    "abandoned": abandon,
                }),
            )?;
            record = transaction
                .query_row(
                    "SELECT transaction_id, run_id, steps_json, next_step_index, status, resume_attempts, max_resume_attempts, last_refusal_json, updated_at_ns FROM transactions WHERE transaction_id = ?",
                    [transaction_id],
                    transaction_from_row,
                )?;
            transaction.commit()?;
            return Ok(TransactionResume {
                transaction: record,
                outcome: if abandon { "abandoned" } else { "refused" }.into(),
            });
        }

        let next_step_index = record.next_step_index.saturating_add(1);
        if next_step_index > step_count {
            return Err(Error::Contract(
                "transaction next_step_index is beyond its step count".into(),
            ));
        }
        let status = if next_step_index == step_count {
            "completed"
        } else {
            "pending"
        };
        transaction.execute(
            "UPDATE transactions SET next_step_index = ?, status = ?, last_refusal_json = NULL, updated_at_ns = ? WHERE transaction_id = ? AND status = 'pending'",
            params![next_step_index, status, now_ns()?, transaction_id],
        )?;
        append_event_transaction(
            &transaction,
            &record.run_id,
            if status == "completed" {
                "transaction.completed"
            } else {
                "transaction.step_advanced"
            },
            json!({
                "transaction_id": transaction_id,
                "next_step_index": next_step_index,
                "step_count": step_count,
            }),
        )?;
        record = transaction
            .query_row(
                "SELECT transaction_id, run_id, steps_json, next_step_index, status, resume_attempts, max_resume_attempts, last_refusal_json, updated_at_ns FROM transactions WHERE transaction_id = ?",
                [transaction_id],
                transaction_from_row,
            )?;
        transaction.commit()?;
        Ok(TransactionResume {
            transaction: record,
            outcome: if status == "completed" {
                "completed"
            } else {
                "advanced"
            }
            .into(),
        })
    }

    pub fn get_run(&self, run_id: &str) -> Result<RunRecord> {
        self.connect()?
            .query_row(
                if self.read_only { "SELECT run_id, environment_id, protocol_version, kind, status, started_at_ns, finished_at_ns, exit_code, CASE WHEN length(CAST(metadata_json AS BLOB)) <= 16384 THEN metadata_json ELSE '{\"observation_truncated\":true}' END AS metadata_json FROM runs WHERE run_id = ?" } else { "SELECT * FROM runs WHERE run_id = ?" },
                [run_id],
                run_from_row,
            )
            .optional()?
            .ok_or_else(|| Error::Contract(format!("unknown run_id: {run_id}")))
    }

    pub fn observation_runs(
        &self,
        environment_id: &str,
        before: Option<&str>,
    ) -> Result<Vec<RunRecord>> {
        let connection = self.connect()?;
        let mut query = connection.prepare("SELECT run_id, environment_id, protocol_version, kind, status, started_at_ns, finished_at_ns, exit_code, CASE WHEN length(CAST(metadata_json AS BLOB)) <= 16384 THEN metadata_json ELSE '{\"observation_truncated\":true}' END AS metadata_json FROM runs WHERE environment_id = ?1 AND (?2 IS NULL OR (started_at_ns,run_id) < (SELECT started_at_ns,run_id FROM runs WHERE run_id=?2 AND environment_id=?1)) ORDER BY started_at_ns DESC, run_id DESC LIMIT 100")?;
        Ok(query
            .query_map(params![environment_id, before], run_from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn list_runs(
        &self,
        environment_id: &str,
        status: Option<&str>,
        limit: u32,
    ) -> Result<Vec<RunRecord>> {
        let connection = self.connect()?;
        let mut rows = if let Some(status) = status {
            let mut statement = connection.prepare("SELECT * FROM runs WHERE environment_id = ? AND status = ? ORDER BY started_at_ns DESC LIMIT ?")?;
            statement
                .query_map(params![environment_id, status, limit], run_from_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            let mut statement = connection.prepare(
                "SELECT * FROM runs WHERE environment_id = ? ORDER BY started_at_ns DESC LIMIT ?",
            )?;
            statement
                .query_map(params![environment_id, limit], run_from_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        rows.shrink_to_fit();
        Ok(rows)
    }

    pub fn append_event(&self, run_id: &str, kind: &str, payload: Value) -> Result<()> {
        validate_identifier(kind, "event kind")?;
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        let status: Option<String> = transaction
            .query_row(
                "SELECT status FROM runs WHERE run_id = ?",
                [run_id],
                |row| row.get(0),
            )
            .optional()?;
        if status.as_deref() != Some("running") {
            return Err(Error::Contract(
                "cannot append an event to a missing or terminal run".into(),
            ));
        }
        let sequence: i64 = transaction.query_row(
            "SELECT COALESCE(MAX(sequence_id), 0) + 1 FROM events WHERE run_id = ?",
            [run_id],
            |row| row.get(0),
        )?;
        transaction.execute(
            "INSERT INTO events(run_id, sequence_id, timestamp_ns, kind, episode_id, step_id, payload_json) VALUES (?, ?, ?, ?, NULL, NULL, ?)",
            params![run_id, sequence, now_ns()?, kind, compact_json(&payload)?],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn list_events(&self, run_id: &str) -> Result<Vec<EventRecord>> {
        self.events_after(run_id, -1)
    }

    pub fn events_after(&self, run_id: &str, after: i64) -> Result<Vec<EventRecord>> {
        let connection = self.connect()?;
        let mut statement = connection
            .prepare("SELECT * FROM events WHERE run_id = ? AND sequence_id > ? ORDER BY sequence_id ASC LIMIT 1000")?;
        Ok(statement
            .query_map(params![run_id, after], |row| {
                Ok(EventRecord {
                    run_id: row.get("run_id")?,
                    sequence_id: row.get("sequence_id")?,
                    timestamp_ns: row.get("timestamp_ns")?,
                    kind: row.get("kind")?,
                    episode_id: row.get("episode_id")?,
                    step_id: row.get("step_id")?,
                    payload: parse_json_row(row.get::<_, String>("payload_json")?)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn list_metrics(&self, run_id: &str) -> Result<Vec<MetricRecord>> {
        self.metrics_after(run_id, 0)
    }

    pub fn metrics_after(&self, run_id: &str, after: i64) -> Result<Vec<MetricRecord>> {
        let connection = self.connect()?;
        let mut statement = connection
            .prepare("SELECT * FROM metrics WHERE run_id = ? AND metric_id > ? ORDER BY metric_id ASC LIMIT 1000")?;
        Ok(statement
            .query_map(params![run_id, after], |row| {
                Ok(MetricRecord {
                    run_id: row.get("run_id")?,
                    metric_id: row.get("metric_id")?,
                    timestamp_ns: row.get("timestamp_ns")?,
                    name: row.get("name")?,
                    value: row.get("value")?,
                    step_id: row.get("step_id")?,
                    metadata: parse_json_row(row.get::<_, String>("metadata_json")?)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn latest_metric_id(&self, run_id: &str) -> Result<i64> {
        Ok(self.connect()?.query_row(
            "SELECT COALESCE(MAX(metric_id), 0) FROM metrics WHERE run_id = ?",
            [run_id],
            |row| row.get(0),
        )?)
    }

    pub fn append_metric(
        &self,
        run_id: &str,
        name: &str,
        value: f64,
        step_id: Option<i64>,
        metadata: Value,
    ) -> Result<MetricRecord> {
        validate_identifier(name, "metric name")?;
        if !value.is_finite() {
            return Err(Error::Invalid("metric value must be finite".into()));
        }
        let connection = self.connect()?;
        let status: String = connection
            .query_row(
                "SELECT status FROM runs WHERE run_id = ?",
                [run_id],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| Error::Contract(format!("unknown run_id: {run_id}")))?;
        if status != "running" {
            return Err(Error::Contract(
                "cannot append a metric to a terminal run".into(),
            ));
        }
        connection.execute(
            "INSERT INTO metrics(run_id, timestamp_ns, name, value, step_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            params![run_id, now_ns()?, name, value, step_id, compact_json(&metadata)?],
        )?;
        let metric_id = connection.last_insert_rowid();
        Ok(MetricRecord {
            run_id: run_id.into(),
            metric_id,
            timestamp_ns: now_ns()?,
            name: name.into(),
            value,
            step_id,
            metadata,
        })
    }

    pub fn promotion_metric_value(
        &self,
        run_id: &str,
        after_metric_id: i64,
        evidence: &GoalEvidence,
    ) -> Result<f64> {
        if evidence.run_id != run_id || evidence.authority != Authority::Authoritative {
            return Err(Error::Contract(
                "checkpoint promotion requires authoritative evidence from the current run".into(),
            ));
        }
        Ok(crate::promotion_host::final_measurement(
            &self.connect()?,
            run_id,
            after_metric_id,
            &evidence.metric,
            &evidence.source,
            evidence.value,
        )?
        .value)
    }

    fn checkpoint_connection(&self) -> Result<Connection> {
        let connection = self.connect()?;
        // A busy worker is refused, never stolen after a lease/TTL expiry.
        connection.busy_timeout(Duration::from_secs(1))?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        Ok(connection)
    }

    fn checkpoint_environment(connection: &Connection, run_id: &str) -> Result<String> {
        connection
            .query_row(
                "SELECT environment_id FROM runs WHERE run_id = ?",
                [run_id],
                |row| row.get(0),
            )
            .optional()?
            .ok_or_else(|| Error::Contract("checkpoint journal references an unknown run".into()))
    }

    fn checkpoint_record(
        connection: &Connection,
        goal_id: &str,
    ) -> Result<Option<CheckpointPromotionRecord>> {
        Ok(connection.query_row(
            "SELECT metric, mode, best_metric, checkpoint_sha256, checkpoint_path, run_id, trial_id, updated_at_ns FROM checkpoint_promotions WHERE goal_id = ?",
            [goal_id], |row| Ok(CheckpointPromotionRecord {
                goal_id: goal_id.into(), metric: row.get(0)?, mode: row.get(1)?, best_metric: row.get(2)?,
                checkpoint_sha256: row.get(3)?, checkpoint_path: row.get(4)?,
                run_id: row.get(5)?, trial_id: row.get(6)?, updated_at_ns: row.get(7)?,
            }),
        ).optional()?)
    }

    fn register_checkpoint_location(
        &self,
        request: &CheckpointPromotionRequest<'_>,
        authorization_id: &str,
    ) -> Result<CheckpointBinding> {
        let mut connection = self.checkpoint_connection()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let environment_id = Self::checkpoint_environment(&transaction, request.run_id)?;
        let authorization =
            crate::promotion_host::load_authorization(&transaction, authorization_id)?;
        let scope = &authorization.binding;
        let binding = CheckpointBinding::new(
            &self.path,
            &environment_id,
            request.goal_id,
            request.live,
            &scope.protocol_version,
            &scope.target_id,
            &scope.environment_config_sha256,
            &scope.host_fingerprint,
            &scope.host_epoch,
        )?;
        let previous: Option<CheckpointBinding> = transaction.query_row(
            "SELECT environment_id, goal_id, store_path, live_path, protocol_version, target_id, environment_config_sha256, host_fingerprint, host_epoch FROM checkpoint_promotion_locations WHERE goal_id = ?",
            [request.goal_id], |row| Ok(CheckpointBinding {
                environment_id: row.get(0)?, goal_id: row.get(1)?, store_path: row.get(2)?, live_path: row.get(3)?, protocol_version: row.get(4)?, target_id: row.get(5)?, environment_config_sha256: row.get(6)?, host_fingerprint: row.get(7)?, host_epoch: row.get(8)?,
            }),
        ).optional()?;
        if previous
            .as_ref()
            .is_some_and(|previous| *previous != binding)
        {
            return Err(Error::Contract(
                "checkpoint location/environment/goal binding changed".into(),
            ));
        }
        crate::promotion_journal::claim_location(&binding)?;
        if previous.is_none() {
            transaction.execute(
                "INSERT INTO checkpoint_promotion_locations(goal_id, environment_id, store_path, live_path, protocol_version, target_id, environment_config_sha256, host_fingerprint, host_epoch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![binding.goal_id, binding.environment_id, binding.store_path, binding.live_path, binding.protocol_version, binding.target_id, binding.environment_config_sha256, binding.host_fingerprint, binding.host_epoch],
            )?;
        }
        // Registration is durable before any filesystem work. The mutation
        // transaction below reacquires the lock and rereads the incumbent.
        transaction.commit()?;
        Ok(binding)
    }

    fn reconcile_checkpoint_binding(
        &self,
        connection: &Connection,
        binding: &CheckpointBinding,
    ) -> Result<()> {
        binding.validate(&self.path)?;
        let host: (String, String) = connection.query_row(
            "SELECT fingerprint,epoch FROM promotion_host_authority WHERE singleton=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if host != (binding.host_fingerprint.clone(), binding.host_epoch.clone()) {
            return Err(Error::Contract(
                "checkpoint owner belongs to another database host instance; bytes retained".into(),
            ));
        }
        crate::promotion_journal::verify_owner(binding)?;
        if let Some(journal) = CheckpointJournal::load(binding)? {
            for record in journal
                .previous
                .iter()
                .chain(std::iter::once(&journal.proposed))
            {
                if Self::checkpoint_environment(connection, &record.run_id)?
                    != binding.environment_id
                {
                    return Err(Error::Contract(
                        "checkpoint journal run/environment binding changed; bytes retained".into(),
                    ));
                }
            }
            let authorization_state = crate::promotion_host::verify_recovery(
                connection,
                &journal.authorization_id,
                &self.path,
                &journal.proposed,
                journal.incumbent_sha256.as_deref(),
            )?;
            let current = Self::checkpoint_record(connection, &binding.goal_id)?;
            let committed = if current.as_ref() == Some(&journal.proposed) {
                true
            } else if current == journal.previous {
                false
            } else {
                return Err(Error::Contract("checkpoint database is neither journal predecessor nor successor; bytes retained".into()));
            };
            if authorization_state != if committed { "installed" } else { "approved" } {
                return Err(Error::Contract(
                    "journal/database authorization consumption state conflicts; bytes retained"
                        .into(),
                ));
            }
            journal.reconcile(committed)?;
        }
        if let Some(record) = Self::checkpoint_record(connection, &binding.goal_id)?
            && (Self::checkpoint_environment(connection, &record.run_id)? != binding.environment_id
                || crate::promotion_journal::canonical_live_path(Path::new(
                    &record.checkpoint_path,
                ))? != Path::new(&binding.live_path)
                || !Path::new(&binding.live_path).is_file()
                || sha256_file(Path::new(&binding.live_path))? != record.checkpoint_sha256)
        {
            return Err(Error::Contract(
                "checkpoint incumbent binding/hash is inconsistent; bytes retained".into(),
            ));
        }
        Ok(())
    }

    fn reconcile_checkpoint_promotions(&self) -> Result<()> {
        let mut connection = self.checkpoint_connection()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let bindings = {
            let mut statement = transaction.prepare(
                "SELECT environment_id, goal_id, store_path, live_path, protocol_version, target_id, environment_config_sha256, host_fingerprint, host_epoch FROM checkpoint_promotion_locations ORDER BY goal_id LIMIT 1025",
            )?;
            statement
                .query_map([], |row| {
                    Ok(CheckpointBinding {
                        environment_id: row.get(0)?,
                        goal_id: row.get(1)?,
                        store_path: row.get(2)?,
                        live_path: row.get(3)?,
                        protocol_version: row.get(4)?,
                        target_id: row.get(5)?,
                        environment_config_sha256: row.get(6)?,
                        host_fingerprint: row.get(7)?,
                        host_epoch: row.get(8)?,
                    })
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        if bindings.len() > 1024 {
            return Err(Error::Contract(
                "checkpoint recovery location bound exceeded".into(),
            ));
        }
        for binding in bindings {
            self.reconcile_checkpoint_binding(&transaction, &binding)?;
        }
        transaction.commit()?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn promote_checkpoint(
        &self,
        _request: CheckpointPromotionRequest<'_>,
    ) -> Result<(bool, CheckpointPromotionRecord)> {
        Err(Error::Contract("legacy checkpoint promotion is disabled: persisted host evaluation, supervisor stop and independent review are required".into()))
    }

    pub(crate) fn install_authorized_checkpoint(
        &self,
        request: CheckpointPromotionRequest<'_>,
        authorization_id: &str,
    ) -> Result<(bool, CheckpointPromotionRecord)> {
        match self.promote_checkpoint_inner(request, authorization_id, |_| Ok(())) {
            Ok(result) => Ok(result),
            Err(error) => match self.reconcile_checkpoint_promotions() {
                Ok(()) => Err(error),
                Err(recovery) => Err(Error::Contract(format!(
                    "checkpoint promotion failed: {error}; recovery refused: {recovery}; journal and model bytes retained"
                ))),
            },
        }
    }

    // Fault callbacks are private and supplied only by Rust test fixtures.
    // Production always passes the no-op above; no environment crash switch.
    fn promote_checkpoint_inner(
        &self,
        request: CheckpointPromotionRequest<'_>,
        authorization_id: &str,
        mut observe: impl FnMut(CheckpointPhase) -> Result<()>,
    ) -> Result<(bool, CheckpointPromotionRecord)> {
        validate_identifier(request.goal_id, "goal_id")?;
        validate_identifier(request.metric, "checkpoint promotion metric")?;
        validate_identifier(request.trial_id, "checkpoint promotion trial_id")?;
        if !request.value.is_finite() {
            return Err(Error::Invalid(
                "checkpoint promotion metric must be finite".into(),
            ));
        }
        {
            let connection = self.checkpoint_connection()?;
            crate::promotion_host::verify_authorization(
                &connection,
                authorization_id,
                &self.path,
                &request,
            )?;
        }
        crate::promotion_journal::validate_ancestors(request.candidate)?;
        if !request.candidate.is_file() {
            return Err(Error::Missing(request.candidate.to_path_buf()));
        }
        crate::promotion_journal::validate_ancestors(request.live)?;
        if let Some(parent) = request.live.parent() {
            fs::create_dir_all(parent)?;
        }
        let binding = self.register_checkpoint_location(&request, authorization_id)?;
        let mut connection = self.checkpoint_connection()?;
        let transaction =
            connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        self.reconcile_checkpoint_binding(&transaction, &binding)?;
        let minimum_improvement = crate::promotion_host::verify_authorization(
            &transaction,
            authorization_id,
            &self.path,
            &request,
        )?;
        let previous = Self::checkpoint_record(&transaction, request.goal_id)?;
        let mode = match request.mode {
            PromotionMode::Max => "max",
            PromotionMode::Min => "min",
        };
        if previous
            .as_ref()
            .is_some_and(|stored| stored.metric != request.metric || stored.mode != mode)
        {
            return Err(Error::Contract(
                "checkpoint promotion metric or mode differs from the incumbent".into(),
            ));
        }
        let improved = previous.as_ref().is_none_or(|stored| match request.mode {
            PromotionMode::Max => request.value - stored.best_metric > minimum_improvement,
            PromotionMode::Min => stored.best_metric - request.value > minimum_improvement,
        });
        if !improved {
            if transaction.execute("UPDATE host_checkpoint_promotion_trials SET state='declined' WHERE authorization_id=? AND state='approved'", [authorization_id])? != 1 { return Err(Error::Contract("authorization has already been consumed".into())); }
            transaction.commit()?;
            return Ok((
                false,
                previous.ok_or_else(|| {
                    Error::Contract("checkpoint promotion state disappeared".into())
                })?,
            ));
        }
        let journal = CheckpointJournal::stage(
            binding,
            authorization_id,
            request.candidate,
            previous,
            CheckpointPromotionRecord {
                goal_id: request.goal_id.into(),
                metric: request.metric.into(),
                mode: mode.into(),
                best_metric: request.value,
                checkpoint_sha256: String::new(),
                checkpoint_path: String::new(),
                run_id: request.run_id.into(),
                trial_id: request.trial_id.into(),
                updated_at_ns: now_ns()?,
            },
        )?;
        crate::promotion_host::verify_recovery(
            &transaction,
            authorization_id,
            &self.path,
            &journal.proposed,
            journal.incumbent_sha256.as_deref(),
        )?;
        observe(CheckpointPhase::Staged)?;
        journal.persist()?;
        observe(CheckpointPhase::Journaled)?;
        let record = journal.proposed.clone();
        transaction.execute(
            "INSERT INTO checkpoint_promotions(goal_id, metric, mode, best_metric, checkpoint_sha256, checkpoint_path, run_id, trial_id, updated_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(goal_id) DO UPDATE SET metric=excluded.metric, mode=excluded.mode, best_metric=excluded.best_metric, checkpoint_sha256=excluded.checkpoint_sha256, checkpoint_path=excluded.checkpoint_path, run_id=excluded.run_id, trial_id=excluded.trial_id, updated_at_ns=excluded.updated_at_ns",
            params![record.goal_id, record.metric, record.mode, record.best_metric, record.checkpoint_sha256,
                record.checkpoint_path, record.run_id, record.trial_id, record.updated_at_ns],
        )?;
        if transaction.execute("UPDATE host_checkpoint_promotion_trials SET state='installed' WHERE authorization_id=? AND state='approved'", [authorization_id])? != 1 { return Err(Error::Contract("authorization has already been consumed".into())); }
        observe(CheckpointPhase::SqlWritten)?;
        let actual_live = if request.live.try_exists()? {
            Some(sha256_file(request.live)?)
        } else {
            None
        };
        if actual_live != journal.incumbent_sha256 {
            return Err(Error::Contract(
                "live checkpoint drifted after staging; unknown bytes retained".into(),
            ));
        }
        journal.install(&record.checkpoint_sha256)?;
        observe(CheckpointPhase::Replaced)?;
        observe(CheckpointPhase::BeforeCommit)?;
        transaction.commit()?;
        observe(CheckpointPhase::Committed)?;
        // Reacquire the actual SQLite lock after commit. Another worker may
        // already have reconciled this intent; recovery is idempotent.
        self.reconcile_checkpoint_promotions()?;
        Ok((true, record))
    }
    pub fn has_metric_evidence(
        &self,
        run_id: &str,
        after_metric_id: i64,
        evidence: &GoalEvidence,
    ) -> Result<bool> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT value FROM metrics WHERE run_id = ? AND metric_id > ? AND name = ? AND json_extract(metadata_json, '$.source') = ? AND json_extract(metadata_json, '$.authority') = ?",
        )?;
        let values = statement
            .query_map(
                params![
                    run_id,
                    after_metric_id,
                    evidence.metric,
                    evidence.source,
                    evidence.authority.as_str()
                ],
                |row| row.get::<_, f64>(0),
            )?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(values.into_iter().any(|value| {
            (value - evidence.value).abs() <= 1e-12_f64.max(1e-9 * evidence.value.abs())
        }))
    }

    pub fn register_artifact(
        &self,
        run_id: &str,
        relative_path: &str,
        source: &Path,
        role: &str,
        media_type: &str,
    ) -> Result<ArtifactRecord> {
        validate_portable_path(relative_path)?;
        validate_identifier(role, "artifact role")?;
        if source.is_symlink() || !source.is_file() {
            return Err(Error::Missing(source.to_path_buf()));
        }
        let artifact = ArtifactRecord {
            run_id: run_id.into(),
            path: relative_path.into(),
            role: role.into(),
            media_type: media_type.into(),
            sha256: sha256_file(source)?,
            size_bytes: source.metadata()?.len(),
            metadata: json!({}),
        };
        self.connect()?.execute(
            "INSERT INTO artifacts(run_id, path, role, media_type, sha256, size_bytes, metadata_json) VALUES (?, ?, ?, ?, ?, ?, '{}') ON CONFLICT(run_id, path) DO UPDATE SET role=excluded.role, media_type=excluded.media_type, sha256=excluded.sha256, size_bytes=excluded.size_bytes, metadata_json=excluded.metadata_json",
            params![artifact.run_id, artifact.path, artifact.role, artifact.media_type, artifact.sha256, artifact.size_bytes],
        )?;
        Ok(artifact)
    }

    pub fn list_artifacts(&self, run_id: &str) -> Result<Vec<ArtifactRecord>> {
        let connection = self.connect()?;
        let mut statement =
            connection.prepare("SELECT * FROM artifacts WHERE run_id = ? ORDER BY path ASC")?;
        Ok(statement
            .query_map([run_id], |row| {
                Ok(ArtifactRecord {
                    run_id: row.get("run_id")?,
                    path: row.get("path")?,
                    role: row.get("role")?,
                    media_type: row.get("media_type")?,
                    sha256: row.get("sha256")?,
                    size_bytes: row.get("size_bytes")?,
                    metadata: parse_json_row(row.get::<_, String>("metadata_json")?)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn media_artifacts(
        &self,
        run_id: &str,
        path: Option<&str>,
        after: &str,
    ) -> Result<Vec<ArtifactRecord>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare("SELECT run_id,path,role,media_type,sha256,size_bytes FROM artifacts WHERE run_id=? AND (? IS NULL OR path=?) AND path>? ORDER BY path LIMIT 101")?;
        Ok(statement
            .query_map(params![run_id, path, path, after], |row| {
                Ok(ArtifactRecord {
                    run_id: row.get(0)?,
                    path: row.get(1)?,
                    role: row.get(2)?,
                    media_type: row.get(3)?,
                    sha256: row.get(4)?,
                    size_bytes: row.get(5)?,
                    metadata: json!({}),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn query_entities(&self, query: EntityQuery<'_>) -> Result<Vec<SpatialEntity>> {
        let mut clauses = vec!["environment_id = ?".to_string(), "world_id = ?".to_string()];
        let mut parameters = vec![
            SqlValue::Text(query.environment_id.into()),
            SqlValue::Text(query.world_id.into()),
        ];
        if let Some(kind) = query.kind {
            clauses.push("kind = ?".into());
            parameters.push(SqlValue::Text(kind.into()));
        }
        if let Some(name) = query.name {
            clauses.push("label LIKE ?".into());
            parameters.push(SqlValue::Text(format!("%{name}%")));
        }
        let order = if let Some(near) = query.near {
            if near.len() != 3 {
                return Err(Error::Invalid("near requires exactly X Y Z".into()));
            }
            let radius = query
                .radius
                .ok_or_else(|| Error::Invalid("radius is required with near".into()))?;
            if !radius.is_finite() || radius <= 0.0 {
                return Err(Error::Invalid("radius must be positive and finite".into()));
            }
            let distance = "((x - ?) * (x - ?) + (y - ?) * (y - ?) + (z - ?) * (z - ?))";
            clauses.push(format!("{distance} <= ?"));
            for value in [
                near[0],
                near[0],
                near[1],
                near[1],
                near[2],
                near[2],
                radius * radius,
            ] {
                parameters.push(SqlValue::Real(value));
            }
            for value in [near[0], near[0], near[1], near[1], near[2], near[2]] {
                parameters.push(SqlValue::Real(value));
            }
            format!("{distance} ASC, entity_id ASC")
        } else {
            if query.radius.is_some() {
                return Err(Error::Invalid("near is required with radius".into()));
            }
            "observed_at_ns DESC, entity_id ASC".into()
        };
        parameters.push(SqlValue::Integer(i64::from(query.limit)));
        let sql = format!(
            "SELECT * FROM spatial_entities WHERE {} ORDER BY {order} LIMIT ?",
            clauses.join(" AND ")
        );
        let connection = self.connect()?;
        let mut statement = connection.prepare(&sql)?;
        Ok(statement
            .query_map(params_from_iter(parameters), entity_from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn query_routes(
        &self,
        environment_id: &str,
        world_id: &str,
        from_entity: Option<&str>,
        to_entity: Option<&str>,
        limit: u32,
    ) -> Result<Vec<SpatialRoute>> {
        let mut clauses = vec!["environment_id = ?", "world_id = ?"];
        let mut parameters = vec![
            SqlValue::Text(environment_id.into()),
            SqlValue::Text(world_id.into()),
        ];
        if let Some(value) = from_entity {
            clauses.push("from_entity_id = ?");
            parameters.push(SqlValue::Text(value.into()));
        }
        if let Some(value) = to_entity {
            clauses.push("to_entity_id = ?");
            parameters.push(SqlValue::Text(value.into()));
        }
        parameters.push(SqlValue::Integer(i64::from(limit)));
        let sql = format!(
            "SELECT * FROM spatial_routes WHERE {} ORDER BY confidence DESC, verified_at_ns DESC, route_id ASC LIMIT ?",
            clauses.join(" AND ")
        );
        let connection = self.connect()?;
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(params_from_iter(parameters), route_header_from_row)?;
        let mut routes = Vec::new();
        for row in rows {
            let mut route = row?;
            route.waypoints = read_waypoints(&connection, &route)?;
            routes.push(route);
        }
        Ok(routes)
    }

    pub fn upsert_research_bundle(&self, bundle: &ResearchBundle) -> Result<()> {
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        for source in &bundle.sources {
            transaction.execute(
                "INSERT INTO research_sources(source_id, media_type, accessed_at, source_json) VALUES (?, ?, ?, ?) ON CONFLICT(source_id) DO UPDATE SET media_type=excluded.media_type, accessed_at=excluded.accessed_at, source_json=excluded.source_json",
                params![source.source_id, enum_string(&source.media_type)?, source.accessed_at, compact_json(source)?],
            )?;
        }
        for finding in &bundle.findings {
            transaction.execute(
                "INSERT INTO research_findings(finding_id, category, status, scope, scope_id, finding_json) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(finding_id) DO UPDATE SET category=excluded.category, status=excluded.status, scope=excluded.scope, scope_id=excluded.scope_id, finding_json=excluded.finding_json",
                params![finding.finding_id, enum_string(&finding.category)?, enum_string(&finding.status)?, enum_string(&finding.scope)?, finding.scope_id, compact_json(finding)?],
            )?;
            transaction.execute(
                "DELETE FROM research_finding_sources WHERE finding_id = ?",
                [&finding.finding_id],
            )?;
            for (ordinal, source_id) in finding.source_ids.iter().enumerate() {
                transaction.execute(
                    "INSERT INTO research_finding_sources(finding_id, source_id, ordinal) VALUES (?, ?, ?)",
                    params![finding.finding_id, source_id, ordinal],
                )?;
            }
            transaction.execute(
                "DELETE FROM research_finding_tags WHERE finding_id = ?",
                [&finding.finding_id],
            )?;
            for tag in &finding.tags {
                transaction.execute(
                    "INSERT INTO research_finding_tags(finding_id, tag) VALUES (?, ?)",
                    params![finding.finding_id, tag],
                )?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn query_research(
        &self,
        environment_id: &str,
        environment_family: &str,
        tags: &[String],
        category: Option<&str>,
        verified_only: bool,
        limit: u32,
    ) -> Result<Vec<Value>> {
        let mut clauses = vec![
            "status != ?".to_string(),
            "((scope = ? AND scope_id = ?) OR (scope = ? AND scope_id = ?) OR scope = ?)"
                .to_string(),
        ];
        let mut parameters = vec![
            SqlValue::Text("rejected".into()),
            SqlValue::Text("environment".into()),
            SqlValue::Text(environment_id.into()),
            SqlValue::Text("family".into()),
            SqlValue::Text(environment_family.into()),
            SqlValue::Text("generic".into()),
        ];
        if verified_only {
            clauses.push("status = ?".into());
            parameters.push(SqlValue::Text("runtime-verified".into()));
        }
        if let Some(category) = category {
            clauses.push("category = ?".into());
            parameters.push(SqlValue::Text(category.into()));
        }
        for tag in tags {
            validate_identifier(tag, "research tag")?;
            clauses.push("EXISTS (SELECT 1 FROM research_finding_tags AS tags WHERE tags.finding_id = research_findings.finding_id AND tags.tag = ?)".into());
            parameters.push(SqlValue::Text(tag.clone()));
        }
        parameters.push(SqlValue::Integer(i64::from(limit)));
        let sql = format!(
            "SELECT finding_json FROM research_findings WHERE {} ORDER BY CASE status WHEN 'runtime-verified' THEN 0 ELSE 1 END, finding_id ASC LIMIT ?",
            clauses.join(" AND ")
        );
        let connection = self.connect()?;
        let mut statement = connection.prepare(&sql)?;
        let findings = statement
            .query_map(params_from_iter(parameters), |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        findings
            .into_iter()
            .map(|encoded| {
                let mut finding: Value = serde_json::from_str(&encoded)?;
                let source_ids = finding
                    .get("source_ids")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        Error::Contract("research finding is missing source_ids".into())
                    })?;
                let mut sources = Vec::new();
                for source_id in source_ids {
                    let source_id = source_id
                        .as_str()
                        .ok_or_else(|| Error::Contract("research source_id is not text".into()))?;
                    let source: String = connection
                        .query_row(
                            "SELECT source_json FROM research_sources WHERE source_id = ?",
                            [source_id],
                            |row| row.get(0),
                        )
                        .optional()?
                        .ok_or_else(|| {
                            Error::Contract(format!("missing research source: {source_id}"))
                        })?;
                    sources.push(serde_json::from_str(&source)?);
                }
                let object = finding
                    .as_object_mut()
                    .ok_or_else(|| Error::Contract("research finding is not an object".into()))?;
                object.insert("sources".into(), Value::Array(sources));
                object.insert("action_authority".into(), Value::Bool(false));
                Ok(finding)
            })
            .collect()
    }

    pub fn spatial_bundle(
        &self,
        environment_id: &str,
        protocol_version: &str,
    ) -> Result<SpatialKnowledgeBundle> {
        Ok(SpatialKnowledgeBundle {
            schema_version: "glr.spatial-knowledge.v1".into(),
            environment_id: environment_id.into(),
            protocol_version: protocol_version.into(),
            exported_at_ns: now_ns()?
                .try_into()
                .map_err(|_| Error::Invalid("clock is negative".into()))?,
            entities: self.list_entities(environment_id)?,
            routes: self.list_routes(environment_id)?,
        })
    }

    pub fn import_spatial(
        &self,
        bundle: &SpatialKnowledgeBundle,
        source_run_id: &str,
    ) -> Result<(usize, usize)> {
        for entity in &bundle.entities {
            let mut imported = entity.clone();
            let original_authority = imported.authority.as_str();
            let original_run = imported.source_run_id.clone();
            imported.authority = Authority::Advisory;
            imported.source_run_id = source_run_id.into();
            imported.metadata = merge_metadata(
                &imported.metadata,
                json!({"imported_authority": original_authority, "imported_source_run_id": original_run}),
            );
            self.upsert_entity(&imported)?;
        }
        for route in &bundle.routes {
            let mut imported = route.clone();
            let original_run = imported.source_run_id.clone();
            imported.source_run_id = source_run_id.into();
            imported.metadata = merge_metadata(
                &imported.metadata,
                json!({"imported_source_run_id": original_run, "advisory": true}),
            );
            self.upsert_route(&imported)?;
        }
        Ok((bundle.entities.len(), bundle.routes.len()))
    }

    pub fn import_spatial_graph(
        &self,
        graph: &SpatialKnowledgeGraph,
        source_run_id: &str,
    ) -> Result<SpatialKnowledgeGraph> {
        graph.validate()?;
        validate_identifier(source_run_id, "source_run_id")?;
        let mut imported = graph.clone();
        for node in &mut imported.nodes {
            let original_run = node.source_run_id.clone();
            node.authority = Authority::Advisory;
            node.source_run_id = source_run_id.into();
            node.metadata = merge_metadata(
                &node.metadata,
                json!({"imported_source_run_id": original_run, "advisory": true}),
            );
        }
        for edge in &mut imported.edges {
            let original_run = edge.source_run_id.clone();
            edge.authority = Authority::Advisory;
            edge.source_run_id = source_run_id.into();
            edge.metadata = merge_metadata(
                &edge.metadata,
                json!({"imported_source_run_id": original_run, "advisory": true}),
            );
        }
        imported.validate()?;
        let graph_json = compact_json(&imported)?;
        let graph_id = format!("sha256-{}", sha256_bytes(graph_json.as_bytes()));
        self.connect()?.execute(
            "INSERT INTO spatial_graphs(environment_id, protocol_version, graph_id, exported_at_ns, source_run_id, graph_json) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(environment_id, graph_id) DO NOTHING",
            params![
                imported.environment_id,
                imported.protocol_version,
                graph_id,
                imported.exported_at_ns,
                source_run_id,
                graph_json
            ],
        )?;
        Ok(imported)
    }

    pub fn latest_spatial_graph(
        &self,
        environment_id: &str,
        protocol_version: &str,
    ) -> Result<Option<SpatialKnowledgeGraph>> {
        let encoded: Option<String> = self
            .connect()?
            .query_row(
                "SELECT graph_json FROM spatial_graphs WHERE environment_id = ? AND protocol_version = ? ORDER BY exported_at_ns DESC, graph_id ASC LIMIT 1",
                params![environment_id, protocol_version],
                |row| row.get(0),
            )
            .optional()?;
        encoded
            .map(|value| {
                let graph: SpatialKnowledgeGraph = serde_json::from_str(&value)?;
                graph.validate()?;
                Ok(graph)
            })
            .transpose()
    }

    pub fn upsert_entity(&self, entity: &SpatialEntity) -> Result<()> {
        self.connect()?.execute(
            "INSERT INTO spatial_entities(environment_id, world_id, entity_id, kind, label, x, y, z, coordinate_frame, authority, confidence, observed_at_ns, source_run_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(environment_id, world_id, entity_id) DO UPDATE SET kind=excluded.kind, label=excluded.label, x=excluded.x, y=excluded.y, z=excluded.z, coordinate_frame=excluded.coordinate_frame, authority=excluded.authority, confidence=excluded.confidence, observed_at_ns=excluded.observed_at_ns, source_run_id=excluded.source_run_id, metadata_json=excluded.metadata_json WHERE excluded.observed_at_ns >= spatial_entities.observed_at_ns",
            params![entity.environment_id, entity.world_id, entity.entity_id, entity.kind, entity.label, entity.position[0], entity.position[1], entity.position[2], entity.coordinate_frame, entity.authority.as_str(), entity.confidence, entity.observed_at_ns, entity.source_run_id, compact_json(&entity.metadata)?],
        )?;
        Ok(())
    }

    pub fn upsert_route(&self, route: &SpatialRoute) -> Result<()> {
        let mut connection = self.connect()?;
        let transaction = connection.transaction()?;
        transaction.execute(
            "INSERT INTO spatial_routes(environment_id, world_id, route_id, name, from_entity_id, to_entity_id, coordinate_frame, confidence, verified_at_ns, source_run_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(environment_id, world_id, route_id) DO UPDATE SET name=excluded.name, from_entity_id=excluded.from_entity_id, to_entity_id=excluded.to_entity_id, coordinate_frame=excluded.coordinate_frame, confidence=excluded.confidence, verified_at_ns=excluded.verified_at_ns, source_run_id=excluded.source_run_id, metadata_json=excluded.metadata_json WHERE excluded.verified_at_ns >= spatial_routes.verified_at_ns",
            params![route.environment_id, route.world_id, route.route_id, route.name, route.from_entity_id, route.to_entity_id, route.coordinate_frame, route.confidence, route.verified_at_ns, route.source_run_id, compact_json(&route.metadata)?],
        )?;
        transaction.execute(
            "DELETE FROM route_waypoints WHERE environment_id = ? AND world_id = ? AND route_id = ?",
            params![route.environment_id, route.world_id, route.route_id],
        )?;
        for waypoint in &route.waypoints {
            transaction.execute(
                "INSERT INTO route_waypoints(environment_id, world_id, route_id, waypoint_index, x, y, z, tolerance, label) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params![route.environment_id, route.world_id, route.route_id, waypoint.index, waypoint.position[0], waypoint.position[1], waypoint.position[2], waypoint.tolerance, waypoint.label],
            )?;
        }
        transaction.commit()?;
        Ok(())
    }

    fn list_entities(&self, environment_id: &str) -> Result<Vec<SpatialEntity>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT * FROM spatial_entities WHERE environment_id = ? ORDER BY world_id ASC, entity_id ASC",
        )?;
        Ok(statement
            .query_map([environment_id], entity_from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?)
    }

    fn list_routes(&self, environment_id: &str) -> Result<Vec<SpatialRoute>> {
        let connection = self.connect()?;
        let mut statement = connection.prepare(
            "SELECT * FROM spatial_routes WHERE environment_id = ? ORDER BY world_id ASC, route_id ASC",
        )?;
        let rows = statement.query_map([environment_id], route_header_from_row)?;
        let mut routes = Vec::new();
        for row in rows {
            let mut route = row?;
            route.waypoints = read_waypoints(&connection, &route)?;
            routes.push(route);
        }
        Ok(routes)
    }
}

fn transaction_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TransactionRecord> {
    let steps_json: String = row.get("steps_json")?;
    let steps: Vec<Value> = serde_json::from_str(&steps_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(2, Type::Text, Box::new(error))
    })?;
    let last_refusal = row
        .get::<_, Option<String>>("last_refusal_json")?
        .map(|value| {
            serde_json::from_str(&value).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(7, Type::Text, Box::new(error))
            })
        })
        .transpose()?;
    Ok(TransactionRecord {
        transaction_id: row.get("transaction_id")?,
        run_id: row.get("run_id")?,
        step_count: u32::try_from(steps.len()).map_err(|_| {
            rusqlite::Error::FromSqlConversionFailure(
                2,
                Type::Text,
                "transaction step count overflows u32".into(),
            )
        })?,
        next_step_index: row.get("next_step_index")?,
        status: row.get("status")?,
        resume_attempts: row.get("resume_attempts")?,
        max_resume_attempts: row.get("max_resume_attempts")?,
        last_refusal,
        updated_at_ns: row.get("updated_at_ns")?,
    })
}

fn validate_transaction_refusal(refusal: &TransactionRefusal) -> Result<()> {
    validate_identifier(&refusal.action_id, "transaction refusal action_id")?;
    validate_identifier(&refusal.target_id, "transaction refusal target_id")?;
    if !matches!(refusal.reason_class.as_str(), "transient" | "structural") {
        return Err(Error::Invalid(
            "transaction refusal reason_class must be 'transient' or 'structural'".into(),
        ));
    }
    if refusal.message.is_empty() || refusal.message.len() > 512 {
        return Err(Error::Invalid(
            "transaction refusal message must contain 1-512 characters".into(),
        ));
    }
    Ok(())
}

fn append_event_transaction(
    transaction: &rusqlite::Transaction<'_>,
    run_id: &str,
    kind: &str,
    payload: Value,
) -> Result<()> {
    validate_identifier(kind, "event kind")?;
    let sequence: i64 = transaction.query_row(
        "SELECT COALESCE(MAX(sequence_id), 0) + 1 FROM events WHERE run_id = ?",
        [run_id],
        |row| row.get(0),
    )?;
    transaction.execute(
        "INSERT INTO events(run_id, sequence_id, timestamp_ns, kind, episode_id, step_id, payload_json) VALUES (?, ?, ?, ?, NULL, NULL, ?)",
        params![run_id, sequence, now_ns()?, kind, compact_json(&payload)?],
    )?;
    Ok(())
}

fn run_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RunRecord> {
    Ok(RunRecord {
        run_id: row.get("run_id")?,
        environment_id: row.get("environment_id")?,
        protocol_version: row.get("protocol_version")?,
        kind: row.get("kind")?,
        status: row.get("status")?,
        started_at_ns: row.get("started_at_ns")?,
        finished_at_ns: row.get("finished_at_ns")?,
        exit_code: row.get("exit_code")?,
        metadata: parse_json_row(row.get::<_, String>("metadata_json")?)?,
    })
}

fn entity_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SpatialEntity> {
    let authority: String = row.get("authority")?;
    Ok(SpatialEntity {
        environment_id: row.get("environment_id")?,
        world_id: row.get("world_id")?,
        entity_id: row.get("entity_id")?,
        kind: row.get("kind")?,
        label: row.get("label")?,
        position: [row.get("x")?, row.get("y")?, row.get("z")?],
        coordinate_frame: row.get("coordinate_frame")?,
        authority: if authority == "authoritative" {
            Authority::Authoritative
        } else {
            Authority::Advisory
        },
        confidence: row.get("confidence")?,
        observed_at_ns: row.get("observed_at_ns")?,
        source_run_id: row.get("source_run_id")?,
        metadata: parse_json_row(row.get::<_, String>("metadata_json")?)?,
    })
}

fn route_header_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SpatialRoute> {
    Ok(SpatialRoute {
        environment_id: row.get("environment_id")?,
        world_id: row.get("world_id")?,
        route_id: row.get("route_id")?,
        name: row.get("name")?,
        from_entity_id: row.get("from_entity_id")?,
        to_entity_id: row.get("to_entity_id")?,
        coordinate_frame: row.get("coordinate_frame")?,
        confidence: row.get("confidence")?,
        verified_at_ns: row.get("verified_at_ns")?,
        source_run_id: row.get("source_run_id")?,
        waypoints: Vec::new(),
        metadata: parse_json_row(row.get::<_, String>("metadata_json")?)?,
    })
}

fn read_waypoints(connection: &Connection, route: &SpatialRoute) -> Result<Vec<RouteWaypoint>> {
    let mut statement = connection.prepare(
        "SELECT * FROM route_waypoints WHERE environment_id = ? AND world_id = ? AND route_id = ? ORDER BY waypoint_index ASC",
    )?;
    Ok(statement
        .query_map(
            params![route.environment_id, route.world_id, route.route_id],
            |row| {
                Ok(RouteWaypoint {
                    index: row.get("waypoint_index")?,
                    position: [row.get("x")?, row.get("y")?, row.get("z")?],
                    tolerance: row.get("tolerance")?,
                    label: row.get("label")?,
                })
            },
        )?
        .collect::<std::result::Result<Vec<_>, _>>()?)
}

fn compact_json<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    Ok(serde_json::to_string(value)?)
}

fn parse_json_row(value: String) -> rusqlite::Result<Value> {
    serde_json::from_str(&value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            value.len(),
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn legacy_checkpoint_promotion_cannot_install_without_host_authorization() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        let live = temp.path().join("live.checkpoint");
        let candidate = temp.path().join("candidate.checkpoint");
        fs::write(&live, b"incumbent").unwrap();
        fs::write(&candidate, b"candidate").unwrap();
        assert!(
            store
                .promote_checkpoint(CheckpointPromotionRequest {
                    goal_id: "goal.demo",
                    metric: "victories",
                    mode: PromotionMode::Max,
                    value: 100.0,
                    run_id: &run.run_id,
                    trial_id: "trial-1",
                    candidate: &candidate,
                    live: &live,
                })
                .is_err()
        );
        assert_eq!(fs::read(live).unwrap(), b"incumbent");
        assert_eq!(fs::read(candidate).unwrap(), b"candidate");
        let count: i64 = store
            .connect()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM checkpoint_promotions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn structural_refusal_is_bounded_and_reported_without_implicit_retry() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        let steps = vec![
            json!({"action_id": "move-1"}),
            json!({"action_id": "move-2"}),
        ];
        let started = store
            .begin_transaction(&run.run_id, "txn.demo", &steps, 2)
            .unwrap();
        assert_eq!(started.status, "pending");
        assert_eq!(started.next_step_index, 0);

        let refusal = TransactionRefusal {
            action_id: "move-1".into(),
            target_id: "card-1".into(),
            reason_class: "structural".into(),
            message: "postcondition failed".into(),
            retryable: false,
        };
        let first = store
            .resume_transaction("txn.demo", Some(&refusal))
            .unwrap();
        assert_eq!(first.outcome, "refused");
        assert_eq!(first.transaction.resume_attempts, 1);
        assert_eq!(first.transaction.next_step_index, 0);

        let second = store
            .resume_transaction("txn.demo", Some(&refusal))
            .unwrap();
        assert_eq!(second.outcome, "abandoned");
        assert_eq!(second.transaction.status, "abandoned");
        assert_eq!(second.transaction.resume_attempts, 2);
        assert_eq!(
            store.get_transaction("txn.demo").unwrap().status,
            "abandoned"
        );
        assert!(
            store
                .list_events(&run.run_id)
                .unwrap()
                .iter()
                .any(|event| event.kind == "transaction.abandoned")
        );
        assert_eq!(
            store.resume_transaction("txn.demo", None).unwrap().outcome,
            "already_terminal"
        );
    }

    #[test]
    fn accepted_transaction_steps_advance_explicitly_and_complete() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        store
            .begin_transaction(
                &run.run_id,
                "txn.accepted",
                &[
                    json!({"action_id": "move-1"}),
                    json!({"action_id": "move-2"}),
                ],
                2,
            )
            .unwrap();
        assert_eq!(
            store
                .resume_transaction("txn.accepted", None)
                .unwrap()
                .outcome,
            "advanced"
        );
        let completed = store.resume_transaction("txn.accepted", None).unwrap();
        assert_eq!(completed.outcome, "completed");
        assert_eq!(completed.transaction.next_step_index, 2);
    }
}

fn enum_string<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_value(value)?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| Error::Invalid("enum did not serialize as text".into()))
}

pub(crate) fn now_ns() -> Result<i64> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::Invalid("system clock is before the Unix epoch".into()))?
        .as_nanos();
    i64::try_from(nanos).map_err(|_| Error::Invalid("system clock exceeds SQLite range".into()))
}

fn validate_portable_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains('\\')
        || value.contains(':')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        Err(Error::Invalid(
            "artifact path must be a portable relative path".into(),
        ))
    } else {
        Ok(())
    }
}

fn merge_metadata(original: &Value, additions: Value) -> Value {
    let mut merged = original.as_object().cloned().unwrap_or_default();
    if let Some(additions) = additions.as_object() {
        merged.extend(additions.clone());
    }
    Value::Object(merged)
}

fn sha256_bytes(value: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(value);
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod promotion_tests {
    use super::*;
    fn test_binding(
        store: &Path,
        environment: &str,
        goal: &str,
        live: &Path,
    ) -> Result<CheckpointBinding> {
        let config = store.parent().unwrap().join("journal-fixture-config.json");
        if !config.exists() {
            fs::write(
                &config,
                serde_json::to_vec(
                    &json!({"environment_id":environment,"protocol_version":"1.0","target_id":"fixture-target","evaluator_files":[]}),
                )?,
            )?;
        }
        let connection = Connection::open(store)?;
        let fingerprint = format!("{:x}", Sha256::digest([31u8; 32]));
        connection.execute("INSERT OR IGNORE INTO promotion_host_authority(singleton,fingerprint,epoch) VALUES (1,?,?)",params![fingerprint,"11111111111111111111111111111111"])?;
        let (fingerprint, epoch): (String, String) = connection.query_row(
            "SELECT fingerprint,epoch FROM promotion_host_authority WHERE singleton=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        CheckpointBinding::new(
            store,
            environment,
            goal,
            live,
            "1.0",
            "fixture-target",
            &sha256_file(&config)?,
            &fingerprint,
            &epoch,
        )
    }

    fn journal_fixture() -> (tempfile::TempDir, Store, String, CheckpointBinding) {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        let candidate = temp.path().join("incumbent.candidate");
        let live = temp.path().join("best.checkpoint");
        fs::write(&candidate, b"incumbent").unwrap();
        store
            .test_promote_checkpoint(CheckpointPromotionRequest {
                goal_id: "goal.journal",
                metric: "victories",
                mode: PromotionMode::Max,
                value: 2.0,
                run_id: &run.run_id,
                trial_id: "trial-1",
                candidate: &candidate,
                live: &live,
            })
            .unwrap();
        fs::write(temp.path().join("new.candidate"), b"candidate").unwrap();
        let binding =
            test_binding(&store.path, "example.environment-v1", "goal.journal", &live).unwrap();
        (temp, store, run.run_id, binding)
    }

    fn crash_promotion(temp: &Path, run_id: &str, phase: &str) {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "store::promotion_tests::checkpoint_promotion_crash_child",
            ])
            .env("JOURNAL_TEST_ROOT", temp)
            .env("JOURNAL_TEST_RUN", run_id)
            .env("JOURNAL_TEST_PHASE", phase)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(86),
            "phase {phase}\nstdout {}\nstderr {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    #[ignore]
    fn checkpoint_promotion_crash_child() {
        let root = PathBuf::from(std::env::var_os("JOURNAL_TEST_ROOT").unwrap());
        let run_id = std::env::var("JOURNAL_TEST_RUN").unwrap();
        let phase = std::env::var("JOURNAL_TEST_PHASE").unwrap();
        let store = Store::open(root.join("runs.sqlite3")).unwrap();
        let request = CheckpointPromotionRequest {
            goal_id: "goal.journal",
            metric: "victories",
            mode: PromotionMode::Max,
            value: 4.0,
            run_id: &run_id,
            trial_id: "trial-crash",
            candidate: &root.join("new.candidate"),
            live: &root.join("best.checkpoint"),
        };
        let authorization = crate::promotion_host::fixture_authorization(&store, &request).unwrap();
        store
            .promote_checkpoint_inner(request, &authorization, |observed| {
                if format!("{observed:?}") == phase {
                    // std::process::exit deliberately bypasses all Rust Drop,
                    // transaction rollback, and tempfile cleanup in this fixture.
                    std::process::exit(86);
                }
                Ok(())
            })
            .unwrap();
        panic!("unknown fault phase");
    }

    #[test]
    fn checkpoint_promotion_recovers_each_abrupt_process_exit_boundary() {
        for phase in [
            "Staged",
            "Journaled",
            "SqlWritten",
            "Replaced",
            "BeforeCommit",
            "Committed",
        ] {
            let (temp, _store, run_id, binding) = journal_fixture();
            crash_promotion(temp.path(), &run_id, phase);
            let recovered = Store::open(temp.path().join("runs.sqlite3")).unwrap();
            let record = Store::checkpoint_record(&recovered.connect().unwrap(), "goal.journal")
                .unwrap()
                .unwrap();
            let committed = phase == "Committed";
            assert_eq!(
                record.best_metric,
                if committed { 4.0 } else { 2.0 },
                "{phase}"
            );
            assert_eq!(
                fs::read(&binding.live_path).unwrap(),
                if committed {
                    b"candidate"
                } else {
                    b"incumbent"
                },
                "{phase}"
            );
            assert_eq!(
                sha256_file(Path::new(&binding.live_path)).unwrap(),
                record.checkpoint_sha256
            );
            assert!(!binding.pending().exists(), "{phase}");
            assert_eq!(
                fs::read(temp.path().join("new.candidate")).unwrap(),
                b"candidate"
            );
            let old_digest = sha256_file(&temp.path().join("incumbent.candidate")).unwrap();
            assert_eq!(fs::read(binding.blob(&old_digest)).unwrap(), b"incumbent");
            // A second restart must not replay or reverse the completed decision.
            Store::open(temp.path().join("runs.sqlite3")).unwrap();
        }
    }

    #[test]
    fn checkpoint_promotion_first_attempt_rolls_back_without_deleting_candidate_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        fs::write(temp.path().join("new.candidate"), b"candidate").unwrap();
        crash_promotion(temp.path(), &run.run_id, "Replaced");
        let recovered = Store::open(store.path.clone()).unwrap();
        assert!(!temp.path().join("best.checkpoint").exists());
        assert!(
            Store::checkpoint_record(&recovered.connect().unwrap(), "goal.journal")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            fs::read(temp.path().join("new.candidate")).unwrap(),
            b"candidate"
        );
        let binding = test_binding(
            &store.path,
            "example.environment-v1",
            "goal.journal",
            &temp.path().join("best.checkpoint"),
        )
        .unwrap();
        let abandoned = fs::read_dir(binding.directory())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "abandoned")
            })
            .unwrap();
        assert_eq!(fs::read(abandoned).unwrap(), b"candidate");
    }

    #[test]
    fn checkpoint_promotion_recovery_reinstalls_missing_live_from_verified_blobs() {
        for phase in ["Replaced", "Committed"] {
            let (temp, store, run_id, binding) = journal_fixture();
            crash_promotion(temp.path(), &run_id, phase);
            // Only this fixture's replaceable live projection is removed;
            // both immutable models and original candidates remain present.
            fs::remove_file(&binding.live_path).unwrap();
            Store::open(store.path).unwrap();
            assert_eq!(
                fs::read(&binding.live_path).unwrap(),
                if phase == "Committed" {
                    b"candidate"
                } else {
                    b"incumbent"
                }
            );
        }
    }

    #[test]
    fn checkpoint_promotion_recovery_refuses_unknown_bindings_hashes_and_database_states() {
        for tamper in [
            "environment",
            "schema",
            "blob",
            "live",
            "database",
            "run-environment",
        ] {
            let (temp, store, run_id, binding) = journal_fixture();
            crash_promotion(temp.path(), &run_id, "Replaced");
            let mut journal: Value =
                serde_json::from_slice(&fs::read(binding.pending()).unwrap()).unwrap();
            match tamper {
                "environment" => {
                    journal["binding"]["environment_id"] = json!("different.environment");
                    fs::write(binding.pending(), serde_json::to_vec(&journal).unwrap()).unwrap();
                }
                "schema" => {
                    journal["schema_version"] = json!("unknown.v9");
                    fs::write(binding.pending(), serde_json::to_vec(&journal).unwrap()).unwrap();
                }
                "blob" => {
                    let digest = journal["incumbent_sha256"].as_str().unwrap();
                    fs::write(binding.blob(digest), b"corrupt").unwrap();
                }
                "live" => fs::write(&binding.live_path, b"unknown bytes").unwrap(),
                "database" => {
                    store.connect().unwrap().execute("UPDATE checkpoint_promotions SET best_metric = 999 WHERE goal_id = 'goal.journal'", []).unwrap();
                }
                "run-environment" => {
                    store.connect().unwrap().execute("UPDATE runs SET environment_id = 'different.environment' WHERE run_id = ?", [&run_id]).unwrap();
                }
                _ => unreachable!(),
            }
            let before = fs::read(&binding.live_path).unwrap();
            assert!(
                Store::open(temp.path().join("runs.sqlite3")).is_err(),
                "{tamper}"
            );
            assert_eq!(fs::read(&binding.live_path).unwrap(), before, "{tamper}");
            assert!(binding.pending().is_file(), "{tamper}");
            assert_eq!(
                fs::read(temp.path().join("incumbent.candidate")).unwrap(),
                b"incumbent"
            );
            assert_eq!(
                fs::read(temp.path().join("new.candidate")).unwrap(),
                b"candidate"
            );
        }
    }

    #[test]
    fn checkpoint_promotion_copied_store_cannot_recover_another_store_location() {
        let (temp, store, run_id, binding) = journal_fixture();
        crash_promotion(temp.path(), &run_id, "Replaced");
        let copy = temp.path().join("copied.sqlite3");
        fs::copy(&store.path, &copy).unwrap();
        assert!(Store::open(copy).is_err());
        assert!(binding.pending().exists());
        assert_eq!(fs::read(&binding.live_path).unwrap(), b"candidate");
        Store::open(store.path).unwrap();
        assert_eq!(fs::read(&binding.live_path).unwrap(), b"incumbent");
    }

    #[test]
    fn checkpoint_promotion_never_steals_an_active_sqlite_worker_lock() {
        let (temp, store, run_id, binding) = journal_fixture();
        let mut connection = store.connect().unwrap();
        let transaction = connection
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .unwrap();
        let error = store
            .test_promote_checkpoint(CheckpointPromotionRequest {
                goal_id: "goal.journal",
                metric: "victories",
                mode: PromotionMode::Max,
                value: 4.0,
                run_id: &run_id,
                trial_id: "trial-lock",
                candidate: &temp.path().join("new.candidate"),
                live: Path::new(&binding.live_path),
            })
            .unwrap_err();
        assert!(error.to_string().contains("locked"), "{error}");
        assert_eq!(fs::read(&binding.live_path).unwrap(), b"incumbent");
        assert!(!binding.pending().exists());
        assert_eq!(
            Store::checkpoint_record(&transaction, "goal.journal")
                .unwrap()
                .unwrap()
                .best_metric,
            2.0
        );
        transaction.rollback().unwrap();
    }

    #[test]
    fn checkpoint_promotion_different_store_cannot_claim_the_same_live_path() {
        let (temp, _store, _run_id, binding) = journal_fixture();
        let foreign = Store::open(temp.path().join("foreign.sqlite3")).unwrap();
        let run = foreign
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        let error = foreign
            .test_promote_checkpoint(CheckpointPromotionRequest {
                goal_id: "goal.journal",
                metric: "victories",
                mode: PromotionMode::Max,
                value: 99.0,
                run_id: &run.run_id,
                trial_id: "trial-foreign",
                candidate: &temp.path().join("new.candidate"),
                live: Path::new(&binding.live_path),
            })
            .unwrap_err();
        assert!(
            error.to_string().contains("belongs to another store"),
            "{error}"
        );
        assert_eq!(fs::read(&binding.live_path).unwrap(), b"incumbent");
        assert!(
            Store::checkpoint_record(&foreign.connect().unwrap(), "goal.journal")
                .unwrap()
                .is_none()
        );
        crate::promotion_journal::verify_owner(&binding).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn checkpoint_promotion_case_aliases_share_ownership_before_live_exists() {
        let temp = tempfile::tempdir().unwrap();
        let first = Store::open(temp.path().join("first.sqlite3")).unwrap();
        let second = Store::open(temp.path().join("second.sqlite3")).unwrap();
        let binding = test_binding(
            &first.path,
            "example.environment-v1",
            "goal.journal",
            &temp.path().join("best.checkpoint"),
        )
        .unwrap();
        let alias = test_binding(
            &second.path,
            "example.environment-v1",
            "goal.journal",
            &temp.path().join("BEST.CHECKPOINT"),
        )
        .unwrap();
        assert_eq!(binding.directory(), alias.directory());
        crate::promotion_journal::claim_location(&binding).unwrap();
        assert!(crate::promotion_journal::claim_location(&alias).is_err());
        assert!(!temp.path().join("best.checkpoint").exists());
    }

    #[test]
    fn checkpoint_promotion_owner_tampering_fails_closed_without_changing_bytes() {
        let (temp, store, _run_id, binding) = journal_fixture();
        let mut owner: Value = serde_json::from_slice(&fs::read(binding.owner()).unwrap()).unwrap();
        owner["binding"]["goal_id"] = json!("another.goal");
        fs::write(binding.owner(), serde_json::to_vec(&owner).unwrap()).unwrap();
        assert!(Store::open(store.path).is_err());
        assert_eq!(fs::read(&binding.live_path).unwrap(), b"incumbent");
        assert!(binding.owner().exists());
        assert_eq!(
            fs::read(temp.path().join("incumbent.candidate")).unwrap(),
            b"incumbent"
        );
    }

    #[test]
    fn promotion_metric_is_bound_to_authoritative_persisted_trial_evidence() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        store
            .append_metric(
                &run.run_id,
                "victories",
                3.0,
                None,
                json!({"source": "referee", "authority": "authoritative"}),
            )
            .unwrap();
        let floor = store.latest_metric_id(&run.run_id).unwrap();
        store
            .append_metric(
                &run.run_id,
                "victories",
                4.0,
                None,
                json!({"source": "referee", "authority": "authoritative"}),
            )
            .unwrap();
        for (source, authority, value) in [
            ("trainer", "authoritative", 99.0),
            ("referee", "advisory", 999.0),
        ] {
            store
                .append_metric(
                    &run.run_id,
                    "victories",
                    value,
                    None,
                    json!({"source": source, "authority": authority}),
                )
                .unwrap();
        }
        let evidence = GoalEvidence {
            metric: "victories".into(),
            value: 4.0,
            source: "referee".into(),
            authority: Authority::Authoritative,
            run_id: run.run_id.clone(),
        };
        assert_eq!(
            store
                .promotion_metric_value(&run.run_id, floor, &evidence)
                .unwrap(),
            4.0
        );
        for invalid in [
            GoalEvidence {
                value: 3.0,
                ..evidence.clone()
            },
            GoalEvidence {
                value: 999.0,
                ..evidence.clone()
            },
            GoalEvidence {
                authority: Authority::Advisory,
                ..evidence.clone()
            },
            GoalEvidence {
                run_id: "different-run".into(),
                ..evidence.clone()
            },
        ] {
            assert!(
                store
                    .promotion_metric_value(&run.run_id, floor, &invalid)
                    .is_err()
            );
        }
    }

    #[test]
    fn checkpoint_promotion_database_failure_preserves_incumbent_and_candidate() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        let live = temp.path().join("best.checkpoint");
        let candidate = temp.path().join("candidate.checkpoint");
        fs::write(&live, b"incumbent").unwrap();
        fs::write(&candidate, b"candidate").unwrap();
        store.connect().unwrap().execute_batch(
            "CREATE TRIGGER reject_promotion BEFORE INSERT ON checkpoint_promotions BEGIN SELECT RAISE(ABORT, 'test refusal'); END;",
        ).unwrap();
        assert!(
            store
                .test_promote_checkpoint(CheckpointPromotionRequest {
                    goal_id: "goal.demo",
                    metric: "victories",
                    mode: PromotionMode::Max,
                    value: 4.0,
                    run_id: &run.run_id,
                    trial_id: "trial-1",
                    candidate: &candidate,
                    live: &live,
                })
                .is_err()
        );
        assert_eq!(fs::read(&live).unwrap(), b"incumbent");
        assert_eq!(fs::read(&candidate).unwrap(), b"candidate");
        let count: i64 = store
            .connect()
            .unwrap()
            .query_row("SELECT COUNT(*) FROM checkpoint_promotions", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn checkpoint_promotion_keeps_the_best_bytes_and_retains_candidates() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.environment-v1", "1.0", "goal", json!({}))
            .unwrap();
        let live = temp.path().join("checkpoints/policy.checkpoint");
        let first = temp.path().join("trial-1.checkpoint");
        fs::write(&first, b"first").unwrap();
        let (promoted, record) = store
            .test_promote_checkpoint(CheckpointPromotionRequest {
                goal_id: "goal.demo",
                metric: "victories",
                mode: PromotionMode::Max,
                value: 3.0,
                run_id: &run.run_id,
                trial_id: "trial-1",
                candidate: &first,
                live: &live,
            })
            .unwrap();
        assert!(promoted);
        assert_eq!(record.best_metric, 3.0);
        assert_eq!(record.run_id, run.run_id);
        assert_eq!(record.trial_id, "trial-1");
        assert_eq!(fs::read(&live).unwrap(), b"first");

        let regression = temp.path().join("trial-2.checkpoint");
        fs::write(&regression, b"regression").unwrap();
        let (promoted, record) = store
            .test_promote_checkpoint(CheckpointPromotionRequest {
                goal_id: "goal.demo",
                metric: "victories",
                mode: PromotionMode::Max,
                value: 2.0,
                run_id: &run.run_id,
                trial_id: "trial-2",
                candidate: &regression,
                live: &live,
            })
            .unwrap();
        assert!(!promoted);
        assert_eq!(record.best_metric, 3.0);
        assert_eq!(record.run_id, run.run_id);
        assert_eq!(record.trial_id, "trial-1");
        assert_eq!(fs::read(&live).unwrap(), b"first");
        assert_eq!(fs::read(&regression).unwrap(), b"regression");

        let improvement = temp.path().join("trial-4.checkpoint");
        fs::write(&improvement, b"improvement").unwrap();
        let (promoted, record) = store
            .test_promote_checkpoint(CheckpointPromotionRequest {
                goal_id: "goal.demo",
                metric: "victories",
                mode: PromotionMode::Max,
                value: 4.0,
                run_id: &run.run_id,
                trial_id: "trial-4",
                candidate: &improvement,
                live: &live,
            })
            .unwrap();
        assert!(promoted);
        assert_eq!(record.best_metric, 4.0);
        assert_eq!(record.trial_id, "trial-4");
        assert_eq!(fs::read(&live).unwrap(), b"improvement");

        let tie = temp.path().join("trial-3.checkpoint");
        fs::write(&tie, b"tie").unwrap();
        let (promoted, _) = store
            .test_promote_checkpoint(CheckpointPromotionRequest {
                goal_id: "goal.demo",
                metric: "victories",
                mode: PromotionMode::Max,
                value: 3.0,
                run_id: &run.run_id,
                trial_id: "trial-3",
                candidate: &tie,
                live: &live,
            })
            .unwrap();
        assert!(!promoted);
        assert_eq!(fs::read(&live).unwrap(), b"improvement");
    }
}

#[cfg(test)]
impl Store {
    fn test_promote_checkpoint(
        &self,
        request: CheckpointPromotionRequest<'_>,
    ) -> Result<(bool, CheckpointPromotionRecord)> {
        if let Some(parent) = request.live.parent() {
            fs::create_dir_all(parent)?;
        }
        let authorization = crate::promotion_host::fixture_authorization(self, &request)?;
        self.install_authorized_checkpoint(request, &authorization)
    }
}
