use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::types::Value as SqlValue;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::contracts::{
    Authority, GoalEvidence, PromotionMode, ResearchBundle, RouteWaypoint, SpatialEntity,
    SpatialKnowledgeBundle, SpatialRoute, sha256_file,
};
use crate::error::{Error, Result};
use crate::project::validate_identifier;

pub const RUN_STORE_SCHEMA_VERSION: i64 = 1;

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

#[derive(Debug, Clone, Serialize)]
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
}

impl Store {
    pub fn open(path: PathBuf) -> Result<Self> {
        if path.is_symlink() {
            return Err(Error::Invalid("run store cannot be a symlink".into()));
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let store = Self { path };
        store.initialize()?;
        Ok(store)
    }

    fn connect(&self) -> Result<Connection> {
        let connection = Connection::open(&self.path)?;
        connection.busy_timeout(Duration::from_secs(30))?;
        connection.execute_batch("PRAGMA foreign_keys = ON;")?;
        Ok(connection)
    }

    fn initialize(&self) -> Result<()> {
        let connection = self.connect()?;
        let version: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if !matches!(version, 0 | RUN_STORE_SCHEMA_VERSION) {
            return Err(Error::Contract(format!(
                "unsupported run store schema version: {version}"
            )));
        }
        connection.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
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
            PRAGMA user_version = 1;
            "#,
        )?;
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

    pub fn get_run(&self, run_id: &str) -> Result<RunRecord> {
        self.connect()?
            .query_row(
                "SELECT * FROM runs WHERE run_id = ?",
                [run_id],
                run_from_row,
            )
            .optional()?
            .ok_or_else(|| Error::Contract(format!("unknown run_id: {run_id}")))
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
        let connection = self.connect()?;
        let mut statement = connection
            .prepare("SELECT * FROM events WHERE run_id = ? ORDER BY sequence_id ASC LIMIT 1000")?;
        Ok(statement
            .query_map([run_id], |row| {
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
        let connection = self.connect()?;
        let mut statement = connection
            .prepare("SELECT * FROM metrics WHERE run_id = ? ORDER BY metric_id ASC LIMIT 1000")?;
        Ok(statement
            .query_map([run_id], |row| {
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

    pub fn latest_metric_value(
        &self,
        run_id: &str,
        name: &str,
        after_metric_id: i64,
    ) -> Result<Option<f64>> {
        self.connect()?
            .query_row(
                "SELECT value FROM metrics WHERE run_id = ? AND metric_id > ? AND name = ? ORDER BY metric_id DESC LIMIT 1",
                params![run_id, after_metric_id, name],
                |row| row.get(0),
            )
            .optional()
            .map_err(Error::from)
    }

    pub fn promote_checkpoint(
        &self,
        goal_id: &str,
        metric: &str,
        mode: PromotionMode,
        value: f64,
        run_id: &str,
        trial_id: &str,
        candidate: &Path,
        live: &Path,
    ) -> Result<(bool, CheckpointPromotionRecord)> {
        validate_identifier(goal_id, "goal_id")?;
        validate_identifier(metric, "checkpoint promotion metric")?;
        if !value.is_finite() {
            return Err(Error::Invalid(
                "checkpoint promotion metric must be finite".into(),
            ));
        }
        if candidate.is_symlink() || !candidate.is_file() {
            return Err(Error::Missing(candidate.to_path_buf()));
        }
        if live.is_symlink() {
            return Err(Error::Invalid("live checkpoint cannot be a symlink".into()));
        }
        if let Some(parent) = live.parent() {
            fs::create_dir_all(parent)?;
        }
        let connection = self.connect()?;
        let previous: Option<(String, f64, String)> = connection
            .query_row(
                "SELECT mode, best_metric, checkpoint_sha256 FROM checkpoint_promotions WHERE goal_id = ?",
                [goal_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let mode_name = match mode {
            PromotionMode::Max => "max",
            PromotionMode::Min => "min",
        };
        let improved = previous.as_ref().is_none_or(|(_, best, _)| match mode {
            PromotionMode::Max => value > *best,
            PromotionMode::Min => value < *best,
        });
        let digest = sha256_file(candidate)?;
        if improved {
            let temporary = live.with_extension(format!("tmp-{}", uuid::Uuid::new_v4().simple()));
            fs::copy(candidate, &temporary)?;
            if live.exists() {
                fs::remove_file(live)?;
            }
            fs::rename(&temporary, live)?;
            let updated_at_ns = now_ns()?;
            connection.execute(
                "INSERT INTO checkpoint_promotions(goal_id, metric, mode, best_metric, checkpoint_sha256, checkpoint_path, run_id, trial_id, updated_at_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(goal_id) DO UPDATE SET metric=excluded.metric, mode=excluded.mode, best_metric=excluded.best_metric, checkpoint_sha256=excluded.checkpoint_sha256, checkpoint_path=excluded.checkpoint_path, run_id=excluded.run_id, trial_id=excluded.trial_id, updated_at_ns=excluded.updated_at_ns",
                params![goal_id, metric, mode_name, value, digest, live.to_string_lossy(), run_id, trial_id, updated_at_ns],
            )?;
        }
        let record = if improved {
            CheckpointPromotionRecord {
                goal_id: goal_id.into(),
                metric: metric.into(),
                mode: mode_name.into(),
                best_metric: value,
                checkpoint_sha256: digest,
                checkpoint_path: live.to_string_lossy().into_owned(),
                run_id: run_id.into(),
                trial_id: trial_id.into(),
                updated_at_ns: now_ns()?,
            }
        } else {
            let (stored_mode, best_metric, checkpoint_sha256) = previous
                .ok_or_else(|| Error::Contract("checkpoint promotion state disappeared".into()))?;
            CheckpointPromotionRecord {
                goal_id: goal_id.into(),
                metric: metric.into(),
                mode: stored_mode,
                best_metric,
                checkpoint_sha256,
                checkpoint_path: live.to_string_lossy().into_owned(),
                run_id: run_id.into(),
                trial_id: trial_id.into(),
                updated_at_ns: now_ns()?,
            }
        };
        Ok((improved, record))
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
mod tests {
    use super::*;

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
            .promote_checkpoint(
                "goal.demo",
                "victories",
                PromotionMode::Max,
                3.0,
                &run.run_id,
                "trial-1",
                &first,
                &live,
            )
            .unwrap();
        assert!(promoted);
        assert_eq!(record.best_metric, 3.0);
        assert_eq!(fs::read(&live).unwrap(), b"first");

        let regression = temp.path().join("trial-2.checkpoint");
        fs::write(&regression, b"regression").unwrap();
        let (promoted, record) = store
            .promote_checkpoint(
                "goal.demo",
                "victories",
                PromotionMode::Max,
                2.0,
                &run.run_id,
                "trial-2",
                &regression,
                &live,
            )
            .unwrap();
        assert!(!promoted);
        assert_eq!(record.best_metric, 3.0);
        assert_eq!(fs::read(&live).unwrap(), b"first");
        assert_eq!(fs::read(&regression).unwrap(), b"regression");

        let tie = temp.path().join("trial-3.checkpoint");
        fs::write(&tie, b"tie").unwrap();
        let (promoted, _) = store
            .promote_checkpoint(
                "goal.demo",
                "victories",
                PromotionMode::Max,
                3.0,
                &run.run_id,
                "trial-3",
                &tie,
                &live,
            )
            .unwrap();
        assert!(!promoted);
        assert_eq!(fs::read(&live).unwrap(), b"first");
    }
}

fn enum_string<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_value(value)?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| Error::Invalid("enum did not serialize as text".into()))
}

fn now_ns() -> Result<i64> {
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
