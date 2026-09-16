//! Versioned passive bridge diagnostics. Transport never grants action authority.
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::args::TelemetryCommand;
use crate::error::{Error, Result};
use crate::observation::safe_child;
use crate::project::{Project, validate_identifier};
use crate::store::Store;

pub const SCHEMA: &str = "glr.bridge-telemetry.v1";
pub const MAX_BYTES: usize = 65536;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Batch {
    schema_version: String,
    run_id: String,
    source: String,
    batch_id: String,
    #[serde(default)]
    events: Vec<Event>,
    #[serde(default)]
    metrics: Vec<Metric>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Event {
    kind: String,
    payload: serde_json::Map<String, Value>,
    #[serde(default)]
    step_id: Option<i64>,
    #[serde(default)]
    episode_id: Option<String>,
    #[serde(default)]
    observed_at_ns: Option<i64>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Metric {
    name: String,
    value: f64,
    #[serde(default)]
    step_id: Option<i64>,
}

fn identifier(value: &str, label: &str) -> Result<()> {
    validate_identifier(value, label)?;
    if value.len() > 128 {
        return Err(Error::Invalid(format!("{label} exceeds 128 bytes")));
    }
    Ok(())
}

fn validate(batch: &Batch) -> Result<()> {
    if batch.schema_version != SCHEMA
        || !(1..=64).contains(&(batch.events.len() + batch.metrics.len()))
    {
        return Err(Error::Invalid(
            "telemetry requires glr.bridge-telemetry.v1 and 1..64 records".into(),
        ));
    }
    for (label, value) in [
        ("run_id", &batch.run_id),
        ("source", &batch.source),
        ("batch_id", &batch.batch_id),
    ] {
        identifier(value, label)?;
    }
    for event in &batch.events {
        identifier(&event.kind, "event kind")?;
        if let Some(episode) = &event.episode_id {
            // Episode UUIDs are valid; unlike GLR identifiers they may start with a digit.
            if episode.is_empty()
                || episode.len() > 128
                || !episode
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
            {
                return Err(Error::Invalid("invalid episode_id".into()));
            }
        }
        if event.step_id.is_some_and(|s| s < 0)
            || event.observed_at_ns.is_some_and(|s| s < 0)
            || event.payload.contains_key("_glr")
            || event.payload.contains_key("authority")
            || serde_json::to_vec(&event.payload)?.len() > 12288
        {
            return Err(Error::Invalid(
                "invalid event cursor, reserved provenance, or payload over 12 KiB".into(),
            ));
        }
        if event.kind == "navigation.route_sample"
            && !event
                .payload
                .get("position")
                .and_then(Value::as_array)
                .is_some_and(|p| {
                    (2..=3).contains(&p.len())
                        && p.iter().all(|v| v.as_f64().is_some_and(f64::is_finite))
                })
        {
            return Err(Error::Invalid(
                "route sample requires 2 or 3 finite position coordinates".into(),
            ));
        }
        if event.kind == "bridge.progress"
            && !event
                .payload
                .get("fraction")
                .and_then(Value::as_f64)
                .is_some_and(|f| (0.0..=1.0).contains(&f))
        {
            return Err(Error::Invalid(
                "bridge.progress requires fraction in 0..1".into(),
            ));
        }
    }
    for metric in &batch.metrics {
        identifier(&metric.name, "metric name")?;
        if !metric.value.is_finite() || metric.step_id.is_some_and(|s| s < 0) {
            return Err(Error::Invalid(
                "metric requires finite value and nonnegative step".into(),
            ));
        }
    }
    Ok(())
}

pub fn ingest(data_dir: &Path, environment: &str, bytes: &[u8]) -> Result<Value> {
    if bytes.len() > MAX_BYTES {
        return Err(Error::Invalid("telemetry batch exceeds 64 KiB".into()));
    }
    let batch: Batch = serde_json::from_slice(bytes)?;
    validate(&batch)?;
    let digest = format!("{:x}", Sha256::digest(serde_json::to_vec(&batch)?));
    let path = safe_child(data_dir, Path::new("runs.sqlite3"))?;
    Store::read_only(path.clone())?; // Never invent a run/database or accept an unknown schema.
    let mut db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(2))?;
    db.execute_batch("PRAGMA foreign_keys=ON;")?;
    let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let (run_environment, status): (String, String) = tx.query_row(
        "SELECT environment_id,status FROM runs WHERE run_id=?",
        [&batch.run_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    if run_environment != environment {
        return Err(Error::Contract(
            "telemetry run belongs to another environment".into(),
        ));
    }
    tx.execute_batch(
        "CREATE TABLE IF NOT EXISTS telemetry_batches (
        run_id TEXT NOT NULL REFERENCES runs(run_id), source TEXT NOT NULL, batch_id TEXT NOT NULL,
        digest TEXT NOT NULL, receipt TEXT NOT NULL, PRIMARY KEY(run_id,source,batch_id));
        CREATE TABLE IF NOT EXISTS telemetry_latest (
        run_id TEXT NOT NULL REFERENCES runs(run_id), source TEXT NOT NULL, kind TEXT NOT NULL,
        event_json TEXT NOT NULL, PRIMARY KEY(run_id,source,kind));",
    )?;
    let previous: Option<(String,String)> = tx.query_row(
        "SELECT digest,receipt FROM telemetry_batches WHERE run_id=? AND source=? AND batch_id=?",
        params![batch.run_id,batch.source,batch.batch_id], |r| Ok((r.get(0)?,r.get(1)?))).optional()?;
    if let Some((previous_digest, receipt)) = previous {
        if previous_digest != digest {
            return Err(Error::Contract(
                "batch_id already used with different content".into(),
            ));
        }
        let mut receipt: Value = serde_json::from_str(&receipt)?;
        receipt["duplicate"] = true.into();
        return Ok(receipt);
    }
    if status != "running" {
        return Err(Error::Contract(
            "new telemetry requires a running run; terminal runs are immutable".into(),
        ));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64;
    let mut sequence: i64 = tx.query_row(
        "SELECT COALESCE(MAX(sequence_id),0) FROM events WHERE run_id=?",
        [&batch.run_id],
        |r| r.get(0),
    )?;
    let first_sequence = sequence + 1;
    for event in &batch.events {
        sequence += 1;
        let mut payload = event.payload.clone();
        payload.insert("authority".into(), "diagnostic".into());
        payload.insert("_glr".into(), json!({"source":batch.source,"batch_id":batch.batch_id,"observed_at_ns":event.observed_at_ns}));
        tx.execute("INSERT INTO events(run_id,sequence_id,timestamp_ns,kind,episode_id,step_id,payload_json) VALUES(?,?,?,?,?,?,?)",
            params![batch.run_id,sequence,now,event.kind,event.episode_id,event.step_id,serde_json::to_string(&payload)?])?;
        if matches!(
            event.kind.as_str(),
            "bridge.status" | "bridge.state" | "bridge.progress"
        ) {
            let record = json!({"run_id":batch.run_id,"source":batch.source,"sequence_id":sequence,"timestamp_ns":now,"kind":event.kind,"step_id":event.step_id,"episode_id":event.episode_id,"payload":payload});
            tx.execute("INSERT INTO telemetry_latest(run_id,source,kind,event_json) VALUES(?,?,?,?) ON CONFLICT(run_id,source,kind) DO UPDATE SET event_json=excluded.event_json",
                params![batch.run_id,batch.source,event.kind,record.to_string()])?;
        }
    }
    let mut metric_ids = Vec::new();
    for metric in &batch.metrics {
        tx.execute("INSERT INTO metrics(run_id,timestamp_ns,name,value,step_id,metadata_json) VALUES(?,?,?,?,?,?)",
            params![batch.run_id,now,metric.name,metric.value,metric.step_id,json!({"source":batch.source,"batch_id":batch.batch_id,"authority":"diagnostic"}).to_string()])?;
        metric_ids.push(tx.last_insert_rowid());
    }
    let receipt = json!({"schema_version":SCHEMA,"run_id":batch.run_id,"source":batch.source,"batch_id":batch.batch_id,
        "duplicate":false,"accepted_events":batch.events.len(),"accepted_metrics":batch.metrics.len(),
        "first_sequence_id":if batch.events.is_empty(){None}else{Some(first_sequence)},"last_sequence_id":sequence,"metric_ids":metric_ids});
    tx.execute(
        "INSERT INTO telemetry_batches(run_id,source,batch_id,digest,receipt) VALUES(?,?,?,?,?)",
        params![
            batch.run_id,
            batch.source,
            batch.batch_id,
            digest,
            receipt.to_string()
        ],
    )?;
    tx.commit()?;
    Ok(receipt)
}

pub fn latest(data_dir: &Path, environment: &str, run: &str) -> Result<Value> {
    identifier(run, "run_id")?;
    let path = safe_child(data_dir, Path::new("runs.sqlite3"))?;
    let store = Store::read_only(path.clone())?;
    if store.get_run(run)?.environment_id != environment {
        return Err(Error::Contract(
            "telemetry run belongs to another environment".into(),
        ));
    }
    let db = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    db.busy_timeout(Duration::from_secs(1))?;
    let exists: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='telemetry_latest')",
        [],
        |r| r.get(0),
    )?;
    let mut states = Vec::new();
    if exists {
        let mut query = db.prepare(
            "SELECT event_json FROM telemetry_latest WHERE run_id=? ORDER BY source,kind LIMIT 101",
        )?;
        for row in query.query_map([run], |r| r.get::<_, String>(0))? {
            states.push(serde_json::from_str::<Value>(&row?)?);
        }
    }
    let truncated = states.len() > 100;
    states.truncate(100);
    Ok(
        json!({"schema_version":SCHEMA,"run_id":run,"states":states,"truncated":truncated,"authority":"diagnostic"}),
    )
}

pub fn schema() -> Value {
    serde_json::from_str(include_str!(
        "../../../docs/schemas/bridge-telemetry.schema.json"
    ))
    .expect("embedded schema")
}

pub fn execute(project: &Project, command: &TelemetryCommand) -> Result<Value> {
    match command {
        TelemetryCommand::Schema => Ok(schema()),
        TelemetryCommand::State { run_id } => {
            latest(&project.data_dir, &project.environment_id, run_id)
        }
        TelemetryCommand::Ingest { file, jsonl } => {
            let input: Box<dyn Read> = if file.as_os_str() == "-" {
                Box::new(std::io::stdin())
            } else {
                Box::new(std::fs::File::open(file)?)
            };
            let mut reader = BufReader::new(input);
            let mut receipts = Vec::new();
            loop {
                let mut bytes = Vec::new();
                if *jsonl {
                    reader
                        .by_ref()
                        .take((MAX_BYTES + 2) as u64)
                        .read_until(b'\n', &mut bytes)?;
                    if bytes.is_empty() {
                        break;
                    }
                    if bytes.last() == Some(&b'\n') {
                        bytes.pop();
                    }
                    if bytes.last() == Some(&b'\r') {
                        bytes.pop();
                    }
                } else {
                    reader
                        .by_ref()
                        .take((MAX_BYTES + 1) as u64)
                        .read_to_end(&mut bytes)?;
                }
                if receipts.len() >= 1000 {
                    return Err(Error::Invalid(
                        "JSONL import exceeds 1000 batches; split the file".into(),
                    ));
                }
                receipts.push(ingest(&project.data_dir, &project.environment_id, &bytes)?);
                if !jsonl {
                    break;
                }
            }
            Ok(json!({"schema_version":SCHEMA,"receipts":receipts,"atomicity":"per_batch"}))
        }
    }
}
