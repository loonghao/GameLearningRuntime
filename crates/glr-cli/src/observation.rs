//! Read-only projections shared by the HTTP dashboard and external report clients.
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::project::validate_identifier;
use crate::store::Store;

pub const SCHEMA: &str = "glr.observation.v1";
pub const LOG_BYTES: u64 = 65536;
const LOG_NAMES: &[&str] = &[
    "trainer.log",
    "capture.log",
    "runtime.log",
    "playback.log",
    "researcher.log",
    "planner.log",
    "evaluator.log",
];

pub struct Observation {
    pub data_dir: PathBuf,
    pub environment_id: String,
}

impl Observation {
    fn store(&self) -> Result<Store> {
        safe_child(&self.data_dir, Path::new("runs.sqlite3"))?;
        Store::read_only(self.data_dir.join("runs.sqlite3"))
    }

    pub fn runs(&self, before: Option<&str>) -> Result<Value> {
        // A fresh project can be observed before its first run exists.
        let runs = if self.data_dir.join("runs.sqlite3").exists() {
            self.store()?
                .observation_runs(&self.environment_id, before)?
        } else {
            Vec::new()
        };
        Ok(json!({"schema_version": SCHEMA, "runs": runs, "limit": 100,
            "next_before": if runs.len()==100 { runs.last().map(|r|&r.run_id) } else { None }}))
    }

    pub fn snapshot(
        &self,
        run_id: &str,
        events_after: i64,
        metrics_after: i64,
        limit: u32,
    ) -> Result<Value> {
        if events_after < -1 || metrics_after < 0 || !(1..=250).contains(&limit) {
            return Err(Error::Invalid(
                "invalid observation cursor or page size".into(),
            ));
        }
        let store = self.store()?;
        self.run_dir(run_id)?;
        let run = store.get_run(run_id)?;
        if run.environment_id != self.environment_id {
            return Err(Error::Invalid("run belongs to another environment".into()));
        }
        let events = store.observation_page(run_id, "events", events_after, limit)?;
        let metrics = store.observation_page(run_id, "metrics", metrics_after, limit)?;
        let next_event = events
            .last()
            .and_then(|v| v["sequence_id"].as_i64())
            .unwrap_or(events_after);
        let next_metric = metrics
            .last()
            .and_then(|v| v["metric_id"].as_i64())
            .unwrap_or(metrics_after);
        let (logs, logs_truncated) = self.logs(run_id)?;
        Ok(json!({"schema_version": SCHEMA, "run": run,
            "events": events, "metrics": metrics, "logs": logs, "logs_truncated": logs_truncated,
            "cursor": {"events_after": next_event, "metrics_after": next_metric},
            "more": {"events": events.len() == limit as usize, "metrics": metrics.len() == limit as usize},
            "limits": {"page": limit, "record_payload_bytes": 16384, "log_chunk_bytes": LOG_BYTES},
            "authority": "read_only_projection"}))
    }

    fn run_dir(&self, run_id: &str) -> Result<PathBuf> {
        validate_identifier(run_id, "run_id")?;
        safe_child(&self.data_dir, &Path::new("runs").join(run_id))
    }

    fn logs(&self, run_id: &str) -> Result<(Vec<String>, bool)> {
        let root = self.run_dir(run_id)?;
        let mut logs = Vec::new();
        for name in LOG_NAMES {
            if safe_child(&root, Path::new(name))?.is_file() {
                logs.push((*name).into());
            }
        }
        let trials = safe_child(&root, Path::new("trials"))?;
        let mut truncated = false;
        if trials.is_dir() {
            for (index, entry) in fs::read_dir(&trials)?.enumerate() {
                if index == 128 {
                    truncated = true;
                    break;
                }
                let entry = entry?;
                let name = entry.file_name().to_string_lossy().into_owned();
                if validate_identifier(&name, "trial").is_err() || !entry.file_type()?.is_dir() {
                    continue;
                }
                for log in LOG_NAMES {
                    let relative = format!("trials/{name}/{log}");
                    if safe_child(&root, Path::new(&relative))?.is_file() {
                        logs.push(relative);
                    }
                }
            }
        }
        logs.sort();
        Ok((logs, truncated))
    }

    pub fn log(&self, run_id: &str, relative: &str, offset: Option<u64>) -> Result<Value> {
        self.log_page(run_id, relative, offset, None)
    }

    pub fn log_page(
        &self,
        run_id: &str,
        relative: &str,
        offset: Option<u64>,
        before: Option<u64>,
    ) -> Result<Value> {
        if offset.is_some() && before.is_some() {
            return Err(Error::Invalid("choose offset or before, not both".into()));
        }
        let store = self.store()?;
        if store.get_run(run_id)?.environment_id != self.environment_id {
            return Err(Error::Invalid("run belongs to another environment".into()));
        }
        let parts: Vec<_> = relative.split('/').collect();
        let valid = match parts.as_slice() {
            [name] => LOG_NAMES.contains(name),
            ["trials", trial, name] => {
                validate_identifier(trial, "trial").is_ok() && LOG_NAMES.contains(name)
            }
            _ => false,
        };
        if !valid {
            return Err(Error::Invalid(
                "only managed role logs are observable".into(),
            ));
        }
        let path = safe_child(&self.run_dir(run_id)?, Path::new(relative))?;
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        let reset =
            offset.is_some_and(|value| value > size) || before.is_some_and(|value| value > size);
        let end = before.unwrap_or(size).min(size);
        let mut start = if reset {
            0
        } else {
            offset.unwrap_or_else(|| end.saturating_sub(LOG_BYTES))
        };
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        (&mut file)
            .take(LOG_BYTES.min(end.saturating_sub(start)))
            .read_to_end(&mut bytes)?;
        // Keep independently fetched pages composable across UTF-8 boundaries.
        let leading = bytes.iter().take_while(|b| **b & 0xc0 == 0x80).count();
        bytes.drain(..leading);
        start += leading as u64;
        if let Some(index) = bytes.iter().rposition(|b| b & 0xc0 != 0x80) {
            let width = match bytes[index] {
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf4 => 4,
                _ => 1,
            };
            if bytes.len() - index < width {
                bytes.truncate(index);
            }
        }
        let next = start + bytes.len() as u64;
        let mut preceding = *b"\n";
        if start > 0 {
            file.seek(SeekFrom::Start(start - 1))?;
            file.read_exact(&mut preceding)?;
        }
        let partial_start = start > 0 && !matches!(preceding[0], b'\n' | b'\r');
        let partial_end = bytes.last().is_some_and(|b| !matches!(b, b'\n' | b'\r'));
        Ok(
            json!({"schema_version": SCHEMA, "run_id": run_id, "path": relative,
            "offset": start, "next_offset": next, "size_bytes": size, "reset": reset,
            "tail_truncated": offset.is_none() && start > 0, "more": next < size,
            "partial_start":partial_start,"partial_end":partial_end,
            "text": String::from_utf8_lossy(&bytes)}),
        )
    }
}

/// Reject lexical escapes and all existing link/junction components, including
/// storage ancestors. The web API never resolves arbitrary project files.
pub(crate) fn safe_child(root: &Path, relative: &Path) -> Result<PathBuf> {
    if relative
        .components()
        .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err(Error::Invalid(
            "observation path must stay inside storage".into(),
        ));
    }
    for component in relative.components() {
        let text = component.as_os_str().to_string_lossy();
        if text.contains(':') || text.ends_with(['.', ' ']) {
            return Err(Error::Invalid(
                "observation path must use portable components".into(),
            ));
        }
    }
    let target = root.join(relative);
    for path in target.ancestors() {
        if let Ok(metadata) = fs::symlink_metadata(path) {
            let linked = metadata.file_type().is_symlink();
            #[cfg(windows)]
            let linked = {
                use std::os::windows::fs::MetadataExt;
                linked || metadata.file_attributes() & 0x400 != 0
            };
            if linked {
                return Err(Error::Invalid(
                    "observation path cannot traverse links".into(),
                ));
            }
        }
    }
    Ok(target)
}
