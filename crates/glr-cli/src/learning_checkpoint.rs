use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::project::validate_identifier;

pub const LEARNING_CHECKPOINT_SCHEMA_VERSION: &str = "glr.learning-checkpoint.v1";

#[derive(Debug, Serialize)]
pub struct LearningCheckpoint<'a> {
    pub schema_version: &'static str,
    pub environment_id: &'a str,
    pub goal_id: &'a str,
    pub run_id: &'a str,
    pub trial_id: &'a str,
    pub stage: &'a str,
    pub stage_index: u8,
    pub learning_status: &'a str,
    pub total_training_steps: u64,
    pub state: Value,
}

pub fn ensure_goal_root(data_dir: &Path, environment_id: &str, goal_id: &str) -> Result<PathBuf> {
    validate_identifier(environment_id, "checkpoint environment_id")?;
    validate_identifier(goal_id, "checkpoint goal_id")?;
    let mut current = data_dir.to_path_buf();
    for segment in ["checkpoints", environment_id, goal_id] {
        current.push(segment);
        ensure_directory(&current)?;
    }
    Ok(current)
}

pub fn write_stage(data_dir: &Path, checkpoint: &LearningCheckpoint<'_>) -> Result<PathBuf> {
    for (value, label) in [
        (checkpoint.run_id, "checkpoint run_id"),
        (checkpoint.trial_id, "checkpoint trial_id"),
        (checkpoint.stage, "checkpoint stage"),
    ] {
        validate_identifier(value, label)?;
    }
    let goal_directory = ensure_goal_root(data_dir, checkpoint.environment_id, checkpoint.goal_id)?;
    let mut directory = goal_directory.clone();
    let mut run_directory = None;
    for segment in ["runs", checkpoint.run_id, checkpoint.trial_id] {
        directory.push(segment);
        ensure_directory(&directory)?;
        if segment == checkpoint.run_id {
            run_directory = Some(directory.clone());
        }
    }
    let path = directory.join(format!(
        "{:02}-{}.json",
        checkpoint.stage_index, checkpoint.stage
    ));
    write_atomic(&path, checkpoint)?;
    write_atomic(&directory.join("latest.json"), checkpoint)?;
    write_atomic(
        &run_directory
            .ok_or_else(|| Error::Invalid("learning checkpoint run directory is missing".into()))?
            .join("latest.json"),
        checkpoint,
    )?;
    write_atomic(&goal_directory.join("latest.json"), checkpoint)?;
    Ok(path)
}

fn ensure_directory(path: &Path) -> Result<()> {
    if path.exists() {
        if path.is_symlink() || !path.is_dir() {
            return Err(Error::Contract(
                "checkpoint path components must be regular directories".into(),
            ));
        }
    } else {
        fs::create_dir(path)?;
    }
    Ok(())
}

fn write_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if path.exists() && (path.is_symlink() || !path.is_file()) {
        return Err(Error::Contract(
            "learning checkpoint target must be a regular file".into(),
        ));
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("learning checkpoint has no parent".into()))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&bytes)?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| Error::Io(error.error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{LEARNING_CHECKPOINT_SCHEMA_VERSION, LearningCheckpoint, write_stage};
    use serde_json::{Value, json};

    #[test]
    fn stage_checkpoints_are_namespaced_and_latest_is_refreshed() {
        let temporary = tempfile::tempdir().unwrap();
        let checkpoint = LearningCheckpoint {
            schema_version: LEARNING_CHECKPOINT_SCHEMA_VERSION,
            environment_id: "example.adventure-v1",
            goal_id: "reach-boss",
            run_id: "run-1234",
            trial_id: "trial-1",
            stage: "trainer",
            stage_index: 3,
            learning_status: "completed",
            total_training_steps: 128,
            state: json!({"policy_candidate": "checkpoint.candidate"}),
        };
        let path = write_stage(temporary.path(), &checkpoint).unwrap();
        assert!(path.ends_with(
            "checkpoints/example.adventure-v1/reach-boss/runs/run-1234/trial-1/03-trainer.json"
        ));
        let latest: Value = serde_json::from_slice(
            &std::fs::read(path.parent().unwrap().join("latest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(latest["schema_version"], LEARNING_CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(latest["learning_status"], "completed");
        assert_eq!(latest["total_training_steps"], 128);
        let goal_latest: Value = serde_json::from_slice(
            &std::fs::read(
                temporary
                    .path()
                    .join("checkpoints/example.adventure-v1/reach-boss/latest.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(goal_latest["stage"], "trainer");
    }
}
