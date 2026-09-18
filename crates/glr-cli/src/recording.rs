//! Training-run recording integration.
//!
//! `glr train` records the main window of the trainer process it spawns, from
//! before the first step to after the last one, and registers every segment as
//! run evidence. Recording is best effort: a session that cannot start is
//! reported as skipped and the run continues unchanged.

use std::path::Path;

use glr_recording::{
    RecordingReport, RecordingSession, RecordingSkip, RecordingStart, RecordingTarget,
};
use serde_json::Value;

use crate::error::Result;
use crate::project::Project;
use crate::store::Store;

/// A recording started for a run, or the reason none was started.
pub struct TrainingRecording {
    start: Option<RecordingStart>,
    target: RecordingTarget,
}

impl TrainingRecording {
    /// Starts recording the trainer process below the run directory.
    #[must_use]
    pub fn start(project: &Project, run_dir: &Path, pid: u32) -> Self {
        let target = RecordingTarget::new(pid, "trainer");
        let start = if Path::new(&project.recording.output_dir).is_absolute() {
            // Run evidence must stay inside the run directory. An absolute
            // output directory would place every segment outside it, so the
            // run could not register them; skip explicitly instead of
            // degrading silently per segment.
            RecordingStart::Skipped(RecordingSkip::Unavailable(format!(
                "recording.output_dir {:?} is absolute; segments outside the run \
                 directory cannot be registered as run evidence",
                project.recording.output_dir
            )))
        } else {
            RecordingSession::start(target.clone(), &project.recording, run_dir)
        };
        Self {
            start: Some(start),
            target,
        }
    }

    /// Stops recording, registers the segments and returns the report envelope.
    #[must_use]
    pub fn stop(self, store: &Store, run_id: &str, run_dir: &Path) -> Value {
        let Some(start) = self.start else {
            return Value::Null;
        };
        let report = start.finish(&self.target);
        for warning in &report.warnings {
            eprintln!("GLR recording warning: {warning}");
        }
        if let Some(reason) = &report.skipped {
            eprintln!("GLR recording skipped: {reason}");
        }
        register_segments(store, run_id, run_dir, &report);
        report.to_json()
    }
}

/// Registers every segment as run evidence; registration never fails a run.
fn register_segments(store: &Store, run_id: &str, run_dir: &Path, report: &RecordingReport) {
    for segment in &report.segments {
        register(
            store,
            run_id,
            run_dir,
            &segment.video,
            "capture_video",
            "video/mp4",
        );
        register(
            store,
            run_id,
            run_dir,
            &segment.index,
            "capture_index",
            "text/plain",
        );
    }
}

fn register(
    store: &Store,
    run_id: &str,
    run_dir: &Path,
    path: &Path,
    role: &str,
    media_type: &str,
) {
    let relative = match crate::process::relative_portable(run_dir, path) {
        Ok(relative) => relative,
        Err(error) => {
            eprintln!("GLR recording warning: {path:?} is not run-relative ({error})");
            return;
        }
    };
    if let Err(error) = store.register_artifact(run_id, &relative, path, role, media_type) {
        eprintln!("GLR recording warning: cannot register {relative} ({error})");
    }
}

/// Records the main window of an arbitrary process (Windows only).
///
/// # Errors
///
/// Never fails: an unavailable session is reported as skipped in the report.
pub fn record_window(
    project: &Project,
    pid: u32,
    seconds: u64,
    output: Option<&Path>,
) -> Result<RecordingReport> {
    let mut config = project.recording.clone();
    if let Some(path) = output {
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            project.root.join(path)
        };
        config.output_dir = resolved.display().to_string();
    }
    // An explicit command records even when the project default is disabled.
    config.enabled = true;
    let target = RecordingTarget::new(pid, "process");
    let start = RecordingSession::start(target.clone(), &config, &project.root);
    Ok(match start {
        RecordingStart::Started(session) => {
            std::thread::sleep(std::time::Duration::from_secs(seconds));
            session.stop()
        }
        RecordingStart::Skipped(reason) => RecordingReport::skipped(&target, &reason),
    })
}

#[cfg(test)]
mod tests {
    use super::{TrainingRecording, register_segments};
    use crate::store::Store;
    use glr_recording::{RecordingReport, RecordingSegment};
    use serde_json::json;
    use std::fs;
    use std::path::Path;

    fn project(root: &Path, recording: serde_json::Value) -> crate::project::Project {
        fs::create_dir_all(root.join("bridge")).unwrap();
        fs::write(
            root.join("glr-project.json"),
            serde_json::to_vec_pretty(&json!({
                "schema_version": "glr.project.v1",
                "environment_id": "example.context-v1",
                "environment_family": "test",
                "protocol_version": "1.0",
                "data_dir": ".glr",
                "bridge_path": "bridge",
                "runtime": {"argv": ["true"]},
                "trainer": {"argv": ["true"]},
                "player": {"argv": ["true"]},
                "capture": null,
                "recording": recording
            }))
            .unwrap(),
        )
        .unwrap();
        crate::project::load_project(root).unwrap()
    }

    fn report(segments: Vec<RecordingSegment>) -> RecordingReport {
        RecordingReport {
            pid: 8124,
            label: "trainer".into(),
            frames: segments.iter().map(|segment| segment.frames).sum(),
            width: 640,
            height: 480,
            segments,
            warnings: Vec::new(),
            skipped: None,
        }
    }

    #[test]
    fn registers_segments_as_run_evidence_inside_the_run_directory() {
        let temp = tempfile::tempdir().unwrap();
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.context-v1", "1.0", "training", json!({}))
            .unwrap();
        let run_dir = temp.path().join("run");
        fs::create_dir_all(&run_dir).unwrap();
        let video = run_dir.join("20260918T024500Z_8124_trainer_seg0.mp4");
        let index = run_dir.join("20260918T024500Z_8124_trainer_seg0.frames.jsonl");
        fs::write(&video, b"mp4").unwrap();
        fs::write(&index, b"{\"frame\":0}\n").unwrap();

        register_segments(
            &store,
            &run.run_id,
            &run_dir,
            &report(vec![RecordingSegment {
                video,
                index,
                frames: 1,
            }]),
        );

        let artifacts = store.list_artifacts(&run.run_id).unwrap();
        let roles: Vec<&str> = artifacts
            .iter()
            .map(|artifact| artifact.role.as_str())
            .collect();
        assert_eq!(roles, vec!["capture_index", "capture_video"]);
        let paths: Vec<&str> = artifacts
            .iter()
            .map(|artifact| artifact.path.as_str())
            .collect();
        assert_eq!(
            paths,
            vec![
                "20260918T024500Z_8124_trainer_seg0.frames.jsonl",
                "20260918T024500Z_8124_trainer_seg0.mp4"
            ]
        );
    }

    #[test]
    fn absolute_output_dir_skips_training_recording() {
        let temp = tempfile::tempdir().unwrap();
        let absolute = temp.path().join("outside-recordings");
        let project = project(
            temp.path(),
            json!({ "output_dir": absolute.display().to_string() }),
        );
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.context-v1", "1.0", "training", json!({}))
            .unwrap();
        let run_dir = temp.path().join("run");

        let recording = TrainingRecording::start(&project, &run_dir, 1234);
        let envelope = recording.stop(&store, &run.run_id, &run_dir);

        assert_eq!(envelope["schema_version"], "glr.recording.v1");
        assert!(envelope["skipped"].as_str().unwrap().contains("absolute"));
        assert!(envelope["segments"].as_array().unwrap().is_empty());
    }

    #[cfg(not(windows))]
    #[test]
    fn training_recording_skips_on_non_windows() {
        let temp = tempfile::tempdir().unwrap();
        let project = project(temp.path(), json!({}));
        let store = Store::open(temp.path().join("runs.sqlite3")).unwrap();
        let run = store
            .create_run("example.context-v1", "1.0", "training", json!({}))
            .unwrap();
        let run_dir = temp.path().join("run");

        let recording = TrainingRecording::start(&project, &run_dir, 1234);
        let envelope = recording.stop(&store, &run.run_id, &run_dir);

        assert_eq!(envelope["schema_version"], "glr.recording.v1");
        assert_eq!(
            envelope["skipped"].as_str(),
            Some("recording requires Windows; the session is skipped on this platform")
        );
        assert!(envelope["segments"].as_array().unwrap().is_empty());
    }
}
