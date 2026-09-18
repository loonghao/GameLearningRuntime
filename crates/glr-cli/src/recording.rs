//! Training-run recording integration.
//!
//! `glr train` records the main window of the trainer process it spawns, from
//! before the first step to after the last one, and registers every segment as
//! run evidence. Recording is best effort: a session that cannot start is
//! reported as skipped and the run continues unchanged.

use std::path::Path;

use glr_recording::{RecordingReport, RecordingSession, RecordingStart, RecordingTarget};
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
        let start = RecordingSession::start(target.clone(), &project.recording, run_dir);
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
        register(store, run_id, run_dir, &segment.video, "capture_video", "video/mp4");
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
