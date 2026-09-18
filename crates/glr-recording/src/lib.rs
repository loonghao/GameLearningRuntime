//! PID-scoped window recording for GLR training runs.
//!
//! A recording session captures the main window of one process — normally the
//! trainer spawned by `glr train` — with Windows Graphics Capture and encodes it
//! to H.264 MP4 plus a per-frame index used to align frames with learner steps.
//!
//! Recording is always best effort. Every failure mode (unsupported platform,
//! no desktop session, no capturable window, encoder unavailable) skips the
//! session with a reason instead of failing the run that requested it.

mod canvas;
mod config;
mod index;

#[cfg(windows)]
mod platform_windows;
#[cfg(windows)]
use platform_windows as platform;

#[cfg(not(windows))]
mod platform_stub;
#[cfg(not(windows))]
use platform_stub as platform;

use std::path::{Path, PathBuf};

pub use canvas::{FrameGeometry, composite_bgra, plan_size};
pub use config::{RecordingConfig, RecordingConfigError};
pub use index::FrameIndexWriter;

/// Envelope schema of [`RecordingReport::to_json`].
pub const RECORDING_SCHEMA_VERSION: &str = "glr.recording.v1";
/// Line schema of the per-frame index written next to every segment.
pub const FRAME_INDEX_SCHEMA_VERSION: &str = "glr.recording-frame.v1";

/// The process whose main window is recorded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingTarget {
    /// Operating system process id of the recorded process.
    pub pid: u32,
    /// Short label used in file names when no process name is available.
    pub label: String,
}

impl RecordingTarget {
    /// Builds a target from a process id and a file-name label.
    #[must_use]
    pub fn new(pid: u32, label: impl Into<String>) -> Self {
        Self {
            pid,
            label: label.into(),
        }
    }
}

/// Why a recording session did not start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordingSkip {
    /// Configuration disabled recording.
    Disabled,
    /// The host platform has no recording backend.
    UnsupportedPlatform,
    /// Windows Graphics Capture is unavailable on this device.
    CaptureUnsupported,
    /// The session runs inside a remote desktop or server session.
    RemoteSession,
    /// No capturable main window belongs to the target process.
    NoMainWindow,
    /// The main window is minimized and auto restore is disabled.
    Minimized,
    /// The main window stayed minimized after an automatic restore attempt.
    StillMinimized,
    /// Any other start failure, described by the message.
    Unavailable(String),
}

impl RecordingSkip {
    /// Human readable reason, safe to log and to surface to operators.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::Disabled => "recording is disabled by configuration".into(),
            Self::UnsupportedPlatform => {
                "recording requires Windows; the session is skipped on this platform".into()
            }
            Self::CaptureUnsupported => {
                "Windows Graphics Capture is not supported on this device".into()
            }
            Self::RemoteSession => {
                "Windows Graphics Capture is unavailable in a remote desktop session".into()
            }
            Self::NoMainWindow => "the target process has no capturable main window".into(),
            Self::Minimized => {
                "the target window is minimized and auto_restore_minimized is disabled".into()
            }
            Self::StillMinimized => {
                "the target window stayed minimized after an automatic restore attempt".into()
            }
            Self::Unavailable(detail) => format!("recording unavailable: {detail}"),
        }
    }
}

/// Outcome of [`RecordingSession::start`].
#[derive(Debug)]
pub enum RecordingStart {
    /// A capture session is running in the background.
    Started(RecordingSession),
    /// No session was started; training continues unaffected.
    Skipped(RecordingSkip),
}

impl RecordingStart {
    /// Whether a capture session is running.
    #[must_use]
    pub const fn is_started(&self) -> bool {
        matches!(self, Self::Started(_))
    }

    /// Stops a running session, or reports why none was started.
    #[must_use]
    pub fn finish(self, target: &RecordingTarget) -> RecordingReport {
        match self {
            Self::Started(session) => session.stop(),
            Self::Skipped(reason) => RecordingReport::skipped(target, &reason),
        }
    }
}

/// A running recording session. Stops on [`RecordingSession::stop`] or on drop.
#[derive(Debug)]
pub struct RecordingSession {
    inner: platform::Session,
}

impl RecordingSession {
    /// Starts a session for `target`, writing below `base_dir`.
    ///
    /// Never fails: an unavailable session is reported as
    /// [`RecordingStart::Skipped`] so callers can log a warning and continue.
    #[must_use]
    pub fn start(
        target: RecordingTarget,
        config: &RecordingConfig,
        base_dir: &Path,
    ) -> RecordingStart {
        match platform::start(target, config, base_dir) {
            platform::StartOutcome::Started(session) => {
                RecordingStart::Started(Self { inner: session })
            }
            platform::StartOutcome::Skipped(reason) => RecordingStart::Skipped(reason),
        }
    }

    /// Stops capture, finalizes every segment and returns what was written.
    #[must_use]
    pub fn stop(self) -> RecordingReport {
        self.inner.stop()
    }
}

/// One encoded MP4 file and its frame index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingSegment {
    /// Absolute path of the MP4 file.
    pub video: PathBuf,
    /// Absolute path of the `frames.jsonl` index for this segment.
    pub index: PathBuf,
    /// Frames written to this segment.
    pub frames: u64,
}

/// What a finished (or skipped) recording produced.
#[derive(Debug, Clone)]
pub struct RecordingReport {
    /// Recorded process id.
    pub pid: u32,
    /// Label of the recorded process.
    pub label: String,
    /// Frames delivered to the encoder across all segments.
    pub frames: u64,
    /// Encoded width in pixels.
    pub width: u32,
    /// Encoded height in pixels.
    pub height: u32,
    /// Segments written, oldest first.
    pub segments: Vec<RecordingSegment>,
    /// Non-fatal problems worth surfacing to the operator.
    pub warnings: Vec<String>,
    /// Why nothing was recorded, when the session was skipped.
    pub skipped: Option<String>,
}

impl RecordingReport {
    /// Builds a report for a session that never started.
    #[must_use]
    pub fn skipped(target: &RecordingTarget, reason: &RecordingSkip) -> Self {
        Self {
            pid: target.pid,
            label: target.label.clone(),
            frames: 0,
            width: 0,
            height: 0,
            segments: Vec::new(),
            warnings: Vec::new(),
            skipped: Some(reason.message()),
        }
    }

    /// Whether anything was written to disk.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Machine readable envelope used by `glr train` output.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema_version": RECORDING_SCHEMA_VERSION,
            "frame_index_schema_version": FRAME_INDEX_SCHEMA_VERSION,
            "pid": self.pid,
            "label": self.label,
            "frames": self.frames,
            "width": self.width,
            "height": self.height,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "segments": self.segments.iter().map(|segment| serde_json::json!({
                "video": segment.video.display().to_string(),
                "index": segment.index.display().to_string(),
                "frames": segment.frames,
            })).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{RecordingReport, RecordingSkip, RecordingTarget};

    #[test]
    fn skipped_report_carries_the_reason() {
        let target = RecordingTarget::new(8124, "trainer");
        let report = RecordingReport::skipped(&target, &RecordingSkip::NoMainWindow);
        assert!(report.is_empty());
        assert_eq!(report.pid, 8124);
        assert_eq!(
            report.skipped.as_deref(),
            Some("the target process has no capturable main window")
        );
        assert_eq!(report.to_json()["frames"], 0);
    }

    #[test]
    fn skip_reasons_are_stable_messages() {
        assert_eq!(
            RecordingSkip::Minimized.message(),
            "the target window is minimized and auto_restore_minimized is disabled"
        );
        assert_eq!(
            RecordingSkip::StillMinimized.message(),
            "the target window stayed minimized after an automatic restore attempt"
        );
        assert!(
            RecordingSkip::Unavailable("encoder busy".into())
                .message()
                .contains("encoder busy")
        );
    }
}
