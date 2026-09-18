//! No-op backend for platforms without Windows Graphics Capture.
//!
//! Recording is a Windows-only feature. On every other platform a session is
//! skipped with [`RecordingSkip::UnsupportedPlatform`] and no file is created,
//! so the crate and its callers compile and run unchanged.

use std::path::Path;

use crate::{RecordingConfig, RecordingReport, RecordingSkip, RecordingTarget};

/// Result of a platform start attempt.
pub(super) enum StartOutcome {
    /// A capture session is running.
    Started(Session),
    /// No session was started.
    Skipped(RecordingSkip),
}

/// Placeholder session; nothing is ever started on this platform.
#[derive(Debug)]
pub(super) struct Session;

/// Always skips: recording needs Windows Graphics Capture.
pub(super) fn start(
    target: RecordingTarget,
    config: &RecordingConfig,
    _base_dir: &Path,
) -> StartOutcome {
    if !config.enabled {
        return StartOutcome::Skipped(RecordingSkip::Disabled);
    }
    StartOutcome::Skipped(RecordingSkip::UnsupportedPlatform)
}

impl Session {
    /// Returns an empty report; no file was written.
    pub(super) fn stop(self) -> RecordingReport {
        RecordingReport::skipped(
            &RecordingTarget::new(0, "unsupported"),
            &RecordingSkip::UnsupportedPlatform,
        )
    }
}
