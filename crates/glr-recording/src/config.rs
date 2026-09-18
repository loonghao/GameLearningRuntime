//! Recording configuration: defaults, TOML loading and sanitization.
//!
//! Every field falls back to a documented default. A missing, unreadable or
//! invalid configuration is never fatal: [`RecordingConfig::load`] returns the
//! defaults so a training run can proceed.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Default output directory, relative to the run or project directory.
pub const DEFAULT_OUTPUT_DIR: &str = "recordings";
/// Highest frame rate accepted from configuration.
pub const MAX_FPS: u32 = 60;
/// Highest long-edge limit accepted from configuration.
pub const MAX_WIDTH_LIMIT: u32 = 7680;
/// Highest bitrate accepted from configuration, in megabits per second.
pub const MAX_BITRATE_MBPS: u32 = 200;

/// Recording behaviour for a project.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecordingConfig {
    /// Master switch. When false no capture thread starts and no file is created.
    pub enabled: bool,
    /// Output directory; relative paths resolve against the run or project root.
    pub output_dir: String,
    /// Target frame rate used for the capture interval and the encoded stream.
    pub fps: u32,
    /// Long-edge cap; larger windows are downscaled and even-aligned.
    pub max_width: u32,
    /// Encoder target bitrate in megabits per second.
    pub bitrate_mbps: u32,
    /// Segment length in seconds; `0` writes a single segment.
    pub segment_seconds: u64,
    /// Whether the system cursor is composited into the recording.
    pub include_cursor: bool,
    /// Restore (without activating) a minimized target window before capture.
    pub auto_restore_minimized: bool,
    /// Reserved for process loopback audio, which is not implemented yet.
    pub audio: bool,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            output_dir: DEFAULT_OUTPUT_DIR.into(),
            fps: 30,
            max_width: 1920,
            bitrate_mbps: 6,
            segment_seconds: 600,
            include_cursor: false,
            auto_restore_minimized: true,
            audio: false,
        }
    }
}

/// Why a configuration file could not be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecordingConfigError {
    /// The file could not be read.
    #[error("recording configuration unreadable: {0}")]
    Unreadable(String),
    /// The TOML did not parse.
    #[error("recording configuration is invalid TOML: {0}")]
    Invalid(String),
    /// The TOML parsed but carries no `[recording]` table.
    #[error("recording configuration has no [recording] table")]
    MissingTable,
}

#[derive(Deserialize)]
struct RecordingDocument {
    #[serde(default)]
    recording: Option<RecordingConfig>,
}

impl RecordingConfig {
    /// Parses a TOML document carrying a `[recording]` table.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingConfigError`] when the text is not valid TOML or the
    /// table is missing.
    pub fn from_toml(text: &str) -> Result<Self, RecordingConfigError> {
        let document: RecordingDocument = toml::from_str(text)
            .map_err(|error| RecordingConfigError::Invalid(error.to_string()))?;
        let mut config = document
            .recording
            .ok_or(RecordingConfigError::MissingTable)?;
        let _ = config.sanitize();
        Ok(config)
    }

    /// Loads a configuration file, falling back to defaults on any failure.
    #[must_use]
    pub fn load(path: &Path) -> Self {
        load_with_warnings(path).0
    }

    /// Clamps out-of-range values in place and returns the applied corrections.
    pub fn sanitize(&mut self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.fps == 0 || self.fps > MAX_FPS {
            warnings.push(format!(
                "recording.fps {} out of range; using {}",
                self.fps,
                self.fps.clamp(1, MAX_FPS)
            ));
            self.fps = self.fps.clamp(1, MAX_FPS);
        }
        if self.max_width > MAX_WIDTH_LIMIT {
            warnings.push(format!(
                "recording.max_width {} exceeds {MAX_WIDTH_LIMIT}; using {MAX_WIDTH_LIMIT}",
                self.max_width
            ));
            self.max_width = MAX_WIDTH_LIMIT;
        }
        if self.max_width == 1 {
            warnings.push("recording.max_width 1 is too small; using 2".into());
            self.max_width = 2;
        }
        if self.bitrate_mbps == 0 || self.bitrate_mbps > MAX_BITRATE_MBPS {
            warnings.push(format!(
                "recording.bitrate_mbps {} out of range; using {}",
                self.bitrate_mbps,
                self.bitrate_mbps.clamp(1, MAX_BITRATE_MBPS)
            ));
            self.bitrate_mbps = self.bitrate_mbps.clamp(1, MAX_BITRATE_MBPS);
        }
        if self.output_dir.trim().is_empty() {
            warnings.push(format!(
                "recording.output_dir is empty; using {DEFAULT_OUTPUT_DIR:?}"
            ));
            self.output_dir = DEFAULT_OUTPUT_DIR.into();
        }
        if self.audio {
            warnings.push(
                "recording.audio is not implemented; audio is ignored for this release".into(),
            );
            self.audio = false;
        }
        warnings
    }

    /// Frames per second as a capture interval, never dividing by zero.
    #[must_use]
    pub const fn frame_interval_nanos(&self) -> u64 {
        let fps = if self.fps == 0 { 1 } else { self.fps };
        1_000_000_000 / fps as u64
    }
}

/// Loads a configuration file together with the warnings it produced.
#[must_use]
pub fn load_with_warnings(path: &Path) -> (RecordingConfig, Vec<String>) {
    let fallback = |reason: std::fmt::Arguments<'_>| {
        (
            RecordingConfig::default(),
            vec![format!(
                "recording configuration {} unusable ({reason}); using defaults",
                path.display()
            )],
        )
    };
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => return fallback(format_args!("{error}")),
    };
    let document: RecordingDocument = match toml::from_str(&text) {
        Ok(document) => document,
        Err(error) => return fallback(format_args!("{error}")),
    };
    let Some(mut config) = document.recording else {
        return (RecordingConfig::default(), Vec::new());
    };
    let warnings = config.sanitize();
    (config, warnings)
}

#[cfg(test)]
mod tests {
    use super::{MAX_FPS, RecordingConfig, RecordingConfigError};

    #[test]
    fn defaults_match_the_contract() {
        let config = RecordingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.output_dir, "recordings");
        assert_eq!(config.fps, 30);
        assert_eq!(config.max_width, 1920);
        assert_eq!(config.bitrate_mbps, 6);
        assert_eq!(config.segment_seconds, 600);
        assert!(!config.include_cursor);
        assert!(config.auto_restore_minimized);
        assert!(!config.audio);
    }

    #[test]
    fn toml_table_overrides_defaults() {
        let config = RecordingConfig::from_toml(
            "[recording]\nenabled = false\nfps = 24\noutput_dir = \"clips\"\nsegment_seconds = 60\n",
        )
        .expect("table parses");
        assert!(!config.enabled);
        assert_eq!(config.fps, 24);
        assert_eq!(config.output_dir, "clips");
        assert_eq!(config.segment_seconds, 60);
        assert_eq!(config.max_width, 1920);
    }

    #[test]
    fn missing_table_is_reported() {
        let error = RecordingConfig::from_toml("[other]\nenabled = false\n")
            .expect_err("no recording table");
        assert_eq!(error, RecordingConfigError::MissingTable);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        assert!(RecordingConfig::from_toml("[recording]\nfps = 30\nmystery = 1\n").is_err());
    }

    #[test]
    fn sanitize_clamps_out_of_range_values() {
        let mut config = RecordingConfig {
            fps: 0,
            max_width: 100_000,
            bitrate_mbps: 0,
            output_dir: "  ".into(),
            audio: true,
            ..RecordingConfig::default()
        };
        let warnings = config.sanitize();
        assert_eq!(config.fps, 1);
        assert_eq!(config.max_width, 7680);
        assert_eq!(config.bitrate_mbps, 1);
        assert_eq!(config.output_dir, "recordings");
        assert!(!config.audio);
        assert_eq!(warnings.len(), 5);
    }

    #[test]
    fn sanitize_reports_fps_ceiling() {
        let mut config = RecordingConfig {
            fps: MAX_FPS + 1,
            ..RecordingConfig::default()
        };
        assert!(!config.sanitize().is_empty());
        assert_eq!(config.fps, MAX_FPS);
    }

    #[test]
    fn frame_interval_is_finite() {
        let mut config = RecordingConfig::default();
        assert_eq!(config.frame_interval_nanos(), 33_333_333);
        config.fps = 0;
        assert_eq!(config.frame_interval_nanos(), 1_000_000_000);
    }
}
