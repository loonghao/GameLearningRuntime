//! Windows backend: Windows Graphics Capture of one process' main window.
//!
//! The session resolves a process id to its main window, restores a minimized
//! window without activating it, and encodes captured frames to H.264 MP4 with
//! a per-frame index beside every segment.

use std::ffi::c_void;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use windows::Graphics::Capture::GraphicsCaptureSession;
use windows::Win32::Foundation::{HWND, LPARAM, RECT};
use windows::Win32::Graphics::Dwm::{DWMWA_CLOAKED, DwmGetWindowAttribute};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GW_OWNER, GetSystemMetrics, GetWindow, GetWindowRect, GetWindowThreadProcessId,
    HWND_BOTTOM, IsIconic, IsWindowVisible, SM_REMOTESESSION, SW_SHOWNOACTIVATE, SWP_NOACTIVATE,
    SWP_NOMOVE, SWP_NOSIZE, SetWindowPos, ShowWindow,
};
use windows::core::BOOL;
use windows_capture::capture::{CaptureControl, Context, GraphicsCaptureApiHandler};
use windows_capture::encoder::{
    AudioSettingsBuilder, ContainerSettingsBuilder, VideoEncoder, VideoSettingsBuilder,
    VideoSettingsSubType,
};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};
use windows_capture::window::Window;

use crate::canvas::{self, FrameGeometry};
use crate::config::RecordingConfig;
use crate::index::FrameIndexWriter;
use crate::{RecordingReport, RecordingSegment, RecordingSkip, RecordingTarget};

/// How long a session waits for the target process to create its main window.
const WINDOW_SEARCH_TIMEOUT: Duration = Duration::from_secs(3);
/// Poll interval while waiting for the main window.
const WINDOW_SEARCH_POLL: Duration = Duration::from_millis(100);
/// Capture and encoder timestamps are 100 ns ticks.
const TICKS_PER_SECOND: i64 = 10_000_000;

/// Result of a platform start attempt.
pub(super) enum StartOutcome {
    /// A capture session is running.
    Started(Session),
    /// No session was started.
    Skipped(RecordingSkip),
}

/// A top-level window considered for capture.
///
/// Kept free of window handles' lifetime concerns so the selection rule is unit
/// tested without a window station.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WindowCandidate {
    hwnd: usize,
    pid: u32,
    visible: bool,
    cloaked: bool,
    owned: bool,
    area: i64,
}

/// Picks the main window of `pid`: visible, not cloaked, unowned, largest area.
fn select_main_window(candidates: &[WindowCandidate], pid: u32) -> Option<&WindowCandidate> {
    candidates
        .iter()
        .filter(|candidate| {
            candidate.pid == pid
                && candidate.visible
                && !candidate.cloaked
                && !candidate.owned
                && candidate.area > 0
        })
        .max_by_key(|candidate| candidate.area)
}

/// Result of resolving a process id to a capturable window.
enum WindowLookup {
    /// A visible, capturable window was found.
    Found(HWND),
    /// A window exists but stays minimized (auto restore disabled).
    Minimized,
    /// A window exists but stayed minimized after an automatic restore attempt.
    StillMinimized,
    /// No capturable window appeared before the deadline.
    Missing,
}

struct WindowSearch {
    candidates: Vec<WindowCandidate>,
}

/// Collects every top-level window with the properties the selection needs.
///
/// # Safety
///
/// `lparam` must point at a [`WindowSearch`] that outlives the enumeration.
unsafe extern "system" fn collect_window(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let search = unsafe { &mut *(lparam.0 as *mut WindowSearch) };
    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    search.candidates.push(WindowCandidate {
        hwnd: hwnd.0 as usize,
        pid,
        visible: unsafe { IsWindowVisible(hwnd) }.as_bool(),
        cloaked: is_cloaked(hwnd),
        owned: unsafe { GetWindow(hwnd, GW_OWNER) }.is_ok(),
        area: window_area(hwnd),
    });
    BOOL::from(true)
}

/// Area of a window in square pixels; `0` when the rectangle is unavailable.
fn window_area(hwnd: HWND) -> i64 {
    let mut rect = RECT::default();
    if unsafe { GetWindowRect(hwnd, &mut rect) }.is_err() {
        return 0;
    }
    let width = i64::from(rect.right.saturating_sub(rect.left)).max(0);
    let height = i64::from(rect.bottom.saturating_sub(rect.top)).max(0);
    width * height
}

/// Whether the shell has cloaked the window (virtual desktops, UWP suspension).
fn is_cloaked(hwnd: HWND) -> bool {
    let mut cloaked = 0i32;
    let read = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_CLOAKED,
            std::ptr::addr_of_mut!(cloaked).cast::<c_void>(),
            u32::try_from(size_of::<i32>()).unwrap_or(4),
        )
    };
    read.is_ok() && cloaked != 0
}

/// Whether the window is currently minimized.
fn is_minimized(hwnd: HWND) -> bool {
    unsafe { IsIconic(hwnd) }.as_bool()
}

/// Restores a minimized window without stealing focus: shown, kept at the
/// bottom of the Z order, never activated.
fn restore_without_activation(hwnd: HWND) {
    unsafe {
        let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        // No SWP_NOZORDER: HWND_BOTTOM must apply, SWP_NOACTIVATE keeps focus.
        let _ = SetWindowPos(
            hwnd,
            Some(HWND_BOTTOM),
            0,
            0,
            0,
            0,
            SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE,
        );
    }
}

/// Largest capturable window owned by `pid`, if any.
fn main_window(pid: u32) -> Option<HWND> {
    let mut search = WindowSearch {
        candidates: Vec::new(),
    };
    let enumerated = unsafe {
        EnumWindows(
            Some(collect_window),
            LPARAM(std::ptr::addr_of_mut!(search) as isize),
        )
    };
    if enumerated.is_err() {
        return None;
    }
    select_main_window(&search.candidates, pid).map(|candidate| HWND(candidate.hwnd as *mut c_void))
}

/// Waits up to [`WINDOW_SEARCH_TIMEOUT`] for the target's main window.
fn wait_for_main_window(pid: u32, auto_restore_minimized: bool) -> WindowLookup {
    let deadline = Instant::now() + WINDOW_SEARCH_TIMEOUT;
    loop {
        let mut lookup = WindowLookup::Missing;
        if let Some(hwnd) = main_window(pid) {
            if is_minimized(hwnd) {
                if !auto_restore_minimized {
                    return WindowLookup::Minimized;
                }
                restore_without_activation(hwnd);
                lookup = WindowLookup::StillMinimized;
            } else {
                return WindowLookup::Found(hwnd);
            }
        }
        if Instant::now() >= deadline {
            return lookup;
        }
        thread::sleep(WINDOW_SEARCH_POLL);
    }
}

/// Whether Windows Graphics Capture is available on this device.
fn capture_supported() -> bool {
    GraphicsCaptureSession::IsSupported().unwrap_or(false)
}

/// Whether this is a remote desktop or server session, where capture is banned.
fn remote_session() -> bool {
    let session = unsafe { GetSystemMetrics(SM_REMOTESESSION) };
    session != 0
}

/// Resolves `output_dir` against `base_dir`, refusing to escape the base.
fn resolve_output_dir(base_dir: &Path, output_dir: &str) -> PathBuf {
    let configured = Path::new(output_dir);
    if configured.is_absolute() {
        return configured.to_path_buf();
    }
    let mut resolved = base_dir.to_path_buf();
    for component in configured.components() {
        match component {
            Component::Normal(part) => resolved.push(part),
            Component::CurDir => {}
            // Never let configuration escape the directory GLR owns.
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {}
        }
    }
    resolved
}

/// File-name safe process label.
fn sanitize_label(label: &str) -> String {
    let mut sanitized = String::new();
    for character in label.chars().take(32) {
        if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
            sanitized.push(character.to_ascii_lowercase());
        } else if !sanitized.ends_with('_') {
            sanitized.push('_');
        }
    }
    let trimmed = sanitized.trim_matches('_');
    if trimmed.is_empty() {
        "process".into()
    } else {
        trimmed.to_string()
    }
}

/// Civil date from days since the Unix epoch (Howard Hinnant's algorithm).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_progress = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_progress + 2) / 5 + 1) as u32;
    let month = if month_progress < 10 {
        month_progress + 3
    } else {
        month_progress - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// UTC timestamp used in file names, e.g. `20260918T024500Z`.
fn utc_stamp(now: Duration) -> String {
    let seconds = now.as_secs();
    let (year, month, day) = civil_from_days((seconds / 86_400) as i64);
    let time_of_day = seconds % 86_400;
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        time_of_day / 3600,
        time_of_day % 3600 / 60,
        time_of_day % 60
    )
}

/// Wall clock milliseconds, used by the per-frame index.
fn utc_now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis() as i64)
        .unwrap_or_default()
}

/// Everything a recording session needs to open and name its segments.
struct RecordingFlags {
    config: RecordingConfig,
    target: RecordingTarget,
    directory: PathBuf,
    stem: String,
}

impl RecordingFlags {
    /// Paths of one segment: the MP4 and its frame index.
    fn segment_paths(&self, segment: u64) -> (PathBuf, PathBuf) {
        (
            self.directory
                .join(format!("{}_seg{segment}.mp4", self.stem)),
            self.directory
                .join(format!("{}_seg{segment}.frames.jsonl", self.stem)),
        )
    }
}

/// Handler error reported back through the capture thread.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("recording handler failed: {0}")]
struct HandlerError(String);

/// Capture callback writing frames to the encoder and the frame index.
struct RecordingHandler {
    flags: RecordingFlags,
    geometry: Option<FrameGeometry>,
    /// When true frames go straight to the encoder as GPU surfaces.
    direct: bool,
    segment: u64,
    segment_frames: u64,
    segment_start_ticks: i64,
    frames: u64,
    encoder: Option<VideoEncoder>,
    index: Option<FrameIndexWriter>,
    /// Reused source buffer; never reallocated per frame.
    scratch: Vec<u8>,
    /// Reused destination canvas; keeps the stream resolution stable.
    canvas: Vec<u8>,
    segments: Vec<RecordingSegment>,
    warnings: Vec<String>,
    error: Option<String>,
}

impl RecordingHandler {
    fn new(flags: RecordingFlags) -> Self {
        Self {
            flags,
            geometry: None,
            direct: true,
            segment: 0,
            segment_frames: 0,
            segment_start_ticks: 0,
            frames: 0,
            encoder: None,
            index: None,
            scratch: Vec::new(),
            canvas: Vec::new(),
            segments: Vec::new(),
            warnings: Vec::new(),
            error: None,
        }
    }

    /// Whether the current segment reached `segment_seconds`.
    fn segment_expired(&self, ticks: i64) -> bool {
        let limit = self.flags.config.segment_seconds;
        if limit == 0 {
            return false;
        }
        let length = i64::try_from(limit)
            .unwrap_or(i64::MAX)
            .saturating_mul(TICKS_PER_SECOND);
        ticks.saturating_sub(self.segment_start_ticks) >= length
    }

    /// Encodes one frame and records it in the index.
    fn record(&mut self, frame: &mut Frame) -> Result<(), String> {
        let ticks = frame
            .timestamp()
            .map(|value| value.Duration)
            .unwrap_or_default();
        let utc_ms = utc_now_ms();
        let size = FrameGeometry {
            width: frame.width(),
            height: frame.height(),
        };
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let geometry = match self.geometry {
            Some(geometry) => geometry,
            None => {
                let planned =
                    canvas::plan_size(size.width, size.height, self.flags.config.max_width);
                self.direct = planned.width == size.width && planned.height == size.height;
                self.canvas = vec![0u8; planned.byte_len()];
                self.geometry = Some(planned);
                planned
            }
        };
        if self.encoder.is_none() || self.segment_expired(ticks) {
            self.rotate(ticks, geometry)?;
        }
        if self.direct {
            let encoder = self.encoder.as_mut().ok_or("encoder is not open")?;
            encoder
                .send_frame(frame)
                .map_err(|error| format!("encoder rejected the frame: {error}"))?;
        } else {
            let buffer = frame
                .buffer()
                .map_err(|error| format!("frame buffer unavailable: {error}"))?;
            let source = buffer.as_nopadding_buffer(&mut self.scratch);
            canvas::composite_bgra(
                source,
                size.width,
                size.height,
                &mut self.canvas,
                geometry.width,
                geometry.height,
            );
            let encoder = self.encoder.as_mut().ok_or("encoder is not open")?;
            encoder
                .send_frame_buffer(&self.canvas, ticks)
                .map_err(|error| format!("encoder rejected the frame: {error}"))?;
        }
        let index = self.index.as_mut().ok_or("frame index is not open")?;
        index
            .write(self.frames, ticks, utc_ms)
            .map_err(|error| format!("frame index write failed: {error}"))?;
        self.frames += 1;
        self.segment_frames += 1;
        Ok(())
    }

    /// Finalizes the current segment and opens the next one.
    fn rotate(&mut self, ticks: i64, geometry: FrameGeometry) -> Result<(), String> {
        self.close_segment();
        let (video, index) = self.flags.segment_paths(self.segment);
        let encoder = VideoEncoder::new(
            VideoSettingsBuilder::new(geometry.width, geometry.height)
                .sub_type(VideoSettingsSubType::H264)
                .bitrate(self.flags.config.bitrate_mbps.saturating_mul(1_000_000))
                .frame_rate(self.flags.config.fps),
            AudioSettingsBuilder::default().disabled(true),
            ContainerSettingsBuilder::default(),
            &video,
        )
        .map_err(|error| format!("encoder unavailable: {error}"))?;
        let index = FrameIndexWriter::create(&index)
            .map_err(|error| format!("frame index unavailable: {error}"))?;
        self.encoder = Some(encoder);
        self.index = Some(index);
        self.segment_start_ticks = ticks;
        Ok(())
    }

    /// Finalizes the open segment and records it in the report.
    fn close_segment(&mut self) {
        let Some(encoder) = self.encoder.take() else {
            return;
        };
        if let Err(error) = encoder.finish() {
            self.warnings
                .push(format!("segment {} finalize failed: {error}", self.segment));
        }
        if let Some(mut index) = self.index.take() {
            let _ = index.flush();
        }
        let (video, index_path) = self.flags.segment_paths(self.segment);
        if self.segment_frames > 0 {
            self.segments.push(RecordingSegment {
                video,
                index: index_path,
                frames: self.segment_frames,
            });
        }
        self.segment += 1;
        self.segment_frames = 0;
    }

    /// Finalizes everything still open.
    fn finish(&mut self) {
        self.close_segment();
    }
}

impl GraphicsCaptureApiHandler for RecordingHandler {
    type Flags = RecordingFlags;
    type Error = HandlerError;

    fn new(context: Context<Self::Flags>) -> Result<Self, Self::Error> {
        Ok(Self::new(context.flags))
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        if let Err(error) = self.record(frame) {
            self.error = Some(error.clone());
            self.finish();
            capture_control.stop();
            return Err(HandlerError(error));
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        self.finish();
        Ok(())
    }
}

/// A running Windows capture session.
pub(super) struct Session {
    control: Option<CaptureControl<RecordingHandler, HandlerError>>,
    handler: Arc<Mutex<RecordingHandler>>,
}

impl Session {
    /// Stops capture and reports what was written.
    pub(super) fn stop(mut self) -> RecordingReport {
        self.shutdown();
        let handler = self.handler.lock();
        report(&handler)
    }

    /// Stops the capture thread and finalizes any open segment.
    fn shutdown(&mut self) {
        let Some(control) = self.control.take() else {
            return;
        };
        if control.stop().is_err() {
            self.handler
                .lock()
                .warnings
                .push("capture thread did not stop cleanly".into());
        }
        self.handler.lock().finish();
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Session")
            .field("capturing", &self.control.is_some())
            .finish()
    }
}

/// Builds the report from a finished handler.
fn report(handler: &RecordingHandler) -> RecordingReport {
    let geometry = handler.geometry.unwrap_or(FrameGeometry {
        width: 0,
        height: 0,
    });
    let skipped = if handler.segments.is_empty() {
        Some(
            handler
                .error
                .clone()
                .unwrap_or_else(|| "no frame was delivered before the session ended".into()),
        )
    } else {
        None
    };
    RecordingReport {
        pid: handler.flags.target.pid,
        label: handler.flags.target.label.clone(),
        frames: handler.frames,
        width: geometry.width,
        height: geometry.height,
        segments: handler.segments.clone(),
        warnings: handler.warnings.clone(),
        skipped,
    }
}

/// Starts a Windows recording session for `target`.
pub(super) fn start(
    target: RecordingTarget,
    config: &RecordingConfig,
    base_dir: &Path,
) -> StartOutcome {
    if !config.enabled {
        return StartOutcome::Skipped(RecordingSkip::Disabled);
    }
    if !capture_supported() {
        return StartOutcome::Skipped(RecordingSkip::CaptureUnsupported);
    }
    if remote_session() {
        return StartOutcome::Skipped(RecordingSkip::RemoteSession);
    }
    let hwnd = match wait_for_main_window(target.pid, config.auto_restore_minimized) {
        WindowLookup::Found(hwnd) => hwnd,
        WindowLookup::Minimized => return StartOutcome::Skipped(RecordingSkip::Minimized),
        WindowLookup::StillMinimized => {
            return StartOutcome::Skipped(RecordingSkip::StillMinimized);
        }
        WindowLookup::Missing => return StartOutcome::Skipped(RecordingSkip::NoMainWindow),
    };
    let window = Window::from_raw_hwnd(hwnd.0);
    let label = window
        .process_name()
        .ok()
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| target.label.clone());
    let directory = resolve_output_dir(base_dir, &config.output_dir);
    if let Err(error) = fs::create_dir_all(&directory) {
        return StartOutcome::Skipped(RecordingSkip::Unavailable(format!(
            "output directory {} unusable: {error}",
            directory.display()
        )));
    }
    let stem = format!(
        "{}_{}_{}",
        utc_stamp(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
        ),
        target.pid,
        sanitize_label(&label)
    );
    let flags = RecordingFlags {
        config: config.clone(),
        target: target.clone(),
        directory,
        stem,
    };
    let settings = Settings::new(
        window,
        if config.include_cursor {
            CursorCaptureSettings::WithCursor
        } else {
            CursorCaptureSettings::WithoutCursor
        },
        DrawBorderSettings::Default,
        SecondaryWindowSettings::Include,
        MinimumUpdateIntervalSettings::Custom(Duration::from_nanos(config.frame_interval_nanos())),
        DirtyRegionSettings::ReportOnly,
        ColorFormat::Bgra8,
        flags,
    );
    match RecordingHandler::start_free_threaded(settings) {
        Ok(control) => {
            let handler = control.callback();
            StartOutcome::Started(Session {
                control: Some(control),
                handler,
            })
        }
        Err(error) => StartOutcome::Skipped(RecordingSkip::Unavailable(format!(
            "capture start failed: {error}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        WindowCandidate, civil_from_days, resolve_output_dir, sanitize_label, select_main_window,
        utc_stamp,
    };
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    fn candidate(hwnd: usize, pid: u32, area: i64) -> WindowCandidate {
        WindowCandidate {
            hwnd,
            pid,
            visible: true,
            cloaked: false,
            owned: false,
            area,
        }
    }

    #[test]
    fn selects_the_largest_window_of_the_process() {
        let candidates = [
            candidate(1, 10, 100),
            candidate(2, 11, 90_000),
            candidate(3, 10, 40_000),
            candidate(4, 10, 80_000),
        ];
        let selected = select_main_window(&candidates, 10).expect("a window of pid 10");
        assert_eq!(selected.hwnd, 4);
    }

    #[test]
    fn rejects_invisible_cloaked_owned_and_empty_windows() {
        let mut hidden = candidate(1, 10, 500);
        hidden.visible = false;
        let mut cloaked = candidate(2, 10, 600);
        cloaked.cloaked = true;
        let mut owned = candidate(3, 10, 700);
        owned.owned = true;
        let empty = candidate(4, 10, 0);
        let candidates = [hidden, cloaked, owned, empty];
        assert!(select_main_window(&candidates, 10).is_none());
    }

    #[test]
    fn ignores_other_processes() {
        let candidates = [candidate(1, 12, 1000)];
        assert!(select_main_window(&candidates, 10).is_none());
    }

    #[test]
    fn output_dir_stays_inside_the_base() {
        let base = PathBuf::from("C:\\project\\run");
        assert_eq!(
            resolve_output_dir(&base, "recordings"),
            Path::new("C:\\project\\run\\recordings")
        );
        assert_eq!(
            resolve_output_dir(&base, "../escape"),
            Path::new("C:\\project\\run\\escape")
        );
        assert_eq!(
            resolve_output_dir(&base, "D:\\absolute"),
            Path::new("D:\\absolute")
        );
    }

    #[test]
    fn labels_are_file_name_safe() {
        assert_eq!(sanitize_label("My Game.exe"), "my_game_exe");
        assert_eq!(sanitize_label("glr"), "glr");
        assert_eq!(sanitize_label(""), "process");
        assert_eq!(sanitize_label("___"), "process");
    }

    #[test]
    fn utc_stamp_matches_the_contract() {
        assert_eq!(
            utc_stamp(Duration::from_secs(1_757_000_000)),
            "20250904T153320Z"
        );
    }

    #[test]
    fn civil_from_days_covers_the_epoch_and_a_leap_year() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 2021-05-15 is 18762 days after the epoch; 2020 was a leap year.
        assert_eq!(civil_from_days(18_762), (2021, 5, 15));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }
}
