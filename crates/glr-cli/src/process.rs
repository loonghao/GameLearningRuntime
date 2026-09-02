use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::contracts::{CaptureManifestInput, build_capture_manifest};
use crate::error::{Error, Result};
use crate::project::{CaptureSessionConfig, Project, ProjectCommand};
use crate::store::Store;

pub const CAPTURE_SESSION_SCHEMA_VERSION: &str = "glr.capture-session.v1";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CaptureState {
    Starting,
    Healthy,
    Degraded,
    Stopped,
    Failed,
    Completed,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CaptureStatus {
    schema_version: String,
    session_id: String,
    state: CaptureState,
    frames_written: u64,
    #[serde(default)]
    steps_written: u64,
    #[serde(default)]
    last_frame_timestamp_ns: Option<u64>,
    #[serde(default)]
    dropped_frames: u64,
    recorder_state: String,
    timestamp_ns: u64,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaptureStartReceipt {
    pub schema_version: &'static str,
    pub session_id: String,
    pub run_id: String,
    pub video_path: String,
    pub index_path: String,
    pub status_path: String,
    pub codec: String,
    pub frame_rate: f64,
    pub width: u32,
    pub height: u32,
    pub started_at_ns: u64,
    pub recorder_pid: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaptureLifecycle {
    pub schema_version: &'static str,
    pub receipt: CaptureStartReceipt,
    pub state: CaptureState,
    pub reason: Option<String>,
    pub frames_written: u64,
    pub steps_written: u64,
    pub last_frame_timestamp_ns: Option<u64>,
    pub dropped_frames: u64,
    pub finished_at_ns: u64,
}

pub struct CaptureSession {
    child: Option<Child>,
    log_path: PathBuf,
    status_path: PathBuf,
    receipt_path: PathBuf,
    receipt: CaptureStartReceipt,
    config: Option<CaptureSessionConfig>,
    startup_error: Option<String>,
}

pub fn command_context(
    project: &Project,
    run_id: &str,
    run_dir: &Path,
    bundle: Option<&Path>,
    extra: &HashMap<String, PathBuf>,
) -> HashMap<String, PathBuf> {
    let mut values = HashMap::from([
        ("project_root".into(), project.root.clone()),
        ("bridge_path".into(), project.bridge_path.clone()),
        ("run_id".into(), PathBuf::from(run_id)),
        ("run_dir".into(), run_dir.to_path_buf()),
        (
            "capture_video".into(),
            run_dir.join(
                project
                    .capture
                    .as_ref()
                    .map_or("capture.mp4", |value| value.video_file.as_str()),
            ),
        ),
        (
            "capture_index".into(),
            run_dir.join(
                project
                    .capture
                    .as_ref()
                    .map_or("capture-index.jsonl", |value| value.index_file.as_str()),
            ),
        ),
        (
            "capture_status".into(),
            run_dir.join(
                project
                    .capture
                    .as_ref()
                    .and_then(|value| value.session.as_ref())
                    .map_or("capture-status.jsonl", |value| value.status_file.as_str()),
            ),
        ),
    ]);
    if let Some(bundle) = bundle {
        values.insert("bundle".into(), bundle.to_path_buf());
    }
    for (key, value) in extra {
        values.insert(key.clone(), value.clone());
    }
    values
}

fn configure_command(
    command: &ProjectCommand,
    project: &Project,
    run_id: &str,
    run_dir: &Path,
    bundle: Option<&Path>,
    extra: &HashMap<String, PathBuf>,
) -> Result<Command> {
    let context = command_context(project, run_id, run_dir, bundle, extra);
    let argv = command.expand(&context)?;
    let (program, arguments) = argv
        .split_first()
        .ok_or_else(|| Error::Invalid("project command is empty".into()))?;
    let mut process = Command::new(program);
    process
        .args(arguments)
        .current_dir(&project.root)
        .env("GLR_PROJECT_ROOT", &project.root)
        .env("GLR_BRIDGE_PATH", &project.bridge_path)
        .env("GLR_RUN_ID", run_id)
        .env("GLR_RUN_DIR", run_dir)
        .env("GLR_STORE_PATH", project.data_dir.join("runs.sqlite3"))
        .env("GLR_ENVIRONMENT_ID", &project.environment_id)
        .env("GLR_ENVIRONMENT_FAMILY", &project.environment_family)
        .env("GLR_PROTOCOL_VERSION", &project.protocol_version)
        .env("GLR_CAPTURE_VIDEO", &context["capture_video"])
        .env("GLR_CAPTURE_INDEX", &context["capture_index"])
        .env("GLR_CAPTURE_STATUS", &context["capture_status"]);
    if let Some(progress) = &project.progress {
        process
            .env("GLR_PROGRESS_SIGNAL", &progress.signal)
            .env(
                "GLR_PROGRESS_WINDOW_STEPS",
                progress.window_steps.to_string(),
            )
            .env(
                "GLR_PROGRESS_MAX_STALLED_ROUNDS",
                progress.max_stalled_rounds.to_string(),
            );
    }
    if let Some(bundle) = bundle {
        process.env("GLR_MODEL_BUNDLE", bundle);
    }
    for (key, value) in extra {
        process.env(format!("GLR_{}", key.to_ascii_uppercase()), value);
    }
    Ok(process)
}

pub struct CommandInvocation<'a> {
    pub command: &'a ProjectCommand,
    pub project: &'a Project,
    pub run_id: &'a str,
    pub run_dir: &'a Path,
    pub log_path: &'a Path,
    pub bundle: Option<&'a Path>,
    pub extra: &'a HashMap<String, PathBuf>,
    pub timeout: Option<Duration>,
}

pub fn run_command(invocation: CommandInvocation<'_>) -> Result<i32> {
    if let Some(parent) = invocation.log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log = File::create(invocation.log_path)?;
    let stderr = log.try_clone()?;
    let mut process = configure_command(
        invocation.command,
        invocation.project,
        invocation.run_id,
        invocation.run_dir,
        invocation.bundle,
        invocation.extra,
    )?;
    let mut child = process
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr))
        .spawn()?;
    wait_for_child(&mut child, invocation.timeout)
}

pub fn start_capture(project: &Project, run_id: &str, run_dir: &Path) -> Result<CaptureSession> {
    let capture = project
        .capture
        .as_ref()
        .ok_or_else(|| Error::Contract("capture configuration is missing".into()))?;
    let session_id = format!("capture-{}", Uuid::new_v4().simple());
    let log_path = run_dir.join("capture.log");
    let status_path = run_dir.join(
        capture
            .session
            .as_ref()
            .map_or("capture-status.jsonl", |value| value.status_file.as_str()),
    );
    let receipt_path = run_dir.join("capture-session.json");
    reject_symlink(&status_path)?;
    reject_symlink(&receipt_path)?;
    if let Some(parent) = status_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if capture.session.is_some() {
        File::create(&status_path)?;
    }
    let mut receipt = CaptureStartReceipt {
        schema_version: CAPTURE_SESSION_SCHEMA_VERSION,
        session_id,
        run_id: run_id.into(),
        video_path: relative_portable(run_dir, &run_dir.join(&capture.video_file))?,
        index_path: relative_portable(run_dir, &run_dir.join(&capture.index_file))?,
        status_path: relative_portable(run_dir, &status_path)?,
        codec: capture.codec.clone(),
        frame_rate: capture.frame_rate,
        width: capture.width,
        height: capture.height,
        started_at_ns: now_ns(),
        recorder_pid: None,
    };
    write_json_file(&receipt_path, &receipt)?;
    let log = File::create(&log_path)?;
    let stderr = log.try_clone()?;
    let command = capture.command();
    let mut process = configure_command(&command, project, run_id, run_dir, None, &HashMap::new())?;
    process
        .env("GLR_CAPTURE_SESSION_ID", &receipt.session_id)
        .env("GLR_CAPTURE_RECEIPT", &receipt_path);
    let spawn = process
        .stdin(if capture.stop == "stdin-q" {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr))
        .spawn();
    let (child, startup_error) = match spawn {
        Ok(mut child) => {
            receipt.recorder_pid = Some(child.id());
            write_json_file(&receipt_path, &receipt)?;
            let startup_error = capture.session.as_ref().and_then(|config| {
                wait_for_healthy(&status_path, &receipt.session_id, &mut child, config)
            });
            (Some(child), startup_error)
        }
        Err(error) => (None, Some(format!("recorder failed to start: {error}"))),
    };
    Ok(CaptureSession {
        child,
        log_path,
        status_path,
        receipt_path,
        receipt,
        config: capture.session.clone(),
        startup_error,
    })
}

pub fn finish_capture(
    project: &Project,
    store: &Store,
    run_id: &str,
    capture_dir: &Path,
    artifact_root: &Path,
    mut session: CaptureSession,
) -> Result<CaptureLifecycle> {
    let capture = project
        .capture
        .as_ref()
        .ok_or_else(|| Error::Contract("capture configuration is missing".into()))?;
    let mut reasons = Vec::new();
    if session.receipt_path.is_symlink() {
        reasons.push("capture start receipt must not be a symlink".into());
    }
    if session.log_path.is_symlink() {
        reasons.push("capture log must not be a symlink".into());
    }
    let capture_exit = stop_capture(capture.stop.as_str(), &mut session.child, 10, &mut reasons);
    let latest = match read_capture_statuses(&session.status_path, &session.receipt.session_id) {
        Ok(value) => value,
        Err(error) => {
            reasons.push(error);
            None
        }
    };
    let video = capture_dir.join(&capture.video_file);
    let index = capture_dir.join(&capture.index_file);
    let manifest = capture_dir.join("capture.manifest.json");
    let mut manifest_valid = false;
    if capture_exit == Some(0) && video.is_file() && index.is_file() {
        match build_capture_manifest(CaptureManifestInput {
            manifest_path: &manifest,
            environment_id: &project.environment_id,
            run_id,
            video_path: &video,
            index_path: &index,
            codec: &capture.codec,
            frame_rate: capture.frame_rate,
            width: capture.width,
            height: capture.height,
        }) {
            Ok(_) => manifest_valid = true,
            Err(error) => reasons.push(format!("capture manifest invalid: {error}")),
        }
    } else {
        if capture_exit != Some(0) {
            reasons.push(format!("recorder exited with {:?}", capture_exit));
        }
        if !video.is_file() {
            reasons.push("capture video was not produced".into());
        }
        if !index.is_file() {
            reasons.push("capture index was not produced".into());
        }
    }
    let (frames_written, steps_written, last_frame_timestamp_ns, dropped_frames, terminal_state) =
        latest.as_ref().map_or((0, 0, None, 0, None), |status| {
            (
                status.frames_written,
                status.steps_written,
                status.last_frame_timestamp_ns,
                status.dropped_frames,
                Some(status.state),
            )
        });
    if let Some(error) = &session.startup_error {
        reasons.push(error.clone());
    }
    if let Some(config) = &session.config {
        match latest.as_ref() {
            None => reasons.push("capture recorder did not publish a status heartbeat".into()),
            Some(status) => {
                if status.state != CaptureState::Completed {
                    reasons.push(format!("capture ended in {:?}", status.state));
                }
                if status.state != CaptureState::Completed
                    && heartbeat_is_stale(status, config.heartbeat_timeout_seconds, now_ns())
                {
                    reasons.push("capture heartbeat stalled".into());
                }
                if status.frames_written > 0 && status.last_frame_timestamp_ns.is_none() {
                    reasons.push("capture frame progress has no timestamp".into());
                }
                if status.frames_written < config.minimum_frames {
                    reasons.push(format!(
                        "capture wrote {} frames, minimum is {}",
                        status.frames_written, config.minimum_frames
                    ));
                }
                if status.steps_written < config.minimum_steps {
                    reasons.push(format!(
                        "capture wrote {} steps, minimum is {}",
                        status.steps_written, config.minimum_steps
                    ));
                }
            }
        }
        if latest.is_none() {
            reasons.push("capture startup handshake was not observed".into());
        }
    }
    let state = if reasons.is_empty() && manifest_valid {
        CaptureState::Completed
    } else if terminal_state == Some(CaptureState::Degraded) {
        CaptureState::Degraded
    } else if terminal_state == Some(CaptureState::Stopped) {
        CaptureState::Stopped
    } else if !capture.required && terminal_state != Some(CaptureState::Failed) {
        CaptureState::Degraded
    } else {
        CaptureState::Failed
    };
    let lifecycle = CaptureLifecycle {
        schema_version: CAPTURE_SESSION_SCHEMA_VERSION,
        receipt: session.receipt.clone(),
        state,
        reason: (!reasons.is_empty()).then(|| reasons.join("; ")),
        frames_written,
        steps_written,
        last_frame_timestamp_ns,
        dropped_frames,
        finished_at_ns: now_ns(),
    };
    for (path, role, media_type) in [
        (&session.log_path, "capture-log", "text/plain"),
        (&session.receipt_path, "capture-session", "application/json"),
    ] {
        if path.is_file() && !path.is_symlink() {
            store.register_artifact(
                run_id,
                &relative_portable(artifact_root, path)?,
                path,
                role,
                media_type,
            )?;
        }
    }
    if session.config.is_some()
        && session.status_path.is_file()
        && !session.status_path.is_symlink()
    {
        store.register_artifact(
            run_id,
            &relative_portable(artifact_root, &session.status_path)?,
            &session.status_path,
            "capture-status",
            "application/x-ndjson",
        )?;
    }
    if lifecycle.state == CaptureState::Completed {
        for (path, role, media_type) in [
            (&video, "review-video", "video/mp4"),
            (&index, "capture-index", "application/x-ndjson"),
            (&manifest, "capture-manifest", "application/json"),
        ] {
            store.register_artifact(
                run_id,
                &relative_portable(artifact_root, path)?,
                path,
                role,
                media_type,
            )?;
        }
    }
    store.append_event(
        run_id,
        "capture.lifecycle",
        serde_json::to_value(&lifecycle)?,
    )?;
    Ok(lifecycle)
}

fn stop_capture(
    stop: &str,
    child: &mut Option<Child>,
    timeout_seconds: u64,
    reasons: &mut Vec<String>,
) -> Option<i32> {
    let child = child.as_mut()?;
    match child.try_wait() {
        Ok(Some(status)) => return status.code(),
        Ok(None) => {}
        Err(error) => reasons.push(format!("could not inspect recorder: {error}")),
    }
    if stop == "stdin-q" {
        if let Some(stdin) = child.stdin.as_mut() {
            let _ = stdin.write_all(b"q\n");
            let _ = stdin.flush();
        }
    } else if let Err(error) = child.kill() {
        reasons.push(format!("could not stop recorder: {error}"));
    }
    match wait_for_child(child, Some(Duration::from_secs(timeout_seconds))) {
        Ok(code) => Some(code),
        Err(error) => {
            reasons.push(error.to_string());
            None
        }
    }
}

fn wait_for_healthy(
    status_path: &Path,
    session_id: &str,
    child: &mut Child,
    config: &CaptureSessionConfig,
) -> Option<String> {
    let deadline = Instant::now() + Duration::from_secs_f64(config.startup_timeout_seconds);
    loop {
        match read_capture_statuses(status_path, session_id) {
            Ok(Some(status)) if status.state == CaptureState::Healthy => return None,
            Ok(Some(status))
                if matches!(
                    status.state,
                    CaptureState::Failed | CaptureState::Degraded | CaptureState::Stopped
                ) =>
            {
                return Some(format!("capture startup ended in {:?}", status.state));
            }
            Err(error) => return Some(error),
            _ => {}
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                return Some(format!(
                    "recorder exited before startup handshake with {:?}",
                    status.code()
                ));
            }
            Ok(None) => {}
            Err(error) => return Some(format!("could not inspect recorder: {error}")),
        }
        if Instant::now() >= deadline {
            return Some(format!(
                "capture startup handshake timed out after {:.1}s",
                config.startup_timeout_seconds
            ));
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn read_capture_statuses(
    path: &Path,
    session_id: &str,
) -> std::result::Result<Option<CaptureStatus>, String> {
    if path.is_symlink() {
        return Err("capture status must not be a symlink".into());
    }
    if !path.is_file() {
        return Ok(None);
    }
    let file =
        File::open(path).map_err(|error| format!("could not read capture status: {error}"))?;
    let file = BufReader::new(file);
    let mut latest = None;
    let mut previous_timestamp = 0;
    let mut previous_frames = 0;
    let mut previous_steps = 0;
    for (line_number, line) in file.lines().enumerate() {
        let line = line.map_err(|error| format!("could not read capture status line: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let status: CaptureStatus = serde_json::from_str(&line).map_err(|error| {
            format!(
                "invalid capture status at line {}: {error}",
                line_number + 1
            )
        })?;
        if status.schema_version != CAPTURE_SESSION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported capture status schema: {}",
                status.schema_version
            ));
        }
        if status.session_id != session_id {
            return Err("capture status session_id does not match the start receipt".into());
        }
        if status.timestamp_ns < previous_timestamp
            || status.frames_written < previous_frames
            || status.steps_written < previous_steps
        {
            return Err("capture status counters or timestamps are not monotonic".into());
        }
        previous_timestamp = status.timestamp_ns;
        previous_frames = status.frames_written;
        previous_steps = status.steps_written;
        latest = Some(status);
    }
    Ok(latest)
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn reject_symlink(path: &Path) -> Result<()> {
    if path.is_symlink() {
        return Err(Error::Contract(format!(
            "capture path must not be a symlink: {}",
            path.display()
        )));
    }
    Ok(())
}

fn duration_ns(seconds: f64) -> u64 {
    Duration::from_secs_f64(seconds)
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

fn heartbeat_is_stale(status: &CaptureStatus, timeout_seconds: f64, now: u64) -> bool {
    now.saturating_sub(status.timestamp_ns) > duration_ns(timeout_seconds)
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.as_nanos().min(u64::MAX as u128) as u64
        })
}

pub fn relative_portable(root: &Path, path: &Path) -> Result<String> {
    Ok(path
        .strip_prefix(root)
        .map_err(|_| Error::Invalid("artifact must stay inside the run directory".into()))?
        .to_string_lossy()
        .replace('\\', "/"))
}

fn wait_for_child(child: &mut Child, timeout: Option<Duration>) -> Result<i32> {
    let deadline = timeout.map(|value| Instant::now() + value);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status.code().unwrap_or(1));
        }
        if deadline.is_some_and(|value| Instant::now() >= value) {
            child.kill()?;
            let _ = child.wait()?;
            return Err(Error::Contract(format!(
                "project command exceeded {:.1}s",
                timeout.unwrap_or_default().as_secs_f64()
            )));
        }
        thread::sleep(Duration::from_millis(20));
    }
}

pub fn executable_available(project: &Project, command: &ProjectCommand) -> bool {
    let Some(program) = command.argv.first() else {
        return false;
    };
    let candidate = Path::new(program);
    if candidate.is_absolute() {
        return candidate.is_file();
    }
    if candidate.components().count() > 1 {
        return project.root.join(candidate).is_file();
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|directory| {
            let plain = directory.join(program);
            if plain.is_file() {
                return true;
            }
            #[cfg(windows)]
            {
                ["exe", "cmd", "bat"]
                    .iter()
                    .any(|extension| plain.with_extension(extension).is_file())
            }
            #[cfg(not(windows))]
            {
                false
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn status(
        session_id: &str,
        state: CaptureState,
        frames: u64,
        steps: u64,
        timestamp_ns: u64,
    ) -> CaptureStatus {
        CaptureStatus {
            schema_version: CAPTURE_SESSION_SCHEMA_VERSION.into(),
            session_id: session_id.into(),
            state,
            frames_written: frames,
            steps_written: steps,
            last_frame_timestamp_ns: (frames > 0).then_some(timestamp_ns),
            dropped_frames: 0,
            recorder_state: "fake".into(),
            timestamp_ns,
            reason: None,
        }
    }

    fn write_statuses(path: &Path, statuses: &[CaptureStatus]) {
        let mut file = File::create(path).expect("status file");
        for status in statuses {
            serde_json::to_writer(&mut file, status).expect("status json");
            file.write_all(b"\n").expect("status newline");
        }
    }

    #[test]
    fn capture_session_parser_distinguishes_start_zero_stall_and_valid_states() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("capture-status.jsonl");
        let session_id = "capture-test";

        assert_eq!(read_capture_statuses(&path, session_id).unwrap(), None);

        write_statuses(
            &path,
            &[status(session_id, CaptureState::Completed, 0, 0, 10)],
        );
        let zero = read_capture_statuses(&path, session_id).unwrap().unwrap();
        assert_eq!(zero.state, CaptureState::Completed);
        assert_eq!(zero.frames_written, 0);

        let stalled = status(session_id, CaptureState::Healthy, 1, 1, 100);
        assert!(heartbeat_is_stale(&stalled, 0.1, 200_000_000));

        write_statuses(
            &path,
            &[
                status(session_id, CaptureState::Starting, 0, 0, 10),
                status(session_id, CaptureState::Healthy, 1, 1, 20),
                status(session_id, CaptureState::Completed, 1, 1, 30),
            ],
        );
        let valid = read_capture_statuses(&path, session_id).unwrap().unwrap();
        assert_eq!(valid.state, CaptureState::Completed);
        assert_eq!(valid.frames_written, 1);
        assert_eq!(valid.steps_written, 1);
    }

    #[test]
    fn capture_session_parser_rejects_malformed_and_partial_progress() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("capture-status.jsonl");
        let session_id = "capture-test";
        fs::write(&path, b"not-json\n").expect("malformed status");
        assert!(read_capture_statuses(&path, session_id).is_err());

        write_statuses(
            &path,
            &[
                status(session_id, CaptureState::Healthy, 2, 2, 20),
                status(session_id, CaptureState::Healthy, 1, 2, 30),
            ],
        );
        let error = read_capture_statuses(&path, session_id).unwrap_err();
        assert!(error.contains("not monotonic"));
    }
}
