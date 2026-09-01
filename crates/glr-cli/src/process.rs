use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::contracts::{CaptureManifestInput, build_capture_manifest};
use crate::error::{Error, Result};
use crate::project::{Project, ProjectCommand};
use crate::store::Store;

pub struct CaptureSession {
    child: Child,
    log_path: PathBuf,
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
        .env("GLR_CAPTURE_INDEX", &context["capture_index"]);
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
    let log_path = run_dir.join("capture.log");
    let log = File::create(&log_path)?;
    let stderr = log.try_clone()?;
    let command = capture.command();
    let mut process = configure_command(&command, project, run_id, run_dir, None, &HashMap::new())?;
    let child = process
        .stdin(if capture.stop == "stdin-q" {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr))
        .spawn()?;
    Ok(CaptureSession { child, log_path })
}

pub fn finish_capture(
    project: &Project,
    store: &Store,
    run_id: &str,
    capture_dir: &Path,
    artifact_root: &Path,
    mut session: CaptureSession,
) -> Result<bool> {
    let capture = project
        .capture
        .as_ref()
        .ok_or_else(|| Error::Contract("capture configuration is missing".into()))?;
    if session.child.try_wait()?.is_none() {
        if capture.stop == "stdin-q" {
            if let Some(stdin) = session.child.stdin.as_mut() {
                let _ = stdin.write_all(b"q\n");
                let _ = stdin.flush();
            }
        } else {
            session.child.kill()?;
        }
    }
    let capture_exit = wait_for_child(&mut session.child, Some(Duration::from_secs(10)))?;
    if session.log_path.is_file() {
        store.register_artifact(
            run_id,
            &relative_portable(artifact_root, &session.log_path)?,
            &session.log_path,
            "capture-log",
            "text/plain",
        )?;
    }
    let video = capture_dir.join(&capture.video_file);
    let index = capture_dir.join(&capture.index_file);
    if capture_exit != 0 || !video.is_file() || !index.is_file() {
        return Ok(false);
    }
    let manifest = capture_dir.join("capture.manifest.json");
    build_capture_manifest(CaptureManifestInput {
        manifest_path: &manifest,
        environment_id: &project.environment_id,
        run_id,
        video_path: &video,
        index_path: &index,
        codec: &capture.codec,
        frame_rate: capture.frame_rate,
        width: capture.width,
        height: capture.height,
    })?;
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
    Ok(true)
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
