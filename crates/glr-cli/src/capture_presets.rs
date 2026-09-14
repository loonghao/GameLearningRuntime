use serde::Serialize;

use crate::error::{Error, Result};
use crate::project::Project;

pub const DEFAULT_PRESET: &str = "training-balanced";

#[derive(Debug, Serialize)]
pub struct CapturePreset {
    pub name: &'static str,
    pub purpose: &'static str,
    pub codec: &'static str,
    pub encoder: &'static str,
    pub encoder_preset: &'static str,
    pub crf: u8,
    pub pixel_format: &'static str,
    pub frame_rate: u32,
    pub width: u32,
    pub height: u32,
    pub gop_frames: u32,
    pub keyint_min: u32,
    pub scene_cut: bool,
    pub constant_frame_rate: bool,
    pub audio: bool,
    pub ffmpeg_output_argv: [&'static str; 22],
}

const TRAINING: CapturePreset = CapturePreset {
    name: DEFAULT_PRESET,
    purpose: "default random-seekable visual training and human review",
    codec: "h264",
    encoder: "libx264",
    encoder_preset: "fast",
    crf: 18,
    pixel_format: "yuv420p",
    frame_rate: 30,
    width: 1920,
    height: 1080,
    gop_frames: 30,
    keyint_min: 30,
    scene_cut: false,
    constant_frame_rate: true,
    audio: false,
    ffmpeg_output_argv: [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-g",
        "30",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        "-fps_mode",
        "cfr",
        "-movflags",
        "+faststart",
        "-an",
        "{capture_video}",
    ],
};

const REVIEW: CapturePreset = CapturePreset {
    name: "review-compact",
    purpose: "smaller H.264 review media with broad decoder compatibility",
    codec: "h264",
    encoder: "libx264",
    encoder_preset: "medium",
    crf: 20,
    pixel_format: "yuv420p",
    frame_rate: 30,
    width: 1920,
    height: 1080,
    gop_frames: 60,
    keyint_min: 30,
    scene_cut: false,
    constant_frame_rate: true,
    audio: false,
    ffmpeg_output_argv: [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-g",
        "60",
        "-keyint_min",
        "30",
        "-sc_threshold",
        "0",
        "-fps_mode",
        "cfr",
        "-movflags",
        "+faststart",
        "-an",
        "{capture_video}",
    ],
};

pub fn list() -> [&'static CapturePreset; 2] {
    [&TRAINING, &REVIEW]
}

pub fn find(name: &str) -> Result<&'static CapturePreset> {
    list()
        .into_iter()
        .find(|preset| preset.name == name)
        .ok_or_else(|| {
            Error::Invalid(format!(
                "unknown capture preset {name:?}; expected training-balanced or review-compact"
            ))
        })
}

#[derive(Debug, Serialize)]
pub struct ProjectLayout {
    pub schema_version: &'static str,
    pub project_root: String,
    pub data_root: String,
    pub run_store: String,
    pub runs: String,
    pub run_directory: String,
    pub logs: [&'static str; 3],
    pub capture: [&'static str; 5],
    pub training_data: [&'static str; 2],
    pub reports: [&'static str; 2],
    pub checkpoints: String,
    pub checkpoint_best: String,
    pub checkpoint_promotion: String,
    pub learning_checkpoints: String,
    pub latest_learning_checkpoint: String,
}

pub fn layout(project: &Project) -> ProjectLayout {
    let portable = |path: &std::path::Path| path.to_string_lossy().replace('\\', "/");
    let data_root = portable(&project.data_dir);
    ProjectLayout {
        schema_version: "glr.storage-layout.v1",
        project_root: portable(&project.root),
        run_store: format!("{data_root}/runs.sqlite3"),
        runs: format!("{data_root}/runs"),
        run_directory: format!("{data_root}/runs/<run-id>"),
        data_root,
        logs: ["<run>/trainer.log", "<run>/capture.log", "<run>/<role>.log"],
        capture: [
            "<run>/capture.mp4",
            "<run>/capture-index.jsonl",
            "<run>/capture-status.jsonl",
            "<run>/capture-session.json",
            "<run>/capture.manifest.json",
        ],
        training_data: ["<run>/artifacts/<dataset>", "<run>/trainer-result.json"],
        reports: ["<run>/report/index.html", "<run>/review/report/index.html"],
        checkpoints: format!("{}/checkpoints", portable(&project.data_dir)),
        checkpoint_best: format!(
            "{}/checkpoints/<environment-id>/<goal-id>/best.checkpoint",
            portable(&project.data_dir)
        ),
        checkpoint_promotion: format!(
            "{}/checkpoints/<environment-id>/<goal-id>/best.json",
            portable(&project.data_dir)
        ),
        learning_checkpoints: format!(
            "{}/checkpoints/<environment-id>/<goal-id>/runs/<run-id>/<trial-id>/<stage>.json",
            portable(&project.data_dir)
        ),
        latest_learning_checkpoint: format!(
            "{}/checkpoints/<environment-id>/<goal-id>/latest.json",
            portable(&project.data_dir)
        ),
    }
}
