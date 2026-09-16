//! Registered run evidence only. Media preview never turns files into training evidence.
use crate::{
    error::{Error, Result},
    observation::{Observation, safe_child},
    project::validate_identifier,
    store::{ArtifactRecord, Store},
};
use serde_json::{Value, json};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "glr.run-media.v1";
const TEXT_LIMIT: u64 = 256 * 1024;

fn root(observation: &Observation, run: &str) -> Result<(Store, PathBuf)> {
    validate_identifier(run, "run_id")?;
    let db = safe_child(&observation.data_dir, Path::new("runs.sqlite3"))?;
    let store = Store::read_only(db)?;
    if store.get_run(run)?.environment_id != observation.environment_id {
        return Err(Error::Invalid("run belongs to another environment".into()));
    }
    Ok((
        store,
        safe_child(&observation.data_dir, &Path::new("runs").join(run))?,
    ))
}

fn kind(path: &str) -> (&'static str, &'static str) {
    match Path::new(path)
        .extension()
        .and_then(|p| p.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "mp4" => ("video", "video/mp4"),
        "webm" => ("video", "video/webm"),
        "png" => ("image", "image/png"),
        "jpg" | "jpeg" => ("image", "image/jpeg"),
        "webp" => ("image", "image/webp"),
        "gif" => ("image", "image/gif"),
        "md" | "markdown" => ("markdown", "text/plain; charset=utf-8"),
        "txt" | "log" | "json" | "jsonl" => ("text", "text/plain; charset=utf-8"),
        // HTML, SVG and other active formats are download-only.
        _ => ("download", "application/octet-stream"),
    }
}

pub fn catalog(observation: &Observation, run: &str, after: &str) -> Result<Value> {
    let (store, root) = root(observation, run)?;
    let mut records = store.media_artifacts(run, None, after)?;
    let more = records.len() > 100;
    records.truncate(100);
    let next = if more {
        records.last().map(|r| r.path.clone())
    } else {
        None
    };
    let items: Vec<_> = records.into_iter().map(|a| {
        let (kind, mime) = kind(&a.path);
        let available = safe_child(&root, Path::new(&a.path)).ok().and_then(|p| p.metadata().ok()).is_some_and(|m| m.is_file() && m.len() == a.size_bytes);
        json!({"path":a.path,"role":a.role,"kind":kind,"mime":mime,"size_bytes":a.size_bytes,"recorded_sha256":a.sha256,"available":available,"integrity":"not_reverified"})
    }).collect();
    Ok(json!({"schema_version":SCHEMA,"run_id":run,"items":items,"next_after":next}))
}

pub struct MediaFile {
    pub file: File,
    pub size: u64,
    pub mime: &'static str,
    pub download: bool,
}

fn registered(observation: &Observation, run: &str, path: &str) -> Result<(ArtifactRecord, File)> {
    let (store, root) = root(observation, run)?;
    let record = store
        .media_artifacts(run, Some(path), "")?
        .into_iter()
        .next()
        .ok_or_else(|| Error::Invalid("file is not registered to this run".into()))?;
    let file = File::open(safe_child(&root, Path::new(path))?)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != record.size_bytes {
        return Err(Error::Contract(
            "artifact is missing or its size changed; refresh its registration".into(),
        ));
    }
    Ok((record, file))
}

pub fn open(observation: &Observation, run: &str, path: &str) -> Result<MediaFile> {
    let (record, file) = registered(observation, run, path)?;
    let (kind, mime) = kind(path);
    Ok(MediaFile {
        file,
        size: record.size_bytes,
        mime,
        download: kind == "download",
    })
}

pub fn document(observation: &Observation, run: &str, path: &str) -> Result<Value> {
    let (record, file) = registered(observation, run, path)?;
    if !matches!(kind(path).0, "text" | "markdown") {
        return Err(Error::Invalid(
            "artifact does not support text preview".into(),
        ));
    }
    let mut bytes = Vec::new();
    file.take(TEXT_LIMIT).read_to_end(&mut bytes)?;
    Ok(
        json!({"schema_version":SCHEMA,"path":path,"text":String::from_utf8_lossy(&bytes),"truncated":record.size_bytes>TEXT_LIMIT}),
    )
}

// Read a bounded, registered capture manifest selected explicitly by the viewer.
pub fn frames(observation: &Observation, run: &str, manifest: &str, video: &str) -> Result<Value> {
    let (record, file) = registered(observation, run, manifest)?;
    if record.role != "capture-manifest" || record.size_bytes > 8 * 1024 * 1024 {
        return Err(Error::Invalid(
            "select a registered capture manifest of at most 8 MiB".into(),
        ));
    }
    let mut bytes = Vec::new();
    file.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    use sha2::{Digest, Sha256};
    if format!("{:x}", Sha256::digest(&bytes)) != record.sha256 {
        return Err(Error::Contract("capture manifest checksum mismatch".into()));
    }
    let manifest_data: crate::contracts::CaptureManifest = serde_json::from_slice(&bytes)?;
    let (_, run_root) = root(observation, run)?;
    let manifest_parent = safe_child(&run_root, Path::new(manifest))?
        .parent()
        .unwrap()
        .to_path_buf();
    let bound_video = safe_child(&manifest_parent, Path::new(&manifest_data.video.path))?;
    let (video_record, _) = registered(observation, run, video)?;
    if manifest_data.schema_version != crate::contracts::CAPTURE_MANIFEST_SCHEMA_VERSION
        || bound_video != safe_child(&run_root, Path::new(video))?
        || manifest_data.run_id != run
        || manifest_data.environment_id != observation.environment_id
        || manifest_data.video.sha256 != video_record.sha256
        || manifest_data.video.size_bytes != video_record.size_bytes
    {
        return Err(Error::Contract(
            "capture manifest does not bind this run and video".into(),
        ));
    }
    let mut previous = None;
    for frame in &manifest_data.frames {
        if frame.schema_version != "glr.capture-frame.v1"
            || frame.run_id != run
            || previous.is_some_and(|(index, pts)| frame.frame_index <= index || frame.pts_ns < pts)
        {
            return Err(Error::Contract(
                "invalid capture frame binding or non-monotonic mapping".into(),
            ));
        }
        previous = Some((frame.frame_index, frame.pts_ns));
    }
    let frames: Vec<_> = manifest_data.frames.iter().take(25000).map(|f|json!({"episode_id":f.episode_id,"step_id":f.step_id,"frame_index":f.frame_index,"seconds":f.pts_ns as f64 / 1e9})).collect();
    Ok(
        json!({"schema_version":SCHEMA,"frames":frames,"truncated":manifest_data.frames.len()>25000,"binding":"registered_capture_manifest","video_integrity":"not_reverified"}),
    )
}

/// One HTTP byte range, including open-ended and suffix requests.
pub fn byte_range(value: &str, size: u64) -> Option<(u64, u64)> {
    let (start, end) = value.strip_prefix("bytes=")?.split_once('-')?;
    if size == 0 {
        return None;
    }
    if start.is_empty() {
        let length: u64 = end.parse().ok()?;
        return (length > 0).then_some((size.saturating_sub(length), size - 1));
    }
    let start: u64 = start.parse().ok()?;
    let end = if end.is_empty() {
        size - 1
    } else {
        end.parse::<u64>().ok()?.min(size - 1)
    };
    (start <= end && start < size).then_some((start, end))
}
