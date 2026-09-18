# Window recording

`glr train` can record the main window of the trainer process it spawns. Recording uses
Windows Graphics Capture (WGC), encodes H.264 into MP4, and writes a per-frame index next
to every segment so downstream tools can align video time with training time. Recording is
best effort: when it cannot run, GLR logs a warning and training continues unchanged.

Recording is Windows only. On other platforms every session reports
`recording requires Windows; the session is skipped on this platform` and nothing is
written.

## Configuration

Add a `[recording]` table to `glr-project.toml` (or the JSON manifest). Every key is
optional; unknown keys are rejected at load time, and out-of-range values are clamped with
a warning instead of failing the run.

```toml
[recording]
enabled = true
output_dir = "recordings"
fps = 30
max_width = 1920
bitrate_mbps = 6
segment_seconds = 600
include_cursor = false
auto_restore_minimized = true
audio = false
```

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Record the trainer window during `glr train`. |
| `output_dir` | `recordings` | Directory relative to the run directory. `glr train` skips recording when this is an absolute path (segments outside the run directory cannot become run evidence); `recording run --output` accepts absolute paths. |
| `fps` | `30` | Target frame rate; clamped to `1..=60`. |
| `max_width` | `1920` | Long-edge cap for the encoded video; clamped to `2..=7680`. |
| `bitrate_mbps` | `6` | H.264 target bitrate; clamped to `1..=200`. |
| `segment_seconds` | `600` | Rotate to a new file after this many seconds; `0` disables rotation. |
| `include_cursor` | `false` | Draw the system cursor into the capture. |
| `auto_restore_minimized` | `true` | Restore a minimized target window without activating it. |
| `audio` | `false` | Reserved. Audio capture is not implemented; a non-default value logs a warning and is treated as `false`. |

## Outputs

Each segment produces two files below the resolved output directory:

```text
.glr/runs/<run-id>/recordings/
  20260918T024500Z_8124_trainer_exe_seg0.mp4
  20260918T024500Z_8124_trainer_exe_seg0.frames.jsonl
```

- Name: `<UTC timestamp>_<pid>_<process name>_seg<segment>.mp4`, for example
  `20260918T024500Z_8124_trainer_exe_seg0.mp4`. The timestamp is UTC
  (`YYYYMMDDTHHMMSSZ`), the process name is sanitized for file names, and the segment
  counter starts at `0`.
- Resolution is decided once, from the first frame, and never changes mid-recording.
  Frames larger than `max_width` are downscaled into a fixed canvas; smaller or odd-sized
  frames are letterboxed. This keeps one MP4 playable end to end.
- `frames.jsonl` holds one JSON object per frame:

```json
{"frame": 0, "timestamp_qpc": 32369756126, "timestamp_utc_ms": 1789701530019}
```

  `frame` is a monotonic counter that continues across segments, `timestamp_qpc` is the
  capture timestamp in 100 ns ticks, and `timestamp_utc_ms` is the wall clock in
  milliseconds.

`glr train` registers every segment as run evidence (`capture_video` for the MP4,
`capture_index` for the JSONL) and returns a `recording` object in its output envelope:

```json
{
  "schema_version": "glr.recording.v1",
  "frame_index_schema_version": "glr.recording-frame.v1",
  "pid": 8124,
  "label": "trainer",
  "frames": 2910,
  "width": 1280,
  "height": 840,
  "skipped": null,
  "warnings": [],
  "segments": [{"video": "...", "index": "...", "frames": 586}]
}
```

## Commands

Recording during training is on by default; skip it for one run with `--no-recording`:

```powershell
glr --project . train
glr --project . train --no-recording
```

Record any process for a fixed duration — useful to check that a PID resolves to the
window you expect before wiring a trainer:

```powershell
glr --project . --json recording run --pid 8124 --seconds 10
glr --project . --json recording run --pid 8124 --seconds 10 --output .glr/probe
```

`recording run` records even when `[recording] enabled = false`, resolves a relative
`--output` against the project root, and returns the same envelope.

## Behavior and limits

- **Window choice.** The session enumerates top-level windows, keeps those owned by the
  target PID that are visible, not cloaked, and not owned by another window, and picks the
  largest by area. It waits up to three seconds for that window to appear.
- **Minimized windows.** A minimized window delivers no content. With
  `auto_restore_minimized` the window is shown with `SW_SHOWNOACTIVATE` and pushed to the
  bottom of the Z order, so recording works without stealing focus. Disable the option to
  skip instead.
- **Occluded windows.** WGC delivers the window's own content, so a covered or
  off-screen window still records correctly.
- **Remote desktop.** Capture is unavailable in remote desktop and server sessions; the
  session is skipped with a warning.
- **Windows that opt out.** A window marked with `WDA_EXCLUDEFROMCAPTURE` is never
  captured. GLR does not bypass capture protection; Windows also draws its capture border
  while recording.
- **Throughput.** WGC delivers frames when the window updates, so a static window yields
  few frames; the recorder adds roughly 5 ms per frame on the GPU path and about 30 ms per
  1080p frame on the CPU downscale path. Hardware encoding and display composition set the
  real ceiling.
- **Audio.** Not implemented. The MP4 container may carry an empty audio track; the
  recording itself is silent.
- **Failures.** `GraphicsCaptureSession::IsSupported() == false`, a missing window, or an
  encoder failure only produce a warning. `glr train` still finishes and reports the run
  normally.

## Skip reasons

| `skipped` message | Cause |
| --- | --- |
| `recording is disabled by configuration` | `[recording] enabled = false`. |
| `recording requires Windows; ...` | Session started on a non-Windows platform. |
| `Windows Graphics Capture is not supported on this device` | `IsSupported()` returned false. |
| `Windows Graphics Capture is unavailable in a remote desktop session` | Remote desktop or server session. |
| `the target process has no capturable main window` | No visible, uncloaked window within three seconds. |
| `the target window is minimized and auto_restore_minimized is disabled` | Minimized window and restore disabled. |
| `the target window stayed minimized after an automatic restore attempt` | Minimized window was restored without activation but still no capturable window appeared within three seconds. |
| `recording unavailable: <detail>` | Output directory or encoder could not be opened. |
