# Recording and training data

Use one project-local storage contract for every game. This prevents recordings, logs,
datasets, and reports from drifting into game-specific folders.

## Inspect the default preset

```powershell
glr --project . --json capture preset
glr --project . --json capture layout
```

The default `training-balanced` preset is H.264/libx264, CRF 18, `fast`, 1920x1080 at
30 FPS CFR, yuv420p, GOP 30, fixed keyframes, fast-start metadata, and no audio.

The command returns output arguments only. The reviewed recorder sidecar supplies the
platform-specific exact-window input, consumes `GLR_CAPTURE_VIDEO`,
`GLR_CAPTURE_INDEX`, and `GLR_CAPTURE_STATUS`, and publishes lifecycle heartbeats.

## Canonical project layout

```text
.glr/
  runs.sqlite3
  checkpoints/
  runs/<run-id>/
    trainer.log
    capture.log
    trainer-result.json
    capture.mp4
    capture-index.jsonl
    capture-status.jsonl
    capture-session.json
    capture.manifest.json
    artifacts/
    report/index.html
```

Keep `.glr/` out of Git. Do not add sibling `recordings/`, `logs`, `training-data/`,
or `reports/` roots. Migrate old exports only as an explicit, reviewable operation.

An MP4 without `capture-index.jsonl` is review media only. Every index record must use
`glr.capture-frame.v1` and bind the run, episode, step, frame, video PTS, and observation
timestamp. Build a completed run report with:

```powershell
glr --project . --json report build <run-id>
```

The report is a projection of registered evidence. It does not convert an unindexed
video into training data or prove a live-game outcome.
