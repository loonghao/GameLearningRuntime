# Showcase asset provenance

README media must prove only what it actually shows. Do not publish a local
path, account name, hostname, PID/HWND, authentication material, chat overlay,
private dataset, proprietary telemetry, or an unauthorized runtime trace.

## `glr-counter-collector.gif`

- Source: a real local collection run of the bundled, explicitly synthetic
  `CounterEnvironment` through `ContractEnvironment` and `SyncCollector`.
- Renderer: [`tools/docs/render_readme_demo.py`](../../../tools/docs/render_readme_demo.py).
- Public claim: the core GLR contract validates and collects a terminal-bounded
  sequence. It is not evidence of a commercial-game adapter or trainer.
- Expected artifact: 960 x 540, seven frames, looping GIF.

Regenerate it from a synchronized development environment with ImageMagick on
`PATH`:

```powershell
uv sync --frozen --all-groups
uv run python tools/docs/render_readme_demo.py
magick identify docs/assets/showcase/glr-counter-collector.gif
```

Inspect the first, middle, and final frames after regeneration. Confirm that
the values follow the real collected transitions and that the final frame is
terminal.

## Adding an authorized live-adapter clip

Before adding footage from a local project:

1. establish that the runtime and public recording are authorized;
2. capture only the exact application region needed to support the claim;
3. remove or crop accounts, paths, machine/process identity, notifications,
   private telemetry, and unrelated applications;
4. keep the clip short and label the adapter, transport, lifecycle (`reset` or
   `attach`), and validation boundary accurately;
5. preserve the source/transform commands and inspect representative frames;
6. verify the committed GIF properties and README reference on the final Git
   commit.

Synthetic conformance is not live acceptance. Gameplay footage alone is not
proof that observations, actions, rewards, or episode fencing passed the GLR
contract.

## Curated game-workflow assets

- `glr-agent-learning-hero.png` is an ImageGen-created, non-game-specific hero
  illustration; it is decorative and carries no runtime claim.
- `wukong-training-clean.png` and `wukong-training.mp4` are cropped, muted
  derivatives of the supplied Black Myth: Wukong recording. The crop removes
  the training overlay and unrelated UI; the clip is six seconds.
- `wukong-training-full.mp4` is the complete seven-minute recording, muted,
  cropped, resized to 640px wide, and encoded at roughly 450 kbps for GitHub
  playback. It is still workflow evidence, not a success claim.
- `vampire-survivors-training.png` and `vampire-survivors-training.mp4` are
  six-second, muted derivatives of an authorized local Vampire Survivors run.
- `training-contact-sheet-12x12.png` is a 12×12 frame contact sheet derived from
  that same run for compact visual review.

These assets support the narrow claim that GLR is used in interface-first game
learning workflows. They do not establish convergence, a terminal win, or a
particular token/cost saving for every workload; measure those claims from the
run records and billing telemetry for the target setup.
