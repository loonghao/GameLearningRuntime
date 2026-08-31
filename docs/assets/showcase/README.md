# Showcase asset provenance

README media must prove only what it actually shows. Do not publish a local
path, account name, hostname, PID/HWND, authentication material, chat overlay,
private dataset, proprietary telemetry, or an unauthorized runtime trace.

## `glr-counter-collector.gif`

- Source: a real local collection run of the bundled, explicitly synthetic
  `CounterEnvironment` through `ContractEnvironment` and `SyncCollector`.
- Renderer: [`scripts/render_readme_demo.py`](../../../scripts/render_readme_demo.py).
- Public claim: the core GLR contract validates and collects a terminal-bounded
  sequence. It is not evidence of a commercial-game adapter or trainer.
- Expected artifact: 960 x 540, seven frames, looping GIF.

Regenerate it from a synchronized development environment with ImageMagick on
`PATH`:

```powershell
uv sync --frozen --all-groups
uv run python scripts/render_readme_demo.py
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
