# Web and Three.js game learning

GLR drives a browser game through the same `ContractEnvironment` it uses for a
desktop title. The adapter reads an observation, applies an action, and reports
reward and episode boundaries; the browser is only a transport.

This guide shows both supported paths and states where each one stops.

See [ADR-0048](../decisions/0048-drive-browser-and-three-js-web-games-through-the-same-contract.md)
for the decision and the full capability boundary.

## Install

The runtime package has no browser dependency. The browser path is optional:

```bash
pip install '.[torch]' playwright
playwright install chromium
```

## Path 1: a page you can instrument

Use this when you wrote the page or are authorized to add a hook to it. The page
publishes a small object and the adapter does the rest.

```html
<script type="module">
  window.__glr = {
    ready: true,
    reset(seed) { /* start a deterministic episode from `seed` */ },
    step(action) { /* apply one of N discrete actions */ },
    state() {
      return { features: { x: 0.1, y: -0.2 }, alive: true, score: 12, steps: 12 };
    },
  };
</script>
```

Only two rules matter for learning:

* **Advance the world in `step()`, never in the render loop.** A game that
  mutates state on `requestAnimationFrame` races the learner and its results
  will not reproduce.
* **Honour `reset(seed)`.** A deterministic reset is what makes an evaluation
  comparable across runs.

### Serve it, do not open it

ES module imports and import maps are blocked from a `file://` origin, so a
Three.js page opened as a local file silently loses its renderer. Serve the
directory, even when every asset is local:

```python
from game_learning_runtime.web_game import (
    InstrumentedWebGameEnvironment,
    LocalPageServer,
    PageSpec,
    PlaywrightBrowserBridge,
    bundled_assets_dir,
    dodge_page_spec,
)

with LocalPageServer(bundled_assets_dir()) as server:
    page = PageSpec(
        url=server.url_for("orbital_dodge.html"),
        ready_expression="window.__glr && window.__glr.ready === true",
    )
    with PlaywrightBrowserBridge(page) as bridge:
        env = InstrumentedWebGameEnvironment(bridge, max_steps=256)
        timestep = env.reset(seed=7)
        timestep = env.step({"choice": np.array([2], dtype=np.int64)})
```

### Collect and train

The adapter is an ordinary `GameEnvironment`, so `SyncCollector` and
`ContractEnvironment` work unchanged:

```python
from game_learning_runtime.collector import SyncCollector
from game_learning_runtime.environment import ContractEnvironment

collector = SyncCollector(ContractEnvironment(env), actor_id="actor-0")
unroll = collector.collect(policy, steps=128)
```

## Path 2: an external page you cannot modify

Use `BlackBoxWebGameEnvironment` when the game is someone else's. Supply a
JavaScript expression that returns observable state and the keys that drive it:

```python
env = BlackBoxWebGameEnvironment(
    bridge,
    state_expression="({features: [shipX(), nearestDx(), nearestDy(), secondDx()],"
    " alive: !document.querySelector('.game-over'),"
    " score: Number(document.querySelector('.score').textContent),"
    " steps: window.frame})",
    action_keys=("ArrowLeft", "Space", "ArrowRight"),
    reset_key="KeyR",
)
```

Actions are real keyboard events, so the page cannot distinguish the adapter from
a human player. The trade-off is fidelity: **the learner can only see what your
expression reads.** If the score exists only as pixels, this adapter has nothing
to learn from and you need a screen encoder instead.

## What this does not do

| Situation | Outcome |
|---|---|
| Canvas-only game, score visible only as pixels | Not learnable via this adapter; needs a screen encoder |
| Anti-cheat, DRM, or licensing bypass | Out of scope; no facility provided |
| Game needs an account you do not hold | Out of scope |
| Difficulty depends on wall-clock reaction time | Artificially easy: the contract steps the page, it does not play in real time |
| Bundling a third-party game into this repo | Not done; point the adapter at a URL you supply |

## Reproduce the validation

```bash
python tools/web/validate_web_rl.py --output artifacts/web-rl-validation.json
```

The tool reports a random-policy baseline and a trained PPO policy measured on
the same environment, plus a training curve sampled every 2,400 steps over a
rolling window of 50 episodes. The comparison is the evidence; "it connects" is
not.

Measured on `web.dodge-instrumented-v1`, 120,000 environment steps, seed 7:

| Metric | Random policy | Trained PPO | Change |
|---|---|---|---|
| Mean steps survived | 65.8 | 103.9 | 1.58x |
| Mean episode return | 0.73 | 2.31 | 3.15x |

The full report is checked in as
[web-rl-validation.json](web-rl-validation.json).
