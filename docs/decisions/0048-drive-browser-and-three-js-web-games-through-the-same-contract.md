# ADR-0048: Drive browser and Three.js web games through the same contract

## Status

Accepted.

## Context

GLR already drove desktop titles through `ContractEnvironment`. The open question
was whether the same contract could drive a game that lives in a browser tab:
an arbitrary Three.js or external web game, learned end to end so the AI
actually plays better.

The browser is a different host, but not a different problem. The runtime
contract is about observation, action, mask, reward, and episode boundary --
none of which care whether the process behind them is a game engine or a
Chromium renderer. What differs is the transport: a desktop adapter talks to a
process, while a web adapter evaluates script in a page and synthesizes input.

Two constraints shaped the design:

1. **The runtime must stay importable without a browser.** The core package
   depends only on NumPy. A hard Playwright import would make every GLR
   installation depend on a browser driver.
2. **Capability claims need limits.** "GLR can drive any web game" is false in
   an important way: a web game whose score is only visible as pixels cannot be
   learned from a keyboard-driven adapter. Claiming otherwise would send
   downstream users into a week of dead-end work.

## Decision

### 1. The browser is an injectable transport, not an adapter concern

`BrowserBridge` is a `Protocol` with three operations: `evaluate`, `press`, and
`close`. Adapters depend on it; nothing in the adapter knows Playwright exists.

* `PlaywrightBrowserBridge` drives a real Chromium page. Playwright is imported
  **lazily, in the constructor**, so the package imports cleanly without it and
  fails with an actionable message when it is genuinely needed.
* `ScriptedBrowserBridge` replays a scripted in-page model and records key
  presses. It is the deterministic double the test suite uses, so the whole
  adapter contract is testable with no browser installed and no network.

This keeps a browser dependency at the edge and makes the adapter unit-testable,
which is what let the 62 tests in `tests/test_web_game_*.py` cover the contract
without a Chromium download.

### 2. Two adapters, because there are two genuinely different situations

| | `InstrumentedWebGameEnvironment` | `BlackBoxWebGameEnvironment` |
|---|---|---|
| Page | exposes a `window.__glr` hook | unmodified |
| Observation | structured state from the page | a caller-supplied JS expression |
| Action | structured `step(index)` call | real keyboard events |
| Needs | you own or may instrument the page | a readable score and a key binding |

The instrumented path is the one that learns well, because the observation is
structured state instead of pixels. The black-box path exists because most
external games will never be instrumented, and it is honest about what it can
see: whatever the supplied expression cannot read is invisible to the learner.

### 3. The bundled demonstration page is ours, and the dependency is vendored

`src/game_learning_runtime/web_game/assets/orbital_dodge.html` is authored for
this repository under the repository's own MIT license. Three.js r180 (MIT) is
vendored beside it rather than loaded from a CDN, so a validation run is
reproducible offline and the license provenance is a file in the tree instead of
a URL that can rot.

The page's simulation advances **only** when `step()` is called. The render loop
draws whatever the world currently is and never mutates it. This single property
is what makes a headless run and a visible run produce identical trajectories;
without it, a learner would be racing a `requestAnimationFrame` clock and no
result would reproduce.

### 4. Terminated and truncated are disjoint, and the budget counts adapter steps

A crash is the page's verdict (`terminated`). Running out of step budget is the
harness cutting an episode short (`truncated`). Exhausting the budget is never
also a termination, even when the player was about to crash: the crash was never
observed, so reporting it would invent a fact the page did not report.

The budget counts adapter steps, not page ticks. An adapter configured with
`frames_per_step=2` advances the page twice per action, and comparing the page's
faster tick counter against an adapter-step budget would silently cut every
episode in half.

### 5. The learner stays out of the runtime package

Per ADR-0005, the runtime ships objectives, not learners. The PPO update loop
lives in `tools/web/validate_web_rl.py` and reuses
`integrations.torch_objectives.ppo_loss` and `generalized_advantage_estimate`
for the reusable parts. The adapters ship in the runtime package; the trainer
does not.

## Capability boundary

This is the part downstream users need most, so it is stated as limits rather
than as features.

**Supported**

* Any self-owned or authorized-to-instrument web game, via `window.__glr`. Full
  contract: observation, action, mask, reward, episode boundary, deterministic
  reset by seed.
* Any unmodified page whose observable state can be read by a JavaScript
  expression and whose controls are keyboard-driven.
* Headless or headed Chromium, WebGL included (verified: WebGL 2.0 in headless
  Chromium).
* Serving from a loopback HTTP server. Bundled pages **must** be served over
  HTTP, not opened as `file://`: ES module imports and import maps are blocked
  from a `file://` origin, so a Three.js page opened that way silently loses its
  renderer.

**Not supported**

* **Canvas-only games with no readable state.** If the score exists only as
  pixels, the black-box adapter has nothing to learn from. This needs a screen
  encoder on the capture path, which this ADR does not add.
* **Anti-cheat, DRM, licensing, or access-control bypass.** Out of scope by the
  README boundary, and no facility for it is provided. An adapter automates a
  page the caller is authorized to automate.
* **Games requiring an account the caller does not hold**, or multiplayer titles
  where driving one client affects other players.
* **Timing-critical action games** whose difficulty depends on wall-clock
  reaction time. The contract steps the page, it does not play it in real time;
  a game that punishes slow wall-clock input will look artificially easy.
* **Third-party pages bundled into this repository.** Black-box adapters point at
  a URL the caller supplies; we do not vendor other people's games, since that
  would make us responsible for their license.

## Consequences

**Positive**

* The web path is validated by evidence, not by assertion: a before/after
  comparison from a real training run (see below).
* The adapter contract is covered by 62 tests that need no browser.
* Browser automation is one optional import away, and its absence produces an
  actionable error rather than an `ImportError` at package import.

**Negative**

* Playwright and Chromium are required to run the validation tool. They are not
  installed by the default dependency set, and the browser must be fetched with
  `playwright install chromium`.
* The black-box adapter's fidelity is bounded by the caller's state expression.
  A weak expression yields an unlearnable observation, and nothing in the
  runtime can detect that.

**Neutral**

* `tools/web/validate_web_rl.py` is registered in `tools/registry.toml` under
  the `web` domain, so `just layout-check` keeps covering it.

## Validation evidence

Produced by:

```bash
python tools/web/validate_web_rl.py --train-steps 120000 \
    --unroll-length 128 --evaluation-episodes 30 --max-steps 256 \
    --seed 7 --output artifacts/web-rl-validation.json
```

Full report: [web-rl-validation.json](../examples/web-rl-validation.json).
The tool measures a random policy and the trained PPO policy on the same
environment and reports both, because "it connects" and "it steps" are not
evidence that the framework can make an agent play well.

Environment `web.dodge-instrumented-v1`, 120,000 environment steps, 7,500 policy
updates, seed 7. 30 evaluation episodes per policy:

| Metric | Random policy | Trained PPO | Change |
|---|---|---|---|
| Mean steps survived | 65.8 | 103.9 | **1.58x** |
| Mean episode return | 0.73 | 2.31 | **3.15x** |
| Best episode return | 4.70 | 9.42 | 2.00x |

Mean steps and mean score are the same number because the bundled page scores
one point per survived step.

The training curve confirms the improvement is learning rather than a lucky
evaluation sample. It is a rolling window of 50 completed episodes sampled every
2,400 environment steps; averaging the first five and last five checkpoints:

```
first 5 checkpoints (~12k steps):  70.1 mean steps
last  5 checkpoints (~120k steps): 122.4 mean steps
```

The window matters more than the checkpoint count. Two probe episodes would have
been a coin flip: the page seeds each episode independently, so single-episode
samples swing between 30 and 209 steps and say nothing about the policy.

The run is deliberately not tuned to convergence. It answers "can GLR drive a
browser game and make an agent measurably better at it", not "what is the best
score attainable on this toy task".

## Reproduction

```bash
pip install '.[torch]' playwright
playwright install chromium
python tools/web/validate_web_rl.py --output artifacts/web-rl-validation.json
```
