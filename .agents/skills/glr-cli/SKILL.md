---
name: glr-cli
description: Configure and operate the GameLearningRuntime agent-first CLI for bridge startup, bounded goal-driven research and training, concurrent review capture, run queries, spatial knowledge transfer, and verified model playback. Use for operating an existing GLR project; use glr-adapter-builder when implementing the game adapter itself.
---

# GLR CLI

Operate GLR as an agent control plane while preserving the adapter/learner boundary.

Read [references/commands.md](references/commands.md) before creating a project config,
running a goal, transferring knowledge, or claiming reproduction.

## Select the correct boundary

- Use this Skill when the project already has a reviewed runtime bridge and needs CLI setup or
  operation.
- Use `glr-adapter-builder` when implementing or changing observation, action, lifecycle,
  transport, target binding, or post-action verification.
- Never make the game adapter import a learner algorithm. The configured trainer, planner,
  researcher, evaluator, recorder, and player remain explicit project-owned processes.

## Operate agent-first

1. Resolve the nearest `glr-project.json`; do not guess a bridge path or game target.
2. Inspect the strict project roles and exact `environment_id`, `environment_family`, and
   `protocol_version` before execution.
3. Use `glr runtime start` only for the configured fixed-argv runtime command. Its process exit
   proves command completion, not a live bridge handshake or gameplay success.
4. Express the user objective as `glr.agent-goal.v1` with machine-readable success criteria and
   hard trial, step, time, and research-source budgets.
5. Run `glr goal run`. Let the project researcher gather only allowed sources; let the planner
   emit declarative reward terms; require the trainer/runtime to persist metrics; accept success
   only when evaluator evidence matches those persisted authoritative metrics.
6. Inspect `glr runs show` and query entities, routes, or research before deciding the next action.
   Route and guide results are hints; re-observe and verify postconditions in the live runtime.
7. Use a verified model bundle for playback. A valid hash proves artifact integrity and config
   identity, not policy quality, hardware determinism, or successful live gameplay.

## Preserve knowledge scope

- Environment-scoped positions and routes transfer only across the exact environment and protocol.
  Imports are downgraded to advisory until the new runtime observes them again.
- Family-scoped tutorial/guide findings may inform a similar game, but never transfer coordinates,
  action authority, or model compatibility.
- Keep public source provenance, access time, compact paraphrases, confidence, volatility, and
  runtime-verification status. Never store credentials or full copied guides.
- Exclude rejected findings. Treat unverified findings as hypotheses, never authoritative reward
  or success evidence.

## Recording and training data

When capture is configured, keep it enabled for `glr train` and `glr goal run` unless the user
explicitly opts out. The recorder is a concurrent project-owned sidecar and must emit both a small
H.264 MP4 and `glr.capture-frame.v1` step/frame index. A video without a valid checksummed index is
review media, not supervised-learning data.

Do not claim live-game acceptance from synthetic tests, process exit, video presence, run status,
or model hashes. Report the exact remaining runtime acceptance boundary.
