---
name: glr-cli
description: Configure and operate the GameLearningRuntime agent-first CLI for bridge startup, bounded goal-driven research and training, concurrent review capture, run queries, spatial knowledge transfer, and verified model playback. Use for operating an existing GLR project; use glr-adapter-builder when implementing the game adapter itself.
---

# GLR CLI

Operate GLR through the standalone Rust control plane while preserving the
adapter/learner boundary. The `glr` executable is the canonical deployment and
Agent entrypoint; Python is an optional SDK for project roles, not a CLI runtime
dependency.

## Resolve bundled files portably

This Skill is distributed from both GLR releases and Agent Plugin packages.
Resolve its `references/` directory relative to the directory containing this
`SKILL.md`; do not assume a repository checkout or a user-profile install
path. The `--skills-dir` option below is a project-owned destination for an
explicit update and is separate from the host's installed plugin directory.

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

1. Run `glr --version`, resolve the nearest `glr-project.json`, and run
   `glr --project . --json doctor`; do not guess a bridge path or game target.
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

## Keep the managed runtime current

- `glr update --check` is a read-only release check and is safe to use when
  diagnosing version drift.
- Run `glr update --yes` only when the user explicitly asks to update GLR. It
  verifies the exact platform archive and `SHA256SUMS`, then updates the `glr`
  executable, its sibling `glr-hostd`, and the repository-owned `glr-cli` and
  `glr-adapter-builder` Skills.
- Use `--skills-dir` only for an explicitly selected project Skills directory.
  Use `--no-skills` when the user requested binary-only maintenance.
- The updater does not modify game code, project role dependencies, Python
  environments, models, datasets, `glr-project.json`, or trainer configuration.
- SHA-256 protects same-release artifact integrity; it is not publisher
  signature verification. Report the first unified-release smoke boundary when
  no matching target archive exists yet.
- Public checks use GitHub's latest-release asset link rather than the REST API,
  so they do not consume anonymous API quota. The updater derives the version,
  exact target archive, and digest from the published `SHA256SUMS`.

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

## Build an offline run report

Generate a self-contained, interactive review page from one completed run:

```powershell
glr --project . --json report build <run-id>
glr --project . --json report build <run-id> --output review/report
```

The default output is `.glr/runs/<run-id>/report/index.html`; a custom output
must remain inside that run directory. Before writing, GLR verifies every
registered evidence artifact's portable path, byte size, and SHA-256 digest,
omits prior `run-report` outputs to avoid self-referential hashes, then
registers the HTML as a `run-report` artifact. The page is offline and
filterable: it summarizes metrics, renders `navigation.route_sample` points,
shows `progression.*` unlock/catalog events, lists explicit `match.result`
records (including `match_kind=pvp`), and links authorized screenshots or
videos by their checksummed artifact paths.

Reports are projections over the run store, not a second source of truth. They
do not mutate training data, infer missing unlocks or wins, widen action masks,
or establish live-game acceptance. Keep unsupported panels empty and return to
the adapter/runtime boundary when authoritative evidence is missing.
