# Agent onboarding

This is the self-explaining entry point for this repository. An agent — or a new
human contributor — should be able to clone the project, prove it works, and
prove it is still aligned with upstream by running one command chain, without
reading the source first.

Everything below is copy-paste runnable from a clean clone on Windows
PowerShell, macOS, or Linux.

## The one command chain

```bash
git clone https://github.com/loonghao/GameLearningRuntime.git
cd GameLearningRuntime
vx setup                 # installs the pinned Python, Node.js, uv, just, Rust, .NET inputs
vx just layout-check     # proves the checkout follows its own layout contract
vx just core-check       # ruff + ruff format + mypy + pytest (90% coverage gate)
vx just glr-doctor       # proves the control plane runs against this project
vx just glr-fork-gate    # proves this checkout still tracks canonical upstream
```

If all five succeed, the checkout is bootstrapped, tested, runnable, and
aligned. If any fails, see **Failure criteria** below — every step names its own
exit code so a scheduler or agent can branch on it without parsing output.

## Platform and pinned inputs

| Input | Value | Source of truth |
| --- | --- | --- |
| Python | 3.12.13 | `vx.toml`, `.python-version` |
| Rust | 1.98.0 | `rust-toolchain.toml` |
| .NET SDK | 10.0.400 | `global.json` |
| Node.js / npm | pinned by `vx` | `vx.toml`, `vx.lock` |
| uv, just | pinned by `vx` | `vx.toml` |
| Python dependencies | frozen | `uv.lock` |

Do not hand-install these. `vx setup` reads the committed files and installs the
exact versions; CI uses the same recipe, so local and CI cannot disagree about
which interpreter or toolchain is "correct".

## Environment

| Name | Meaning |
| --- | --- |
| `UV_PROJECT_ENVIRONMENT` | Set to `.venv-glr` by the `justfile`. The project-owned environment; never activate a different one. |
| `MSYS_NO_PATHCONV=1` | Git Bash / MSYS2 only. Set it when passing POSIX-style paths (for example `/usr/bin/true`) as command arguments, or MSYS rewrites them to Windows paths. |

There are no required secrets, tokens, or credentials for the core chain. Engine
provider lanes (`Unity`, `Unreal`, `Godot`, `DCC-CUA`) need their own installed
editor or executable and are optional.

## What each step proves

| Step | Proves | Exit code |
| --- | --- | --- |
| `vx setup` | Dependency resolution from `uv.lock` is reproducible and complete. | non-zero = environment broken |
| `vx just layout-check` | Every tool under `tools/` is registered in `tools/registry.toml` and owned by a capability domain. | non-zero = unregistered or orphaned tool |
| `vx just core-check` | Lint, format, strict typing, and the test suite with the 90% coverage gate. | non-zero = quality gate failed |
| `vx just glr-doctor` | The control plane loads the project and reports role and launch readiness. | non-zero = project not runnable |
| `vx just glr-fork-gate` | `origin` matches upstream and drift is within budget. | `0` aligned, `5` blocked |

## Machine-readable mode

Every control-plane command emits the stable `glr.cli-output.v1` envelope with
`--format json`. Agents should always use this form; human tables are for
reading, not parsing.

```bash
vx uv run --no-sync python -m game_learning_runtime.cli --format json doctor
vx uv run --no-sync python -m game_learning_runtime.cli --format json fork-gate
vx uv run --no-sync python -m game_learning_runtime.cli --format json watchdog tick --source trainer
```

The entry point is `python -m game_learning_runtime`, not
`python -m game_learning_runtime.cli`.

## Failure criteria

Stop and report rather than guessing. Map the failure to its gate:

- **`layout-check` fails** → a tool exists outside `tools/`, or is missing from
  `tools/registry.toml`. Fix by registering it; see
  [repository layout](repository-layout.md). Never delete the gate.
- **`core-check` fails** → one of `ruff check`, `ruff format --check`, `mypy`,
  or `pytest` failed. The printed command is authoritative; fix the reported
  file, do not relax the configuration.
- **`glr-doctor` fails** → the project manifest, a configured role, or game
  launch readiness is wrong. This is a project problem, not a repository
  problem; check `glr-project.json` first.
- **`glr-fork-gate` exits 5** → the checkout has drifted. Read
  [the fork gate guide](fork-gate.md); the JSON report names the blocking
  finding and the exact remediation.
- **`vx setup` fails** → the pinned inputs are unavailable on this host. This is
  an environment blocker; report it instead of substituting a different
  interpreter.

## Where things live

Before adding anything, read [repository layout](repository-layout.md). The
short version: library logic goes in `src/game_learning_runtime/`, automation
goes in `tools/<domain>/` and must be registered, and there is no `scripts/`
directory.

## Scheduled and unattended runs

Unattended training uses the same entry points with explicit exit codes:

- [Supervision and watchdog](supervision-watchdog.md) — heartbeat evaluation,
  bounded restarts, and the cron contract.
- [Anti-fork gate](fork-gate.md) — drift detection for derived checkouts.

```bash
# One supervision pass: 0 healthy, 3 recovered, 4 escalated.
vx just glr-watchdog --source trainer --timeout 30 --restart-limit 3
```
