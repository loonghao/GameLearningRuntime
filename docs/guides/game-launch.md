# Start configured game instances before training

Use the project control-plane module through `vx just glr-train` when a project
owns an authorized game executable and a trainer command. The launcher starts
only the fixed argv declared in the project file; it never invokes a shell or
searches for an unrelated game process. The released Rust `glr` binary remains
the primary runtime control plane; this Python module is intentionally exposed
through the repository's `vx just` recipes rather than a Python console script.

```json
{
  "schema_version": "glr.project.v1",
  "game": {
    "schema_version": "glr.game-launch.v1",
    "game_id": "example.game",
    "command": {
      "argv": ["game.exe", "--instance", "{instance_id}"]
    },
    "instances": 4,
    "parallel": true,
    "max_parallel": 4,
    "working_dir": "tools",
    "environment": {"GLR_MODE": "train"},
    "readiness": {"kind": "file", "path": "ready.json"},
    "startup_timeout_seconds": 30,
    "shutdown_timeout_seconds": 10
  },
  "trainer": {"argv": ["python", "tools/train.py", "{run_dir}"]}
}
```

The example shows the `game` and `trainer` portion to add to a complete
`glr-project.json`; the full project also needs the identity, bridge, runtime,
player, and capture fields from the agent-first CLI guide.

Run it from the project root:

```powershell
python -m game_learning_runtime.cli --project glr-project.json doctor
python -m game_learning_runtime.cli --project glr-project.json train
vx just glr-train
```

For the standalone launcher, the same two-role document can be run with
`python -m game_learning_runtime.game_launcher --project glr-project.json --json`.

`parallel` controls whether more than one process may be started in the same
batch. `max_parallel` bounds each batch, so a configuration with 8 instances
and `max_parallel: 2` starts four bounded batches. Every instance receives
`GLR_GAME_INSTANCE_ID`, `GLR_GAME_INSTANCE_INDEX`, `GLR_GAME_INSTANCE_DIR`, and
`GLR_GAME_RUN_DIR`. The trainer receives `GLR_GAME_INSTANCE_COUNT`,
`GLR_GAME_INSTANCE_IDS`, and `GLR_GAME_INSTANCES_MANIFEST`.

The launcher writes one stdout/stderr pair under `.glr/runs/<run>/games/<id>`
and an atomic `game-instances.json` manifest. A file readiness signal is
removed before launch so a stale file cannot satisfy a new run. The signal only
means that the game process has reached the project's declared startup point;
the adapter must still perform its authoritative bridge handshake and target
binding before accepting training actions.

On trainer exit, timeout, startup failure, or an exception, all processes owned
by the run are terminated and then killed if they do not stop within the
configured shutdown window. Instance count is bounded to 64 and all paths are
project-relative, which keeps a typo from launching an unbounded or unrelated
process tree.

For library callers, use `TrainingLauncher` with a `GameLaunchConfig` and a
`LaunchCommand`. A standalone `glr.game-launch.v1` document can be loaded with
`load_game_launch_config` when a project has a separate trainer coordinator.
