# Project output layout

GLR projects have two different kinds of generated data:

- Run evidence is immutable, run-scoped material and belongs in
  `.glr/runs/<run-id>/`.
- Durable exports are handoff material and belong in `.glr/exports/`, grouped as
  `knowledge/`, `model-bundles/`, or `loader-packages/`.

Source code, bridge files, game installations, and checked-in fixture datasets do
not move into `.glr/`. A role must write generated files below the `GLR_RUN_DIR`
provided by the CLI; it must not invent a sibling `artifacts/`, `recordings/`,
`reports/`, or `logs/` directory.

## Applying the contract to the current game projects

The three working directories currently have different histories and must not be
bulk-moved as one operation:

| Project | Keep as source | Route generated output to |
| --- | --- | --- |
| JK Chess (`jcc`) | `live/`, `seasons/`, `config/`, and the game installation | `.glr/runs/<run-id>/`; durable checkpoints/exports under `.glr/` |
| Wukong (`rl-wukong`) | `bridge/`, `rl/`, and the checked-in `data/legacy-trajectories/` fixture | `.glr/runs/<run-id>/`; do not relabel legacy traces as successful demonstrations |
| Vampire Survivors | `vsrl/`, `adapters/`, `mod/`, `config/`, and `knowledge/` | split run reports/captures into `.glr/runs/<run-id>/` and models/knowledge into `.glr/exports/` |

Before migrating any one project:

1. Add one `glr-project.toml` with `data_dir = ".glr"` and run
   `glr --project . --json doctor`.
2. Change each configured role to consume `GLR_RUN_DIR` (or its role-specific
   path) for logs, datasets, captures, and result JSON.
3. Move existing files only after recording their provenance and mapping them to a
   run or export category. Never overwrite an existing destination.
4. Re-run `glr capture layout`, `glr runs list`, and `glr report build <run-id>`
   for the migrated evidence.

This is deliberately an explicit migration boundary. The CLI does not silently
move existing files, and a video, checkpoint, or report is not made authoritative
merely by placing it below `.glr/`.
