# Season and ruleset selection

This contract requires a season-aware CLI. Python convenience helpers require a
matching SDK; other roles may implement the versioned context receipt contract
directly. A local source build is not an installed release. All examples are synthetic.

## Declare, initialize, then review

The project opts in explicitly; projects without this table retain legacy behavior:

```toml
[seasons]
config = "config/seasons.toml"
```

The reference is root-relative, not relative to the catalog. It may point to a
missing catalog for initialization. GLR does not edit the root manifest for you.

```shell
glr --project . --season example-season --ruleset standard --json season init
glr --project . --json season list
glr --project . --season example-season --ruleset standard --json season show
```

`init` registers the pair and creates only
`config/seasons/example-season/standard.toml` with `status = "pending"`. It never
copies a game, imports code, executes a hook, installs a dependency, or overwrites
an existing declaration. It rewrites the strict catalog through a temporary file
under a short exclusive initialization lock. Review the catalog diff: comments
and formatting are not retained. Interrupted setup may leave empty directories;
it must not replace an existing project file or start a run.

The catalog has exactly these fields (an initial empty catalog uses `entries = []`):

```toml
schema_version = "glr.seasons.v1"
[[entries]]
season_id = "example-season"
ruleset_id = "standard"
config = "config/seasons/example-season/standard.toml"
```

Each declaration has exactly the required identity/status fields below and an
optional `extensions` table. Season/ruleset IDs and namespaces are bounded opaque
identifiers matching `[a-z][a-z0-9_-]{0,63}`. GLR owns no gameplay meaning for them.

```toml
schema_version = "glr.season.v1"
season_id = "example-season"
ruleset_id = "standard"
environment_id = "example.environment-v1"
protocol_version = "1.0"
status = "pending"

[extensions.training]
config = "config/training.toml"
[extensions.preset]
config = "config/preset.toml"
```

Every extension has exactly one `config` field. The extension owns its contents;
GLR reads bytes only and never imports them as code. A nested input such as a
preset must also be registered explicitly to join the frozen scope. Shared source
code need not be copied per season. Machine-local overrides remain local and do
not acquire portable scope or publication authority.

Selection must match both catalog and declaration, and declaration environment
and protocol must match the project. Unknown fields, duplicate pairs, invalid
status values, missing files, path escape, links/reparse points and ambiguous
identity fail closed. Catalogs hold at most 256 entries, declarations at most
32 extensions, UTF-8 paths at most 512 bytes, and each selected input at most
1 MiB. Catalogs/declarations are UTF-8 TOML. Extension format validation remains
owned by its reviewed consumer.

## Readiness and execution gates

`--season` and `--ruleset` must be supplied together. Configured projects require
an explicit pair before any runtime/training/player role. `train`, `goal run` and
`play` additionally require `ready`. Do not create a second active-profile/default
selection source inside an adapter. List and doctor can inspect without selection.

```shell
glr --project . --season example-season --ruleset standard --json doctor
glr --project . --season example-season --ruleset standard --json runtime start
glr --project . --season example-season --ruleset standard --json train
glr --project . --season example-season --ruleset standard --json goal run --goal goals/example.json
```

`runtime start` deliberately permits a selected pending declaration so a reviewed
runtime can probe its actual version and readiness. This is not permission to
queue a match or train implicitly. After project-specific conformance and version
checks, a developer can review and change the declaration to `ready`.

Doctor separates `installation_ready` (local bridge/executable dependencies) from
`training_config_ready` (the selection/status gate); `ready` is their conjunction.
`live_runtime_verified` remains false: no generic configuration status proves the
actual game ruleset, target identity, rendering, action readback or whole-game
acceptance. Keep the live gate in the runtime/adapter.

## Frozen role context and evidence

One invocation freezes `glr.season-context.v1` containing exactly:

- `schema_version`, `season_id`, `ruleset_id`, `environment_id`, `protocol_version`, `status`;
- `project`, `catalog`, `declaration`: each `{path, sha256, size_bytes}`;
- `extensions`: namespace to the same `{path, sha256, size_bytes}` shape;
- `context_sha256`.

Paths are project-root-relative; no machine path or timestamp is included.
`context_sha256` hashes UTF-8 JSON of the entire object without that field, using
recursively sorted keys, no insignificant whitespace, and unescaped Unicode
(Python `ensure_ascii=False`). Only strings, integers and objects occur. Identical
bytes reproduce identical context; newline conversion changes identity, so pin
line endings when transferring byte-bound projects. The shared synthetic fixture
tests this encoding in both languages.

All roles, including capture and each goal stage, receive:

| Variable | Value |
| --- | --- |
| `GLR_SEASON_ID` | Selected season ID |
| `GLR_RULESET_ID` | Selected ruleset ID |
| `GLR_SEASON_CONFIG_SHA256` | Declaration SHA-256 |
| `GLR_SEASON_CONTEXT_SHA256` | Aggregate context SHA-256 |
| `GLR_SEASON_CONTEXT` | Full compact context JSON, bounded to 24 KiB |

GLR clears inherited season variables for an unselected legacy project. It checks
the original frozen bytes before every role spawn, rather than silently loading
a different selection. A catalog edit during a run invalidates its frozen context,
even if the edit adds another pair. Do not mutate frozen inputs during a run.

Python `load_project()` consumes and verifies a complete inherited frozen context
and exposes `project.season_context`. A partial, forged, duplicate-key or stale
context is rejected. `load_season_context(project, environment)` provides explicit
SDK validation; `select_season(project, season, ruleset)` is for deliberate local
selection, not a fallback inside a role missing its CLI context. Consumers should
parse the exact verified input bytes, fence on the runtime's actual ruleset, and
retain their own live guards throughout execution. A pre-spawn file check is not
an immutable filesystem or a guard against arbitrary in-process code.

Each executing run receives a checksummed `season-context.json` artifact and a
`season.selected` event before its first role. Rust also includes the context in
run metadata. This proves which configuration was selected, not that an arbitrary
custom role honored it or learned successfully. Inspect readback and dataset/model
scope separately; do not silently reuse incompatible observations, weights or
rewards across rulesets.
