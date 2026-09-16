# Agent-first training workbench

GLR's workbench follows the agent's recorded goal, decisions, execution receipts,
learning updates, and evidence. CLI, HTTP and the browser read the same durable
run store. The browser supplies human inspection and CLI-backed operations; it
does not introduce a second orchestration loop or infer successful learning from
a running process.

## Game-neutral data views

Adapters can opt into `glr.workbench.v1` inside a `bridge.state` event's
`payload.workbench`. This is presentation data, not executable UI or game logic.
The [schema](../schemas/workbench.schema.json) is also in `$defs.workbench` of
`glr --json telemetry schema` and `/api/v1/telemetry/schema` (inlined at runtime).

| Environment type | Example data to report | Reusable view |
| --- | --- | --- |
| Wukong-style action combat | Health, stamina, encounter phase, action candidates | Stats + candidate table + route samples |
| The Bazaar-style economy/build | Currency, day, inventory, shop candidates, battle outcomes | Stats + inventory/candidate tables |
| Vampire Survivors-style survival | Elapsed time, level, wave, weapons, upgrade choices | Stats + loadout/upgrade tables + metrics |
| Other environments | Adapter-owned scalar fields, row data, summaries | Same primitives; unknown events remain inspectable |

These are mapping examples, not claims that those game adapters already publish
the contract. No game names or reward semantics are hardcoded in the renderer.
Route panels appear only when recorded route samples exist. Metrics and media
retain their existing contracts and can accompany any view.

```python
from game_learning_runtime import BridgeTelemetry

publisher = BridgeTelemetry.from_env("bridge.my-environment")
if publisher is not None:
    batch = publisher.prepare(events=[{
        "kind": "bridge.state", "episode_id": "episode-1", "step_id": 42,
        "payload": {"workbench": {
            "schema_version": "glr.workbench.v1",
            "title": "Encounter training", "agent": "policy.baseline",
            "objective": "Evaluate action selection", "phase": "Encounter",
            "sections": [
                {"id": "state", "title": "Player state", "kind": "stats",
                 "fields": [{"label": "Health", "value": 72, "unit": "%"}]},
                {"id": "choices", "title": "Candidates", "kind": "table",
                 "columns": ["Action", "Score", "Selected"],
                 "rows": [["dodge", 0.8, True], ["attack", 0.3, False]]}
            ]
        }}
    }])
    publisher.send(batch)
```

The values above are synthetic examples. Publish actual observations in a real
adapter. Each `bridge.state` replaces the latest snapshot for that run/source,
so report the complete current view. Multiple producers remain separately
selectable. All historical events remain durable and accessible with
`glr --json runs trace RUN_ID`; `glr --json telemetry state RUN_ID` reads the latest
snapshot. Existing backups preserve both. A view has no authority to execute
actions or satisfy rewards.

Version 1 supports `stats`, `table`, and plain `text`. Maximums: 8 sections,
12 stat fields per section, 8 table columns, 40 rows, 240 characters per scalar
string, and 4,000 characters per text section. Section IDs must be unique; every
row must match the column count. The existing 12 KiB event payload and 64 KiB batch
limits still apply. Null scalar values render as missing; labels/optional fields
must be strings when present. Values are escaped text, never HTML or scripts.
Invalid new views reject the complete telemetry batch. Unsupported historical
views retain the raw inspector, including provenance. Agent clients can inspect
the same schema and values without rendering a web page.

Shared [fixtures](../examples/workbench-views.json) exercise combat, economy,
survival and custom shapes against Rust and TypeScript validation.

## Process view

The light, blue-accent workbench provides a four-lane overview, searchable steps,
source/category filters, and an input/output inspector. `agent.decision` maps to
Model, `agent.execution`/`tool.*`/`process.*`/`bridge.log` to Tools, `learning.*` to
Learning; all other kinds remain visible under System.

Recommended optional fields in an event payload:

| Field | Meaning |
| --- | --- |
| `label`, `summary`, `message` | Producer-recorded human-readable description |
| `input` | Structured input, with credentials omitted |
| `output` or `receipt` | Recorded output or execution receipt |
| `duration_ms` | Finite nonnegative measured duration; absent values show a dash |

No duration is inferred from adjacent events, and no private model reasoning is
generated. The overview uses receipt-time positions with reported-duration bars;
it is not an exact concurrency profiler or a model/tool time accounting report.
Only 400 matching events per lane are rendered. Rows show the latest 100 with
explicit loading of earlier steps from the browser's bounded window. Selecting
a row or marker opens details and updates the shared evidence selection; video
seeking still requires a matching capture manifest. Stages summarize the latest
recorded signals and are not a completion checklist. Pausing observation does
not stop an agent or its training job.
