# Build an interactive run report

GLR can turn a persisted local run into a self-contained HTML review artifact:

```powershell
glr --project . --json report build run-0123456789abcdef
```

The default output is `.glr/runs/<run-id>/report/index.html`. Use `--output`
for another directory inside that run directory. The command reads the
SQLite run projection, verifies every registered artifact against its size and
SHA-256 digest, and registers the generated HTML as a `run-report` artifact.
It never starts a runtime, sends an action, or changes a training dataset.

The report is offline and static. It includes responsive summary cards,
metric bars, a filterable event timeline, route samples, progression events,
match results, and links to checksummed artifacts such as review video and
post-match screenshots. The browser code uses `textContent` for runtime data;
event payloads are not interpreted as HTML or JavaScript.

## Adapter event vocabulary

The report is intentionally game-neutral. Adapters may append bounded events
to the run store using these namespaced kinds:

| Kind | Expected payload | Report surface |
| --- | --- | --- |
| `navigation.route_sample` | `position: [x, y, z]` (finite numbers) | 2D route trace |
| `progression.item_unlocked` | `item_kind`, `item_id`, optional `status` | Unlock table |
| `progression.catalog_snapshot` | `catalog_kind`, bounded snapshot summary | Progression history |
| `match.result` | `match_kind`, `outcome`, optional `turns`, `trophy_delta` | Match table |

These events are observations, not actions. A route is only authoritative when
the adapter obtains coordinates from an authorized runtime state source. A
rendered screenshot without semantic state is shown as media evidence and
must not be used to infer a route, unlock, or match result.

For PvP games, set `match_kind` explicitly to `pvp`; the report does not turn
generic combat events or monster wins into PvP wins. Capture a post-match card
image through the project's authorized recorder, register it with a portable
relative path and digest, and reference it from the match event payload.

## Evidence and privacy

- Keep raw reports local unless their metadata and payloads have been reviewed.
- Do not put account identifiers, process/window identifiers, hostnames,
  absolute paths, credentials, or proprietary state in event payloads.
- Keep media beside the run and let the artifact manifest bind bytes to their
  recorded SHA-256. The report builder fails closed on missing or changed
  artifacts.
- Mark uncertain projections in the event payload (`authority` or `status`)
  and preserve the distinction between authoritative, inferred, advisory, and
  human-validation-needed evidence.
- A report is review evidence, not proof of live-game acceptance. A successful
  HTML build only proves that the persisted run data and referenced artifacts
  were readable and verified.

The report format is `glr.run-report.v1`. It is an additive consumer of the
existing `glr.transition.v1`, run-store, and capture contracts; it does not
change tensor encoding or learner interfaces.
