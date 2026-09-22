# Anti-fork gate

A fork or a copy of GLR does not break loudly. It drifts: the `origin` remote is
repointed, the branch quietly falls hundreds of commits behind, or version and
schema constants are edited locally until the copy can no longer merge back. By
the time anyone notices, the divergence is unrecoverable.

`glr fork-gate` turns that drift into an explicit, machine-readable gate. It is
read-only — it never fetches, never rewrites, never pushes — and it is designed
to be run by a scheduler so drift is caught while it is still cheap to fix.

## What it checks

| Check | Blocking by default | Fails when |
| --- | --- | --- |
| `origin-url` | yes | `origin` is missing, or does not normalize to the canonical URL. |
| `upstream-divergence` | yes | `origin/<default-branch>` is unavailable — usually means nobody ran `git fetch`. |
| `commits-behind` | yes | The checkout is more than `--max-behind` (default 50) commits behind. |
| `commits-ahead` | no — advisory | More than `--max-ahead` (default 200) commits ahead. Being ahead is not drift; it is unreviewed work. |
| `version-alignment` | yes | `pyproject.toml` version disagrees with `.release-please-manifest.json`. |
| `schema-version:<label>` | yes | A pinned wire schema version no longer matches the canonical value. |

Remote comparison is **normalized**, so cosmetic differences never fail the
gate. `git@github.com:org/repo.git`, `https://github.com/org/repo.git`,
`ssh://git@github.com/org/repo` and `https://github.com/org/repo/` are all the
same upstream: scheme, user, trailing slash, and `.git` suffix are stripped and
the result is lowercased.

The schema check is the subtle one. `FORK_GATE_SCHEMA_VERSIONS` in `cli.py`
holds **pinned literals**; the probe reports what this checkout's own modules
actually say (`LOCAL_SCHEMA_VERSIONS`). A derived checkout that edits a schema
constant therefore fails, instead of trivially comparing a constant against
itself.

### Bumping a wire schema version

Because the expectation is pinned, bumping a schema is a two-place change. Both
sides live in `src/game_learning_runtime/cli.py`, and
`tests/test_cli_governance.py::test_pinned_fork_gate_schema_versions_match_this_checkout`
fails if you do only one of them:

1. Bump the constant in its owning module (for example
   `WATCHDOG_SCHEMA_VERSION` in `watchdog.py`), with a migration note and an
   ADR if the payload shape changed.
2. Update the pinned literal for the same label in `FORK_GATE_SCHEMA_VERSIONS`.
3. Run `vx just core-check`. The parity test is the guard: forgetting step 2
   makes the gate report drift on the canonical repository, and forgetting step 1
   makes it report drift on every derived checkout.

Do not "fix" a schema-version blocker by editing only the pinned literal — that
is the drift the check exists to detect.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Gate passed. No blocking findings. |
| `5` | Gate blocked. See `blockers` in the report. |

Nothing else is returned, so a cron job or CI step needs no output parsing.

## Running it

```bash
# Default: canonical origin, main, 50 behind / 200 ahead.
vx just glr-fork-gate

# Explicit, with a fetched upstream.
git fetch --quiet origin
vx uv run --no-sync python -m game_learning_runtime.cli --format json fork-gate

# A deliberate downstream fork: report drift without failing on it.
vx uv run --no-sync python -m game_learning_runtime.cli --format json fork-gate \
  --origin https://github.com/your-org/YourFork.git \
  --allow-foreign-origin
```

### Flags

| Flag | Effect |
| --- | --- |
| `--origin URL` | Expected canonical remote. Defaults to `https://github.com/loonghao/GameLearningRuntime.git`. |
| `--default-branch NAME` | Branch to measure divergence against. Default `main`. |
| `--max-behind N` | Allowed commits behind. Default `50`. |
| `--max-ahead N` | Allowed commits ahead (advisory). Default `200`. |
| `--allow-foreign-origin` | Downgrade a non-canonical origin to an advisory. |
| `--allow-version-drift` | Downgrade version misalignment to an advisory. |
| `--allow-missing-upstream` | Downgrade an unfetched upstream ref to an advisory. |
| `--ignore-schema-versions` | Skip the pinned schema checks. |

Use the `--allow-*` flags **only** for a fork that knowingly tracks a different
upstream. Weakening the gate on the canonical repository defeats its purpose.

## Report shape

```json
{
  "schema_version": "glr.fork-gate-report.v1",
  "passed": false,
  "exit_code": 5,
  "blockers": ["origin-url"],
  "advisories": ["commits-ahead"],
  "findings": [
    {
      "check": "origin-url",
      "passed": false,
      "blocking": true,
      "detail": "origin does not match the canonical upstream",
      "observed": "github.com/someone/else",
      "expected": "github.com/loonghao/gamelearningruntime"
    }
  ]
}
```

Every finding carries `observed` and `expected`, so an alert can state the
remediation without a human opening the code.

## Remediation

| Blocker | Fix |
| --- | --- |
| `origin-url` | `git remote set-url origin https://github.com/loonghao/GameLearningRuntime.git`, or pass `--allow-foreign-origin` if the fork is intentional. |
| `upstream-divergence` | `git fetch origin`. The gate cannot measure divergence without the ref. |
| `commits-behind` | Rebase onto `origin/main`. Do it now: the budget exists so the rebase stays small. |
| `version-alignment` | Release Please owns both values. Restore them rather than editing either by hand. |
| `schema-version:<label>` | Restore the canonical constant, or bump the pinned literal in `cli.py` as a reviewed change with an ADR. |

## Interception points

The gate is cheap enough to run anywhere and is deliberately not wired into
`just check` — a fork may legitimately fail it, and local quality gates must
stay independent of upstream reachability. Put it where drift actually needs to
be caught:

- **Scheduled, on any long-lived checkout** — hourly fetch plus
  `vx just glr-fork-gate`. This is the primary interception point.
- **Before an unattended training run** — a nightly trainer on a drifted copy
  produces evidence nobody can reproduce.
- **In a fork's own CI** — run it with `--allow-foreign-origin` so real drift
  (behind, version, schema) still fails while the intentional origin is accepted.

```cron
7 * * * * cd /srv/GameLearningRuntime && git fetch --quiet origin && vx just glr-fork-gate >> /var/log/glr-fork-gate.log 2>&1
```

## Design boundaries

- **Read-only.** No fetch, no rewrite, no push, no network beyond `git` reading
  a local remote configuration.
- **Bounded.** Every `git` call is a read with a 30-second timeout.
- **Degrades explicitly.** When git or a version file is unavailable the probe
  returns `None` and the finding says so, rather than guessing `0`.
- **Injected runner.** `GitRepositoryProbe` accepts a command runner, so the
  gate is fully testable without a real repository.

## Related

- [Agent onboarding](agent-onboarding.md) — bootstrap and verify chain.
- [Supervision and watchdog](supervision-watchdog.md) — scheduled supervision.
- [Pin one entry point per project](entry-point.md) — drift against upstream is
  here; drift within the project is there.
- [Repository layout](repository-layout.md) — where automation lives.
- ADR-0037 in [the decision index](../decisions/README.md).
