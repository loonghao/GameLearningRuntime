# Gameplay research and reward design

## Research output

The research manifest is design evidence, not a runtime database dump. Keep it
small, reviewable, and refreshable. Each source should record:

- a stable source ID;
- canonical HTTP(S) URL and publisher;
- `official`, `wiki`, `guide`, or `community` source type;
- access timestamp and the source's update timestamp when visible;
- a short paraphrased summary;
- confidence and volatility.

Each claim references one or more source IDs and records:

- category: `mechanic`, `strategy`, `reward-hypothesis`, or `safety`;
- status: `unverified`, `runtime-verified`, or `rejected`;
- a concise paraphrase, not copied prose;
- contradictions or version assumptions.

Search patch notes again when a mechanic is version-sensitive. If two sources
disagree, retain the disagreement and test it; do not silently choose one.

## Convert research into contracts

Use claims to ask concrete adapter questions:

| Claim type | Contract question | Evidence required |
|---|---|---|
| mechanic | Is this observable and stable enough for `observation`? | authoritative state readback |
| mechanic | Is this a discrete/continuous action or an action mask rule? | accepted action plus postcondition |
| strategy | Should the policy receive it as optional context? | freshness and ablation |
| reward-hypothesis | Which named runtime signal proves progress? | bounded trace and counterexample |
| safety | What action must be masked or rejected? | fail-closed negative test |

Do not expose inaccessible hidden state merely because a guide mentions it.
Choose observations that the authorized runtime can truthfully and repeatedly
produce.

## Reward design sequence

1. Define the task objective in game-neutral terms.
2. Start with sparse terminal or milestone signals from authoritative runtime
   state.
3. Add dense shaping only for observable progress that cannot reward cycling,
   stalling, farming, or invalid actions.
4. Give every contribution a stable name, source, weight, raw clip, and required
   authority.
5. Log the immutable reward breakdown, never hidden executable logic.
6. Test missing, duplicate, stale, mismatched-source, non-finite, and extreme
   signals.
7. Run ablations for each shaping term and inspect behavior, not just return.

Guide-derived recommendations may seed a `reward-hypothesis`, but they remain
advisory until runtime evidence validates the underlying signal. Even after
validation, the guide is provenance for the hypothesis; the runtime is the
reward authority.

## Privacy and copyright

- Store links and paraphrased facts, not full guides or large excerpts.
- Never store logins, cookies, account identifiers, telemetry endpoints, local
  paths, process/window identifiers, or proprietary assets.
- Keep public examples synthetic and game-neutral.
- Respect access controls, robots directives, site terms, and source licenses.
- Do not automate access to authenticated sources without explicit authority.
