# Bounded continuous learning

GLR can retain a goal, source revisions, proposals and evaluation decisions
across bounded research and training passes. A project-owned supervisor
chooses when to schedule another pass. This SDK does not launch a continuous
service or promise uninterrupted operation.

## Divide responsibilities

| Component | Responsibility |
| --- | --- |
| Game adapter | Observe current state; expose legal actions, reset or attach, rule entry points and authoritative success evidence |
| Agent or learner | Propose knowledge, interfaces and policies as isolated artifacts with source revisions |
| Shared runtime | Bind targets and configurations; admit budgets/actions; keep leases, evidence, fixed evaluation and review decisions |
| Protected host | Issue scoped capabilities; launch or observe owned workers; confirm terminal states; keep evaluator inputs fixed; require independent review |
| Game owner | Maintain the actual input mutex, authorized interface and deployment boundary |

Do not copy the campaign kernel into each game. A source guide can inform a
proposal; it does not authorize an action or become an authoritative reward.
Coordinates and receipts belong to the exact environment, protocol, target
and configuration. Knowledge from a different version remains advisory.

## Perform one bounded pass

1. Resolve the actual project and owner. Preserve the current checkpoint and
   active processes. Freeze a goal with trial, step, time, source and artifact
   limits, legal action IDs, resource IDs, target and configuration digest.
2. Register content-addressed source revisions and an inert proposal. Admit
   the proposal only if its requested actions and baseline fit that goal.
   Claim a trial before dispatch. Unknown or expired workers retain their
   lease until the supervisor confirms their stop.
3. Run the candidate in an isolated workspace through the reviewed adapter.
   Keep evaluator code, direct script/file inputs and suite bytes outside
   candidate write access. Use a supervisor that can cancel and reap its
   owned processes at the wall deadline.
4. Evaluate the exact artifact against fixed evidence. Reject conflicting
   authoritative values, missing target/configuration, stale observations,
   illegal actions, changed parameters, incorrect reward attribution and
   action intervals. A historical high score cannot override a later low
   value of the same metric/source.
5. Confirm terminal worker evidence, then request independent host review.
   Recheck bytes, final measurements, budget, stop receipt and baseline at
   review and installation. Record rejection and preserve the incumbent if
   any binding changes.

The Python API is in `continuous_learning`, `knowledge_evidence` and
`replay_evaluation`. `CampaignStore` accepts a protected `HostAuthority`;
evaluation, stop confirmation and review require its role capabilities.
Do not serialize an authority or capability into a manifest, suite, log or
worker input. These are trusted host objects, not candidate credentials.

Python review moves an approved artifact reference. Fixed external policy
evaluation can update a logical checkpoint digest and score; it does not
replace a checkpoint file. Inert offline replay cannot approve that policy
reference or authorize a policy installation. The Rust
`PromotionHost` owns the checkpoint installation path and its durable
authorization IDs. There is no automatic conversion between these approval
contracts. Provision an empty Rust host ledger explicitly with
`PromotionHost::provision_empty`; `open` only reopens an already provisioned
ledger. Neither a legacy run nor an empty public run silently grants host
authority. Existing populated ledgers need a reviewed migration.

If an incumbent has no persisted evaluation score, a first authorized install
admits the owner-reviewed measurements; it does not establish improvement
relative to those incumbent bytes. Put required quality thresholds into the
fixed evaluator criteria instead of relying on a goal ID or baseline digest.

The Rust fixed evaluator writes `glr.checkpoint-evaluation.v1`: a
`coverage` mapping for all seven correctness metrics, each explicitly
`measured`, and an `evidence_bundle` using `glr.goal-evidence.v1`. Missing,
unknown or inapplicable coverage rejects installation. Every authoritative
value for those counters must be exactly zero, including persisted rows;
ordinary score tolerance does not apply to correctness counters. The host
binds the complete report bytes. The fixed external producer computes these
measurements; GLR does not infer them from incomplete native logs.

## Existing CLI behavior

`glr goal run` still performs the existing bounded research, planning,
training and evaluation flow. A candidate that passes those checks is
reported as staged for review; it is not installed automatically. The old
`Store::promote_checkpoint` method rejects direct calls. A project integrates
the protected Rust host API to launch declared workers, record final evidence,
review and install an authorized candidate.

For existing manifests, remove the legacy `{promotion_path}` command
placeholder. The native project loader rejects it before creating a run or
starting a worker. Keep `{checkpoint_path}` and `{candidate_checkpoint_path}`
for declared checkpoint and candidate paths; only `PromotionHost` installs
an authorized candidate.

Existing project tasks, lifecycle hooks, supervision and watchdog policy keep
their current boundaries. This change adds SDK contracts rather than a new
persistent scheduler or CLI host command. Follow the existing supervision
guide and commands present in the installed version. A deployment owns its
scheduler and access rights.

## Read replay results honestly

Replay can establish that particular captured receipts and learning updates
agree with fixed rules. Preserve producer sequence, episode, observation,
target, configuration and action bindings. Missing fields stay unknown;
do not manufacture them from an adjacent log row.

An inert knowledge/interface replay may explicitly exclude policy checks.
The report must keep their reason and applicability. It cannot label an
excluded check measured or turn a partial replay into live policy evidence.
Synthetic fixtures test the gate; they do not establish gameplay improvement.

## Resume and recover

Back up the campaign database and its immutable source/artifact references.
For Rust checkpoint recovery retain the database, bound owner/journal
sidecars, checkpoint blobs, the original candidate and evaluation output,
and the exact configuration, evaluator and suite dependencies. A same-path
database with a different epoch is a different owner. Unknown or mixed state
is preserved for diagnosis and refused.

The journal tests cover explicit application exit points. They do not prove
power-loss behavior or directory flush guarantees on every platform. Direct
child stop confirmation does not cover untracked descendants. Before a live
pass, the owner must connect this host to the game's actual process and input
ownership, legal receipts and reset/attach semantics.

A stdio or bridge close, including failed construction, can raise
`CleanupPendingError`. Retain that exception and call its `retry_cleanup()`
to retry the original fenced owner's cleanup. It does not reconnect, launch
another process or resend an action. `cleanup_complete` means that callback
returned successfully; custom adapter cleanup remains trusted application
code. This local recovery handle is not a serialized stop authorization or
a way to recover handles after a host crash.

A sudden host crash can leave direct workers running. Reopening the database
cannot recreate their owned handles or prove stop from a PID. Unresolved
launches remain reserved; a new run cannot take the same target merely by
changing run ID. This version has no API to reattach old handles or release
unknown reservations using a PID, TTL or external stop string. An owner must
verify cleanup and separately review a coherent repair or migration.

Rust host limits apply to a bounded host run/trial. They are not automatically
connected to the Python campaign's aggregate trial and research budgets. Host
role IDs enforce configured separation; they do not authenticate distinct
human reviewers.

## Keep the evaluation independent of learner configuration

Freeze `FixedReplaySuite.reward_training` and `reward_safety` with the suite,
not with a candidate output or its declared metrics. Each `ReplayEpisode`
explicitly binds its captured source `run_id`; this is distinct from the
new evaluation run and the logical replay episode UUID. Both reward
contracts and the source identity contribute to the frozen suite digest.
Source reset epochs must be unique within their explicit source run; a new
logical replay UUID cannot establish that a captured reset was fresh.

The evaluator consumes the captured action and original timestep
`glr.observation-context` and `glr.reward-evidence` through the strict
correlated reward API. It cross-checks producer sequence, lifecycle,
configuration and receipt identity rather than merging contradictory
projections. The composed nonzero reward terms must also agree with the
captured contribution names and values; equal scalar totals are insufficient.
Every positive term, including an outcome term, needs its own confirmed
action-effect claim. Missing contract, source run, context or alive measurement
keeps reward coverage unknown. A numeric sum or legacy contribution list
alone cannot establish causal attribution. Typed adapter evidence records
the adapter claim; it does not authenticate an external game effect or
prove that an independently owned learner table was updated.

`ReplayEnvironment` emits a fresh logical episode identity while retaining
original adapter contexts as source evidence. Those timesteps are for this
replay evaluator; pass `ReplayFrame.snapshot()` to the strict source guard,
not a timestep with its logical identity rewritten. The evaluator digest
includes the replay, correlated reward, training and reward-safety modules.
The host still owns callback globals, other installed dependencies and their
process isolation; a digest is not a sandbox or remote code attestation.

Knowledge lookup fingerprints similarly establish the queried revision,
not consumption. A `DecisionConsumptionReceipt` binds an explicit consumer,
observation and decision to the rule index. Rule and capability-gap data
remain passive and cannot expand the adapter action mask or success authority.

Shared offline parsing accepts bounded data and syntax trees. It neither
executes target source nor repairs missing historical fields. Game-specific
structure selection and live observation/effect semantics remain in the
adapter; public regression fixtures are independent synthetic data.
