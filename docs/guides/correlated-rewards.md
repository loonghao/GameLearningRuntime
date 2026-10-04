# Correlated rewards

Use the strict reward path when positive shaping must be justified by one
observed action effect. Keep game-specific measurements inside the adapter.

The owner creates a `CorrelationPolicy` with the run, environment, protocol,
target and SHA-256 digest of `environment.config_snapshot()`. Construct a
`CorrelatedRewardGuard` with that policy and the reviewed `TrainingConfig`
and `RewardSafetyConfig`, then pass it as `correlated_rewards` to
`SyncCollector`. Binding a store and run also persists validated receipts.

Each reset/attach and step observation supplies an `ObservationContext` as
`timestep.info["glr.observation-context"]`, using its `to_mapping()` result.
Its episode, step, timestamp and producer sequence must identify that exact
observation. Report `phase` and `alive` explicitly; unknown facts do not
become gameplay evidence.

The post-state supplies `timestep.info["glr.reward-evidence"]` containing
exactly `signals` and `attributions` arrays. Signals have `name`, `source`
and finite `value`. Every positive contribution, including a positive terminal
outcome or unbudgeted term, requires an attribution containing `signal_name`, `source`,
`action_id`, `before_sequence`, `after_sequence` and `effect`. Use
`RewardAttribution.to_mapping()` to produce that record. `effect="confirmed"`
is an adapter-measured fact for the specified signal, not a runtime inference
from the action's acceptance.

The action receipt identifies the post-step, the target, the observation
sequence it was issued against, and the fresh authoritative post-sequence.
Positive rewards require both an accepted receipt and confirmed effect
evidence. An accepted action with no effect cannot earn positive credit.
Negative penalties and authoritative terminal failure remain valid.

The collector owns composition on this path and checks the adapter's observed
scalar reward against the bounded result. An incorrect reward, incomplete
context or stale interval closes the episode before returning learner data.
The runtime does not retry the action or infer a missing receipt. The caller
must arrange a fresh authorized reset/attach before resuming.

Strict collection requires `on_error="raise"`; it returns no partial unroll
after an indeterminate outcome. Rewards are finite scalar float32 or float64
values, checked against composition rounded to the declared dtype. The
receipt's `learning_reward` is the actual observed scalar, while
`result.total` retains the composed value. Use `learning_reward` when reporting
TD operands. Consecutive intervals must continue the previously validated
post-state, including its context and tensor hash.

For custom direct composition, call `reset(episode_id)` and then
`compose(before, after, signals, attributions, action=chosen_action)`.
Set `verify_observed_reward=True` to check the post-state's reward. Use the
actual chosen tensor action, not a later policy recommendation. The default
limits are 4096 actions per episode and 4096 unique episodes per guard;
owners may select smaller limits.

In a collector these limits also admit operations before dispatch. Reset/attach
attempts are charged before the adapter call. Action attempts are checked before
the policy runs and charged before `environment.step`; failed attempts are not
refunded. Direct `compose` accepts already captured evidence and does not launch
or authorize an operation.

Validated receipts include hashes of the actual state, action and next state.
`Telemetry.correlated_reward()` records the bounded projection. For scalar
learners, report `ScalarLearningUpdate` using those hashes and the exact
reward receipt digest. Supply the owner-selected `LearningConsumerPolicy`
to `Telemetry.correlated_learning_update()` so a report cannot switch table,
learner or policy version. The runtime checks the reported TD arithmetic and
requires a zero bootstrap value on terminal updates.

Store events use `episode-<uuid>` as their portable episode identifier. The
receipt retains the original UUID, so this projection does not change episode
identity or establish reward authority.

These learner events report instrumentation. They do not prove a table was
mutated, and they do not replace independent evaluation. General neural
learners may keep their existing `learning_update` diagnostics; GLR does not
derive a scalar update or gameplay success from those messages.

Fixed replay suites freeze their own reward configuration and source run
identity. They validate captured actions and observations through the same
strict guard. Missing fields remain unknown coverage, blocking promotion.
Offline replay conformance does not measure live task improvement.
