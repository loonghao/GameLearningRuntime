# Learner-owned decisions

GLR's `Policy` and `SyncCollector` already support learning algorithms. Running
a trainer executable successfully does not establish that its policy changed,
improved, or achieved the objective. Basic `glr train` therefore records
`status_scope=process_execution`, with learning and improvement unverified.
These fields persist with the run; they do not change exit-code semantics.

For dynamic action spaces, `game_learning_runtime.decisions` separates:

- Adapter: enumerate feasible `Candidate` values from observed capabilities.
- Learner: select a candidate key and stamp the policy digest and train/evaluate mode.
- Executor: submit the selected command and parameters and retain its receipt.
- Trainer: update from observed transitions, save a checkpoint, and evaluate it frozen.

The execution helper does not select a fallback after rejection. An actuator may
implement a bounded movement primitive but may not choose another strategic
destination. Its returned provenance includes the decision state and every
candidate's key, command and parameters, including alternatives that were not
selected, so equal-sized candidate sets remain distinguishable. Missing
capabilities should be exposed as limitations, not repaired by silently
reintroducing a scripted controller.

`learning_status` reports only whether recorded transitions accompanied a policy
change. It does not authenticate a trainer's claims or certify improvement. A
held-out evaluation must compare policies on comparable starts and objectives;
changing a digest or incrementing a counter is insufficient. Keep frozen
evaluation out of training data and verify that policy parameters did not change.

This module complements the tensor-based collector; it does not replace it or
require Q-learning, PPO, TorchRL, or any particular model architecture.
