# Route evidence and QA

Campaigns should preserve a bounded transition trace and a final run manifest. Each transition records route index, live position, semantic action, producer state sequence, encounter identity, outcome, recovery action, and the associated recording reference. This supplies training maintenance with route coverage and gives glr-qa deterministic input for replay and regression checks.

Consumers reject traces with missing recording provenance for combat, stale producer bindings, or incompatible environment identity. Replaying a trace must re-observe every checkpoint; a recorded coordinate is never a blind input script.
