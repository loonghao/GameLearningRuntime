# DeepSeek Harness provider

`DeepSeekHarnessProvider` is an optional, disabled-by-default task provider.
It is a control-plane seam for structured prompts, plans, or analysis; it is
not a learner, game adapter, or runtime-action channel. GLR does not discover
credentials, contact a network endpoint, or execute a task unless an
application explicitly enables the provider and supplies a reviewed handler.

Every task contains a bounded JSON payload, an idempotency key, a deadline, and
requested permissions. The provider only grants the permissions declared by
its capability document. `runtime.act` is intentionally not granted by the
default capability set. Results are cached by idempotency key, including
failures and timeouts, so a caller never blindly retries a mutation.

`LocalHarnessOrchestrator` is an optional eventing wrapper. Its ordered
`HarnessEvent` records and `HarnessSnapshot` can be persisted and restored;
active work is never serialized. The snapshot is provider- and schema-bound,
and restoration fails closed when either does not match.

See [`docs/examples/deepseek-harness.json`](../examples/deepseek-harness.json)
for a disabled configuration sample. The provider has no external dependency
or credential requirement.
