# ADR-0019: Add an optional DeepSeek Harness provider boundary

- Status: Accepted
- Date: 2026-09-02

## Context

Applications may want a structured task/control-plane integration with a
DeepSeek Harness while keeping GLR's learner-neutral environment contracts
stable. A model provider must not silently become a game adapter, learner, or
runtime-action authority, and tests must not require external credentials.

## Decision

Add a small Python `harness` port with immutable task, capability, result,
event, and snapshot contracts. `DeepSeekHarnessProvider` implements the port
without a network client: it is disabled by default and requires an explicit
application-supplied handler when enabled. Capabilities are negotiated through
a permission allowlist; the default allowlist excludes `runtime.act`.

Results are cached by caller-supplied idempotency key, including failures and
timeouts. No mutating retry is performed. Snapshots contain completed results
and ordered events only, are bound to provider and schema identity, and fail
closed on restore mismatch. `HarnessOrchestrator` is an optional protocol and
`LocalHarnessOrchestrator` is a minimal eventing implementation.

## Alternatives considered

- Embed a DeepSeek SDK or read environment credentials in GLR: rejected because
  it would add a mandatory external dependency and obscure authorization.
- Add model/task fields to `TimeStep` or `Transition`: rejected because those
  contracts are learner-neutral runtime data contracts.
- Grant a generic execute action: rejected because external model output is
  advisory and cannot widen runtime authority.

## Consequences

The provider can be tested deterministically with a local handler and recovered
after process restart. Integrators still own authentication, transport,
credential handling, and any separate authorization review for a real backend.
The harness does not prove live game integration or model quality.
