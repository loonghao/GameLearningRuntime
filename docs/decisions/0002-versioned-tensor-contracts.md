# ADR-0002: Use nested tensor trees and versioned schemas

## Status

Accepted

## Context

Target games require discrete, continuous, hybrid, parameterized, and
hierarchical actions, dynamic observations, action masks, and cross-language
adapters. A flat fixed discrete action cannot express those requirements.

## Decision

Use recursively nested composite specs with typed tensor leaves. Package a
`glr.v1` Protobuf runtime schema and a `glr.transition.v1` JSONL dataset schema.
Episode UUIDs and monotonic step IDs are mandatory lifecycle identities.

## Consequences

### Positive

- Hybrid and hierarchical actions share one validation model.
- Schemas can generate clients for multiple languages.
- Compatibility boundaries are explicit.

### Negative

- Dynamic shapes cannot map directly to every learner framework.
- Protobuf maps flatten tree paths at the transport adapter boundary.
- JSONL/base64 is portable but not the final high-throughput storage format.

### Neutral

- High-throughput shared memory may carry tensors out of band while preserving
  the same semantic schema.

## Alternatives considered

- JSON-only RPC: rejected for the primary wire contract because typed binary
  tensors and generated cross-language clients are required.
- A fixed tuple layout: rejected because it loses names and hierarchy.
- Shared memory as the only transport: rejected because it is platform-local
  and adds lifecycle complexity before semantics are stable.

