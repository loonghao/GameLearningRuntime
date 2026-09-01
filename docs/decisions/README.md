# Architecture decision records

- [ADR-0001: Keep the runtime contract learner-neutral](0001-learner-neutral-runtime.md)
- [ADR-0002: Use nested tensor trees and versioned schemas](0002-versioned-tensor-contracts.md)
- [ADR-0003: Adapt Gymnasium at the outward boundary](0003-gymnasium-compatibility-boundary.md)
- [ADR-0004: Use Rust for benchmark-proven data-plane work](0004-benchmark-gated-rust-data-plane.md)
- [ADR-0005: Share objective primitives, not learner implementations](0005-share-objectives-not-learners.md)
- [ADR-0006: Distinguish live attach from reset](0006-distinguish-live-attach-from-reset.md)
- [ADR-0007: Standardize bridge lifecycle, not game transports](0007-standardize-bridge-lifecycle.md)
- [ADR-0008: Configure knowledge and rewards as strict data](0008-configure-knowledge-and-rewards-as-data.md)
- [ADR-0009: Profile engine-plugin and external-attach integrations](0009-profile-engine-plugin-and-external-attach.md)
- [ADR-0010: Add authorized loader plugins and reproducible model bundles](0010-add-authorized-loader-plugins-and-reproducible-model-bundles.md)
- [ADR-0011: Enforce episode reward and demonstration safety](0011-enforce-episode-reward-and-demonstration-safety.md)
- [ADR-0012: Use a Runtime Host and engine provider SDKs](0012-use-a-runtime-host-and-engine-provider-sdks.md)
- [ADR-0013: Bind demonstration provenance to trajectory bytes](0013-bind-demonstration-provenance-to-trajectory-bytes.md)
- [ADR-0014: Inject bounded advisory knowledge contexts](0014-inject-bounded-advisory-knowledge-contexts.md)
- [ADR-0015: Add an agent-first local control plane](0015-add-an-agent-first-local-control-plane.md)

Accepted ADRs describe implemented architecture. Proposed future designs belong
in `docs/planning` until accepted and built.
