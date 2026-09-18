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
- [ADR-0016: Make the Rust CLI the distribution entrypoint](0016-make-the-rust-cli-the-distribution-entrypoint.md)
- [ADR-0017: Keep runtime evidence and artifact lineage learner-neutral](0017-runtime-evidence-and-artifact-lineage.md)
- [ADR-0018: Add offline interactive run reports](0018-add-offline-run-reports.md)
- [ADR-0019: Add an optional DeepSeek Harness provider boundary](0019-add-optional-deepseek-harness-provider.md)
- [ADR-0020: Add rollout attempts and local queue barriers](0020-rollout-attempts-and-queue-barriers.md)
- [ADR-0021: Resolve portable projects from one manifest](0021-portable-project-manifests.md)
- [ADR-0023: Add a project-local VX task runner](0023-add-project-local-vx-task-runner.md)
- [ADR-0024: Bind invocation-scoped run contexts](0024-bind-invocation-run-contexts.md)

- [ADR-0025: Record learner-owned dynamic decisions](0025-record-learner-owned-decisions.md)
- [ADR-0026: Add a plugin-first extension system](0026-plugin-first-extension-system.md)
- [ADR-0027: Ship offline source-only project packages](0027-offline-source-packages.md)

- [ADR-0028: Embed a durable training dashboard](0028-embedded-training-dashboard.md)
- [ADR-0029: Bound the runtime start readiness window in the project contract](0029-bound-runtime-start-readiness.md)
- [ADR-0030: Give each workbench server an instance identity](0030-give-workbench-servers-an-instance-identity.md)
- [ADR-0031: Bind every training role to one trial identity](0031-bind-roles-to-one-trial-identity.md)
- [ADR-0032: Host an externally driven loop inside a run](0032-host-an-externally-driven-loop.md)

- [ADR-0033: Package trained-stage installers](0033-package-trained-stage-installers.md)

- [ADR-0034: Bind a default goal per project](0034-bind-a-default-goal-per-project.md)
- [ADR-0035: Own repository automation by capability domain and register it](0035-modular-tools-and-registry.md)
- [ADR-0036: Add a policy-only supervision watchdog with a scheduler exit-code contract](0036-supervision-watchdog-and-scheduler-contract.md)
- [ADR-0037: Add a read-only anti-fork drift gate](0037-anti-fork-drift-gate.md)

Accepted ADRs describe implemented architecture. Proposed future designs belong
in `docs/planning` until accepted and built.
