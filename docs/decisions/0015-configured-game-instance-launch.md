# ADR-0015: Launch configured game instances before training

## Status

Accepted

## Context

Projects currently provide a runtime bridge and a trainer, but operators still
have to start the authorized game manually. That breaks unattended training and
makes multi-instance throughput dependent on ad-hoc scripts. GLR must not
discover processes, inject into games, or turn a configuration file into an
unbounded shell command.

## Decision

Add a separate `glr.game-launch.v1` contract and a small process supervisor.
The contract contains a fixed argv, project-relative working directory and
environment, a bounded instance count, a batch concurrency limit, and an
explicit process-alive or per-instance readiness-file signal. `TrainingLauncher`
starts the configured batches, writes a stable manifest of PIDs and instance
directories, exports the manifest to the trainer, and owns shutdown of every
process it started.

The readiness signal is a startup gate only. The game adapter remains the
authority for protocol version, exact target binding, semantic observations,
actions, and postcondition verification. The launcher does not infer that a
live process is a connected runtime.

## Consequences

- A project can start one game or a bounded set of parallel instances with one
  command.
- Each instance has an isolated directory and logs, and receives a stable ID
  through environment variables.
- Shell expansion and arbitrary command interpolation remain unavailable.
- Projects must publish a readiness file when process-alive is insufficient;
  bridge handshake failures still stop training at the adapter boundary.
- The launcher is local and single-host; distributed actors remain outside this
  contract.
