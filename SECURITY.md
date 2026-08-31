# Security Policy

## Supported versions

The latest minor release receives security fixes. This project is currently
pre-1.0, so public contracts may evolve with documented migration notes.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository. Do not open a
public issue containing an exploit, token, private game artifact, process dump,
or proprietary runtime detail.

Include the affected version, impact, a minimal authorized reproduction, and
suggested remediation when available. Maintainers will acknowledge a complete
report within seven days.

GLR does not accept features intended to bypass anti-cheat, hide unauthorized
instrumentation, or access systems without permission.

`glr-hostd` accepts only its compiled provider allowlist and bounded stdio
envelopes. Reports involving arbitrary provider loading, process discovery,
frame-bound bypass, stale action execution, or child-process cleanup should be
treated as security issues. Do not include a real game's memory, binaries,
account data, or private runtime trace in a report.
