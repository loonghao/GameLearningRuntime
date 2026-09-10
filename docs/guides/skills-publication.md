# Versioned ClawHub skills

Canonical skill sources live in `.agents/skills`; the checked-in agent plugin
is synchronized with `python scripts/package_agent_plugin.py --sync` and
validated with `--check`. Supporting templates use a Markdown extension so
ClawHub includes them in the uploaded bundle.

`python scripts/publish_skills.py` uses the pinned ClawHub CLI 0.23.1 and the
project version in `pyproject.toml`. It previews all three skills without
publishing, checks selected file counts, and writes portable per-file SHA-256
digests plus registry fingerprints to `dist/clawhub-receipts.json`. Node.js
with npx and Python 3.11+ are required (Python 3.10 additionally needs tomli).

The GitHub workflow validates PRs and main pushes. A matching release tag
automatically publishes that version. A manual run publishes only when
`publish=true` and the ref is main or a matching release tag. Concurrent
publication runs are serialized. Configure `CLAWHUB_TOKEN` in the GitHub
environment named `clawhub`, with publishing rights for owner `loonghao`.
Do not put tokens into skill files, workflow inputs or command-line arguments.

Publication performs a fresh preview, verifies the file set and receipt
version/slug/fingerprint, and retains checksums as a workflow artifact. A
pending or malformed receipt fails the job. Explicit versions avoid accidental
patch increments on unrelated main pushes. Existing-version conflicts fail
closed; inspect registry state before retrying an uncertain upload.

A confirmed upload does not prove public availability or security review
completion. Receipts explicitly set `public_visibility_verified=false`.
Inspect the exact owner/slug/version after publication before announcing a
public release. No live publication is performed by local dry runs.

Official references, checked 2026-09-10:

- [ClawHub CLI](https://docs.openclaw.ai/clawhub/cli)
- [ClawHub publishing](https://docs.openclaw.ai/clawhub/publishing)
