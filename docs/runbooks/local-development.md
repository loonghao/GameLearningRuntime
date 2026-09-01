# Local development runbook

## Prerequisites

- Git
- [vx](https://github.com/loonghao/vx)

`vx.toml` and `vx.lock` pin the baseline Python, uv, just, and rustup versions.
`rust-toolchain.toml` pins Rust 1.98.0 and `global.json` pins .NET SDK 10.0.400.
The GitHub Actions jobs use the same just recipes as local development.

## Steps

```powershell
vx setup
vx just check
vx just build
```

For the optional integration:

```powershell
vx just ci-torchrl
vx just ci-gymnasium
```

Validate the cross-language Runtime Host and provider boundary directly:

```powershell
vx just rust-check
vx just provider-sdk-check
vx just host-smoke
```

Validate the standalone Rust control plane and its exact CLI contracts:

```powershell
vx cargo fmt --all --check
vx cargo clippy --workspace --all-targets --locked -- -D warnings
vx cargo test --workspace --locked
vx cargo run --package glr-cli -- --help
```

The CLI is the canonical deployment entrypoint; the Python package remains an
optional SDK. A release distribution is produced with `scripts/package_cli.py`
and must contain `glr`, `glr-hostd`, `glr-release.json`, `install.md`, `LICENSE`,
and both repository Skills.

Run the complete local pre-push surface with `vx just ci`. CI selects Python
3.10 through 3.13 through `vx just ci-core <version>` in isolated jobs.

## Verify

All commands must exit zero. Confirm imports originate from this checkout when
debugging environment drift:

```powershell
vx just origin
```

## Troubleshooting

| Problem | Resolution |
|---|---|
| Lock file changed unexpectedly | Run `vx uv lock --check`; inspect dependency inputs before accepting a relock |
| vx tool versions drifted | Run `vx lock --check`; update `vx.toml` and `vx.lock` together after review |
| Existing `.venv` is unrelated or broken | GLR uses the project-owned `.venv-glr` through just; do not delete another environment as a workaround |
| Torch wheel is large | Run the core suite without the `torchrl` extra; use the dedicated integration job for TorchRL |
| Contract violation | Compare the failing path, dtype, shape, and declared bounds; do not cast silently |
| Protocol test fails | Compile the packaged `runtime.proto`; do not test an unshipped duplicate |
| Runtime Host smoke times out | Run `vx cargo build --locked`, confirm the explicit host binary starts, and inspect only local stderr; never retry a mutating request |
| C++ provider check cannot find a compiler | Install a C++20 compiler; Windows uses the latest Visual Studio VC tools selected by `vswhere` |
