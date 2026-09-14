# GLR plugin system

GLR's plugin layer is a small, declarative control plane inspired by the
bundle/profile workflow in DeepSeek Harness (DSH):
https://deepseek.com/harness/en/. It gives a project a reviewable extension
inventory without making plugin installation an implicit code-execution step.

## Bundle contract

A bundle is a local directory containing glr-plugin.json and its payload:

~~~json
{
  "schema_version": "glr.plugin.v1",
  "id": "torchrl-learner",
  "version": "0.1.0",
  "kind": "learner",
  "name": "TorchRL learner",
  "description": "A project-owned TorchRL learner adapter.",
  "entrypoint": "torchrl_plugin:create",
  "capabilities": ["learner.ppo", "collector.process"],
  "requires": {"glr": ">=0.17.0,<1.0.0", "torchrl": ">=0.13.0,<0.14.0"},
  "platforms": ["windows", "linux", "macos"],
  "isolation": "process",
  "permissions": ["read:environment", "write:checkpoint"]
}
~~~

The id and version determine the immutable project-local storage location
.glr/plugins/<id>/<version>. The entrypoint is metadata only. It is not
imported by inspect, install, health, or profile resolution.

## Profiles

Profiles are saved as .glr/profiles/<name>.json:

~~~json
{
  "schema_version": "glr.profile.v1",
  "name": "training",
  "plugins": [
    {
      "id": "torchrl-learner",
      "version": ">=0.1.0,<1.0.0",
      "enabled": true,
      "permissions": ["read:environment"],
      "config": {"batch_size": 256}
    }
  ]
}
~~~

Profile configuration is canonicalized and included in the profile digest.
Finite JSON numbers, booleans, strings, arrays, and objects are supported;
non-finite values are rejected. Rust enables correctly-rounded float parsing so
Python and Rust preserve the same digest bytes at decimal boundaries.
Text field limits are measured in UTF-8 bytes, so the Python SDK and Rust CLI
apply the same boundary for non-ASCII metadata.

The profile grant must be a subset of the manifest declaration. Dependencies
are selected by highest satisfying semantic version, then emitted in a stable
dependency-first order. Cycles, conflicting requirements, unsupported
platforms, incompatible GLR versions, and missing bundles fail closed. The Rust
CLI checks `requires.glr` against its compiled version; pass `glr_version` to
the Python `PluginManager` when the SDK is enforcing a specific runtime, or
leave it unset for source-side inventory that has no runtime binding. If a
dependency is also listed explicitly, its explicit profile grants and config
are retained rather than silently discarded; incompatible repeated requests
fail closed.
Other `requires` keys (such as `torchrl` or `sample-factory`) are preserved
declarations for the host runner and are not checked for installation by this
control plane. Consequently, `health` reports static bundle/profile readiness,
not proof that every external framework is available.

## CLI workflow

The standalone Rust CLI is the canonical entrypoint; the Python SDK exposes the
same command names for development environments:

~~~powershell
# No role executable or Python import is needed for these control-plane calls.
glr --project . --json plugin inspect --source plugins/torchrl-learner
glr --project . --json plugin install --source plugins/torchrl-learner --sha256 <digest>
glr --project . --json plugin list
glr --project . --json plugin profile enable training torchrl-learner `
  --version ">=0.1.0,<1.0.0" --grant read:environment
glr --project . --json plugin profile resolve training
glr --project . --json plugin health --profile training
glr --project . --json plugin profile disable training torchrl-learner
glr --project . --json plugin remove torchrl-learner --version 0.1.0
~~~

The plugin inspect command can run against a directory that is not yet a GLR
project. The other commands use the --project path only as the owner of the
.glr/plugins and .glr/profiles stores; they do not boot runtime roles. JSON
output uses the stable glr.cli-output.v1 envelope.

## Trust and execution boundary

The first implementation accepts local directories only. It validates strict
JSON, ASCII-only portable relative paths, file sizes, symlink/reparse-point policy, and
SHA-256 inventories, then copies atomically and verifies the staged result.
There is no Git/npm/HTTP fetch, signature verification, import hook, shell
command, build hook, or automatic process launch. A future registry or host
runner must add explicit authorization and signed provenance on top of this
contract; it must not silently execute an entrypoint during installation.

## TorchRL and Sample Factory

The plugin slot is intentionally framework-neutral. Use a TorchRL learner
plugin first when you want the existing optional TorchRL 0.13 integration and
its CI contract. Add a Sample Factory plugin as an independent backend when
its throughput-oriented execution model is useful (especially on Linux). Both
consume the same GameEnvironment, transition, and checkpoint contracts; a
framework plugin must not move game semantics into the learner layer.

The repository currently ships the contract and lifecycle, not bundled
TorchRL/Sample Factory plugin payloads. Keep each project-owned adapter in a
reviewed bundle and pin its digest in deployment documentation or a model
bundle.

## Python API

~~~python
from game_learning_runtime import PluginManager, PluginProfile, PluginRef

manager = PluginManager(".")
inspection = manager.inspect("plugins/torchrl-learner")
manager.install("plugins/torchrl-learner", expected_sha256=inspection.content_sha256)
manager.save_profile(
    PluginProfile(
        name="training",
        plugins=(PluginRef("torchrl-learner", permissions=("read:environment",)),),
    )
)
resolved = manager.resolve_profile("training")
~~~

The returned objects are immutable data records. A host-specific runner owns
the later decision to import or start a plugin and must apply its own trust and
permission policy first.
