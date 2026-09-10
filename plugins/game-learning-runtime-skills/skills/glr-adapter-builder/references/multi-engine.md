# Multi-engine development boundaries

Use `--engine unity --unity-runtime mono` or `--unity-runtime il2cpp` to
record the Unity scripting backend explicitly. `--engine unreal` and
`--engine godot` select the other engine families. All four accept
`--access source` and `--access external`. The generated Python environment
is synthetic. `runtime-selection.json` records this status explicitly.

The Mono BepInEx and Unreal UE4SS loader templates remain unconnected
bootstraps. IL2CPP loader generation is rejected: the Mono assembly template
cannot be used as an IL2CPP bootstrap. No Godot native plugin is generated.
For IL2CPP source access, integrate reviewed semantic code at build time;
for Godot source access, implement the same environment contract in a
project-owned node or extension. Neither path currently has live acceptance.

`game_learning_runtime.engine_detection.inspect_installation(Path(...))`
reads only shallow markers within an explicitly selected directory. It can
distinguish common Windows Unity Mono/IL2CPP layouts, Unreal project markers
and Godot project/PCK markers. Unknown or conflicting layouts stay unknown.
It does not scan processes, load assemblies, select a loader or grant actions.
Embedded exports and non-Windows layouts may need explicit configuration.

The engine adapter owns observations, actions, reward, termination, reset and
game-state projection through `GameEnvironment`. Runtime transport uses the
existing `BridgeDriver`; `RuntimeIntegrationProfile.connect` checks its
declared capabilities. `EnvironmentBridgeDriver` provides episode/step fencing.
An external runtime starts with attach, uses real time, and cannot claim
physical reset, native semantic observations or exact frame stepping.

`game_learning_runtime.external_runtime.InputCaptureSession` composes a
reviewed `InputCaptureBackend` with an immutable command allowlist. Observe
returns RGB pixels and a fresh sequence; act requires that sequence, bounds
input hold time, releases owned inputs and returns a newer capture. Any
provider failure closes the session, preventing a blind mutating retry.
The backend must enforce its own deadlines and input watchdog and validate
exact target identity. The session does not manufacture game rewards or
infer action success from changed pixels. It supplies no OS implementation.

Do not report these Python tests as engine/game acceptance. A concrete
provider still needs target-bound startup, stale-request tests, disconnect
cleanup and a bounded real trace for every capability it declares. DCC-MCP
UI work must use the project dcc-cua/ui-control route and report its version,
PID and HWND before observation/input.
