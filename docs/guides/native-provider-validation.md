# Native sample validation

These are small source-integrated samples, not arbitrary game integrations.
Only the explicitly created Transform, Actor, component, Node3D or allowlisted
sample level is controlled. They do not advertise physics frame stepping,
authentication or generic game reset. Source-level Unity code is AOT compatible;
this does not make a Mono BepInEx bootstrap an IL2CPP loader.

## Reproduce

Use new/empty output directories. Run from the repository environment:

```text
python scripts/check_unity_provider.py --editor <Unity.exe> --output <new-project> --backend mono
python scripts/check_unity_provider.py --editor <Unity.exe> --output <new-project> --backend il2cpp
python scripts/check_unreal_provider.py --editor <UnrealEditor-Cmd.exe> --output <new-project>
python scripts/check_godot_provider.py --godot <godot-executable>
```

Unity needs a valid existing editor license, its matching Windows Player
modules, and the normal C++ build toolchain for IL2CPP. Modules can be installed
using the official Hub headless `install-modules` command. Unreal requires a
complete installation, including Shaders, Content and the Python/Editor
Scripting plugins. Consult the Launcher installation manifest instead of
assuming the default installation directory.

## Observed results on 2026-09-10

- Unity 2022.3.62f3c1: Editor and standalone Mono Player passed three moves,
  Transform readback, stale request rejection and reset. Standalone IL2CPP
  Player passed the same checks after official `windows-il2cpp` installation.
- Unreal 5.8.1: the complete installation passed three Actor moves, one
  SceneComponent move, and allowlisted A/B level transitions, readback, reset
  and stale rejection. This is Editor commandlet validation, not a packaged game.
- Godot 4.6.3: a real Node3D sample passed through `HostBridgeDriver` over
  stdio. The official Windows archive SHA-256 was
  `e39986a178d585ce7ac198fb8de6ea436366dc0cc00e594810c2e3e104c04b90`.
- DCC-CUA 1.8.1: real Unity window pixels were decoded into RGB and a click
  changed the sample counter from 0 to 1. A later capture returned a fresh
  sequence. A repeatable single-session input/post-capture test is not yet
  accepted: the desktop subsequently refused `OpenInputDesktop` with
  `0x80070005`, and the D3D11 device probe failed. The backend rejects an
  unavailable visual/input desktop and never switches to another GUI provider.

## External input sample

Install the optional `cua` dependency. Launch the generated Unity Player with
`--glr-external` using a DCC-CUA `launch_app` Host request carrying a scoped
`allow_app_launch` grant. Bind the returned exact PID/HWND and visibly attest
`provider=dcc-cua runtime=1.8.1 pid=... hwnd=...` before observation/input.
Take a fresh exact-window image and choose the Advance button coordinates.

```text
python scripts/check_cua_provider.py --cli <dcc-cua.exe> --pid <pid> --hwnd <hwnd> --x <x> --y <y> --width <image-width> --height <image-height> --state-file <player-directory>/glr-external-state.json
```

The checker rejects changed image geometry and requires both a fresh frame
and a counter increment; input acknowledgement alone is insufficient. An
explicit `--activate-before-capture` is available when foreground capture is
authorized. Foreground refusal, capture failure, target change and uncertain
responses fail closed without replaying the click. Only complete clicks are
supported; continuous keyboard/gamepad holds are not implemented.
