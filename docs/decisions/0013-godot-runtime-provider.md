# 0013: Godot runtime provider

Godot adapters use the same GLR provider contract as Unity and Unreal. A
Godot GDScript or native extension is a thin main-thread host; it publishes
observations and executes only commands accepted by the GLR dynamic registry.
It does not contain a learner or a second protocol.

The provider declares `engine=godot`, its engine version, adapter build hash,
and whether the session is `live-attach` or `reset`. Commands are registered
in the GLR Python control plane and may be hot-reloaded. The Godot host only
needs to implement the stable primitives and return bounded post-action
receipts, so adding gameplay commands does not require rebuilding or
restarting the game.

Front-end keyboard and mouse input is an explicit fallback capability. When a
Godot semantic command is unavailable, the adapter may route a bounded action
through the project-owned `dcc-cua` provider, subject to its target-window
identity, foreground/readiness checks, input lease, and post-action
observation. Generic computer-use fallback is not part of this provider.
