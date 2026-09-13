# Dynamic command providers

Register commands in the GLR Python registry and keep engine code limited to a
reviewed host implementation. Unity, Unreal, and Godot hosts expose the same
typed dispatch and receipt contract. A registry reload increments its
generation atomically; stale requests are rejected and retried after
renegotiation.

Use the project-owned `dcc-cua` provider only for a bounded foreground
keyboard/mouse fallback when a semantic engine command is unavailable. The
fallback must declare the target identity and prove the post-action state. It
must never be used to hide a missing semantic capability or to bypass the
engine host.
