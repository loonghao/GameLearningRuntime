"""Run only in an isolated Unreal Python commandlet project."""

import json
from pathlib import Path

import unreal
from actor_provider import ActorProvider
from component_provider import ComponentProvider
from level_provider import LevelProvider

actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actor = actors.spawn_actor_from_class(
    unreal.StaticMeshActor, unreal.Vector(0, 0, 0), transient=True
)
if actor is None:
    raise RuntimeError("No commandlet world is available")
try:
    provider = ActorProvider(actor, unreal.Vector)
    initial = provider.reset()
    for step in range(1, 4):
        result = provider.step(initial["episode_id"], step, 1)
        assert actor.get_actor_location().x == step
        try:
            provider.step(initial["episode_id"], step, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Stale request accepted")
    provider.reset()
    assert actor.get_actor_location().x == 0
    provider.close()
    component = ComponentProvider(actor.static_mesh_component, unreal.Vector)
    start = component.reset()
    component.step(start["episode_id"], 1, 1)
    assert actor.static_mesh_component.get_world_location().x == 1
    component.reset()
    assert actor.static_mesh_component.get_world_location().x == 0
    component.close()
    Path(unreal.Paths.project_saved_dir(), "glr-smoke.json").write_text(
        json.dumps(
            {
                "engine": "unreal-editor",
                "steps": 3,
                "reset": True,
                "stale_rejected": True,
            }
        )
    )
finally:
    actors.destroy_actor(actor)

levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
worlds = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
paths = ("/Game/GLRSmoke/A", "/Game/GLRSmoke/B")
for path in paths:
    assert levels.new_level(path)
level_provider = LevelProvider(levels, worlds, paths)
initial = level_provider.reset()
assert initial["level_index"] == 0
assert level_provider.step(initial["episode_id"], 1, 1)["level_index"] == 1
try:
    level_provider.step(initial["episode_id"], 1, 0)
except ValueError:
    pass
else:
    raise AssertionError("Stale level transition accepted")
assert level_provider.reset()["level_index"] == 0
level_provider.close()
Path(unreal.Paths.project_saved_dir(), "glr-smoke.json").write_text(
    json.dumps(
        {
            "engine": "unreal-editor",
            "actor_steps": 3,
            "component_steps": 1,
            "level_transitions": 2,
            "reset": True,
            "stale_rejected": True,
        }
    )
)
