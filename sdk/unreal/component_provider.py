"""Bind the actor sample contract to one explicitly selected SceneComponent."""

from actor_provider import ActorProvider


class _ComponentTarget:
    def __init__(self, component):
        self.component = component

    def get_actor_location(self):
        return self.component.get_world_location()

    def set_actor_location(self, value, sweep, teleport):
        return self.component.set_world_location(value, sweep, teleport)


class ComponentProvider(ActorProvider):
    def __init__(self, component, vector_type):
        super().__init__(_ComponentTarget(component), vector_type)

    def describe(self):
        value = super().describe()
        value["environment_id"] = "glr.unreal.component-v1"
        return value
