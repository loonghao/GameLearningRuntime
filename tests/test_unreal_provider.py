from types import SimpleNamespace

import pytest

from sdk.unreal.actor_provider import ActorProvider


class Actor:
    def __init__(self):
        self.position = vector(9, 0, 0)
        self.writes = 0

    def get_actor_location(self):
        return self.position

    def set_actor_location(self, value, sweep, teleport):
        self.position = value
        self.writes += 1


def vector(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def test_actor_provider_fences_before_native_mutation():
    actor = Actor()
    provider = ActorProvider(actor, vector)
    start = provider.reset()
    provider.step(start["episode_id"], 1, 1)
    assert actor.position.x == 1
    with pytest.raises(ValueError, match="Stale"):
        provider.step(start["episode_id"], 1, 1)
    assert actor.writes == 2
    with pytest.raises(ValueError, match="Move"):
        provider.step(start["episode_id"], 2, 5)
    provider.close()
    with pytest.raises(RuntimeError):
        provider.reset()
