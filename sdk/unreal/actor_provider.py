"""Unreal Editor source provider for one explicitly supplied, owned Actor.

This adapter uses the GLR host tensor representation without importing a
learner or NumPy into Unreal's embedded Python. Transport is owned separately.
It does not provide packaged-game support or physics frame stepping.
"""

from __future__ import annotations

import base64
import math
import struct
import threading
import time
import uuid


def tensor(dtype, fmt, value):
    return {
        "shape": [1],
        "dtype": dtype,
        "data": base64.b64encode(struct.pack(fmt, value)).decode("ascii"),
    }


class ActorProvider:
    def __init__(self, actor, vector_type):
        self.actor = actor
        self.vector_type = vector_type
        self.thread = threading.get_ident()
        self.episode = None
        self.cursor = 0
        self.closed = False

    def _check(self):
        if self.closed or threading.get_ident() != self.thread:
            raise RuntimeError("Provider closed or called outside its engine thread")

    def describe(self):
        self._check()

        def spec(path, dtype, kind):
            return {"path": path, "shape": [1], "dtype": dtype, "kind": kind}

        return {
            "environment_id": "glr.unreal.actor-v1",
            "protocol_version": "1.0",
            "observations": [spec("position", "float32", "continuous")],
            "actions": [spec("move", "int32", "discrete") | {"minimum": -1, "maximum": 1}],
            "action_masks": [],
            "reward": spec("reward", "float32", "continuous"),
            "done": spec("done", "bool", "binary"),
            "capabilities": [
                "reset",
                "live-attach",
                "step",
                "native-action",
                "semantic-observation",
            ],
            "metadata": {},
        }

    def _state(self):
        position = float(self.actor.get_actor_location().x)
        if not math.isfinite(position):
            raise RuntimeError("Invalid actor position")
        done = abs(position) >= 3
        return {
            "episode_id": self.episode,
            "step_id": self.cursor,
            "timestamp_ns": time.time_ns(),
            "observation": {"position": tensor("float32", "<f", position)},
            "reward": tensor("float32", "<f", float(done)),
            "terminated": tensor("bool", "?", done),
            "truncated": tensor("bool", "?", False),
            "action_mask": {},
            "events": [],
            "info": {},
        }

    def reset(self):
        self._check()
        self.actor.set_actor_location(self.vector_type(0, 0, 0), False, True)
        if self.actor.get_actor_location().x != 0:
            raise RuntimeError("Actor reset postcondition failed")
        self.episode, self.cursor = str(uuid.uuid4()), 0
        return self._state()

    def attach(self):
        self._check()
        if abs(self.actor.get_actor_location().x) >= 3:
            raise RuntimeError("Reset the terminal sample first")
        self.episode, self.cursor = str(uuid.uuid4()), 0
        return self._state()

    def step(self, episode_id, expected_step_id, move):
        self._check()
        if (
            self.episode is None
            or episode_id != self.episode
            or expected_step_id != self.cursor + 1
        ):
            raise ValueError("Stale episode or step")
        if type(move) is not int or move not in {-1, 0, 1}:
            raise ValueError("Move must be -1, 0 or 1")
        position = self.actor.get_actor_location()
        if abs(position.x) >= 3:
            raise ValueError("Episode is done")
        expected = position.x + move
        try:
            self.actor.set_actor_location(
                self.vector_type(expected, position.y, position.z), False, True
            )
            if self.actor.get_actor_location().x != expected:
                raise RuntimeError("Actor action postcondition failed")
            self.cursor += 1
            return self._state()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
