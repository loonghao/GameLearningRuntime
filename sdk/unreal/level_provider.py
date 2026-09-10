"""Allowlisted, synchronous editor level transitions with authoritative readback."""

import threading
import uuid


class LevelProvider:
    def __init__(self, level_subsystem, world_subsystem, levels):
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("Levels must be nonempty and unique")
        if any(not path.startswith("/Game/") or ".." in path for path in levels):
            raise ValueError("Levels must be explicit project asset paths")
        self.levels = tuple(levels)
        self.loader = level_subsystem
        self.world = world_subsystem
        self.thread = threading.get_ident()
        self.episode = None
        self.cursor = 0
        self.closed = False

    def _check(self):
        if self.closed or threading.get_ident() != self.thread:
            raise RuntimeError("Provider closed or called outside its engine thread")

    def observe(self):
        self._check()
        path = self.world.get_editor_world().get_path_name().split(".", 1)[0]
        if path not in self.levels:
            raise RuntimeError("Active level is outside the bound level set")
        return {
            "episode_id": self.episode,
            "step_id": self.cursor,
            "level_index": self.levels.index(path),
        }

    def _load(self, index):
        try:
            if not self.loader.load_level(self.levels[index]):
                raise RuntimeError("Level load failed")
            if self.observe()["level_index"] != index:
                raise RuntimeError("Level postcondition failed")
        except BaseException:
            self.closed = True
            raise

    def reset(self):
        self._check()
        self._load(0)
        self.episode, self.cursor = str(uuid.uuid4()), 0
        return self.observe()

    def step(self, episode_id, expected_step_id, level_index):
        self._check()
        if (
            self.episode is None
            or episode_id != self.episode
            or expected_step_id != self.cursor + 1
        ):
            raise ValueError("Stale episode or step")
        if type(level_index) is not int or not 0 <= level_index < len(self.levels):
            raise ValueError("Level index outside allowlist")
        self._load(level_index)
        self.cursor += 1
        return self.observe()

    def close(self):
        self.closed = True
