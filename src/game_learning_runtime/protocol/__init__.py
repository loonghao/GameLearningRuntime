"""Access to the packaged GLR protocol definition."""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path


@contextmanager
def protocol_path() -> Iterator[Path]:
    """Yield a filesystem path to the packaged ``runtime.proto`` schema."""

    resource = files("game_learning_runtime.protocol").joinpath("glr/v1/runtime.proto")
    with as_file(resource) as path:
        yield path
