"""Portable JSONL transition records for BC, replay, and offline learning."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.contracts import Event, TensorTree, Transition

RECORD_SCHEMA = "glr.transition.v1"


def _encode_array(array: NDArray[Any]) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _decode_array(value: Mapping[str, Any]) -> NDArray[Any]:
    dtype = np.dtype(value["dtype"])
    shape = tuple(int(dimension) for dimension in value["shape"])
    raw = base64.b64decode(value["data"], validate=True)
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_size:
        raise ValueError(f"tensor payload has {len(raw)} bytes; expected {expected_size}")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _encode_tree(tree: TensorTree | None) -> dict[str, Any] | None:
    if tree is None:
        return None
    encoded: dict[str, Any] = {}
    for key, value in tree.items():
        encoded[key] = (
            {"tree": _encode_tree(value)}
            if isinstance(value, Mapping)
            else {"tensor": _encode_array(value)}
        )
    return encoded


def _decode_tree(value: Mapping[str, Any] | None) -> TensorTree | None:
    if value is None:
        return None
    decoded: dict[str, Any] = {}
    for key, item in value.items():
        if "tree" in item:
            decoded[key] = _decode_tree(item["tree"])
        elif "tensor" in item:
            decoded[key] = _decode_array(item["tensor"])
        else:
            raise ValueError(f"tree field {key!r} is missing its value kind")
    return decoded


def transition_to_record(transition: Transition) -> dict[str, Any]:
    """Convert a transition to a stable, language-neutral JSON value."""

    record = {
        "schema": RECORD_SCHEMA,
        "episode_id": str(transition.episode_id),
        "step_id": transition.step_id,
        "timestamp_ns": transition.timestamp_ns,
        "observation": _encode_tree(transition.observation),
        "action": _encode_tree(transition.action),
        "action_mask": _encode_tree(transition.action_mask),
        "reward": _encode_array(transition.reward),
        "next_observation": _encode_tree(transition.next_observation),
        "next_action_mask": _encode_tree(transition.next_action_mask),
        "terminated": _encode_array(transition.terminated),
        "truncated": _encode_array(transition.truncated),
        "events": [
            {
                "name": event.name,
                "timestamp_ns": event.timestamp_ns,
                "payload": dict(event.payload),
            }
            for event in transition.events
        ],
        "info": dict(transition.info),
    }
    json.dumps(record, allow_nan=False)
    return record


def transition_from_record(record: Mapping[str, Any]) -> Transition:
    """Parse and validate one ``glr.transition.v1`` JSON value."""

    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"unsupported record schema: {record.get('schema')!r}")
    observation = _decode_tree(record["observation"])
    action = _decode_tree(record["action"])
    next_observation = _decode_tree(record["next_observation"])
    if observation is None or action is None or next_observation is None:
        raise ValueError("observation, action, and next_observation are required")
    return Transition(
        episode_id=UUID(record["episode_id"]),
        step_id=int(record["step_id"]),
        timestamp_ns=int(record["timestamp_ns"]),
        observation=observation,
        action=action,
        action_mask=_decode_tree(record.get("action_mask")),
        reward=_decode_array(record["reward"]),
        next_observation=next_observation,
        next_action_mask=_decode_tree(record.get("next_action_mask")),
        terminated=_decode_array(record["terminated"]),
        truncated=_decode_array(record["truncated"]),
        events=tuple(
            Event(
                name=item["name"],
                timestamp_ns=int(item["timestamp_ns"]),
                payload=item.get("payload", {}),
            )
            for item in record.get("events", [])
        ),
        info=record.get("info", {}),
    )


class JsonlTransitionWriter:
    """Append-only writer for replayable transition datasets."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._stream: IO[str] | None = None

    def __enter__(self) -> JsonlTransitionWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self._path.open("a", encoding="utf-8", newline="\n")
        return self

    def write(self, transition: Transition) -> None:
        if self._stream is None:
            raise RuntimeError("writer must be used as a context manager")
        json.dump(transition_to_record(transition), self._stream, separators=(",", ":"))
        self._stream.write("\n")
        self._stream.flush()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def read_jsonl_transitions(path: str | Path) -> Iterator[Transition]:
    """Stream transitions from a GLR JSONL dataset."""

    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield transition_from_record(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid transition record at line {line_number}") from error
