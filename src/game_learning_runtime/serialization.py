"""Portable JSONL transition records for BC, replay, and offline learning."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from game_learning_runtime.contracts import (
    ActionOutcome,
    ActionReceipt,
    Event,
    TensorTree,
    Transition,
)

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
        "action_receipt": _action_receipt_to_record(transition.action_receipt),
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
        "provenance": dict(transition.provenance) if transition.provenance is not None else None,
    }
    json.dumps(record, allow_nan=False)
    return record


def _action_receipt_to_record(receipt: ActionReceipt | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    return {
        "action_id": receipt.action_id,
        "episode_id": str(receipt.episode_id),
        "step_id": receipt.step_id,
        "outcome": receipt.outcome.value,
        "issued_timestamp_ns": receipt.issued_timestamp_ns,
        "observed_timestamp_ns": receipt.observed_timestamp_ns,
        "postcondition": receipt.postcondition,
        "progress_delta": receipt.progress_delta,
        "authoritative_observation_sequence": receipt.authoritative_observation_sequence,
        "retryable": receipt.retryable,
    }


def _action_receipt_from_record(value: object) -> ActionReceipt | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("action_receipt must be an object or null")
    return ActionReceipt(
        action_id=str(value["action_id"]),
        episode_id=UUID(str(value["episode_id"])),
        step_id=int(value["step_id"]),
        outcome=ActionOutcome(str(value["outcome"])),
        issued_timestamp_ns=int(value["issued_timestamp_ns"]),
        observed_timestamp_ns=int(value["observed_timestamp_ns"]),
        postcondition=str(value.get("postcondition", "unknown")),
        progress_delta=value.get("progress_delta"),
        authoritative_observation_sequence=(
            int(value["authoritative_observation_sequence"])
            if value.get("authoritative_observation_sequence") is not None
            else None
        ),
        retryable=bool(value.get("retryable", False)),
    )


def transition_from_record(record: Mapping[str, Any]) -> Transition:
    """Parse and validate one ``glr.transition.v1`` JSON value."""

    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"unsupported record schema: {record.get('schema')!r}")
    observation = _decode_tree(record["observation"])
    action = _decode_tree(record["action"])
    next_observation = _decode_tree(record["next_observation"])
    if observation is None or action is None or next_observation is None:
        raise ValueError("observation, action, and next_observation are required")
    raw_provenance = record.get("provenance")
    if raw_provenance is not None and not isinstance(raw_provenance, Mapping):
        raise TypeError("provenance must be an object")
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
        action_receipt=_action_receipt_from_record(record.get("action_receipt")),
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
        provenance=raw_provenance,
    )


class JsonlTransitionWriter:
    """Append-only writer for replayable transition datasets."""

    def __init__(self, path: str | Path, *, flush_every: int = 1) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self._path = Path(path)
        self._flush_every = flush_every
        self._pending = 0
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
        self._pending += 1
        if self._pending >= self._flush_every:
            self._stream.flush()
            self._pending = 0

    def write_many(self, transitions: Iterable[Transition]) -> int:
        """Append a batch and flush once after the batch is complete."""
        if self._stream is None:
            raise RuntimeError("writer must be used as a context manager")
        count = 0
        for transition in transitions:
            json.dump(transition_to_record(transition), self._stream, separators=(",", ":"))
            self._stream.write("\n")
            count += 1
        if count:
            self._stream.flush()
            self._pending = 0
        return count

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None


def read_jsonl_transitions(path: str | Path, *, strict: bool = True) -> Iterator[Transition]:
    """Stream transitions from a GLR JSONL dataset."""

    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield transition_from_record(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                if not strict:
                    # A final unterminated line is the expected result of a killed writer.
                    if not line.endswith("\n"):
                        return
                    continue
                raise ValueError(f"invalid transition record at line {line_number}") from error
