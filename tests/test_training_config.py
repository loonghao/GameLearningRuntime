from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import pytest

from game_learning_runtime import (
    KnowledgeAuthority,
    RewardComposer,
    RewardSignal,
    TrainingConfig,
    load_training_config,
)
from game_learning_runtime.errors import ContractViolation


def _config_mapping() -> dict[str, object]:
    return {
        "schema_version": "glr.training.v1",
        "lifecycle": {"start_mode": "attach", "stop_on_done": True},
        "bridge": {
            "required_capabilities": [
                "authenticated",
                "target-bound",
                "postcondition-verified",
            ]
        },
        "knowledge_sources": [
            {
                "id": "runtime",
                "authority": "authoritative",
                "required": True,
                "max_age_seconds": 0,
                "max_payload_bytes": 65536,
            },
            {
                "id": "strategy-prior",
                "authority": "advisory",
                "required": False,
                "max_age_seconds": 5,
                "max_payload_bytes": 4096,
            },
        ],
        "reward": {
            "minimum": -2,
            "maximum": 2,
            "terms": [
                {
                    "name": "progress",
                    "source": "runtime",
                    "weight": 1,
                    "minimum": -1,
                    "maximum": 1,
                    "required": True,
                },
                {
                    "name": "strategy-hint",
                    "source": "strategy-prior",
                    "minimum_authority": "advisory",
                    "weight": 0.1,
                    "minimum": -1,
                    "maximum": 1,
                    "required": False,
                },
            ],
        },
    }


def test_training_config_loads_versioned_immutable_contract() -> None:
    source = _config_mapping()

    config = TrainingConfig.from_mapping(source)

    assert config.schema_version == "glr.training.v1"
    assert config.lifecycle.start_mode == "attach"
    assert config.lifecycle.stop_on_done
    assert config.bridge.required_capabilities == frozenset(
        {"authenticated", "target-bound", "postcondition-verified"}
    )
    assert config.knowledge_sources[0].authority is KnowledgeAuthority.AUTHORITATIVE
    assert config.knowledge_sources[1].authority is KnowledgeAuthority.ADVISORY
    assert config.reward.terms[0].minimum_authority is KnowledgeAuthority.AUTHORITATIVE
    assert isinstance(config.knowledge_by_id, MappingProxyType)

    knowledge = source["knowledge_sources"]
    assert isinstance(knowledge, list)
    runtime = knowledge[0]
    assert isinstance(runtime, dict)
    runtime["id"] = "changed"
    assert config.knowledge_sources[0].source_id == "runtime"


def test_reward_composer_clips_terms_and_total_without_executing_expressions() -> None:
    config = TrainingConfig.from_mapping(_config_mapping())
    composer = RewardComposer(config)

    result = composer.compose(
        [
            RewardSignal(name="progress", source="runtime", value=2),
            RewardSignal(name="strategy-hint", source="strategy-prior", value=0.5),
        ]
    )

    assert result.total == pytest.approx(1.05)
    assert result.contributions == {"progress": 1.0, "strategy-hint": 0.05}
    assert isinstance(result.contributions, MappingProxyType)


def test_reward_composer_requires_configured_authoritative_signal() -> None:
    config = TrainingConfig.from_mapping(_config_mapping())
    composer = RewardComposer(config)

    with pytest.raises(ContractViolation, match=r"missing required reward signals.*progress"):
        composer.compose([RewardSignal(name="strategy-hint", source="strategy-prior", value=0.5)])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: config["knowledge_sources"][0].update(  # type: ignore[index,union-attr]
                {"authority": "advisory"}
            ),
            "requires authoritative",
        ),
        (
            lambda config: config.update({"schema_version": "glr.training.v2"}),
            "schema_version",
        ),
        (
            lambda config: config["reward"].update({"unexpected": True}),  # type: ignore[union-attr]
            "unexpected fields",
        ),
    ],
)
def test_training_config_rejects_unsafe_or_ambiguous_contracts(
    mutate: object, message: str
) -> None:
    config = _config_mapping()
    assert callable(mutate)
    mutate(config)

    with pytest.raises((ContractViolation, ValueError), match=message):
        TrainingConfig.from_mapping(config)


def test_reward_composer_rejects_unknown_mismatched_and_nonfinite_signals() -> None:
    composer = RewardComposer(TrainingConfig.from_mapping(_config_mapping()))

    with pytest.raises(ContractViolation, match="unknown reward signal"):
        composer.compose([RewardSignal(name="unknown", source="runtime", value=1)])
    with pytest.raises(ContractViolation, match="expected source runtime"):
        composer.compose([RewardSignal(name="progress", source="strategy-prior", value=1)])
    with pytest.raises(ValueError, match="finite"):
        RewardSignal(name="progress", source="runtime", value=float("nan"))


def test_load_training_config_reads_json_without_runtime_yaml_dependency(tmp_path: Path) -> None:
    path = tmp_path / "training.json"
    path.write_text(json.dumps(_config_mapping()), encoding="utf-8")

    config = load_training_config(path)

    assert config.lifecycle.start_mode == "attach"
    assert config.reward.terms[0].name == "progress"


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("lifecycle", "start_mode"), "resume", "start_mode"),
        (("lifecycle", "stop_on_done"), 1, "boolean"),
        (("bridge", "required_capabilities"), ["reset", "reset"], "duplicates"),
        (("knowledge_sources", 0, "id"), "local/path", "public identifier|must match"),
        (("knowledge_sources", 0, "max_age_seconds"), -1, "cannot be negative"),
        (("knowledge_sources", 0, "max_payload_bytes"), 0, "positive integer"),
        (("reward", "minimum"), 3, "minimum cannot exceed"),
        (("reward", "terms", 0, "weight"), float("inf"), "finite"),
    ],
)
def test_training_config_rejects_malformed_boundary_values(
    path: tuple[object, ...], replacement: object, message: str
) -> None:
    config = deepcopy(_config_mapping())
    target: object = config
    for segment in path[:-1]:
        target = target[segment]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]

    with pytest.raises((TypeError, ValueError), match=message):
        TrainingConfig.from_mapping(config)


def test_training_config_rejects_duplicate_and_unknown_references() -> None:
    duplicate_source = _config_mapping()
    sources = duplicate_source["knowledge_sources"]
    assert isinstance(sources, list)
    sources.append(deepcopy(sources[0]))
    with pytest.raises(ValueError, match="duplicate ids"):
        TrainingConfig.from_mapping(duplicate_source)

    duplicate_term = _config_mapping()
    reward = duplicate_term["reward"]
    assert isinstance(reward, dict)
    terms = reward["terms"]
    assert isinstance(terms, list)
    terms.append(deepcopy(terms[0]))
    with pytest.raises(ValueError, match="duplicate names"):
        TrainingConfig.from_mapping(duplicate_term)

    unknown_source = _config_mapping()
    reward = unknown_source["reward"]
    assert isinstance(reward, dict)
    terms = reward["terms"]
    assert isinstance(terms, list)
    term = terms[0]
    assert isinstance(term, dict)
    term["source"] = "missing"
    with pytest.raises(ContractViolation, match="unknown source"):
        TrainingConfig.from_mapping(unknown_source)


def test_reward_composer_rejects_duplicates_and_clips_total() -> None:
    config_mapping = _config_mapping()
    reward = config_mapping["reward"]
    assert isinstance(reward, dict)
    reward["maximum"] = 0.5
    composer = RewardComposer(TrainingConfig.from_mapping(config_mapping))
    signal = RewardSignal(name="progress", source="runtime", value=1)

    with pytest.raises(ContractViolation, match="duplicate reward signal"):
        composer.compose([signal, signal])
    with pytest.raises(TypeError, match="RewardSignal"):
        composer.compose([object()])  # type: ignore[list-item]

    result = composer.compose([signal])
    assert result.total == 0.5
    assert result.contributions == {"progress": 1.0}


def test_load_training_config_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "training.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="must be an object"):
        load_training_config(path)
