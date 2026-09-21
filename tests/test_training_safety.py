from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from game_learning_runtime import (
    DemonstrationGate,
    DemonstrationOrigin,
    DemonstrationOutcome,
    DemonstrationPolicyConfig,
    DemonstrationProvenance,
    EpisodeRewardGuard,
    RewardSafetyConfig,
    RewardSignal,
    TrainingConfig,
    load_demonstration_policy_config,
    load_reward_safety_config,
)
from game_learning_runtime.errors import ContractViolation


def _training_config(*, include_item_score: bool = False) -> TrainingConfig:
    terms: list[dict[str, object]] = [
        {
            "name": "progress",
            "source": "runtime",
            "weight": 1,
            "minimum": -10,
            "maximum": 10,
            "required": True,
        },
        {
            "name": "outcome",
            "source": "runtime",
            "weight": 10,
            "minimum": -1,
            "maximum": 1,
            "required": False,
        },
    ]
    if include_item_score:
        terms.insert(
            1,
            {
                "name": "item_score",
                "source": "runtime",
                "weight": 1,
                "minimum": 0,
                "maximum": 5,
                "required": False,
            },
        )
    return TrainingConfig.from_mapping(
        {
            "schema_version": "glr.training.v1",
            "lifecycle": {"start_mode": "reset", "stop_on_done": True},
            "bridge": {"required_capabilities": []},
            "knowledge_sources": [
                {
                    "id": "runtime",
                    "authority": "authoritative",
                    "required": True,
                }
            ],
            "reward": {"minimum": -20, "maximum": 20, "terms": terms},
        }
    )


def _budgeted_safety_mapping(
    *, shaping_signals: list[str], unbudgeted_signals: list[str] | None = None
) -> dict[str, object]:
    mapping = _reward_safety_mapping()
    mapping["shaping_signals"] = shaping_signals
    mapping["max_positive_shaping_per_step"] = 10
    mapping["max_positive_shaping_per_episode"] = 25
    if unbudgeted_signals is not None:
        mapping["unbudgeted_signals"] = unbudgeted_signals
    return mapping


def _item_score_episode(steps: int = 6) -> tuple[list[RewardSignal], ...]:
    """Six identical steps: 5.0 of progress plus 5.0 of item score."""

    return tuple(
        [RewardSignal("progress", "runtime", 5), RewardSignal("item_score", "runtime", 5)]
        for _ in range(steps)
    )


def _reward_safety_mapping() -> dict[str, object]:
    return {
        "schema_version": "glr.reward-safety.v1",
        "outcome_signal": "outcome",
        "shaping_signals": ["progress"],
        "max_positive_shaping_per_step": 1,
        "max_positive_shaping_per_episode": 2,
        "failure_episode_maximum": 0,
        "require_terminal_outcome": True,
    }


def _demonstration_policy_mapping() -> dict[str, object]:
    return {
        "schema_version": "glr.demonstration-policy.v1",
        "allowed_origins": ["human", "scripted-expert"],
        "allowed_outcomes": ["success"],
        "origin_weights": {"human": 1.5, "scripted-expert": 1},
        "outcome_weights": {"success": 2},
        "reject_unknown": True,
    }


def test_reward_guard_caps_dense_positive_shaping_across_an_episode() -> None:
    guard = EpisodeRewardGuard(
        _training_config(), RewardSafetyConfig.from_mapping(_reward_safety_mapping())
    )

    first = guard.compose([RewardSignal("progress", "runtime", 8)])
    second = guard.compose([RewardSignal("progress", "runtime", 8)])
    exhausted = guard.compose([RewardSignal("progress", "runtime", 8)])
    terminal = guard.compose(
        [
            RewardSignal("progress", "runtime", 8),
            RewardSignal("outcome", "runtime", 1),
        ],
        terminal=True,
    )

    assert first.total == 1
    assert second.total == 1
    assert exhausted.total == 0
    assert exhausted.suppressed_positive_shaping == 8
    assert terminal.total == 10
    assert terminal.episode_total == 12
    assert terminal.positive_shaping_total == 2
    assert isinstance(terminal.contributions, MappingProxyType)


def test_reward_guard_makes_a_failed_episode_non_positive() -> None:
    guard = EpisodeRewardGuard(
        _training_config(), RewardSafetyConfig.from_mapping(_reward_safety_mapping())
    )
    guard.compose([RewardSignal("progress", "runtime", 10)])
    guard.compose([RewardSignal("progress", "runtime", 10)])

    terminal = guard.compose(
        [
            RewardSignal("progress", "runtime", 10),
            RewardSignal("outcome", "runtime", -0.1),
        ],
        terminal=True,
    )

    assert terminal.episode_total == 0
    assert terminal.contributions["outcome"] == -1
    assert terminal.contributions["guardrail.failure-correction"] == -1
    with pytest.raises(ContractViolation, match="reset"):
        guard.compose([RewardSignal("progress", "runtime", 0)])

    guard.reset()
    assert guard.compose([RewardSignal("progress", "runtime", 1)]).episode_total == 1


def test_reward_guard_requires_terminal_only_authoritative_outcome() -> None:
    guard = EpisodeRewardGuard(
        _training_config(), RewardSafetyConfig.from_mapping(_reward_safety_mapping())
    )

    with pytest.raises(ContractViolation, match="only be emitted on a terminal"):
        guard.compose(
            [
                RewardSignal("progress", "runtime", 0),
                RewardSignal("outcome", "runtime", 1),
            ]
        )
    with pytest.raises(ContractViolation, match="terminal outcome"):
        guard.compose([RewardSignal("progress", "runtime", 0)], terminal=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "glr.reward-safety.v2", "schema_version"),
        ("shaping_signals", ["progress", "progress"], "duplicates"),
        ("outcome_signal", "local/path", "must match"),
        ("max_positive_shaping_per_step", -1, "non-negative"),
        ("max_positive_shaping_per_episode", 0.5, "cannot exceed"),
        ("unbudgeted_signals", ["item-score", "item-score"], "duplicates"),
        ("unbudgeted_signals", ["progress"], "cannot also be shaping signals"),
        ("unbudgeted_signals", ["outcome"], "cannot also be an unbudgeted signal"),
        ("unbudgeted_signals", ["local/path"], "must match"),
        ("unbudgeted_signals", "item-score", "must be an array"),
    ],
)
def test_reward_safety_config_rejects_ambiguous_or_unsafe_values(
    field: str, value: object, message: str
) -> None:
    mapping = _reward_safety_mapping()
    mapping[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        RewardSafetyConfig.from_mapping(mapping)


def test_reward_guard_rejects_unknown_or_non_positive_outcome_contract() -> None:
    safety = RewardSafetyConfig.from_mapping(_reward_safety_mapping())
    mapping = _reward_safety_mapping()
    mapping["outcome_signal"] = "missing"

    with pytest.raises(ContractViolation, match="unknown outcome"):
        EpisodeRewardGuard(_training_config(), RewardSafetyConfig.from_mapping(mapping))

    training = _training_config()
    object.__setattr__(training.reward.terms[1], "weight", -1)
    with pytest.raises(ContractViolation, match="positive weight"):
        EpisodeRewardGuard(training, safety)


def test_reward_guard_budgets_every_declared_shaping_term() -> None:
    guard = EpisodeRewardGuard(
        _training_config(include_item_score=True),
        RewardSafetyConfig.from_mapping(
            _budgeted_safety_mapping(shaping_signals=["progress", "item_score"])
        ),
    )

    results = [guard.compose(signals) for signals in _item_score_episode()]

    assert [result.episode_total for result in results] == [
        pytest.approx(value) for value in (10, 20, 25, 25, 25, 25)
    ]
    assert results[-1].positive_shaping_total == pytest.approx(25.0)


def test_reward_guard_fails_closed_on_an_unclassified_reward_term() -> None:
    safety = RewardSafetyConfig.from_mapping(_budgeted_safety_mapping(shaping_signals=["progress"]))

    with pytest.raises(ContractViolation, match="neither the outcome signal"):
        EpisodeRewardGuard(_training_config(include_item_score=True), safety)


def test_reward_guard_accepts_a_fully_classified_configuration() -> None:
    safety = RewardSafetyConfig.from_mapping(
        _budgeted_safety_mapping(shaping_signals=["progress", "item_score"])
    )

    assert "outcome" not in safety.shaping_signals
    assert safety.unbudgeted_signals == ()
    assert EpisodeRewardGuard(_training_config(include_item_score=True), safety) is not None


def test_reward_guard_unbudgeted_opt_in_keeps_the_legacy_behaviour() -> None:
    guard = EpisodeRewardGuard(
        _training_config(include_item_score=True),
        RewardSafetyConfig.from_mapping(
            _budgeted_safety_mapping(
                shaping_signals=["progress"], unbudgeted_signals=["item_score"]
            )
        ),
    )

    results = [guard.compose(signals) for signals in _item_score_episode()]

    assert [result.episode_total for result in results] == [
        pytest.approx(value) for value in (10, 20, 30, 40, 50, 55)
    ]


def test_reward_guard_rejects_an_unbudgeted_signal_without_a_reward_term() -> None:
    safety = RewardSafetyConfig.from_mapping(
        _budgeted_safety_mapping(shaping_signals=["progress"], unbudgeted_signals=["item_score"])
    )

    with pytest.raises(ContractViolation, match="unknown unbudgeted signals"):
        EpisodeRewardGuard(_training_config(), safety)


def test_demonstration_gate_rejects_policy_self_imitation_and_failed_episodes() -> None:
    gate = DemonstrationGate(
        DemonstrationPolicyConfig.from_mapping(_demonstration_policy_mapping())
    )

    accepted = gate.validate(
        DemonstrationProvenance(
            origin=DemonstrationOrigin.HUMAN,
            outcome=DemonstrationOutcome.SUCCESS,
        )
    )
    assert accepted.sample_weight == 3

    with pytest.raises(ContractViolation, match="origin policy"):
        gate.validate(
            DemonstrationProvenance(
                origin=DemonstrationOrigin.POLICY,
                outcome=DemonstrationOutcome.SUCCESS,
                policy_id="candidate-v1",
            )
        )
    with pytest.raises(ContractViolation, match="outcome failure"):
        gate.validate(
            DemonstrationProvenance(
                origin=DemonstrationOrigin.SCRIPTED_EXPERT,
                outcome=DemonstrationOutcome.FAILURE,
            )
        )


def test_demonstration_gate_fails_closed_on_unknown_provenance() -> None:
    gate = DemonstrationGate(
        DemonstrationPolicyConfig.from_mapping(_demonstration_policy_mapping())
    )

    with pytest.raises(ContractViolation, match="unknown"):
        gate.validate(
            DemonstrationProvenance(
                origin=DemonstrationOrigin.UNKNOWN,
                outcome=DemonstrationOutcome.UNKNOWN,
            )
        )
    with pytest.raises(ContractViolation, match="policy_id"):
        DemonstrationProvenance(
            origin=DemonstrationOrigin.HUMAN,
            outcome=DemonstrationOutcome.SUCCESS,
            policy_id="mislabelled-policy",
        )


def test_training_safety_loaders_read_strict_data_only_json(tmp_path: Path) -> None:
    reward_path = tmp_path / "reward-safety.json"
    demonstration_path = tmp_path / "demonstration-policy.json"
    reward_path.write_text(json.dumps(_reward_safety_mapping()), encoding="utf-8")
    demonstration_path.write_text(json.dumps(_demonstration_policy_mapping()), encoding="utf-8")

    reward = load_reward_safety_config(reward_path)
    demonstration = load_demonstration_policy_config(demonstration_path)

    assert reward.outcome_signal == "outcome"
    assert demonstration.origin_weights[DemonstrationOrigin.HUMAN] == 1.5


def test_demonstration_policy_rejects_missing_weights_unknowns_and_extra_fields() -> None:
    missing_weight = _demonstration_policy_mapping()
    missing_weight["origin_weights"] = {"human": 1}
    with pytest.raises(ValueError, match="exactly match"):
        DemonstrationPolicyConfig.from_mapping(missing_weight)

    unknown = _demonstration_policy_mapping()
    unknown["allowed_origins"] = ["human", "unknown"]
    unknown["origin_weights"] = {"human": 1, "unknown": 1}
    with pytest.raises(ValueError, match="reject_unknown"):
        DemonstrationPolicyConfig.from_mapping(unknown)

    extra = _demonstration_policy_mapping()
    extra["callback"] = "module:function"
    with pytest.raises(ValueError, match="unexpected fields"):
        DemonstrationPolicyConfig.from_mapping(extra)
