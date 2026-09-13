import pytest

from game_learning_runtime.decisions import Candidate, Decision, execute_decision, learning_status


@pytest.mark.parametrize(
    "key,command,parameters",
    [
        ("", "walk", "{}"),
        ("a", " ", "{}"),
        ("a", "walk", "[]"),
        ("a", "walk", "invalid"),
        ("a", "walk", '{"nested": {"x": Infinity}}'),
    ],
)
def test_invalid_candidates_are_rejected(key, command, parameters):
    with pytest.raises(ValueError):
        Candidate(key, command, parameters)


@pytest.mark.parametrize(
    "state,items,key,digest,mode",
    [
        ("", (Candidate("a", "walk"),), "a", "digest", "train"),
        ("s", [Candidate("a", "walk")], "a", "digest", "train"),
        ("s", ("invalid",), "a", "digest", "train"),
        ("s", (Candidate("a", "walk"),), "a", "", "train"),
        ("s", (Candidate("a", "walk"),), "a", "digest", "unknown"),
    ],
)
def test_invalid_decisions_are_rejected(state, items, key, digest, mode):
    with pytest.raises(ValueError):
        Decision(state, items, key, digest, mode)


def test_executor_exception_does_not_retry():
    decision = Decision("s", (Candidate("a", "walk"),), "a", "digest", "train")
    calls = []

    def execute(command, parameters):
        calls.append(command)
        raise RuntimeError("outcome unknown")

    with pytest.raises(RuntimeError, match="outcome unknown"):
        execute_decision(decision, execute)
    assert calls == ["walk"]


@pytest.mark.parametrize(
    "transitions,updates,initial,final,expected",
    [
        (1, 0, "a", "a", "learning_unverified"),
        (1, 1, "", "a", "learning_unverified"),
        (1, 1, "a", "a", "policy_unchanged"),
    ],
)
def test_evidence_classifies_missing_and_unchanged_policy(
    transitions, updates, initial, final, expected
):
    assert (
        learning_status(
            transitions=transitions, updates=updates, initial_digest=initial, final_digest=final
        )
        == expected
    )


@pytest.mark.parametrize("transitions,updates", [(-1, 0), (0, -1), (True, 1), (1, 0.5)])
def test_invalid_evidence_counts_are_rejected(transitions, updates):
    with pytest.raises(ValueError):
        learning_status(
            transitions=transitions, updates=updates, initial_digest="a", final_digest="b"
        )


def test_policy_cannot_choose_outside_the_supplied_action_set():
    with pytest.raises(ValueError, match="uniquely"):
        Decision("s", (Candidate("a", "walk"),), "b", "digest", "train")
    with pytest.raises(ValueError, match="uniquely"):
        Decision("s", (Candidate("a", "walk"), Candidate("a", "rest")), "a", "digest", "train")


def test_executor_does_not_replace_rejected_policy_choice():
    decision = Decision(
        "s", (Candidate("rest", "rest"), Candidate("walk", "walk")), "walk", "digest", "evaluate"
    )
    calls = []

    def execute(command, parameters):
        calls.append(command)
        return {"accepted": False}

    result = execute_decision(decision, execute)
    assert calls == ["walk"]
    assert result["selected_key"] == "walk"
    assert result["receipt"]["accepted"] is False


def test_parameter_payload_is_defensively_decoded_and_finite():
    item = Candidate("a", "walk", '{"x": 1}')
    item.parameters["x"] = 8
    assert item.parameters == {"x": 1}
    with pytest.raises(ValueError):
        Candidate("a", "walk", '{"x": NaN}')


def test_training_counters_never_certify_improvement():
    assert (
        learning_status(transitions=0, updates=30, initial_digest="a", final_digest="b")
        == "no_transitions"
    )
    assert (
        learning_status(transitions=30, updates=30, initial_digest="a", final_digest="b")
        == "policy_changed_improvement_unverified"
    )
