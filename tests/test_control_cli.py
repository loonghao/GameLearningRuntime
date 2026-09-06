import pytest

from game_learning_runtime.control_cli import ControlCommand, command


def test_commands_are_provider_neutral_and_contract_bound():
    assert command("train").argv() == ("glr", "train")
    assert command("status", run_id="run-1").argv() == ("glr", "status", "--run-id", "run-1")


def test_diagnostic_and_stop_commands_require_run_identity():
    for name in ("status", "feedback", "reflect", "stop"):
        with pytest.raises(ValueError, match="run_id"):
            command(name)


def test_unknown_command_rejected():
    with pytest.raises(ValueError, match="unknown GLR"):
        command("kill")
    assert ControlCommand.RESTART.value == "restart"
