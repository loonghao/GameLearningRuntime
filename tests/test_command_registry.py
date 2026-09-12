from game_learning_runtime.command_registry import CommandRegistry, CommandSpec

def test_registry_is_deterministic_and_generation_tracked():
    r = CommandRegistry((CommandSpec("shrine_rest"), CommandSpec("shop_buy", params=("item_id",))))
    first = r.capabilities()
    assert r.names() == ("shop_buy", "shrine_rest")
    assert first["generation"] == 1
    r.replace((CommandSpec("equip_item"),))
    assert r.generation == 2
    assert r.require("equip_item").name == "equip_item"

def test_duplicate_commands_rejected():
    try:
        CommandRegistry((CommandSpec("x"), CommandSpec("x")))
    except ValueError:
        return
    assert False
