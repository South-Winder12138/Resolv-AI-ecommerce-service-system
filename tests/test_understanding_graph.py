from tests.test_understanding_service import accept, require


def test_graph_has_isolated_json_state_and_saved_end(understanding_service):
    s = require(understanding_service)
    row = accept(s)
    s.run("c", row.understanding_id)
    saver = s.checkpoint_factory.saver
    checkpoint = saver.get_tuple(
        {"configurable": {"thread_id": "understanding:" + row.understanding_id}}
    )
    assert checkpoint is not None
    values = checkpoint.checkpoint["channel_values"]
    assert (
        values["phase"] == "end" and values["understanding_id"] == row.understanding_id
    )
    assert "execution_epoch" not in values and "messages" not in values
    assert saver.get_tuple({"configurable": {"thread_id": "c"}}) is None
