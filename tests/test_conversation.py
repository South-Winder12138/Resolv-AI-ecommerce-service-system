from langchain_core.messages import AIMessage, HumanMessage

from resolv.conversation import ConversationStore


def test_sessions_are_isolated_and_history_is_a_snapshot() -> None:
    store = ConversationStore()
    store.commit_turn("session-A", "我叫小明", "你好，小明")
    store.commit_turn("session-B", "我叫小红", "你好，小红")

    history_a = store.get_history("session-A")
    history_b = store.get_history("session-B")

    assert history_a == [
        HumanMessage(content="我叫小明"),
        AIMessage(content="你好，小明"),
    ]
    assert history_b == [
        HumanMessage(content="我叫小红"),
        AIMessage(content="你好，小红"),
    ]
    history_a.clear()
    assert len(store.get_history("session-A")) == 2


def test_clear_only_removes_selected_session() -> None:
    store = ConversationStore()
    store.commit_turn("session-A", "你好", "你好")
    store.commit_turn("session-B", "再见", "再见")

    store.clear("session-A")

    assert store.get_history("session-A") == []
    assert len(store.get_history("session-B")) == 2
