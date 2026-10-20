from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

import resolv.chat as chat_module


def test_stream_chat_uses_chat_prompt_and_yields_chunks(monkeypatch) -> None:
    pieces = [AIMessageChunk(content=text) for text in ("你", "好", "！")]

    class FakeModel:
        def stream(self, messages):
            assert len(messages) == 2
            assert isinstance(messages[0], SystemMessage)
            assert isinstance(messages[1], HumanMessage)
            assert messages[1].content == "你好"
            yield from pieces

    monkeypatch.setattr(chat_module, "get_chat_model", lambda: FakeModel())

    chunks = list(chat_module.stream_chat("你好"))

    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
    assert [chunk.text for chunk in chunks] == ["你", "好", "！"]


def test_message_chunks_add_into_complete_reply() -> None:
    chunks = [AIMessageChunk(content=text) for text in ("你", "好", "！")]

    full = chunks[0]
    for chunk in chunks[1:]:
        full += chunk

    assert isinstance(full, AIMessageChunk)
    assert full.text == "你好！"


def test_stream_chat_places_history_before_current_question(monkeypatch) -> None:
    history = [HumanMessage(content="我叫小明"), AIMessage(content="你好，小明")]

    class FakeModel:
        def stream(self, messages):
            assert [type(message) for message in messages] == [
                SystemMessage,
                HumanMessage,
                AIMessage,
                HumanMessage,
            ]
            assert [message.content for message in messages[1:]] == [
                "我叫小明",
                "你好，小明",
                "我叫什么？",
            ]
            yield AIMessageChunk(content="小明")

    monkeypatch.setattr(chat_module, "get_chat_model", lambda: FakeModel())

    assert [chunk.text for chunk in chat_module.stream_chat("我叫什么？", history)] == [
        "小明"
    ]
