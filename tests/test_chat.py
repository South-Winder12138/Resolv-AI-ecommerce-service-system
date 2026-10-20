from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import resolv.chat as chat_module


def test_chat_passes_prompt_messages_to_model(monkeypatch) -> None:
    class FakeModel:
        def invoke(self, messages):
            assert isinstance(messages[0], SystemMessage)
            assert isinstance(messages[1], HumanMessage)
            assert messages[1].content == "你好"
            return AIMessage(content="你好，我是 Resolv。")

    monkeypatch.setattr(chat_module, "get_chat_model", lambda: FakeModel())

    assert chat_module.chat("你好") == "你好，我是 Resolv。"
