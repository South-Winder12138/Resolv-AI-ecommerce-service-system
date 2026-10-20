from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from resolv.core.prompts import CHAT_PROMPT


def test_prompt_has_system_then_human_message() -> None:
    prompt_value = CHAT_PROMPT.invoke({"history": [], "question": "你好"})

    assert isinstance(prompt_value, ChatPromptValue)
    messages = prompt_value.to_messages()
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


def test_question_becomes_human_message() -> None:
    messages = CHAT_PROMPT.invoke(
        {"history": [], "question": "我的订单是 MH001"}
    ).to_messages()

    assert messages[1].content == "我的订单是 MH001"


def test_system_prompt_states_data_boundary() -> None:
    messages = CHAT_PROMPT.invoke(
        {"history": [], "question": "订单到哪里了？"}
    ).to_messages()
    system_text = messages[0].content

    assert "无法查询" in system_text
    assert "不得编造" in system_text
    assert "订单" in system_text


def test_history_precedes_current_question_once() -> None:
    messages = CHAT_PROMPT.invoke(
        {
            "history": [
                HumanMessage(content="我叫小明"),
                AIMessage(content="你好，小明"),
            ],
            "question": "我叫什么？",
        }
    ).to_messages()

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
