from collections.abc import Iterator, Sequence

from langchain_core.messages import AIMessageChunk, BaseMessage

from resolv.core.llm import get_chat_model
from resolv.core.prompts import CHAT_PROMPT


def chat(question: str, history: Sequence[BaseMessage] = ()) -> str:
    """Answer one question using explicit prompt and model steps."""
    prompt_value = CHAT_PROMPT.invoke({"history": list(history), "question": question})
    messages = prompt_value.to_messages()
    model = get_chat_model()
    response = model.invoke(messages)
    return response.text


def stream_chat(
    question: str, history: Sequence[BaseMessage] = ()
) -> Iterator[AIMessageChunk]:
    """Yield the model's incremental message chunks for one question."""
    prompt_value = CHAT_PROMPT.invoke({"history": list(history), "question": question})
    messages = prompt_value.to_messages()
    model = get_chat_model()
    yield from model.stream(messages)
