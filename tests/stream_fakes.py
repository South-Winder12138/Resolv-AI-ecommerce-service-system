"""Turn scripted complete model decisions into single-chunk fake streams."""

from langchain_core.messages import AIMessage, AIMessageChunk


def as_chunk(message: AIMessage) -> AIMessageChunk:
    return AIMessageChunk(content=message.content, tool_calls=message.tool_calls)
