"""Finish and validate one public model decision before any tool can execute."""

import json

from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message

from resolv.tools.registry import ToolInputError


class ToolCallParseError(ToolInputError):
    pass


def stream_model_decision(model, messages):
    aggregate = None
    text_deltas = []
    for chunk in model.stream(messages):
        if not isinstance(chunk, AIMessageChunk):
            raise TypeError("Agent model returned a non-AI message chunk")
        aggregate = chunk if aggregate is None else aggregate + chunk
        if chunk.text:
            text_deltas.append(chunk.text)
    if aggregate is None:
        raise RuntimeError("Agent model stream returned no chunks")
    if aggregate.invalid_tool_calls or (
        aggregate.tool_call_chunks
        and len(aggregate.tool_calls) != len(aggregate.tool_call_chunks)
    ):
        raise ToolCallParseError("Incomplete streamed tool call")
    for chunk in aggregate.tool_call_chunks:
        try:
            parsed_args = json.loads(chunk["args"])
        except (json.JSONDecodeError, TypeError):
            raise ToolCallParseError("Incomplete streamed tool call") from None
        if not isinstance(parsed_args, dict):
            raise ToolCallParseError("Streamed tool arguments must be an object")
    response = message_chunk_to_message(aggregate)
    if not isinstance(response, AIMessage):
        raise TypeError("Agent model stream did not produce an AIMessage")
    for call in response.tool_calls:
        if (
            not call.get("name")
            or not call.get("id")
            or not isinstance(call.get("args"), dict)
        ):
            raise ToolCallParseError("Incomplete streamed tool call")
    return response, text_deltas
