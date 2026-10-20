import json

from langchain_core.messages import AIMessage, ToolMessage

from resolv.core.llm import get_chat_model
from resolv.core.prompts import TOOL_CALLING_PROMPT
from resolv.tools.executor import ToolExecutionResult, execute_tool_call
from resolv.tools.order import query_order


class UnsupportedToolCallError(RuntimeError):
    """The model requested a tool-call pattern beyond this teaching phase."""


def request_tool_call(question: str) -> AIMessage:
    """Ask the model whether it wants a tool; do not execute the tool."""
    prompt_value = TOOL_CALLING_PROMPT.invoke({"question": question})
    messages = prompt_value.to_messages()
    model = get_chat_model()
    model_with_tools = model.bind_tools([query_order])
    return model_with_tools.invoke(messages)


def tool_result_to_message(result: ToolExecutionResult) -> ToolMessage:
    """Encode one Python tool result for the matching model tool call."""
    content = json.dumps(
        result.output, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return ToolMessage(
        content=content,
        tool_call_id=result.tool_call_id,
        name=result.name,
    )


def answer_with_tool(question: str) -> str:
    """Complete at most one model-requested tool call and return the final text."""
    prompt_value = TOOL_CALLING_PROMPT.invoke({"question": question})
    messages = prompt_value.to_messages()
    model = get_chat_model()
    model_with_tools = model.bind_tools([query_order])

    first_response = model_with_tools.invoke(messages)
    if not first_response.tool_calls:
        return first_response.text
    if len(first_response.tool_calls) > 1:
        raise UnsupportedToolCallError("Phase 9 supports at most one tool call.")

    execution_result = execute_tool_call(first_response.tool_calls[0])
    tool_message = tool_result_to_message(execution_result)
    final_response = model_with_tools.invoke([*messages, first_response, tool_message])
    if final_response.tool_calls:
        raise UnsupportedToolCallError("Phase 9 does not support a second tool call.")
    return final_response.text
