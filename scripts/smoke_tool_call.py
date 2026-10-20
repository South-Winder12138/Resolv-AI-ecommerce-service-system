"""Observe tool-call requests without executing any tools."""

from pydantic import ValidationError

from resolv.core.config import Settings
from resolv.tool_calling import request_tool_call


def main() -> None:
    try:
        Settings()
    except ValidationError as exc:
        raise SystemExit(
            "Check OPENAI_API_KEY and OPENAI_BASE_URL in .env or the environment."
        ) from exc

    questions = (
        "帮我查订单 MH001 到哪里了。",
        "你好，请介绍一下自己。",
    )
    for question in questions:
        response = request_tool_call(question)
        print(f"question: {question}")
        print(f"response type: {type(response).__name__}")
        print(f"response.text: {response.text!r}")
        print(f"response.tool_calls: {response.tool_calls}")
        for call in response.tool_calls:
            print(f"  name: {call['name']}")
            print(f"  args: {call['args']}")
            print(f"  id: {call['id']}")


if __name__ == "__main__":
    main()
