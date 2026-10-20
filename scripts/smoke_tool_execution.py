"""Observe one model tool call, then execute it without a second model request."""

from pydantic import ValidationError

from resolv.core.config import Settings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.tool_calling import request_tool_call
from resolv.tools.executor import execute_tool_call


def main() -> None:
    try:
        Settings()
    except ValidationError as exc:
        raise SystemExit(
            "Check OPENAI_API_KEY and OPENAI_BASE_URL in .env or the environment."
        ) from exc

    engine = create_db_engine(Settings().database_url)
    try:
        init_db(engine)
        with create_session_factory(engine).begin() as session:
            seed_data(session)
    finally:
        engine.dispose()

    response = request_tool_call("帮我查订单 MH001 到哪里了。")
    for call in response.tool_calls:
        print("tool_call:")
        print(f"  name: {call['name']}")
        print(f"  args: {call['args']}")
        print(f"  id: {call['id']}")
        result = execute_tool_call(call)
        print("execution_result:")
        print(f"  tool_call_id: {result.tool_call_id}")
        print(f"  name: {result.name}")
        print(f"  output: {result.output}")

    if not response.tool_calls:
        print(f"No tool call requested. Model text: {response.text}")

    direct_call = {
        "name": "query_order",
        "args": {"order_id": "MH001"},
        "id": "test-call-1",
    }
    print(f"direct_executor_result: {execute_tool_call(direct_call)}")


if __name__ == "__main__":
    main()
