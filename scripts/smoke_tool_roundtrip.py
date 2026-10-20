"""Verify a single-tool roundtrip and a no-tool reply with real model calls."""

from pydantic import ValidationError

from resolv.core.config import Settings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.tool_calling import answer_with_tool


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

    for question in (
        "帮我查订单 MH001 到哪里了。",
        "你好，请介绍一下自己。",
    ):
        print(f"question: {question}", flush=True)
        print(f"final_answer: {answer_with_tool(question)}", flush=True)


if __name__ == "__main__":
    main()
