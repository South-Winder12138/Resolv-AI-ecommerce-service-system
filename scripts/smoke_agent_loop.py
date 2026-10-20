"""Observe real bounded agent runs without HTTP or a framework agent API."""

import logging

from pydantic import ValidationError

from resolv.agent_loop import AgentApprovalRequired, AgentCompleted, run_agent
from resolv.core.config import Settings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data


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

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for question in (
        "你好",
        "帮我查 MH001 的订单状态。",
        "帮我查 MH001 的订单，如果已经发货，再告诉我物流状态。",
    ):
        print(f"question: {question}", flush=True)
        result = run_agent(question)
        if isinstance(result, AgentCompleted):
            print(f"final_answer: {result.text}", flush=True)
        elif isinstance(result, AgentApprovalRequired):
            print(f"approval_required: {result.request.message}", flush=True)


if __name__ == "__main__":
    main()
