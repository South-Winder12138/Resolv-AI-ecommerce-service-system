"""Observe refund READ and real-model WRITE approval with temporary SQLite."""

import os
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from resolv.agent_loop import (
    AgentApprovalRequired,
    AgentCompleted,
    resume_agent,
    run_agent,
)
from resolv.core.config import Settings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.tools.refund import apply_refund, query_refund_status


def main() -> None:
    try:
        Settings()
    except ValidationError as exc:
        raise SystemExit(
            "Check OPENAI_API_KEY and OPENAI_BASE_URL in .env or the environment."
        ) from exc

    previous_url = os.environ.get("DATABASE_URL")
    try:
        with TemporaryDirectory() as directory:
            database_url = f"sqlite:///{directory}/refund-smoke.db"
            os.environ["DATABASE_URL"] = database_url
            engine = create_db_engine(database_url)
            try:
                init_db(engine)
                with create_session_factory(engine).begin() as session:
                    seed_data(session)
            finally:
                engine.dispose()

            print(
                "Before:",
                query_refund_status.invoke({"order_id": "MH001"}),
                flush=True,
            )
            first = run_agent("订单 MH001 到货有问题，我想申请退款，原因是商品损坏。")
            if not isinstance(first, AgentApprovalRequired):
                print("Model did not request a WRITE Tool:", first, flush=True)
                return

            pending = first.request.pending_call
            print("Approval required:", first.request.message, flush=True)
            print("Frozen action:", pending.name, pending.args, pending.tool_call_id)
            print(
                "Before approval:",
                query_refund_status.invoke({"order_id": "MH001"}),
                flush=True,
            )
            if pending.name != apply_refund.name:
                print("The model requested a different WRITE Tool; not approving.")
                return

            # Developer smoke explicitly approves only the frozen refund action.
            resumed = resume_agent(first.state, approved=True)
            if isinstance(resumed, AgentCompleted):
                print("Final answer:", resumed.text, flush=True)
            else:
                print("Further approval required:", resumed.request.message, flush=True)
            print(
                "After:",
                query_refund_status.invoke({"order_id": "MH001"}),
                flush=True,
            )
            print(
                "Repeat application:",
                apply_refund.invoke({"order_id": "MH001", "reason": "不想要了"}),
                flush=True,
            )
    finally:
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url


if __name__ == "__main__":
    main()
