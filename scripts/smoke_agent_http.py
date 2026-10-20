"""Exercise two HTTP requests and an approval against a temporary database."""

import os
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import ValidationError

from resolv.core.config import Settings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.main import app
from resolv.tools.order import query_order


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
            database_url = f"sqlite:///{directory}/agent-http-smoke.db"
            os.environ["DATABASE_URL"] = database_url
            engine = create_db_engine(database_url)
            try:
                init_db(engine)
                with create_session_factory(engine).begin() as session:
                    seed_data(session)
            finally:
                engine.dispose()

            with TestClient(app) as client:
                conversation_id = client.post("/conversations").json()[
                    "conversation_id"
                ]
                first = client.post(
                    "/agent/chat",
                    json={
                        "session_id": conversation_id,
                        "request_id": uuid4().hex,
                        "question": "请取消订单 MH002。",
                    },
                )
                print("Request #1:", first.json(), flush=True)
                if first.json().get("status") != "approval_required":
                    print("Model did not request a WRITE Tool.", flush=True)
                    return
                print("Before approval:", query_order.invoke({"order_id": "MH002"}))
                approval_id = first.json()["approval_id"]
                second = client.post(
                    f"/agent/approvals/{approval_id}",
                    json={"session_id": conversation_id, "approved": True},
                )
                print("Request #2:", second.json(), flush=True)
                print("After approval:", query_order.invoke({"order_id": "MH002"}))
    finally:
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url


if __name__ == "__main__":
    main()
