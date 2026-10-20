"""Observe real Agent step events over SSE against temporary seeded SQLite."""

import os
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import ValidationError

from resolv.core.config import Settings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.main import app


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
            database_url = f"sqlite:///{directory}/agent-stream-smoke.db"
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
                with client.stream(
                    "POST",
                    "/agent/stream",
                    json={
                        "session_id": conversation_id,
                        "request_id": uuid4().hex,
                        "question": "查一下 MH001，如果已经发货，再查它现在到哪里了。",
                    },
                ) as response:
                    print("HTTP status:", response.status_code, flush=True)
                    print("Content-Type:", response.headers["content-type"], flush=True)
                    for line in response.iter_lines():
                        if line:
                            print(line, flush=True)

    finally:
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url


if __name__ == "__main__":
    main()
