"""Direct developer-only WRITE Tool smoke against a temporary SQLite database."""

import os
from tempfile import TemporaryDirectory

from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.tools.cancel import cancel_order
from resolv.tools.order import query_order


def main() -> None:
    previous_url = os.environ.get("DATABASE_URL")
    try:
        with TemporaryDirectory() as directory:
            database_url = f"sqlite:///{directory}/write-smoke.db"
            os.environ["DATABASE_URL"] = database_url
            engine = create_db_engine(database_url)
            try:
                init_db(engine)
                with create_session_factory(engine).begin() as session:
                    seed_data(session)
            finally:
                engine.dispose()

            print("Before:", query_order.invoke({"order_id": "MH002"}))
            print("First call:", cancel_order.invoke({"order_id": "MH002"}))
            print("After:", query_order.invoke({"order_id": "MH002"}))
            print("Second call:", cancel_order.invoke({"order_id": "MH002"}))
    finally:
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url


if __name__ == "__main__":
    main()
