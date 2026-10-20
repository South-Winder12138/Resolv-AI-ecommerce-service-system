"""Verify that LangChain Tools read seeded SQLite data without calling GPT."""

from resolv.core.config import DatabaseSettings
from resolv.db.engine import create_db_engine, create_session_factory
from resolv.db.init_db import init_db, seed_data
from resolv.tools.logistics import query_logistics
from resolv.tools.order import query_order


def main() -> None:
    engine = create_db_engine(DatabaseSettings().database_url)
    try:
        init_db(engine)
        with create_session_factory(engine).begin() as session:
            seed_data(session)
    finally:
        engine.dispose()

    print("MH001:", query_order.invoke({"order_id": "MH001"}))
    print("SF001:", query_logistics.invoke({"tracking_no": "SF001"}))
    print("MH999:", query_order.invoke({"order_id": "MH999"}))


if __name__ == "__main__":
    main()
