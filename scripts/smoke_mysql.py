"""Check the knowledge MySQL connection without creating tables or writing data."""

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from resolv.core.config import KnowledgeDatabaseSettings
from resolv.db.knowledge import KnowledgeStoreError, knowledge_engine


def main() -> None:
    try:
        settings = KnowledgeDatabaseSettings()
        with knowledge_engine(settings) as engine:
            connection = None
            body_error: BaseException | None = None
            try:
                connection = engine.connect()
                result = connection.scalar(text("SELECT 1"))
                if type(result) is not int or result != 1:
                    raise KnowledgeStoreError("Knowledge database returned an invalid connectivity result.")
                print("Knowledge MySQL SELECT 1:", result)
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception as exc:
                        if body_error is None:
                            raise KnowledgeStoreError("Knowledge database connection cleanup failed.") from exc
                        body_error.add_note("Knowledge database connection cleanup also failed.")
    except ValidationError as exc:
        raise SystemExit("Knowledge MySQL configuration invalid. Check KNOWLEDGE_DATABASE_URL.") from exc
    except (KnowledgeStoreError, SQLAlchemyError) as exc:
        raise SystemExit("Knowledge MySQL smoke failed. Check server, credentials and connectivity.") from exc
    print("MySQL smoke passed; resources closed.")


if __name__ == "__main__":
    main()
