"""Knowledge mappings use their own metadata, separate from business SQLite."""

from sqlalchemy.orm import DeclarativeBase


class KnowledgeBase(DeclarativeBase):
    pass
