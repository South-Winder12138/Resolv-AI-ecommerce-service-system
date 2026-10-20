import pytest
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from resolv.models.knowledge import KnowledgeChunkRecord
from resolv.models.knowledge_hybrid import KnowledgeChunkIndex, KnowledgeIndexVersion
from resolv.repositories.knowledge_hybrid import HybridIndexRepository
from tests.hybrid_fixtures import profile
from tests.test_knowledge_hybrid_repository import hybrid_factory

__all__ = ["hybrid_factory"]


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "ready"},
        {"state": "building "},
        {"profile_hash": "short"},
        {"collection_name": ""},
        {"manifest_hash": "short"},
    ],
)
def test_version_constraints(hybrid_factory, changes):
    factory, _, m = hybrid_factory
    with pytest.raises(IntegrityError), factory.begin() as session:
        HybridIndexRepository(session).create_version("one", profile(), m)
        row = session.get(KnowledgeIndexVersion, "one")
        for key, value in changes.items():
            setattr(row, key, value)
        session.flush()


@pytest.mark.parametrize(
    "changes",
    [
        {"index_status": "done"},
        {"index_status": "pending "},
        {"indexed_hash": "short"},
        {"attempt_count": -1},
        {"document_hash": "short"},
        {"section_instance_id": ""},
    ],
)
def test_chunk_index_constraints(hybrid_factory, changes):
    factory, _, m = hybrid_factory
    with pytest.raises(IntegrityError), factory.begin() as session:
        HybridIndexRepository(session).create_version("one", profile(), m)
        row = session.scalar(select(KnowledgeChunkIndex))
        for key, value in changes.items():
            setattr(row, key, value)
        session.flush()


def test_confirmations_restrict_source_and_version_deletion(hybrid_factory):
    factory, _, m = hybrid_factory
    with factory.begin() as session:
        HybridIndexRepository(session).create_version("one", profile(), m)
    for cls in (KnowledgeChunkRecord, KnowledgeIndexVersion):
        with pytest.raises(IntegrityError), factory.begin() as session:
            session.execute(delete(cls))
