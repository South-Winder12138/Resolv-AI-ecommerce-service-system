"""Fingerprints pin meaningful ingest/index inputs without service resources."""

import importlib
import subprocess
import sys

import pytest

from resolv.schemas.knowledge import Document, KnowledgeChunk, KnowledgeFileConfig

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def _profile(**changes):
    return importlib.import_module("resolv.schemas.knowledge").KnowledgeIndexProfile(
        model_revision=REVISION, **changes
    )


def _document(**changes):
    return Document(document_id="d" * 64, source="refund.md", text="正文", **changes)


def _chunk(**changes):
    return KnowledgeChunk(**{
        "chunk_id": "c" * 64, "document_id": "d" * 64, "chunk_index": 0,
        "source": "refund.md", "category": "售后", "questions": "退款",
        "answer": "正文", "section_path": "售后 > 退款", "content_type": "refund",
        "is_key_clause": 1, "text": "售后 > 退款\n\n正文", **changes,
    })


def _hash(name, *args, **kwargs):
    return getattr(importlib.import_module("resolv.knowledge"), name)(*args, **kwargs)


def test_ingest_hash_matches_canonical_utf8_golden():
    result = _hash("knowledge_ingest_hash", _document(), KnowledgeFileConfig(content_type="refund"))
    assert result == "82f9de10735cd28d921cf2b29f64816516bb8e1f650c1ec6aeda20a811d1223d"
    assert result == _hash("knowledge_ingest_hash", _document(), KnowledgeFileConfig(content_type="refund"))


@pytest.mark.parametrize("change", ["body", "source", "type", "keywords", "max_chars", "overlap"])
def test_ingest_hash_changes_with_each_ingestion_input(change):
    document = _document()
    config = KnowledgeFileConfig(content_type="refund")
    original = _hash("knowledge_ingest_hash", document, config)
    options = {}
    if change == "body":
        document = document.model_copy(update={"text": "修改正文"})
    elif change == "source":
        document = document.model_copy(update={"source": "another.md"})
    elif change == "type":
        config = KnowledgeFileConfig(content_type="support")
    elif change == "keywords":
        config = KnowledgeFileConfig(content_type="refund", key_clause_keywords=("正文",))
    else:
        options[change] = 900 if change == "max_chars" else 50
    assert _hash("knowledge_ingest_hash", document, config, **options) != original


def test_index_hash_matches_canonical_utf8_golden():
    assert _hash("knowledge_index_hash", _chunk(), _profile()) == "40502da4dcf51d8dadeb5106f399db4908b92db64185a61a99ec02083fa866f1"


@pytest.mark.parametrize("changes", [
    {"chunk_id": "e" * 64}, {"document_id": "e" * 64}, {"chunk_index": 1},
    {"source": "another.md"}, {"text": "修改正文"}, {"section_path": "其他章节"},
    {"content_type": "support"}, {"is_key_clause": 0},
])
def test_index_hash_covers_every_index_projection_field(changes):
    assert _hash("knowledge_index_hash", _chunk(**changes), _profile()) != _hash(
        "knowledge_index_hash", _chunk(), _profile()
    )


@pytest.mark.parametrize("changes", [
    {"model_revision": "a" * 40}, {"collection_name": "another_v2"},
    {"passage_max_length": 1024},
])
def test_index_hash_covers_profile(changes):
    schemas = importlib.import_module("resolv.schemas.knowledge")
    other = schemas.KnowledgeIndexProfile(**{
        "model_revision": REVISION, **changes,
    })
    assert _hash("knowledge_index_hash", _chunk(), other) != _hash(
        "knowledge_index_hash", _chunk(), _profile()
    )


@pytest.mark.parametrize("changes", [
    {"model_revision": "main"}, {"model_revision": "latest"},
    {"model_revision": "a" * 39}, {"model_revision": "A" * 40},
    {"collection_name": ""}, {"collection_name": "bad/name"},
    {"collection_name": "中文"}, {"passage_max_length": True},
    {"passage_max_length": 0}, {"passage_max_length": 8193},
    {"unexpected": True},
])
def test_profile_rejects_unpinned_or_invalid_configuration(changes):
    schemas = importlib.import_module("resolv.schemas.knowledge")
    with pytest.raises(ValueError):
        schemas.KnowledgeIndexProfile(**{"model_revision": REVISION, **changes})


def test_profile_is_immutable():
    with pytest.raises(ValueError):
        _profile().collection_name = "mutated"


@pytest.mark.parametrize("options", [
    {"max_chars": True}, {"overlap": True}, {"max_chars": 0},
    {"overlap": -1}, {"max_chars": 10, "overlap": 10},
])
def test_ingest_hash_rejects_invalid_chunk_rules(options):
    with pytest.raises(ValueError):
        _hash("knowledge_ingest_hash", _document(), KnowledgeFileConfig(content_type="refund"), **options)


def test_plain_hashing_imports_no_database_or_model(tmp_path):
    code = """
import sys
from resolv.knowledge import knowledge_ingest_hash, knowledge_index_hash
from resolv.schemas.knowledge import KnowledgeIndexProfile
assert not {"sqlalchemy", "torch", "FlagEmbedding", "pymilvus"} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
