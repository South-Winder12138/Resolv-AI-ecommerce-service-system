"""Literal knowledge records shared by persistence tests only."""

import importlib


def document_record(**overrides):
    models = importlib.import_module("resolv.models.knowledge")
    return models.KnowledgeDocument(**{
        "document_id": "d" * 64, "source": "knowledge/refund.md",
        "content_type": "refund_policy",
        "raw_text": "# 售后政策\n## 软件商品退款\n软件商品一旦激活，不支持无理由退款。",
        "ingest_hash": "a" * 64, **overrides,
    })


def chunk_record(**overrides):
    models = importlib.import_module("resolv.models.knowledge")
    return models.KnowledgeChunkRecord(**{
        "chunk_id": "c" * 64, "document_id": "d" * 64, "chunk_index": 0,
        "source": "knowledge/refund.md", "category": "售后政策",
        "questions": "软件商品退款", "answer": "软件商品一旦激活，不支持无理由退款。",
        "section_path": "售后政策 > 软件商品退款", "content_type": "refund_policy",
        "text": "售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。",
        "is_key_clause": 1, "index_hash": "1" * 64, **overrides,
    })
