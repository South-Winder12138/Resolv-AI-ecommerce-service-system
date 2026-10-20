"""Literal hybrid fixtures; no files, engines, models, or external resources."""
from resolv.schemas.hybrid_knowledge import (
    HybridCandidate,
    HybridCorpusManifest,
    HybridIndexProfile,
    HybridManifestChunk,
    HybridManifestDocument,
    HybridSourceDocument,
)
from resolv.schemas.knowledge import Document, KnowledgeChunk, KnowledgeFileConfig


def chunk(**changes):
    data = {'chunk_id': 'a' * 64, 'document_id': 'd' * 64, 'text': '退货\n\n七天内可申请。',
                'source': 'synthetic.md', 'chunk_index': 0, 'category': '', 'questions': '退货',
                'answer': '七天内可申请。', 'section_path': '退货', 'content_type': 'policy', 'is_key_clause': 0}
    data.update(changes)
    return KnowledgeChunk(**data)


def source(**changes):
    data = {'document': Document(document_id='d' * 64, source='synthetic.md', text='# 退货\n七天内可申请。\n'),
                'ingest_hash': 'e' * 64, 'chunks': (chunk(),)}
    data.update(changes)
    return HybridSourceDocument(**data)


def profile(**changes):
    return HybridIndexProfile(collection_name='mhhybrid_gate_fixture', **changes)


def manifest_chunk(**changes):
    data = {'chunk': chunk(), 'content_hash': 'c' * 64, 'section_instance_id': 'f' * 64, 'expandable': True}
    data.update(changes)
    return HybridManifestChunk(**data)


def manifest(**changes):
    data = {'document_id': 'd' * 64, 'document_hash': 'b' * 64, 'ingest_hash': 'e' * 64,
                'file_config': KnowledgeFileConfig(content_type='policy'), 'max_chars': 800,
                'overlap': 100, 'chunks': (manifest_chunk(),)}
    data.update(changes)
    return HybridCorpusManifest(documents=(HybridManifestDocument(**data),))


def candidate(route='dense', rank=1, **changes):
    data = {'index_version': 'fixture-v3', 'chunk_id': 'a' * 64, 'document_id': 'd' * 64,
                'index_hash': 'c' * 64, 'route': route, 'rank': rank, 'raw_score': -0.2}
    data.update(changes)
    return HybridCandidate(**data)
