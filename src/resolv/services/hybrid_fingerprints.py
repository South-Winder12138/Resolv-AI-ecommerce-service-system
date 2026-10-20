"""Byte-preserving canonical UTF-8 JSON fingerprints, separate from v2."""
import json
from hashlib import sha256

from resolv.schemas.hybrid_knowledge import (
    HybridCorpusManifest,
    HybridIndexProfile,
    HybridManifestChunk,
    HybridSourceDocument,
)
from resolv.schemas.knowledge import KnowledgeChunk


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return sha256(encoded).hexdigest()


def content_hash(chunk: KnowledgeChunk) -> str:
    return canonical_hash({'rule': 'hybrid-content-v1', 'chunk': chunk.model_dump(mode='json')})


def document_hash(source: HybridSourceDocument) -> str:
    return canonical_hash({'rule': 'hybrid-document-v1', 'document': source.document.model_dump(mode='json'),
                           'ingest_hash': source.ingest_hash})


def profile_hash(profile: HybridIndexProfile) -> str:
    return canonical_hash(profile.model_dump(mode='json'))


def manifest_hash(manifest: HybridCorpusManifest) -> str:
    return canonical_hash(manifest.model_dump(mode='json'))


def index_hash(chunk: HybridManifestChunk, profile: HybridIndexProfile, version_id: str) -> str:
    return canonical_hash({'rule': 'hybrid-index-v1', 'index_version': version_id,
                           'profile': profile.model_dump(mode='json'),
                           'manifest_chunk': chunk.model_dump(mode='json')})
