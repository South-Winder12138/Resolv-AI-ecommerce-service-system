"""Reconstruct selected SQL payloads with existing chunk rules and section parser."""

from collections.abc import Mapping, Sequence

from resolv.knowledge import (
    _body_windows,
    _validate_chunk_rules,
    build_embedding_text,
    chunk_knowledge_document,
    iter_document_sections,
    knowledge_ingest_hash,
)
from resolv.schemas.hybrid_knowledge import (
    HybridCorpusManifest,
    HybridManifestChunk,
    HybridManifestDocument,
    HybridRetrievalError,
    HybridSourceDocument,
)
from resolv.schemas.knowledge import KnowledgeFileConfig
from resolv.services.hybrid_fingerprints import (
    canonical_hash,
    content_hash,
    document_hash,
)


def build_corpus_manifest(
    sources: Sequence[HybridSourceDocument],
    file_configs: Mapping[str, KnowledgeFileConfig],
    *,
    max_chars: int = 800,
    overlap: int = 100,
) -> HybridCorpusManifest:
    try:
        _validate_chunk_rules(max_chars, overlap)
        if not sources:
            raise ValueError("Explicit source selection required")
        documents = []
        ids = set()
        for source in sources:
            if source.document.document_id in ids:
                raise ValueError("Duplicate source")
            ids.add(source.document.document_id)
            config = file_configs[source.document.source]
            if (
                knowledge_ingest_hash(
                    source.document, config, max_chars=max_chars, overlap=overlap
                )
                != source.ingest_hash
            ):
                raise ValueError("Source/ingest/config mismatch")
            rebuilt = chunk_knowledge_document(
                source.document, config, max_chars=max_chars, overlap=overlap
            )
            if [c.model_dump() for c in rebuilt] != [
                c.model_dump() for c in source.chunks
            ]:
                raise ValueError("Stored chunks differ from reconstructed source")
            dh = document_hash(source)
            entries = []
            index = 0
            for ordinal, titles, body in iter_document_sections(source.document):
                body = body.strip()
                if not body:
                    continue
                category = " > ".join(titles[:-1])
                questions = titles[-1] if titles else ""
                budget = max_chars - len(build_embedding_text(category, questions, ""))
                section = canonical_hash(
                    {
                        "rule": "document-ordinal-v1",
                        "document_hash": dh,
                        "ordinal": ordinal,
                    }
                )
                for _ in _body_windows(body, budget, overlap):
                    chunk = source.chunks[index]
                    index += 1
                    entries.append(
                        HybridManifestChunk(
                            chunk=chunk,
                            content_hash=content_hash(chunk),
                            section_instance_id=section,
                            expandable=bool(titles),
                        )
                    )
            if index != len(source.chunks):
                raise ValueError("Section mapping mismatch")
            documents.append(
                HybridManifestDocument(
                    document_id=source.document.document_id,
                    document_hash=dh,
                    ingest_hash=source.ingest_hash,
                    file_config=config,
                    max_chars=max_chars,
                    overlap=overlap,
                    chunks=tuple(entries),
                )
            )
        return HybridCorpusManifest(
            documents=tuple(sorted(documents, key=lambda d: d.document_id))
        )
    except Exception as exc:
        raise HybridRetrievalError("hybrid_manifest_source_mismatch") from exc
