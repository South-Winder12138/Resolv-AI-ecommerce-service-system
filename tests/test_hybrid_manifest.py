from hashlib import sha256

import pytest

from resolv.knowledge import (
    chunk_knowledge_document,
    iter_document_sections,
    knowledge_ingest_hash,
)
from resolv.schemas.hybrid_knowledge import HybridRetrievalError, HybridSourceDocument
from resolv.schemas.knowledge import Document, KnowledgeFileConfig
from resolv.services.hybrid_manifest import build_corpus_manifest

CONFIG = KnowledgeFileConfig(
    content_type="synthetic-policy", key_clause_keywords=("不能",)
)


def source(text, source="synthetic.md", max_chars=800, overlap=100):
    d = Document(
        document_id=sha256(source.encode()).hexdigest(), source=source, text=text
    )
    return HybridSourceDocument(
        document=d,
        ingest_hash=knowledge_ingest_hash(
            d, CONFIG, max_chars=max_chars, overlap=overlap
        ),
        chunks=tuple(
            chunk_knowledge_document(d, CONFIG, max_chars=max_chars, overlap=overlap)
        ),
    )


def build(s, **kw):
    return build_corpus_manifest((s,), {s.document.source: CONFIG}, **kw)


def test_repeated_headings_have_distinct_section_instances():
    s = source("# 商品\n## 退货\n第一处不能退货。\n# 商品\n## 退货\n第二处可以退货。")
    first, second = build(s).documents[0].chunks
    assert first.chunk.section_path == second.chunk.section_path == "商品 > 退货"
    assert first.section_instance_id != second.section_instance_id
    assert first.expandable and second.expandable
    assert tuple(x.chunk for x in build(s).documents[0].chunks) == s.chunks


def test_sections_with_multiple_chunks_share_identity_and_prelude_is_unexpandable():
    s = source("无标题前言。\n# 退货\n" + "这是不可拆封的退货政策。" * 180)
    chunks = build(s).documents[0].chunks
    assert chunks[0].expandable is False
    assert len(chunks) > 3
    assert len({x.section_instance_id for x in chunks[1:]}) == 1
    assert all(x.expandable for x in chunks[1:])


@pytest.mark.parametrize(
    "change",
    [
        "raw",
        "tail",
        "ingest",
        "missing",
        "extra",
        "payload",
        "order",
        "config",
        "rules",
    ],
)
def test_changed_source_or_stored_chunks_refused(change):
    s = source("# A\n一段正文。\n# B\n第二段正文。")
    options = {}
    config = CONFIG
    if change in ("raw", "tail"):
        s = s.model_copy(
            update={
                "document": s.document.model_copy(
                    update={
                        "text": s.document.text + ("新正文" if change == "raw" else " ")
                    }
                )
            }
        )
    elif change == "ingest":
        s = s.model_copy(update={"ingest_hash": "f" * 64})
    elif change == "missing":
        s = s.model_copy(update={"chunks": s.chunks[:-1]})
    elif change == "extra":
        s = s.model_copy(update={"chunks": s.chunks + s.chunks[:1]})
    elif change == "payload":
        s = s.model_copy(
            update={
                "chunks": (
                    s.chunks[0].model_copy(update={"answer": s.chunks[0].answer + " "}),
                    s.chunks[1],
                )
            }
        )
    elif change == "order":
        s = s.model_copy(update={"chunks": tuple(reversed(s.chunks))})
    elif change == "config":
        config = KnowledgeFileConfig(content_type="other")
    else:
        options = {"max_chars": 900}
    with pytest.raises(HybridRetrievalError):
        build_corpus_manifest((s,), {s.document.source: config}, **options)


def test_code_fences_empty_headings_and_skipped_levels_use_existing_rules():
    s = source("# A\n正文\n```md\n# 伪标题\n```\n### C\n跳级正文\n#\n无标题正文")
    m = build(s)
    assert [c.chunk.section_path for c in m.documents[0].chunks] == ["A", "A > C", ""]
    assert [c.expandable for c in m.documents[0].chunks] == [True, True, False]
    assert "# 伪标题" in m.documents[0].chunks[0].chunk.answer
    sections = list(iter_document_sections(s.document))
    assert [ordinal for ordinal, _, _ in sections] == [0, 1, 2, 3]


def test_txt_and_no_title_markdown_use_only_center():
    for s in (
        source("# 文本中的标题\n正文", "synthetic.txt"),
        source("无明确小节标题。"),
    ):
        assert all(not x.expandable for x in build(s).documents[0].chunks)


def test_manifest_is_sorted_and_document_identity_binds_sections():
    a = source("# A\n同样正文。", "a.md")
    b = source("# A\n同样正文。", "b.md")
    m = build_corpus_manifest((a, b), {"a.md": CONFIG, "b.md": CONFIG})
    assert [x.document_id for x in m.documents] == sorted(
        (a.document.document_id, b.document.document_id)
    )
    assert (
        m.documents[0].chunks[0].section_instance_id
        != m.documents[1].chunks[0].section_instance_id
    )
    with pytest.raises(HybridRetrievalError):
        build_corpus_manifest((a, a), {"a.md": CONFIG})
    with pytest.raises(HybridRetrievalError):
        build_corpus_manifest((a,), {})
