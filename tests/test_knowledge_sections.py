"""Chapter metadata must survive body splitting without changing v0.1 text."""

import pytest

from resolv import knowledge
from resolv.schemas.knowledge import Document, KnowledgeFileConfig


def _document(text, source="refund.md"):
    return Document(document_id="doc", source=source, text=text)


def _config(keywords=()):
    return KnowledgeFileConfig(content_type="refund_policy", key_clause_keywords=keywords)


def test_refund_chunk_has_category_question_answer_and_full_path():
    chunks = knowledge.chunk_knowledge_document(
        _document("# 售后政策\n\n## 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。\n"),
        _config(["一旦激活", "不支持"]),
    )
    assert len(chunks) == 1
    assert chunks[0].model_dump() == {
        "chunk_id": "09ffccc88c6f9b2568ea5a24621c456b74423871e94f9e024bf231b1696316ff",
        "document_id": "doc", "source": "refund.md", "chunk_index": 0,
        "text": "售后政策 > 软件商品退款\n\n软件商品一旦激活，不支持无理由退款。",
        "category": "售后政策", "questions": "软件商品退款",
        "answer": "软件商品一旦激活，不支持无理由退款。",
        "section_path": "售后政策 > 软件商品退款",
        "content_type": "refund_policy", "is_key_clause": 1,
    }


def test_literal_path_separator_inside_titles_is_not_an_extra_level():
    chunks = knowledge.chunk_knowledge_document(
        _document("# 售后 > 服务\nparent\n## 软件 > 订阅\nchild\n### 激活 > 退款\nleaf"),
        _config(),
    )
    assert [(c.category, c.questions, c.section_path, c.answer) for c in chunks] == [
        ("", "售后 > 服务", "售后 > 服务", "parent"),
        ("售后 > 服务", "软件 > 订阅", "售后 > 服务 > 软件 > 订阅", "child"),
        ("售后 > 服务 > 软件 > 订阅", "激活 > 退款",
         "售后 > 服务 > 软件 > 订阅 > 激活 > 退款", "leaf"),
    ]


def test_sections_preserve_preamble_parent_sibling_jump_and_empty_heading_reset():
    chunks = knowledge.chunk_knowledge_document(
        _document("intro\n# Root\nparent\n### Child\nchild\n## Peer\npeer\n## \nroot again\n#\nend"),
        _config(),
    )
    assert [(c.category, c.questions, c.answer) for c in chunks] == [
        ("", "", "intro"), ("", "Root", "parent"), ("Root", "Child", "child"),
        ("Root", "Peer", "peer"), ("", "Root", "root again"), ("", "", "end"),
    ]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("source", ["policy.md", "policy.markdown", "POLICY.MD", "POLICY.MARKDOWN"])
def test_enhanced_chunking_recognizes_markdown_extensions(source):
    chunk = knowledge.chunk_knowledge_document(_document("# A\nbody", source), _config())[0]
    assert (chunk.category, chunk.questions, chunk.section_path, chunk.answer) == ("", "A", "A", "body")


def test_txt_hash_lines_are_body_and_have_no_heading_metadata():
    text = "# A\nbody\n## B\nchild"
    chunk = knowledge.chunk_knowledge_document(_document(text, "policy.TXT"), _config())[0]
    assert (chunk.category, chunk.questions, chunk.section_path) == ("", "", "")
    assert chunk.answer == chunk.text == text


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_fenced_headings_do_not_change_chapter_metadata(fence):
    body = f"{fence}python\n## Fake\ncode\n{fence}"
    chunks = knowledge.chunk_knowledge_document(
        _document(f"# A\n{body}\n## Real\npolicy"), _config()
    )
    assert [(c.section_path, c.answer) for c in chunks] == [
        ("A", body), ("A > Real", "policy"),
    ]


def test_matching_long_fence_and_unclosed_fence_keep_body_until_close():
    body = "````python\n# Fake\n```\n## Still fake\n~~~~\n````"
    chunks = knowledge.chunk_knowledge_document(
        _document(f"# A\n{body}\n## B\n```\n## Not real\nbody"), _config()
    )
    assert [c.questions for c in chunks] == ["A", "B"]
    assert chunks[1].answer == "```\n## Not real\nbody"


def test_optional_closing_hashes_are_not_part_of_questions():
    chunks = knowledge.chunk_knowledge_document(
        _document("# Refund ##\npolicy\n## Rate#\nbody"), _config()
    )
    assert [(c.category, c.questions) for c in chunks] == [("", "Refund"), ("Refund", "Rate#")]


def test_metadata_repeats_in_each_window_with_header_budget_and_overlap():
    chunks = knowledge.chunk_knowledge_document(
        _document("# P\nabcdefghijklmno"), _config(), max_chars=10, overlap=2
    )
    assert [c.answer for c in chunks] == ["abcdefg", "fghijkl", "klmno"]
    assert [c.text for c in chunks] == ["P\n\nabcdefg", "P\n\nfghijkl", "P\n\nklmno"]
    assert all(c.questions == "P" and c.section_path == "P" and c.category == "" for c in chunks)
    assert all(len(c.text) <= 10 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == 3


def test_overlap_never_carries_body_or_headers_into_next_section():
    chunks = knowledge.chunk_knowledge_document(
        _document("# A\nabcdefghi\n# B\n123456789"), _config(), max_chars=8, overlap=2
    )
    assert [(c.questions, c.answer) for c in chunks] == [
        ("A", "abcde"), ("A", "defgh"), ("A", "ghi"),
        ("B", "12345"), ("B", "45678"), ("B", "789"),
    ]


def test_keyword_match_is_per_answer_window_not_entire_section():
    chunks = knowledge.chunk_knowledge_document(
        _document("abcdefghijBANklmnopqr", "policy.txt"), _config(["BAN"]),
        max_chars=10, overlap=0,
    )
    assert [c.answer for c in chunks] == ["abcdefghij", "BANklmnopq", "r"]
    assert [c.is_key_clause for c in chunks] == [0, 1, 0]
    assert all(type(c.is_key_clause) is int for c in chunks)


def test_keyword_only_in_heading_does_not_mark_body():
    chunks = knowledge.chunk_knowledge_document(
        _document("# BAN\nabcdefghij"), _config(["BAN"]), max_chars=10, overlap=0
    )
    assert all(c.questions == "BAN" for c in chunks)
    assert [c.is_key_clause for c in chunks] == [0, 0]


def test_keyword_matching_is_literal_and_case_sensitive():
    chunks = knowledge.chunk_knowledge_document(
        _document("banana ordinary", "policy.txt"), _config(["BAN", ".*"])
    )
    assert chunks[0].is_key_clause == 0


def test_keyword_split_across_nonoverlapping_windows_does_not_match():
    chunks = knowledge.chunk_knowledge_document(
        _document("abBLOCKcd", "policy.txt"), _config(["BLOCK"]), max_chars=5, overlap=0
    )
    assert [c.answer for c in chunks] == ["abBLO", "CKcd"]
    assert [c.is_key_clause for c in chunks] == [0, 0]


def test_default_window_budget_is_800_characters_with_100_overlap():
    chunks = knowledge.chunk_knowledge_document(_document("中" * 900, "policy.txt"), _config())
    assert [len(c.answer) for c in chunks] == [800, 200]
    assert [c.is_key_clause for c in chunks] == [0, 0]


def test_structural_paragraph_split_retains_chapter_metadata():
    chunks = knowledge.chunk_knowledge_document(
        _document("# A\naaaa\n\nbbbb\ncccc"), _config(), max_chars=15, overlap=0
    )
    assert [c.answer for c in chunks] == ["aaaa", "bbbb\ncccc"]
    assert [c.section_path for c in chunks] == ["A", "A"]


def test_enhanced_text_and_ids_match_legacy_pipeline_and_are_deterministic():
    document = _document("intro\n# A\nabcdefghijklmno\n## B\nbeta")
    legacy = knowledge.chunk_document(document, max_chars=10, overlap=2)
    chunks = knowledge.chunk_knowledge_document(document, _config(), max_chars=10, overlap=2)
    assert [(c.chunk_id, c.chunk_index, c.text) for c in chunks] == [
        (c.chunk_id, c.chunk_index, c.text) for c in legacy
    ]
    assert chunks == knowledge.chunk_knowledge_document(document, _config(), max_chars=10, overlap=2)
    other = document.model_copy(update={"document_id": "different"})
    assert chunks[0].chunk_id != knowledge.chunk_knowledge_document(
        other, _config(), max_chars=10, overlap=2
    )[0].chunk_id


@pytest.mark.parametrize("text", ["", " \n\t", "# A\n## B\n# C\n"])
def test_empty_body_does_not_create_metadata_only_chunk(text):
    assert knowledge.chunk_knowledge_document(_document(text), _config()) == []


@pytest.mark.parametrize(("max_chars", "overlap"), [
    (0, 0), (True, 0), (10.5, 0), (10, -1), (10, 10), (10, False), (10, 0.5),
])
def test_enhanced_chunker_rejects_invalid_window_parameters(max_chars, overlap):
    with pytest.raises(ValueError):
        knowledge.chunk_knowledge_document(
            _document("body"), _config(), max_chars=max_chars, overlap=overlap
        )


@pytest.mark.parametrize(("text", "overlap"), [
    ("# abcdefghi\nbody", 0), ("# abcde\nbody", 3), ("# abcde\nbody", 4),
])
def test_enhanced_chunker_rejects_header_leaving_unusable_body_budget(text, overlap):
    with pytest.raises(ValueError, match="body budget"):
        knowledge.chunk_knowledge_document(_document(text), _config(), max_chars=10, overlap=overlap)


@pytest.mark.parametrize(("category", "questions", "expected"), [
    ("售后政策", "软件退款", "售后政策 > 软件退款\n\n正文"),
    ("", "软件退款", "软件退款\n\n正文"),
    ("售后政策", "", "售后政策\n\n正文"),
    ("", "", "正文"),
])
def test_embedding_text_joins_context_once(category, questions, expected):
    assert knowledge.build_embedding_text(category, questions, "正文") == expected
