import importlib

import pytest
from pydantic import ValidationError

from resolv.schemas.knowledge import Document


@pytest.mark.parametrize(
    "filename",
    [
        "policy.md",
        "policy.markdown",
        "policy.txt",
        "POLICY.MD",
        "POLICY.MARKDOWN",
        "POLICY.TXT",
    ],
)
def test_loader_accepts_text_extensions(tmp_path, filename) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    path = tmp_path / filename
    path.write_text("软件商品退款政策。", encoding="utf-8")
    document = knowledge.load_document(filename, source_root=tmp_path)
    assert document.text == "软件商品退款政策。"
    assert document.source == filename


def test_loader_normalizes_bom_and_newlines(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    path = tmp_path / "policy.txt"
    path.write_bytes("\ufeff第一行\r\n第二行\r第三行\n".encode("utf-8"))
    document = knowledge.load_document(path, source_root=tmp_path)
    assert document.text == "第一行\n第二行\n第三行\n"


def test_loader_accepts_empty_text(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    assert knowledge.load_document("empty.txt", source_root=tmp_path).text == ""


def test_loader_derives_stable_id_from_normalized_source(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    directory = tmp_path / "knowledge"
    directory.mkdir()
    path = directory / "refund.md"
    path.write_text("退款政策", encoding="utf-8")
    absolute = knowledge.load_document(path, source_root=tmp_path)
    normalized = knowledge.load_document(
        "knowledge/../knowledge/./refund.md", source_root=tmp_path
    )
    assert absolute == normalized
    assert absolute.source == "knowledge/refund.md"
    assert (
        absolute.document_id
        == "c8c842d195a89114604bfddcf36a38ed2f36289266e412a1b4e64c2fbee8a210"
    )


def test_loader_content_edit_keeps_document_identity(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    path = tmp_path / "refund.md"
    path.write_text("旧政策", encoding="utf-8")
    before = knowledge.load_document(path, source_root=tmp_path)
    path.write_text("新政策", encoding="utf-8")
    after = knowledge.load_document(path, source_root=tmp_path)
    assert after.document_id == before.document_id
    assert after.text == "新政策"
    assert after.text != before.text


def test_loader_identity_is_portable_between_source_roots(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    documents = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "refund.txt").write_text("相同政策", encoding="utf-8")
        documents.append(knowledge.load_document("refund.txt", source_root=root))
    assert documents[0] == documents[1]


def test_loader_default_root_does_not_follow_cwd(tmp_path, monkeypatch) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    monkeypatch.chdir(tmp_path)
    document = knowledge.load_document("README.md")
    assert document.source == "README.md"
    assert document.text.startswith("# Resolv")


def test_loader_rejects_unsupported_file_type(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    (tmp_path / "policy.pdf").write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="file type"):
        knowledge.load_document("policy.pdf", source_root=tmp_path)


def test_loader_preserves_missing_file_error(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    with pytest.raises(FileNotFoundError):
        knowledge.load_document("missing.txt", source_root=tmp_path)


def test_loader_preserves_decode_error(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    (tmp_path / "broken.txt").write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        knowledge.load_document("broken.txt", source_root=tmp_path)


def test_loader_requires_directory_root(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    path = tmp_path / "policy.txt"
    path.write_text("政策", encoding="utf-8")
    with pytest.raises(ValueError, match="source_root"):
        knowledge.load_document(path, source_root=path)


@pytest.mark.parametrize("relative", [False, True])
def test_loader_rejects_path_outside_source_root(tmp_path, relative) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("外部政策", encoding="utf-8")
    path = "../outside.txt" if relative else outside
    with pytest.raises(ValueError, match="source_root"):
        knowledge.load_document(path, source_root=root)


def test_loader_rejects_symlink_escape(tmp_path) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("外部政策", encoding="utf-8")
    (root / "alias.md").symlink_to(outside)
    with pytest.raises(ValueError, match="source_root"):
        knowledge.load_document("alias.md", source_root=root)


def test_chunk_and_search_hit_have_separate_score_contracts() -> None:
    schemas = importlib.import_module("resolv.schemas.knowledge")
    payload = {
        "chunk_id": "chunk",
        "document_id": "document",
        "text": "退款政策",
        "source": "refund.md",
        "chunk_index": 0,
    }
    chunk = schemas.Chunk(**payload)
    hit = schemas.SearchHit(**payload, score=0.75)
    assert chunk.model_dump() == payload
    assert hit.model_dump() == {**payload, "score": 0.75}


def test_chunk_rejects_negative_index() -> None:
    schemas = importlib.import_module("resolv.schemas.knowledge")
    with pytest.raises(ValidationError, match="chunk_index"):
        schemas.Chunk(
            chunk_id="chunk",
            document_id="document",
            text="政策",
            source="refund.md",
            chunk_index=-1,
        )


@pytest.mark.parametrize("score", [float("inf"), float("-inf"), float("nan")])
def test_search_hit_rejects_nonfinite_score(score) -> None:
    schemas = importlib.import_module("resolv.schemas.knowledge")
    with pytest.raises(ValidationError, match="score"):
        schemas.SearchHit(
            chunk_id="chunk",
            document_id="document",
            text="政策",
            source="refund.md",
            chunk_index=0,
            score=score,
        )


def _document(text: str, source: str = "refund.md") -> Document:
    return Document(document_id="doc", source=source, text=text)


def test_chunker_preserves_preamble_and_heading_hierarchy() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    document = _document("intro\n\n# A\nalpha\n### C\nchild\n## B\nbeta\n# Z\nlast")
    chunks = knowledge.chunk_document(document)
    assert [chunk.text for chunk in chunks] == [
        "intro",
        "A\n\nalpha",
        "A > C\n\nchild",
        "A > B\n\nbeta",
        "Z\n\nlast",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3, 4]
    assert all(
        chunk.document_id == "doc" and chunk.source == "refund.md" for chunk in chunks
    )


@pytest.mark.parametrize(
    "source", ["refund.md", "refund.markdown", "REFUND.MD", "REFUND.MARKDOWN"]
)
def test_chunker_recognizes_markdown_source_extensions(source) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(_document("# A\nalpha", source))
    assert [chunk.text for chunk in chunks] == ["A\n\nalpha"]


def test_chunker_omits_empty_sections() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(_document("# A\n## Empty\n## B\ntext\n# Tail\n"))
    assert [chunk.text for chunk in chunks] == ["A > B\n\ntext"]
    assert chunks[0].chunk_index == 0


def test_chunker_handles_optional_closing_hashes() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(_document("# Refund ##\npolicy\n# Rate#\nbody"))
    assert [chunk.text for chunk in chunks] == ["Refund\n\npolicy", "Rate#\n\nbody"]


def test_chunker_keeps_non_atx_lines_as_body() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    text = "#NoSpace\n####### TooDeep\n    # Indented"
    assert [chunk.text for chunk in knowledge.chunk_document(_document(text))] == [text]


def test_chunker_treats_txt_headings_as_body() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    text = "# A\nalpha\n## B\nbeta"
    chunks = knowledge.chunk_document(_document(text, "refund.TXT"))
    assert [chunk.text for chunk in chunks] == [text]


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_chunker_ignores_headings_in_fenced_code(fence) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    text = f"# A\n{fence}python\n## Fake\ncode\n{fence}\n## Real\nbody"
    chunks = knowledge.chunk_document(_document(text))
    assert [chunk.text for chunk in chunks] == [
        f"A\n\n{fence}python\n## Fake\ncode\n{fence}",
        "A > Real\n\nbody",
    ]


def test_chunker_requires_matching_fence_character_and_length() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    code = "````python\n# Fake\n```\n## Still fake\n~~~~\n````"
    chunks = knowledge.chunk_document(_document(f"# A\n{code}\n## Real\nbody"))
    assert [chunk.text for chunk in chunks] == [f"A\n\n{code}", "A > Real\n\nbody"]


def test_chunker_preserves_unclosed_fence_to_eof() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(_document("# A\n```\n## Fake\nbody"))
    assert [chunk.text for chunk in chunks] == ["A\n\n```\n## Fake\nbody"]


@pytest.mark.parametrize("line", ["```example```", "```python `hint`"])
def test_chunker_invalid_backtick_fence_info_keeps_following_heading(line) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(_document(f"# A\n{line}\n## B\nrefund policy"))
    assert [chunk.text for chunk in chunks] == [
        f"A\n\n{line}",
        "A > B\n\nrefund policy",
    ]


def test_chunker_allows_backticks_in_tilde_fence_info() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    body = "~~~python `hint`\n## Fake\ncode\n~~~"
    chunks = knowledge.chunk_document(_document(f"# A\n{body}\n## B\nrefund policy"))
    assert [chunk.text for chunk in chunks] == [
        f"A\n\n{body}",
        "A > B\n\nrefund policy",
    ]


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        ("aaaa\n\nbbbb\ncccc", 12, ["aaaa", "bbbb\ncccc"]),
        ("aaaa\n \nbbbb\ncccc", 12, ["aaaa", "bbbb\ncccc"]),
        ("aaaa\nbbbb\ncccc", 10, ["aaaa\nbbbb", "cccc"]),
        ("甲乙。丙丁。戊己。", 5, ["甲乙。", "丙丁。", "戊己。"]),
        ("One. Two. Three.", 9, ["One. Two.", "Three."]),
    ],
)
def test_chunker_prefers_structural_boundaries(text, max_chars, expected) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(
        _document(text, "policy.txt"), max_chars=max_chars, overlap=0
    )
    assert [chunk.text for chunk in chunks] == expected


def test_chunker_hard_windows_preserve_content_with_overlap() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(
        _document("abcdefghijklmnopqrstuv", "policy.txt"), max_chars=10, overlap=3
    )
    assert [chunk.text for chunk in chunks] == ["abcdefghij", "hijklmnopq", "opqrstuv"]
    assert (
        chunks[0].chunk_id
        == "09ffccc88c6f9b2568ea5a24621c456b74423871e94f9e024bf231b1696316ff"
    )
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert len({chunk.chunk_id for chunk in chunks}) == 3


def test_chunker_counts_header_prefix_in_budget() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(
        _document("# P\nabcdefghijklmno"), max_chars=10, overlap=2
    )
    assert [chunk.text for chunk in chunks] == [
        "P\n\nabcdefg",
        "P\n\nfghijkl",
        "P\n\nklmno",
    ]
    assert all(len(chunk.text) <= 10 for chunk in chunks)


def test_chunker_overlap_does_not_cross_sections() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(
        _document("# A\nabcdefghi\n# B\n123456789"), max_chars=8, overlap=2
    )
    assert [chunk.text for chunk in chunks] == [
        "A\n\nabcde",
        "A\n\ndefgh",
        "A\n\nghi",
        "B\n\n12345",
        "B\n\n45678",
        "B\n\n789",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3, 4, 5]


def test_chunker_close_overlap_budget_still_makes_progress() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(
        _document("abcdefg", "policy.txt"), max_chars=3, overlap=2
    )
    assert [chunk.text for chunk in chunks] == ["abc", "bcd", "cde", "def", "efg"]


def test_chunker_defaults_apply_character_window_and_overlap() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(_document("中" * 900, "policy.txt"))
    assert [len(chunk.text) for chunk in chunks] == [800, 200]


def test_chunker_counts_unicode_characters_not_bytes() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    chunks = knowledge.chunk_document(
        _document("🐱猫" * 6, "policy.txt"), max_chars=4, overlap=0
    )
    assert [chunk.text for chunk in chunks] == ["🐱猫🐱猫", "🐱猫🐱猫", "🐱猫🐱猫"]


def test_chunker_is_deterministic_for_same_document() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    document = _document("# A\nalpha\n# B\nbeta")
    assert knowledge.chunk_document(document) == knowledge.chunk_document(document)


def test_chunker_ids_include_document_identity() -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    first = _document("policy", "policy.txt")
    second = first.model_copy(update={"document_id": "different"})
    assert (
        knowledge.chunk_document(first)[0].chunk_id
        != knowledge.chunk_document(second)[0].chunk_id
    )


@pytest.mark.parametrize("text", ["", " \n\t", "# A\n## B\n\n# C\n"])
def test_chunker_empty_content_has_no_chunks(text) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    assert knowledge.chunk_document(_document(text)) == []


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10.5, 0),
        (True, 0),
        (10, -1),
        (10, 10),
        (10, 11),
        (10, 0.5),
        (10, False),
    ],
)
def test_chunker_rejects_invalid_parameters(max_chars, overlap) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    with pytest.raises(ValueError):
        knowledge.chunk_document(
            _document("policy"), max_chars=max_chars, overlap=overlap
        )


@pytest.mark.parametrize(
    ("text", "overlap"),
    [("# abcdefghi\nbody", 0), ("# abcde\nbody", 3), ("# abcde\nbody", 4)],
)
def test_chunker_rejects_unusable_body_budget(text, overlap) -> None:
    knowledge = importlib.import_module("resolv.knowledge")
    with pytest.raises(ValueError, match="body budget"):
        knowledge.chunk_document(_document(text), max_chars=10, overlap=overlap)
