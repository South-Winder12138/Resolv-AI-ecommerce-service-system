"""File metadata contracts: reject ambiguous mappings before indexing."""

import pytest
from pydantic import ValidationError

from resolv import knowledge
from resolv.schemas import knowledge as schemas


def _payload():
    return {
        "chunk_id": "chunk", "document_id": "doc", "source": "policy.md",
        "chunk_index": 0, "text": "售后政策 > 软件退款\n\n不能退款。",
        "category": "售后政策", "questions": "软件退款", "answer": "不能退款。",
        "section_path": "售后政策 > 软件退款", "content_type": "refund",
        "is_key_clause": 1,
    }


def _config_path(tmp_path, text):
    path = tmp_path / "files.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_enhanced_chunk_preserves_fields_without_query_score():
    chunk = schemas.KnowledgeChunk(**_payload())
    assert chunk.model_dump() == _payload()
    assert type(chunk.is_key_clause) is int
    with pytest.raises(ValidationError):
        schemas.KnowledgeChunk(**_payload(), score=0.9)


@pytest.mark.parametrize("flag", [-1, 2, True, False, "1", "0", 1.0, None])
def test_enhanced_chunk_rejects_non_integer_binary_flag(flag):
    with pytest.raises(ValidationError):
        schemas.KnowledgeChunk(**{**_payload(), "is_key_clause": flag})


@pytest.mark.parametrize("flag", [0, 1])
def test_enhanced_chunk_accepts_integer_binary_flag(flag):
    assert schemas.KnowledgeChunk(**{**_payload(), "is_key_clause": flag}).is_key_clause == flag


@pytest.mark.parametrize("content_type", ["", " \t", 12, True])
def test_enhanced_chunk_requires_nonblank_business_type(content_type):
    with pytest.raises(ValidationError):
        schemas.KnowledgeChunk(**{**_payload(), "content_type": content_type})


def test_enhanced_chunk_requires_body():
    with pytest.raises(ValidationError):
        schemas.KnowledgeChunk(**{**_payload(), "answer": ""})


def test_file_config_normalizes_values_and_defaults_empty_keywords():
    config = schemas.KnowledgeFileConfig(content_type=" refund ")
    assert config.content_type == "refund"
    assert config.key_clause_keywords == ()
    configured = schemas.KnowledgeFileConfig(
        content_type="refund", key_clause_keywords=[" 一旦激活 ", "不支持"]
    )
    assert configured.key_clause_keywords == ("一旦激活", "不支持")


@pytest.mark.parametrize("payload", [
    {}, {"content_type": ""}, {"content_type": "  "}, {"content_type": 42},
    {"content_type": "refund", "unknown": "value"},
    {"content_type": "refund", "key_clause_keywords": "不能退款"},
    {"content_type": "refund", "key_clause_keywords": [""]},
    {"content_type": "refund", "key_clause_keywords": [" \t"]},
    {"content_type": "refund", "key_clause_keywords": [123]},
])
def test_file_config_rejects_missing_invalid_or_unknown_fields(payload):
    with pytest.raises(ValidationError):
        schemas.KnowledgeFileConfig(**payload)


def test_toml_maps_normalized_markdown_and_txt_paths(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    config = _config_path(tmp_path, '''
[files."knowledge/../knowledge/refund.markdown"]
content_type = "refund"
key_clause_keywords = ["激活", "不支持"]
[files."policy.TXT"]
content_type = "general"
''')
    files = knowledge.load_knowledge_file_configs(config, source_root=root)
    assert set(files) == {"knowledge/refund.markdown", "policy.TXT"}
    assert files["knowledge/refund.markdown"].key_clause_keywords == ("激活", "不支持")
    assert files["policy.TXT"].key_clause_keywords == ()
    assert knowledge.get_knowledge_file_config(files, "policy.TXT").content_type == "general"


def test_config_lookup_matches_loader_source_alias(tmp_path):
    (tmp_path / "policy.md").write_text("正文", encoding="utf-8")
    config = _config_path(tmp_path, '[files."./policy.md"]\ncontent_type = "policy"\n')
    files = knowledge.load_knowledge_file_configs(config, source_root=tmp_path)
    document = knowledge.load_document("folder/../policy.md", source_root=tmp_path)
    assert document.source == "policy.md"
    assert knowledge.get_knowledge_file_config(files, document.source).content_type == "policy"


def test_file_mapping_is_portable_between_source_roots(tmp_path):
    config = _config_path(tmp_path, '[files."policy.md"]\ncontent_type = "policy"\n')
    roots = [tmp_path / "first", tmp_path / "second"]
    for root in roots:
        root.mkdir()
    first, second = [
        knowledge.load_knowledge_file_configs(config, source_root=root) for root in roots
    ]
    assert first == second
    assert list(first) == ["policy.md"]


def test_normalized_aliases_are_rejected_as_ambiguous(tmp_path):
    config = _config_path(tmp_path, '''
[files."policy.md"]
content_type = "first"
[files."./policy.md"]
content_type = "second"
''')
    with pytest.raises(ValueError, match="Duplicate"):
        knowledge.load_knowledge_file_configs(config, source_root=tmp_path)


@pytest.mark.parametrize("file_key", ["../outside.md", "", "policy.pdf"])
def test_config_rejects_escaping_empty_or_unsupported_source(tmp_path, file_key):
    config = _config_path(tmp_path, f'[files."{file_key}"]\ncontent_type = "policy"\n')
    with pytest.raises(ValueError):
        knowledge.load_knowledge_file_configs(config, source_root=tmp_path)


def test_config_rejects_symlink_outside_root(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("外部文件", encoding="utf-8")
    (root / "alias.md").symlink_to(outside)
    config = _config_path(tmp_path, '[files."alias.md"]\ncontent_type = "policy"\n')
    with pytest.raises(ValueError):
        knowledge.load_knowledge_file_configs(config, source_root=root)


@pytest.mark.parametrize("text", [
    "", "files = []", "[other]\nname = 'unknown'",
    "[files]\n[other]\nname = 'unknown'",
    '[files."policy.md"]\ncontent_type = "policy"\nunknown = true',
    '[files."policy.md"]\ncontent_type = "  "',
    '[files."policy.md"]\ncontent_type = "policy"\nkey_clause_keywords = [""]',
    "[files.\n",
])
def test_config_rejects_invalid_toml_or_shape(tmp_path, text):
    config = _config_path(tmp_path, text)
    with pytest.raises(ValueError):
        knowledge.load_knowledge_file_configs(config, source_root=tmp_path)


def test_empty_files_table_is_valid_but_missing_requested_source_fails(tmp_path):
    config = _config_path(tmp_path, "[files]\n")
    files = knowledge.load_knowledge_file_configs(config, source_root=tmp_path)
    assert files == {}
    with pytest.raises(ValueError, match="source"):
        knowledge.get_knowledge_file_config(files, "unconfigured.md")


def test_config_requires_real_directory_root(tmp_path):
    config = _config_path(tmp_path, "[files]\n")
    with pytest.raises(ValueError, match="source_root"):
        knowledge.load_knowledge_file_configs(config, source_root=tmp_path / "missing")


def test_config_preserves_missing_file_and_utf8_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        knowledge.load_knowledge_file_configs(tmp_path / "missing.toml", source_root=tmp_path)
    broken = tmp_path / "broken.toml"
    broken.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        knowledge.load_knowledge_file_configs(broken, source_root=tmp_path)


def test_config_rejects_existing_filesystem_case_aliases(tmp_path):
    original = tmp_path / "policy.md"
    original.write_text("政策", encoding="utf-8")
    alias = tmp_path / "POLICY.MD"
    if not alias.exists() or not original.samefile(alias):
        pytest.skip("Requires a case-insensitive filesystem")
    config = _config_path(tmp_path, '''
[files."policy.md"]
content_type = "first"
[files."POLICY.MD"]
content_type = "second"
''')
    with pytest.raises(ValueError, match="Duplicate"):
        knowledge.load_knowledge_file_configs(config, source_root=tmp_path)


def test_config_rejects_existing_physical_file_aliases(tmp_path):
    original = tmp_path / "policy.md"
    original.write_text("政策", encoding="utf-8")
    (tmp_path / "alias.md").hardlink_to(original)
    config = _config_path(tmp_path, '''
[files."policy.md"]
content_type = "first"
[files."alias.md"]
content_type = "second"
''')
    with pytest.raises(ValueError, match="Duplicate"):
        knowledge.load_knowledge_file_configs(config, source_root=tmp_path)


def test_distinct_case_sensitive_files_keep_separate_mapping(tmp_path):
    original = tmp_path / "policy.md"
    original.write_text("第一份政策", encoding="utf-8")
    second = tmp_path / "POLICY.MD"
    if second.exists():
        pytest.skip("Requires a case-sensitive filesystem")
    second.write_text("第二份政策", encoding="utf-8")
    config = _config_path(tmp_path, '''
[files."policy.md"]
content_type = "first"
[files."POLICY.MD"]
content_type = "second"
''')
    files = knowledge.load_knowledge_file_configs(config, source_root=tmp_path)
    assert files["policy.md"].content_type == "first"
    assert files["POLICY.MD"].content_type == "second"
