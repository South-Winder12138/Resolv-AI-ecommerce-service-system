from resolv.schemas.hybrid_knowledge import HybridRetrievalProfile
from resolv.services.hybrid_fingerprints import (
    content_hash,
    document_hash,
    index_hash,
    manifest_hash,
    profile_hash,
)
from tests.hybrid_fixtures import chunk, manifest, manifest_chunk, profile, source


def test_fingerprints_stable_across_json_roundtrip():
    p = profile()
    m = manifest()
    assert profile_hash(p) == profile_hash(type(p).model_validate_json(p.model_dump_json()))
    assert manifest_hash(m) == manifest_hash(type(m).model_validate_json(m.model_dump_json()))
    assert len(content_hash(chunk())) == 64


def test_document_raw_tail_and_source_are_bound_without_trimming():
    s = source()
    assert document_hash(s) != document_hash(source(document=s.document.model_copy(update={'text': s.document.text+' '})))
    assert document_hash(s) != document_hash(source(document=s.document.model_copy(update={'source':'other.md'})))
    assert document_hash(s) != document_hash(source(ingest_hash='f'*64))


def test_content_covers_full_payload_including_tail_metadata_and_answer():
    original = chunk()
    for name, value in [('text', original.text+' '), ('answer', original.answer+' '),
                        ('source','other.md'), ('chunk_index',1), ('section_path','另一节'),
                        ('is_key_clause',1), ('questions','另一问题')]:
        assert content_hash(original) != content_hash(original.model_copy(update={name:value}))


def test_index_binds_profile_version_section_and_manifest_payload():
    c, p = manifest_chunk(), profile()
    baseline = index_hash(c,p,'v3')
    assert baseline != index_hash(c,p,'v4')
    assert baseline != index_hash(c,p.model_copy(update={'collection_name':'other'}),'v3')
    assert baseline != index_hash(c.model_copy(update={'section_instance_id':'b'*64}),p,'v3')
    assert baseline != index_hash(c.model_copy(update={'expandable':False}),p,'v3')
    assert baseline != index_hash(c.model_copy(update={'chunk':chunk(answer='条件不同')}),p,'v3')
    assert baseline != index_hash(c.model_copy(update={'content_hash':'f'*64}),p,'v3')
    # Retrieval choices never enter the index profile or the index fingerprint.
    HybridRetrievalProfile(reranker_revision='a'*40)
    assert baseline == index_hash(c,p,'v3')


def test_manifest_binds_config_chunk_rules_order_and_hashes():
    baseline = manifest_hash(manifest())
    assert baseline != manifest_hash(manifest(max_chars=900))
    assert baseline != manifest_hash(manifest(overlap=0))
    assert baseline != manifest_hash(manifest(document_hash='f'*64))
    assert baseline != manifest_hash(manifest(chunks=(manifest_chunk(expandable=False),)))
    from resolv.schemas.knowledge import KnowledgeFileConfig
    assert baseline != manifest_hash(manifest(file_config=KnowledgeFileConfig(content_type='faq')))
