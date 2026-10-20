import pytest
from pydantic import ValidationError

from resolv.core.hybrid_knowledge_config import (
    default_index_profile,
    default_retrieval_profile,
)
from resolv.schemas.hybrid_knowledge import (
    HybridDiagnostics,
    HybridEvidenceWindow,
    HybridRetrievalProfile,
    HybridRetrievalResult,
    VerifiedHybridChunk,
)
from resolv.services.hybrid_ranking import fuse_rrf
from tests.hybrid_fixtures import candidate, chunk, profile


def evidence(**changes):
    fused = fuse_rrf((candidate(),), ())[0]
    data = {'window_id': 'f'*64, 'document_id': 'd'*64, 'section_instance_id': 'e'*64,
                'chunk_ids': ('a'*64,), 'center_chunk_ids': ('a'*64,), 'center_candidates': (fused,),
                'best_rrf': fused.rrf_score, 'rerank_score': -2.0}
    data.update(changes)
    return HybridEvidenceWindow(**data)


def body():
    return VerifiedHybridChunk(chunk=chunk(), document_hash='b'*64, content_hash='c'*64,
                               index_hash='c'*64, section_instance_id='e'*64, expandable=True)


def result(**changes):
    data = {'index_version': 'fixture-v3', 'index_profile_hash': 'b'*64,
                'retrieval_profile': HybridRetrievalProfile(), 'windows': (evidence(),),
                'chunks': {'a'*64: body()}, 'diagnostics': HybridDiagnostics()}
    data.update(changes)
    return HybridRetrievalResult(**data)


def test_fixed_profiles_reject_silent_runtime_changes():
    p = default_index_profile('mhhybrid_gate_fixture')
    r = default_retrieval_profile()
    assert p.dimension == 1024 and p.device == 'cpu' and p.precision == 'fp32'
    assert p.model_revision == '5617a9f61b028005a4858fdac845db406aefb181'
    assert (r.dense_limit, r.bm25_limit, r.center_limit, r.output_limit, r.rrf_k) == (50,50,50,10,60)
    assert (r.batch_size, r.query_max_tokens, r.pair_max_tokens) == (4,512,3072)
    assert (p.analyzer, p.bm25_k1, p.bm25_b) == ('chinese',1.2,0.75)
    for changes in ({'device':'cpu'}, {'batch_size':8}, {'pair_max_tokens':3073}, {'dense_limit':51}):
        with pytest.raises(ValidationError):
            HybridRetrievalProfile(**changes)
    with pytest.raises(ValidationError):
        p.collection_name = 'changed'
    with pytest.raises(ValidationError):
        profile(dimension='1024')
    with pytest.raises(ValidationError):
        profile(unexpected=True)


def test_empty_result_is_valid():
    assert result(windows=(), chunks={}).windows == ()


@pytest.mark.parametrize('changes', [
    {'chunks':{}}, {'chunks':{'b'*64:body()}},
    {'windows':()},
    {'windows':tuple(evidence(window_id=f'{i:064x}') for i in range(11))},
])
def test_result_rejects_dangling_or_orphan_bodies_and_excess_windows(changes):
    with pytest.raises(ValidationError):
        result(**changes)


def test_window_rejects_duplicate_chunks_unrelated_centers_and_nonfinite_logits():
    for changes in ({'chunk_ids':('a'*64,'a'*64)}, {'center_chunk_ids':('b'*64,)},
                    {'rerank_score':float('nan')}):
        with pytest.raises(ValidationError):
            evidence(**changes)


def test_manifest_document_order_and_duplicate_ids_rejected():
    from resolv.schemas.hybrid_knowledge import HybridCorpusManifest
    from tests.hybrid_fixtures import manifest
    m = manifest()
    with pytest.raises(ValidationError):
        HybridCorpusManifest(documents=m.documents*2)
