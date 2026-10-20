import pytest
from pydantic import ValidationError

from resolv.schemas.hybrid_knowledge import HybridRetrievalError
from resolv.services.hybrid_ranking import fuse_rrf
from tests.hybrid_fixtures import candidate


def test_rrf_uses_original_ranks_and_full_identity():
    fused = fuse_rrf((candidate(rank=1), candidate(rank=9, index_hash='b'*64)),
                     (candidate('bm25', rank=2),))
    assert fused[0].rrf_score == pytest.approx(1/61 + 1/62)
    assert len(fused) == 2
    assert fused[1].dense.rank == 9
    assert fused[1].bm25 is None


def test_duplicates_use_first_and_do_not_renumber_filtered_ranks():
    first = candidate(rank=7)
    fused = fuse_rrf((first, candidate(rank=8, raw_score=20.0)), ())
    assert len(fused) == 1
    assert fused[0].dense == first
    assert fused[0].rrf_score == pytest.approx(1/67)


def test_same_id_different_version_is_independent():
    fused = fuse_rrf((candidate(),), (candidate('bm25', index_version='old-v2'),))
    assert len(fused) == 2
    assert all(x.rrf_score == pytest.approx(1/61) for x in fused)


def test_ties_use_best_rank_then_chunk_id_and_are_repeatable():
    fused = fuse_rrf((candidate(chunk_id='b'*64), candidate(rank=2, chunk_id='c'*64)),
                     (candidate('bm25', chunk_id='a'*64),))
    assert [x.chunk_id for x in fused] == ['a'*64, 'b'*64, 'c'*64]
    assert fuse_rrf((), ()) == ()


def test_full_union_is_not_cut_to_fifty_before_authority():
    dense = tuple(candidate(rank=i+1, chunk_id=f'{i:064x}') for i in range(50))
    bm25 = tuple(candidate('bm25', rank=i+1, chunk_id=f'{i+50:064x}') for i in range(50))
    assert len(fuse_rrf(dense, bm25)) == 100


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_score_rejected(score):
    with pytest.raises(ValidationError):
        candidate(raw_score=score)


def test_wrong_route_or_inconsistent_document_is_fault():
    with pytest.raises(HybridRetrievalError):
        fuse_rrf((candidate('bm25'),), ())
    with pytest.raises(HybridRetrievalError):
        fuse_rrf((candidate(),), (candidate('bm25', document_id='b'*64),))
