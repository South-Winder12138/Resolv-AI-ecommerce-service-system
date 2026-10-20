import pytest

from resolv.schemas.hybrid_knowledge import (
    HybridCandidate,
    HybridRetrievalError,
    VerifiedHybridCenter,
    VerifiedHybridChunk,
)
from resolv.services.hybrid_ranking import fuse_rrf
from resolv.services.hybrid_windows import build_windows, select_evidence
from tests.hybrid_fixtures import chunk


def body(i, **changes):
    c = chunk(
        chunk_id=f"{i:064x}",
        chunk_index=i,
        text="退货\n\n公共重叠正文 " + str(i),
        answer="公共重叠正文 " + str(i),
    )
    data = {
        "chunk": c,
        "document_hash": "d" * 64,
        "content_hash": f"{i + 300:064x}",
        "index_hash": f"{i + 400:064x}",
        "section_instance_id": "f" * 64,
        "expandable": True,
    }
    data.update(changes)
    return VerifiedHybridChunk(**data)


def center(c, rank=1):
    v = HybridCandidate(
        index_version="v3",
        chunk_id=c.chunk.chunk_id,
        document_id=c.chunk.document_id,
        index_hash=c.index_hash,
        route="dense",
        rank=rank,
        raw_score=0.1,
    )
    return VerifiedHybridCenter(verified_chunk=c, fused_candidate=fuse_rrf((v,), ())[0])


def mapping(values):
    return {c.chunk.chunk_id: c for c in values}


def test_partial_overlap_preserves_windows_and_unique_bodies():
    chunks = mapping([body(i) for i in (7, 8, 9, 10)])
    windows = build_windows(
        (center(chunks[f"{8:064x}"]), center(chunks[f"{9:064x}"])), chunks
    )
    assert [w.chunk_ids for w in windows] == [
        tuple(f"{i:064x}" for i in (7, 8, 9)),
        tuple(f"{i:064x}" for i in (8, 9, 10)),
    ]
    evidence, bodies = select_evidence(windows, (0.7, 0.4), chunks)
    assert len(windows) == 2 and len(bodies) == 4 and len(evidence) == 2
    assert windows[0].body_text.count("退货") == 1
    assert windows[0].body_text.count("公共重叠正文") == 3
    assert not hasattr(evidence[0], "body_text") and not hasattr(
        bodies[f"{8:064x}"], "rerank_score"
    )


def test_identical_windows_merge_all_centers_without_expanding():
    values = [body(i) for i in (7, 8)]
    chunks = mapping(values)
    windows = build_windows(tuple(center(c) for c in values), chunks)
    assert len(windows) == 1 and windows[0].center_chunk_ids == tuple(
        f"{i:064x}" for i in (7, 8)
    )
    assert len(windows[0].chunk_ids) == 2


@pytest.mark.parametrize(
    "case", ["different_section", "txt", "no_title", "missing", "different_doc", "gap"]
)
def test_neighbor_boundaries_are_not_crossed(case):
    c = body(8)
    left = body(7)
    right = body(9)
    if case == "different_section":
        left = body(7, section_instance_id="a" * 64)
        right = body(9, section_instance_id="b" * 64)
    elif case in ("txt", "no_title"):
        c = body(8, expandable=False)
    elif case == "different_doc":
        left = body(7, chunk=left.chunk.model_copy(update={"document_id": "b" * 64}))
        right = body(9, chunk=right.chunk.model_copy(update={"document_id": "b" * 64}))
    elif case == "gap":
        left = body(6)
        right = body(10)
    values = [c] if case == "missing" else [left, c, right]
    assert build_windows((center(c),), mapping(values))[0].chunk_ids == (f"{8:064x}",)


def test_missing_one_adjacent_block_does_not_find_a_farther_one():
    c = body(8)
    chunks = mapping([body(6), c, body(9), body(10)])
    assert build_windows((center(c),), chunks)[0].chunk_ids == tuple(
        f"{i:064x}" for i in (8, 9)
    )


def test_fifty_windows_top_ten_and_thirty_unique_body_limit():
    chunks = mapping([body(i) for i in range(150)])
    centers = tuple(center(chunks[f"{3 * i + 1:064x}"], rank=i + 1) for i in range(50))
    windows = build_windows(centers, chunks)
    assert len(windows) == 50
    evidence, bodies = select_evidence(
        windows, tuple(float(i) for i in range(50)), chunks
    )
    assert len(evidence) == 10 and len(bodies) == 30
    assert set(bodies) == {cid for w in evidence for cid in w.chunk_ids}


def test_score_ties_use_peak_rrf_then_stable_window_id():
    chunks = mapping([body(i) for i in (1, 10, 20)])
    windows = build_windows(
        (
            center(chunks[f"{1:064x}"], 2),
            center(chunks[f"{10:064x}"], 1),
            center(chunks[f"{20:064x}"], 2),
        ),
        chunks,
    )
    selected, _ = select_evidence(tuple(reversed(windows)), (0.1, 0.1, 0.1), chunks)
    assert selected[0].center_chunk_ids == (f"{10:064x}",)
    assert [w.window_id for w in selected[1:]] == sorted(
        w.window_id for w in windows if w.center_chunk_ids != (f"{10:064x}",)
    )


@pytest.mark.parametrize("scores", [(float("nan"),), (float("inf"),), (), (1.0, 2.0)])
def test_invalid_scores_never_select_partial_evidence(scores):
    c = body(8)
    chunks = mapping([c])
    windows = build_windows((center(c),), chunks)
    with pytest.raises(HybridRetrievalError):
        select_evidence(windows, scores, chunks)
