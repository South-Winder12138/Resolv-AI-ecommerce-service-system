"""Frozen authoritative retrieval, complete prefix and unchanged evidence closure."""

from types import SimpleNamespace

import pytest

from resolv.schemas.hybrid_knowledge import HybridRetrievalError, HybridVersionSnapshot
from resolv.schemas.knowledge_release import ReleaseSelection
from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_flow import PreparedRefundQuery
from resolv.services.hybrid_fingerprints import manifest_hash, profile_hash
from tests.hybrid_fixtures import manifest
from tests.hybrid_fixtures import profile as index_profile
from tests.knowledge_branch_fixtures import hybrid_result
from tests.refund_fixtures import refund_owned
from tests.test_knowledge_hybrid_repository import hybrid_factory

__all__ = ["hybrid_factory", "refund_owned"]


class Counter:
    def count_text(self, text):
        return len(text)


def selected():
    p = index_profile()
    return ReleaseSelection(
        status="configured",
        release_id="refund-fixture",
        index_version="fixture-v3",
        collection=p.collection_name,
        profile=p,
        profile_hash=profile_hash(p),
        manifest_hash=manifest_hash(manifest()),
    )


def harness(result=None):
    class Retrievers:
        def __init__(self):
            self.calls = []

        def validate_selection(self, selection):
            return HybridVersionSnapshot(
                version_id=selection.index_version,
                profile=selection.profile,
                manifest=manifest(),
                state="ready",
            )

        def retrieve(self, selection, query, **kwargs):
            self.calls.append((query, kwargs))
            return result if result is not None else hybrid_result((3.0, 2.0, 1.0))

    return Retrievers()


def retrieve(h, retrievers=None, counter=None, selection=None):
    from resolv.services.refund_evidence import retrieve_refund_evidence

    return retrieve_refund_evidence(
        PreparedRefundQuery(
            standard="申请整单退货",
            expanded=("退款", "退货条件"),
            bm25_text="申请整单退货\n退款\n退货条件",
            status="prepared",
            input_hash="a" * 64,
        ),
        selection=selection or selected(),
        retrievers=retrievers or harness(),
        profile=h.profile.model_copy(update={"selection": selection or selected()}),
        counter=counter or Counter(),
    )


def test_complete_evidence_prefix_and_closed_citations(refund_owned):
    h = refund_owned
    r = harness()

    # Counter preserves real serialized content and only substitutes token counting.
    class WindowsCounter:
        def count_text(self, text):
            import json

            count = len(json.loads(text.split("\n", 1)[1])[0]["citations"])
            return {0: 10, 1: 100, 2: 3001, 3: 200}[count]

    result = retrieve(h, r, WindowsCounter())
    assert (
        r.calls[0][0] == "申请整单退货"
        and r.calls[0][1]["bm25_text"] == "申请整单退货\n退款\n退货条件"
    )
    assert len(result.result.windows) == 3 and len(result.bundle.windows) == 1
    assert {cid for w in result.bundle.windows for cid in w.chunk_ids} == {
        c.chunk.chunk_id for c in result.bundle.chunks
    }
    assert (
        "self_check" not in result.model_dump() and "signals" not in result.model_dump()
    )
    with pytest.raises(RefundFlowError, match="refund_evidence_budget_exceeded"):
        retrieve(h, harness(), SimpleNamespace(count_text=lambda _: 3001))


def test_empty_retrieval_is_valid_but_disabled_and_faults_are_distinct(refund_owned):
    h = refund_owned
    assert retrieve(h, harness(hybrid_result(()))).bundle.windows == ()
    with pytest.raises(RefundFlowError, match="refund_unavailable"):
        retrieve(h, selection=ReleaseSelection(status="disabled"))
    with pytest.raises(RefundFlowError) as error:
        retrieve(
            h, selection=ReleaseSelection(status="invalid", error_code="bad-selector")
        )
    assert error.value.code != "refund_unavailable"

    class Broken:
        def validate_selection(self, _):
            raise OSError("SQL failed")

    with pytest.raises(RefundFlowError) as error:
        retrieve(h, Broken())
    assert error.value.retryable

    class SearchBroken:
        validate_selection = harness().validate_selection

        def retrieve(self, *a, **kw):
            raise HybridRetrievalError("dense_query_failed")

    with pytest.raises(RefundFlowError):
        retrieve(h, SearchBroken())


def test_wrong_result_version_cannot_be_frozen(refund_owned):
    result = hybrid_result(()).model_copy(update={"index_version": "other"})
    with pytest.raises(RefundFlowError):
        retrieve(refund_owned, harness(result))


def test_selection_cannot_replace_frozen_run_selection(refund_owned):
    from resolv.services.refund_evidence import retrieve_refund_evidence

    query = PreparedRefundQuery(
        standard="退货",
        expanded=(),
        bm25_text="退货",
        status="prepared",
        input_hash="a" * 64,
    )
    with pytest.raises(RefundFlowError, match="refund_release_conflict"):
        retrieve_refund_evidence(
            query,
            selection=selected(),
            retrievers=harness(),
            profile=refund_owned.profile,
            counter=SimpleNamespace(count_text=lambda _: 10),
        )


def test_standard_reaches_actual_dense_and_reranker_bm25_receives_expansions(
    refund_owned, hybrid_factory
):
    from tests.test_hybrid_retrieval import retriever

    factory, _, m = hybrid_factory
    service, embedding, store, reranker = retriever(factory, m)
    p = index_profile()
    selection = ReleaseSelection(
        status="configured",
        release_id="fixture-release",
        index_version="v3",
        collection=p.collection_name,
        profile=p,
        profile_hash=profile_hash(p),
        manifest_hash=manifest_hash(m),
    )

    class Pool:
        def validate_selection(self, s):
            return HybridVersionSnapshot(
                version_id="v3", profile=p, manifest=m, state="ready"
            )

        def retrieve(self, s, query, **kwargs):
            return service.retrieve(query, bm25_text=kwargs["bm25_text"])

    result = retrieve(
        refund_owned,
        Pool(),
        SimpleNamespace(count_text=lambda text: len(text) // 100),
        selection,
    )
    assert result.bundle.windows and result.result.diagnostics.bm25_count == 0
    assert embedding.queries == reranker.queries == ["申请整单退货"]
    assert store.queries == ["申请整单退货\n退款\n退货条件"]
    store.error = RuntimeError("BM25 fault")
    with pytest.raises(RefundFlowError, match="refund_retrieval_failed"):
        retrieve(
            refund_owned,
            Pool(),
            SimpleNamespace(count_text=lambda text: len(text) // 100),
            selection,
        )
