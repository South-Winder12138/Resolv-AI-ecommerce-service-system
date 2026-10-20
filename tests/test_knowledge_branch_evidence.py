"""Complete ranked windows, shared bodies and actual serialized token budgets."""

import socket

import pytest

from resolv.schemas.knowledge_branch import KnowledgeBranchError
from resolv.services.context import TokenCounter, knowledge_text
from resolv.services.hybrid_fingerprints import canonical_hash
from resolv.services.knowledge_branch_evidence import (
    select_knowledge_evidence,
    serialize_knowledge_evidence,
)
from tests.knowledge_branch_fixtures import bundle, hybrid_result


class Characters:
    def count_text(self, text):
        return len(text)


def test_complete_prefix_and_shared_body_budget():
    result = hybrid_result((3.0, 2.0, 1.0), overlapping=True)
    original = result.model_dump_json()
    budget = len(bundle((3.0, 2.0), overlapping=True).serialized)
    selected = select_knowledge_evidence(
        result, counter=Characters(), max_tokens=budget
    )
    assert [c.citation_number for c in selected.citations] == [1, 2]
    assert len(selected.chunks) == 4
    assert selected.windows == result.windows[:2]
    assert selected.input_tokens == budget
    assert selected.serialized == knowledge_text([selected.payload])
    assert serialize_knowledge_evidence(selected) == selected.serialized
    assert selected.evidence_hash == canonical_hash(
        {"rule": "knowledge-evidence-v1", "text": selected.serialized}
    )
    assert result.model_dump_json() == original
    assert all("rerank_score" not in c for c in selected.payload["chunks"].values())


def test_first_window_cannot_fit_is_error():
    size = len(bundle((1.0,)).serialized)
    with pytest.raises(KnowledgeBranchError, match="evidence_budget_exceeded"):
        select_knowledge_evidence(
            hybrid_result((1.0,)), counter=Characters(), max_tokens=size - 1
        )
    assert (
        len(
            select_knowledge_evidence(
                hybrid_result((1.0,)), counter=Characters(), max_tokens=size
            ).windows
        )
        == 1
    )


def test_budget_never_skips_second_for_smaller_third():
    result = hybrid_result((3.0, 2.0, 1.0))
    cid = result.windows[1].chunk_ids[0]
    body = result.chunks[cid]
    result.chunks[cid] = body.model_copy(
        update={"chunk": body.chunk.model_copy(update={"answer": "长" * 10000})}
    )
    budget = len(bundle((3.0, 2.0)).serialized)
    selected = select_knowledge_evidence(
        result, counter=Characters(), max_tokens=budget
    )
    assert selected.windows == result.windows[:1]


def test_empty_evidence_is_a_closed_valid_bundle():
    selected = select_knowledge_evidence(hybrid_result(()), counter=Characters())
    assert selected.windows == selected.citations == selected.chunks == ()
    assert selected.payload == {"citations": [], "chunks": {}}
    assert selected.input_tokens == len(knowledge_text([selected.payload]))


def test_metadata_counts_toward_budget():
    result = hybrid_result((1.0,))
    body_chars = sum(len(c.chunk.text) for c in result.chunks.values())
    with pytest.raises(KnowledgeBranchError, match="evidence_budget_exceeded"):
        select_knowledge_evidence(
            result, counter=Characters(), max_tokens=body_chars + 10
        )


def test_orphan_or_tampered_legacy_result_is_rejected():
    result = hybrid_result((1.0,))
    result.chunks.clear()
    with pytest.raises(KnowledgeBranchError, match="knowledge_evidence_invalid"):
        select_knowledge_evidence(result, counter=Characters())


@pytest.mark.parametrize("tokens", [True, -1, 2.5])
def test_counter_must_return_a_nonnegative_integer(tokens):
    class InvalidCounter:
        def count_text(self, text):
            return tokens

    with pytest.raises(KnowledgeBranchError, match="evidence_token_count_invalid"):
        select_knowledge_evidence(hybrid_result((1.0,)), counter=InvalidCounter())


def test_real_local_tokenizer_boundary_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Tokenizer attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    counter = TokenCounter("o200k_base")
    size = counter.count_text(bundle((1.0,)).serialized)
    selected = select_knowledge_evidence(
        hybrid_result((1.0,)), counter=counter, max_tokens=size
    )
    assert selected.input_tokens == size < len(selected.serialized)
    with pytest.raises(KnowledgeBranchError, match="evidence_budget_exceeded"):
        select_knowledge_evidence(
            hybrid_result((1.0,)), counter=counter, max_tokens=size - 1
        )
