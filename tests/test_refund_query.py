"""One prepared refund query; failures may not hide storage or ownership faults."""

import json

import pytest
from pydantic import ValidationError

from resolv.schemas.refund import RefundFlowError
from resolv.schemas.refund_flow import RefundQueryOutput, RefundRequest
from tests.refund_fixtures import refund_owned
from tests.test_refund_repository import bound

__all__ = ["refund_owned"]


class Client:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def prepare(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        try:
            return RefundQueryOutput.model_validate_json(
                self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
            )
        except (ValidationError, ValueError) as exc:
            raise RefundFlowError("refund_model_output_invalid", True) from exc


def prepare(h, client, count=len, **kwargs):
    from resolv.services.refund_query import prepare_refund_query

    return prepare_refund_query(
        RefundRequest(operation="apply"),
        h.primary,
        bound(h),
        client=client,
        profile=h.profile,
        query_token_count=count,
        audit_hook=lambda _: None,
        **kwargs,
    )


def test_refund_standard_and_expanded_routing(refund_owned):
    h = refund_owned
    client = Client(
        {"standard": "申请整单退货", "expanded": ["退款", "退货条件", " ", "退款"]}
    )
    prepared = prepare(h, client)
    assert prepared.bm25_text == "申请整单退货\n退款\n退货条件"
    sent = json.loads(client.calls[0][0][1].content)
    assert sent["primary_text"] == h.primary.primary.text
    assert (
        sent["request"]["operation"] == "apply"
        and sent["binding"]["order_id"] == "A001"
    )
    assert "knowledge_check" not in str(client.calls)
    client = Client(
        {"standard": "申请整单退货", "expanded": [str(i) * 32 for i in range(8)]}
    )
    assert len(prepare(h, client).expanded) == 8


@pytest.mark.parametrize(
    "reply",
    [
        TimeoutError(),
        RefundFlowError("refund_model_unavailable", True),
        '{"standard":',
        {"standard": " ", "expanded": []},
        {"standard": "退货", "expanded": ["a" * 33]},
        {"standard": "退货", "expanded": [str(i) for i in range(9)]},
    ],
)
def test_query_fallback_respects_failure_boundaries(refund_owned, reply):
    h = refund_owned
    prepared = prepare(h, Client(reply))
    assert (
        prepared.status == "fallback"
        and prepared.standard == prepared.bm25_text == h.primary.primary.text
        and prepared.expanded == ()
    )
    assert prepared.fallback_reason


@pytest.mark.parametrize(
    "error",
    [
        OSError("SQL"),
        ValueError("storage hash"),
        RefundFlowError("refund_execution_conflict"),
        RefundFlowError("refund_source_conflict"),
    ],
)
def test_ownership_storage_and_tokenizer_faults_never_fallback(refund_owned, error):
    with pytest.raises(type(error)):
        prepare(refund_owned, Client(error))

    def broken(_):
        raise error

    with pytest.raises(type(error)):
        prepare(refund_owned, Client({"standard": "退货", "expanded": []}), broken)


def test_query_length_falls_back_once_and_original_overflow_fails(refund_owned):
    h = refund_owned
    prepared = prepare(h, Client({"standard": "x" * 513, "expanded": []}))
    assert prepared.standard == h.primary.primary.text and prepared.status == "fallback"
    with pytest.raises(RefundFlowError, match="refund_query_too_long"):
        prepare(h, Client({"standard": "x" * 513, "expanded": []}), lambda _: 513)
    with pytest.raises(RefundFlowError, match="refund_query_token_count_invalid"):
        prepare(h, Client({"standard": "退货", "expanded": []}), lambda _: True)
    from resolv.schemas.hybrid_knowledge import HybridRetrievalError

    def long(_):
        raise HybridRetrievalError("query_too_long")

    with pytest.raises(RefundFlowError, match="refund_query_too_long"):
        prepare(h, Client(TimeoutError()), long)


def test_wrong_binding_rejected_before_model(refund_owned):
    from resolv.services.refund_query import prepare_refund_query

    h = refund_owned
    client = Client({"standard": "退货", "expanded": []})
    from tests.refund_fixtures import binding

    with pytest.raises(RefundFlowError):
        prepare_refund_query(
            RefundRequest(operation="apply"),
            h.primary,
            binding(),
            client=client,
            profile=h.profile,
            query_token_count=len,
            audit_hook=lambda _: None,
        )
    assert client.calls == []


def test_default_profile_freezes_refund_specific_query_prompt(refund_owned):
    from resolv.core.refund_config import default_refund_profile
    from resolv.core.refund_prompts import QUERY_PROMPT, QUERY_PROMPT_VERSION

    h = refund_owned
    profile = default_refund_profile(
        understanding_config=h.profile.query_model,
        business_config=h.profile.decision_model,
        summary_profile={},
        adapter_descriptor=h.profile.adapter,
    )
    assert profile.query_model.system_prompt == QUERY_PROMPT
    assert profile.query_model.prompt_version == QUERY_PROMPT_VERSION
